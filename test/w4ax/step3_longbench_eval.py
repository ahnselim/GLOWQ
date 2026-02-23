"""
w4ax/step3_longbench_eval.py
Evaluates FP16 and quantized-plus-SVD W4Ax model variants on LongBench with built-in scoring.
output :
<dirname(csv_path)>/
`-- <basename(csv_path)>   (results appended; columns: method,dataset,score)
"""

import argparse
import csv
import json
import math
import os
import re
import string
import difflib
import gc
from collections import Counter
from contextlib import contextmanager
from typing import Optional, List, Dict

try:
    from fuzzywuzzy import fuzz as fuzzy_fuzz
except ImportError:
    fuzzy_fuzz = None

try:
    from rouge import Rouge as RougeMetric
except ImportError:
    RougeMetric = None

_ROUGE = RougeMetric() if RougeMetric is not None else None

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from time import perf_counter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer





class MiniGroupCache:
    __slots__ = ("r", "valid", "uses_left")

    def __init__(self):
        self.r: Optional[torch.Tensor] = None
        self.valid: bool = False
        self.uses_left: int = 0

    def set(self, r: torch.Tensor, uses: int):
        self.r = r
        self.valid = True
        self.uses_left = uses

    def consume(self):
        if self.valid and self.uses_left > 0 and self.r is not None:
            self.uses_left -= 1
            out = self.r
            if self.uses_left == 0:
                self.valid = False
                self.r = None
            return out, True
        return None, False

    def clear(self):
        self.r = None
        self.valid = False
        self.uses_left = 0


GROUP_CORR_CACHE: Dict[str, MiniGroupCache] = {}


def clear_group_cache():
    for cache in GROUP_CORR_CACHE.values():
        cache.clear()



TARGET_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
}





def _cuda_sync(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def temp_generation_overrides(model, **overrides):
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        yield
        return
    old_vals = {k: getattr(gen_cfg, k, None) for k in overrides}
    for k, v in overrides.items():
        try:
            setattr(gen_cfg, k, v)
        except Exception:
            pass
    try:
        yield
    finally:
        for k, v in old_vals.items():
            try:
                setattr(gen_cfg, k, v)
            except Exception:
                pass





class AddSVDCorrection(nn.Module):


    def __init__(
        self,
        inner: nn.Module,
        A_q: torch.Tensor,
        B_q: torch.Tensor,
        gkey: str,
        is_group: bool,
        role: str,
        group_cache: Optional[MiniGroupCache],
        alpha_svd: float = 1.0,
    ):
        super().__init__()
        self.inner = inner
        self.register_buffer("A_q", A_q.to(torch.float16), persistent=False)
        self.register_buffer("B_q", B_q.to(torch.float16), persistent=False)
        self.gkey = gkey
        self.is_group = is_group
        self.role = role
        self.group_cache = group_cache
        self.alpha_svd = alpha_svd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)

        if self.alpha_svd == 0.0:
            return z

        A_q_dev = self.A_q
        B_q_dev = self.B_q

        try:

            if self.is_group:
                if self.role in ("q", "gate"):
                    intermediate_r = F.linear(x, B_q_dev)
                    uses = 2 if self.role == "q" else 1
                    if self.group_cache is not None:
                        self.group_cache.set(intermediate_r, uses)
                elif self.role in ("k", "v", "up"):
                    cached_val, ok = (
                        self.group_cache.consume()
                        if self.group_cache is not None
                        else (None, False)
                    )
                    if ok and cached_val is not None:
                        intermediate_r = cached_val
                    else:
                        intermediate_r = F.linear(x, B_q_dev)
                else:
                    intermediate_r = F.linear(x, B_q_dev)
            else:
                intermediate_r = F.linear(x, B_q_dev)

            svd_raw = F.linear(intermediate_r, A_q_dev)


            if z.shape != svd_raw.shape:
                if len(z.shape) == len(svd_raw.shape):
                    if z.shape[:-1] == svd_raw.shape[:-1]:
                        min_last_dim = min(z.shape[-1], svd_raw.shape[-1])
                        svd_raw = svd_raw[..., :min_last_dim]
                        if z.shape[-1] > min_last_dim:
                            pad_size = z.shape[-1] - min_last_dim
                            svd_raw = F.pad(svd_raw, (0, pad_size))
                    else:
                        if svd_raw.numel() == z.numel():
                            svd_raw = svd_raw.reshape(z.shape)
                        else:
                            return z
                else:
                    if svd_raw.numel() == z.numel():
                        svd_raw = svd_raw.reshape(z.shape)
                    else:
                        return z

            return z.add_(svd_raw, alpha=self.alpha_svd)

        except RuntimeError:

            return z
        except Exception:

            return z





def fake_quantize_activation(
    x: torch.Tensor,
    group_size: int = 128,
    num_bits: int = 8,
) -> torch.Tensor:
    if num_bits <= 0:
        return x
    if x.dim() == 0:
        return x

    original_shape = x.shape
    last_dim = original_shape[-1]
    if last_dim == 0:
        return x

    x_2d = x.reshape(-1, last_dim)
    pad = (group_size - (last_dim % group_size)) % group_size
    if pad:
        x_2d = F.pad(x_2d, (0, pad), mode="constant", value=0.0)

    x_2d = x_2d.to(torch.float32)
    grouped = x_2d.view(-1, group_size)

    min_vals = grouped.min(dim=-1, keepdim=True).values
    max_vals = grouped.max(dim=-1, keepdim=True).values
    qmax = (1 << num_bits) - 1
    ranges = max_vals - min_vals

    scales = torch.clamp(ranges / max(qmax, 1), min=1e-8)
    zeros = torch.round(-min_vals / scales).clamp(0, qmax)
    quant = torch.round(grouped / scales + zeros).clamp(0, qmax)
    dequant = (quant - zeros) * scales


    tiny_mask = (ranges < 1e-8).expand_as(grouped)
    if tiny_mask.any():
        dequant = torch.where(tiny_mask, grouped, dequant)

    dequant = dequant.view(-1, group_size)
    if pad:
        dequant = dequant[:, :last_dim]

    return dequant.reshape(*original_shape).to(x.dtype)


class ActivationFakeQuantWrapper(nn.Module):
    def __init__(self, inner: nn.Module, act_bits: int = 8, group_size: int = 128):
        super().__init__()
        self.inner = inner
        self.act_bits = act_bits
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qx = fake_quantize_activation(
            x, group_size=self.group_size, num_bits=self.act_bits
        )
        return self.inner(qx)


def _role_from_suffix(module_suffix: str) -> str:
    if module_suffix.endswith("q_proj"):
        return "q"
    if module_suffix.endswith("k_proj"):
        return "k"
    if module_suffix.endswith("v_proj"):
        return "v"
    if module_suffix.endswith("gate_proj"):
        return "gate"
    if module_suffix.endswith("up_proj"):
        return "up"
    return "solo"


def _unwrap_base_linear(module: nn.Module) -> nn.Module:

    while isinstance(module, (AddSVDCorrection, ActivationFakeQuantWrapper)):
        module = module.inner
    return module





def get_parent_module(model, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _is_target_module_name(name: str) -> bool:
    if not name:
        return False
    if not (
        "layers" in name
        or "encoder.layers" in name
        or "model.layers" in name
    ):
        return False
    return name.split(".")[-1] in TARGET_SUFFIXES


def strip_wrappers(model: nn.Module, module_names: List[str]):

    unique = sorted(set(module_names))
    for module_name in unique:
        try:
            parent, attr_name = get_parent_module(model, module_name)
        except AttributeError:
            continue
        current = getattr(parent, attr_name, None)
        if current is None:
            continue
        base = _unwrap_base_linear(current)
        setattr(parent, attr_name, base)


@torch.no_grad()
def apply_quantized_weights(model: nn.Module, qweights: dict):
    injected, missing, mismatch = 0, 0, 0
    for wkey, Wq in qweights.items():
        if not (
            isinstance(wkey, str)
            and wkey.endswith(".weight")
            and getattr(Wq, "ndim", 0) == 2
        ):
            continue
        module_name = wkey[:-7]
        try:
            parent, attr_name = get_parent_module(model, module_name)
        except AttributeError:
            missing += 1
            continue
        target = getattr(parent, attr_name, None)
        if target is None:
            missing += 1
            continue
        inner = _unwrap_base_linear(target)
        if not hasattr(inner, "weight"):
            missing += 1
            continue
        if inner.weight.shape != Wq.shape:
            mismatch += 1
            continue
        inner.weight.data.copy_(
            Wq.to(device=inner.weight.device, dtype=inner.weight.dtype)
        )
        injected += 1
    print(
        f"[FakeQuant] injected={injected}, missing={missing}, shape_mismatch={mismatch}"
    )


@torch.no_grad()
def apply_activation_fake_quant(
    model: nn.Module,
    module_names: List[str],
    act_bits: int = 8,
    group_size: int = 128,
):

    if act_bits <= 0:
        print("[Activation Quant] act_bits <= 0, activation quantization disabled for this variant.")
        return model

    unique_modules = sorted(set(module_names))
    wrapped = 0
    for module_name in unique_modules:
        if not _is_target_module_name(module_name):
            continue
        try:
            parent, attr_name = get_parent_module(model, module_name)
        except AttributeError:
            continue
        current = getattr(parent, attr_name, None)
        if current is None:
            continue
        if isinstance(current, ActivationFakeQuantWrapper):

            current.act_bits = act_bits
            current.group_size = group_size
            continue
        setattr(
            parent,
            attr_name,
            ActivationFakeQuantWrapper(current, act_bits=act_bits, group_size=group_size),
        )
        wrapped += 1
    print(
        f"[Activation Quant] Wrapped {wrapped} modules with fake int{act_bits} activation quantization (group_size={group_size})."
    )
    return model


def patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0):
    patched_count, skipped_count = 0, 0
    for weight_name, bkey in tqdm(bmap.items(), desc="Patching SVD Correction"):
        module_name = weight_name.replace(".weight", "")
        B_q = shared.get(bkey)
        is_group = "B_shared" in bkey

        if is_group:
            gkey = bkey.replace(".B_shared", "")
            module_suffix = module_name.split(".")[-1]
            a_key = f"{gkey}.{module_suffix}.A"
            role = _role_from_suffix(module_suffix)
            cache = GROUP_CORR_CACHE.setdefault(gkey, MiniGroupCache())
        else:
            if name_match := re.match(r"(model\.layers\.\d+\..*?)\.B", bkey):
                gkey = name_match.group(1)
            else:
                gkey = bkey.replace(".B", "")
            a_key = gkey + ".A"
            role = "solo"
            cache = None

        A_q = shared.get(a_key)
        if A_q is None or B_q is None:
            skipped_count += 1
            continue

        try:
            parent, attr_name = get_parent_module(model, module_name)
            current = getattr(parent, attr_name)
        except AttributeError as e:
            print(f"AttributeError for {module_name}: {e}")
            skipped_count += 1
            continue

        inner = _unwrap_base_linear(current)
        if not hasattr(inner, "weight"):
            skipped_count += 1
            continue

        wrapped = AddSVDCorrection(
            inner, A_q, B_q, gkey, is_group, role, cache, alpha_svd
        )


        if isinstance(current, ActivationFakeQuantWrapper):
            current.inner = wrapped
            setattr(parent, attr_name, current)
        else:
            setattr(parent, attr_name, wrapped)

        patched_count += 1

    print(
        f"SVD Correction Patching Summary: {patched_count} patched, {skipped_count} skipped"
    )
    return model





def normalize_answer(s: str) -> str:

    def lower(text):
        return text.lower()

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_single(pred: str, truth: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()

    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def qa_f1_max(pred: str, answers: List[str]) -> float:
    if not answers:
        return 0.0
    return max(qa_f1_single(pred, a) for a in answers)


def rouge_l_single(pred: str, truth: str) -> float:
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    m, n = len(pred_tokens), len(truth_tokens)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == truth_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs_len = dp[m][n]
    prec = lcs_len / m
    rec = lcs_len / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l_max(pred: str, answers: List[str]) -> float:
    if not answers:
        return 0.0
    return max(rouge_l_single(pred, a) for a in answers)


def exact_match_accuracy(pred: str, answers: List[str]) -> float:
    if not answers:
        return 0.0
    pred_norm = normalize_answer(pred)
    return 1.0 if any(pred_norm == normalize_answer(a) for a in answers) else 0.0


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def edit_similarity(pred: str, truth: str) -> float:

    pred_tokens = pred.split()
    truth_tokens = truth.split()
    max_len = max(len(pred_tokens), len(truth_tokens))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(pred_tokens, truth_tokens)
    return 1.0 - dist / max_len


def edit_sim_max(pred: str, answers: List[str]) -> float:
    if not answers:
        return 0.0
    return max(edit_similarity(pred, a) for a in answers)


def qa_f1_metric(prediction: str, ground_truth: str, **kwargs) -> float:
    return qa_f1_single(prediction, ground_truth)


def rouge_l_metric(prediction: str, ground_truth: str, **kwargs) -> float:
    if _ROUGE is not None:
        try:
            scores = _ROUGE.get_scores([prediction], [ground_truth], avg=True)
            return float(scores["rouge-l"]["f"])
        except Exception:
            pass
    return rouge_l_single(prediction, ground_truth)


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_classes = kwargs.get("all_classes") or []
    em_matches = [cls for cls in all_classes if cls in prediction]
    filtered_matches = [
        cls for cls in em_matches if not (cls in ground_truth and cls != ground_truth)
    ]
    if filtered_matches:
        return 1.0 / len(filtered_matches) if ground_truth in filtered_matches else 0.0
    if not all_classes:
        return float(normalize_answer(prediction) == normalize_answer(ground_truth))

    best_match = None
    highest_similarity = 0.0
    for candidate in all_classes:
        similarity = difflib.SequenceMatcher(None, candidate, prediction).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = candidate
    return 1.0 if best_match == ground_truth else 0.0


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"Paragraph\s*(\d+)", ground_truth)
    target_id = matches[0] if matches else None
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    correct = sum(1 for num in numbers if target_id is not None and num == target_id)
    return correct / len(numbers)


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            candidate = line
            break
    if not candidate:
        candidate = prediction
    if fuzzy_fuzz is not None:
        return fuzzy_fuzz.ratio(candidate, ground_truth) / 100.0
    return difflib.SequenceMatcher(None, candidate, ground_truth).ratio()


def exact_match_metric(prediction: str, ground_truth: str, **kwargs) -> float:
    return exact_match_accuracy(prediction, [ground_truth])







LONG_BENCH_TASKS = {
    "NarrativeQA":   {"subset": "narrativeqa",        "metric": "f1"},
    "Qasper":        {"subset": "qasper",             "metric": "f1"},
    "MultiFieldQA":  {"subset": "multifieldqa_en",    "metric": "f1"},
    "HotpotQA":      {"subset": "hotpotqa",           "metric": "f1"},
    "MuSiQue":       {"subset": "musique",            "metric": "f1"},
    "2WikiMQA":      {"subset": "2wikimqa",           "metric": "f1"},
    "GovReport":     {"subset": "gov_report",         "metric": "rougeL"},
    "QMSum":         {"subset": "qmsum",              "metric": "rougeL"},
    "MultiNews":     {"subset": "multi_news",         "metric": "rougeL"},
    "LCC":           {"subset": "lcc",                "metric": "edit_sim"},
    "RepoBench-P":   {"subset": "repobench-p",        "metric": "edit_sim"},
    "TriviaQA":      {"subset": "triviaqa",           "metric": "f1"},
    "SAMSum":        {"subset": "samsum",             "metric": "rougeL"},
    "TREC":          {"subset": "trec",               "metric": "accuracy"},
    "PR":            {"subset": "passage_retrieval_en", "metric": "accuracy"},
}

DEFAULT_TASK_ORDER = [
    "NarrativeQA",
    "Qasper",
    "MultiFieldQA",
    "HotpotQA",
    "MuSiQue",
    "2WikiMQA",
    "GovReport",
    "QMSum",
    "MultiNews",
    "LCC",
    "RepoBench-P",
    "TriviaQA",
    "SAMSum",
    "TREC",
    "PR",
]




DATASET_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

DATASET_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "musique": 32,
    "2wikimqa": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

SPECIAL_SINGLE_LINE_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}

DATASET_METRIC_FNS = {
    "narrativeqa": qa_f1_metric,
    "qasper": qa_f1_metric,
    "multifieldqa_en": qa_f1_metric,
    "hotpotqa": qa_f1_metric,
    "musique": qa_f1_metric,
    "2wikimqa": qa_f1_metric,
    "gov_report": rouge_l_metric,
    "qmsum": rouge_l_metric,
    "multi_news": rouge_l_metric,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "triviaqa": qa_f1_metric,
    "samsum": rouge_l_metric,
    "trec": classification_score,
    "passage_retrieval_en": retrieval_score,
}





def build_longbench_prompt(example: dict, subset: Optional[str] = None) -> str:

    instruction = example.get("input", "")
    context = example.get("context", "")
    all_classes = example.get("all_classes", None)

    if subset and subset in DATASET_PROMPTS:
        template = DATASET_PROMPTS[subset]
        return template.format(input=instruction, context=context)

    prompt = instruction.strip() + "\n\n" + context.strip() + "\n\nAnswer:"

    if all_classes:

        prompt += "\n\nPossible options:\n"
        for cls in all_classes:
            prompt += f"- {cls}\n"

    return prompt


@torch.no_grad()
def evaluate_longbench_task(
    model,
    tokenizer,
    device,
    dataset_id: str,
    subset: str,
    metric_type: str,
    max_input_tokens: int,
    max_new_tokens: int,
    max_test_samples: Optional[int] = None,
    desc_prefix: str = "",
) -> float:

    split_name = "test"


    data = load_dataset(
        dataset_id,
        subset,
        split=split_name,
        trust_remote_code=True,
    )

    if max_test_samples is not None and max_test_samples > 0:
        n = min(max_test_samples, len(data))
        data = data.select(range(n))

    scores = []
    pad_id = tokenizer.eos_token_id
    dataset_metric_fn = DATASET_METRIC_FNS.get(subset)
    fallback_metric_map = {
        "f1": qa_f1_metric,
        "rougeL": rouge_l_metric,
        "accuracy": exact_match_metric,
        "edit_sim": edit_similarity,
    }
    metric_fn = dataset_metric_fn or fallback_metric_map.get(metric_type)
    if metric_fn is None:
        raise ValueError(f"No metric registered for subset '{subset}' (metric_type={metric_type})")

    dataset_max_new = DATASET_MAX_NEW_TOKENS.get(subset, max_new_tokens)

    with temp_generation_overrides(model, temperature=0.0, top_p=1.0):
        pbar = tqdm(data, desc=f"{desc_prefix} [{subset}]")
        for ex in pbar:
            clear_group_cache()

            prompt = build_longbench_prompt(ex, subset=subset)
            answers = ex.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            all_classes = ex.get("all_classes", []) or []

            tokenized_prompt = tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt",
            ).input_ids[0]
            if tokenized_prompt.shape[-1] > max_input_tokens:
                half = max_input_tokens // 2
                prompt = tokenizer.decode(
                    tokenized_prompt[:half], skip_special_tokens=True
                ) + tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True
                )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            ).to(device)

            input_len = inputs["input_ids"].shape[1]

            try:
                _cuda_sync(device)
                start = perf_counter()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=dataset_max_new,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
                _cuda_sync(device)
                _ = perf_counter() - start
            except RuntimeError as e:
                print(f"[WARN] Generation failed on {subset}: {e}")
                scores.append(0.0)
                continue

            gen_ids = outputs[0, input_len:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if subset in SPECIAL_SINGLE_LINE_DATASETS:
                pred = pred.lstrip("\n").split("\n")[0]

            if not answers:
                scores.append(0.0)
                continue

            sample_score = 0.0
            for truth in answers:
                score_val = metric_fn(
                    pred,
                    truth,
                    all_classes=all_classes,
                )
                sample_score = max(sample_score, score_val)

            scores.append(sample_score)
            pbar.set_postfix({"score": f"{sample_score * 100:.2f}"})

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores) * 100.0)





def load_base_model_and_tokenizer(args):
    print(f"📥 Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"📥 Loading base FP16 model from HF: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )

    print(f"📦 Loading original weights from: {args.original_weights_path}")
    original_weights = torch.load(
        args.original_weights_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(original_weights)
    del original_weights
    gc.collect()

    return model, tokenizer



VARIANT_CONFIGS = {
    "fp16": {
        "use_fakequant": False,
        "use_ab": False,
        "act_bits": 0,
    },
    "w4a16_ab": {
        "use_fakequant": True,
        "use_ab": True,
        "act_bits": 0,
    },
    "w4a8_ab": {
        "use_fakequant": True,
        "use_ab": True,
        "act_bits": 8,
    },
    "w4a4_ab": {
        "use_fakequant": True,
        "use_ab": True,
        "act_bits": 4,
    },
}


def build_model_variant(
    base_model,
    variant: str,
    args,
    shared,
    bmap,
    module_names: List[str],
    activation_group_size: int,
):

    cfg = VARIANT_CONFIGS[variant]
    model = base_model


    if cfg["use_fakequant"]:
        print(f"📦 Loading fake-quant weights from: {args.quantized_weights_path}")
        fake_quant_weights = torch.load(
            args.quantized_weights_path, map_location="cpu", weights_only=True
        )
        apply_quantized_weights(model, fake_quant_weights)
        del fake_quant_weights
        gc.collect()


    if cfg["act_bits"] > 0:
        apply_activation_fake_quant(
            model,
            module_names,
            act_bits=cfg["act_bits"],
            group_size=activation_group_size,
        )


    if cfg["use_ab"]:
        print(f"🧩 Patching SVD correction wrappers (α=1.0, act_bits={cfg['act_bits']})")
        patch_svd_correction_wrappers(model, shared, bmap, alpha_svd=1.0)

    return model





def get_csv_writer(csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    fieldnames = ["method", "dataset", "score"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists or os.stat(csv_path).st_size == 0:
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())
    return f, writer





def parse_args():
    p = argparse.ArgumentParser(
        description="LongBench evaluation for FP16 vs W4A16+AB vs W4A8+AB vs W4A4+AB."
    )
    p.add_argument("--model_name", required=True)
    p.add_argument("--shared_path", required=True)
    p.add_argument("--bmap_path", required=True)
    p.add_argument(
        "--original_weights_path",
        required=True,
        help="Path to original FP16 weights (from step1).",
    )
    p.add_argument(
        "--quantized_weights_path",
        required=True,
        help="Path to fake-quant (dequantized) weights saved in step1 (W4).",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Required for some model families (e.g., Qwen). Also used for tokenizer.",
    )
    p.add_argument(
        "--longbench_dataset_id",
        default="THUDM/LongBench",
        help="HF dataset id for LongBench (default: THUDM/LongBench).",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "Subset of LongBench tasks to run. "
            "If omitted, uses default English set "
            "[NarrativeQA, Qasper, MultiFieldQA, HotpotQA, MuSiQue, 2WikiMQA, "
            "GovReport, QMSum, MultiNews, LCC, RepoBench-P, TriviaQA, SAMSum, TREC, PR]."
        ),
    )
    p.add_argument(
        "--max_input_tokens",
        type=int,
        default=4096,
        help="Maximum input tokens (prompt) for LongBench evaluation.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate for each example.",
    )
    p.add_argument(
        "--max_test_samples",
        type=int,
        default=0,
        help="Max samples per dataset (0 => all test samples).",
    )
    p.add_argument(
        "--csv_path",
        required=True,
        help="Path to CSV file where results (method, dataset, score) are appended.",
    )
    p.add_argument(
        "--activation_group_size",
        type=int,
        default=128,
        help="Group size for activation fake quantization (for W4A8/W4A4).",
    )
    p.add_argument(
        "--variant",
        choices=["fp16", "w4a16_ab", "w4a8_ab", "w4a4_ab"],
        default="fp16",
        help=(
            "Which method variant to evaluate: "
            "'fp16', 'w4a16_ab', 'w4a8_ab', or 'w4a4_ab'."
        ),
    )
    return p.parse_args()



def main():
    args = parse_args()
    device = torch.device(args.device)

    selected_tasks = args.tasks if args.tasks is not None else DEFAULT_TASK_ORDER
    for t in selected_tasks:
        if t not in LONG_BENCH_TASKS:
            raise ValueError(f"Unknown task '{t}'. Valid keys: {list(LONG_BENCH_TASKS.keys())}")

    csv_file, csv_writer = get_csv_writer(args.csv_path)


    base_model, tokenizer = load_base_model_and_tokenizer(args)
    max_ctx = getattr(base_model.config, "max_position_embeddings", args.max_input_tokens)
    max_input_tokens = min(args.max_input_tokens, max_ctx)


    shared_artifacts = torch.load(args.shared_path, map_location="cpu", weights_only=True)
    with open(args.bmap_path, "r") as f:
        bmap = json.load(f)
    module_names = [name.replace(".weight", "") for name in bmap.keys()]

    method_label_map = {
        "fp16": "fp16",
        "w4a16_ab": "W4A16+AB",
        "w4a8_ab": "W4A8+AB",
        "w4a4_ab": "W4A4+AB",
    }


    method_variants = [args.variant]

    try:
        for variant in method_variants:

            print("\n" + "=" * 80)
            print(f"🔎 Evaluating method = {method_label_map[variant]} ({variant})")
            print("=" * 80)


            print("🔄 Stripping previous wrappers (SVD / act quant) and resetting model to original FP16 weights...")
            strip_wrappers(base_model, module_names)
            original_weights = torch.load(
                args.original_weights_path, map_location="cpu", weights_only=True
            )
            base_model.load_state_dict(original_weights)
            del original_weights
            gc.collect()

            model = build_model_variant(
                base_model,
                variant=variant,
                args=args,
                shared=shared_artifacts,
                bmap=bmap,
                module_names=module_names,
                activation_group_size=args.activation_group_size,
            )

            model.to(device)
            model.eval()

            per_task_scores = []

            for task_name in selected_tasks:
                spec = LONG_BENCH_TASKS[task_name]
                subset = spec["subset"]
                metric_type = spec["metric"]
                print(f"\n▶ Task: {task_name} (subset={subset}, metric={metric_type})")

                max_samples = args.max_test_samples if args.max_test_samples > 0 else None

                score = evaluate_longbench_task(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    dataset_id=args.longbench_dataset_id,
                    subset=subset,
                    metric_type=metric_type,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=args.max_new_tokens,
                    max_test_samples=max_samples,
                    desc_prefix=f"{method_label_map[variant]}",
                )

                per_task_scores.append(score)
                print(f"✅ {task_name} ({method_label_map[variant]}) score = {score:.4f}")


                csv_writer.writerow(
                    {
                        "method": method_label_map[variant],
                        "dataset": task_name,
                        "score": f"{score:.6f}",
                    }
                )
                csv_file.flush()
                os.fsync(csv_file.fileno())


            if per_task_scores:
                avg_score = float(sum(per_task_scores) / len(per_task_scores))
            else:
                avg_score = 0.0

            print(f"\n📊 {method_label_map[variant]} Avg over {len(per_task_scores)} tasks = {avg_score:.4f}")

            csv_writer.writerow(
                {
                    "method": method_label_map[variant],
                    "dataset": "Avg",
                    "score": f"{avg_score:.6f}",
                }
            )
            csv_file.flush()
            os.fsync(csv_file.fileno())


            model.to("cpu")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        csv_file.close()
        del base_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n🎉 LongBench evaluation finished. Results saved to:", args.csv_path)


if __name__ == "__main__":
    main()
