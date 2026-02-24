#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "configs"

ANSI_RESET = "\033[0m"
STEP_COLORS = {
    "step1": "\033[38;5;213m",  # pink
    "step2": "\033[38;5;208m",  # orange
    "step3": "\033[38;5;117m",  # sky blue
}


class _ColorizedStream:
    def __init__(self, stream, color_code: str):
        self._stream = stream
        self._color_code = color_code

    def write(self, text):
        if not text:
            return 0
        return self._stream.write(f"{self._color_code}{text}{ANSI_RESET}")

    def flush(self):
        return self._stream.flush()

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _load_toml(path: Path) -> dict:
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)

    for mod_name in ("tomli", "toml"):
        try:
            mod = importlib.import_module(mod_name)
            break
        except ModuleNotFoundError:
            mod = None
    if mod is None:
        raise RuntimeError(
            "TOML parser not found. Install `tomli` (Py<3.11) or use Python 3.11+."
        )

    with path.open("rb") as f:
        if mod.__name__ == "toml":
            return mod.loads(f.read().decode("utf-8"))
        return mod.load(f)


def _resolve_config_path(config_arg: str) -> Path:
    raw = Path(config_arg)
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(CONFIG_DIR / raw.name)
        if raw.suffix == "":
            candidates.append(CONFIG_DIR / f"{raw.name}.toml")

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(f"Config not found: {config_arg}\nSearched:\n{searched}")


def _cfg_get(cfg: dict, key: str, default=None):
    if key in cfg:
        return cfg[key]
    for section in ("glowq", "pipeline", "run", "step2", "step3"):
        sec = cfg.get(section)
        if isinstance(sec, dict) and key in sec:
            return sec[key]
    return default


def _require(cfg: dict, key: str):
    val = _cfg_get(cfg, key)
    if val is None:
        raise KeyError(f"Missing required config key: {key}")
    return val


def _ensure_src_on_path() -> None:
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


@contextlib.contextmanager
def _temporary_argv(argv: list[str]):
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


def _should_use_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return stream.isatty()
    except Exception:
        return False


@contextlib.contextmanager
def _step_color_output(step_name: str):
    color = STEP_COLORS.get(step_name)
    if not color or (not _should_use_color(sys.stdout) and not _should_use_color(sys.stderr)):
        yield
        return

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _ColorizedStream(old_stdout, color)
    sys.stderr = _ColorizedStream(old_stderr, color)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _import_src_module(module_name: str):
    _ensure_src_on_path()
    return importlib.import_module(module_name)


def _append_flag(argv: list[str], enabled: bool, flag: str):
    if enabled:
        argv.append(flag)


def _build_run_paths(run_root: Path) -> dict[str, Path]:
    step1_dir = run_root / "step1"
    step2_dir = run_root / "step2"
    step1_dir.mkdir(parents=True, exist_ok=True)
    step2_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_root": run_root,
        "step1_dir": step1_dir,
        "step2_dir": step2_dir,
        "quant_err_path": step1_dir / "quant_error.pt",
        "original_weights_path": step1_dir / "original_weights.pt",
        "shared_path": step2_dir / "low_rank_shared.pt",
        "bmap_path": step2_dir / "b_ref_map.json",
    }


def _normalize_ppl_dataset(dataset_key: str) -> str:
    mapping = {
        "wiki": "wikitext2",
        "wikitext2": "wikitext2",
        "c4": "c4",
        "ptb": "ptb",
    }
    norm = str(dataset_key).strip().lower()
    if norm not in mapping:
        raise ValueError(
            f"Unsupported ppl_dataset='{dataset_key}'. Use one of: wiki, c4, ptb."
        )
    return mapping[norm]


def run_step1(cfg: dict, paths: dict[str, Path]) -> None:
    with _step_color_output("step1"):
        step1 = _import_src_module("step1_quantize_error_integrated")
        argv = [
            "step1_quantize_error_integrated.py",
            "--model_name",
            str(_require(cfg, "model_name")),
            "--out_quant_err",
            str(paths["quant_err_path"]),
            "--out_original_weights",
            str(paths["original_weights_path"]),
            "--device",
            str(_cfg_get(cfg, "device", "cuda")),
            "--group_size",
            str(int(_cfg_get(cfg, "group_size", 128))),
            "--seed",
            str(int(_cfg_get(cfg, "seed", 42))),
        ]
        _append_flag(argv, bool(_cfg_get(cfg, "trust_remote_code", False)), "--trust_remote_code")

        print("\n[GlowQ] Step1 start: quantization error extraction")
        print(f"[GlowQ] Step1 outputs: {paths['quant_err_path']}, {paths['original_weights_path']}")
        with _temporary_argv(argv):
            step1.main()
        print("[GlowQ] Step1 done")


def run_step2(cfg: dict, paths: dict[str, Path]) -> None:
    with _step_color_output("step2"):
        step2 = _import_src_module("step2_randomized_gsvd_integrated")
        cov_stats_path = _cfg_get(cfg, "cov_stats_path", None)
        args = argparse.Namespace(
            model_name=str(_require(cfg, "model_name")),
            err_path=str(paths["quant_err_path"]),
            output_path=str(paths["step2_dir"]),
            trust_remote_code=bool(_cfg_get(cfg, "trust_remote_code", False)),
            max_rank=int(_require(cfg, "svd_rank")),
            nsamples=int(_require(cfg, "calibration_n_samples")),
            calib_dataset=str(
                _cfg_get(cfg, "calibration_dataset", "DKYoon/SlimPajama-6B")
            ),
            calib_config=_cfg_get(cfg, "calibration_dataset_config", _cfg_get(cfg, "calib_config", None)),
            seqlen=int(_cfg_get(cfg, "calibration_seq_len", 2048)),
            shrinkage_alpha=float(_cfg_get(cfg, "shrinkage_alpha", 0.05)),
            cov_store_device=str(_cfg_get(cfg, "cov_store_device", "cpu")),
            oversamples=int(_cfg_get(cfg, "oversamples", 10)),
            power_iters=int(_cfg_get(cfg, "power_iters", 2)),
            cov_stats_path=(str(cov_stats_path) if cov_stats_path is not None else None),
            reuse_cov_stats=bool(_cfg_get(cfg, "reuse_cov_stats", False)),
            matmul_dtype=str(_cfg_get(cfg, "matmul_dtype", "float32")),
        )

        print("\n[GlowQ] Step2 start: randomized GSVD")
        print(
            f"[GlowQ] Step2 config: rank={args.max_rank}, calib_dataset={args.calib_dataset}, nsamples={args.nsamples}"
        )
        step2.main(args)
        print("[GlowQ] Step2 done")


def run_step3(cfg: dict, paths: dict[str, Path]) -> None:
    model_name = str(_require(cfg, "model_name"))
    trust_remote_code = bool(_cfg_get(cfg, "trust_remote_code", False))
    device = str(_cfg_get(cfg, "device", "cuda:0"))
    group_size = int(_cfg_get(cfg, "group_size", 128))
    use_cuda_w4a16 = bool(_cfg_get(cfg, "use_cuda_w4a16", False))
    lm_harness = bool(_require(cfg, "lm_harness"))
    ppl_dataset = str(_require(cfg, "ppl_dataset"))

    common_argv = [
        "--model_name",
        model_name,
        "--shared_path",
        str(paths["shared_path"]),
        "--bmap_path",
        str(paths["bmap_path"]),
        "--original_weights_path",
        str(paths["original_weights_path"]),
        "--device",
        device,
        "--group_size",
        str(group_size),
    ]
    # run_glowq.py focuses on final GlowQ results; skip Wq-only baseline measurement/output.
    common_argv.append("--skip_baseline_eval")
    if trust_remote_code:
        common_argv.append("--trust_remote_code")
    if use_cuda_w4a16:
        common_argv.append("--use_cuda_w4a16")

    with _step_color_output("step3"):
        if lm_harness:
            if _normalize_ppl_dataset(ppl_dataset) != "wikitext2":
                print(
                    f"[GlowQ] Note: lm_harness=true uses step3_decode_cache_lm.py (built-in PPL eval); ppl_dataset='{ppl_dataset}' is ignored in this mode."
                )
            step3_lm = _import_src_module("step3_decode_cache_lm")
            argv = ["step3_decode_cache_lm.py", *common_argv, "--enable_harness"]

            if bool(_cfg_get(cfg, "skip_gen", False)):
                argv.append("--skip_gen")
            for key, flag in (
                ("gen_do_sample", "--gen_do_sample"),
                ("clear_cache_before_harness", "--clear_cache_before_harness"),
            ):
                _append_flag(argv, bool(_cfg_get(cfg, key, False)), flag)
            for key in (
                "gen_max_new_tokens",
                "gen_repeats",
                "gen_num_beams",
                "gen_temperature",
                "gen_top_p",
                "harness_tasks",
                "harness_batch_size",
                "harness_num_fewshot",
                "harness_limit",
                "save_harness_results",
            ):
                value = _cfg_get(cfg, key, None)
                if value is not None:
                    argv.extend([f"--{key}", str(value)])

            print("\n[GlowQ] Step3 start: step3_decode_cache_lm.py (lm-eval-harness enabled)")
            with _temporary_argv(argv):
                step3_lm.main()
            print("[GlowQ] Step3 done")
            return

        step3_eval = _import_src_module("step3_decode_cache_nodict_eval")
        argv = [
            "step3_decode_cache_nodict_eval.py",
            *common_argv,
            "--eval_dataset",
            _normalize_ppl_dataset(ppl_dataset),
        ]

        if bool(_cfg_get(cfg, "skip_gen", False)):
            argv.append("--skip_gen")
        for key, flag in (("gen_do_sample", "--gen_do_sample"),):
            _append_flag(argv, bool(_cfg_get(cfg, key, False)), flag)
        for key in (
            "gen_max_new_tokens",
            "gen_repeats",
            "gen_num_beams",
            "gen_temperature",
            "gen_top_p",
            "eval_split",
            "eval_max_docs",
            "eval_max_chars",
            "eval_seq_len",
            "use_fast_tokenizer",
            "eval_hf_name",
            "eval_hf_config",
            "eval_text_field",
        ):
            value = _cfg_get(cfg, key, None)
            if value is not None:
                argv.extend([f"--{key}", str(value)])

        print(
            f"\n[GlowQ] Step3 start: step3_decode_cache_nodict_eval.py (ppl_dataset={ppl_dataset})"
        )
        with _temporary_argv(argv):
            step3_eval.main()
        print("[GlowQ] Step3 done")


def main():
    parser = argparse.ArgumentParser(
        description="Run GlowQ pipeline (step1 -> step2 -> step3) from a TOML config."
    )
    parser.add_argument(
        "config",
        help="TOML config path or config file name under GlowQ/configs (e.g., llama_3_2_3b.toml)",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    cfg = _load_toml(config_path)

    run_root_cfg = _cfg_get(cfg, "output_dir", None)
    if run_root_cfg is None:
        run_root = ROOT_DIR / "outputs" / config_path.stem
    else:
        run_root = Path(run_root_cfg)
        if not run_root.is_absolute():
            run_root = ROOT_DIR / run_root
    paths = _build_run_paths(run_root)

    print(f"[GlowQ] Config: {config_path}")
    print(f"[GlowQ] Run directory: {paths['run_root']}")

    run_step1(cfg, paths)
    run_step2(cfg, paths)
    run_step3(cfg, paths)

    print("\n[GlowQ] Pipeline completed")
    print(f"[GlowQ] Step1 quant err: {paths['quant_err_path']}")
    print(f"[GlowQ] Step1 original weights: {paths['original_weights_path']}")
    print(f"[GlowQ] Step2 shared tensors: {paths['shared_path']}")
    print(f"[GlowQ] Step2 B-map: {paths['bmap_path']}")


if __name__ == "__main__":
    main()
