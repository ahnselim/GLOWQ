#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
from pathlib import Path

from run_glowq import _load_toml, _resolve_config_path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
RESTORATION_DIR = SRC_DIR / "restoration"

ANSI_RESET = "\033[0m"
STEP_COLORS = {
    "step1": "\033[38;5;213m",  # pink
    "step2": "\033[38;5;208m",  # orange
    "step3": "\033[38;5;226m",  # yellow
    "step4": "\033[38;5;117m",  # sky blue
    "step5": "\033[38;5;183m",  # light purple
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


def _cfg_get(cfg: dict, key: str, default=None):
    if key in cfg:
        return cfg[key]
    for section in ("restoration", "glowq", "pipeline", "run", "step2", "step3"):
        sec = cfg.get(section)
        if isinstance(sec, dict) and key in sec:
            return sec[key]
    return default


def _require(cfg: dict, key: str):
    val = _cfg_get(cfg, key)
    if val is None:
        raise KeyError(f"Missing required config key: {key}")
    return val


def _importance_metrics_cfg(cfg: dict) -> str:
    for key in ("importance_metric", "importance_metrics", "restoration_importance_metrics"):
        val = _cfg_get(cfg, key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            return ",".join(str(x) for x in val)
        return str(val)
    return "gsvd,norm_error"


def _ensure_restoration_on_path() -> None:
    for p in (RESTORATION_DIR, SRC_DIR):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


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


def _import_restoration_module(name: str):
    _ensure_restoration_on_path()
    return importlib.import_module(name)


def _import_src_module(name: str):
    _ensure_restoration_on_path()
    return importlib.import_module(name)


def _build_paths(cfg: dict, config_stem: str) -> dict[str, Path]:
    base_output = _cfg_get(cfg, "restoration_output_dir")
    if base_output is None:
        normal_output = _cfg_get(cfg, "output_dir")
        if normal_output is not None:
            run_root = Path(normal_output)
            if not run_root.is_absolute():
                run_root = ROOT_DIR / run_root
            run_root = run_root / "restoration"
        else:
            run_root = ROOT_DIR / "outputs" / config_stem / "restoration"
    else:
        run_root = Path(base_output)
        if not run_root.is_absolute():
            run_root = ROOT_DIR / run_root

    dirs = {
        "run_root": run_root,
        "step1_dir": run_root / "step1",
        "step2_dir": run_root / "step2",
        "step3_1_dir": run_root / "step3_1",
        "step4_dir": run_root / "step4",
        "step5_dir": run_root / "step5",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    dirs.update(
        {
            "quant_err_path": dirs["step1_dir"] / "quant_error.pt",
            "original_weights_path": dirs["step1_dir"] / "original_weights.pt",
            "shared_path": dirs["step2_dir"] / "low_rank_shared.pt",
            "bmap_path": dirs["step2_dir"] / "b_ref_map.json",
            "rankings_json": dirs["step3_1_dir"] / "importance_rankings.json",
            "step4_csv_path": dirs["step4_dir"] / "cumulative_results.csv",
            "step5_plot_path": dirs["step5_dir"] / "final_ppl_comparison_plot.png",
        }
    )
    return dirs


def run_step1(cfg: dict, paths: dict[str, Path]) -> None:
    with _step_color_output("step1"):
        mod = _import_src_module("step1_quantize")
        argv = [
            "step1_quantize.py",
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
        if bool(_cfg_get(cfg, "trust_remote_code", False)):
            argv.append("--trust_remote_code")

        print("\n[GlowQ-S] Step1 start: quantization error extraction (src)")
        with _temporary_argv(argv):
            mod.main()
        print("[GlowQ-S] Step1 done")


def run_step2(cfg: dict, paths: dict[str, Path]) -> None:
    with _step_color_output("step2"):
        mod = _import_src_module("step2_rsvd")
        cov_stats_path = _cfg_get(
            cfg,
            "restoration_cov_stats_path",
            _cfg_get(cfg, "cov_stats_path", None),
        )
        args = argparse.Namespace(
            model_name=str(_require(cfg, "model_name")),
            err_path=str(paths["quant_err_path"]),
            output_path=str(paths["step2_dir"]),
            trust_remote_code=bool(_cfg_get(cfg, "trust_remote_code", False)),
            max_rank=int(_require(cfg, "svd_rank")),
            nsamples=int(_cfg_get(cfg, "restoration_calibration_n_samples", _require(cfg, "calibration_n_samples"))),
            seqlen=int(_cfg_get(cfg, "restoration_calibration_seq_len", _cfg_get(cfg, "calibration_seq_len", 2048))),
            shrinkage_alpha=float(_cfg_get(cfg, "restoration_shrinkage_alpha", 0.05)),
            calib_dataset=str(_cfg_get(cfg, "restoration_calibration_dataset", _cfg_get(cfg, "calibration_dataset", "DKYoon/SlimPajama-6B"))),
            calib_config=_cfg_get(
                cfg,
                "restoration_calibration_dataset_config",
                _cfg_get(cfg, "calibration_dataset_config", _cfg_get(cfg, "calib_config", None)),
            ),
            cov_store_device=str(
                _cfg_get(cfg, "restoration_cov_store_device", _cfg_get(cfg, "cov_store_device", "cpu"))
            ),
            oversamples=int(_cfg_get(cfg, "restoration_oversamples", _cfg_get(cfg, "oversamples", 10))),
            power_iters=int(_cfg_get(cfg, "restoration_power_iters", _cfg_get(cfg, "power_iters", 2))),
            cov_stats_path=(str(cov_stats_path) if cov_stats_path is not None else None),
            reuse_cov_stats=bool(
                _cfg_get(cfg, "restoration_reuse_cov_stats", _cfg_get(cfg, "reuse_cov_stats", False))
            ),
            matmul_dtype=str(
                _cfg_get(cfg, "restoration_matmul_dtype", _cfg_get(cfg, "matmul_dtype", "float32"))
            ),
        )

        print("\n[GlowQ-S] Step2 start: randomized GSVD (src integrated)")
        print(
            f"[GlowQ-S] Step2 config: rank={args.max_rank}, calib_dataset={args.calib_dataset}, nsamples={args.nsamples}"
        )
        mod.main(args)
        print("[GlowQ-S] Step2 done")


def run_step3_1(cfg: dict, paths: dict[str, Path]) -> None:
    with _step_color_output("step3"):
        mod = _import_restoration_module("step3_1_calculate_importance")
        paths["rankings_json"].parent.mkdir(parents=True, exist_ok=True)
        args = argparse.Namespace(
            err_path=str(paths["quant_err_path"]),
            original_weights_path=str(paths["original_weights_path"]),
            shared_path=str(paths["shared_path"]),
            bmap_path=str(paths["bmap_path"]),
            output_json=str(_cfg_get(cfg, "restoration_rankings_json", paths["rankings_json"])),
            metrics=_importance_metrics_cfg(cfg),
            include_component_rankings=bool(
                _cfg_get(cfg, "restoration_include_component_rankings", True)
            ),
            include_layer_order=bool(_cfg_get(cfg, "restoration_include_layer_order", True)),
            device=str(_cfg_get(cfg, "device", "auto")),
        )
        print("\n[GlowQ-S] Step3_1 start: calculate importance rankings")
        print(f"[GlowQ-S] Step3_1 metrics: {args.metrics}")
        mod.main(args)
        print("[GlowQ-S] Step3_1 done")
        paths["rankings_json"] = Path(args.output_json)


def run_step4(cfg: dict, paths: dict[str, Path]) -> None:
    with _step_color_output("step4"):
        mod = _import_restoration_module("step4_evaluate_cumulative")
        step4_output_dir = Path(_cfg_get(cfg, "restoration_step4_output_dir", paths["step4_dir"]))
        if not step4_output_dir.is_absolute():
            step4_output_dir = ROOT_DIR / step4_output_dir
        step4_output_dir.mkdir(parents=True, exist_ok=True)

        args = argparse.Namespace(
            model_name=str(_require(cfg, "model_name")),
            original_weights_path=str(paths["original_weights_path"]),
            shared_path=str(paths["shared_path"]),
            bmap_path=str(paths["bmap_path"]),
            rankings_json=str(paths["rankings_json"]),
            output_dir=str(step4_output_dir),
            device=str(_cfg_get(cfg, "device", "cuda:0")),
            trust_remote_code=bool(_cfg_get(cfg, "trust_remote_code", False)),
        )
        print("\n[GlowQ-S] Step4 start: cumulative evaluation")
        mod.main(args)
        print("[GlowQ-S] Step4 done")
        paths["step4_dir"] = step4_output_dir
        paths["step4_csv_path"] = step4_output_dir / "cumulative_results.csv"


def run_step5(cfg: dict, paths: dict[str, Path]) -> None:
    with _step_color_output("step5"):
        mod = _import_restoration_module("step5_plot_comparison")
        output_plot = Path(_cfg_get(cfg, "restoration_step5_output_plot", paths["step5_plot_path"]))
        if not output_plot.is_absolute():
            output_plot = ROOT_DIR / output_plot
        output_plot.parent.mkdir(parents=True, exist_ok=True)

        args = argparse.Namespace(
            csv_path=str(paths["step4_csv_path"]),
            output_plot=str(output_plot),
        )
        print("\n[GlowQ-S] Step5 start: final comparison plot")
        mod.main(args)
        print("[GlowQ-S] Step5 done")
        paths["step5_plot_path"] = output_plot


def main():
    p = argparse.ArgumentParser(
        description="Run GlowQ restoration pipeline (step1 -> step2 -> step3_1 -> step4 -> step5) from a TOML config."
    )
    p.add_argument(
        "config",
        help="TOML config path or config file name under GlowQ/configs (e.g., llama_3_2_3b.toml)",
    )
    args = p.parse_args()

    config_path = _resolve_config_path(args.config)
    cfg = _load_toml(config_path)
    paths = _build_paths(cfg, config_path.stem)

    print(f"[GlowQ-S] Config: {config_path}")
    print(f"[GlowQ-S] Run directory: {paths['run_root']}")

    run_step1(cfg, paths)
    run_step2(cfg, paths)
    run_step3_1(cfg, paths)
    run_step4(cfg, paths)
    run_step5(cfg, paths)

    print("\n[GlowQ-S] Pipeline completed")
    print(f"[GlowQ-S] Step1 quant err: {paths['quant_err_path']}")
    print(f"[GlowQ-S] Step1 original weights: {paths['original_weights_path']}")
    print(f"[GlowQ-S] Step2 shared tensors: {paths['shared_path']}")
    print(f"[GlowQ-S] Step2 B-map: {paths['bmap_path']}")
    print(f"[GlowQ-S] Step3_1 rankings: {paths['rankings_json']}")
    print(f"[GlowQ-S] Step4 CSV: {paths['step4_csv_path']}")
    print(f"[GlowQ-S] Step5 plot: {paths['step5_plot_path']}")


if __name__ == "__main__":
    main()
