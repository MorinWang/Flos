from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .benchmark import BenchmarkInstance, dataset_seed, make_heavy_tail_benchmark
from .datasets import latitude_bands, load_dataset_config, prepare_climate_matrix, spatial_index_to_coord
from .metrics import curve_metrics, evaluate_decomposition, summarize_mean_std
from .models import ModelConfig, ModelResult, run_named_model
from .reporting import write_report
from .visualization import write_visualizations


PROFILE = {
    "smoke": {
        "datasets": ["noaa_ncep_air"],
        "lat_size": 8,
        "lon_size": 16,
        "time_steps": 72,
        "seeds": [0],
        "max_iter": 5,
        "max_rank": 5,
        "param_seeds": [0],
    },
    "standard": {
        "datasets": ["noaa_ncep_air", "noaa_cmap_precip", "noaa_ncep_pressure"],
        "lat_size": 18,
        "lon_size": 36,
        "time_steps": 180,
        "seeds": [0, 1, 2, 3, 4],
        "max_iter": 10,
        "max_rank": 10,
        "param_seeds": [0, 1, 2, 3, 4],
    },
    "paper": {
        "datasets": ["noaa_ncep_air", "noaa_cmap_precip", "noaa_ncep_pressure"],
        "lat_size": 24,
        "lon_size": 48,
        "time_steps": 240,
        "seeds": [0, 1, 2, 3, 4],
        "max_iter": 16,
        "max_rank": 12,
        "param_seeds": [0, 1, 2],
    },
    "broad": {
        "datasets": [
            "noaa_ncep_air",
            "noaa_ncep_pressure",
            "noaa_cmap_precip",
            "noaa_ncep_mslp",
            "noaa_ncep_omega_sig995",
            "noaa_ncep_pottmp_sig995",
            "noaa_ncep_rhum_sig995",
            "noaa_ncep_lftx",
        ],
        "lat_size": 18,
        "lon_size": 36,
        "time_steps": 180,
        "seeds": [0, 1, 2, 3, 4],
        "max_iter": 10,
        "max_rank": 10,
        "param_seeds": [0, 1, 2, 3, 4],
    },
}

OVERALL_METHODS = ["EOF-PCA", "GoDec-RPCA", "PCP-L1", "TV-RPCA", "ST-TV-RPCA", "Graph-RPCA", "FLoS-RPCA"]


def _matrix_cache_path(data_root: Path, name: str, lat_size: int, lon_size: int, time_steps: int) -> Path:
    return data_root / "processed" / f"{name}_lat{lat_size}_lon{lon_size}_t{time_steps}.npz"


def load_or_prepare_matrix(
    name: str,
    spec: dict,
    data_root: Path,
    lat_size: int,
    lon_size: int,
    time_steps: int,
):
    cache = _matrix_cache_path(data_root, name, lat_size, lon_size, time_steps)
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        from .datasets import ClimateMatrix

        return ClimateMatrix(
            name=name,
            title=str(data["title"]),
            matrix=data["matrix"],
            lat=data["lat"],
            lon=data["lon"],
            time=data["time"],
            scale=float(data["scale"]),
            source_path=Path(str(data["source_path"])),
        )
    matrix = prepare_climate_matrix(name, spec, data_root, lat_size, lon_size, time_steps)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        title=matrix.title,
        matrix=matrix.matrix,
        lat=matrix.lat,
        lon=matrix.lon,
        time=matrix.time,
        scale=matrix.scale,
        source_path=str(matrix.source_path),
    )
    return matrix


def _base_cfg(profile: dict, data_root: Path) -> ModelConfig:
    return ModelConfig(
        lam=0.18,
        smooth=0.85,
        frac_s=0.25,
        lorentz_p=1.25,
        max_iter=int(profile["max_iter"]),
        min_iter=4,
        patience=2,
        proposed_min_iter=10,
        proposed_patience=99,
        max_rank=int(profile["max_rank"]),
        tol=5e-4,
        svd_backend="randomized",
        fast_fractional=False,
        cache_dir=data_root / "cache",
    )


def _benchmark(matrix, seed: int) -> BenchmarkInstance:
    return make_heavy_tail_benchmark(
        matrix.matrix,
        matrix.lat,
        matrix.lon,
        seed=dataset_seed(seed, matrix.name),
        rank_true=6,
        event_rate=0.07,
        cluster_len=4,
        spatial_radius_deg=20.0,
        tail_alpha=1.35,
        dense_noise_scale=0.035,
    )


def _result_row(dataset: str, seed: int, method_result: ModelResult, bench: BenchmarkInstance) -> dict[str, float | int | str]:
    metrics = evaluate_decomposition(
        method_result.low_rank,
        method_result.sparse,
        bench.low_rank,
        bench.sparse,
        bench.support,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method_result.method,
        "rank": method_result.rank,
        "iterations": method_result.iterations,
        "seconds": method_result.seconds,
        "peak_mb": method_result.peak_mb,
        **metrics,
    }


def run_overall(matrices: dict, seeds: list[int], cfg: ModelConfig) -> tuple[pd.DataFrame, dict[tuple[str, int], BenchmarkInstance], dict[tuple[str, int, str], ModelResult]]:
    rows: list[dict[str, float | int | str]] = []
    benches: dict[tuple[str, int], BenchmarkInstance] = {}
    model_outputs: dict[tuple[str, int, str], ModelResult] = {}
    for dataset, matrix in matrices.items():
        for seed in seeds:
            bench = _benchmark(matrix, seed)
            benches[(dataset, seed)] = bench
            for method in OVERALL_METHODS:
                print(f"[overall] {dataset} seed={seed} method={method}", flush=True)
                result = run_named_model(method, bench.observed, matrix.lat, matrix.lon, cfg)
                model_outputs[(dataset, seed, method)] = result
                rows.append(_result_row(dataset, seed, result, bench))
    return pd.DataFrame(rows), benches, model_outputs


def run_parameter_analysis(matrix, seeds: list[int], cfg: ModelConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for param_name, values in {
        "frac_s": [0.25, 0.4, 0.6, 0.8, 1.0],
        "lorentz_p": [1.1, 1.25, 1.35, 1.55, 1.8],
        "smooth": [0.05, 0.15, 0.32, 0.55, 0.85],
    }.items():
        for value in values:
            cfg_i = replace(cfg, **{param_name: value})
            for seed in seeds:
                bench = _benchmark(matrix, seed)
                result = run_named_model("FLoS-RPCA", bench.observed, matrix.lat, matrix.lon, cfg_i)
                row = _result_row(matrix.name, seed, result, bench)
                row["parameter"] = param_name
                row["value"] = value
                rows.append(row)
    raw = pd.DataFrame(rows)
    summary = summarize_mean_std(raw, ["dataset", "parameter", "value"], ["sparse_auprc", "sparse_f1", "low_rank_relerr"])
    return summary.sort_values(["parameter", "value"])


def run_ablation(matrix, seeds: list[int], cfg: ModelConfig) -> pd.DataFrame:
    methods = ["FLoS-RPCA", "FLoS-L1", "Graph-Lorentz", "Graph-RPCA", "LowRank-Lorentz", "ST-TV-RPCA", "EOF-PCA"]
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        bench = _benchmark(matrix, seed)
        for method in methods:
            result = run_named_model(method, bench.observed, matrix.lat, matrix.lon, cfg)
            row = _result_row(matrix.name, seed, result, bench)
            row["ablation"] = {
                "FLoS-RPCA": "full",
                "FLoS-L1": "replace Lorentz with L1",
                "Graph-Lorentz": "replace fractional with integer graph",
                "Graph-RPCA": "remove fractional and Lorentz",
                "LowRank-Lorentz": "remove spherical smoothing",
                "ST-TV-RPCA": "replace fractional geometry with spatio-temporal TV",
                "EOF-PCA": "remove robust sparse module",
            }[method]
            rows.append(row)
    raw = pd.DataFrame(rows)
    return summarize_mean_std(raw, ["dataset", "method", "ablation"], ["sparse_auprc", "sparse_f1", "low_rank_relerr"])


def run_efficiency(overall_raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_mean_std(overall_raw, ["method"], ["seconds", "peak_mb", "iterations"])


def run_compatibility(matrix, seeds: list[int], cfg: ModelConfig) -> pd.DataFrame:
    pairs = [
        ("EOF backbone", "EOF-PCA", "LowRank-Lorentz"),
        ("Graph backbone", "Graph-RPCA", "Graph-Lorentz"),
        ("Fractional backbone", "FLoS-L1", "FLoS-RPCA"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        bench = _benchmark(matrix, seed)
        for backbone, base, plugin in pairs:
            base_res = run_named_model(base, bench.observed, matrix.lat, matrix.lon, cfg)
            plugin_res = run_named_model(plugin, bench.observed, matrix.lat, matrix.lon, cfg)
            base_metrics = evaluate_decomposition(base_res.low_rank, base_res.sparse, bench.low_rank, bench.sparse, bench.support)
            plugin_metrics = evaluate_decomposition(plugin_res.low_rank, plugin_res.sparse, bench.low_rank, bench.sparse, bench.support)
            rows.append(
                {
                    "dataset": matrix.name,
                    "seed": seed,
                    "backbone": backbone,
                    "base_method": base,
                    "plugin_method": plugin,
                    "base_sparse_auprc": base_metrics["sparse_auprc"],
                    "plugin_sparse_auprc": plugin_metrics["sparse_auprc"],
                    "delta_sparse_auprc": plugin_metrics["sparse_auprc"] - base_metrics["sparse_auprc"],
                    "base_low_rank_relerr": base_metrics["low_rank_relerr"],
                    "plugin_low_rank_relerr": plugin_metrics["low_rank_relerr"],
                    "delta_low_rank_relerr": plugin_metrics["low_rank_relerr"] - base_metrics["low_rank_relerr"],
                }
            )
    return summarize_mean_std(
        pd.DataFrame(rows),
        ["dataset", "backbone", "base_method", "plugin_method"],
        ["base_sparse_auprc", "plugin_sparse_auprc", "delta_sparse_auprc", "delta_low_rank_relerr"],
    )


def _best_parameter(parameter_summary: pd.DataFrame, parameter: str, default: float) -> float:
    sub = parameter_summary[parameter_summary["parameter"] == parameter].copy()
    if sub.empty:
        return default
    sub["score"] = sub["sparse_auprc_mean"] - 0.25 * sub["low_rank_relerr_mean"]
    return float(sub.sort_values("score", ascending=False).iloc[0]["value"])


def run_transferability(matrices: dict, seeds: list[int], cfg: ModelConfig, parameter_summary: pd.DataFrame) -> pd.DataFrame:
    best_s = _best_parameter(parameter_summary, "frac_s", cfg.frac_s)
    best_p = _best_parameter(parameter_summary, "lorentz_p", cfg.lorentz_p)
    source_tuned = replace(cfg, frac_s=best_s, lorentz_p=best_p)
    rows: list[dict[str, float | int | str]] = []
    for dataset, matrix in matrices.items():
        for seed in seeds:
            bench = _benchmark(matrix, seed)
            for label, cfg_i in [("default", cfg), ("source_tuned_transfer", source_tuned)]:
                result = run_named_model("FLoS-RPCA", bench.observed, matrix.lat, matrix.lon, cfg_i)
                row = _result_row(dataset, seed, result, bench)
                row["setting"] = label
                row["transfer_frac_s"] = cfg_i.frac_s
                row["transfer_lorentz_p"] = cfg_i.lorentz_p
                rows.append(row)
    return summarize_mean_std(pd.DataFrame(rows), ["dataset", "setting", "transfer_frac_s", "transfer_lorentz_p"], ["sparse_auprc", "sparse_f1", "low_rank_relerr"])


def run_curve_fitting(model_outputs: dict, benches: dict, matrices: dict) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (dataset, seed, method), result in model_outputs.items():
        bench = benches[(dataset, seed)]
        rows.append({"dataset": dataset, "seed": seed, "method": method, **curve_metrics(result.low_rank, bench.low_rank)})
    return summarize_mean_std(pd.DataFrame(rows), ["dataset", "method"], ["curve_rmse", "curve_mae", "curve_corr"])


def run_feature_analysis(matrix, seeds: list[int], cfg: ModelConfig) -> pd.DataFrame:
    bands = latitude_bands(matrix.lat, matrix.lon, n_bands=6)
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        bench = _benchmark(matrix, seed)
        result = run_named_model("FLoS-RPCA", bench.observed, matrix.lat, matrix.lon, cfg)
        for band, spatial_mask in bands.items():
            mask2 = np.repeat(spatial_mask[:, None], bench.support.shape[1], axis=1)
            y = bench.support[mask2].astype(int)
            score = np.abs(result.sparse[mask2])
            auprc = float(average_precision_score(y, score)) if y.sum() > 0 else 0.0
            rows.append(
                {
                    "dataset": matrix.name,
                    "seed": seed,
                    "feature_group": f"latitude {band}",
                    "support_count": int(y.sum()),
                    "sparse_auprc": auprc,
                    "mean_abs_sparse_score": float(np.mean(score)),
                }
            )
    return summarize_mean_std(pd.DataFrame(rows), ["dataset", "feature_group"], ["sparse_auprc", "mean_abs_sparse_score", "support_count"])


def run_case_study(matrix, cfg: ModelConfig) -> pd.DataFrame:
    seed = 0
    bench = _benchmark(matrix, seed)
    proposed = run_named_model("FLoS-RPCA", bench.observed, matrix.lat, matrix.lon, cfg)
    baseline = run_named_model("Graph-RPCA", bench.observed, matrix.lat, matrix.lon, cfg)
    diff = np.abs(proposed.sparse) - np.abs(baseline.sparse)
    flat_order = np.argsort(np.abs(bench.sparse).ravel())[::-1][:12]
    rows: list[dict[str, float | int | str]] = []
    for rank, flat in enumerate(flat_order, start=1):
        spatial_idx, time_idx = np.unravel_index(flat, bench.sparse.shape)
        lat, lon = spatial_index_to_coord(int(spatial_idx), matrix.lat, matrix.lon)
        rows.append(
            {
                "dataset": matrix.name,
                "seed": seed,
                "case_rank": rank,
                "time_index": int(time_idx),
                "time": str(matrix.time[int(time_idx)]) if int(time_idx) < len(matrix.time) else str(time_idx),
                "lat": lat,
                "lon": lon,
                "true_sparse": float(bench.sparse[spatial_idx, time_idx]),
                "proposed_sparse": float(proposed.sparse[spatial_idx, time_idx]),
                "best_baseline": "Graph-RPCA",
                "baseline_sparse": float(baseline.sparse[spatial_idx, time_idx]),
                "abs_score_gain": float(diff[spatial_idx, time_idx]),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE), default="standard")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.json"))
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = dict(PROFILE[args.profile])
    if args.datasets:
        profile["datasets"] = args.datasets

    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    specs = load_dataset_config(config_path)
    matrices = {}
    for name in profile["datasets"]:
        print(f"[data] preparing {name}", flush=True)
        matrices[name] = load_or_prepare_matrix(
            name,
            specs[name],
            args.data_root,
            int(profile["lat_size"]),
            int(profile["lon_size"]),
            int(profile["time_steps"]),
        )

    cfg = _base_cfg(profile, args.data_root)
    seeds = list(profile["seeds"])
    param_seeds = list(profile["param_seeds"])
    overall_raw, benches, model_outputs = run_overall(matrices, seeds, cfg)

    first_matrix = matrices[next(iter(matrices))]
    parameter = run_parameter_analysis(first_matrix, param_seeds, cfg)
    ablation = run_ablation(first_matrix, param_seeds, cfg)
    efficiency = run_efficiency(overall_raw)
    compatibility = run_compatibility(first_matrix, param_seeds, cfg)
    transferability = run_transferability(matrices, param_seeds, cfg, parameter)
    curve = run_curve_fitting(model_outputs, benches, matrices)
    feature = run_feature_analysis(first_matrix, param_seeds, cfg)
    case = run_case_study(first_matrix, cfg)

    write_report(
        args.out_dir,
        overall_raw,
        parameter,
        ablation,
        efficiency,
        compatibility,
        transferability,
        curve,
        feature,
        case,
    )
    write_visualizations(args.out_dir)
    print(f"[done] wrote results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
