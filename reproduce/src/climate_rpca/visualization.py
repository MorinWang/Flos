from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "EOF-PCA": "#8a8f98",
    "GoDec-RPCA": "#a96a36",
    "PCP-L1": "#4f7f9f",
    "TV-RPCA": "#d08b2c",
    "ST-TV-RPCA": "#be4b5b",
    "Graph-RPCA": "#4f9a72",
    "FLoS-RPCA": "#1f5f8b",
}

METHOD_ORDER = [
    "EOF-PCA",
    "GoDec-RPCA",
    "PCP-L1",
    "Graph-RPCA",
    "TV-RPCA",
    "ST-TV-RPCA",
    "FLoS-RPCA",
]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bar_positions(n_groups: int, n_bars: int) -> tuple[np.ndarray, float]:
    x = np.arange(n_groups)
    width = min(0.82 / max(n_bars, 1), 0.14)
    return x, width


def _ordered_methods(methods: list[str]) -> list[str]:
    ordered = [m for m in METHOD_ORDER if m in methods]
    extras = [m for m in methods if m not in ordered]
    return ordered + extras


def plot_overall(results_dir: Path, figures_dir: Path) -> None:
    df = pd.read_csv(results_dir / "overall_summary.csv")
    datasets = list(df["dataset"].drop_duplicates())
    methods = _ordered_methods(list(df["method"].drop_duplicates()))
    x, width = _bar_positions(len(datasets), len(methods))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("dataset")
        vals = [sub.loc[d, "sparse_auprc_mean"] for d in datasets]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=method, color=COLORS.get(method, None))
    ax.set_title("Sparse Extreme Detection (AUPRC)")
    ax.set_ylabel("AUPRC higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    _save(fig, figures_dir / "overall_sparse_auprc.png")


def plot_efficiency(results_dir: Path, figures_dir: Path) -> None:
    df = pd.read_csv(results_dir / "efficiency.csv")
    methods = list(df["method"])
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(x, df["seconds_mean"], yerr=df["seconds_std"], capsize=3, color=[COLORS.get(m, "#777777") for m in methods])
    axes[0].set_title("Runtime")
    axes[0].set_ylabel("Seconds lower is better")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, df["peak_mb_mean"], yerr=df["peak_mb_std"], capsize=3, color=[COLORS.get(m, "#777777") for m in methods])
    axes[1].set_title("Peak Memory")
    axes[1].set_ylabel("MB lower is better")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, figures_dir / "efficiency_runtime_memory.png")


def plot_parameter(results_dir: Path, figures_dir: Path) -> None:
    df = pd.read_csv(results_dir / "parameter_analysis.csv")
    params = list(df["parameter"].drop_duplicates())
    fig, axes = plt.subplots(1, len(params), figsize=(13.5, 4.2), sharey=True)
    if len(params) == 1:
        axes = [axes]
    for ax, param in zip(axes, params):
        sub = df[df["parameter"] == param].sort_values("value")
        ax.errorbar(sub["value"], sub["sparse_auprc_mean"], yerr=sub["sparse_auprc_std"], marker="o", color="#1f5f8b")
        ax.set_title(param)
        ax.set_xlabel("value")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Sparse AUPRC higher is better")
    _save(fig, figures_dir / "parameter_analysis.png")


def plot_transferability(results_dir: Path, figures_dir: Path) -> None:
    df = pd.read_csv(results_dir / "transferability.csv")
    datasets = list(df["dataset"].drop_duplicates())
    settings = list(df["setting"].drop_duplicates())
    x, width = _bar_positions(len(datasets), len(settings))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, setting in enumerate(settings):
        sub = df[df["setting"] == setting].set_index("dataset")
        vals = [sub.loc[d, "sparse_auprc_mean"] for d in datasets]
        errs = [sub.loc[d, "sparse_auprc_std"] for d in datasets]
        ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=2, label=setting)
    ax.set_title("Transferability")
    ax.set_ylabel("Sparse AUPRC higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, figures_dir / "transferability.png")


def plot_curve(results_dir: Path, figures_dir: Path) -> None:
    df = pd.read_csv(results_dir / "curve_fitting.csv")
    datasets = list(df["dataset"].drop_duplicates())
    methods = list(df["method"].drop_duplicates())
    x, width = _bar_positions(len(datasets), len(methods))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("dataset")
        vals = [sub.loc[d, "curve_corr_mean"] for d in datasets]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=method, color=COLORS.get(method, None))
    ax.set_title("Temporal Energy Curve Fit")
    ax.set_ylabel("Correlation higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    _save(fig, figures_dir / "curve_fitting_corr.png")


def plot_case(results_dir: Path, figures_dir: Path) -> None:
    df = pd.read_csv(results_dir / "case_study.csv")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x, df["true_sparse"], marker="o", label="truth", color="#111111", linewidth=2)
    ax.plot(x, df["proposed_sparse"], marker="s", label="FLoS-RPCA", color=COLORS["FLoS-RPCA"], linewidth=2)
    ax.plot(x, df["baseline_sparse"], marker="^", label=str(df["best_baseline"].iloc[0]), color=COLORS["Graph-RPCA"], linewidth=2)
    ax.set_title("Case Study: Top Injected Extremes")
    ax.set_xlabel("case rank")
    ax.set_ylabel("sparse amplitude")
    ax.set_xticks(x)
    ax.set_xticklabels(df["case_rank"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, figures_dir / "case_study_sparse_recovery.png")


def write_visualizations(out_dir: Path) -> None:
    results_dir = out_dir / "results"
    figures_dir = out_dir / "reports" / "figures"
    plot_overall(results_dir, figures_dir)
    plot_efficiency(results_dir, figures_dir)
    plot_parameter(results_dir, figures_dir)
    plot_transferability(results_dir, figures_dir)
    plot_curve(results_dir, figures_dir)
    plot_case(results_dir, figures_dir)
