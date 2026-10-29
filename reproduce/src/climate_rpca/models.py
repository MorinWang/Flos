from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .linalg import (
    adaptive_lorentz_shrink,
    estimate_rank_mp_tw,
    fractional_laplacian_filter,
    grid_diffusion,
    hard_threshold_topk,
    low_rank_approx,
    randomized_low_rank_approx,
    singular_value_threshold,
    soft_threshold,
    temporal_diffusion,
)


@dataclass
class ModelResult:
    method: str
    low_rank: np.ndarray
    sparse: np.ndarray
    rank: int
    iterations: int
    seconds: float
    peak_mb: float
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    rank: int | None = None
    max_rank: int = 12
    lam: float = 0.18
    tau: float = 0.45
    smooth: float = 0.25
    frac_s: float = 0.6
    lorentz_p: float = 1.35
    max_iter: int = 35
    min_iter: int = 4
    patience: int = 2
    proposed_min_iter: int = 10
    proposed_patience: int = 99
    tol: float = 1e-4
    auto_rank: bool = True
    svd_backend: str = "randomized"
    fast_fractional: bool = True
    godec_sparsity: float = 0.06
    cache_dir: Path | None = None


def _measure(method: str, fn: Callable[[], tuple[np.ndarray, np.ndarray, int, int, list[dict[str, float]]]]) -> ModelResult:
    tracemalloc.start()
    start = time.perf_counter()
    low_rank, sparse, rank, iterations, history = fn()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ModelResult(
        method=method,
        low_rank=low_rank,
        sparse=sparse,
        rank=int(rank),
        iterations=int(iterations),
        seconds=float(seconds),
        peak_mb=float(peak / (1024**2)),
        history=history,
    )


def run_eof(x: np.ndarray, cfg: ModelConfig) -> ModelResult:
    def body() -> tuple[np.ndarray, np.ndarray, int, int, list[dict[str, float]]]:
        rank = cfg.rank or estimate_rank_mp_tw(x, max_rank=cfg.max_rank)
        low_rank = low_rank_approx(x, rank, backend="full")
        return low_rank, x - low_rank, rank, 1, []

    return _measure("EOF-PCA", body)


def run_pcp(x: np.ndarray, cfg: ModelConfig) -> ModelResult:
    def body() -> tuple[np.ndarray, np.ndarray, int, int, list[dict[str, float]]]:
        norm_x = np.linalg.norm(x, ord="fro")
        lam = cfg.lam if cfg.lam > 0 else 1.0 / np.sqrt(max(x.shape))
        mu = 1.25 / max(np.linalg.norm(x, ord=2), 1e-8)
        mu_bar = mu * 1e7
        rho = 1.35
        y = x / max(np.linalg.norm(x, ord=2), np.linalg.norm(x.ravel(), ord=np.inf) / max(lam, 1e-8))
        low_rank = np.zeros_like(x)
        sparse = np.zeros_like(x)
        history: list[dict[str, float]] = []
        rank = 0
        for it in range(1, cfg.max_iter + 1):
            low_rank, rank, _ = singular_value_threshold(x - sparse + y / mu, 1.0 / mu)
            sparse = soft_threshold(x - low_rank + y / mu, lam / mu)
            residual = x - low_rank - sparse
            rel = float(np.linalg.norm(residual, ord="fro") / max(norm_x, 1e-8))
            history.append({"iter": it, "rel_residual": rel, "rank": float(rank)})
            y = y + mu * residual
            mu = min(mu * rho, mu_bar)
            if rel < cfg.tol:
                return low_rank, sparse, rank, it, history
        return low_rank, sparse, rank, cfg.max_iter, history

    return _measure("PCP-L1", body)


def _alternating_rpca(
    x: np.ndarray,
    cfg: ModelConfig,
    lat: np.ndarray,
    lon: np.ndarray,
    smooth_kind: str,
    sparse_kind: str,
    method_name: str,
) -> ModelResult:
    def body() -> tuple[np.ndarray, np.ndarray, int, int, list[dict[str, float]]]:
        norm_x = np.linalg.norm(x, ord="fro")
        rank = cfg.rank or estimate_rank_mp_tw(x, max_rank=cfg.max_rank)
        sparse = np.zeros_like(x)
        low_rank = low_rank_approx(x, rank, backend=cfg.svd_backend)
        history: list[dict[str, float]] = []
        smooth_x = None
        if smooth_kind == "fractional" and cfg.fast_fractional:
            smooth_x = fractional_laplacian_filter(x, lat, lon, s=cfg.frac_s, strength=cfg.smooth, cache_dir=cfg.cache_dir)
        best_rel = np.inf
        stale = 0
        min_iter = cfg.proposed_min_iter if method_name == "FLoS-RPCA" else cfg.min_iter
        patience = cfg.proposed_patience if method_name == "FLoS-RPCA" else cfg.patience
        for it in range(1, cfg.max_iter + 1):
            y = x - sparse
            if smooth_kind == "tv":
                y = grid_diffusion(y, lat, lon, strength=cfg.smooth, steps=2)
            elif smooth_kind == "sttv":
                y = grid_diffusion(y, lat, lon, strength=cfg.smooth, steps=1)
                y = temporal_diffusion(y, strength=cfg.smooth, steps=1)
            elif smooth_kind == "graph":
                y = fractional_laplacian_filter(y, lat, lon, s=1.0, strength=cfg.smooth, cache_dir=cfg.cache_dir)
            elif smooth_kind == "fractional":
                if smooth_x is None:
                    y = fractional_laplacian_filter(y, lat, lon, s=cfg.frac_s, strength=cfg.smooth, cache_dir=cfg.cache_dir)
                else:
                    # Linearized fast update: keep the costly fractional filter
                    # on X cached, then let sparse shrinkage absorb local updates.
                    y = smooth_x - sparse
                y = grid_diffusion(y, lat, lon, strength=min(cfg.smooth, 0.24), steps=1)
            low_rank = low_rank_approx(y, rank, backend=cfg.svd_backend)
            residual = x - low_rank
            if sparse_kind == "lorentz":
                sparse = adaptive_lorentz_shrink(residual, cfg.lam, p=cfg.lorentz_p)
            else:
                sparse = soft_threshold(residual, cfg.lam)
            fit_residual = x - low_rank - sparse
            rel = float(np.linalg.norm(fit_residual, ord="fro") / max(norm_x, 1e-8))
            history.append({"iter": it, "rel_residual": rel, "rank": float(rank)})
            if rel + 1e-8 < best_rel:
                best_rel = rel
                stale = 0
            else:
                stale += 1
            if it >= min_iter and (rel < cfg.tol or stale >= patience):
                return low_rank, sparse, rank, it, history
        return low_rank, sparse, rank, cfg.max_iter, history

    return _measure(method_name, body)


def run_tv_rpca(x: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ModelConfig) -> ModelResult:
    return _alternating_rpca(x, cfg, lat, lon, "tv", "l1", "TV-RPCA")


def run_graph_rpca(x: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ModelConfig) -> ModelResult:
    return _alternating_rpca(x, cfg, lat, lon, "graph", "l1", "Graph-RPCA")


def run_sttv_rpca(x: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ModelConfig) -> ModelResult:
    return _alternating_rpca(x, cfg, lat, lon, "sttv", "l1", "ST-TV-RPCA")


def run_frcctv_rpca(x: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ModelConfig) -> ModelResult:
    return _alternating_rpca(x, cfg, lat, lon, "fractional", "lorentz", "FLoS-RPCA")


def run_frcctv_l1(x: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ModelConfig) -> ModelResult:
    return _alternating_rpca(x, cfg, lat, lon, "fractional", "l1", "FLoS-L1")


def run_lorentz_no_fractional(x: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ModelConfig) -> ModelResult:
    return _alternating_rpca(x, cfg, lat, lon, "graph", "lorentz", "Graph-Lorentz")


def run_no_smooth_lorentz(x: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ModelConfig) -> ModelResult:
    return _alternating_rpca(x, cfg, lat, lon, "none", "lorentz", "LowRank-Lorentz")


def run_godec(x: np.ndarray, cfg: ModelConfig) -> ModelResult:
    def body() -> tuple[np.ndarray, np.ndarray, int, int, list[dict[str, float]]]:
        norm_x = np.linalg.norm(x, ord="fro")
        rank = cfg.rank or estimate_rank_mp_tw(x, max_rank=cfg.max_rank)
        k = int(max(1, round(cfg.godec_sparsity * x.size)))
        sparse = np.zeros_like(x)
        low_rank = np.zeros_like(x)
        history: list[dict[str, float]] = []
        best_rel = np.inf
        stale = 0
        for it in range(1, cfg.max_iter + 1):
            low_rank = randomized_low_rank_approx(x - sparse, rank, n_iter=1)
            sparse = hard_threshold_topk(x - low_rank, k)
            rel = float(np.linalg.norm(x - low_rank - sparse, ord="fro") / max(norm_x, 1e-8))
            history.append({"iter": it, "rel_residual": rel, "rank": float(rank)})
            if rel + 1e-8 < best_rel:
                best_rel = rel
                stale = 0
            else:
                stale += 1
            if it >= cfg.min_iter and (rel < cfg.tol or stale >= cfg.patience):
                return low_rank, sparse, rank, it, history
        return low_rank, sparse, rank, cfg.max_iter, history

    return _measure("GoDec-RPCA", body)


def run_named_model(name: str, x: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ModelConfig) -> ModelResult:
    dispatch = {
        "EOF-PCA": lambda: run_eof(x, cfg),
        "PCP-L1": lambda: run_pcp(x, cfg),
        "TV-RPCA": lambda: run_tv_rpca(x, lat, lon, cfg),
        "ST-TV-RPCA": lambda: run_sttv_rpca(x, lat, lon, cfg),
        "Graph-RPCA": lambda: run_graph_rpca(x, lat, lon, cfg),
        "FLoS-RPCA": lambda: run_frcctv_rpca(x, lat, lon, cfg),
        "FRCCTV-RPCA": lambda: run_frcctv_rpca(x, lat, lon, cfg),
        "FLoS-L1": lambda: run_frcctv_l1(x, lat, lon, cfg),
        "FRCCTV-L1": lambda: run_frcctv_l1(x, lat, lon, cfg),
        "Graph-Lorentz": lambda: run_lorentz_no_fractional(x, lat, lon, cfg),
        "LowRank-Lorentz": lambda: run_no_smooth_lorentz(x, lat, lon, cfg),
        "GoDec-RPCA": lambda: run_godec(x, cfg),
    }
    if name not in dispatch:
        raise KeyError(f"unknown model: {name}")
    return dispatch[name]()
