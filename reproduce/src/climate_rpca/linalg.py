from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from sklearn.utils.extmath import randomized_svd


def _require_finite(x: np.ndarray, operation: str) -> np.ndarray:
    """Reject genuine numerical failures after BLAS operations.

    Some macOS BLAS builds emit spurious floating-point warnings for finite
    matrix products. We suppress those local warnings but still fail loudly if
    the computed array actually contains a non-finite value.
    """
    if not np.all(np.isfinite(x)):
        raise FloatingPointError(f"{operation} produced non-finite values")
    return x


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def adaptive_lorentz_shrink(x: np.ndarray, lam: float, p: float = 1.35, eps: float = 1e-6) -> np.ndarray:
    """Lorentz-inspired heavy-tail shrinkage.

    Large residuals receive weaker shrinkage than small residuals, which keeps
    extreme-event amplitudes from being flattened by an L1 penalty. This is the
    practical prox used in the prototype; the exact continuous Lorentz prox can
    be swapped in later without changing experiment orchestration.
    """
    abs_x = np.abs(x)
    med = np.median(abs_x)
    mad = np.median(np.abs(abs_x - med)) / 0.6745
    scale = mad if mad > 1e-8 else np.std(abs_x)
    scale = float(scale if scale > 1e-8 else 1.0)
    normalized = abs_x / scale
    adaptive = lam / np.power(normalized + eps, max(p - 1.0, 1e-3))
    return np.sign(x) * np.maximum(abs_x - adaptive, 0.0)


def singular_value_threshold(x: np.ndarray, tau: float) -> tuple[np.ndarray, int, np.ndarray]:
    u, s, vt = linalg.svd(x, full_matrices=False, overwrite_a=False, check_finite=False)
    shrunk = np.maximum(s - tau, 0.0)
    rank = int(np.sum(shrunk > 0))
    if rank == 0:
        return np.zeros_like(x), 0, s
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        low_rank = (u[:, :rank] * shrunk[:rank]) @ vt[:rank, :]
    return _require_finite(low_rank, "singular-value reconstruction"), rank, s


def truncated_svd(x: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = int(max(1, min(rank, min(x.shape) - 1)))
    u, s, vt = linalg.svd(x, full_matrices=False, overwrite_a=False, check_finite=False)
    return u[:, :rank], s[:rank], vt[:rank, :]


def randomized_low_rank_approx(x: np.ndarray, rank: int, n_iter: int = 1, random_state: int = 0) -> np.ndarray:
    rank = int(max(1, min(rank, min(x.shape) - 1)))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        u, s, vt = randomized_svd(
            x,
            n_components=rank,
            n_iter=n_iter,
            random_state=random_state,
            power_iteration_normalizer="auto",
        )
        low_rank = (u * s) @ vt
    return _require_finite(low_rank, "randomized SVD reconstruction")


def low_rank_approx(x: np.ndarray, rank: int, backend: str = "full") -> np.ndarray:
    if backend == "randomized":
        return randomized_low_rank_approx(x, rank)
    u, s, vt = truncated_svd(x, rank)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        low_rank = (u * s) @ vt
    return _require_finite(low_rank, "truncated SVD reconstruction")


def hard_threshold_topk(x: np.ndarray, k: int) -> np.ndarray:
    flat = np.abs(x).ravel()
    k = int(np.clip(k, 1, flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros_like(x).ravel()
    src = x.ravel()
    out[idx] = src[idx]
    return out.reshape(x.shape)


def estimate_rank_mp_tw(x: np.ndarray, max_rank: int = 12, min_rank: int = 2, tw_eta: float = 0.05) -> int:
    """Data-driven rank estimate using robust MP hard thresholding plus TW slack."""
    m, n = x.shape
    if min(m, n) <= 2:
        return 1
    _, s, _ = linalg.svd(x, full_matrices=False, overwrite_a=False, check_finite=False)
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median_s = float(np.median(s))
    # A small Tracy-Widom-inspired finite-size correction. The 0.95 quantile of
    # TW1 is about 0.98; this keeps the selector conservative for short series.
    tw_q95 = 0.98
    tw_slack = tw_q95 * median_s * (min(m, n) ** (-2.0 / 3.0)) * (1.0 + tw_eta)
    threshold = omega * median_s + tw_slack
    rank = int(np.sum(s > threshold))
    return int(np.clip(rank, min_rank, max_rank))


def _haversine_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * np.arctan2(np.sqrt(a), np.sqrt(max(1.0 - a, 0.0))))


def sphere_graph_laplacian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    n_lat, n_lon = len(lat), len(lon)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add_edge(a: int, b: int, w: float) -> None:
        rows.extend([a, b])
        cols.extend([b, a])
        vals.extend([w, w])

    for i in range(n_lat):
        for j in range(n_lon):
            idx = i * n_lon + j
            if i + 1 < n_lat:
                dst = (i + 1) * n_lon + j
                dist = _haversine_deg(float(lat[i]), float(lon[j]), float(lat[i + 1]), float(lon[j]))
                add_edge(idx, dst, 1.0 / max(dist, 1e-6))
            dst = i * n_lon + ((j + 1) % n_lon)
            dist = _haversine_deg(float(lat[i]), float(lon[j]), float(lat[i]), float(lon[(j + 1) % n_lon]))
            add_edge(idx, dst, 1.0 / max(dist, 1e-6))

    n = n_lat * n_lon
    w = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    degree = np.asarray(w.sum(axis=1)).ravel()
    lap = np.diag(degree) - w.toarray()
    scale = float(np.max(np.diag(lap)))
    return lap / scale if scale > 0 else lap


def _cache_key(lat: np.ndarray, lon: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(lat, dtype=np.float64).tobytes())
    h.update(np.asarray(lon, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def laplacian_eigendecomp(lat: np.ndarray, lon: np.ndarray, cache_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"lap_eig_{_cache_key(lat, lon)}.npz"
        if path.exists():
            data = np.load(path)
            return data["evals"], data["evecs"]

    lap = sphere_graph_laplacian(lat, lon)
    evals, evecs = linalg.eigh(lap, check_finite=False)
    evals = np.maximum(evals, 0.0)
    max_eval = float(np.max(evals))
    if max_eval > 0:
        evals = evals / max_eval

    if cache_dir is not None:
        np.savez_compressed(path, evals=evals, evecs=evecs)
    return evals, evecs


def fractional_laplacian_filter(
    x: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    s: float,
    strength: float,
    cache_dir: Path | None = None,
) -> np.ndarray:
    if strength <= 0:
        return x
    evals, evecs = laplacian_eigendecomp(lat, lon, cache_dir=cache_dir)
    frac = np.power(evals + 1e-12, s)
    denom = 1.0 + strength * frac
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        coeff = evecs.T @ x
        filtered = evecs @ (coeff / denom[:, None])
    return _require_finite(filtered, "fractional Laplacian filtering")


def grid_diffusion(x: np.ndarray, lat: np.ndarray, lon: np.ndarray, strength: float, steps: int = 2) -> np.ndarray:
    if strength <= 0 or steps <= 0:
        return x
    n_lat, n_lon = len(lat), len(lon)
    arr = x.reshape(n_lat, n_lon, -1).copy()
    alpha = min(strength, 0.24)
    for _ in range(steps):
        north = np.roll(arr, 1, axis=0)
        south = np.roll(arr, -1, axis=0)
        east = np.roll(arr, 1, axis=1)
        west = np.roll(arr, -1, axis=1)
        north[0, :, :] = arr[0, :, :]
        south[-1, :, :] = arr[-1, :, :]
        arr = (1 - 4 * alpha) * arr + alpha * (north + south + east + west)
    return arr.reshape(n_lat * n_lon, -1)


def temporal_diffusion(x: np.ndarray, strength: float, steps: int = 1) -> np.ndarray:
    if strength <= 0 or steps <= 0:
        return x
    arr = x.copy()
    alpha = min(strength, 0.24)
    for _ in range(steps):
        prev_t = np.roll(arr, 1, axis=1)
        next_t = np.roll(arr, -1, axis=1)
        prev_t[:, 0] = arr[:, 0]
        next_t[:, -1] = arr[:, -1]
        arr = (1 - 2 * alpha) * arr + alpha * (prev_t + next_t)
    return arr
