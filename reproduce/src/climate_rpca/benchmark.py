from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .linalg import low_rank_approx


@dataclass(frozen=True)
class BenchmarkInstance:
    observed: np.ndarray
    low_rank: np.ndarray
    sparse: np.ndarray
    support: np.ndarray
    dense_noise: np.ndarray
    event_table: list[dict[str, float | int]]


def dataset_seed(seed: int, dataset_name: str) -> int:
    """Derive a stable benchmark seed for a dataset.

    Using the same integer seed for every product makes identical latitude-
    longitude grids receive identical injected event supports. That is useful
    for method comparisons within one product, but it creates artificial
    cross-product repetition in aggregate tables. This helper keeps the seed
    deterministic while decorrelating event locations across products.
    """
    digest = hashlib.sha1(dataset_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((int(seed) + offset) % (2**32 - 1))


def _spherical_distance_grid(lat: np.ndarray, lon: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    phi1 = np.deg2rad(lat_grid)
    phi2 = np.deg2rad(center_lat)
    dphi = np.deg2rad(lat_grid - center_lat)
    dlambda = np.deg2rad(lon_grid - center_lon)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))


def _elliptic_distance_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    angle: float,
    aspect: float,
) -> np.ndarray:
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    dlat = lat_grid - center_lat
    dlon = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
    dlon = dlon * np.cos(np.deg2rad(center_lat))
    ca, sa = np.cos(angle), np.sin(angle)
    x = ca * dlon + sa * dlat
    y = -sa * dlon + ca * dlat
    return np.sqrt((x / max(aspect, 1e-6)) ** 2 + y**2)


def _event_patch(
    rng: np.random.Generator,
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius: float,
    event_shape: str,
) -> tuple[np.ndarray, np.ndarray]:
    event_shape = event_shape.lower()
    if event_shape == "spherical":
        dist_rad = _spherical_distance_grid(lat, lon, center_lat, center_lon)
        radius_rad = np.deg2rad(radius)
        return dist_rad <= radius_rad, dist_rad / max(radius_rad, 1e-8)

    if event_shape == "elongated":
        angle = float(rng.uniform(0.0, np.pi))
        aspect = float(rng.uniform(1.8, 3.2))
        dist_deg = _elliptic_distance_grid(lat, lon, center_lat, center_lon, angle, aspect)
        return dist_deg <= radius, dist_deg / max(radius, 1e-8)

    if event_shape == "multicore":
        patch = np.zeros((len(lat), len(lon)), dtype=bool)
        taper_dist = np.full((len(lat), len(lon)), np.inf)
        for _ in range(int(rng.integers(2, 4))):
            lat_j = float(center_lat + rng.normal(0.0, radius * 0.35))
            lon_j = float(center_lon + rng.normal(0.0, radius * 0.35))
            radius_j = float(radius * rng.uniform(0.45, 0.8))
            dist_rad = _spherical_distance_grid(lat, lon, lat_j, lon_j)
            radius_rad = np.deg2rad(radius_j)
            patch_j = dist_rad <= radius_rad
            patch |= patch_j
            taper_dist = np.minimum(taper_dist, dist_rad / max(radius_rad, 1e-8))
        return patch, taper_dist

    if event_shape in {"irregular", "nonconvex"}:
        dist_rad = _spherical_distance_grid(lat, lon, center_lat, center_lon)
        radius_rad = np.deg2rad(radius)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        wobble = 1.0 + 0.28 * np.sin(np.deg2rad(lat_grid * 2.0) + phase)
        wobble += 0.18 * np.cos(np.deg2rad(lon_grid * 3.0) - phase)
        patch = dist_rad <= radius_rad * np.clip(wobble, 0.55, 1.45)
        dropout = rng.random(patch.shape) < 0.12
        patch = patch & ~dropout
        return patch, dist_rad / max(radius_rad, 1e-8)

    raise ValueError(f"unknown event_shape={event_shape!r}")


def _sample_magnitude(
    rng: np.random.Generator,
    amplitude_dist: str,
    tail_alpha: float,
    base_scale: float,
) -> float:
    amplitude_dist = amplitude_dist.lower()
    if amplitude_dist == "pareto":
        raw = rng.pareto(tail_alpha) + 1.0
    elif amplitude_dist == "lognormal":
        raw = rng.lognormal(mean=0.25, sigma=0.85)
    elif amplitude_dist in {"student", "student_t", "student-t"}:
        raw = abs(rng.standard_t(df=3.0)) + 0.5
    elif amplitude_dist == "mixture":
        if rng.random() < 0.8:
            raw = rng.lognormal(mean=0.0, sigma=0.45)
        else:
            raw = rng.pareto(max(tail_alpha, 1.1)) + 1.0
    else:
        raise ValueError(f"unknown amplitude_dist={amplitude_dist!r}")
    return float(raw * base_scale * rng.uniform(2.5, 5.5))


def make_heavy_tail_benchmark(
    x: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    seed: int,
    rank_true: int = 6,
    event_rate: float = 0.06,
    cluster_len: int = 4,
    spatial_radius_deg: float = 18.0,
    tail_alpha: float = 1.35,
    dense_noise_scale: float = 0.03,
    event_shape: str = "spherical",
    amplitude_dist: str = "pareto",
) -> BenchmarkInstance:
    rng = np.random.default_rng(seed)
    l0 = low_rank_approx(x, rank_true)
    residual_scale = float(np.std(x - l0))
    base_scale = residual_scale if residual_scale > 1e-8 else float(np.std(x))
    base_scale = base_scale if base_scale > 1e-8 else 1.0

    n_space, n_time = x.shape
    s0 = np.zeros_like(x)
    support = np.zeros_like(x, dtype=bool)
    n_events = max(2, int(np.ceil(n_time * event_rate / max(cluster_len, 1))))
    event_table: list[dict[str, float | int]] = []

    for event_id in range(n_events):
        t0 = int(rng.integers(0, max(n_time - 1, 1)))
        duration = int(max(1, rng.poisson(cluster_len - 1) + 1))
        times = np.arange(t0, min(n_time, t0 + duration))

        center_lat = float(rng.choice(lat))
        center_lon = float(rng.choice(lon))
        radius = float(spatial_radius_deg * rng.uniform(0.6, 1.4))
        patch, taper_distance = _event_patch(rng, lat, lon, center_lat, center_lon, radius, event_shape)
        if patch.sum() < 3:
            flat_idx = int(rng.integers(0, n_space))
            patch = np.zeros((len(lat), len(lon)), dtype=bool)
            patch.reshape(-1)[flat_idx] = True
            taper_distance = np.ones_like(patch, dtype=float)

        taper = np.exp(-0.5 * taper_distance**2).reshape(-1)
        patch_flat = patch.reshape(-1)
        sign = float(rng.choice([-1.0, 1.0]))
        magnitude = _sample_magnitude(rng, amplitude_dist, tail_alpha, base_scale)

        for offset, t in enumerate(times):
            temporal = np.exp(-0.5 * (offset / max(duration / 2.0, 1.0)) ** 2)
            amp = sign * magnitude * temporal
            s0[patch_flat, t] += amp * taper[patch_flat]
            support[patch_flat, t] = True

        event_table.append(
            {
                "event_id": event_id,
                "t_start": int(t0),
                "duration": int(len(times)),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "radius_deg": radius,
                "signed_magnitude": sign * magnitude,
                "support_size": int(patch_flat.sum() * len(times)),
                "event_shape": event_shape,
                "amplitude_dist": amplitude_dist,
            }
        )

    dense_noise = rng.normal(0.0, dense_noise_scale * base_scale, size=x.shape)
    observed = l0 + s0 + dense_noise
    return BenchmarkInstance(
        observed=observed,
        low_rank=l0,
        sparse=s0,
        support=support,
        dense_noise=dense_noise,
        event_table=event_table,
    )
