from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class ClimateMatrix:
    name: str
    title: str
    matrix: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray
    scale: float
    source_path: Path


def load_dataset_config(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def download_file(url: str, out_path: Path, retries: int = 5, force: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if force and out_path.exists():
        out_path.unlink()
    if not force and out_path.exists() and out_path.stat().st_size > 1024:
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "climate-rpca/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp_path.open("wb") as f:
                total = resp.headers.get("Content-Length")
                total_i = int(total) if total and total.isdigit() else None
                seen = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    seen += len(chunk)
                    if total_i:
                        pct = 100.0 * seen / total_i
                        print(f"download {out_path.name}: {pct:5.1f}%", end="\r", flush=True)
                if total_i is not None and seen != total_i:
                    raise RuntimeError(f"incomplete download for {url}: got {seen} bytes, expected {total_i}")
            tmp_path.replace(out_path)
            print(f"download {out_path.name}: done{' ' * 20}")
            return out_path
        except Exception as exc:  # pragma: no cover - network branch
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            wait = 3 * attempt
            print(f"download failed ({attempt}/{retries}): {exc}; retrying in {wait}s")
            time.sleep(wait)
    return out_path


def _coord_name(ds: xr.Dataset, preferred: str, candidates: tuple[str, ...]) -> str:
    if preferred in ds.coords or preferred in ds.dims:
        return preferred
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"could not find coordinate among {preferred!r} or {candidates!r}")


def _select_even(values: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= 0 or target_size >= len(values):
        return np.arange(len(values))
    idx = np.linspace(0, len(values) - 1, target_size)
    return np.unique(np.round(idx).astype(int))


def _monthly_anomaly(da: xr.DataArray, time_name: str) -> xr.DataArray:
    try:
        return da.groupby(f"{time_name}.month") - da.groupby(f"{time_name}.month").mean(time_name)
    except Exception:
        return da - da.mean(time_name)


def _robust_scale(x: np.ndarray) -> float:
    finite = np.isfinite(x)
    if not finite.any():
        return 1.0
    med = np.nanmedian(x[finite])
    mad = np.nanmedian(np.abs(x[finite] - med)) / 0.6745
    std = np.nanstd(x[finite])
    scale = mad if mad > 1e-8 else std
    return float(scale if scale > 1e-8 else 1.0)


def prepare_climate_matrix(
    name: str,
    spec: dict[str, Any],
    data_root: Path,
    lat_size: int,
    lon_size: int,
    time_steps: int,
) -> ClimateMatrix:
    file_name = os.path.basename(spec["url"])
    source_path = download_file(spec["url"], data_root / "raw" / name / file_name)

    try:
        ds = xr.open_dataset(source_path)
    except OSError:
        if source_path.exists():
            source_path.unlink()
        source_path = download_file(spec["url"], data_root / "raw" / name / file_name, force=True)
        ds = xr.open_dataset(source_path)
    var_name = spec["variable"]
    if var_name not in ds:
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError(f"{source_path} contains no data variables")
        var_name = data_vars[0]

    lat_name = _coord_name(ds, spec.get("lat_name", "lat"), ("latitude", "y"))
    lon_name = _coord_name(ds, spec.get("lon_name", "lon"), ("longitude", "x"))
    time_name = _coord_name(ds, spec.get("time_name", "time"), ("date",))

    da = ds[var_name]
    extra_dims = [d for d in da.dims if d not in {time_name, lat_name, lon_name}]
    for dim in extra_dims:
        da = da.isel({dim: 0})

    if time_steps > 0 and da.sizes[time_name] > time_steps:
        da = da.isel({time_name: slice(da.sizes[time_name] - time_steps, None)})

    lat_values = np.asarray(da[lat_name].values, dtype=float)
    lon_values = np.asarray(da[lon_name].values, dtype=float)
    lat_idx = _select_even(lat_values, lat_size)
    lon_idx = _select_even(lon_values, lon_size)
    da = da.isel({lat_name: lat_idx, lon_name: lon_idx})

    lat = np.asarray(da[lat_name].values, dtype=float)
    lon = np.asarray(da[lon_name].values, dtype=float)
    if lat[0] > lat[-1]:
        da = da.sortby(lat_name)
        lat = np.asarray(da[lat_name].values, dtype=float)

    da = _monthly_anomaly(da, time_name)
    da = da.transpose(time_name, lat_name, lon_name)
    values = np.asarray(da.load().values, dtype=np.float64)

    scale = _robust_scale(values)
    values = values / scale
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove any remaining spatial mean so low-rank recovery is not dominated by
    # a constant offset introduced by masked land/ocean cells.
    values = values - values.mean(axis=(1, 2), keepdims=True)
    matrix = values.reshape(values.shape[0], -1).T

    time_values = np.asarray(da[time_name].values)
    if time_values.size and np.issubdtype(time_values.dtype, np.datetime64):
        time_values = time_values.astype("datetime64[M]").astype(str)
    else:
        time_values = np.asarray([str(i) for i in range(matrix.shape[1])])

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} preprocessing produced non-finite values")

    expected_n = len(lat) * len(lon)
    if matrix.shape[0] != expected_n:
        raise ValueError(f"{name} matrix shape mismatch: {matrix.shape[0]} vs {expected_n}")

    return ClimateMatrix(
        name=name,
        title=spec.get("title", name),
        matrix=matrix,
        lat=lat,
        lon=lon,
        time=time_values,
        scale=scale,
        source_path=source_path,
    )


def latitude_bands(lat: np.ndarray, lon: np.ndarray, n_bands: int = 6) -> dict[str, np.ndarray]:
    lat_grid = np.repeat(lat, len(lon))
    edges = np.linspace(float(np.min(lat)), float(np.max(lat)), n_bands + 1)
    bands: dict[str, np.ndarray] = {}
    for i in range(n_bands):
        left, right = edges[i], edges[i + 1]
        if i == n_bands - 1:
            mask = (lat_grid >= left) & (lat_grid <= right)
        else:
            mask = (lat_grid >= left) & (lat_grid < right)
        bands[f"{left:.1f}..{right:.1f}"] = mask
    return bands


def spatial_index_to_coord(idx: int, lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    n_lon = len(lon)
    i_lat = int(math.floor(idx / n_lon))
    i_lon = int(idx % n_lon)
    return float(lat[i_lat]), float(lon[i_lon])
