"""Loading a monthly field as anomalies, in the units used throughout the paper."""
import numpy as np


def load_anomalies(path, variable=None):
    """Read a monthly NetCDF field on a latitude-longitude mesh and return anomalies.

    The calendar-month mean is removed at every cell and the result is divided by one global robust scale
    (median absolute anomaly / 0.6745). Returns (Z, lat, lon, time, scale) with Z of shape (time, lat, lon).
    Use `Z.reshape(len(time), -1).T` for the cells x time matrix that FLoS takes.
    """
    import xarray as xr
    with xr.open_dataset(path) as d:
        da = d[variable or list(d.data_vars)[0]]
        lat = da.lat.values.astype(float); lon = da.lon.values.astype(float)
        x = da.values.astype(np.float64); months = da.time.dt.month.values; time = da.time.values
    a = np.empty_like(x)
    for m in np.unique(months):
        sel = months == m
        a[sel] = 0.0 if sel.sum() < 3 else x[sel] - np.nanmean(x[sel], 0)
    a = np.nan_to_num(a)
    aa = np.abs(a)
    scale = float(np.median(aa[aa > 0]) / 0.6745)
    return a / scale, lat, lon, time, scale
