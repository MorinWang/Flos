#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ERA5 monthly means -> product files in the same format as NCEP R1.
The generator load_anom only reads the first 3-D variable in a file, with coordinates named time / lat / lon.
So this script renames coordinates, unifies units, keeps one variable per file, and sanity-checks the units of each.
Output goes to $FLOS_DATA/era5/clean/, with file names matching the NCEP files one to one.
"""
import os, sys
import numpy as np, xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import DATA
SRC = os.path.join(DATA, "era5")
DST = os.path.join(SRC, "clean")
os.makedirs(DST, exist_ok=True)

def tidy(da, name, units):
    """Unify coordinate names, drop auxiliary coordinates such as number/expver, keep a single 3-D variable."""
    ren = {}
    for a, b in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon")):
        if a in da.dims or a in da.coords: ren[a] = b
    da = da.rename(ren)
    # Pressure-level data has an extra pressure_level dimension (only one level was fetched); squeeze it out
    for d in list(da.dims):
        if d not in ("time", "lat", "lon"):
            da = da.isel({d: 0}, drop=True)
    for c in list(da.coords):
        if c not in ("time", "lat", "lon"): da = da.drop_vars(c)
    da = da.transpose("time", "lat", "lon")
    da.name = name; da.attrs = {"units": units, "long_name": name, "source": "ERA5 monthly means"}
    return da

def check(name, da, lo, hi, units):
    v = da.values
    n_nan = int(np.isnan(v).sum())
    vmin, vmax, vmed = float(np.nanmin(v)), float(np.nanmax(v)), float(np.nanmedian(v))
    ok = (vmin >= lo) and (vmax <= hi) and (da.shape[1], da.shape[2]) == (73, 144)
    print("  %-8s %-10s shape %s  range %.2f .. %.2f  median %.2f  missing %d  %s"
          % (name, units, da.shape, vmin, vmax, vmed, n_nan, "OK" if ok else "!! outside expected range [%g,%g]" % (lo, hi)))
    return ok

# NCEP R1 time axis: both datasets must align month by month, otherwise they are not the same task
NCEP_REF = os.path.join(DATA, "raw", "air.mon.mean.nc")
_ref = xr.open_dataset(NCEP_REF)
T0, T1 = _ref.time.values[0], _ref.time.values[-1]
NT = len(_ref.time); _ref.close()

def load(fn):
    p = os.path.join(SRC, fn)
    if not os.path.exists(p): return None
    d = xr.open_dataset(p)
    k = [n for n in d.data_vars if d[n].ndim >= 3][0]
    da = d[k]
    tname = "valid_time" if "valid_time" in da.dims else "time"
    da = da.sel({tname: slice(T0, T1)})          # crop to the same period as NCEP
    if len(da[tname]) != NT:
        print("  !! %s has %d months after cropping, NCEP has %d months" % (fn, len(da[tname]), NT))
    return da

def save(da, fn):
    p = os.path.join(DST, fn)
    da.to_dataset(name=da.name).to_netcdf(p)
    print("  wrote %s  %.1f MB" % (fn, os.path.getsize(p)/1e6))

print("=== ERA5 conversion and sanity checks ===")
allok = True

t2m = load("era5_t2m.mon.nc"); sp = load("era5_sp.mon.nc")
msl = load("era5_msl.mon.nc"); d2m = load("era5_d2m.mon.nc")
cape = load("era5_cape.mon.nc"); w = load("era5_w1000.mon.nc")

# 1) Air temperature: K -> degC, matching NCEP air
if t2m is not None:
    air = tidy(t2m, "air", "degC") - 273.15
    air = tidy(air, "air", "degC")
    allok &= check("air", air, -95, 60, "degC"); save(air, "era5_air.mon.nc")

# 2) Sea-level pressure: Pa -> millibars
if msl is not None:
    slp = tidy(msl, "slp", "millibars") / 100.0
    slp = tidy(slp, "slp", "millibars")
    allok &= check("slp", slp, 900, 1100, "millibars"); save(slp, "era5_slp.mon.nc")

# 3) Surface pressure: Pa -> millibars
if sp is not None:
    pres = tidy(sp, "pres", "millibars") / 100.0
    pres = tidy(pres, "pres", "millibars")
    allok &= check("pres", pres, 400, 1100, "millibars"); save(pres, "era5_pres.mon.nc")

# 4) Potential temperature: theta = T * (1000/p_hPa)^0.2857, with p the surface pressure
if t2m is not None and sp is not None:
    T = tidy(t2m, "t", "K"); P = tidy(sp, "p", "Pa") / 100.0
    theta = T * (1000.0 / P) ** 0.2857
    theta = tidy(theta, "pottmp", "degK")
    allok &= check("pottmp", theta, 200, 400, "degK"); save(theta, "era5_pottmp.mon.nc")

# 5) Relative humidity (Magnus formula): e(T)=6.112*exp(17.67*Tc/(Tc+243.5)), RH=100*e(Td)/e(T)
if t2m is not None and d2m is not None:
    Tc = tidy(t2m, "t", "K") - 273.15
    Dc = tidy(d2m, "d", "K") - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    ed = 6.112 * np.exp(17.67 * Dc / (Dc + 243.5))
    rh = (100.0 * ed / es).clip(0, 100)
    rh = tidy(rh, "rhum", "%")
    allok &= check("rhum", rh, 0, 100, "%"); save(rh, "era5_rhum.mon.nc")

# 6) Vertical velocity: Pa/s, same unit as NCEP omega
if w is not None:
    om = tidy(w, "omega", "Pascal/s")
    allok &= check("omega", om, -5, 5, "Pascal/s"); save(om, "era5_omega.mon.nc")

# 7) CAPE: substitutes for the lifted index; note that it is NOT the same variable
if cape is not None:
    cp = tidy(cape, "cape", "J/kg")
    allok &= check("cape", cp, 0, 8000, "J/kg"); save(cp, "era5_cape.mon.nc")

print("\n==== Overall: %s" % ("all checks passed" if allok else "some items outside expected range, see the !! lines above"))
