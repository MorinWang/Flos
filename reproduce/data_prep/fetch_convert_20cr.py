#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20CRv3 monthly means -> product files in the same format as NCEP R1 (third reanalysis dataset).
Source: NOAA PSL THREDDS OPeNDAP, subset year by year, 1948-01..2015-12 only (20CRv3 ends at 2015-12).
Grid: native 1-degree 181x360, bilinearly interpolated to the NCEP R1 73x144 grid (lat 90->-90, lon 0->357.5, longitude wrapped).
The seven products match the ERA5 ones one to one: air, slp, pres, pottmp, rhum, omega, cape.
  air   = air.2m (K->degC)              slp  = prmsl (Pa->millibars)  pres = pres.sfc (Pa->millibars)
  pottmp= air.2m*(1000/pres_hPa)^0.2857 (same formula as for ERA5)
  rhum  = rhum.2m (%)                   omega= omega 1000 hPa (Pa/s)  cape = cape (J/kg, substitute for lftx, as for ERA5)
"""
import os, sys, time
import numpy as np, xarray as xr
BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/20thC_ReanV3/Monthlies/"
SRC = {"air2m": ("2mSI-MO/air.2m.mon.mean.nc", "air", None),
       "prmsl": ("miscSI-MO/prmsl.mon.mean.nc", "prmsl", None),
       "pres":  ("sfcSI-MO/pres.sfc.mon.mean.nc", "pres", None),
       "rhum2m":("2mSI-MO/rhum.2m.mon.mean.nc", "rhum", None),
       "omega": ("prsSI-MO/omega.mon.mean.nc", "omega", 1000.0),
       "cape":  ("sfcSI-MO/cape.mon.mean.nc", "cape", None)}
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import DATA
OUT = os.path.join(DATA, "20cr"); DST = OUT + "/clean"
ref = xr.open_dataset(os.path.join(DATA, "raw", "air.mon.mean.nc"))
LAT, LON = ref.lat.values, ref.lon.values; ref.close()

def fetch(key):
    path = "%s/raw_%s_1deg.nc" % (OUT, key)
    if os.path.exists(path):
        print("[skip] %s already exists" % key, flush=True); return xr.open_dataarray(path)
    rel, var, lev = SRC[key]
    d = xr.open_dataset(BASE + rel)
    da = d[var]
    if lev is not None: da = da.sel(level=lev)
    parts = []
    for y in range(1948, 2016):
        for k in range(5):
            try:
                parts.append(da.sel(time=slice("%d-01-01" % y, "%d-12-31" % y)).load()); break
            except Exception as e:
                print("  retry %s %d: %s" % (key, y, str(e)[:80]), flush=True); time.sleep(10)
        else:
            raise RuntimeError("failed to fetch %s %d" % (key, y))
    da = xr.concat(parts, "time")
    for c in list(da.coords):
        if c not in ("time", "lat", "lon"): da = da.drop_vars(c)
    da.attrs = {}; da.name = key
    da.to_netcdf(path); print("[fetched] %s %s" % (key, da.shape), flush=True)
    return da

def regrid(da):
    # Wrap longitude: append a column at 360 before interpolating
    da = da.sortby("lat")
    wrap = da.isel(lon=0).assign_coords(lon=360.0)
    da = xr.concat([da, wrap], "lon")
    return da.interp(lat=LAT, lon=LON, method="linear").transpose("time", "lat", "lon")

def save(da, name, units, lo, hi):
    da = da.astype("float32"); da.name = name
    da.attrs = {"units": units, "long_name": name, "source": "20CRv3 monthly means, bilinear to 2.5deg"}
    v = da.values; ok = (np.nanmin(v) >= lo) and (np.nanmax(v) <= hi) and da.shape == (816, 73, 144) and not np.isnan(v).any()
    print("  %-7s %-9s shape %s range %.2f..%.2f median %.2f missing %d %s" % (name, units, da.shape, np.nanmin(v), np.nanmax(v),
          np.nanmedian(v), int(np.isnan(v).sum()), "OK" if ok else "!! check failed"), flush=True)
    da.to_dataset(name=name).to_netcdf("%s/20cr_%s.mon.nc" % (DST, name))
    return ok

raw = {k: fetch(k) for k in SRC}
print("=== conversion and sanity checks ===", flush=True)
g = {k: regrid(v) for k, v in raw.items()}
ok = True
ok &= save(g["air2m"] - 273.15, "air", "degC", -95, 60)
ok &= save(g["prmsl"] / 100.0, "slp", "millibars", 900, 1100)
ok &= save(g["pres"] / 100.0, "pres", "millibars", 400, 1100)
ok &= save(g["air2m"] * (1000.0 / (g["pres"] / 100.0)) ** 0.2857, "pottmp", "degK", 200, 400)
ok &= save(g["rhum2m"].clip(0, 100), "rhum", "%", 0, 100)
ok &= save(g["omega"], "omega", "Pascal/s", -5, 5)
ok &= save(g["cape"], "cape", "J/kg", 0, 8000)
print("==== Overall: %s" % ("all checks passed" if ok else "some checks failed, see the !! lines"), flush=True)
