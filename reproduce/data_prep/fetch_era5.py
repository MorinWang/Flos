#!/usr/bin/env python3
# Download ERA5 monthly means on the NCEP R1 2.5-degree 73x144 grid for 1948-01..2026-02.
# Six variables downloaded directly, plus CAPE as a substitute for lftx (ERA5 has no lifted-index variable).
import cdsapi, os, sys, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import DATA
OUT = os.path.join(DATA, "era5")
YEARS = [str(y) for y in range(1948, 2027)]
MONTHS = ["%02d" % m for m in range(1, 13)]
GRID = [2.5, 2.5]

SINGLE = {
    "t2m":  "2m_temperature",                            # -> air
    "msl":  "mean_sea_level_pressure",                   # -> slp
    "sp":   "surface_pressure",                          # -> pres
    "d2m":  "2m_dewpoint_temperature",                   # -> used for rhum
    "cape": "convective_available_potential_energy",     # -> substitute for lftx
}

c = cdsapi.Client()

def grab(tag, req, ds, fname):
    path = os.path.join(OUT, fname)
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        print("[skip] %s already exists %.1f MB" % (fname, os.path.getsize(path)/1e6), flush=True)
        return True
    print("[start] %s <- %s" % (fname, tag), flush=True)
    try:
        c.retrieve(ds, req, path)
        print("[done] %s  %.1f MB" % (fname, os.path.getsize(path)/1e6), flush=True)
        return True
    except Exception as e:
        print("[failed] %s : %s" % (fname, str(e)[:300]), flush=True)
        traceback.print_exc()
        return False

ok, bad = [], []
for short, longname in SINGLE.items():
    r = {"product_type": ["monthly_averaged_reanalysis"], "variable": [longname],
         "year": YEARS, "month": MONTHS, "time": ["00:00"],
         "grid": GRID, "data_format": "netcdf"}
    (ok if grab(longname, r, "reanalysis-era5-single-levels-monthly-means",
                "era5_%s.mon.nc" % short) else bad).append(short)

# omega: vertical velocity at the 1000 hPa pressure level
r = {"product_type": ["monthly_averaged_reanalysis"], "variable": ["vertical_velocity"],
     "pressure_level": ["1000"], "year": YEARS, "month": MONTHS, "time": ["00:00"],
     "grid": GRID, "data_format": "netcdf"}
(ok if grab("vertical_velocity@1000hPa", r, "reanalysis-era5-pressure-levels-monthly-means",
            "era5_w1000.mon.nc") else bad).append("w1000")

print("\n==== succeeded %d: %s" % (len(ok), ok), flush=True)
print("==== failed %d: %s" % (len(bad), bad), flush=True)
