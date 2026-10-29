#!/bin/bash
# NCEP/NCAR Reanalysis 1 (seven monthly surface variables) and GPCP v2.3 monthly precipitation from NOAA PSL.
# Originals go to raw/_orig/; raw/ keeps only the main variable of each file (some carry time_bnds, and the
# benchmark generator reads the first data variable), cut to the months used in the paper:
# NCEP 1948-01..2026-02 (938 months), GPCP 1979-01..2026-06 (570 months).
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
RAW=${FLOS_DATA:-$HERE/data}/raw; ORIG=$RAW/_orig; mkdir -p "$ORIG"
B=https://psl.noaa.gov/thredds/fileServer/Datasets
declare -A SRC=(
  [air.mon.mean.nc]=ncep.reanalysis.derived/surface/air.mon.mean.nc
  [slp.mon.mean.nc]=ncep.reanalysis.derived/surface/slp.mon.mean.nc
  [lftx.sfc.mon.mean.nc]=ncep.reanalysis.derived/surface/lftx.sfc.mon.mean.nc
  [pres.sfc.mon.mean.nc]=ncep.reanalysis.derived/surface/pres.sfc.mon.mean.nc
  [omega.sig995.mon.mean.nc]=ncep.reanalysis.derived/surface/omega.sig995.mon.mean.nc
  [pottmp.sig995.mon.mean.nc]=ncep.reanalysis.derived/surface/pottmp.sig995.mon.mean.nc
  [rhum.sig995.mon.mean.nc]=ncep.reanalysis.derived/surface/rhum.sig995.mon.mean.nc
  [gpcp.mon.mean.nc]=gpcp/precip.mon.mean.nc
)
for f in "${!SRC[@]}"; do
  if [ ! -s "$ORIG/$f" ]; then
    # THREDDS HTTP/2 streams can drop mid-transfer; force HTTP/1.1 and retry by hand
    for k in 1 2 3 4 5 6 7 8 9 10; do
      curl -fL --http1.1 -C - -o "$ORIG/$f.part" "$B/${SRC[$f]}" && break
      echo "retry $k: $f"; sleep 5
    done
    mv "$ORIG/$f.part" "$ORIG/$f"
  fi
done
RAW="$RAW" $PY - <<'PYEOF'
import os, xarray as xr
RAW = os.environ["RAW"]; ORIG = os.path.join(RAW, "_orig")
END = {"gpcp.mon.mean.nc": "2026-06"}
for f in sorted(os.listdir(ORIG)):
    if not f.endswith(".nc"): continue
    d = xr.open_dataset(os.path.join(ORIG, f), decode_times=False)   # time_bnds units do not decode
    main = [k for k, v in d.data_vars.items() if {"time", "lat", "lon"} <= set(v.dims)]
    assert len(main) == 1, (f, list(d.data_vars))
    out = xr.decode_cf(d[[main[0]]].load()); d.close()
    out = out.sel(time=slice(None, END.get(f, "2026-02")))
    out.to_netcdf(os.path.join(RAW, f))
    print(f, main[0], dict(out.sizes), str(out.time.values[0])[:7], "->", str(out.time.values[-1])[:7])
PYEOF
