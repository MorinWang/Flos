#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A38(e'): where the benchmark's extremes are (descriptive, no prediction).

For every variable: the share of cells above the global 99th percentile of |z| (the pool from which patches are
drawn and the cells excluded from scoring) that lie poleward of 60 degrees, against the area share of that band;
and the share of the record's first and second half among those cells (trend proxy).  CPU only.
Output: $FLOS_OUT/A38_pool_lat.csv and A38_pool_zonal.npz (extreme-cell counts per latitude row)
"""
import os, numpy as np
from paths import DATA, OUT
from A1p_honest_benchmark import load_anom, QTHRESH

SRC = {"ncep": (DATA + "/raw", {"air": "air.mon.mean.nc", "slp": "slp.mon.mean.nc", "rhum": "rhum.sig995.mon.mean.nc",
                                "lftx": "lftx.sfc.mon.mean.nc", "pres": "pres.sfc.mon.mean.nc",
                                "omega": "omega.sig995.mon.mean.nc", "pottmp": "pottmp.sig995.mon.mean.nc",
                                "gpcp": "gpcp.mon.mean.nc"}),
       "era5": (DATA + "/era5/clean", {k: "era5_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}),
       "20cr": (DATA + "/20cr/clean", {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]})}
out = open(os.path.join(OUT, "A38_pool_lat.csv"), "w"); zon = {}
out.write("source,product,n_extreme,share_poleward60,area_share_poleward60,cellcount_share_poleward60,share_second_half\n")
for src, (raw, prods) in SRC.items():
    for p, fn in prods.items():
        z, lat, lon, sg = load_anom(os.path.join(raw, fn))
        T = z.shape[0]
        az = np.abs(z); thr = float(np.quantile(az[az > 0], QTHRESH)); m = az > thr
        pol = (np.abs(lat) >= 60)[None, :, None]
        w = np.cos(np.deg2rad(lat)).clip(min=0)
        area = float(w[np.abs(lat) >= 60].sum() / w.sum())
        cells = float((np.abs(lat) >= 60).mean())
        sh = float((m & pol).sum() / m.sum())
        half = float(m[T // 2:].sum() / m.sum())
        out.write("%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n" % (src, p, int(m.sum()), sh, area, cells, half)); out.flush()
        zon["%s_%s_count" % (src, p)] = m.sum(axis=(0, 2)).astype(float); zon["%s_%s_lat" % (src, p)] = np.asarray(lat, float)
        print("%-5s %-6s extremes %7d  poleward60 %.3f (area %.3f, cells %.3f)  second half %.3f" % (src, p, m.sum(), sh, area, cells, half), flush=True)
np.savez(os.path.join(OUT, "A38_pool_zonal.npz"), **zon)
