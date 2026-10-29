#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A37(d): GLOSS tuned on an absolute scale.

The paper's rule lambda = gamma = 1/||Y||_0 gives ~1e-7 on our data and leaves 99.8% of the entries in S at every
multiplier, i.e. GLOSS was effectively untuned.  Here lambda = gamma takes absolute values; everything else is the
validated port (A5_gloss.run_gloss, the authors' remaining parameter rules, month x year x lat x lon tensor, whole
years only, score |S|).  CPU only.
"""
import os, sys, time, csv, argparse
import numpy as np
from paths import DATA, OUT as OUTDIR, CACHE
import A5_leaderboard as L
from A5_gloss import run_gloss

SRC = {"ncep": (DATA + "/raw", {"air": "air.mon.mean.nc", "slp": "slp.mon.mean.nc", "rhum": "rhum.sig995.mon.mean.nc",
                                "lftx": "lftx.sfc.mon.mean.nc", "pres": "pres.sfc.mon.mean.nc",
                                "omega": "omega.sig995.mon.mean.nc", "pottmp": "pottmp.sig995.mon.mean.nc",
                                "gpcp": "gpcp.mon.mean.nc"}),
       "era5": (DATA + "/era5/clean", {k: "era5_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}),
       "20cr": (DATA + "/20cr/clean", {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]})}
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=list(SRC))
ap.add_argument("--products", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1,1,10")
ap.add_argument("--shard", required=True)
a = ap.parse_args()
L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
OUT = "%s/A37_gloss_%s.csv" % (OUTDIR, a.shard)
if not os.path.exists(OUT):
    open(OUT, "w").write("source,product,seed,lam,auprc,prec_at_k,nnz_frac,iters,seconds\n")
done = {(r["product"], int(r["seed"]), float(r["lam"])) for r in csv.DictReader(open(OUT))}
for p in a.products.split(","):
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in map(int, a.seeds.split(",")):
        todo = [float(x) for x in a.lams.split(",") if (p, sd, float(x)) not in done]
        if not todo: continue
        b = L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        O3 = b["observed"]; T, h, w = O3.shape; ny = T // 12; Tn = ny * 12
        Y = O3[:Tn].reshape(ny, 12, h, w).transpose(1, 0, 2, 3).copy()
        for lam in todo:
            t0 = time.time()
            Lg, Sg, info = run_gloss(Y, lam=lam, gam=lam)
            S3 = np.zeros_like(O3); S3[:Tn] = Sg.transpose(1, 0, 2, 3).reshape(Tn, h, w)
            row = L.score(b, scores=np.abs(S3))
            open(OUT, "a").write("%s,%s,%d,%g,%.5f,%.5f,%.5f,%s,%.0f\n" % (a.source, p, sd, lam, row["auprc"], row["prec_at_k"],
                                 float((S3[:Tn] != 0).mean()), info.get("iters", ""), time.time() - t0))
            print("  %-5s %-6s seed%d lam=%-6g AUPRC %.5f nnz %.4f %.0fs" % (a.source, p, sd, lam, row["auprc"],
                  float((S3[:Tn] != 0).mean()), time.time() - t0), flush=True)
print("A37 gloss shard %s done %s" % (a.shard, time.strftime("%F %T")), flush=True)
