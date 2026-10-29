#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A17: held-out validation of "adaptivity helps in the sparse regime"; the evaluation criteria were fixed in advance.
Two arms, A adaptive (default FLoS) and B fixed threshold, twelve λ values, otherwise identical to A15/A16.
Runs only on the twelve held-out products that were not examined in this setting during development.
Runs are sharded; each shard writes its own csv to avoid write conflicts."""
import sys, os, time, csv, argparse
import numpy as np, torch
from paths import DATA, OUT as OUTDIR, CACHE, device; DEVICE = device()
SYNC = torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
FREE = torch.cuda.empty_cache if DEVICE == "cuda" else (lambda: None)
import A5_leaderboard as L
import flos_torch as FT
SRC={"ncep": (DATA+"/raw",
              {"lftx":"lftx.sfc.mon.mean.nc","pres":"pres.sfc.mon.mean.nc",
               "omega":"omega.sig995.mon.mean.nc","pottmp":"pottmp.sig995.mon.mean.nc",
               "gpcp":"gpcp.mon.mean.nc"}),
     "era5": (DATA+"/era5/clean",
              {"air":"era5_air.mon.nc","slp":"era5_slp.mon.nc","pres":"era5_pres.mon.nc",
               "pottmp":"era5_pottmp.mon.nc","rhum":"era5_rhum.mon.nc",
               "omega":"era5_omega.mon.nc","cape":"era5_cape.mon.nc"})}
ap=argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["ncep","era5"])
ap.add_argument("--products", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--lams", default="0.045,0.09,0.18,0.27,0.36,0.5,0.72,1.0,1.44,2.0,2.88,4.0")
ap.add_argument("--shard", required=True)
a=ap.parse_args()
L.CACHE=CACHE; L.RAW,L.PRODUCTS=SRC[a.source]
prods=a.products.split(","); seeds=[int(x) for x in a.seeds.split(",")]
lams=[float(x) for x in a.lams.split(",")]
OD=OUTDIR; os.makedirs(OD, exist_ok=True)
OUT="%s/A17_holdout_%s.csv"%(OD,a.shard)
done=set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)): done.add((r["source"],r["product"],r["arm"],float(r["lam"]),int(r["seed"])))
else:
    open(OUT,"w").write("source,product,arm,lam,seed,seconds,rank,iters,converged,auprc,prec_at_k,pos_rate,n_pos,nnz_frac\n")
print("shard %s  source %s  products %s  seeds %s  λ %d values"%(a.shard,a.source,prods,seeds,len(lams)), flush=True)
for p in prods:
    path=os.path.join(L.RAW,L.PRODUCTS[p]); filt=None
    for sd in seeds:
        todo=[(arm,lm) for lm in lams for arm in ["A-adapt","B-fixed"]
              if (a.source,p,arm,lm,sd) not in done]
        if not todo: continue
        b=L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T,nlat,nlon=b["observed"].shape
        X=b["observed"].reshape(T,-1).T.copy(); r=L.mp_rank(X)
        if filt is None:
            fn=L.SphereFilter(b["lat"],b["lon"],L.FLOS["alpha"],L.FLOS["s"])
            filt=FT.SphereFilterT(fn.U,fn.inv,fn.node_of,device=DEVICE,dtype=torch.float32)
        Xt=torch.as_tensor(X,dtype=torch.float32,device=DEVICE)
        to3=lambda M: M.T.reshape(T,nlat,nlon)
        for arm,lm in todo:
            SYNC(); t0=time.perf_counter()
            Lm,S,info=FT.run_flos_ablate(Xt,filt,lam=lm,p=L.FLOS["p"],
                max_iter=L.FLOS["max_iter"],tol=L.FLOS["tol"],
                paper_iters=L.FLOS["paper_iters"],rank=r,
                ablation=("fixed-thresh" if arm=="B-fixed" else None))
            SYNC(); sec=time.perf_counter()-t0
            nnz=float((S!=0).float().mean())
            row=L.score(b,scores=np.abs(to3(S.double().cpu().numpy())))
            open(OUT,"a").write("%s,%s,%s,%g,%d,%.1f,%s,%s,%s,%.5f,%.5f,%.2g,%d,%.5f\n"%(
                a.source,p,arm,lm,sd,sec,info["rank"],info["iters"],info["converged"],
                row["auprc"],row["prec_at_k"],row["pos_rate"],row["n_pos"],nnz))
            print("  %-6s %-6s seed %d λ=%-6g %-8s AUPRC %.5f kept %.5f %.0f s"%(
                a.source,p,sd,lm,arm,row["auprc"],nnz,sec), flush=True)
            del Lm,S
        del Xt; FREE()
print("shard %s done"%a.shard, flush=True)
