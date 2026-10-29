#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A15: sweep λ to test whether "fixed threshold beats adaptive threshold" is an artefact of the threshold level.
Two arms, A adaptive (default FLoS) and B fixed threshold, otherwise identical.
The benchmark, scoring, α, s and p are kept fixed."""
import sys, os, time, csv, argparse
import numpy as np, torch
from paths import DATA, OUT as OUTDIR, CACHE, device; DEVICE = device()
SYNC = torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
FREE = torch.cuda.empty_cache if DEVICE == "cuda" else (lambda: None)
import A5_leaderboard as L
import flos_torch as FT
L.CACHE=CACHE; L.RAW=DATA+"/raw"
L.PRODUCTS={"air":"air.mon.mean.nc","slp":"slp.mon.mean.nc","rhum":"rhum.sig995.mon.mean.nc",
            "lftx":"lftx.sfc.mon.mean.nc","pres":"pres.sfc.mon.mean.nc"}
ap=argparse.ArgumentParser()
ap.add_argument("--products", default="air,slp,rhum")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--lams", default="0.045,0.09,0.18,0.27,0.36,0.72")
a=ap.parse_args()
prods=a.products.split(","); seeds=[int(s) for s in a.seeds.split(",")]
lams=[float(x) for x in a.lams.split(",")]
OUT=OUTDIR+"/A15_lam_sweep.csv"
done=set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)): done.add((r["product"],r["arm"],float(r["lam"]),int(r["seed"])))
else:
    open(OUT,"w").write("product,arm,lam,seed,seconds,rank,iters,converged,auprc,prec_at_k,nnz_frac\n")
print("products %s seeds %s λ %s"%(prods,seeds,lams), flush=True)
for p in prods:
    path=os.path.join(L.RAW,L.PRODUCTS[p]); f_main=None
    for sd in seeds:
        todo=[(arm,lm) for lm in lams for arm in ["A-adapt","B-fixed"] if (p,arm,lm,sd) not in done]
        if not todo: continue
        b=L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T,nlat,nlon=b["observed"].shape
        X=b["observed"].reshape(T,-1).T.copy(); r=L.mp_rank(X)
        if f_main is None:
            fn=L.SphereFilter(b["lat"],b["lon"],L.FLOS["alpha"],L.FLOS["s"])
            f_main=FT.SphereFilterT(fn.U,fn.inv,fn.node_of,device=DEVICE,dtype=torch.float32)
        Xt=torch.as_tensor(X,dtype=torch.float32,device=DEVICE)
        to3=lambda M: M.T.reshape(T,nlat,nlon)
        for arm,lm in todo:
            SYNC(); t0=time.perf_counter()
            Lm,S,info=FT.run_flos_ablate(Xt,f_main,lam=lm,p=L.FLOS["p"],
                max_iter=L.FLOS["max_iter"],tol=L.FLOS["tol"],
                paper_iters=L.FLOS["paper_iters"],rank=r,
                ablation=("fixed-thresh" if arm=="B-fixed" else None))
            SYNC(); sec=time.perf_counter()-t0
            nnz=float((S!=0).float().mean())
            row=L.score(b,scores=np.abs(to3(S.double().cpu().numpy())))
            open(OUT,"a").write("%s,%s,%g,%d,%.1f,%s,%s,%s,%.5f,%.5f,%.4f\n"%(
                p,arm,lm,sd,sec,info["rank"],info["iters"],info["converged"],
                row["auprc"],row["prec_at_k"],nnz))
            print("  %-6s seed %-2d λ=%-6g %-8s AUPRC %.5f nonzero %.3f %.0f s"%(
                p,sd,lm,arm,row["auprc"],nnz,sec), flush=True)
            del Lm,S
        del Xt; FREE()
print("done", flush=True)
