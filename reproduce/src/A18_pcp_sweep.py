#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A18: give PCP the same tuning budget as FLoS by sweeping its shrinkage strength, yielding a sparsity curve
that can be compared side by side with ours.
PCP's λ = 1/√max(m,n) is fixed by its paper; this script only multiplies it by a factor and leaves everything else
(ρ, tol, iterations, solver) unchanged.
CPU only, so it can run alongside the GPU-based A17 without competing for resources."""
import sys, os, time, csv, argparse
import numpy as np
from scipy import linalg
from paths import DATA, OUT as OUTDIR, CACHE
import A5_leaderboard as L
SRC={"ncep": (DATA+"/raw",
              {"air":"air.mon.mean.nc","lftx":"lftx.sfc.mon.mean.nc","slp":"slp.mon.mean.nc",
               "pres":"pres.sfc.mon.mean.nc","omega":"omega.sig995.mon.mean.nc",
               "pottmp":"pottmp.sig995.mon.mean.nc","rhum":"rhum.sig995.mon.mean.nc",
               "gpcp":"gpcp.mon.mean.nc"}),
     "era5": (DATA+"/era5/clean",
              {"air":"era5_air.mon.nc","slp":"era5_slp.mon.nc","pres":"era5_pres.mon.nc",
               "pottmp":"era5_pottmp.mon.nc","rhum":"era5_rhum.mon.nc",
               "omega":"era5_omega.mon.nc","cape":"era5_cape.mon.nc"})}

def run_pcp_mult(X, mult):
    """Identical to A5_leaderboard.run_pcp except that lam is multiplied by mult."""
    c=L.PCP; m,n=X.shape; lam=mult*1.0/np.sqrt(max(m,n)); dn=np.linalg.norm(X)
    norm_two=linalg.svdvals(X, check_finite=False)[0]
    Y=X/max(norm_two, np.abs(X).max()/lam)
    mu=1.25/norm_two; mu_bar=mu*1e7
    A=np.zeros_like(X); E=np.zeros_like(X); conv=False; svp=0
    for it in range(1, c["max_iter"]+1):
        Tt=X-A+Y/mu
        E=np.maximum(Tt-lam/mu,0)+np.minimum(Tt+lam/mu,0)
        u,s,vt=linalg.svd(X-E+Y/mu, full_matrices=False, check_finite=False)
        svp=int((s>1/mu).sum())
        A=(u[:,:svp]*(s[:svp]-1/mu))@vt[:svp]
        Z=X-A-E; Y=Y+mu*Z; mu=min(mu*c["rho"], mu_bar)
        if np.linalg.norm(Z)/dn < c["tol"]: conv=True; break
    return A,E,dict(rank=svp,iters=it,converged=conv)

ap=argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["ncep","era5"])
ap.add_argument("--products", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--mults", default="0.5,1,2,4,8,16,32,64")
ap.add_argument("--shard", required=True)
a=ap.parse_args()
L.CACHE=CACHE; L.RAW,L.PRODUCTS=SRC[a.source]
prods=a.products.split(","); seeds=[int(x) for x in a.seeds.split(",")]
mults=[float(x) for x in a.mults.split(",")]
OD=OUTDIR; os.makedirs(OD, exist_ok=True)
OUT="%s/A18_pcp_%s.csv"%(OD,a.shard)
done=set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)): done.add((r["source"],r["product"],float(r["mult"]),int(r["seed"])))
else:
    open(OUT,"w").write("source,product,method,mult,seed,seconds,rank,iters,converged,auprc,prec_at_k,nnz_frac\n")
print("shard %s  %s  %s  multipliers %s"%(a.shard,a.source,prods,mults), flush=True)
for p in prods:
    path=os.path.join(L.RAW,L.PRODUCTS[p])
    for sd in seeds:
        todo=[m for m in mults if (a.source,p,m,sd) not in done]
        if not todo: continue
        b=L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T,nlat,nlon=b["observed"].shape
        X=b["observed"].reshape(T,-1).T.copy()
        to3=lambda M: M.T.reshape(T,nlat,nlon)
        for mu_ in todo:
            t0=time.perf_counter(); A,E,info=run_pcp_mult(X,mu_); sec=time.perf_counter()-t0
            nnz=float((E!=0).mean())
            row=L.score(b, scores=np.abs(to3(E)))
            open(OUT,"a").write("%s,%s,PCP-L1,%g,%d,%.1f,%s,%s,%s,%.5f,%.5f,%.5f\n"%(
                a.source,p,mu_,sd,sec,info["rank"],info["iters"],info["converged"],
                row["auprc"],row["prec_at_k"],nnz))
            print("  %-5s %-6s seed %d mult %-5g AUPRC %.5f kept %.5f %.0f s"%(
                a.source,p,sd,mu_,row["auprc"],nnz,sec), flush=True)
print("shard %s done"%a.shard, flush=True)
