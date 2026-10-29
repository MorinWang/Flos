#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A21: give the remaining three baselines the same tuning budget as FLoS.
Each baseline's regularisation is multiplied by a single factor, keeping the internal ratios from its paper
unchanged; nothing else is modified. CPU only."""
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
ap=argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["ncep","era5"])
ap.add_argument("--products", required=True)
ap.add_argument("--methods", default="TV-RPCA-M,GLOSS,EOF-PCA,raw-abs")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--mults", default="0.25,0.5,1,2,4")
ap.add_argument("--shard", required=True)
a=ap.parse_args()
L.CACHE=CACHE; L.RAW,L.PRODUCTS=SRC[a.source]
prods=a.products.split(","); seeds=[int(x) for x in a.seeds.split(",")]
mults=[float(x) for x in a.mults.split(",")]; meths=a.methods.split(",")
OD=OUTDIR; os.makedirs(OD,exist_ok=True)
OUT="%s/A21_opp_%s.csv"%(OD,a.shard)
done=set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)): done.add((r["source"],r["product"],r["method"],float(r["mult"]),int(r["seed"])))
else:
    open(OUT,"w").write("source,product,method,mult,seed,seconds,auprc,nnz_frac,extra\n")
print("shard %s  %s  %s  methods %s  multipliers %s"%(a.shard,a.source,prods,meths,mults), flush=True)
for p in prods:
    path=os.path.join(L.RAW,L.PRODUCTS[p])
    for sd in seeds:
        todo=[(m,mu) for m in meths for mu in (mults if m!="raw-abs" else [1.0])
              if (a.source,p,m,mu,sd) not in done]
        if not todo: continue
        b=L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T,nlat,nlon=b["observed"].shape
        X=b["observed"].reshape(T,-1).T.copy()
        to3=lambda M: M.T.reshape(T,nlat,nlon)
        for m,mu in todo:
            t0=time.perf_counter(); extra=""
            try:
                if m=="TV-RPCA-M":
                    from A5_tvrpca import run_tvrpca
                    B3,F3,info=run_tvrpca(b["observed"], lam=tuple(mu*np.array([0.4,2.0,1.0])), return_all=True)
                    S3=np.abs(info["M"])                       # scored channel M = E + F
                elif m=="GLOSS":
                    # do not modify the validated port; reshape the tensor here exactly as run_gloss_field does and only scale the paper's rule by the factor
                    from A5_gloss import run_gloss
                    O3=b["observed"]; Tt,hh,ww=O3.shape; ny=Tt//12; Tn=ny*12
                    Yt=O3[:Tn].reshape(ny,12,hh,ww).transpose(1,0,2,3).copy()
                    rule=1.0/np.count_nonzero(Yt)              # paper p.16: λ = γ = 1/‖P_Ω(Y)‖₀
                    Lg,Sg,info=run_gloss(Yt, lam=rule*mu, gam=rule*mu)
                    S3=np.zeros_like(O3); S3[:Tn]=Sg.transpose(1,0,2,3).reshape(Tn,hh,ww)
                    S3=np.abs(S3); extra="rule=%.3g x%g"%(rule,mu)
                elif m=="EOF-PCA":
                    r=max(1,int(round(mu*L.mp_rank(X))))
                    u,s,vt=linalg.svd(X, full_matrices=False, check_finite=False)
                    Lm=(u[:,:r]*s[:r])@vt[:r]; S3=np.abs(to3(X-Lm)); extra="rank=%d"%r
                elif m=="raw-abs":
                    S3=np.abs(b["observed"])
                else: continue
            except Exception as e:
                print("  ✗ %s %s mult %g failed: %s"%(p,m,mu,str(e)[:80]), flush=True); continue
            sec=time.perf_counter()-t0
            nnz=float((S3!=0).mean())
            row=L.score(b, scores=S3)
            open(OUT,"a").write("%s,%s,%s,%g,%d,%.1f,%.5f,%.5f,%s\n"%(a.source,p,m,mu,sd,sec,row["auprc"],nnz,extra))
            print("  %-5s %-6s seed %d %-10s mult %-5g AUPRC %.5f kept %.4f %.0f s %s"%(
                a.source,p,sd,m,mu,row["auprc"],nnz,sec,extra), flush=True)
print("shard %s done"%a.shard, flush=True)
