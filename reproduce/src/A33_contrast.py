#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A33: the high-pass contrast that the theory says governs detection.

Measures the high-pass contrast on the benchmark.  No decomposition is run.  For every variable and
seed we build the benchmark and measure, with the background operator F of FLoS (alpha=0.85, s=0.25) and K = I - F,
    rho_K = (||K E||^2 / ||E||^2) / (||K B||^2 / ||B||^2),      E = inserted events, B = observed field before insertion,
the factor by which the high-pass K multiplies the event-to-background energy ratio (Proposition 3 in the paper);
rho_K_evm restricts B to the months that receive events.  For reference also the plain Dirichlet ratio
Q(E)/Q(B) with Q(Y) = tr((AY)' D (AY)) / tr((AY)'(AY)) (exponent 1, the quantity of A1''/A5).
Additional measurements:
  * prefix dominance of Proposition 3: energy CDFs of E and of B (event months) over the Laplacian eigenmodes sorted
    by roughness; we record the share of modes k with CDF_E(k) <= CDF_B(k) and the largest violation;
  * per-variable energy histograms over 24 log-spaced roughness bins (E and B), and the normalised singular values of
    the observed field (seed 0) for the spectrum figure -> A33_spectra.npz
"""
import sys, os, time, csv, argparse
import numpy as np, torch
from paths import DATA, OUT as OUTDIR, CACHE, device
import A5_leaderboard as L
import flos_torch as FT

DEVICE = device()
SRC = {"ncep": (DATA + "/raw", {"air": "air.mon.mean.nc", "slp": "slp.mon.mean.nc", "rhum": "rhum.sig995.mon.mean.nc",
                                "lftx": "lftx.sfc.mon.mean.nc", "pres": "pres.sfc.mon.mean.nc",
                                "omega": "omega.sig995.mon.mean.nc", "pottmp": "pottmp.sig995.mon.mean.nc",
                                "gpcp": "gpcp.mon.mean.nc"}),
       "era5": (DATA + "/era5/clean", {k: "era5_%s.mon.nc" % k for k in
                                       ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}),
       "20cr": (DATA + "/20cr/clean", {k: "20cr_%s.mon.nc" % k for k in
                                       ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]})}
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args()
OUT = "%s/A33_contrast.csv" % OUTDIR
open(OUT, "w").write("source,product,seed,rho_K,rho_K_evm,gainE,gainB,gainB_evm,Q_E,Q_B,Q_ratio,"
                     "dom_share,dom_maxviol\n")
L.CACHE = CACHE
geo = None
spectra = {}
BINS = np.logspace(-5, 0, 25)
for src, (raw, prods) in SRC.items():
    L.RAW, L.PRODUCTS = raw, prods
    for p in prods:
        for sd in map(int, a.seeds.split(",")):
            b = L.make_benchmark(os.path.join(raw, prods[p]), seed=sd, n_events=L.N_EVENTS)
            T, nlat, nlon = b["observed"].shape
            key = (nlat, nlon)
            if geo != key:
                fn = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], L.FLOS["s"])
                F = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=DEVICE, dtype=torch.float64)
                ev_np = L.graph_eig(b["lat"], b["lon"])[0]
                ev = torch.as_tensor(ev_np, dtype=torch.float64, device=DEVICE)
                order = torch.argsort(ev)
                geo = key
            E = torch.as_tensor(b["S"].reshape(T, -1).T, dtype=torch.float64, device=DEVICE)
            B = torch.as_tensor(b["background"].reshape(T, -1).T, dtype=torch.float64, device=DEVICE)
            evm = torch.as_tensor(np.unique(np.argwhere(b["support"])[:, 0]), device=DEVICE)
            KE = E - F(E); KB = B - F(B)
            gE = float((KE ** 2).sum() / (E ** 2).sum()); gB = float((KB ** 2).sum() / (B ** 2).sum())
            gBm = float((KB[:, evm] ** 2).sum() / (B[:, evm] ** 2).sum())

            def modal_energy(Y):
                acc = torch.zeros(F.n, Y.shape[1], dtype=Y.dtype, device=Y.device)
                acc.index_add_(0, F.node, Y)
                C = F.U.T @ (acc * F.w)
                return (C ** 2).sum(1)
            eE, eB = modal_energy(E[:, evm]), modal_energy(B[:, evm])
            qE = float((eE * ev).sum() / eE.sum()); qB = float((eB * ev).sum() / eB.sum())
            cE = torch.cumsum(eE[order], 0) / eE.sum(); cB = torch.cumsum(eB[order], 0) / eB.sum()
            dom_share = float((cE <= cB + 1e-12).double().mean()); dom_viol = float(torch.clamp(cE - cB, min=0).max())
            open(OUT, "a").write("%s,%s,%d,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n" % (
                src, p, sd, gE / gB, gE / gBm, gE, gB, gBm, qE, qB, qE / qB, dom_share, dom_viol))
            print("%-5s %-6s seed%d rho_K %.3f (evm %.3f)  Q ratio %.3f  dominance %.3f (max viol %.3f)" % (
                src, p, sd, gE / gB, gE / gBm, qE / qB, dom_share, dom_viol), flush=True)
            if sd == 0:
                X = b["observed"].reshape(T, -1).T
                sv = np.linalg.svd(X, compute_uv=False)
                hE = np.histogram(ev_np, bins=BINS, weights=eE.cpu().numpy())[0]
                hB = np.histogram(ev_np, bins=BINS, weights=eB.cpu().numpy())[0]
                spectra["%s_%s_sv" % (src, p)] = sv / sv[0]
                spectra["%s_%s_hE" % (src, p)] = hE / hE.sum()
                spectra["%s_%s_hB" % (src, p)] = hB / hB.sum()
            del E, B, KE, KB
spectra["bins"] = BINS
np.savez("%s/A33_spectra.npz" % OUTDIR, **spectra)
print("A33 done", time.strftime("%F %T"))
