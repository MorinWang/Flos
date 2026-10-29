#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A23: LR-STSS of Mondal et al. 2025 (arXiv 2510.00460) evaluated on our benchmark.
Model and solver settings are copied verbatim from lr_stss_select and var2 in the official script
LRSTSS/Varying_Amplitude/jobs_amp_2/LRSTSS_amp.py:
  SNN__LOGN_GTV; low rank = sum of nuclear norms over all modes; sparse = zero-hop neighbour groups (i.e. element-wise l1)
  + spatial-graph GTMV(p=1, symmetric) + temporal-graph GTV(p=1);
  max_iter=100, rho=4*mean|Y|, err_tol=1e-6, rho_update=1, rho_update_thr=100; three parameters psi, lda_l, lda_t,
  with lda=1-psi; search ranges as in the original: psi in [0,1], lda_l and lda_t in [1e-8,10] on a log scale.
Four choices that the original code does not make and that we have to fix:
  1. Tensor layout: grid point x month x year (10512 x 12 x 78), dropping the last two months; the same layout as GLOSS
     on the leaderboard, except that latitude and longitude are merged into a single graph mode.
  2. Spatial graph: unweighted 4-neighbour lat-lon grid, periodic in longitude but not in latitude; the original
     synthetic experiments also use an unweighted grid graph.
  3. Scoring: AUPRC of |S| outside the exclusion zone, the same metric as for all other methods on the leaderboard;
     the original NLL score hard-codes the block size of its synthetic data and is not ported.
  4. Tuning: the original picks a Pareto point of ROC-AUC and F1 using the ground truth; we apply the same rule we use
     for our own method, i.e. tune once against the ground truth on the three development products, maximising AUPRC.
Three modes:
  --probe  timing and kept fraction only, no scoring;
  --tune   tuning on the development set: Optuna TPE (sampler seed 0) for --trials trials; each trial runs once on
           seed 0 of NCEP air temperature, sea-level pressure and relative humidity; objective = mean of the three
           AUPRCs; stored in SQLite so that interrupted runs can resume;
  --eval   run with fixed parameters and report the AUPRC of |S|, plus the same ladder of post-hoc thresholds as
           A21_rawthresh, so that results lie on the same kept-fraction axis.
--dry: do not run the model; S is set to |observed|. Only used to test the scoring, CSV-writing and Optuna pipeline.
"""
import sys, os, time, csv, argparse, resource
import numpy as np

ap = argparse.ArgumentParser()
mode = ap.add_mutually_exclusive_group(required=True)
mode.add_argument("--probe", action="store_true")
mode.add_argument("--tune", action="store_true")
mode.add_argument("--eval", action="store_true")
ap.add_argument("--source", default="ncep", choices=["ncep", "era5", "20cr"])
ap.add_argument("--products", default="air")
ap.add_argument("--seeds", default="0")
ap.add_argument("--params", default="0.5,0.01,0.01", help="psi,lda_l,lda_t; with --probe, several sets may be separated by semicolons")
ap.add_argument("--trials", type=int, default=36)          # 36 trials x 3 products = 108 runs, the same budget as our lambda tuning (12 values x 3 products x 3 seeds)
ap.add_argument("--db", default="A23_optuna.db")
ap.add_argument("--max_iter", type=int, default=100)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--dry", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import torch, networkx as nx
if a.threads:
    torch.set_num_threads(a.threads)
# Paths default to paths.py; on another machine set environment variables instead of editing code
DIAG11 = os.path.dirname(os.path.abspath(__file__))                     # all modules live in src/
from paths import DATA as _DATA, EXTERNAL
DATA = _DATA
if os.environ.get("FLOS_REFIMPL"):
    sys.path.insert(0, os.environ["FLOS_REFIMPL"])
sys.path.insert(0, DIAG11)
import A5_leaderboard as L
MONDAL = os.environ.get("FLOS_MONDAL", os.path.join(EXTERNAL, "Anomaly_Detection"))
if MONDAL not in sys.path:
    sys.path.append(MONDAL)
if not hasattr(nx.Graph, "out_degree"):              # as in A3: undirected graphs in networkx 3 have no out_degree
    nx.Graph.out_degree = property(lambda self: self.degree)
from src.models.lr_ssd.snn__logn_gtv import SNN__LOGN_GTV

ERA5 = {"air": "era5_air.mon.nc", "slp": "era5_slp.mon.nc", "pres": "era5_pres.mon.nc", "pottmp": "era5_pottmp.mon.nc",
        "rhum": "era5_rhum.mon.nc", "omega": "era5_omega.mon.nc", "cape": "era5_cape.mon.nc"}
TWENTYCR = {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}   # third reanalysis (A25)
SRC = {"ncep": (DATA + "/raw", L.PRODUCTS), "era5": (DATA + "/era5/clean", ERA5), "20cr": (DATA + "/20cr/clean", TWENTYCR)}
CFG = dict(lr_modes=[1, 2, 3], graph_modes=[1], grouping="neighbor", weighing="uniform", r_hop=0,
           dtype=torch.float64, verbose=1, report_freq=10,
           gtvr_config=[{"graph": "spatial", "mode": [1], "p": 1, "variation_type": "GTMV", "normalization": "symmetric"},
                        {"graph": "temporal", "mode": [2], "variation_type": "GTV", "p": 1}])
DEV = [("ncep", "air"), ("ncep", "slp"), ("ncep", "rhum")]
LADDER = [1.0, 0.5, 0.3, 0.17, 0.08, 0.04, 0.024, 0.014, 0.0076, 0.003, 0.001, 0.0005]   # identical to A21_rawthresh


def build(source, product, seed):
    raw, prods = SRC[source]
    b = L.make_benchmark(os.path.join(raw, prods[product]), seed=seed, n_events=L.N_EVENTS)
    O3 = b["observed"]
    assert not np.isnan(O3).any(), "observations contain missing values"
    T, nlat, nlon = O3.shape; ny = T // 12; Tn = ny * 12
    Y = O3[:Tn].reshape(ny, 12, nlat, nlon).transpose(2, 3, 1, 0).reshape(nlat * nlon, 12, ny).copy()
    back = Y.reshape(nlat, nlon, 12, ny).transpose(3, 2, 0, 1).reshape(Tn, nlat, nlon)
    assert np.array_equal(back, O3[:Tn])                                         # layout round-trips exactly
    G = nx.grid_2d_graph(nlat, nlon, periodic=[False, True])
    assert list(G.nodes) == [(i, j) for i in range(nlat) for j in range(nlon)]  # node order = flattening order
    return dict(b=b, Y=Y, G=G, shape=(T, nlat, nlon, ny, Tn))


def run(case, psi, lda_l, lda_t):
    T, nlat, nlon, ny, Tn = case["shape"]; Y = case["Y"]
    if a.dry:
        return np.abs(case["b"]["observed"]), dict(build_s=0.0, iter_s=0.0, iters=0, converged=False)
    t0 = time.perf_counter()
    model = SNN__LOGN_GTV(Y, case["G"], device=a.device, **CFG)
    t1 = time.perf_counter()
    _, St = model(psis=[psi] * 3, lda=1 - psi, lda_gtvs=[lda_l, lda_t], max_iter=a.max_iter,
                  rho=4 * np.abs(Y).sum() / Y.size, err_tol=1e-6, rho_update=1, rho_update_thr=100)
    t2 = time.perf_counter()
    S = St.detach().cpu().numpy().reshape(nlat, nlon, 12, ny).transpose(3, 2, 0, 1).reshape(Tn, nlat, nlon)
    S3 = np.zeros((T, nlat, nlon)); S3[:Tn] = np.abs(S)
    info = dict(build_s=t1 - t0, iter_s=t2 - t1, iters=model.it, converged=model.converged)
    del model, St
    if a.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return S3, info


def writer(path, header):
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(header); fh.flush()
    return fh, w


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


print("A23 %s device %s threads %d max_iter %d%s" % ("probe" if a.probe else "tune" if a.tune else "eval", a.device,
      torch.get_num_threads(), a.max_iter, " [dry]" if a.dry else ""), flush=True)

if a.probe:
    fh, w = writer(a.out, ["source", "product", "seed", "psi", "lda_l", "lda_t", "device", "build_s", "iter_s", "iters",
                           "converged", "nnz_frac", "peak_rss_gb"])
    params = [tuple(float(x) for x in s.split(",")) for s in a.params.split(";")]
    for p in a.products.split(","):
        for sd in [int(s) for s in a.seeds.split(",")]:
            case = build(a.source, p, sd)
            for psi, lda_l, lda_t in params:
                S3, info = run(case, psi, lda_l, lda_t)
                nnz = float((S3 != 0).mean())
                w.writerow([a.source, p, sd, psi, lda_l, lda_t, a.device, "%.1f" % info["build_s"], "%.1f" % info["iter_s"],
                            info["iters"], info["converged"], "%.5f" % nnz, "%.1f" % rss_gb()]); fh.flush()
                print("  %s %s seed %d build %.1fs iterate %.1fs %d iters converged %s kept %.4f peak RSS %.1fG" % (
                    a.source, p, sd, info["build_s"], info["iter_s"], info["iters"], info["converged"], nnz, rss_gb()), flush=True)

elif a.tune:
    import optuna
    fh, w = writer(a.out, ["trial", "source", "product", "seed", "psi", "lda_l", "lda_t", "device", "build_s", "iter_s",
                           "iters", "converged", "auprc", "nnz_frac"])
    cases = {pp: build(pp[0], pp[1], 0) for pp in DEV}
    study = optuna.create_study(study_name="A23_mondal_dev", storage="sqlite:///" + os.path.abspath(a.db),
                                direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=True)

    def objective(trial):
        psi = trial.suggest_float("psi", 0, 1)
        lda_l = trial.suggest_float("lda_l", 1e-8, 10, log=True)
        lda_t = trial.suggest_float("lda_t", 1e-8, 10, log=True)
        vals = []
        for (src, p), case in cases.items():
            S3, info = run(case, psi, lda_l, lda_t)
            au = L.score(case["b"], scores=S3)["auprc"]; vals.append(au)
            w.writerow([trial.number, src, p, 0, psi, lda_l, lda_t, a.device, "%.1f" % info["build_s"], "%.1f" % info["iter_s"],
                        info["iters"], info["converged"], "%.5f" % au, "%.5f" % float((S3 != 0).mean())]); fh.flush()
            print("  trial %d %s %s iterate %.0fs %d iters written" % (trial.number, src, p, info["iter_s"], info["iters"]), flush=True)
        return float(np.mean(vals))

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    study.optimize(objective, n_trials=max(0, a.trials - done), catch=(Exception,))
    best = study.best_trial
    print("A23 tuning done: %d trials completed, best trial #%d psi=%.6g lda_l=%.6g lda_t=%.6g" % (
        len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]), best.number,
        best.params["psi"], best.params["lda_l"], best.params["lda_t"]), flush=True)

else:
    psi, lda_l, lda_t = (float(x) for x in a.params.split(","))
    fh, w = writer(a.out, ["source", "product", "arm", "psi", "lda_l", "lda_t", "nnz_target", "seed", "auprc", "prec_at_k",
                           "pos_rate", "n_pos", "nnz_frac", "iter_s", "iters", "converged"])
    done = set()
    for r in csv.DictReader(open(a.out)):
        done.add((r["source"], r["product"], int(r["seed"])))
    for p in a.products.split(","):
        for sd in [int(s) for s in a.seeds.split(",")]:
            if (a.source, p, sd) in done:
                continue
            case = build(a.source, p, sd)
            try:
                S3, info = run(case, psi, lda_l, lda_t)
                if not np.isfinite(S3).all():
                    raise FloatingPointError("S contains inf or NaN")
            except Exception as e:                           # divergence or error: write a row with empty scores and continue; counted as a random ranking when aggregating
                w.writerow([a.source, p, "Mondal-LRSTSS", psi, lda_l, lda_t, 1.0, sd, "", "", "", "", "", "", "", "fail:" + str(e)[:60]]); fh.flush()
                print("  %s %s seed %d failed: %s" % (a.source, p, sd, str(e)[:80]), flush=True)
                continue
            flat = S3.ravel()
            for tgt in LADDER:
                if tgt >= 1.0:
                    Sc = S3
                else:
                    cut = np.quantile(flat, 1.0 - tgt); Sc = np.where(S3 >= cut, S3, 0.0)
                row = L.score(case["b"], scores=Sc)
                w.writerow([a.source, p, "Mondal-LRSTSS", psi, lda_l, lda_t, tgt, sd, "%.5f" % row["auprc"], "%.5f" % row["prec_at_k"],
                            "%.2g" % row["pos_rate"], row["n_pos"], "%.5f" % float((Sc != 0).mean()), "%.1f" % info["iter_s"],
                            info["iters"], info["converged"]])
            fh.flush()
            print("  %s %s seed %d iterate %.0fs %d iters, %d thresholds written" % (a.source, p, sd, info["iter_s"], info["iters"], len(LADDER)), flush=True)
print("A23 done", flush=True)
