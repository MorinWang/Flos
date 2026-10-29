#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A24: tensor robust decomposition by scaled gradient descent (ScaledGD) of Dong, Tong, Ma and Chi
(official repository @a0f97aa) evaluated on our benchmark.
We call rpca(Y, ranks, z0, z1, eta, decay, T, epsilon, device) from its src/rpca.py unmodified;
settings follow its own scripts: T=100, epsilon=1e-7, single precision (the code assumes torch's default dtype);
search ranges follow its synthetic_bayesian_train.py: eta in [0,3], decay in [0.001,0.999].
Four choices that the original code does not make and that we have to fix:
  1. Tensor layout: grid point x month x year (10512 x 12 x 78), dropping the last two months; same layout as A23 (Mondal).
  2. Ranks: the original is always given the true ranks. We include the three mode ranks in the development-set search:
     grid-point mode in [5,900] on a log scale (at most the short side 936 of the unfolding), month in [1,12], year in [1,78].
  3. Thresholds: its z0 and z1 are absolute values with range [0.001,0.05], matched to the amplitude of its synthetic
     data. Our benchmark is already in units of a global robust scale, so we search [0.01,100] on a log scale, which
     covers the 0.045-23 range of our own lambda sweep.
  4. Tuning and scoring: as in A23, mean AUPRC over seed 0 of the three development products, tuned once and then
     frozen; |S| is scored; on the held-out set the same ladder of post-hoc thresholds is added.
The self-supervised unrolled variant is not included: from a reading of its code we estimate that the backpropagation
graph for full-size training with T=100 needs about 38-78 GB, which does not fit in memory.
Usage as in A23: --probe / --tune / --eval; --dry only tests the pipeline.
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
ap.add_argument("--params", default="300,6,20,1.0,1.0,0.5,0.9",
                help="r_space,r_month,r_year,z0,z1,eta,decay; with --probe, several sets may be separated by semicolons")
ap.add_argument("--trials", type=int, default=100)
ap.add_argument("--db", default="A24_optuna.db")
ap.add_argument("--T", type=int, default=100)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--dry", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.dont_write_bytecode = True                                   # do not leave __pycache__ in the external repository
import torch
torch.set_num_threads(a.threads)
DIAG11 = os.path.dirname(os.path.abspath(__file__))                     # all modules live in src/
from paths import DATA as _DATA, EXTERNAL
DATA = _DATA
if os.environ.get("FLOS_REFIMPL"):
    sys.path.insert(0, os.environ["FLOS_REFIMPL"])
sys.path.insert(0, DIAG11)
import A5_leaderboard as L
sys.path.insert(0, os.environ.get("FLOS_SCALEDGD", os.path.join(EXTERNAL, "Tensor_RPCA_ScaledGD", "src")))
from rpca import rpca                                            # sets the tensorly backend to pytorch on import

ERA5 = {"air": "era5_air.mon.nc", "slp": "era5_slp.mon.nc", "pres": "era5_pres.mon.nc", "pottmp": "era5_pottmp.mon.nc",
        "rhum": "era5_rhum.mon.nc", "omega": "era5_omega.mon.nc", "cape": "era5_cape.mon.nc"}
TWENTYCR = {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}   # third reanalysis (A25)
SRC = {"ncep": (DATA + "/raw", L.PRODUCTS), "era5": (DATA + "/era5/clean", ERA5), "20cr": (DATA + "/20cr/clean", TWENTYCR)}
DEV = [("ncep", "air"), ("ncep", "slp"), ("ncep", "rhum")]
LADDER = [1.0, 0.5, 0.3, 0.17, 0.08, 0.04, 0.024, 0.014, 0.0076, 0.003, 0.001, 0.0005]   # identical to A21_rawthresh
KEYS = ["r_space", "r_month", "r_year", "z0", "z1", "eta", "decay"]


def build(source, product, seed):
    raw, prods = SRC[source]
    b = L.make_benchmark(os.path.join(raw, prods[product]), seed=seed, n_events=L.N_EVENTS)
    O3 = b["observed"]
    assert not np.isnan(O3).any(), "observations contain missing values"
    T, nlat, nlon = O3.shape; ny = T // 12; Tn = ny * 12
    Y = O3[:Tn].reshape(ny, 12, nlat, nlon).transpose(2, 3, 1, 0).reshape(nlat * nlon, 12, ny).copy()
    back = Y.reshape(nlat, nlon, 12, ny).transpose(3, 2, 0, 1).reshape(Tn, nlat, nlon)
    assert np.array_equal(back, O3[:Tn])
    return dict(b=b, Y=Y, shape=(T, nlat, nlon, ny, Tn))


def run(case, prm):
    T, nlat, nlon, ny, Tn = case["shape"]
    if a.dry:
        return np.abs(case["b"]["observed"]), dict(sec=0.0, finite=True)
    Y = torch.as_tensor(case["Y"], dtype=torch.float32).to(a.device)
    ranks = [int(prm["r_space"]), int(prm["r_month"]), int(prm["r_year"])]
    t0 = time.perf_counter()
    _, St = rpca(Y, ranks, prm["z0"], prm["z1"], prm["eta"], prm["decay"], a.T, 1e-7, a.device)
    sec = time.perf_counter() - t0
    S = St.detach().abs().double().cpu().numpy().reshape(nlat, nlon, 12, ny).transpose(3, 2, 0, 1).reshape(Tn, nlat, nlon)
    S3 = np.zeros((T, nlat, nlon)); S3[:Tn] = S
    del St, Y
    return S3, dict(sec=sec, finite=bool(np.isfinite(S).all()))


def need_finite(S3, info):
    if not info.get("finite", True):
        raise FloatingPointError("S contains inf or NaN; the algorithm diverged")


def writer(path, header):
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(header); fh.flush()
    return fh, w


def parse(s):
    v = [float(x) for x in s.split(",")]
    return dict(zip(KEYS, v))


print("A24 %s device %s threads %d T=%d%s" % ("probe" if a.probe else "tune" if a.tune else "eval", a.device,
      torch.get_num_threads(), a.T, " [dry]" if a.dry else ""), flush=True)

if a.probe:
    fh, w = writer(a.out, ["source", "product", "seed"] + KEYS + ["device", "sec", "finite", "nnz_frac", "peak_rss_gb"])
    for p in a.products.split(","):
        for sd in [int(s) for s in a.seeds.split(",")]:
            case = build(a.source, p, sd)
            for ps in a.params.split(";"):
                prm = parse(ps)
                S3, info = run(case, prm)
                nnz = float((S3 != 0).mean()); rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
                w.writerow([a.source, p, sd] + [prm[k] for k in KEYS] + [a.device, "%.1f" % info["sec"], info["finite"], "%.5f" % nnz, "%.1f" % rss]); fh.flush()
                print("  %s %s seed %d %.1fs finite %s kept %.4f peak RSS %.1fG" % (a.source, p, sd, info["sec"], info["finite"], nnz, rss), flush=True)

elif a.tune:
    import optuna
    fh, w = writer(a.out, ["trial", "source", "product", "seed"] + KEYS + ["device", "sec", "auprc", "nnz_frac"])
    cases = {pp: build(pp[0], pp[1], 0) for pp in DEV}
    study = optuna.create_study(study_name="A24_scaledgd_dev", storage="sqlite:///" + os.path.abspath(a.db),
                                direction="maximize", sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=True)

    def objective(trial):
        prm = dict(r_space=trial.suggest_int("r_space", 5, 900, log=True),
                   r_month=trial.suggest_int("r_month", 1, 12),
                   r_year=trial.suggest_int("r_year", 1, 78),
                   z0=trial.suggest_float("z0", 0.01, 100, log=True),
                   z1=trial.suggest_float("z1", 0.01, 100, log=True),
                   eta=trial.suggest_float("eta", 0, 3),
                   decay=trial.suggest_float("decay", 0.001, 0.999))
        vals = []
        for (src, p), case in cases.items():
            S3, info = run(case, prm); need_finite(S3, info)
            au = L.score(case["b"], scores=S3)["auprc"]; vals.append(au)
            w.writerow([trial.number, src, p, 0] + [prm[k] for k in KEYS] + [a.device, "%.1f" % info["sec"], "%.5f" % au,
                        "%.5f" % float((S3 != 0).mean())]); fh.flush()
        print("  trial %d written %.0fs" % (trial.number, info["sec"]), flush=True)
        return float(np.mean(vals))

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    study.optimize(objective, n_trials=max(0, a.trials - done), catch=(Exception,))
    best = study.best_trial
    print("A24 tuning done: %d trials completed, best trial #%d %s" % (
        len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]), best.number,
        ",".join("%s=%.6g" % (k, best.params[k]) for k in KEYS)), flush=True)

else:
    prm = parse(a.params)
    fh, w = writer(a.out, ["source", "product", "arm"] + KEYS + ["nnz_target", "seed", "auprc", "prec_at_k", "pos_rate", "n_pos",
                                                                 "nnz_frac", "sec"])
    done = set()
    for r in csv.DictReader(open(a.out)):
        done.add((r["source"], r["product"], int(r["seed"])))
    for p in a.products.split(","):
        for sd in [int(s) for s in a.seeds.split(",")]:
            if (a.source, p, sd) in done:
                continue
            case = build(a.source, p, sd)
            try:
                S3, info = run(case, prm); need_finite(S3, info)
            except Exception as e:                           # divergence or error: write a row with empty scores and continue; counted as a random ranking when aggregating
                w.writerow([a.source, p, "ScaledGD"] + [prm[k] for k in KEYS] + [1.0, sd, "", "", "", "", "", "fail:" + str(e)[:60]]); fh.flush()
                print("  %s %s seed %d failed: %s" % (a.source, p, sd, str(e)[:80]), flush=True)
                continue
            flat = S3.ravel()
            for tgt in LADDER:
                if tgt >= 1.0:
                    Sc = S3
                else:
                    cut = np.quantile(flat, 1.0 - tgt); Sc = np.where(S3 >= cut, S3, 0.0)
                row = L.score(case["b"], scores=Sc)
                w.writerow([a.source, p, "ScaledGD"] + [prm[k] for k in KEYS] + [tgt, sd, "%.5f" % row["auprc"], "%.5f" % row["prec_at_k"],
                            "%.2g" % row["pos_rate"], row["n_pos"], "%.5f" % float((Sc != 0).mean()), "%.1f" % info["sec"]])
            fh.flush()
            print("  %s %s seed %d %.0fs %d thresholds written" % (a.source, p, sd, info["sec"], len(LADDER)), flush=True)
print("A24 done", flush=True)
