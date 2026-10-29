#!/usr/bin/env python3
"""Summarise the results of A36 (tapered benchmark) and A37 (fairness checks).
Usage: python3 eval_A36_A37.py <analysis/data dir>"""
import csv, glob, sys, statistics as st
from collections import defaultdict
from scipy.stats import spearmanr, wilcoxon

D = sys.argv[1]
DEV = {("ncep", "air"), ("ncep", "slp"), ("ncep", "rhum")}
mt = {(r["source"], r["product"]): r for r in csv.DictReader(open(D + "/main/main_table.csv"))}
ALL = sorted(mt); HO = [k for k in ALL if k not in DEV]
FL = st.mean(float(mt[k]["FLoS"]) for k in HO); SRT = st.mean(float(mt[k]["Sort"]) for k in HO)
print("sharp benchmark reference: FLoS %.4f  Sort %.4f  (diff %+.4f)" % (FL, SRT, FL - SRT))


def rows(pat):                       # a configuration re-run after a crash appears twice; keep its first record
    out, seen = [], set()
    for f in sorted(glob.glob(D + "/tpami/" + pat)):
        for r in csv.DictReader(open(f)):
            key = tuple((k, v) for k, v in r.items() if k not in ("auprc", "prec_at_k", "seconds", "iters", "nnz", "nnz_frac"))
            if key not in seen: seen.add(key); out.append(r)
    return out


# ------------------------------------------------------------------ A36
t = defaultdict(lambda: defaultdict(list))
for r in rows("A36_taper_*.csv"): t[r["method"]][(r["source"], r["product"])].append(float(r["auprc"]))
c = defaultdict(lambda: defaultdict(list))
for r in rows("A36_contrast_*.csv"):
    for col in ("rhoK_sharp", "rhoK_taper", "rho_obs", "dom_viol_sharp", "dom_viol_taper"): c[col][(r["source"], r["product"])].append(float(r[col]))
if t:
    complete = [k for k in HO if all(len(t[m].get(k, [])) == 3 for m in ("FLoS", "CORE", "Sort", "FRPCAG"))]
    print("\nA36: held-out variables complete: %d / 19" % len(complete))
    if complete:
        mean = {m: st.mean(st.mean(t[m][k]) for k in complete) for m in ("FLoS", "CORE", "Sort", "FRPCAG")}
        print("  tapered held-out means:", {m: round(v, 4) for m, v in mean.items()})
        print("  sharp FLoS-Sort on the same variables: %+.4f" % st.mean(float(mt[k]["FLoS"]) - float(mt[k]["Sort"]) for k in complete))
        d = [st.mean(t["FLoS"][k]) - st.mean(t["Sort"][k]) for k in complete]
        print("  P2 tapered FLoS - Sort = %+.4f (threshold +0.03) -> %s ; wins %d/%d" % (st.mean(d), "PASS" if st.mean(d) >= 0.03 else "FAIL", sum(x > 0 for x in d), len(d)))
        print("  P3 FLoS highest tapered mean ->", "PASS" if max(mean, key=mean.get) == "FLoS" else "FAIL (%s highest)" % max(mean, key=mean.get))
        kk = [k for k in complete if k in c["rhoK_taper"]]
        if len(kk) >= 5:
            print("  P4 Spearman(rhoK taper, FLoS-Sort taper) = %.3f" % spearmanr([st.mean(c["rhoK_taper"][k]) for k in kk], [st.mean(t["FLoS"][k]) - st.mean(t["Sort"][k]) for k in kk]).correlation)
        for k in complete:
            print("   %-5s %-6s taper FLoS %.3f CORE %.3f Sort %.3f FRPCAG %.3f | sharp FLoS %.3f Sort %.3f" % (k + tuple(st.mean(t[m][k]) for m in ("FLoS", "CORE", "Sort", "FRPCAG")) + (float(mt[k]["FLoS"]), float(mt[k]["Sort"]))))
    ks = [k for k in ALL if len(c["rhoK_taper"].get(k, [])) == 3]
    if ks:
        lower = sum(st.mean(c["rhoK_taper"][k]) < st.mean(c["rhoK_sharp"][k]) for k in ks)
        print("  P1 rhoK taper < sharp on %d / %d variables (threshold 18/22)" % (lower, len(ks)))
        ho = [k for k in HO if k in ks]
        if len(ho) >= 5:
            print("  P5 Spearman(rho_obs, sharp FLoS-Sort) over %d held-out = %.3f" % (len(ho), spearmanr([st.mean(c["rho_obs"][k]) for k in ho], [float(mt[k]["FLoS"]) - float(mt[k]["Sort"]) for k in ho]).correlation))
            print("  extra: exact-H rhoK sharp vs sharp gain Spearman = %.3f" % spearmanr([st.mean(c["rhoK_sharp"][k]) for k in ho], [float(mt[k]["FLoS"]) - float(mt[k]["Sort"]) for k in ho]).correlation)
            print("  extra: rho_obs per variable:", {k: round(st.mean(c["rho_obs"][k]), 3) for k in ho})
            print("  extra: exact-H prefix dominance violations sharp (max):", round(max(max(c["dom_viol_sharp"][k]) for k in ks), 4),
                  " variables with no violation:", sum(max(c["dom_viol_sharp"][k]) <= 1e-9 for k in ks))


# ------------------------------------------------------------------ A37 families: select on dev rows, evaluate on held-out
def select_eval(name, rs, keyf, valf=lambda r: float(r["auprc"])):
    g = defaultdict(lambda: defaultdict(list))
    for r in rs: g[keyf(r)][(r["source"], r["product"])].append(valf(r))
    dev = {cfg: st.mean(st.mean(v[k]) for k in DEV) for cfg, v in g.items() if all(len(v.get(k, [])) == 3 for k in DEV)}
    if not dev: return None
    best = max(dev, key=dev.get)
    ho = [k for k in HO if len(g[best].get(k, [])) == 3]
    hm = st.mean(st.mean(g[best][k]) for k in ho) if ho else float("nan")
    print("\n%s: dev-selected %s (dev %.4f; grid %s) held-out mean %.4f over %d vars (FLoS %.4f, Sort %.4f)" % (
        name, best, dev[best], sorted(dev), hm, len(ho), FL, SRT))
    return best, dev, hm, len(ho)

select_eval("A37a EOF", [r for r in rows("A37_eof_*.csv")], lambda r: int(r["param1"]))
hp = rows("A37_hp_*.csv")
select_eval("A37b one-shot high-pass", [r for r in hp if r["family"] == "HP"], lambda r: (float(r["param1"]), float(r["param2"])))
select_eval("A37b low-mode removal", [r for r in hp if r["family"] == "LOWMODE"], lambda r: int(r["param1"]))
gl = rows("A37_gloss_*.csv")
select_eval("A37d GLOSS absolute", gl, lambda r: float(r["lam"]))
rp = rows("A37_rpcag_*.csv")
select_eval("A37e RPCAG", rp, lambda r: (float(r["s"]), float(r["lam_mult"]), float(r["gamma"])))
ff = rows("A37_frpcagfrac_*.csv")
select_eval("A37f FRPCAG-frac", ff, lambda r: (float(r["g_s"]), float(r["g_t"])))
ro = rows("A37_rounds_*.csv")
if ro:
    g = defaultdict(lambda: defaultdict(list))
    for r in ro: g[int(r["param1"])][(r["source"], r["product"])].append(float(r["auprc"]))
    ks = [k for k in HO if all(len(g[it].get(k, [])) == 3 for it in g)]
    if ks:
        print("\nA37c FLoS AUPRC by round over %d held-out vars:" % len(ks), {it: round(st.mean(st.mean(g[it][k]) for k in ks), 4) for it in sorted(g)})
        if 500 in g and 1000 in g:
            dd = st.mean(st.mean(g[1000][k]) for k in ks) - st.mean(st.mean(g[500][k]) for k in ks)
            print("  P(c) |AUPRC(1000) - AUPRC(500)| = %.4f -> %s" % (abs(dd), "PASS" if abs(dd) <= 0.01 else "FAIL"))
