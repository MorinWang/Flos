#!/usr/bin/env python3
"""Summarise the A39 truncation ablation from the result CSVs and run checks (a)-(e), printing pass/fail for each.
Usage: python3 a39_eval.py <analysis/data dir>"""
import sys, csv, glob, statistics as st
from collections import defaultdict
from scipy.stats import spearmanr, wilcoxon

D = sys.argv[1]
DEV = {("ncep", "air"), ("ncep", "slp"), ("ncep", "rhum")}
ROUGH = {("era5", "rhum"), ("20cr", "rhum"), ("ncep", "gpcp")}
main = defaultdict(list)
for r in csv.DictReader(open(D + "/main/main_table_long.csv")):
    if r["method"] == "FLoS": main[(r["source"], r["product"])].append(float(r["auprc"]))
a39 = defaultdict(list)
for f in glob.glob(D + "/tpami/A39_notrunc_*.csv"):
    if "sanity" in f: continue
    for r in csv.DictReader(open(f)):
        a39[(r["source"], r["product"], r["arm"], float(r["lam"]))].append(float(r["auprc"]))
adapt = defaultdict(list)
for f in glob.glob(D + "/tpami/A30_core_ho_adapt_*.csv"):
    for r in csv.DictReader(open(f)): adapt[(r["source"], r["product"])].append(float(r["auprc"]))
con = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(D + "/tpami/A33_contrast.csv")):
    for c in ("rho_K", "gainB"): con[c][(r["source"], r["product"])].append(float(r[c]))

HO = sorted(k for k in main if k not in DEV)
mean = lambda d, k: st.mean(d[k]) if len(d.get(k, [])) else float("nan")
fl = {k: mean(main, k) for k in HO}
full = {k: mean(a39, k + ("full", 0.72)) for k in HO}
noT = {k: mean(a39, k + ("noT", 0.72)) for k in HO}
ok = lambda d: all(x == x for x in d.values()) and len(d) == 19
print("held-out variables:", len(HO), "| complete: full", ok(full), "noT", ok(noT))
M = lambda d: st.mean(d.values())
if ok(full):
    print("(a) reproduction: full %.4f vs main table %.4f, diff %+.4f, max per-variable |diff| %.4f -> %s" % (
        M(full), M(fl), M(full) - M(fl), max(abs(full[k] - fl[k]) for k in HO), "pass" if abs(M(full) - M(fl)) <= 0.005 else "fail"))
if ok(noT):
    g = {k: fl[k] - noT[k] for k in HO}
    print("(b) noT %.4f, FLoS %.4f, gap %.4f (need >= 0.03) -> %s; Wilcoxon p=%.2g, FLoS better on %d/19" % (
        M(noT), M(fl), M(g), "pass" if M(g) >= 0.03 else "fail", wilcoxon([fl[k] for k in HO], [noT[k] for k in HO]).pvalue,
        sum(g[k] > 0 for k in HO)))
    ad = M({k: mean(adapt, k) for k in HO})
    print("(c) noT %.4f vs FLoS-C adaptive %.4f, diff %+.4f (need |diff| <= 0.015) -> %s" % (
        M(noT), ad, M(noT) - ad, "pass" if abs(M(noT) - ad) <= 0.015 else "fail"))
    up = [k for k in ROUGH if noT[k] > fl[k]]; down = [k for k in HO if k not in ROUGH and noT[k] < fl[k]]
    print("(d) noT above FLoS on %d of the 3 rough variables %s; below FLoS on %d of the other 16 -> %s" % (
        len(up), up, len(down), "pass" if len(up) >= 2 and len(down) >= 13 else "fail"))
    for c in ("rho_K", "gainB"):
        x = [st.mean(con[c][k]) for k in HO]; rho, p = spearmanr(x, [g[k] for k in HO])
        print("    measure: Spearman(truncation gain, %s) = %.2f (p=%.2g)" % (c, rho, p))
    for k in sorted(HO, key=lambda k: g[k]): print("      %-5s %-7s FLoS %.3f noT %.3f gain %+.3f" % (k[0], k[1], fl[k], noT[k], g[k]))
# development selection for (e)
dev = defaultdict(list)
for (s, p, arm, lam), v in a39.items():
    if (s, p) in DEV and arm == "noT": dev[lam] += v
if dev:
    print("dev noT:", {l: round(st.mean(v), 4) for l, v in sorted(dev.items())}, "n", {l: len(v) for l, v in sorted(dev.items())})
    best = max(dev, key=lambda l: st.mean(dev[l])); print("    selected lambda", best)
    sel = {k: mean(a39, k + ("noT", best)) for k in HO}
    if ok(sel):
        print("(e) noT at selected lambda %g: %.4f, gap to FLoS %.4f (need >= 0.02) -> %s" % (
            best, M(sel), M(fl) - M(sel), "pass" if M(fl) - M(sel) >= 0.02 else "fail"))
