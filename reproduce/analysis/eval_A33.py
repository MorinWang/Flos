import csv, sys, statistics as st
from scipy.stats import spearmanr
D = sys.argv[1]
mt = {(r['source'], r['product']): r for r in csv.DictReader(open(D + '/main/main_table.csv'))}
rows = list(csv.DictReader(open(D + '/tpami/A33_contrast.csv')))
agg = {}
for r in rows:
    k = (r['source'], r['product']); agg.setdefault(k, []).append(r)
def m(k, c): return st.mean(float(x[c]) for x in agg[k])
keys = sorted(agg)
print("%-6s %-7s %7s %7s %7s %7s %6s %8s %8s" % ("src", "prod", "rhoK", "rhoKevm", "Qratio", "domshr", "viol", "FLoS-Srt", "FLoS-Bst"))
base = ['ScaledGD', 'Sort', 'Mondal', 'PCP', 'EOF', 'GLOSS']
res = {}
for k in keys:
    t = mt[k]; fl = float(t['FLoS']); srt = float(t['Sort']); bst = max(float(t[b]) for b in base)
    res[k] = (m(k, 'rho_K'), m(k, 'rho_K_evm'), m(k, 'Q_ratio'), m(k, 'dom_share'), max(float(x['dom_maxviol']) for x in agg[k]), fl - srt, fl - bst)
    print("%-6s %-7s %7.3f %7.3f %7.3f %7.3f %6.3f %+8.4f %+8.4f" % ((k[0], k[1]) + res[k]))
sub = [k for k in keys if k[0] in ('era5', '20cr')]
r1 = spearmanr([res[k][0] for k in sub], [res[k][5] for k in sub])
print("\nP1 (14 ERA5+20CRv3): Spearman(rho_K, FLoS-Sort) = %.3f  p=%.4f  -> %s" % (r1.correlation, r1.pvalue, "PASS" if r1.correlation >= 0.5 else "FAIL"))
print("P2: rho_K > 1 on %d / 22" % sum(res[k][0] > 1 for k in keys))
lose = {('ncep','rhum'),('ncep','gpcp'),('era5','rhum'),('era5','omega'),('20cr','rhum')}
low5 = sorted(keys, key=lambda k: res[k][0])[:5]
print("P3: lowest-5 rho_K:", low5, " overlap with FLoS losses:", len(set(low5) & lose), "-> %s" % ("PASS" if len(set(low5) & lose) >= 3 else "FAIL"))
for nm, idx in (("all 22 rho_K", 0), ("all 22 rho_K_evm", 1), ("all 22 Q ratio", 2)):
    r = spearmanr([res[k][idx] for k in keys], [res[k][5] for k in keys])
    print("extra: Spearman(%s, FLoS-Sort) = %.3f p=%.4f" % (nm, r.correlation, r.pvalue))
r = spearmanr([res[k][0] for k in sub], [res[k][6] for k in sub]); print("extra: 14-var Spearman(rho_K, FLoS-best) = %.3f p=%.4f" % (r.correlation, r.pvalue))
print("extra: dominance share min/median: %.3f / %.3f ; max violation over all: %.4f" % (min(res[k][3] for k in keys), st.median(res[k][3] for k in keys), max(res[k][4] for k in keys)))
print("extra: variables with full prefix dominance (viol==0):", sum(res[k][4] <= 1e-9 for k in keys), "/ 22")
