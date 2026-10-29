import csv, glob, statistics as st, sys
from collections import defaultdict
D = sys.argv[1]
mt = {(r['source'], r['product']): r for r in csv.DictReader(open(D + '/main/main_table.csv'))}
DEV = {('ncep','air'),('ncep','slp'),('ncep','rhum')}
HO = [k for k in mt if k not in DEV]
fl = st.mean(float(mt[k]['FLoS']) for k in HO); srt = st.mean(float(mt[k]['Sort']) for k in HO)
print("reference held-out means: FLoS %.4f  Sort %.4f  Fixed %.4f" % (fl, srt, st.mean(float(mt[k]['Fixed']) for k in HO)))
# A30 held-out
rows = []
for f in glob.glob(D + '/tpami/A30_core_ho_*.csv') + glob.glob(D + '/tpami/A30_core_hp?.csv'): rows += list(csv.DictReader(open(f)))
per = defaultdict(lambda: defaultdict(list)); inc = defaultdict(list); conv = defaultdict(list); its = defaultdict(list)
for r in rows:
    per[r['arm']][(r['source'], r['product'])].append(float(r['auprc']))
    if r['arm'].startswith('CORE'):
        conv[r['arm']].append(r['converged'] == 'True'); its[r['arm']].append(int(r['iters']))
        if r['obj_increases'] != '': inc[r['arm']].append(int(r['obj_increases']))
for arm in ('HP1', 'HP1T', 'CORE-soft', 'CORE-adapt'):
    ks = [k for k in HO if k in per[arm]]
    m = st.mean(st.mean(per[arm][k]) for k in ks)
    win_sort = sum(st.mean(per[arm][k]) > float(mt[k]['Sort']) for k in ks)
    print("%-10s n_vars=%d held-out mean %.4f  (FLoS-it %+.4f, it-Sort %+.4f, beats Sort on %d/%d)" % (arm, len(ks), m, fl - m, m - srt, win_sort, len(ks)), end='')
    if arm in conv: print("  converged %d/%d, iters median %d, max %d, runs with objective increase: %s" % (sum(conv[arm]), len(conv[arm]), st.median(its[arm]), max(its[arm]), (sum(x > 0 for x in inc[arm]) if inc[arm] else 'n/a')))
    else: print()
# dev objective check too
dev = list(csv.DictReader(open(D + '/tpami/A30_core_dev.csv')))
soft_dev = [r for r in dev if r['arm'] == 'CORE-soft']
print("dev CORE-soft runs: %d, any objective increase: %d, converged %d" % (len(soft_dev), sum(int(r['obj_increases']) > 0 for r in soft_dev), sum(r['converged'] == 'True' for r in soft_dev)))
allsoft = soft_dev + [r for r in rows if r['arm'] == 'CORE-soft']
print("P30-1: CORE-soft runs with obj increase = %d / %d ; converged within 2000 = %d / %d (%.1f%%)" % (
    sum(int(r['obj_increases']) > 0 for r in allsoft), len(allsoft), sum(r['converged'] == 'True' for r in allsoft), len(allsoft), 100 * sum(r['converged'] == 'True' for r in allsoft) / len(allsoft)))
print("max relative increase observed:", max(float(r['obj_max_rel_increase']) for r in allsoft))
# A32 selection
a32 = list(csv.DictReader(open(D + '/tpami/A32_frpcag_dev.csv')))
g = defaultdict(list)
for r in a32: g[(float(r['g_s']), float(r['g_t']))].append(float(r['auprc']))
best = sorted(g, key=lambda k: -st.mean(g[k]))
for k in best[:6]: print("A32 dev g_s=%g g_t=%g  %.4f (n=%d)" % (k + (st.mean(g[k]), len(g[k]))))
