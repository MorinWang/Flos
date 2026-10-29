"""Largest consecutive singular-value ratio of each observed field (seed 0), from A33_spectra.npz."""
import numpy as np, csv, sys
D = sys.argv[1]
z = np.load(D + '/tpami/A33_spectra.npz')
spec = {(r['source'], r['product']): r for r in csv.DictReader(open(D + '/ablation/A28_spectrum.csv'))}
print("%-5s %-6s %5s %5s %5s | max ratio j>=1 (at j) | max ratio for 5<=j<=mp_rank+50 | ratio at mp_rank | sv[mp]/sv1" % ("src","prod","c95","c99","mp"))
allmax = []
for k in sorted(spec):
    sv = z['%s_%s_sv' % k]; r = np.asarray(sv[:-1] / sv[1:])
    mp = int(spec[k]['mp_rank'])
    j1 = int(np.argmax(r[:400])); lo, hi = 5, min(mp + 50, len(r))
    j2 = lo + int(np.argmax(r[lo:hi]))
    allmax.append(r[lo:hi].max())
    print("%-5s %-6s %5s %5s %5d | %.3f (j=%d) | %.3f (j=%d) | %.4f | %.4f" % (k[0], k[1], spec[k]['comp95'], spec[k]['comp99'], mp,
          r[j1], j1 + 1, r[j2], j2 + 1, r[mp - 1], sv[mp - 1]))
print("max over variables of the largest ratio in [5, mp+50]: %.3f" % max(allmax))
