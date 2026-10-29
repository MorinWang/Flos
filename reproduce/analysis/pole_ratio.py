"""Polar vs equatorial weighted degree of the 2.5-degree graph, with and without merging the polar rings.
Mirrors A5_leaderboard.fixed_sphere_laplacian (4-neighbour, longitude wrapped, weight = 1/central angle)."""
import numpy as np
def hav(l1, o1, l2, o2):
    p1, p2 = np.deg2rad(l1), np.deg2rad(l2); dp = np.deg2rad(l2 - l1); dl = np.deg2rad(o2 - o1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(max(1 - a, 0.0)))
lat = np.arange(90, -90.1, -2.5); lon = np.arange(0, 360, 2.5)
def degrees(merge):
    nl, no = len(lat), len(lon); node = np.empty(nl * no, int); k = 0
    for i in range(nl):
        if merge and abs(abs(lat[i]) - 90) < 1e-6:
            node[i*no:(i+1)*no] = k; k += 1
        else:
            node[i*no:(i+1)*no] = np.arange(k, k + no); k += no
    deg = np.zeros(k); seen = set()
    for i in range(nl):
        for j in range(no):
            a = node[i*no+j]
            for ii, jj in ((i+1, j), (i, (j+1) % no)):
                if ii >= nl: continue
                b = node[ii*no+jj]; key = (min(a, b), max(a, b))
                if a == b or key in seen: continue
                seen.add(key); w = 1.0 / max(hav(lat[i], lon[j], lat[ii], lon[jj]), 1e-6)
                deg[a] += w; deg[b] += w
    return deg, node, k
for merge in (False, True):
    deg, node, n = degrees(merge)
    eq = deg[node[36*144 + 0]]          # lat index 36 = equator
    pole = deg[node[0]]                  # north pole cell / node
    near = deg[node[1*144]]              # first ring off the pole (87.5N)
    print("merge=%s nodes=%d  deg(pole)=%.4g deg(87.5N)=%.4g deg(equator)=%.4g  pole/equator=%.4g  87.5N/equator=%.4g  max/min=%.4g"
          % (merge, n, pole, near, eq, pole/eq, near/eq, deg.max()/deg.min()))
