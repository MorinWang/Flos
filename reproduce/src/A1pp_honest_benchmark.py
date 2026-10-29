# -*- coding: utf-8 -*-
"""A1'': honest benchmark v2.2, with injections by a same-location year swap.
Acceptance checks (printed at the end of __main__):
  (1) the median ratio of injected-event amplitude to local standard deviation differs from the median of the
      sampled population itself by <= 15%;
  (2) the peak pool is not capped: 99th percentile / maximum < 0.97;
  (3) the upper-20% Hill index of the peak pool lies between that of all events and that of the largest 400
      (air 4.64–7.45, lftx 4.92–8.13, slp 6.28–8.62);
  (4) injected events are still rougher than the background; the ratio is > 1 for all three products.
  Also measured, without a criterion: fraction of the injected support that overlaps the exclusion zone.
Differences from A1' (scaling, exclusion zone and roughness check follow A1' exactly; its functions are imported):
  1) no random placement and no random sign: each injected event is the dominant-month slice of a real event,
     moved with its original grid cells, sign and values to the same calendar month of another year (t0 + 12k, k != 0);
  2) sampled population = all real extreme connected components with >= 3 cells in their dominant month
     (A1 / A1' took shapes only from the largest 400).
make_benchmark() can be imported directly by the leaderboard scripts.
"""
import numpy as np, glob, csv, sys, os
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, OUT
from A1p_honest_benchmark import (load_anom, pole_merged_laplacian, to_phys, roughness, hill,
                                  QTHRESH, OLD_CAP)

MIN_CELLS = 3

def make_benchmark(path, seed=0, n_events=60):
    """Return one benchmark instance: observed = background + S; score_mask is the scoring mask that excludes the exclusion zone. Unit = global robust scale."""
    z, lat, lon, sg = load_anom(path)
    T, nlat, nlon = z.shape
    az = np.abs(z); thr = float(np.quantile(az[az>0], QTHRESH))
    real_mask = az > thr                                   # exclusion zone: real extremes already present in the background
    lab, nlab = ndimage.label(real_mask, structure=ndimage.generate_binary_structure(3, 1))
    ids = np.arange(1, nlab+1)
    peaks = np.asarray(ndimage.maximum(az, lab, ids))      # peaks of all real events = same population as M3′
    peak_pos = ndimage.maximum_position(az, lab, ids)
    sizes = np.asarray(ndimage.sum(real_mask, lab, ids))
    top400 = set((np.argsort(sizes, kind="stable")[::-1][:400] + 1).tolist())
    pool = []                                              # (dominant month, grid cells, signed values, among the largest 400)
    for i, slc in enumerate(ndimage.find_objects(lab), start=1):
        idx = np.argwhere(lab[slc] == i) + np.array([s.start for s in slc])
        t0 = int(np.bincount(idx[:,0]).argmax())
        sl = idx[idx[:,0]==t0][:,1:]
        if len(sl) < MIN_CELLS: continue
        pool.append((t0, sl, z[t0, sl[:,0], sl[:,1]].copy(), i in top400))
    rng = np.random.default_rng(seed)
    S = np.zeros_like(z); support = np.zeros(z.shape, dtype=bool); inj = []
    for _ in range(n_events):
        t0, sl, vals, _ = pool[rng.integers(len(pool))]
        ks = [k for k in range(-(t0//12), (T-1-t0)//12 + 1) if k != 0]
        tn = t0 + 12*int(rng.choice(ks))                   # same calendar month, different year
        S[tn, sl[:,0], sl[:,1]] += vals
        support[tn, sl[:,0], sl[:,1]] = True
        k = int(np.argmax(np.abs(vals))); inj.append((abs(float(vals[k])), int(sl[k,0]), int(sl[k,1])))
    return dict(observed=z + S, background=z, S=S, support=support, score_mask=~real_mask,
                real_mask=real_mask, lat=lat, lon=lon, global_scale=sg, thr=thr,
                peaks=peaks, peak_pos=peak_pos, pool=pool, injected=inj)

def acceptance(b, name, n_events):
    z, S, support, pool = b["background"], b["S"], b["support"], b["pool"]
    peaks = b["peaks"]; inj = np.array(b["injected"])
    L, node_of, npz = pole_merged_laplacian(b["lat"], b["lon"])
    tt = np.unique(np.argwhere(support)[:,0])[:40]
    q_evt = roughness(to_phys(S[tt], node_of, npz), L)
    q_bg  = roughness(to_phys(z[tt], node_of, npz), L)
    lsd = z.std(0)                                         # per-cell standard deviation over the full period, in global-scale units
    real_loc = np.array([peaks[i] / lsd[p[1], p[2]] for i, p in enumerate(b["peak_pos"])])
    pk = np.array([np.abs(v).max() for _, _, v, _ in pool])
    pc = [sl[int(np.argmax(np.abs(v)))] for _, sl, v, _ in pool]
    pool_loc = pk / np.array([lsd[c[0], c[1]] for c in pc])
    pk400 = np.array([np.abs(v).max() for _, _, v, t in pool if t])
    inj_loc = inj[:,0] / lsd[inj[:,1].astype(int), inj[:,2].astype(int)]
    p99, pmax = float(np.quantile(pk, .99)), float(pk.max())
    lr_pool, lr_inj = float(np.median(pool_loc)), float(np.median(inj_loc))
    return dict(product=name, T=z.shape[0], n_events=n_events, global_scale=round(b["global_scale"],4),
                n_real_events=len(peaks), n_pool=len(pool), pool_cells_p50=float(np.median([len(p[1]) for p in pool])),
                real_hill_20=round(hill(peaks, .20),2), pool_hill_20=round(hill(pk, .20),2),
                top400_hill_20=round(hill(pk400, .20),2),
                pool_p50=round(float(np.median(pk)),2), pool_p99=round(p99,2), pool_max=round(pmax,2),
                pool_capped="yes" if p99/pmax > 0.97 else "no",
                inj_amp_p50=round(float(np.median(inj[:,0])),2), inj_amp_max=round(float(inj[:,0].max()),2),
                inj_over_oldcap=int((inj[:,0] > OLD_CAP).sum()),
                local_ratio_real_p50=round(float(np.median(real_loc)),2),
                local_ratio_pool_p50=round(lr_pool,2), local_ratio_inj_p50=round(lr_inj,2),
                local_ratio_gap=round(abs(lr_inj/lr_pool - 1),3),
                support_density=float(support.mean()), excluded_frac=float(b["real_mask"].mean()),
                overlap_with_excluded=round(float((support & b["real_mask"]).sum() / support.sum()),4),
                Q_event=round(q_evt,5), Q_background=round(q_bg,5), ratio=round(q_evt/max(q_bg,1e-12),2),
                verdict="events rougher ✓" if q_evt > q_bg else "events smoother ✗")

if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "raw", "*.mon.mean.nc"))):
        nm = f.split("/")[-1].split(".")[0]
        if nm == "precip": continue          # the precipitation product is excluded (replaced by another product)
        b = make_benchmark(f, seed=0, n_events=60)
        r = acceptance(b, nm, 60); rows.append(r); del b
        print("%-5s pool %4d/%4d median cells %.0f  Hill20 all %.2f pool %.2f top400 %.2f  pool p99/max %.2f/%.2f capped %s  "
              "injected median %.2f max %.2f above old cap %d/60  local ratio real %.2f pool %.2f injected %.2f gap %.1f%%  "
              "overlap with exclusion zone %.1f%%  roughness %.4f/%.4f ratio %.2f  %s"
              % (r["product"], r["n_pool"], r["n_real_events"], r["pool_cells_p50"], r["real_hill_20"],
                 r["pool_hill_20"], r["top400_hill_20"], r["pool_p99"], r["pool_max"], r["pool_capped"],
                 r["inj_amp_p50"], r["inj_amp_max"], r["inj_over_oldcap"], r["local_ratio_real_p50"],
                 r["local_ratio_pool_p50"], r["local_ratio_inj_p50"], 100*r["local_ratio_gap"],
                 100*r["overlap_with_excluded"], r["Q_event"], r["Q_background"], r["ratio"], r["verdict"]), flush=True)
    out = os.path.join(OUT, "A1pp_honest_benchmark_spec.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(); print("Output:", out)
    print("(1)", "✓" if all(r["local_ratio_gap"] <= 0.15 for r in rows) else "✗", "local-ratio gap <= 15%")
    print("(2)", "✓" if all(r["pool_capped"] == "no" for r in rows) else "✗", "not capped")
    rng3 = {"air": (4.64, 7.45), "lftx": (4.92, 8.13), "slp": (6.28, 8.62)}
    print("(3)", "✓" if all(rng3[r["product"]][0] <= r["pool_hill_20"] <= rng3[r["product"]][1] for r in rows) else "✗",
          "pool Hill index lies between all events and the top 400")
    print("(4)", "✓" if all(r["Q_event"] > r["Q_background"] for r in rows) else "✗", "events rougher than background")
