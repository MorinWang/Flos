# -*- coding: utf-8 -*-
"""A1': honest benchmark v2.1 (native 2.5-degree grid, 73x144), with amplitudes on the same scale as the M3' tail analysis.
Acceptance checks (printed at the end of __main__):
  (1) injected amplitudes of the three products are no longer capped: peak-pool 99th percentile / maximum < 0.97,
      and the largest injected amplitude exceeds the old cap 8.72;
  (2) injected events are still rougher than the background; the ratio is > 1 for all three products;
  (3) the upper-20% Hill index of real peaks computed inside the generator matches M3' exactly
      (air 4.64, lftx 4.92, slp 6.28).
  Also measured, without a criterion: injected amplitude / local standard deviation at the injection site,
  compared with the same ratio for real events in place.
Differences from A1 (everything else is unchanged):
  1) load_anom only subtracts the per-cell monthly mean and no longer divides by the per-cell monthly standard
     deviation (that quantity has a mathematical upper bound of (n-1)/sqrt(n)); the whole field is divided by a
     global robust scale (median absolute deviation / 0.6745), so background and injections share one unit;
  2) the peak pool contains all real extreme connected components, the same population as M3' (A1 used only
     the largest 400); shape templates are still the largest 400;
  3) connected components are sorted by size with a stable sort (with A1's default sort the event roughness
     differed between two machines while the background matched exactly, presumably because ties were ordered
     differently; not verified).
make_benchmark() can be imported directly by the leaderboard scripts.
"""
import numpy as np, xarray as xr, glob, csv
from scipy import ndimage

QTHRESH   = 0.99     # quantile threshold for real extremes (same definition as M1-M3 and M3′)
POLE_TOL  = 1e-6
HALFWIDTH = {"air": 8.75, "lftx": 8.75, "precip": 3.75, "slp": 13.75}   # median measured in M1 (kept for reference only since A1)
DURATION  = 1        # median duration measured in M2 (months)
OLD_CAP   = 8.72     # mathematical upper bound of the old per-cell monthly z-score at monthly resolution

def load_anom(path):
    """Remove the seasonal mean only, then divide by the global robust scale. Returns (z, lat, lon, global scale). Same definition as deseason + tail in M3′."""
    d = xr.open_dataset(path); var = [k for k in d.data_vars][0]; da = d[var]
    lat = da.lat.values.astype(float); lon = da.lon.values.astype(float)
    x = da.values.astype(np.float64); months = da.time.dt.month.values; d.close()
    a = np.empty_like(x)
    for m in np.unique(months):
        sel = months == m
        if sel.sum() < 3: a[sel] = 0.0; continue
        a[sel] = x[sel] - np.nanmean(x[sel], 0)
    a = np.nan_to_num(a)
    aa = np.abs(a)
    sg = float(np.median(aa[aa > 0]) / 0.6745)
    return a / sg, lat, lon, sg

def pole_merged_laplacian(lat, lon):
    """Pole fix: the longitude cells on a pole latitude are merged into one physical node."""
    nlat, nlon = len(lat), len(lon)
    node_of = np.empty(nlat*nlon, dtype=int); coords = []
    for i in range(nlat):
        if abs(abs(lat[i]) - 90.0) < 1e-6 or abs(np.cos(np.radians(lat[i]))) < POLE_TOL:
            k = len(coords); coords.append((lat[i], 0.0)); node_of[i*nlon:(i+1)*nlon] = k
        else:
            for j in range(nlon):
                k = len(coords); coords.append((lat[i], lon[j])); node_of[i*nlon+j] = k
    n = len(coords); co = np.array(coords)
    la = np.radians(co[:,0]); lo = np.radians(co[:,1])
    # 4-neighbour connectivity only; weights are a Gaussian kernel of the great-circle distance
    W = np.zeros((n, n))
    def gc(a, b):
        dl = lo[b]-lo[a]
        h = np.sin((la[b]-la[a])/2)**2 + np.cos(la[a])*np.cos(la[b])*np.sin(dl/2)**2
        return 2*np.arctan2(np.sqrt(h), np.sqrt(max(1-h, 0.0)))
    ds = []
    edges = []
    for i in range(nlat):
        for j in range(nlon):
            a = node_of[i*nlon+j]
            for (ii, jj) in (((i+1), j), (i, (j+1) % nlon)):
                if ii >= nlat: continue
                b = node_of[ii*nlon+jj]
                if a == b: continue
                edges.append((a, b)); ds.append(gc(a, b))
    ds = np.array(ds); sc = np.median(ds)*1.5
    for (a, b), dd in zip(edges, ds):
        w = np.exp(-(dd/sc)**2); W[a,b] = w; W[b,a] = w
    L = np.diag(W.sum(1)) - W
    return L, node_of, n

def to_phys(field, node_of, n):
    """(T, nlat, nlon) -> (n_phys, T); pole cells are averaged."""
    T = field.shape[0]; flat = field.reshape(T, -1).T          # (cells, T)
    out = np.zeros((n, T)); cnt = np.zeros(n)
    np.add.at(out, node_of, flat); np.add.at(cnt, node_of, 1.0)
    return out / np.maximum(cnt, 1)[:, None]

def roughness(A, L):
    num = float(np.sum(A * (L @ A))); den = float(np.sum(A*A))
    return num / max(den, 1e-30)

def hill(peaks, frac):
    """Same formula as tail() in M3′."""
    p = np.sort(peaks)[::-1]
    k = min(max(10, int(frac * len(p))), len(p) - 1)
    return float(k / np.sum(np.log(p[:k] / p[k])))

def make_benchmark(path, seed=0, n_events=60):
    """Return one benchmark instance: observed = background + S; score_mask is the scoring mask that excludes the exclusion zone. Unit = global robust scale."""
    z, lat, lon, sg = load_anom(path)
    T, nlat, nlon = z.shape
    az = np.abs(z); thr = float(np.quantile(az[az>0], QTHRESH))
    real_mask = az > thr                                   # exclusion zone: real extremes already present in the background
    st = ndimage.generate_binary_structure(3, 1)
    lab, nlab = ndimage.label(real_mask, structure=st)
    ids = np.arange(1, nlab+1)
    peaks = np.asarray(ndimage.maximum(az, lab, ids))      # peaks of all real events = same population as M3′
    peak_pos = ndimage.maximum_position(az, lab, ids)
    sizes = np.asarray(ndimage.sum(real_mask, lab, ids))
    good = np.argsort(sizes, kind="stable")[::-1][:400] + 1   # shape-template library: the largest components
    objs = ndimage.find_objects(lab)
    templates = []
    for i in good:
        slc = objs[i-1]
        idx = np.argwhere(lab[slc] == i) + np.array([s.start for s in slc])
        t0 = int(np.bincount(idx[:,0]).argmax())
        sl = idx[idx[:,0]==t0][:,1:]
        if len(sl) < 3: continue
        vals = az[t0, sl[:,0], sl[:,1]]
        sl = sl - sl.mean(0).astype(int)                   # shift to the origin
        templates.append((sl, vals/vals.max()))
    rng = np.random.default_rng(seed)
    S = np.zeros_like(z); support = np.zeros(z.shape, dtype=bool); inj = []
    for _ in range(n_events):
        sl, prof = templates[rng.integers(len(templates))]
        amp = float(rng.choice(peaks)) * rng.choice((-1.0, 1.0))
        t0 = int(rng.integers(0, T - DURATION + 1))
        ci = int(rng.integers(4, nlat-4)); cj = int(rng.integers(0, nlon))
        yy = np.clip(sl[:,0] + ci, 0, nlat-1); xx = (sl[:,1] + cj) % nlon
        for dt in range(DURATION):
            S[t0+dt, yy, xx] += amp * prof
            support[t0+dt, yy, xx] = True
        k = int(np.argmax(prof)); inj.append((abs(amp), int(yy[k]), int(xx[k])))
    return dict(observed=z + S, background=z, S=S, support=support, score_mask=~real_mask,
                real_mask=real_mask, lat=lat, lon=lon, global_scale=sg, thr=thr,
                peaks=peaks, peak_pos=peak_pos, top400_peaks=peaks[good-1],
                templates=len(templates), injected=inj)

def acceptance(b, name, n_events):
    z, S, support = b["background"], b["S"], b["support"]
    peaks = b["peaks"]; inj = np.array(b["injected"])
    # roughness of injected events vs background (as in A1: the first 40 time steps that contain events)
    L, node_of, npz = pole_merged_laplacian(b["lat"], b["lon"])
    tt = np.unique(np.argwhere(support)[:,0])[:40]
    q_evt = roughness(to_phys(S[tt], node_of, npz), L)
    q_bg  = roughness(to_phys(z[tt], node_of, npz), L)
    # also measured: amplitude / local standard deviation (per-cell standard deviation over the full period, in global-scale units)
    lsd = z.std(0)
    real_loc = np.array([peaks[i] / lsd[p[1], p[2]] for i, p in enumerate(b["peak_pos"])])
    inj_loc  = inj[:,0] / lsd[inj[:,1].astype(int), inj[:,2].astype(int)]
    pool_p99 = float(np.quantile(peaks, .99)); pool_max = float(peaks.max())
    return dict(product=name, T=z.shape[0], grid="%dx%d" % z.shape[1:], n_events=n_events,
                global_scale=round(b["global_scale"],4), thr=round(b["thr"],3),
                templates=b["templates"], n_real_events=len(peaks),
                real_hill_20=round(hill(peaks, .20),2), top400_hill_20=round(hill(b["top400_peaks"], .20),2),
                pool_p50=round(float(np.median(peaks)),2), top400_p50=round(float(np.median(b["top400_peaks"])),2),
                pool_p99=round(pool_p99,2), pool_max=round(pool_max,2),
                pool_capped="yes" if pool_p99/pool_max > 0.97 else "no",
                inj_amp_p50=round(float(np.median(inj[:,0])),2), inj_amp_max=round(float(inj[:,0].max()),2),
                inj_over_oldcap=int((inj[:,0] > OLD_CAP).sum()),
                support_density=float(support.mean()), excluded_frac=float(b["real_mask"].mean()),
                Q_event=round(q_evt,5), Q_background=round(q_bg,5), ratio=round(q_evt/max(q_bg,1e-12),2),
                local_ratio_real_p50=round(float(np.median(real_loc)),2),
                local_ratio_inj_p50=round(float(np.median(inj_loc)),2),
                verdict="events rougher ✓" if q_evt > q_bg else "events smoother ✗")

if __name__ == "__main__":
    import os
    from paths import DATA, OUT
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "raw", "*.mon.mean.nc"))):
        nm = f.split("/")[-1].split(".")[0]
        if nm == "precip": continue          # the precipitation product is excluded (replaced by another product)
        b = make_benchmark(f, seed=0, n_events=60)
        r = acceptance(b, nm, 60); rows.append(r); del b
        print("%-5s scale %.3f real events %5d  Hill20 real %.2f / top400 %.2f  pool p99/max %.2f/%.2f capped %s  "
              "injected median %.2f max %.2f above old cap %d/60  roughness %.4f/%.4f ratio %.2f  local ratio real %.2f injected %.2f  %s"
              % (r["product"], r["global_scale"], r["n_real_events"], r["real_hill_20"], r["top400_hill_20"],
                 r["pool_p99"], r["pool_max"], r["pool_capped"], r["inj_amp_p50"], r["inj_amp_max"],
                 r["inj_over_oldcap"], r["Q_event"], r["Q_background"], r["ratio"],
                 r["local_ratio_real_p50"], r["local_ratio_inj_p50"], r["verdict"]), flush=True)
    out = os.path.join(OUT, "A1p_honest_benchmark_spec.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(); print("Output:", out)
    print("Check (1):", "not capped ✓" if all(r["pool_capped"]=="no" and r["inj_amp_max"]>OLD_CAP for r in rows) else "✗")
    print("Check (2):", "all passed ✓" if all(r["Q_event"]>r["Q_background"] for r in rows) else "some failed")
    print("Reference (3): M3′ upper-20% Hill index: air 4.64, lftx 4.92, slp 6.28")
