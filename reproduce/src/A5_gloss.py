# -*- coding: utf-8 -*-
"""Port of GLOSS (Sofuoglu & Aviyente, Signal Processing 192:108370, 2022; arXiv 2010.02889).
Algorithm: line-by-line port of lib/functions/gloss.m (function gloss_3) from the first author's repository
      https://github.com/mrsfgl/tenandet @2505470cc91a, together with get_graphL.m, soft_hosvd.m and graph_reg_update2.m.
Parameters: the rules of paper §5.2 p.16–17 (we follow the authors' published parameter rules; the first author's
      set_params.m and their NYC experiment each use a different setting, and the three are mutually inconsistent):
      λ = γ = 1/‖P_Ω(Y)‖₀ (fully observed = 1/number of nonzeros); β1..β5 = 1/(5·std(vec Y));
      ψ_n = p / Tr(√Σ_Y(n)), with p chosen so that min ψ_n = 1; θ = (∏ψ_n)^(1/4); α = 0 (code default).
      The paper gives no iteration count or tolerance, so we use the first author's set_params.m: max_iter = 100,
      err_tol = 0.01. The graph uses the 5 nearest neighbours hard-coded in the code.
Tensor layout (our choice; the paper uses urban traffic "hour×weekday×week×location", and the code hard-codes 4 modes
      with temporal differencing along mode 1):
      month(12) × year × lat(73) × lon(144); only complete years are used; trailing months of an incomplete year
      are excluded from GLOSS and get S = 0.
Event score = |S|.
"""
import numpy as np
from scipy import linalg

def _unfold(T, n):
    return np.moveaxis(T, n, 0).reshape(T.shape[n], -1)

def _fold(M, n, shape):
    rest = [s for i, s in enumerate(shape) if i != n]
    return np.moveaxis(M.reshape([shape[n]] + rest), 0, n)

def _soft(x, t):
    return np.sign(x)*np.maximum(np.abs(x) - t, 0)

def get_graphL(Y, n, K=5):
    """get_graphL(permute(Y,[mods,i]),5): each index of mode n is a node whose feature is the full slice at that index; binary K-nearest-neighbour graph, symmetrised."""
    X = _unfold(Y, n)                                   # (S, features)
    S = X.shape[0]; sq = np.sum(X*X, 1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2*X @ X.T, 0)
    D = D + np.diag(D.sum(0))                           # D = D + diag(sum(D)): excludes each node from its own neighbours
    kth = np.sort(D, axis=0)[K - 1]                     # max(mink(D,K)): K-th smallest value per column
    W = (D <= kth[None, :]) & (D > 0)                   # D(D>sdW)=0; W = D; every nonzero entry is an edge
    W = W | W.T
    W = W.astype(float)
    return np.diag(W.sum(1)) - W

def run_gloss(Y, max_iter=100, err_tol=0.01, K=5, lam=None, gam=None, psi=None, theta=None):
    """Y: 4th-order tensor. Returns (L, S, info)."""
    N = Y.ndim; sz = Y.shape
    beta = 1.0/(5*np.std(Y))
    rule = 1.0/np.count_nonzero(Y)                     # paper p.16: λ = γ = 1/‖P_Ω(Y)‖₀
    lam = rule if lam is None else lam; gam = rule if gam is None else gam
    tr = []
    for n in range(N):
        ev = np.linalg.eigvalsh(np.cov(_unfold(Y, n)))
        tr.append(np.sum(np.sqrt(np.maximum(ev, 0))))
    tr = np.array(tr); psi_rule = tr.max()/tr           # ψ_n = p/Tr(√Σ), min ψ = 1
    psi = psi_rule if psi is None else np.asarray(psi, float)
    theta = float(np.prod(psi_rule)**(1.0/N)) if theta is None else float(theta)
    b1 = b2 = b3 = b4 = b5 = beta; alpha = 0.0
    inv_Phi = [np.linalg.inv((theta/b3)*get_graphL(Y, n, K) + np.eye(sz[n])) for n in range(N)]
    I1 = sz[0]
    Dm = np.eye(I1) - np.roll(np.eye(I1), 1, axis=1)     # cyclic forward difference: (Dw)_i = w_i − w_{i+1}
    invD = np.linalg.inv(b5*np.eye(I1) + b4*(Dm.T @ Dm))
    Lx = [np.zeros(sz) for _ in range(N)]; G = [np.zeros(sz) for _ in range(N)]
    Lam2 = [np.zeros(sz) for _ in range(N)]; Lam3 = [np.zeros(sz) for _ in range(N)]
    S = np.zeros(sz); L = np.zeros(sz); Nt = np.zeros(sz); W = np.zeros(sz); Z = np.zeros(sz)
    Lam1 = np.zeros(sz); Lam4 = np.zeros(sz); Lam5 = np.zeros(sz)
    nY = np.linalg.norm(Y); it = 0; conv = False
    DW = lambda Wt: _fold(Dm @ _unfold(Wt, 0), 0, sz)
    while True:
        it += 1
        temp = np.zeros(sz)
        for i in range(N):
            temp += b2*(Lx[i] - Lam2[i]) + b3*(G[i] + Lam3[i])
        L = (b1*(Y - S + Lam1) + temp)/(b1 + N*(b2 + b3))
        for i in range(N):                              # soft_hosvd(L, Lam2, psi, 1/β2)
            u, s, vt = linalg.svd(_unfold(L + Lam2[i], i), full_matrices=False, check_finite=False)
            s = s - psi[i]/b2; k = int((s > 0).sum())
            Lx[i] = _fold((u[:, :k]*s[:k]) @ vt[:k], i, sz)
        for i in range(N):                              # graph_reg_update2
            G[i] = _fold(inv_Phi[i] @ _unfold(L - Lam3[i], i), i, sz)
        Sold = S
        S = _soft(b1*(Y - L - Nt + Lam1) + b5*(W + Lam5), lam)/(b1 + b5)
        Nt = (b1/(b1 + alpha))*(Y + Lam1 - L - S)
        W = _fold(invD @ (b5*_unfold(S - Lam5, 0) + b4*Dm.T @ _unfold(Z + Lam4, 0)), 0, sz)
        Z = _soft(DW(W) - Lam4, gam/(b4 + np.finfo(float).eps))
        tmp = 0.0
        for i in range(N):
            up = L - Lx[i]; tmp += np.sum(up*up); Lam2[i] = Lam2[i] + up
            Lam3[i] = Lam3[i] + (G[i] - L)
        Lam1 = Lam1 + Y - L - S - Nt
        tmp = np.sqrt(tmp)/(np.sqrt(N)*nY)
        Lam4 = Lam4 - DW(W) + Z
        Lam5 = Lam5 - S + W
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.linalg.norm(S - Sold)/np.linalg.norm(Sold)
        err = np.nanmax([rel, tmp])                     # MATLAB max ignores NaN
        if err <= err_tol: conv = True; break
        if it >= max_iter: break                        # MATLAB increments iter before testing iter > max_iter, i.e. it runs exactly max_iter iterations
    return L, S, dict(rank="", iters=it, converged=conv, lam=lam, beta=beta, theta=theta, psi=psi.round(3).tolist())

def run_gloss_field(O3):
    """O3: (T, nlat, nlon), starting in January. Arranged as month×year×lat×lon using complete years only. Returns (L3, S3, info) with the same shape as O3."""
    T, h, w = O3.shape; ny = T//12; Tn = ny*12
    Y = O3[:Tn].reshape(ny, 12, h, w).transpose(1, 0, 2, 3).copy()
    L, S, info = run_gloss(Y)
    back = lambda A: A.transpose(1, 0, 2, 3).reshape(Tn, h, w)
    L3 = O3.copy(); S3 = np.zeros_like(O3)
    L3[:Tn] = back(L); S3[:Tn] = back(S)
    info["months_dropped"] = T - Tn
    return L3, S3, info

def ee_score_field(S3, n_jobs=2):
    """The GLOSS paper's own score GLOSS-EE (first author's mahal_dist.m): robust Mahalanobis distance along each fibre.
    In the NYC data the fibres run along "week"; here the counterpart runs along "year": for fixed (month, lat, lon),
    MinCovDet (in place of MATLAB robustcov) is fitted to S across years.
    Returns scores with the same shape as S3; trailing months of an incomplete year get 0."""
    from sklearn.covariance import MinCovDet
    from joblib import Parallel, delayed
    T, h, w = S3.shape; ny = T//12; Tn = ny*12
    S4 = S3[:Tn].reshape(ny, 12, h, w)                        # (year, month, lat, lon)
    fib = S4.transpose(1, 2, 3, 0).reshape(-1, ny)             # each row is one "year" fibre
    def one(v):
        if not np.any(v != 0) or np.all(v == v[0]): return np.zeros(ny)
        x = v.reshape(-1, 1)
        try: d = MinCovDet(random_state=123).fit(x).mahalanobis(x)
        except Exception: d = np.zeros(ny)
        if not np.any(d != 0) or not np.all(np.isfinite(d)):
            x = x + np.sqrt(0.01)*np.random.default_rng(123).standard_normal(x.shape)
            d = MinCovDet(random_state=123).fit(x).mahalanobis(x)
        return np.sqrt(np.maximum(d, 0))
    res = np.array(Parallel(n_jobs=n_jobs, batch_size=512)(delayed(one)(v) for v in fib))
    out = np.zeros_like(S3)
    out[:Tn] = res.reshape(12, h, w, ny).transpose(3, 0, 1, 2).reshape(Tn, h, w)
    return np.nan_to_num(out)
