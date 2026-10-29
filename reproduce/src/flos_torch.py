#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch backend for FLoS. Device-agnostic: the same code runs on CPU and CUDA.
The method, hyperparameters, benchmark and scoring are unchanged; only two hot spots are moved to torch:
   (1) the two dense matrix products of the sphere filter  (2) the randomized SVD
The randomized SVD mirrors sklearn.utils.extmath.randomized_svd line by line (n_iter=1, normalizer=none,
random_state=0, n_oversamples=10); the random projection is still drawn with numpy RandomState so that the
random stream is identical.
Note: svd_flip only flips signs and (u*s)@vt is invariant to them, so it is omitted.
"""
import numpy as np, torch

def _median(t):
    """Same definition as numpy.median: for even length, average the two middle values (torch.median returns the lower one)."""
    f = t.reshape(-1)
    n = f.numel()
    s, _ = torch.sort(f)
    if n % 2: return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) * 0.5

def randomized_low_rank(M, rank, n_iter=1, random_state=0, n_oversamples=10):
    """Mirror sklearn's randomized_svd, then reconstruct; returns the low-rank approximation."""
    rank = int(max(1, min(rank, min(M.shape) - 1)))
    n_random = rank + n_oversamples
    ns, nf = M.shape
    transpose = ns < nf
    A = M.T if transpose else M
    rs = np.random.RandomState(random_state)
    Q = torch.as_tensor(rs.normal(size=(A.shape[1], n_random)), dtype=A.dtype, device=A.device)
    for _ in range(n_iter):                 # normalizer = none
        Q = A @ Q
        Q = A.T @ Q
    Q, _ = torch.linalg.qr(A @ Q, mode="reduced")
    B = Q.T @ A
    Uh, s, Vt = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uh
    if transpose:
        u, sv, vt = Vt[:rank].T, s[:rank], U[:, :rank].T
    else:
        u, sv, vt = U[:, :rank], s[:rank], Vt[:rank]
    return (u * sv) @ vt

def adaptive_lorentz_shrink(x, lam, p=1.25, eps=1e-6):
    """Step-by-step match of the numpy version: scale = MAD of |x|, threshold lam/(|x|/scale+eps)^(p-1)."""
    ax = torch.abs(x)
    med = _median(ax)
    mad = _median(torch.abs(ax - med)) / 0.6745
    scale = mad if float(mad) > 1e-8 else torch.std(ax)
    scale = float(scale) if float(scale) > 1e-8 else 1.0
    adaptive = lam / torch.pow(ax / scale + eps, max(p - 1.0, 1e-3))
    return torch.sign(x) * torch.clamp(ax - adaptive, min=0.0)

class SphereFilterT:
    """F(Y) = U diag(1/(1+α μ^s)) Uᵀ (A Y), then broadcast back to the original grid cells via node_of."""
    def __init__(self, U, inv, node_of, device, dtype=torch.float64, out_dtype=None):
        self.dtype = dtype
        self.out_dtype = out_dtype or dtype
        self.U = torch.as_tensor(U, dtype=dtype, device=device)
        self.inv = torch.as_tensor(inv, dtype=dtype, device=device)[:, None]
        self.node = torch.as_tensor(np.asarray(node_of), dtype=torch.long, device=device)
        n = int(self.node.max()) + 1
        cnt = torch.bincount(self.node, minlength=n).to(dtype)
        self.w = (1.0 / cnt)[:, None]
        self.n = n
    def __call__(self, Y):
        # A @ Y: average the duplicated pole cells per node
        Yl = Y.to(self.dtype)
        acc = torch.zeros(self.n, Yl.shape[1], dtype=self.dtype, device=Yl.device)
        acc.index_add_(0, self.node, Yl)
        C = self.U.T @ (acc * self.w)
        return (self.U @ (C * self.inv))[self.node].to(self.out_dtype)

def run_flos(X, filt, lam=0.18, p=1.25, max_iter=500, tol=1e-5, paper_iters=10, rank=None):
    """Same structure as A5_leaderboard.run_flos; rank is passed in by the caller (the MP estimate still uses the numpy version and is computed once)."""
    S = torch.zeros_like(X); L = torch.zeros_like(X)
    nx = float(torch.linalg.norm(X)); conv = False; snap = None; d = float("nan")
    for it in range(1, max_iter + 1):
        Ln = randomized_low_rank(filt(X - S), rank)
        Sn = adaptive_lorentz_shrink(X - Ln, lam, p=p)
        d = float(max(torch.linalg.norm(Ln - L), torch.linalg.norm(Sn - S)) / nx)
        L, S = Ln, Sn
        if it == paper_iters: snap = (L.clone(), S.clone(), d)
        if d < tol: conv = True; break
    return L, S, dict(rank=rank, iters=it, converged=conv, last_change=d, snap=snap)

# ---------------- Replacement components for ablations
def soft_threshold_fixed(x, lam):
    """Ablation 1: plain soft thresholding with a fixed threshold, removing the magnitude adaptivity."""
    ax = torch.abs(x)
    return torch.sign(x) * torch.clamp(ax - lam, min=0.0)

class IdentityFilter:
    """Ablation 2: remove the spherical smoothing; the background step is a plain low-rank truncation."""
    def __call__(self, Y): return Y

def run_flos_ablate(X, filt, lam=0.18, p=1.25, max_iter=500, tol=1e-5,
                    paper_iters=10, rank=None, ablation=None):
    """Same structure as run_flos; ablation is one of None / fixed-thresh / no-smooth.
    The ablation that fixes the filter exponent to one is not implemented here; the caller builds the filter with s=1.0."""
    if ablation == "no-smooth": filt = IdentityFilter()
    S = torch.zeros_like(X); L = torch.zeros_like(X)
    nx = float(torch.linalg.norm(X)); conv = False; snap = None; d = float("nan")
    for it in range(1, max_iter + 1):
        Ln = randomized_low_rank(filt(X - S), rank)
        R = X - Ln
        if ablation == "fixed-thresh":
            Sn = soft_threshold_fixed(R, lam)
        else:
            Sn = adaptive_lorentz_shrink(R, lam, p=p)
        d = float(max(torch.linalg.norm(Ln - L), torch.linalg.norm(Sn - S)) / nx)
        L, S = Ln, Sn
        if it == paper_iters: snap = (L.clone(), S.clone(), d)
        if d < tol: conv = True; break
    return L, S, dict(rank=rank, iters=it, converged=conv, last_change=d, snap=snap)

# ---------------- A11: three arms with different scale normalisations
def _median_rows(t):
    """numpy-style median along dim=1: for even length, average the two middle values (same definition as _median)."""
    n = t.shape[1]
    s, _ = torch.sort(t, dim=1)
    if n % 2: return s[:, n // 2:n // 2 + 1]
    return (s[:, n // 2 - 1:n // 2] + s[:, n // 2:n // 2 + 1]) * 0.5

def _scale_rows(x):
    """Per-cell robust scale σ_i along time, defined exactly as in adaptive_lorentz_shrink (MAD of |x|)."""
    ax = torch.abs(x)
    med = _median_rows(ax)
    mad = _median_rows(torch.abs(ax - med)) / 0.6745
    sd = ax.std(dim=1, keepdim=True)
    sc = torch.where(mad > 1e-8, mad, sd)
    return torch.where(sc > 1e-8, sc, torch.ones_like(sc))

def _scale_global(x):
    """Global robust scale σ: identical to the corresponding three lines in adaptive_lorentz_shrink."""
    ax = torch.abs(x)
    med = _median(ax)
    mad = _median(torch.abs(ax - med)) / 0.6745
    scale = mad if float(mad) > 1e-8 else torch.std(ax)
    return float(scale) if float(scale) > 1e-8 else 1.0

def shrink_pix_fixed(x, lam):
    """Arm C: per-cell fixed threshold τ_i = λ·σ_i."""
    sig = _scale_rows(x)
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam * sig, min=0.0)

def shrink_pix_adapt(x, lam, p=1.25, eps=1e-6):
    """Arm D: per-cell adaptive, scale-equivariant threshold τ_it = λ·σ_i^(2−q)·|x_it|^(q−1), q−1 = 1−p.
    Its threshold equals that of arm C exactly at |x| = σ_i, so the two arms differ only in magnitude adaptivity."""
    sig = _scale_rows(x); ax = torch.abs(x)
    e = max(p - 1.0, 1e-3)                       # = 1 − q
    tau = lam * torch.pow(sig, 1.0 + e) / torch.pow(ax + eps * sig, e)
    return torch.sign(x) * torch.clamp(ax - tau, min=0.0)

def shrink_glob_equiv(x, lam, p=1.25, eps=1e-6):
    """Arm E: global σ, but with the scale-equivariant threshold τ = λ·σ^(2−q)·|x|^(q−1)."""
    sig = _scale_global(x); ax = torch.abs(x)
    e = max(p - 1.0, 1e-3)
    tau = lam * (sig ** (1.0 + e)) / torch.pow(ax + eps * sig, e)
    return torch.sign(x) * torch.clamp(ax - tau, min=0.0)

_SHRINK = {"pix-fixed": shrink_pix_fixed, "pix-adapt": shrink_pix_adapt, "glob-equiv": shrink_glob_equiv}

def run_flos_shrink(X, filt, lam=0.18, p=1.25, max_iter=500, tol=1e-5,
                    paper_iters=10, rank=None, shrink="pix-adapt"):
    """Same structure as run_flos; only the shrinkage of the event step is replaced, the background step is unchanged."""
    fn = _SHRINK[shrink]
    S = torch.zeros_like(X); L = torch.zeros_like(X)
    nx = float(torch.linalg.norm(X)); conv = False; snap = None; d = float("nan")
    for it in range(1, max_iter + 1):
        Ln = randomized_low_rank(filt(X - S), rank)
        Sn = fn(X - Ln, lam) if shrink == "pix-fixed" else fn(X - Ln, lam, p=p)
        d = float(max(torch.linalg.norm(Ln - L), torch.linalg.norm(Sn - S)) / nx)
        L, S = Ln, Sn
        if it == paper_iters: snap = (L.clone(), S.clone(), d)
        if d < tol: conv = True; break
    return L, S, dict(rank=rank, iters=it, converged=conv, last_change=d, snap=snap)

# ---------------- A14: two shrinkage rules that use "external" information
# The order-preservation theorem covers thresholds determined only by the value being shrunk. To escape it,
# the threshold must carry information from outside that point.
def run_flos_reweight(X, filt, lam=0.18, p=1.25, eps_w=0.1, max_iter=500, tol=1e-5,
                      paper_iters=10, rank=None):
    """Reweighted variant: weights come from the previous iteration's S, not from the current residual itself
    (the Candès–Wakin–Boyd form). In the first iteration all weights are one (a plain fixed threshold), which
    avoids the deadlock of S staying identically zero."""
    S = torch.zeros_like(X); L = torch.zeros_like(X)
    nx = float(torch.linalg.norm(X)); conv = False; snap = None; d = float("nan")
    e = max(p - 1.0, 1e-3)
    for it in range(1, max_iter + 1):
        Ln = randomized_low_rank(filt(X - S), rank)
        R = X - Ln
        if it == 1:
            Sn = soft_threshold_fixed(R, lam)
        else:
            sig = _scale_global(R)
            w = torch.pow(torch.abs(S) / sig + eps_w, -e)       # S is still from the previous iteration
            Sn = torch.sign(R) * torch.clamp(torch.abs(R) - lam * w, min=0.0)
        d = float(max(torch.linalg.norm(Ln - L), torch.linalg.norm(Sn - S)) / nx)
        L, S = Ln, Sn
        if it == paper_iters: snap = (L.clone(), S.clone(), d)
        if d < tol: conv = True; break
    return L, S, dict(rank=rank, iters=it, converged=conv, last_change=d, snap=snap)

def run_flos_rough(X, filt, rough, lam=0.18, gamma=0.25, eps_r=0.1, max_iter=500, tol=1e-5,
                   paper_iters=10, rank=None):
    """Roughness-weighted variant: the threshold comes from the spatial roughness at the point, not from its own magnitude.
    Roughness = |graph Laplacian applied to the residual|, normalised by its median. Rougher ⟹ more event-like ⟹
    smaller threshold, lighter shrinkage. This is the mechanism described in the paper: large-scale, low-roughness
    structure goes to the background; local, high-roughness structure goes to the events."""
    S = torch.zeros_like(X); L = torch.zeros_like(X)
    nx = float(torch.linalg.norm(X)); conv = False; snap = None; d = float("nan")
    for it in range(1, max_iter + 1):
        Ln = randomized_low_rank(filt(X - S), rank)
        R = X - Ln
        q = torch.abs(rough(R))
        med = _median(q); med = med if float(med) > 1e-12 else torch.ones_like(med)
        tau = lam * torch.pow(q / med + eps_r, -gamma)
        Sn = torch.sign(R) * torch.clamp(torch.abs(R) - tau, min=0.0)
        d = float(max(torch.linalg.norm(Ln - L), torch.linalg.norm(Sn - S)) / nx)
        L, S = Ln, Sn
        if it == paper_iters: snap = (L.clone(), S.clone(), d)
        if d < tol: conv = True; break
    return L, S, dict(rank=rank, iters=it, converged=conv, last_change=d, snap=snap)
