# -*- coding: utf-8 -*-
"""Line-by-line port of the official MATLAB code of TV-RPCA (Cao, Yang, Guo, IEEE TCyb 46(4), 2016).
Official code: https://yangliang.github.io/code/TVRPCA.rar
  tv_alm_rpca3.m      → run_tvrpca
  private/deconvtvl2.m → deconvtvl2 (Stanley Chan's 3-D total variation, frequency-domain ADMM)
Parameters follow the official demo.m and paper p.1020: λ = [0.4, 2, 1] × 1/√(height·width) (λ3 = 1 for dynamic backgrounds);
fixed 20 outer iterations (the tolerance test in the official code is commented out, so it always runs maxIter = 20);
μ0 = 1.25/‖O‖₂, ρ = 1.5.
Video layout: one frame = the global field of one month, height = latitude, width = longitude, frames = months.
The total-variation boundary is periodic (official FFT implementation): correct in longitude, but in latitude it
joins the southern and northern edges. We keep the official code unchanged and note this as a known difference.
Foreground = F_hat (the official demo saves F_hat), event score = |F_hat|; background = B_hat.
The only deviation: the official code uses PROPACK's lansvd for a partial SVD; here a full economy SVD is used,
which yields the same truncation mathematically.
"""
import numpy as np
from scipy import linalg

def _soft(x, t):
    return np.maximum(x - t, 0) + np.minimum(x + t, 0)

def _D(U, beta):
    Dux = beta[0]*np.concatenate([np.diff(U, axis=1), U[:, :1] - U[:, -1:]], axis=1)
    Duy = beta[1]*np.concatenate([np.diff(U, axis=0), U[:1] - U[-1:]], axis=0)
    Duz = beta[2]*np.concatenate([np.diff(U, axis=2), U[:, :, :1] - U[:, :, -1:]], axis=2)
    return Dux, Duy, Duz

def _Dt(X, Y, Z, beta):
    out = beta[0]*np.concatenate([X[:, -1:] - X[:, :1], -np.diff(X, axis=1)], axis=1)
    out = out + beta[1]*np.concatenate([Y[-1:] - Y[:1], -np.diff(Y, axis=0)], axis=0)
    out = out + beta[2]*np.concatenate([Z[:, :, -1:] - Z[:, :, :1], -np.diff(Z, axis=2)], axis=2)
    return out

def _eig_kernel(shape, idx2):
    k = np.zeros(shape); k[0, 0, 0] = 1.0; k[idx2] = -1.0
    return np.abs(np.fft.fftn(k))**2

def deconvtvl2(g, mu, beta=(1, 1, 1), rho_r=2.0, gamma=2.0, max_itr=20, tol=1e-3, alpha=0.7):
    """min mu/2 ||f - g||² + ||f||_TV (H = 1), official default parameters."""
    sh = g.shape
    eigDtD = beta[0]**2*_eig_kernel(sh, (0, 1, 0)) + beta[1]**2*_eig_kernel(sh, (1, 0, 0))
    eigEtE = beta[2]**2*_eig_kernel(sh, (0, 0, 1)) if sh[2] > 1 else 0.0
    rho = rho_r; f = g.copy()
    y1 = np.zeros(sh); y2 = np.zeros(sh); y3 = np.zeros(sh)
    u1 = np.zeros(sh); u2 = np.zeros(sh); u3 = np.zeros(sh)
    Df1, Df2, Df3 = _D(f, beta)
    rnorm = np.sqrt(np.sum(Df1**2) + np.sum(Df2**2) + np.sum(Df3**2))
    for itr in range(1, max_itr + 1):
        f_old = f
        rhs = np.fft.fftn((mu/rho)*g + _Dt(u1 - y1/rho, u2 - y2/rho, u3 - y3/rho, beta))
        eigA = (mu/rho) + eigDtD + eigEtE
        f = np.real(np.fft.ifftn(rhs/eigA))
        Df1, Df2, Df3 = _D(f, beta)
        v1 = Df1 + y1/rho; v2 = Df2 + y2/rho; v3 = Df3 + y3/rho
        v = np.sqrt(v1**2 + v2**2 + v3**2); v[v == 0] = 1
        v = np.maximum(v - 1/rho, 0)/v
        u1 = v1*v; u2 = v2*v; u3 = v3*v
        y1 = y1 - rho*(u1 - Df1); y2 = y2 - rho*(u2 - Df2); y3 = y3 - rho*(u3 - Df3)
        rnorm_old = rnorm
        rnorm = np.sqrt(np.sum((Df1 - u1)**2) + np.sum((Df2 - u2)**2) + np.sum((Df3 - u3)**2))
        if rnorm > alpha*rnorm_old: rho = rho*gamma
        relchg = np.linalg.norm(f - f_old)/np.linalg.norm(f_old)
        if relchg < tol: break
    return f

def run_tvrpca(O3, lam=(0.4, 2.0, 1.0), max_iter=20, return_all=False):
    """O3: (T, height, width). Returns (B, F, info); B and F have the same shape as O3."""
    T, h, w = O3.shape; m = h*w
    O = O3.reshape(T, m).T.copy()                      # columns = frames
    to_video = lambda M: M.T.reshape(T, h, w).transpose(1, 2, 0)
    to_mat = lambda V: V.transpose(2, 0, 1).reshape(T, m).T
    l_ = 1/np.sqrt(h*w); l1, l2, l3 = l_*lam[0], l_*lam[1], l_*lam[2]
    norm_two = linalg.svdvals(O, check_finite=False)[0]
    X = O/max(norm_two, np.abs(O).max()/l1)
    Y = np.zeros_like(O); B = np.zeros_like(O); F = np.zeros_like(O); E = np.zeros_like(O)
    mu = 1.25/norm_two; mu_bar = mu*1e7; rho = 1.5; svp = 0
    for it in range(1, max_iter + 1):
        tS = ((mu*O + X - mu*B) + (mu*E + mu*F - Y))/(2*mu)
        M = _soft(tS, l1/(2*mu))
        u, s, vt = linalg.svd(O - M + X/mu, full_matrices=False, check_finite=False)
        svp = int((s > 1/mu).sum())
        B = (u[:, :svp]*(s[:svp] - 1/mu)) @ vt[:svp]
        E = _soft(M - F + Y/mu, l2/mu)
        g = M + Y/mu - E
        if np.all(g.max(0) != 0):                      # MATLAB: if max(g_source) ~= 0 (all column maxima nonzero)
            F = to_mat(deconvtvl2(to_video(g), mu/l3, beta=(1, 1, 1)))
        X = X + mu*(O - B - M)
        Y = Y + mu*(M - E - F)
        mu = min(mu*rho, mu_bar)
    back = lambda M_: M_.T.reshape(T, h, w)
    info = dict(rank=svp, iters=it, converged="official setting, fixed 20 rounds")
    if return_all:                                      # used by the TV-RPCA-M row in A5: also return the disturbance E and M = E + F
        info["E"] = back(E); info["M"] = back(M)
    return back(B), back(F), info
