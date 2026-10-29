"""FLoS (Fractional Laplacian on a Spherical graph) and its convex core FLoS-C."""
import numpy as np
import torch

from .ops import FractionalFilter, adaptive_shrink, estimate_rank, randomized_truncation, soft_threshold


def _as_tensor(X, device, dtype):
    return X.to(device=device, dtype=dtype) if torch.is_tensor(X) else torch.as_tensor(np.asarray(X), dtype=dtype, device=device)


class FLoS:
    """Decompose a field X (cells x time) into a smooth background L and sparse events S, X = L + S + noise.

    Each round filters the background with a fractional power of the graph Laplacian, truncates it to a rank read
    from the data, and shrinks the residual into events:

        L_k = T_r(F(X - S_{k-1})),   S_k = psi_lambda(X - L_k).

    The detection score of entry (i, t) is |S_it|. Defaults are the settings used throughout the paper.

    Parameters
    ----------
    graph     : flos.Graph of the mesh.
    alpha, s  : strength and exponent of the smoothness prior, gain 1 / (1 + alpha mu^s).
    lam, p    : threshold and exponent of the event shrinkage (p = 1 is soft thresholding).
    rank      : truncation rank; None estimates it once from X; 0 disables the truncation.
    max_iter, tol : stop when the relative change of (L, S) falls below tol, or after max_iter rounds.
    """

    def __init__(self, graph, alpha=0.85, s=0.25, lam=0.72, p=1.25, rank=None, max_iter=500, tol=1e-5,
                 device=None, dtype=torch.float32):
        self.graph, self.alpha, self.s, self.lam, self.p = graph, alpha, s, lam, p
        self.rank, self.max_iter, self.tol, self.dtype = rank, max_iter, tol, dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.filter = FractionalFilter(graph, alpha, s, device=self.device, dtype=dtype)

    def fit(self, X, callback=None):
        """Run the decomposition. `callback(k, L, S)` is called after every round (e.g. to record snapshots)."""
        Xn = np.asarray(X.cpu() if torch.is_tensor(X) else X, dtype=np.float64)
        self.rank_ = estimate_rank(Xn) if self.rank is None else int(self.rank)
        X = _as_tensor(X, self.device, self.dtype)
        S = torch.zeros_like(X); L = torch.zeros_like(X)
        nx = float(torch.linalg.norm(X)); self.converged_ = False
        for k in range(1, self.max_iter + 1):
            Ln = self.filter(X - S)
            if self.rank_ > 0:
                Ln = randomized_truncation(Ln, self.rank_)
            Sn = adaptive_shrink(X - Ln, self.lam, p=self.p)
            change = float(max(torch.linalg.norm(Ln - L), torch.linalg.norm(Sn - S)) / nx)
            L, S = Ln, Sn
            if callback is not None:
                callback(k, L, S)
            if change < self.tol:
                self.converged_ = True; break
        self.L_, self.S_, self.n_iter_, self.last_change_ = L, S, k, change
        return self

    @property
    def scores_(self):
        return torch.abs(self.S_)


class FLoSCore:
    """FLoS-C, the convex core of FLoS: no truncation, soft thresholding, and the exact background step

        L_k = H (X - S_{k-1}),   H = P (Omega + alpha Lap^s)^{-1} P^T,   S_k = soft_lambda(X - L_k),

    with Omega the number of cells of each node. This is ISTA with unit step on
    Phi(S) = 1/2 <X - S, (I - H)(X - S)> + lambda ||S||_1, so Phi never increases (objective_ records it).
    Set adaptive=True to use the shrinkage of FLoS instead of soft thresholding.
    """

    def __init__(self, graph, alpha=0.85, s=0.25, lam=0.36, p=1.25, adaptive=False, max_iter=2000, tol=1e-5,
                 device=None, dtype=torch.float32):
        self.alpha, self.s, self.lam, self.p, self.adaptive = alpha, s, lam, p, adaptive
        self.max_iter, self.tol, self.dtype = max_iter, tol, dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        node = np.asarray(graph.node_of)
        n = int(node.max()) + 1
        cnt = np.bincount(node, minlength=n).astype(np.float64)
        U = torch.as_tensor(graph.U, dtype=torch.float64, device=self.device)
        d = torch.as_tensor(alpha * np.power(graph.ev + 1e-12, s), dtype=torch.float64, device=self.device)
        M = (U * d) @ U.T
        M.diagonal().add_(torch.as_tensor(cnt, dtype=torch.float64, device=self.device))
        C = torch.linalg.cholesky(0.5 * (M + M.T))
        self.Minv = torch.cholesky_inverse(C).to(dtype)
        self.node = torch.as_tensor(node, dtype=torch.long, device=self.device)
        self.n = n

    def background(self, Y):
        acc = torch.zeros(self.n, Y.shape[1], dtype=Y.dtype, device=Y.device)
        acc.index_add_(0, self.node, Y)
        return (self.Minv @ acc)[self.node]

    def fit(self, X):
        X = _as_tensor(X, self.device, self.dtype)
        nx = float(torch.linalg.norm(X))
        S = torch.zeros_like(X); L = torch.zeros_like(X)
        self.objective_, self.converged_ = [], False
        for k in range(1, self.max_iter + 1):
            Y = X - S
            Ln = self.background(Y)
            if not self.adaptive:
                self.objective_.append(0.5 * float(torch.sum(Y.double() * (Y - Ln).double()))
                                       + self.lam * float(torch.sum(torch.abs(S).double())))
            R = X - Ln
            Sn = adaptive_shrink(R, self.lam, p=self.p) if self.adaptive else soft_threshold(R, self.lam)
            change = float(max(torch.linalg.norm(Ln - L), torch.linalg.norm(Sn - S)) / nx)
            L, S = Ln, Sn
            if change < self.tol:
                self.converged_ = True; break
        self.L_, self.S_, self.n_iter_ = L, S, k
        return self

    @property
    def scores_(self):
        return torch.abs(self.S_)
