"""Building blocks of FLoS in PyTorch: the fractional graph filter, the rank estimate, the randomized truncation and
the shrinkage rules. Each one mirrors the implementation that produced the numbers in the paper, so that results
agree to floating-point precision on the same device and dtype."""
import numpy as np
import torch
from scipy import linalg


def median(t):
    """numpy.median semantics (mean of the two middle values for even length); torch.median returns the lower."""
    f, _ = torch.sort(t.reshape(-1))
    n = f.numel()
    return f[n // 2] if n % 2 else (f[n // 2 - 1] + f[n // 2]) * 0.5


class FractionalFilter:
    """F(Y) = P U diag(1 / (1 + alpha mu^s)) U^T A Y.

    A averages the cells of each node, the gain discounts each pattern by its roughness mu, and P copies the node
    values back to the cells. This is the minimiser of ||Z - A Y||^2 + alpha tr(Z^T Lap^s Z) on the nodes.
    """

    def __init__(self, graph, alpha=0.85, s=0.25, device="cpu", dtype=torch.float32):
        self.dtype = dtype
        gain = 1.0 / (1.0 + alpha * np.power(graph.ev + 1e-12, s))
        self.U = torch.as_tensor(graph.U, dtype=dtype, device=device)
        self.gain = torch.as_tensor(gain, dtype=dtype, device=device)[:, None]
        self.node = torch.as_tensor(np.asarray(graph.node_of), dtype=torch.long, device=device)
        n = int(self.node.max()) + 1
        self.w = (1.0 / torch.bincount(self.node, minlength=n).to(dtype))[:, None]
        self.n = n

    def __call__(self, Y):
        Y = Y.to(self.dtype)
        acc = torch.zeros(self.n, Y.shape[1], dtype=self.dtype, device=Y.device)
        acc.index_add_(0, self.node, Y)
        C = self.U.T @ (acc * self.w)
        return (self.U @ (C * self.gain))[self.node]


def estimate_rank(X, min_rank=2, max_rank=None, tw_eta=0.05):
    """Optimal hard threshold for singular values under unknown noise (Gavish and Donoho), with a small
    Tracy-Widom-type finite-size margin. X is a NumPy array (cells x time). The rank is not assumed small."""
    X = np.asarray(X)
    m, n = X.shape
    if min(m, n) <= 2:
        return 1
    _, s, _ = linalg.svd(X, full_matrices=False, overwrite_a=False, check_finite=False)
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    med = float(np.median(s))
    threshold = omega * med + 0.98 * med * min(m, n) ** (-2.0 / 3.0) * (1.0 + tw_eta)
    max_rank = min(m, n) - 1 if max_rank is None else max_rank
    return int(np.clip(int(np.sum(s > threshold)), min_rank, max_rank))


def randomized_truncation(M, rank, n_iter=1, random_state=0, n_oversamples=10):
    """Rank-`rank` approximation of M by randomized SVD; mirrors sklearn.utils.extmath.randomized_svd
    (normalizer 'none', NumPy random projection) so that the same random numbers are drawn on every device."""
    rank = int(max(1, min(rank, min(M.shape) - 1)))
    transpose = M.shape[0] < M.shape[1]
    A = M.T if transpose else M
    rs = np.random.RandomState(random_state)
    Q = torch.as_tensor(rs.normal(size=(A.shape[1], rank + n_oversamples)), dtype=A.dtype, device=A.device)
    for _ in range(n_iter):
        Q = A.T @ (A @ Q)
    Q, _ = torch.linalg.qr(A @ Q, mode="reduced")
    Uh, sv, Vt = torch.linalg.svd(Q.T @ A, full_matrices=False)
    U = Q @ Uh
    if transpose:
        u, sv, vt = Vt[:rank].T, sv[:rank], U[:, :rank].T
    else:
        u, sv, vt = U[:, :rank], sv[:rank], Vt[:rank]
    return (u * sv) @ vt


def adaptive_shrink(R, lam, p=1.25, eps=1e-6):
    """Event update of FLoS: threshold lam / (|R| / scale + eps)^(p - 1), with scale the normalised median absolute
    deviation of |R|. For p = 1 this is soft thresholding; for p > 1 large residuals are shrunk less."""
    a = torch.abs(R)
    med = median(a)
    mad = median(torch.abs(a - med)) / 0.6745
    scale = mad if float(mad) > 1e-8 else torch.std(a)
    scale = float(scale) if float(scale) > 1e-8 else 1.0
    tau = lam / torch.pow(a / scale + eps, max(p - 1.0, 1e-3))
    return torch.sign(R) * torch.clamp(a - tau, min=0.0)


def soft_threshold(R, lam):
    return torch.sign(R) * torch.clamp(torch.abs(R) - lam, min=0.0)
