"""Graphs on a mesh and the eigendecomposition of their Laplacian.

A smoothness prior only needs a notion of neighbourhood. Cells of the mesh that sit at the same physical point
(for example the cells along a pole of a latitude-longitude mesh) are merged into one node; neighbouring nodes are
joined with weight 1 / distance, so that the Dirichlet energy measures variation per unit of physical distance.
The Laplacian is divided by its largest degree and its eigenvalues are rescaled to [0, 1]; eigenvector j is a
spatial pattern and its eigenvalue mu_j is the roughness of that pattern.
"""
import hashlib
import os
import time

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix

POLE_TOL = 1e-6


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in radians between two points given in degrees."""
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi, dlam = np.deg2rad(lat2 - lat1), np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * np.arctan2(np.sqrt(a), np.sqrt(max(1.0 - a, 0.0))))


def latlon_graph(lat, lon):
    """Weights of the spherical graph of a regular latitude-longitude mesh.

    Cells are ordered row by row (latitude major). The cells of a polar row are merged into one node; every other
    cell is its own node. Each node is joined to its northern and eastern neighbour (longitude wraps around) with
    weight 1 / great-circle distance.

    Returns (W, node_of): the dense symmetric weight matrix over the nodes and, for every cell, the index of its node.
    """
    nlat, nlon = len(lat), len(lon)
    node_of = np.empty(nlat * nlon, dtype=int)
    n = 0
    for i in range(nlat):
        if abs(abs(float(lat[i])) - 90.0) < POLE_TOL:
            node_of[i * nlon:(i + 1) * nlon] = n; n += 1
        else:
            node_of[i * nlon:(i + 1) * nlon] = np.arange(n, n + nlon); n += nlon
    rows, cols, vals, seen = [], [], [], set()
    for i in range(nlat):
        for j in range(nlon):
            a = int(node_of[i * nlon + j])
            for ii, jj in ((i + 1, j), (i, (j + 1) % nlon)):
                if ii >= nlat:
                    continue
                b = int(node_of[ii * nlon + jj]); key = (min(a, b), max(a, b))
                if a == b or key in seen:
                    continue
                seen.add(key)
                w = 1.0 / max(haversine(float(lat[i]), float(lon[j]), float(lat[ii]), float(lon[jj])), 1e-6)
                rows += [a, b]; cols += [b, a]; vals += [w, w]
    W = coo_matrix((vals, (rows, cols)), shape=(n, n)).toarray()
    return W, node_of


def laplacian_eigen(W):
    """Eigenvalues (rescaled to [0, 1]) and eigenvectors of the Laplacian of W, divided by its largest degree."""
    lap = np.diag(W.sum(1)) - W
    lap /= float(np.max(np.diag(lap)))
    ev, U = linalg.eigh(lap, overwrite_a=True, check_finite=False)
    ev = np.maximum(ev, 0.0)
    return ev / ev.max(), U


class Graph:
    """Spectral data of a mesh graph: roughness `ev` of every pattern, patterns `U` (columns) and `node_of`."""

    def __init__(self, ev, U, node_of):
        self.ev, self.U, self.node_of = np.asarray(ev), np.asarray(U), np.asarray(node_of)

    @property
    def n_nodes(self):
        return self.U.shape[0]

    @property
    def n_cells(self):
        return len(self.node_of)

    @classmethod
    def from_weights(cls, W, node_of=None):
        """Any mesh: W is the symmetric node-by-node weight matrix; node_of maps cells to nodes (identity if None)."""
        ev, U = laplacian_eigen(np.asarray(W, dtype=float))
        return cls(ev, U, np.arange(len(W)) if node_of is None else node_of)

    @classmethod
    def from_latlon(cls, lat, lon, cache_dir=None, verbose=True):
        """Spherical graph of a regular latitude-longitude mesh, with the eigendecomposition cached on disk.

        The cache key is the same as in the experiment scripts (reproduce/src), so a cache built by either is reused.
        """
        cache_dir = cache_dir or os.environ.get("FLOS_CACHE")
        p = None
        if cache_dir:
            h = hashlib.sha1(); h.update(np.asarray(lat, float).tobytes()); h.update(np.asarray(lon, float).tobytes())
            p = os.path.join(cache_dir, "fixed_eig_%s" % h.hexdigest()[:16])
            if os.path.exists(p + "_U.npy"):
                return cls(np.load(p + "_ev.npy"), np.load(p + "_U.npy"), np.load(p + "_node.npy"))
        t = time.perf_counter()
        W, node_of = latlon_graph(lat, lon)
        ev, U = laplacian_eigen(W)
        if verbose:
            print("graph eigendecomposition: %d nodes in %.0f s" % (len(ev), time.perf_counter() - t), flush=True)
        if p:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(p + "_ev.npy", ev); np.save(p + "_node.npy", node_of); np.save(p + "_U.npy", U)
        return cls(ev, U, node_of)
