"""FLoS: decomposition of a field into a smooth background and sparse events under a spherical smoothness prior."""
from .graph import Graph, latlon_graph
from .model import FLoS, FLoSCore
from .ops import FractionalFilter, adaptive_shrink, estimate_rank, randomized_truncation, soft_threshold
from .data import load_anomalies

__all__ = ["Graph", "latlon_graph", "FLoS", "FLoSCore", "FractionalFilter", "adaptive_shrink", "estimate_rank",
           "randomized_truncation", "soft_threshold", "load_anomalies"]
__version__ = "1.0.0"
