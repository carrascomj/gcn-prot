"""Expose only the models."""
from .models import GCN_simple, GCN_normed, sparsize
from .train import fit_network, forward_step

__all__ = [
    "GCN_simple",
    "GCN_normed",
    "fit_network",
    "forward_step",
    "sparsize",
]
