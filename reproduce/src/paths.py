# -*- coding: utf-8 -*-
"""Where data, outputs and caches live. Every script imports its locations from here.

Defaults are folders next to src/; override any of them with an environment variable.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FLOS_DATA", os.path.join(ROOT, "data"))            # raw/, era5/clean/, 20cr/clean/
OUT = os.environ.get("FLOS_OUT", os.path.join(ROOT, "outputs"))           # every CSV, table and figure
CACHE = os.environ.get("FLOS_CACHE", os.path.join(ROOT, "cache"))         # graph eigendecompositions
EXTERNAL = os.environ.get("FLOS_EXTERNAL", os.path.join(ROOT, "external"))  # third-party baseline repositories

for _d in (OUT, CACHE):
    os.makedirs(_d, exist_ok=True)


def device():
    """Torch device for the FLoS runs: $FLOS_DEVICE if set, else CUDA when available."""
    if os.environ.get("FLOS_DEVICE"):
        return os.environ["FLOS_DEVICE"]
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
