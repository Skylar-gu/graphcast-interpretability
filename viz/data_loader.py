"""Data loading utilities for the SAE feature atlas."""

from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

DATA_DIR = Path(__file__).parent / "data"


# ── Registry / catalog ────────────────────────────────────────────────────────

def load_registry() -> dict:
    path = DATA_DIR / "sae_registry.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_catalog() -> pd.DataFrame:
    path = DATA_DIR / "feature_catalog.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return (grid_lat, grid_lon) saved during precompute.py."""
    graph_dir = DATA_DIR / "graph"
    return (
        np.load(str(graph_dir / "grid_lat.npy")),
        np.load(str(graph_dir / "grid_lon.npy")),
    )


# ── Zarr access (cached handles) ──────────────────────────────────────────────

@lru_cache(maxsize=8)
def _open_stores(sae_id: str) -> tuple:
    """Open Zarr index/value stores and timestamps for a given SAE. Cached."""
    root = DATA_DIR / "activations" / sae_id
    idx = zarr.open(str(root / "indices.zarr"), mode="r")
    val = zarr.open(str(root / "values.zarr"), mode="r")
    ts = list(np.load(str(root / "timestamps.npy"), allow_pickle=True))
    return idx, val, ts


def get_timestamps(sae_id: str) -> list[str]:
    _, _, ts = _open_stores(sae_id)
    return ts


# ── Activation retrieval ──────────────────────────────────────────────────────

def get_sparse_codes(sae_id: str, t_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) for one timestep. Both shape: (n_nodes, k_active)."""
    idx_store, val_store, _ = _open_stores(sae_id)
    return idx_store[t_idx], val_store[t_idx]


def get_feature_activation(sae_id: str, t_idx: int,
                            feature_id: int, n_nodes: int) -> np.ndarray:
    """Dense (n_nodes,) float32 activation for one feature at one timestep."""
    indices, values = get_sparse_codes(sae_id, t_idx)
    dense = np.zeros(n_nodes, dtype=np.float32)
    mask = indices == feature_id          # (n_nodes, k_active) bool
    rows, _ = np.where(mask)
    dense[rows] = values[mask]
    return dense


def get_feature_timeseries(sae_id: str, feature_id: int,
                            n_time: int) -> tuple[list[str], np.ndarray]:
    """
    Return (timestamps, mean_activation_per_timestep).
    mean_activation is taken over active (nonzero) nodes only.
    """
    idx_store, val_store, timestamps = _open_stores(sae_id)
    means = np.zeros(n_time, dtype=np.float32)
    for t in range(n_time):
        idxs = idx_store[t]     # (n_nodes, k_active)
        vals = val_store[t]
        mask = idxs == feature_id
        if mask.any():
            means[t] = float(vals[mask].mean())
    return timestamps, means
