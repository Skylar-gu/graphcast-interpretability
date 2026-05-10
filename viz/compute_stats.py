"""
Compute per-feature statistics from precomputed sparse activations.
Outputs feature_catalog.parquet for the feature browser.

Usage:
    python viz/compute_stats.py --sae_id layer8_k32_d4096 [--data_dir viz/data]

Memory note: loads full (T, N, K) index array into RAM. For n_time=100, n_nodes=40962,
k_active=32 this is ~250 MB. For much larger datasets pass --streaming to process
one timestep at a time (slower but uses O(N*K) memory instead).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


KNOWN_LABELS: dict[int, str] = {
    3243: "Tropical cyclone / hurricane tracker",
    1820: "Atmospheric river / IVT plume",
}


def compute_stats_batch(
    indices: np.ndarray,   # (T, N, K) int16
    values: np.ndarray,    # (T, N, K) float32
    n_features: int,
    n_time: int,
    n_nodes: int,
    k_active: int,
    timestamps: list[str],
    top_k_examples: int,
) -> pd.DataFrame:
    """Vectorized stats computation. Requires O(T*N*K) extra RAM."""

    flat_idx = indices.ravel().astype(np.int32)   # (T*N*K,)
    flat_val = values.ravel()                       # (T*N*K,)

    # ── Global counts / mean / max ────────────────────────────────────────
    counts = np.bincount(flat_idx, minlength=n_features)            # fires per feature
    val_sums = np.bincount(flat_idx, weights=flat_val, minlength=n_features)
    mean_act = np.where(counts > 0, val_sums / counts, 0.0)

    max_act = np.zeros(n_features, dtype=np.float32)
    np.maximum.at(max_act, flat_idx, flat_val)

    # activation_frequency = fraction of (t, node) pairs where this feature fires
    freq = counts / (n_time * n_nodes)

    # ── Per-timestep peak (for top-example ranking) ───────────────────────
    # compound index (t * n_features + feat)
    t_ids = np.repeat(np.arange(n_time, dtype=np.int32), n_nodes * k_active)
    tf_idx = t_ids * n_features + flat_idx
    time_feat_max = np.zeros(n_time * n_features, dtype=np.float32)
    np.maximum.at(time_feat_max, tf_idx, flat_val)
    time_feat_max = time_feat_max.reshape(n_time, n_features)  # (T, F)

    # ── Artifact score ────────────────────────────────────────────────────
    # For each (node, feature) pair: count across time.
    # compound index (feat * n_nodes + node)
    node_ids = np.tile(
        np.repeat(np.arange(n_nodes, dtype=np.int32), k_active), n_time
    )
    fn_idx = flat_idx * n_nodes + node_ids
    pair_counts = np.bincount(fn_idx, minlength=n_features * n_nodes)
    # pair_counts[f * n_nodes + n] = # timesteps feature f fires at node n
    pair_freq = pair_counts.reshape(n_features, n_nodes).astype(np.float32) / n_time

    # Normalized spatial std: features that fire at the same nodes every time score high.
    nf_mean = pair_freq.mean(axis=1)   # (F,)
    nf_std = pair_freq.std(axis=1)     # (F,)
    artifact_raw = np.where(nf_mean > 0, nf_std / (nf_mean + 1e-8), 0.0)
    artifact_max = artifact_raw.max()
    artifact_score = artifact_raw / artifact_max if artifact_max > 0 else artifact_raw

    # ── Assemble catalog ──────────────────────────────────────────────────
    records = []
    for i in range(n_features):
        top_t = np.argsort(time_feat_max[:, i])[::-1][:top_k_examples]
        top_ts = [timestamps[t] for t in top_t if time_feat_max[t, i] > 0]
        records.append({
            "feature_id": i,
            "activation_frequency": float(freq[i]),
            "mean_activation": float(mean_act[i]),
            "max_activation": float(max_act[i]),
            "n_active_tokens": int(counts[i]),
            "artifact_score": float(artifact_score[i]),
            "dead": bool(counts[i] == 0),
            "candidate_label": KNOWN_LABELS.get(i, ""),
            "top_timestamps": top_ts,
        })
    return pd.DataFrame(records)


def compute_stats_streaming(
    idx_store: zarr.Array,
    val_store: zarr.Array,
    n_features: int,
    n_time: int,
    n_nodes: int,
    k_active: int,
    timestamps: list[str],
    top_k_examples: int,
) -> pd.DataFrame:
    """Single-pass streaming stats. O(N*K) peak RAM."""
    counts = np.zeros(n_features, dtype=np.int64)
    val_sums = np.zeros(n_features, dtype=np.float64)
    max_act = np.zeros(n_features, dtype=np.float32)
    time_feat_max = np.zeros((n_time, n_features), dtype=np.float32)
    node_fire_sum = np.zeros((n_features, n_nodes), dtype=np.float32)

    for t in range(n_time):
        print(f"  stats pass [{t + 1}/{n_time}]", end="\r", flush=True)
        idxs = idx_store[t].ravel().astype(np.int32)   # (N*K,)
        vals = val_store[t].ravel()

        np.add.at(counts, idxs, 1)
        np.add.at(val_sums, idxs, vals)
        np.maximum.at(max_act, idxs, vals)

        # time_feat_max: max val per (t, feat)
        np.maximum.at(time_feat_max[t], idxs, vals)

        # per-(feat, node) fire count
        node_idx = np.repeat(np.arange(n_nodes, dtype=np.int32), k_active)
        fn_idx = idxs * n_nodes + node_idx
        np.add.at(node_fire_sum.ravel(), fn_idx, 1.0)

    print()  # newline after \r progress

    mean_act = np.where(counts > 0, val_sums / counts, 0.0).astype(np.float32)
    freq = counts / (n_time * n_nodes)
    pair_freq = node_fire_sum / n_time
    nf_mean = pair_freq.mean(axis=1)
    nf_std = pair_freq.std(axis=1)
    artifact_raw = np.where(nf_mean > 0, nf_std / (nf_mean + 1e-8), 0.0)
    artifact_max = artifact_raw.max()
    artifact_score = artifact_raw / artifact_max if artifact_max > 0 else artifact_raw

    records = []
    for i in range(n_features):
        top_t = np.argsort(time_feat_max[:, i])[::-1][:top_k_examples]
        top_ts = [timestamps[t] for t in top_t if time_feat_max[t, i] > 0]
        records.append({
            "feature_id": i,
            "activation_frequency": float(freq[i]),
            "mean_activation": float(mean_act[i]),
            "max_activation": float(max_act[i]),
            "n_active_tokens": int(counts[i]),
            "artifact_score": float(artifact_score[i]),
            "dead": bool(counts[i] == 0),
            "candidate_label": KNOWN_LABELS.get(i, ""),
            "top_timestamps": top_ts,
        })
    return pd.DataFrame(records)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_id", default="layer8_k32_d4096")
    p.add_argument("--data_dir", default="viz/data")
    p.add_argument("--top_k_examples", type=int, default=5)
    p.add_argument("--streaming", action="store_true",
                   help="Process one timestep at a time (lower RAM, slower)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    zarr_root = data_dir / "activations" / args.sae_id

    with open(data_dir / "sae_registry.json") as f:
        info = json.load(f)[args.sae_id]

    n_time = info["n_time"]
    n_nodes = info["n_nodes"]
    n_features = info["latent"]
    k_active = info["k_active"]
    timestamps = info["timestamps"]

    idx_store = zarr.open(str(zarr_root / "indices.zarr"), mode="r")
    val_store = zarr.open(str(zarr_root / "values.zarr"), mode="r")

    if args.streaming:
        print("Streaming mode (single-pass)...")
        df = compute_stats_streaming(
            idx_store, val_store, n_features, n_time, n_nodes,
            k_active, timestamps, args.top_k_examples,
        )
    else:
        print("Loading full index array into memory...")
        indices = idx_store[:]
        values = val_store[:]
        print(f"  indices: {indices.shape}, {indices.nbytes / 1e6:.0f} MB")
        print("Computing stats...")
        df = compute_stats_batch(
            indices, values, n_features, n_time, n_nodes,
            k_active, timestamps, args.top_k_examples,
        )

    out_path = data_dir / "feature_catalog.parquet"
    df.to_parquet(out_path, index=False)

    n_dead = df["dead"].sum()
    print(f"Saved {len(df)} features to {out_path}")
    print(f"  Dead: {n_dead}  Active: {len(df) - n_dead}")
    print(f"  Mean freq: {df['activation_frequency'].mean():.5f}")
    print(f"  High-artifact (>0.8): {(df['artifact_score'] > 0.8).sum()}")


if __name__ == "__main__":
    main()
