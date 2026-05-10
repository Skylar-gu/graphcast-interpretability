"""
Target-centered causal graph recovery from SAE feature activations.

Pipeline:
  1. Extract scalar feature time series from sparse Zarr activations
  2. MI screening — rank all features by mutual information with focal feature
  3. PCMCI+ — recover time-lag causal graph over selected features
  4. ANM orientation — resolve any remaining undirected edges
  5. Validate recovered edges against expected physical structure
  6. Export causal_graph.png + results.json + validation.txt

Usage:
    python scripts/causal_graph.py \\
        --focal_feature 3243 \\
        --seed_features 117,402,911,2088 \\
        --n_mi_candidates 20 \\
        --tau_max 4 \\
        --sae_id layer8_k32_d4096 \\
        --data_dir viz/data \\
        --out_dir results/causal

Expected graph structure (Hurricane Ida case):
    Feature 117 (moist inflow)    ─6h→  Feature 3243 (hurricane-core)
    Feature 402 (low-pressure)    ─6h→  Feature 3243
    Feature 911 (warm-core)       ─12h→ Feature 3243
    Feature 3243                  ─6h→  Feature 2088 (outflow)
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional heavy imports — give clear errors if missing ────────────────────
def _require(pkg_name: str, import_name: str | None = None):
    import importlib
    name = import_name or pkg_name
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"ERROR: '{pkg_name}' not installed.  pip install {pkg_name}")
        sys.exit(1)


# ── Known physical edges for validation ──────────────────────────────────────
# (source_feature, target_feature, expected_lag_steps, description)
KNOWN_EDGES: list[tuple[int, int, int, str]] = [
    (117,  3243, 1, "moist inflow → TC core, 6h"),
    (402,  3243, 1, "low-pressure → TC core, 6h"),
    (911,  3243, 2, "warm-core → TC intensification, 12h"),
    (3243, 2088, 1, "TC core → upper outflow, 6h"),
]


# ── Feature time series extraction ───────────────────────────────────────────

def extract_timeseries(
    sae_id: str,
    feature_ids: list[int],
    data_dir: Path,
    aggregation: str = "max",
) -> tuple[np.ndarray, list[str]]:
    """
    Load sparse Zarr activations and return a (T, F) float32 matrix.

    aggregation:
      'max'  — peak activation across nodes per timestep (good for rare/localized features)
      'mean' — mean over active nodes per timestep
      'p95'  — 95th percentile across nodes per timestep
    """
    import zarr
    zarr_root = data_dir / "activations" / sae_id
    idx_store = zarr.open(str(zarr_root / "indices.zarr"), mode="r")
    val_store = zarr.open(str(zarr_root / "values.zarr"), mode="r")
    timestamps = list(np.load(str(zarr_root / "timestamps.npy"), allow_pickle=True))

    n_time = idx_store.shape[0]
    n_nodes = idx_store.shape[1]
    k_active = idx_store.shape[2]
    F = len(feature_ids)

    feat_set = set(feature_ids)
    feat_pos = {f: i for i, f in enumerate(feature_ids)}

    series = np.zeros((n_time, F), dtype=np.float32)

    print(f"Extracting time series for {F} features over {n_time} timesteps...")
    for t in range(n_time):
        if t % 100 == 0:
            print(f"  {t}/{n_time}", end="\r", flush=True)
        idxs = idx_store[t]   # (n_nodes, k_active) int16
        vals = val_store[t]   # (n_nodes, k_active) float32

        for feat_id in feat_set:
            col = feat_pos[feat_id]
            mask = idxs == feat_id                  # (n_nodes, k_active) bool
            node_vals = vals[mask.any(axis=-1)]     # active node values (variable length)
            if node_vals.size == 0:
                continue
            # Aggregate across nodes for this timestep
            if aggregation == "max":
                series[t, col] = node_vals.max()
            elif aggregation == "mean":
                series[t, col] = node_vals.mean()
            elif aggregation == "p95":
                series[t, col] = np.percentile(node_vals, 95)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

    print(f"\nDone. Series shape: {series.shape}")
    return series, timestamps


# ── MI screening ─────────────────────────────────────────────────────────────

def mi_screening(
    all_series: np.ndarray,          # (T, n_all_features)
    focal_col: int,
    tau_max: int,
    n_candidates: int,
    n_neighbors: int = 5,
) -> list[int]:
    """
    Rank all features by max mutual information with the focal feature across lags 0..tau_max.
    Returns indices (into all_series columns) of top-n_candidates features.
    Uses sklearn's k-NN MI estimator (Kraskov et al.).
    """
    from sklearn.feature_selection import mutual_info_regression

    T, N = all_series.shape
    focal = all_series[:, focal_col]
    scores = np.zeros(N)

    for tau in range(0, tau_max + 1):
        if tau == 0:
            X = all_series
            y = focal
        else:
            X = all_series[:-tau]
            y = focal[tau:]

        mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=0)
        scores = np.maximum(scores, mi)

    # Exclude focal itself
    scores[focal_col] = -1.0
    top_idx = np.argsort(scores)[::-1][:n_candidates]
    return top_idx.tolist()


# ── PCMCI+ causal discovery ───────────────────────────────────────────────────

def run_pcmciplus(
    series: np.ndarray,    # (T, N)
    var_names: list[str],
    tau_max: int,
    alpha: float,
    cond_ind_test: str,
) -> dict:
    """
    Run PCMCI+ from the tigramite package.
    Returns a dict with val_matrix, p_matrix, and graph arrays.
    """
    tigramite = _require("tigramite")
    pp = _require("tigramite.data_processing", "tigramite.data_processing")

    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    if cond_ind_test == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        ci_test = ParCorr(significance="analytic")
    elif cond_ind_test == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        ci_test = CMIknn(significance="shuffle_test", knn=10)
    else:
        raise ValueError(f"Unknown cond_ind_test: {cond_ind_test}")

    dataframe = pp.DataFrame(
        series,
        datatime={0: np.arange(series.shape[0])},
        var_names=var_names,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=1)
    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=alpha,
    )
    return results


# ── Additive Noise Model edge orientation ────────────────────────────────────

def anm_orient(x: np.ndarray, y: np.ndarray) -> tuple[str, float, float]:
    """
    Test direction of a causal edge using additive noise model.
    Returns ('x->y' | 'y->x' | 'inconclusive', score_xy, score_yx).
    Lower score = more independent residuals = preferred direction.

    Uses GP regression + HSIC independence test via causal-learn.
    """
    try:
        from causallearn.search.FCMBased.ANM.ANM import ANM
        anm = ANM()
        p_val_xy, _ = anm.cause_or_effect(x.reshape(-1, 1), y.reshape(-1, 1))
        p_val_yx, _ = anm.cause_or_effect(y.reshape(-1, 1), x.reshape(-1, 1))
        if p_val_xy > 0.05 and p_val_yx <= 0.05:
            return "x->y", p_val_xy, p_val_yx
        elif p_val_yx > 0.05 and p_val_xy <= 0.05:
            return "y->x", p_val_xy, p_val_yx
        else:
            return "inconclusive", p_val_xy, p_val_yx
    except Exception as e:
        warnings.warn(f"ANM failed: {e}")
        return "inconclusive", float("nan"), float("nan")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_edges(
    val_matrix: np.ndarray,   # (N, N, tau_max+1)
    p_matrix: np.ndarray,     # same shape
    feature_ids: list[int],
    alpha: float,
) -> list[dict]:
    """Compare PCMCI+ results against KNOWN_EDGES."""
    feat_pos = {f: i for i, f in enumerate(feature_ids)}
    rows = []
    for src, tgt, lag, desc in KNOWN_EDGES:
        if src not in feat_pos or tgt not in feat_pos:
            rows.append({
                "edge": desc, "src": src, "tgt": tgt, "lag": lag,
                "status": "MISSING — feature not in graph",
                "val": None, "p": None,
            })
            continue
        i, j = feat_pos[src], feat_pos[tgt]
        if lag > val_matrix.shape[2] - 1:
            rows.append({"edge": desc, "status": "lag out of range",
                         "val": None, "p": None})
            continue
        val = float(val_matrix[i, j, lag])
        pval = float(p_matrix[i, j, lag])
        status = "RECOVERED ✓" if pval < alpha and val > 0 else (
            "RECOVERED (negative)" if pval < alpha else "NOT SIGNIFICANT"
        )
        rows.append({
            "edge": desc, "src": src, "tgt": tgt, "lag": lag,
            "status": status, "val": round(val, 4), "p": round(pval, 4),
        })
    return rows


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_causal_graph(
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    feature_ids: list[int],
    var_names: list[str],
    tau_max: int,
    alpha: float,
    out_path: Path,
    focal_feature: int,
) -> None:
    """Draw a lag-graph diagram highlighting edges to/from the focal feature."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    N = len(feature_ids)
    fig, axes = plt.subplots(1, tau_max + 1, figsize=(4 * (tau_max + 1), 4), squeeze=False)
    fig.suptitle(f"Causal lag graph — focal feature {focal_feature}", fontsize=12)

    focal_col = feature_ids.index(focal_feature) if focal_feature in feature_ids else -1

    for tau in range(tau_max + 1):
        ax = axes[0, tau]
        ax.set_title(f"lag τ={tau} ({tau*6}h)")
        mat = val_matrix[:, :, tau]
        pmat = p_matrix[:, :, tau]
        sig = pmat < alpha
        display = np.where(sig, mat, np.nan)

        im = ax.imshow(display, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(var_names, rotation=90, fontsize=7)
        ax.set_yticklabels(var_names, fontsize=7)
        ax.set_xlabel("target")
        ax.set_ylabel("source")

        # Highlight focal feature row/col
        if focal_col >= 0:
            for spine in ax.spines.values():
                pass
            ax.axhline(focal_col - 0.5, color="orange", linewidth=1.5, alpha=0.6)
            ax.axhline(focal_col + 0.5, color="orange", linewidth=1.5, alpha=0.6)
            ax.axvline(focal_col - 0.5, color="orange", linewidth=1.5, alpha=0.6)
            ax.axvline(focal_col + 0.5, color="orange", linewidth=1.5, alpha=0.6)

        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved causal graph: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Target-centered causal graph from SAE features")
    p.add_argument("--focal_feature",    type=int, default=3243,
                   help="Central feature to explain (hurricane-core)")
    p.add_argument("--seed_features",   default="117,402,911,2088",
                   help="Comma-separated feature IDs to include unconditionally")
    p.add_argument("--n_mi_candidates", type=int, default=20,
                   help="Additional features selected by MI screening")
    p.add_argument("--tau_max",         type=int, default=4,
                   help="Max lag in timesteps (1 step = 6h); default 4 = 24h")
    p.add_argument("--alpha",           type=float, default=0.05,
                   help="Significance threshold for PCMCI+")
    p.add_argument("--cond_ind_test",   default="parcorr",
                   choices=["parcorr", "cmiknn"],
                   help="Conditional independence test (parcorr=fast, cmiknn=nonlinear)")
    p.add_argument("--aggregation",     default="max",
                   choices=["max", "mean", "p95"],
                   help="How to aggregate activations across mesh nodes per timestep")
    p.add_argument("--sae_id",          default="layer8_k32_d4096")
    p.add_argument("--data_dir",        default="viz/data")
    p.add_argument("--out_dir",         default="results/causal")
    p.add_argument("--skip_anm",        action="store_true",
                   help="Skip ANM orientation step (faster)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_features = [int(x) for x in args.seed_features.split(",") if x.strip()]
    focal = args.focal_feature

    # ── Load feature catalog for labels ───────────────────────────────────
    catalog_path = data_dir / "feature_catalog.parquet"
    labels: dict[int, str] = {}
    if catalog_path.exists():
        cat = pd.read_parquet(catalog_path)
        for _, row in cat.iterrows():
            if row.get("candidate_label"):
                labels[int(row["feature_id"])] = str(row["candidate_label"])
    labels.setdefault(3243, "TC-core")
    labels.setdefault(117,  "moist-inflow")
    labels.setdefault(402,  "low-pressure")
    labels.setdefault(911,  "warm-core")
    labels.setdefault(2088, "outflow")

    # ── Step 1: Load full feature time series for MI screening ────────────
    import zarr
    zarr_root = data_dir / "activations" / args.sae_id
    reg_path  = data_dir / "sae_registry.json"
    import json as _json
    with open(reg_path) as f:
        info = _json.load(f)[args.sae_id]

    n_features = info["latent"]
    n_time     = info["n_time"]
    n_nodes    = info["n_nodes"]

    print(f"Dataset: {n_time} timesteps, {n_nodes} nodes, {n_features} features")
    print(f"Focal feature: {focal}  |  Seed features: {seed_features}")

    # Extract time series for ALL features (needed for MI screening)
    # To avoid loading all 4096×T at once, do two passes:
    # Pass 1: all features for MI scoring (cheap: just look at scalar)
    # Pass 2: selected features (already cached from pass 1 if seed+MI overlap)

    print("\n=== Pass 1: Extract time series for all features (MI screening) ===")
    all_series, timestamps = extract_timeseries(
        args.sae_id,
        list(range(n_features)),
        data_dir,
        aggregation=args.aggregation,
    )

    # ── Step 2: MI screening ───────────────────────────────────────────────
    print("\n=== Step 2: MI screening ===")
    focal_col = focal
    mi_top_idx = mi_screening(
        all_series,
        focal_col=focal_col,
        tau_max=args.tau_max,
        n_candidates=args.n_mi_candidates,
    )
    mi_features = [int(i) for i in mi_top_idx]
    print(f"Top-{args.n_mi_candidates} MI features: {mi_features}")

    # ── Assemble final feature set ─────────────────────────────────────────
    selected = list(dict.fromkeys([focal] + seed_features + mi_features))
    selected_series = all_series[:, selected]
    var_names = [labels.get(f, f"f{f}") for f in selected]
    N = len(selected)
    T = selected_series.shape[0]
    print(f"\nFinal graph nodes ({N}): {selected}")
    print(f"T={T}, p={N}, T/p={T/N:.1f}")
    if T / N < 10:
        warnings.warn(
            f"T/p = {T/N:.1f} is low. PCMCI+ results will be unreliable. "
            "Generate more timesteps or reduce --n_mi_candidates."
        )

    # ── Step 3: PCMCI+ ────────────────────────────────────────────────────
    print("\n=== Step 3: PCMCI+ ===")
    pcmci_results = run_pcmciplus(
        selected_series,
        var_names=var_names,
        tau_max=args.tau_max,
        alpha=args.alpha,
        cond_ind_test=args.cond_ind_test,
    )
    val_matrix = pcmci_results["val_matrix"]   # (N, N, tau_max+1)
    p_matrix   = pcmci_results["p_matrix"]
    graph      = pcmci_results.get("graph", None)

    # ── Step 4: ANM orientation ────────────────────────────────────────────
    anm_results: list[dict] = []
    if not args.skip_anm:
        print("\n=== Step 4: ANM edge orientation ===")
        # Find contemporaneous edges (tau=0) that are undirected in the PAG
        if graph is not None:
            for i in range(N):
                for j in range(i + 1, N):
                    edge_ij = graph[i, j, 0]
                    edge_ji = graph[j, i, 0]
                    # "o-o" or "-o" pattern in PAG = undirected
                    if "o" in str(edge_ij) or "o" in str(edge_ji):
                        xi = selected_series[:, i]
                        xj = selected_series[:, j]
                        direction, pxy, pyx = anm_orient(xi, xj)
                        anm_results.append({
                            "var_i": var_names[i], "feat_i": selected[i],
                            "var_j": var_names[j], "feat_j": selected[j],
                            "direction": direction,
                            "p_ij": round(pxy, 4), "p_ji": round(pyx, 4),
                        })
                        print(f"  {var_names[i]} ↔ {var_names[j]}: ANM → {direction}")
        else:
            print("  (graph object not available; skipping ANM)")

    # ── Step 5: Validation ─────────────────────────────────────────────────
    print("\n=== Step 5: Validation against known physical edges ===")
    validation = validate_edges(val_matrix, p_matrix, selected, args.alpha)
    for row in validation:
        print(f"  [{row['status']}]  {row['edge']}  "
              f"val={row['val']}  p={row['p']}")

    # ── Save outputs ───────────────────────────────────────────────────────
    print("\n=== Saving outputs ===")

    # Causal graph figure
    plot_causal_graph(
        val_matrix, p_matrix, selected, var_names,
        tau_max=args.tau_max, alpha=args.alpha,
        out_path=out_dir / "causal_graph.png",
        focal_feature=focal,
    )

    # Full results JSON
    results = {
        "config": vars(args),
        "selected_features": selected,
        "var_names": var_names,
        "n_timesteps": int(T),
        "t_over_p": float(T / N),
        "val_matrix": val_matrix.tolist(),
        "p_matrix": p_matrix.tolist(),
        "mi_top_features": mi_features,
        "anm_results": anm_results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Validation table
    val_df = pd.DataFrame(validation)
    val_df.to_csv(out_dir / "validation.txt", index=False, sep="\t")
    print(val_df.to_string(index=False))

    # Summary
    n_recovered = sum(1 for r in validation if "RECOVERED ✓" in r.get("status", ""))
    print(f"\n{'='*60}")
    print(f"Recovered {n_recovered}/{len(KNOWN_EDGES)} expected physical edges")
    print(f"Results: {out_dir}/")


if __name__ == "__main__":
    main()
