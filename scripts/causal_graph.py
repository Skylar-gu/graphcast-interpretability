"""
Lagged dependency / causal graph from SAE feature activations.

Two modes:

  --mode lagged_corr  (default)
      Fast demo mode: pairwise lagged Pearson cross-correlations between seed
      features only.  Works with as few as 32 timesteps (e.g. one Colab run).
      Needs only scipy + networkx.  Produces lag_dep_graph.png in ~5 seconds.

  --mode pcmciplus
      Full cluster mode: MI screening over all features + PCMCI+ + ANM
      orientation.  Needs tigramite + causal-learn.  Requires ~750+ timesteps.

Usage — demo (32 timesteps, Colab output):
    python scripts/causal_graph.py \\
        --focal_feature 3243 \\
        --seed_features 117,402,911,2088 \\
        --tau_max 4 \\
        --mode lagged_corr \\
        --sae_id layer8_k32_d4096 \\
        --data_dir viz/data \\
        --out_dir results/causal

Usage — cluster (1464 timesteps, full PCMCI+):
    python scripts/causal_graph.py \\
        --focal_feature 3243 \\
        --seed_features 117,402,911,2088 \\
        --n_mi_candidates 20 \\
        --tau_max 4 \\
        --mode pcmciplus \\
        --sae_id layer8_k32_d4096 \\
        --data_dir viz/data \\
        --out_dir results/causal

Expected dependency structure (Hurricane Ida case):
    Feature 117 (moist inflow)  ─6h→  Feature 3243 (hurricane-core)
    Feature 402 (low-pressure)  ─6h→  Feature 3243
    Feature 911 (warm-core)    ─12h→  Feature 3243
    Feature 3243               ─6h→  Feature 2088 (outflow)
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_LABELS: dict[int, str] = {
    3243: "TC-core",
    117:  "moist-inflow",
    402:  "low-pressure",
    911:  "warm-core",
    2088: "outflow",
}

# (source_feature, target_feature, expected_lag_steps, description)
KNOWN_EDGES: list[tuple[int, int, int, str]] = [
    (117,  3243, 1, "moist inflow → TC core, 6h"),
    (402,  3243, 1, "low-pressure → TC core, 6h"),
    (911,  3243, 2, "warm-core → TC intensification, 12h"),
    (3243, 2088, 1, "TC core → upper outflow, 6h"),
]


def _require(pkg_name: str, import_name: str | None = None):
    import importlib
    name = import_name or pkg_name
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"ERROR: '{pkg_name}' not installed.  pip install {pkg_name}")
        sys.exit(1)


# ── Feature time series extraction ───────────────────────────────────────────

def extract_timeseries(
    sae_id: str,
    feature_ids: list[int],
    data_dir: Path,
    aggregation: str = "max",
) -> tuple[np.ndarray, list[str]]:
    """Return (T, F) float32 matrix + list of timestamp strings."""
    import zarr
    zarr_root = data_dir / "activations" / sae_id
    idx_store = zarr.open(str(zarr_root / "indices.zarr"), mode="r")
    val_store = zarr.open(str(zarr_root / "values.zarr"), mode="r")
    timestamps = list(np.load(str(zarr_root / "timestamps.npy"), allow_pickle=True))

    n_time   = idx_store.shape[0]
    F        = len(feature_ids)
    feat_pos = {f: i for i, f in enumerate(feature_ids)}

    series = np.zeros((n_time, F), dtype=np.float32)

    print(f"Extracting time series for {F} features over {n_time} timesteps...")
    for t in range(n_time):
        if t % 100 == 0:
            print(f"  {t}/{n_time}", end="\r", flush=True)
        idxs = idx_store[t]   # (n_nodes, k_active)
        vals = val_store[t]   # (n_nodes, k_active)

        for feat_id, col in feat_pos.items():
            mask      = idxs == feat_id
            node_vals = vals[mask.any(axis=-1)]
            if node_vals.size == 0:
                continue
            if aggregation == "max":
                series[t, col] = node_vals.max()
            elif aggregation == "mean":
                series[t, col] = node_vals.mean()
            elif aggregation == "p95":
                series[t, col] = np.percentile(node_vals, 95)

    print(f"\nDone. Shape: {series.shape}")
    return series, timestamps


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Lagged cross-correlation (fast demo, no tigramite)
# ══════════════════════════════════════════════════════════════════════════════

def lagged_pearson(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """
    corr_at_lag[τ] = Pearson(x[:-τ], y[τ:]) for τ=1..max_lag.
    Returns array of shape (max_lag,).
    """
    from scipy.stats import pearsonr
    corrs = np.zeros(max_lag)
    for tau in range(1, max_lag + 1):
        if len(x) - tau < 3:
            corrs[tau - 1] = 0.0
            continue
        r, _ = pearsonr(x[:-tau], y[tau:])
        corrs[tau - 1] = r if np.isfinite(r) else 0.0
    return corrs


def run_lagged_corr(
    series: np.ndarray,       # (T, F)
    feature_ids: list[int],
    var_names: list[str],
    tau_max: int,
    threshold: float,
    out_dir: Path,
    focal_feature: int,
) -> dict:
    """
    Compute all pairwise lagged cross-correlations.

    For each ordered pair (i→j), we record the lag at which corr(i[t], j[t+τ])
    is largest in absolute value above `threshold`.  Both directions are tested
    independently so A→B and B→A can coexist if supported by the data.

    Returns a dict with edges and the full correlation matrix.
    """
    F = len(feature_ids)
    step_h = 6  # timestep in hours

    # Full correlation cube: corr_cube[i, j, tau-1] = corr(i[t], j[t+tau])
    corr_cube = np.zeros((F, F, tau_max), dtype=np.float32)
    for i in range(F):
        for j in range(F):
            if i == j:
                continue
            corr_cube[i, j] = lagged_pearson(series[:, i], series[:, j], tau_max)

    # Find best lag for each directed pair
    edges = []
    for i in range(F):
        for j in range(F):
            if i == j:
                continue
            best_tau_idx = np.argmax(np.abs(corr_cube[i, j]))
            best_r       = float(corr_cube[i, j, best_tau_idx])
            best_lag     = int(best_tau_idx + 1)       # 1-indexed steps
            if abs(best_r) >= threshold:
                edges.append({
                    "src": feature_ids[i],
                    "tgt": feature_ids[j],
                    "src_name": var_names[i],
                    "tgt_name": var_names[j],
                    "lag_steps": best_lag,
                    "lag_hours": best_lag * step_h,
                    "r": round(best_r, 4),
                })

    # Validation against known physical edges
    edge_lookup = {
        (e["src"], e["tgt"]): e for e in edges
    }
    validation = []
    for src, tgt, exp_lag, desc in KNOWN_EDGES:
        if src not in feature_ids or tgt not in feature_ids:
            status = "MISSING — feature not in graph"
            val_row = {"edge": desc, "status": status, "r": None, "lag": None}
        elif (src, tgt) in edge_lookup:
            e = edge_lookup[(src, tgt)]
            lag_match = e["lag_steps"] == exp_lag
            status = (
                f"RECOVERED ✓ (lag {e['lag_hours']}h)"
                if lag_match else
                f"RECOVERED — wrong lag (got {e['lag_hours']}h, expected {exp_lag*step_h}h)"
            )
            val_row = {"edge": desc, "status": status,
                       "r": e["r"], "lag": e["lag_hours"]}
        else:
            status = f"NOT SIGNIFICANT (r < {threshold})"
            val_row = {"edge": desc, "status": status, "r": None, "lag": None}
        validation.append(val_row)

    results = {
        "mode": "lagged_corr",
        "threshold": threshold,
        "tau_max": tau_max,
        "n_timesteps": int(series.shape[0]),
        "feature_ids": feature_ids,
        "var_names": var_names,
        "edges": edges,
        "corr_cube": corr_cube.tolist(),
        "validation": validation,
    }

    # Save JSON
    with open(out_dir / "lag_corr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Validation table
    val_df = pd.DataFrame(validation)
    val_df.to_csv(out_dir / "validation.txt", index=False, sep="\t")
    print("\nValidation against known edges:")
    print(val_df.to_string(index=False))

    n_ok = sum(1 for r in validation if "RECOVERED ✓" in r["status"])
    print(f"\nRecovered {n_ok}/{len(KNOWN_EDGES)} expected edges at threshold r≥{threshold}")

    return results


def plot_lag_graph(
    results: dict,
    focal_feature: int,
    out_path: Path,
) -> None:
    """
    Draw a directed network graph where:
      - Nodes are laid out with the focal feature at center, seeds around it.
      - Edge thickness/color encodes correlation strength |r|.
      - Edge label shows lag in hours.
      - Focal feature node is highlighted in orange.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    feature_ids = results["feature_ids"]
    var_names   = results["var_names"]
    edges       = results["edges"]
    F           = len(feature_ids)

    G = nx.DiGraph()
    for fid, name in zip(feature_ids, var_names):
        G.add_node(fid, label=name)
    for e in edges:
        G.add_edge(e["src"], e["tgt"],
                   r=e["r"], lag_h=e["lag_hours"],
                   weight=abs(e["r"]))

    # Layout: focal at center, others in a ring
    pos = {}
    others = [f for f in feature_ids if f != focal_feature]
    angles = np.linspace(0, 2 * np.pi, len(others), endpoint=False)
    for fid, angle in zip(others, angles):
        pos[fid] = (np.cos(angle) * 1.5, np.sin(angle) * 1.5)
    pos[focal_feature] = (0.0, 0.0)

    node_colors = [
        "#FF8C00" if fid == focal_feature else "#4A90D9"
        for fid in G.nodes()
    ]
    node_sizes = [2200 if fid == focal_feature else 1600 for fid in G.nodes()]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(
        f"Lagged Dependency Graph — focal: F{focal_feature} "
        f"(threshold |r|≥{results['threshold']}, τ_max={results['tau_max']*6}h)",
        fontsize=11,
    )

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.92)
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={fid: G.nodes[fid]["label"] for fid in G.nodes()},
        font_size=8, font_weight="bold",
    )

    # Draw edges colored by |r|
    edge_list = list(G.edges(data=True))
    if edge_list:
        r_vals = np.array([abs(d["r"]) for _, _, d in edge_list])
        r_norm = (r_vals - r_vals.min()) / (r_vals.max() - r_vals.min() + 1e-8)
        cmap   = plt.cm.Oranges
        colors = [cmap(0.4 + 0.6 * v) for v in r_norm]
        widths = [1.5 + 3.0 * v for v in r_norm]

        for (src, tgt, data), color, width in zip(edge_list, colors, widths):
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=[(src, tgt)],
                edge_color=[color], width=width,
                arrows=True, arrowsize=20,
                connectionstyle="arc3,rad=0.12",
                min_source_margin=25, min_target_margin=25,
            )

        edge_labels = {(e["src"], e["tgt"]): f'{e["lag_hours"]}h\nr={e["r"]:+.2f}'
                       for e in results["edges"]}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=7, label_pos=0.35,
        )

    focal_patch = mpatches.Patch(color="#FF8C00", label=f"F{focal_feature} (focal)")
    seed_patch  = mpatches.Patch(color="#4A90D9", label="Seed features")
    ax.legend(handles=[focal_patch, seed_patch], loc="lower right", fontsize=8)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_corr_heatmap(
    results: dict,
    tau_max: int,
    out_path: Path,
) -> None:
    """Heatmap of corr(i[t], j[t+τ]) for each pair and lag."""
    import matplotlib.pyplot as plt

    feature_ids = results["feature_ids"]
    var_names   = results["var_names"]
    cube        = np.array(results["corr_cube"])   # (F, F, tau_max)
    F           = len(feature_ids)
    step_h      = 6

    fig, axes = plt.subplots(1, tau_max, figsize=(3.5 * tau_max, 3.5))
    if tau_max == 1:
        axes = [axes]

    for tau_idx in range(tau_max):
        ax  = axes[tau_idx]
        mat = cube[:, :, tau_idx]
        im  = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"lag τ={(tau_idx+1)*step_h}h", fontsize=9)
        ax.set_xticks(range(F))
        ax.set_yticks(range(F))
        ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(var_names, fontsize=7)
        ax.set_xlabel("target j", fontsize=7)
        ax.set_ylabel("source i", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Cross-correlation:  corr(i[t], j[t+τ])", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: PCMCI+ + ANM (full cluster run, needs tigramite + causal-learn)
# ══════════════════════════════════════════════════════════════════════════════

def mi_screening(
    all_series: np.ndarray,
    focal_col: int,
    tau_max: int,
    n_candidates: int,
    n_neighbors: int = 5,
) -> list[int]:
    from sklearn.feature_selection import mutual_info_regression

    T, N = all_series.shape
    focal  = all_series[:, focal_col]
    scores = np.zeros(N)

    for tau in range(0, tau_max + 1):
        X = all_series if tau == 0 else all_series[:-tau]
        y = focal       if tau == 0 else focal[tau:]
        mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=0)
        scores = np.maximum(scores, mi)

    scores[focal_col] = -1.0
    return np.argsort(scores)[::-1][:n_candidates].tolist()


def run_pcmciplus(
    series: np.ndarray,
    var_names: list[str],
    tau_max: int,
    alpha: float,
    cond_ind_test: str,
) -> dict:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    if cond_ind_test == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        ci_test = ParCorr(significance="analytic")
    else:
        from tigramite.independence_tests.cmiknn import CMIknn
        ci_test = CMIknn(significance="shuffle_test", knn=10)

    dataframe = pp.DataFrame(
        series,
        datatime={0: np.arange(series.shape[0])},
        var_names=var_names,
    )
    pcmci   = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=1)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha)
    return results


def anm_orient(x: np.ndarray, y: np.ndarray) -> tuple[str, float, float]:
    try:
        from causallearn.search.FCMBased.ANM.ANM import ANM
        anm = ANM()
        p_xy, _ = anm.cause_or_effect(x.reshape(-1, 1), y.reshape(-1, 1))
        p_yx, _ = anm.cause_or_effect(y.reshape(-1, 1), x.reshape(-1, 1))
        if p_xy > 0.05 and p_yx <= 0.05:
            return "x->y", p_xy, p_yx
        elif p_yx > 0.05 and p_xy <= 0.05:
            return "y->x", p_xy, p_yx
        return "inconclusive", p_xy, p_yx
    except Exception as e:
        warnings.warn(f"ANM failed: {e}")
        return "inconclusive", float("nan"), float("nan")


def validate_pcmci(
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    feature_ids: list[int],
    alpha: float,
) -> list[dict]:
    feat_pos = {f: i for i, f in enumerate(feature_ids)}
    rows = []
    for src, tgt, lag, desc in KNOWN_EDGES:
        if src not in feat_pos or tgt not in feat_pos:
            rows.append({"edge": desc, "status": "MISSING", "val": None, "p": None})
            continue
        i, j = feat_pos[src], feat_pos[tgt]
        if lag > val_matrix.shape[2] - 1:
            rows.append({"edge": desc, "status": "lag out of range", "val": None, "p": None})
            continue
        val  = float(val_matrix[i, j, lag])
        pval = float(p_matrix[i, j, lag])
        status = ("RECOVERED ✓" if pval < alpha and val > 0
                  else "RECOVERED (negative)" if pval < alpha
                  else "NOT SIGNIFICANT")
        rows.append({"edge": desc, "src": src, "tgt": tgt, "lag": lag,
                     "status": status, "val": round(val, 4), "p": round(pval, 4)})
    return rows


def plot_pcmci_graph(
    val_matrix, p_matrix, feature_ids, var_names,
    tau_max, alpha, focal_feature, out_path,
):
    import matplotlib.pyplot as plt

    N = len(feature_ids)
    fig, axes = plt.subplots(1, tau_max + 1,
                             figsize=(4 * (tau_max + 1), 4), squeeze=False)
    fig.suptitle(f"PCMCI+ lag graph — focal F{focal_feature}", fontsize=12)
    focal_col = feature_ids.index(focal_feature) if focal_feature in feature_ids else -1

    for tau in range(tau_max + 1):
        ax  = axes[0, tau]
        mat = np.where(p_matrix[:, :, tau] < alpha, val_matrix[:, :, tau], np.nan)
        im  = ax.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_title(f"lag τ={tau} ({tau*6}h)")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(var_names, rotation=90, fontsize=7)
        ax.set_yticklabels(var_names, fontsize=7)
        if focal_col >= 0:
            for v in (focal_col - 0.5, focal_col + 0.5):
                ax.axhline(v, color="orange", lw=1.5, alpha=0.6)
                ax.axvline(v, color="orange", lw=1.5, alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Lagged dependency / causal graph from SAE features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", default="lagged_corr",
                   choices=["lagged_corr", "pcmciplus"],
                   help="lagged_corr=fast demo (32 ts); pcmciplus=full cluster (1464 ts)")
    p.add_argument("--focal_feature",    type=int, default=3243)
    p.add_argument("--seed_features",    default="117,402,911,2088",
                   help="Comma-separated feature IDs (always included)")
    p.add_argument("--n_mi_candidates",  type=int, default=20,
                   help="[pcmciplus only] Additional features from MI screening")
    p.add_argument("--tau_max",          type=int, default=4,
                   help="Max lag in 6h steps (4 = 24h)")
    p.add_argument("--threshold",        type=float, default=0.30,
                   help="[lagged_corr] Min |r| to draw an edge")
    p.add_argument("--alpha",            type=float, default=0.05,
                   help="[pcmciplus] Significance level")
    p.add_argument("--cond_ind_test",    default="parcorr",
                   choices=["parcorr", "cmiknn"])
    p.add_argument("--aggregation",      default="max",
                   choices=["max", "mean", "p95"])
    p.add_argument("--sae_id",           default="layer8_k32_d4096")
    p.add_argument("--data_dir",         default="viz/data")
    p.add_argument("--out_dir",          default="results/causal")
    p.add_argument("--skip_anm",         action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_features = [int(x) for x in args.seed_features.split(",") if x.strip()]
    focal         = args.focal_feature

    # Load feature labels from catalog if available
    labels = dict(FEATURE_LABELS)
    catalog_path = data_dir / "feature_catalog.parquet"
    if catalog_path.exists():
        cat = pd.read_parquet(catalog_path)
        for _, row in cat.iterrows():
            if row.get("candidate_label"):
                labels[int(row["feature_id"])] = str(row["candidate_label"])

    # ── Lagged correlation mode ───────────────────────────────────────────────
    if args.mode == "lagged_corr":
        feature_ids = list(dict.fromkeys([focal] + seed_features))
        var_names   = [labels.get(f, f"F{f}") for f in feature_ids]

        print(f"Mode: lagged_corr  |  features: {feature_ids}")
        print(f"Threshold |r| >= {args.threshold}  |  tau_max = {args.tau_max} steps ({args.tau_max*6}h)")

        series, timestamps = extract_timeseries(
            args.sae_id, feature_ids, data_dir, args.aggregation
        )
        T = series.shape[0]
        print(f"T = {T} timesteps")

        results = run_lagged_corr(
            series, feature_ids, var_names,
            tau_max=args.tau_max,
            threshold=args.threshold,
            out_dir=out_dir,
            focal_feature=focal,
        )
        plot_lag_graph(results, focal, out_dir / "lag_dep_graph.png")
        plot_corr_heatmap(results, args.tau_max, out_dir / "lag_corr_heatmap.png")

        print(f"\nOutputs in {out_dir}/")
        print("  lag_dep_graph.png    — network graph")
        print("  lag_corr_heatmap.png — correlation heatmap by lag")
        print("  lag_corr_results.json")
        print("  validation.txt")
        return

    # ── PCMCI+ mode ───────────────────────────────────────────────────────────
    _require("tigramite")
    import zarr, json as _json
    zarr_root = data_dir / "activations" / args.sae_id
    with open(data_dir / "sae_registry.json") as f:
        info = _json.load(f)[args.sae_id]

    n_features = info["latent"]
    print(f"Mode: pcmciplus  |  {info['n_time']} timesteps, {n_features} features")

    print("\n=== Pass 1: All features (MI screening) ===")
    all_series, timestamps = extract_timeseries(
        args.sae_id, list(range(n_features)), data_dir, args.aggregation
    )

    print("\n=== Step 2: MI screening ===")
    mi_top = mi_screening(all_series, focal_col=focal,
                          tau_max=args.tau_max, n_candidates=args.n_mi_candidates)
    print(f"Top-{args.n_mi_candidates} MI features: {mi_top}")

    selected      = list(dict.fromkeys([focal] + seed_features + [int(i) for i in mi_top]))
    selected_s    = all_series[:, selected]
    var_names     = [labels.get(f, f"F{f}") for f in selected]
    T, N          = selected_s.shape
    print(f"\n{N} nodes, T={T}, T/p={T/N:.1f}")
    if T / N < 10:
        warnings.warn(f"T/p={T/N:.1f} — results unreliable; increase timesteps or reduce --n_mi_candidates")

    print("\n=== Step 3: PCMCI+ ===")
    pcmci_res  = run_pcmciplus(selected_s, var_names, args.tau_max, args.alpha, args.cond_ind_test)
    val_matrix = pcmci_res["val_matrix"]
    p_matrix   = pcmci_res["p_matrix"]
    graph      = pcmci_res.get("graph", None)

    anm_results: list[dict] = []
    if not args.skip_anm and graph is not None:
        print("\n=== Step 4: ANM orientation ===")
        for i in range(N):
            for j in range(i + 1, N):
                if "o" in str(graph[i, j, 0]) or "o" in str(graph[j, i, 0]):
                    direction, pxy, pyx = anm_orient(selected_s[:, i], selected_s[:, j])
                    anm_results.append({
                        "var_i": var_names[i], "feat_i": selected[i],
                        "var_j": var_names[j], "feat_j": selected[j],
                        "direction": direction, "p_ij": round(pxy, 4), "p_ji": round(pyx, 4),
                    })
                    print(f"  {var_names[i]} ↔ {var_names[j]}: ANM → {direction}")

    print("\n=== Step 5: Validation ===")
    validation = validate_pcmci(val_matrix, p_matrix, selected, args.alpha)
    for r in validation:
        print(f"  [{r['status']}]  {r['edge']}  val={r['val']}  p={r['p']}")

    plot_pcmci_graph(val_matrix, p_matrix, selected, var_names,
                     args.tau_max, args.alpha, focal, out_dir / "causal_graph.png")

    results = {
        "mode": "pcmciplus",
        "config": vars(args),
        "selected_features": selected,
        "var_names": var_names,
        "n_timesteps": int(T),
        "t_over_p": float(T / N),
        "val_matrix": val_matrix.tolist(),
        "p_matrix": p_matrix.tolist(),
        "mi_top_features": [int(i) for i in mi_top],
        "anm_results": anm_results,
        "validation": validation,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    val_df = pd.DataFrame(validation)
    val_df.to_csv(out_dir / "validation.txt", index=False, sep="\t")
    n_ok = sum(1 for r in validation if "RECOVERED ✓" in r["status"])
    print(f"\nRecovered {n_ok}/{len(KNOWN_EDGES)} expected edges  |  Results: {out_dir}/")


if __name__ == "__main__":
    main()
