"""
ERA5 Feature Correlation Analysis

Pixel-wise Pearson r between SAE feature activation maps and ERA5 physical fields.
Implements the spatial correlation workflow from docs/era5_feature_correlation.md.

Usage:
    python scripts/era5_feature_correlation.py \
        --feature 3243 \
        --sae_id layer8_k32_d4096 \
        --data_dir viz/data \
        --era5_dir data/era5_daily \
        --out_dir results/era5_correlation
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from graphcast_interpretability.interpolate_utils import GridMeshMapper


# ── Priority ERA5 fields from docs/era5_feature_correlation.md ───────────────
ERA5_TARGETS = [
    ("mean_sea_level_pressure", None,  "MSLP"),
    ("geopotential",            850,   "GPH 850 hPa"),
    ("geopotential",            500,   "GPH 500 hPa"),
    ("temperature",             300,   "T 300 hPa"),
    ("specific_humidity",       850,   "q 850 hPa"),
    ("specific_humidity",       700,   "q 700 hPa"),
    ("u_component_of_wind",     850,   "U 850 hPa"),
    ("v_component_of_wind",     850,   "V 850 hPa"),
    ("u_component_of_wind",     200,   "U 200 hPa"),
    ("v_component_of_wind",     200,   "V 200 hPa"),
    ("vertical_velocity",       500,   "ω 500 hPa"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pearson_r_maps(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Vectorised pixel-wise Pearson r.
    X, Y: (T, lat, lon) float32
    Returns: (lat, lon) float32 in [-1, 1]
    """
    X_z = X - X.mean(0, keepdims=True)
    X_z /= X_z.std(0, keepdims=True) + 1e-8
    Y_z = Y - Y.mean(0, keepdims=True)
    Y_z /= Y_z.std(0, keepdims=True) + 1e-8
    return (X_z * Y_z).mean(0).astype(np.float32)


def _plot_r_map(r_map, lat, lon, title, out_path):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.5)

    img = ax.pcolormesh(
        lon, lat, r_map,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=-1, vmax=1,
        rasterized=True,
    )
    plt.colorbar(img, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7,
                 label="Pearson r (pixel-wise over T timesteps)")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _load_era5_field(era5_dir: Path, timestamps: list[str],
                     variable: str, level: int | None) -> np.ndarray:
    """
    Load ERA5 field for the given timestamps → (T, lat, lon) float32.
    Timestamps are strings like '2021-09-04T12'.
    """
    arrays = []
    for ts in timestamps:
        date = ts[:10]  # YYYY-MM-DD
        hour = int(ts[11:13])
        nc_path = era5_dir / f"era5_{date}.nc"
        if not nc_path.exists():
            raise FileNotFoundError(f"Missing ERA5 file: {nc_path}")
        ds = xr.open_dataset(nc_path)
        da = ds[variable]
        # select time step by hour
        t_idx = hour // 6
        if level is not None:
            arr = da.isel(time=t_idx).sel(level=level).values.astype(np.float32)
        else:
            arr = da.isel(time=t_idx).values.astype(np.float32)
        arrays.append(arr)
        ds.close()
    return np.stack(arrays, axis=0)  # (T, lat, lon)


def _extract_feature_maps(
    zarr_root: Path,
    feature_id: int,
    mapper: GridMeshMapper,
) -> tuple[np.ndarray, list[str]]:
    """
    Load feature activations from zarr, scatter-max to mesh nodes,
    interpolate to grid → (T, lat, lon) float32.
    """
    idx_store = zarr.open(str(zarr_root / "indices.zarr"))
    val_store = zarr.open(str(zarr_root / "values.zarr"))
    timestamps = list(np.load(str(zarr_root / "timestamps.npy"), allow_pickle=True))
    T, N, K = idx_store.shape

    print(f"Extracting feature {feature_id} across {T} timesteps ({N} nodes)…")
    grid_lat = mapper.grid_lat
    grid_lon = mapper.grid_lon
    act_maps = np.zeros((T, len(grid_lat), len(grid_lon)), dtype=np.float32)

    for t in range(T):
        if t % 20 == 0:
            print(f"  t={t}/{T}")
        idxs = idx_store[t].astype(np.int32)   # (N, K)
        vals = val_store[t]                      # (N, K)

        # scatter-max: for each node, take max activation if feature fires there
        node_act = np.zeros(N, dtype=np.float32)
        mask = idxs == feature_id               # (N, K) bool
        for k in range(K):
            fired = mask[:, k]
            node_act[fired] = np.maximum(node_act[fired], vals[fired, k])

        act_maps[t] = mapper.apply_mesh_to_grid(node_act)

    return act_maps, timestamps


def _temporal_correlation_table(
    act_maps: np.ndarray, timestamps: list[str],
    era5_dir: Path,
) -> list[tuple[str, float]]:
    """
    Temporal correlation: max activation over globe vs ERA5 spatial statistics.
    Returns list of (label, r) sorted by |r| descending.
    """
    act_ts = act_maps.max(axis=(1, 2))  # (T,) scalar time series

    results = []
    for var, level, label in ERA5_TARGETS:
        try:
            era5_field = _load_era5_field(era5_dir, timestamps, var, level)
        except FileNotFoundError as e:
            print(f"  skip {label}: {e}")
            continue
        # spatial max as scalar time series
        era5_ts = era5_field.max(axis=(1, 2)) if level is None else era5_field.max(axis=(1, 2))
        r = np.corrcoef(act_ts, era5_ts)[0, 1]
        results.append((label, float(r)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", type=int, default=3243)
    p.add_argument("--sae_id", default="layer8_k32_d4096")
    p.add_argument("--data_dir", default="viz/data")
    p.add_argument("--era5_dir", default="data/era5_daily")
    p.add_argument("--out_dir", default="results/era5_correlation")
    p.add_argument("--mapper_cache", default="viz/data/graph/mapper_cache")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_root = Path(args.data_dir) / "activations" / args.sae_id
    era5_dir = Path(args.era5_dir)

    # ── Build GridMeshMapper ──────────────────────────────────────────────
    grid_lat = np.load(f"{args.data_dir}/graph/grid_lat.npy")
    grid_lon = np.load(f"{args.data_dir}/graph/grid_lon.npy")
    mapper = GridMeshMapper(
        grid_lat=grid_lat, grid_lon=grid_lon,
        splits=6, cache_dir=args.mapper_cache,
    )
    # ensure mesh2grid mapping is built
    mapper.precompute_mesh_to_grid()

    # ── Extract activation maps ───────────────────────────────────────────
    act_maps, timestamps = _extract_feature_maps(zarr_root, args.feature, mapper)
    T = len(timestamps)
    print(f"\nActivation maps: {act_maps.shape}, T={T}, range [{act_maps.min():.3f}, {act_maps.max():.3f}]")
    print(f"Timestamps: {timestamps[0]} → {timestamps[-1]}")

    # log1p transform (sparse activations are right-skewed)
    act_maps_tr = np.log1p(act_maps)

    # ── Spatial correlation maps ──────────────────────────────────────────
    print("\n=== Spatial Correlation Maps ===")
    r_summary = []
    for var, level, label in ERA5_TARGETS:
        print(f"  {label}…", end=" ", flush=True)
        try:
            era5_field = _load_era5_field(era5_dir, timestamps, var, level)
        except FileNotFoundError as e:
            print(f"SKIP ({e})")
            continue

        r_map = _pearson_r_maps(act_maps_tr, era5_field)
        r_abs_max = np.abs(r_map).max()
        r_mean = r_map.mean()
        print(f"max|r|={r_abs_max:.3f}, mean_r={r_mean:.3f}")
        r_summary.append((label, r_abs_max, r_mean))

        slug = label.replace(" ", "_").replace("/", "").replace("ω", "omega")
        out_path = out_dir / f"spatial_r_F{args.feature}_{slug}.pdf"
        title = f"Feature {args.feature} × {label}  [T={T} timesteps, log1p activations]"
        _plot_r_map(r_map, grid_lat, grid_lon, title, out_path)

    # ── Summary plot ─────────────────────────────────────────────────────
    print("\n=== Summary: max |r| by ERA5 field ===")
    r_summary.sort(key=lambda x: x[1], reverse=True)
    for label, max_r, mean_r in r_summary:
        print(f"  {label:<20} max|r|={max_r:.3f}  mean_r={mean_r:+.3f}")

    # bar chart
    labels_s = [x[0] for x in r_summary]
    maxr_s   = [x[1] for x in r_summary]
    meanr_s  = [x[2] for x in r_summary]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels_s))
    bars = ax.bar(x, maxr_s, color=[
        "steelblue" if r >= 0 else "tomato" for r in meanr_s
    ], alpha=0.8)
    ax.axhline(0.4, color="k", linestyle="--", linewidth=0.8, label="|r|=0.4 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_s, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Max pixel-wise |r|")
    ax.set_title(f"Feature {args.feature}: Peak spatial correlation with ERA5 fields (T={T})")
    ax.legend()
    plt.tight_layout()
    summary_path = out_dir / f"summary_F{args.feature}.pdf"
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    print(f"\nSummary saved: {summary_path}")

    # ── Temporal correlation table ────────────────────────────────────────
    print("\n=== Temporal Correlation (global max activation vs ERA5 spatial max) ===")
    temp_results = _temporal_correlation_table(act_maps_tr, timestamps, era5_dir)
    for label, r in temp_results:
        print(f"  {label:<20}  r={r:+.3f}")


if __name__ == "__main__":
    main()
