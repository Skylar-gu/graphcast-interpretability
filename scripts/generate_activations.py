"""
Generate GraphCast layer-8 activation .npy files.

Downloads ERA5 data and GraphCast checkpoint from public GCS buckets,
runs GraphCast with activation hooks on layer 8, and writes:

    <acts_dir>/layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t<TIMESTAMP>.npy

Usage:
    python scripts/generate_activations.py \
        --start 2021-08-29 --end 2021-08-30 \
        --acts_dir data/activations_raw

Default case reproduces the Hurricane Ida (2021-08-29) demo.
"""

from __future__ import annotations
import argparse
import dataclasses
import functools
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on the path regardless of editable-install .pth loading
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import xarray as xr
import jax
import haiku as hk
import gcsfs
from google.cloud import storage

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
)
from graphcast.deep_typed_graph_net import get_activation_manager


# ── ERA5 loading ──────────────────────────────────────────────────────────────

ERA5_VARS = [
    "geopotential", "specific_humidity", "temperature",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr",
    "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask",
]

ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2022-full_37-6h-0p25deg_derived.zarr"
)


def load_era5(start: str, end: str) -> xr.Dataset:
    print(f"Downloading ERA5 {start} → {end} from GCS (anonymous)…")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ERA5_ZARR[5:])
    ds = xr.open_zarr(store, consolidated=True)

    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])

    ds = ds.sel(time=slice(np.datetime64(start), np.datetime64(end)))
    ds = ds[[v for v in ERA5_VARS if v in ds.data_vars]]
    ds = ds.load()
    print(f"  ERA5 loaded: {dict(ds.dims)}")
    return ds


def write_daily_nc(ds: xr.Dataset, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for day, ds_day in ds.groupby("time.date"):
        day_str = str(day)[:10]
        path = os.path.join(out_dir, f"era5_{day_str}.nc")
        ds_day.to_netcdf(path)
        print(f"  Wrote {path}")


# ── Three-step windowing (copied faithfully from notebook) ────────────────────

def _open_and_trim(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if "time" in ds.dims and ds.sizes["time"] > 4:
        ds = ds.isel(time=slice(0, 4))
    return ds


def three_step_window(data_dir: str, center_time: str) -> xr.Dataset | None:
    t0      = np.datetime64(center_time)
    t_minus = t0 - np.timedelta64(6, "h")
    t_plus  = t0 + np.timedelta64(6, "h")

    needed_days = sorted({
        np.datetime64(t_minus, "D"),
        np.datetime64(t0, "D"),
        np.datetime64(t_plus, "D"),
    })
    file_paths = [
        os.path.join(data_dir, f"era5_{str(d)[:10]}.nc")
        for d in needed_days
    ]
    if any(not os.path.exists(p) for p in file_paths):
        return None

    daily = [_open_and_trim(p) for p in file_paths]
    var_time   = [v for v, da in daily[0].data_vars.items() if "time" in da.dims]
    var_static = [v for v, da in daily[0].data_vars.items() if "time" not in da.dims]

    ds_time   = xr.concat([d[var_time] for d in daily], dim="time").sortby("time")
    ds_static = daily[0][var_static]
    ds = xr.merge([ds_time, ds_static])

    target_times = np.array([t_minus, t0, t_plus], dtype=ds.time.dtype)
    if not all(t in ds.time.values for t in target_times):
        return None
    ds = ds.sel(time=target_times)

    # Add batch dimension
    ds_new = ds.copy()
    for v in ds_new.data_vars:
        if "time" in ds_new[v].dims:
            ds_new[v] = ds_new[v].expand_dims("batch")
    for c in ds.coords:
        if "time" in ds[c].dims:
            ds_new = ds_new.assign_coords({c: ds[c].expand_dims("batch")})

    time_orig  = ds["time"]
    t_ref      = time_orig.values[0]
    time_delta = time_orig - t_ref
    ds_new = ds_new.assign_coords(time=time_delta)
    ds_new = ds_new.assign_coords(datetime=("time", time_orig.values))
    ds_new = ds_new.assign_coords(
        {"datetime": ds_new["datetime"].expand_dims("batch")}
    )
    return ds_new


# ── GraphCast model loading ───────────────────────────────────────────────────

GCS_BUCKET = "dm_graphcast"
GCS_PREFIX = "graphcast/"
MODEL_FILE = (
    "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 "
    "- mesh 2to6 - precipitation input and output.npz"
)


def load_graphcast_from_gcs():
    print("Downloading GraphCast checkpoint from GCS…")
    gcs = storage.Client.create_anonymous_client()
    bucket = gcs.get_bucket(GCS_BUCKET)

    with bucket.blob(f"{GCS_PREFIX}params/{MODEL_FILE}").open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    print("Downloading normalization stats…")
    stats = {}
    for name in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level"):
        with bucket.blob(f"{GCS_PREFIX}stats/{name}.nc").open("rb") as f:
            stats[name] = xr.load_dataset(f).compute()

    return ckpt, stats


# ── GraphCast JIT forward ─────────────────────────────────────────────────────

def build_jit_forward(ckpt, stats,
                      mesh_sae_params=None,
                      mesh_sae_steps=None,
                      mesh_sae_node_sets=None,
                      mesh_sae_alpha=None):
    model_config = ckpt.model_config
    task_config  = ckpt.task_config
    params       = ckpt.params
    state        = {}

    def construct(mc, tc):
        # SAEInjector is an hk.Module — must be constructed inside hk.transform
        injector = None
        if mesh_sae_params is not None:
            from graphcast.deep_typed_graph_net import SAEInjector
            injector = SAEInjector(mesh_sae_params)
        pred = graphcast.GraphCast(mc, tc,
                                   mesh_sae_injector=injector,
                                   mesh_sae_steps=mesh_sae_steps,
                                   mesh_sae_node_sets=mesh_sae_node_sets,
                                   mesh_sae_alpha=mesh_sae_alpha)
        pred = casting.Bfloat16Cast(pred)
        pred = normalization.InputsAndResiduals(
            pred,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"],
            stddev_by_level=stats["stddev_by_level"],
        )
        pred = autoregressive.Predictor(pred, gradient_checkpointing=True)
        return pred

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        return construct(model_config, task_config)(
            inputs, targets_template=targets_template, forcings=forcings
        )

    run_jit = jax.jit(
        functools.partial(run_forward.apply, model_config=model_config, task_config=task_config)
    )

    def forward(**kw):
        return run_jit(params=params, state=state, **kw)[0]

    return forward, task_config


# ── Main ─────────────────────────────────────────────────────────────────────

def load_graphcast_cached(cache_dir: str):
    """Download GraphCast checkpoint and stats, caching locally to avoid re-download."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache / "graphcast_ckpt.npz"
    gcs = storage.Client.create_anonymous_client()
    bucket = gcs.get_bucket(GCS_BUCKET)

    if not ckpt_path.exists():
        print("Downloading GraphCast checkpoint from GCS (~500 MB, once only)…")
        bucket.blob(f"{GCS_PREFIX}params/{MODEL_FILE}").download_to_filename(str(ckpt_path))
    else:
        print(f"Using cached GraphCast checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    stats = {}
    for name in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level"):
        stat_path = cache / f"{name}.nc"
        if not stat_path.exists():
            print(f"  Downloading {name}…")
            bucket.blob(f"{GCS_PREFIX}stats/{name}.nc").download_to_filename(str(stat_path))
        with open(stat_path, "rb") as f:
            stats[name] = xr.load_dataset(f).compute()

    return ckpt, stats


# Named presets: (era5_start, era5_end, center_start, center_end)
# Hurricane seasons chosen because Feature 3243 (TC tracker) fires primarily
# during Atlantic/Pacific TC activity. 2020 and 2021 were both very active seasons.
PRESETS: dict[str, tuple[str, str, str, str]] = {
    # Single landfall day (4 timesteps) — quick smoke-test
    "hurricane_ida":          ("2021-08-28", "2021-08-30", "2021-08-29T00", "2021-08-30T00"),
    # Full Ida week (32 timesteps) — proof-of-concept demo
    "hurricane_ida_week":     ("2021-08-23", "2021-09-01", "2021-08-24T00", "2021-09-01T00"),
    # Full Atlantic hurricane seasons — cluster / causal analysis
    "hurricane_season_2020":  ("2020-05-31", "2020-12-01", "2020-06-01T00", "2020-12-01T00"),
    "hurricane_season_2021":  ("2021-05-31", "2021-12-01", "2021-06-01T00", "2021-12-01T00"),
    "hurricane_seasons_both": ("2020-05-31", "2021-12-01", "2020-06-01T00", "2021-12-01T00"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate GraphCast layer-8 activations")
    p.add_argument("--preset", choices=list(PRESETS.keys()),
                   help="Named date-range preset (overrides --start/--end/--center_start/--center_end)")
    p.add_argument("--start", default="2021-08-28",
                   help="ERA5 download start (needs day before first center)")
    p.add_argument("--end",   default="2021-08-30",
                   help="ERA5 download end (needs day after last center)")
    p.add_argument("--center_start", default="2021-08-29T00",
                   help="First GraphCast center time (ISO, UTC)")
    p.add_argument("--center_end",   default="2021-08-30T00",
                   help="Exclusive end for center times")
    p.add_argument("--step_hours",   type=int, default=6)
    p.add_argument("--acts_dir",  default="data/activations_raw",
                   help="Where to write layer*.npy files")
    p.add_argument("--era5_dir",  default="data/era5_daily_nc",
                   help="Where to cache daily ERA5 .nc files")
    p.add_argument("--ckpt_cache", default="data/graphcast_cache",
                   help="Local cache for GraphCast checkpoint and stats")
    p.add_argument("--layer",     type=int, default=8)

    # ── SAE intervention (optional) ───────────────────────────────────────
    p.add_argument("--ablate_feature", type=int, default=None,
                   help="SAE feature ID to zero-ablate during the forward pass")
    p.add_argument("--steer_feature",  type=int, default=None,
                   help="SAE feature ID to steer during the forward pass")
    p.add_argument("--steer_strength", type=float, default=1.0,
                   help="Steering multiplier added to alpha for --steer_feature "
                        "(e.g. 2.0 triples the feature, -1.0 ablates it)")
    p.add_argument("--sae_ckpt", default=None,
                   help="Path to SAE .pt checkpoint (default: auto-download from HuggingFace)")
    p.add_argument("--sae_cache", default="data/sae_cache",
                   help="Directory for cached SAE checkpoint")
    args = p.parse_args()

    if args.preset:
        args.start, args.end, args.center_start, args.center_end = PRESETS[args.preset]
        print(f"Preset '{args.preset}': {args.center_start} → {args.center_end}")

    # ── JAX persistent compilation cache ──────────────────────────────────
    # Survives restarts — avoids recompiling the full GraphCast graph each run.
    jax_cache = Path(args.ckpt_cache) / "jax_compile_cache"
    jax_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(jax_cache))
    print(f"JAX compilation cache: {jax_cache}")

    # ── ERA5 data (skip days already on disk) ──────────────────────────────
    era5_dir = args.era5_dir
    Path(era5_dir).mkdir(parents=True, exist_ok=True)
    Path(args.acts_dir).mkdir(parents=True, exist_ok=True)

    needed_days = set(
        str(d)[:10] for d in np.arange(
            np.datetime64(args.start), np.datetime64(args.end) + np.timedelta64(1, "D"),
            np.timedelta64(1, "D")
        )
    )
    missing_days = [d for d in sorted(needed_days)
                    if not (Path(era5_dir) / f"era5_{d}.nc").exists()]
    if missing_days:
        print(f"Downloading ERA5 for {len(missing_days)} missing days (in monthly chunks)…")
        # Group missing days into monthly buckets to avoid loading the full
        # date range into RAM at once (~3.8 GB/day × 90 days > 300 GB).
        from itertools import groupby
        def _ym(d): return d[:7]  # "YYYY-MM"
        for month, days_iter in groupby(missing_days, key=_ym):
            days = list(days_iter)
            # Extend the fetch window one day on each side for windowing overlap
            chunk_start = str(np.datetime64(days[0])  - np.timedelta64(1, "D"))[:10]
            chunk_end   = str(np.datetime64(days[-1]) + np.timedelta64(1, "D"))[:10]
            print(f"  Fetching {month} ({len(days)} days): {chunk_start} → {chunk_end}")
            ds_chunk = load_era5(chunk_start, chunk_end)
            write_daily_nc(ds_chunk, era5_dir)
            del ds_chunk
    else:
        print(f"All ERA5 daily files already cached in {era5_dir}/")

    # ── GraphCast (cached locally) ─────────────────────────────────────────
    ckpt, stats = load_graphcast_cached(args.ckpt_cache)

    # ── SAE intervention (optional) ───────────────────────────────────────
    sae_params     = None
    mesh_sae_alpha = None
    intervening    = args.ablate_feature is not None or args.steer_feature is not None

    if intervening:
        import jax.numpy as jnp
        from graphcast_interpretability.model import load_sae_params_from_torch

        # Resolve SAE checkpoint
        sae_ckpt_path = args.sae_ckpt
        if sae_ckpt_path is None:
            from huggingface_hub import hf_hub_download
            sae_ckpt_path = hf_hub_download(
                "theodoremacmillan/sae-graphcast-k32-lat4096-lay08",
                "sae_step0334221_t2300M.pt",
                local_dir=args.sae_cache,
            )
        print(f"SAE checkpoint: {sae_ckpt_path}")

        sae_params = load_sae_params_from_torch(
            sae_ckpt_path, unit_norm_decoder=True, k_active=32
        )
        # Cast to bfloat16 to match GraphCast's compute dtype — SAEInjector output
        # inherits the dtype of its weights, and Bfloat16Cast rejects float32 outputs.
        sae_params = dataclasses.replace(
            sae_params,
            enc_w=jnp.asarray(sae_params.enc_w, dtype=jnp.bfloat16),
            dec_w=jnp.asarray(sae_params.dec_w, dtype=jnp.bfloat16),
            b_pre=jnp.asarray(sae_params.b_pre, dtype=jnp.bfloat16),
        )

        # Build alpha: zeros = no change; set target feature(s)
        # Must be bfloat16 to match GraphCast's compute dtype — float32 alpha
        # promotes intermediate tensors and Bfloat16Cast rejects float32 output.
        latent = sae_params.enc_w.shape[1]
        alpha  = jnp.zeros(latent, dtype=jnp.bfloat16)

        if args.ablate_feature is not None:
            alpha = alpha.at[args.ablate_feature].set(-1.0)
            print(f"Intervention: ablate feature {args.ablate_feature} (alpha=-1)")

        if args.steer_feature is not None:
            alpha = alpha.at[args.steer_feature].add(args.steer_strength)
            print(f"Intervention: steer feature {args.steer_feature} "
                  f"(alpha+={args.steer_strength})")

        mesh_sae_alpha = alpha

    run_forward, task_config = build_jit_forward(
        ckpt, stats,
        mesh_sae_params=sae_params,
        mesh_sae_steps=[args.layer],
        mesh_sae_node_sets=["mesh_nodes"],
        mesh_sae_alpha=mesh_sae_alpha,
    )

    # ── Activation manager ────────────────────────────────────────────────
    am = get_activation_manager()
    am.__init__(
        enabled=True,
        save_dir=args.acts_dir,
        save_steps=[args.layer],
        save_node_sets=["mesh_nodes"],
        mode="post_res",
    )

    # ── Run loop ──────────────────────────────────────────────────────────
    centers = np.arange(
        np.datetime64(args.center_start),
        np.datetime64(args.center_end),
        np.timedelta64(args.step_hours, "h"),
    )
    print(f"\nRunning {len(centers)} timesteps…")

    t_start = time.time()
    done = 0
    for center in centers:
        center_str = np.datetime_as_string(center, unit="h")

        # Skip if output already exists (resume after crash)
        expected = Path(args.acts_dir) / (
            f"layer{args.layer:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t{center_str}.npy"
        )
        if expected.exists():
            print(f"[{center_str}] SKIP (already done)")
            done += 1
            continue

        print(f"[{center_str}] windowing…", end=" ", flush=True)
        ds = three_step_window(era5_dir, center_str)
        if ds is None:
            print("SKIP (missing adjacent day)")
            continue

        am.set_time(center_str)

        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            ds,
            target_lead_times=slice("6h", "6h"),
            **dataclasses.asdict(task_config),
        )

        t0 = time.time()
        _ = rollout.chunked_prediction(
            run_forward,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets * np.nan,
            forcings=forcings,
        )
        elapsed = time.time() - t0
        done += 1
        print(f"done ({elapsed:.1f}s)")

    total = time.time() - t_start
    print(f"\nFinished {done}/{len(centers)} timesteps in {total:.1f}s")
    print(f"Activations saved to: {args.acts_dir}/")

    # List output files
    npy_files = sorted(Path(args.acts_dir).glob(f"layer{args.layer:04d}_*.npy"))
    print(f"Output files ({len(npy_files)}):")
    for f in npy_files:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
