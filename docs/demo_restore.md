# Restoring the SAE Feature Atlas demo

The demo (`viz/app.py`) was originally built on a Lambda cluster at
`/lambda/nfs/demo/graphcast-interpretability`, which is gone. This document
records where every input came from and how to bring the demo back on a fresh
machine, either by rehydrating from S3 (minutes) or by regenerating from the
public sources (hours).

**S3 home:** `s3://interp-990002768295/graphcast-demo/` (us-east-1)

---

## What the demo is

Streamlit app, four tabs:

| Tab | Needs |
|---|---|
| Feature Inspector | sparse SAE zarr, mapper cache, `era5-surface/` (wind arrows, 2m temp) |
| Feature Map | sparse SAE zarr, mapper cache |
| Feature Atlas | `feature_catalog.parquet` |
| Event Browser | all of the above |

Focal feature is **3243** (tropical cyclone tracker). The labelled features shown
in the sidebar dropdown live in `KNOWN_LABELS` in `viz/compute_stats.py` — edit
there and re-run `compute_stats.py` to change them.

---

## Provenance of every input

| Input | Source | Notes |
|---|---|---|
| ERA5 | `gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr` | public, anonymous; 14 vars, 37 levels, 6-hourly, 0.25° |
| GraphCast params + norm stats | `gs://dm_graphcast/graphcast/` | `GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz` |
| SAE checkpoint | HF `theodoremacmillan/sae-graphcast-k32-lat4096-lay08` | `sae_step0334221_t2300M.pt`, k=32, latent=4096, d_in=512 |
| Activation hooks | fork `github.com/theodoremacmillan/graphcast@a64f7934` | **required** — upstream graphcast has no `get_activation_manager` / `SAEInjector` |
| Activations | derived: GraphCast layer 8, `mesh_gnn_post_res`, `mesh_nodes` | `(40962, 1, 512)` bfloat16 per timestep, 41.9 MB each |

Coverage: **396 timesteps, 2021-08-24T00 → 2021-11-30T18**, 6-hourly.

---

## S3 layout

```
s3://interp-990002768295/graphcast-demo/          51 GiB total
├── checkpoints/     187.3 MiB   6 objects
│   ├── graphcast/   graphcast_ckpt.npz + {diffs_stddev,mean,stddev}_by_level.nc
│   └── sae/         sae_step0334221_t2300M.pt + config.json
├── activations/      15.5 GiB   396 x layer0008_..._t<ISO>.npy, 41,945,216 B each
├── era5-surface/      3.2 GiB   101 daily .nc — 10m u/v, 2m temperature, land_sea_mask
├── era5-levels/      29.5 GiB   101 daily .nc — 6 vars at 200/300/500/700/850 hPa + MSLP
└── viz-data/          2.5 GiB   activations/<sae_id>/{indices,values}.zarr, timestamps.npy,
                                 feature_catalog.parquet, sae_registry.json,
                                 graph/{grid_lat,grid_lon}.npy, graph/mapper_cache/
```

ERA5 days run 2021-08-23 … 2021-12-01 — one day wider than the center range on
each side, because a center at time *t* needs ERA5 at *t*−6h and *t*+6h.

`era5-levels/` is only needed to re-run `scripts/era5_feature_correlation.py`;
it is 58% of the archive. Skip it if you just want the app — the fast path below
already does.

Only `viz-data/` (2.5 GiB) and `era5-surface/` (3.2 GiB) are needed to run the
demo. `activations/` is kept so the SAE can be re-encoded, or a different SAE
swapped in, without re-running GraphCast.

---

## Fast path — rehydrate and run

```bash
git clone https://github.com/theodoremacmillan/graphcast-interpretability.git
cd graphcast-interpretability
./setup_venv.sh && source .venv/bin/activate

S3=s3://interp-990002768295/graphcast-demo
aws s3 sync $S3/viz-data/      viz/data/            # 2.5 GiB
aws s3 sync $S3/era5-surface/  data/era5_daily/     # 3.2 GiB, globe overlays
aws s3 sync $S3/checkpoints/   data/                # only for new inference

streamlit run viz/app.py --server.port 8501
```

No GPU needed for this path — the app only reads precomputed zarr. ~6 GiB and a
few minutes of transfer. `setup_venv.sh` still needs the graphcast fork to
import, but no GraphCast inference runs.

---

## Full regeneration from public sources

Needs a CUDA GPU. `scripts/stream_rebuild.py` drives the whole pipeline in
short windows so peak disk stays ~30 GB rather than the ~383 GB the full ERA5
span would occupy:

```bash
python scripts/stream_rebuild.py \
    --center_start 2021-08-24T00 --center_end 2021-12-01T00 \
    --window_days 5 --s3 s3://interp-990002768295/graphcast-demo

python viz/precompute.py    --acts_dir data/activations_raw --out_dir viz/data
python viz/compute_stats.py --sae_id layer8_k32_d4096 --data_dir viz/data
```

It is resumable — finished timesteps and slimmed days are skipped on re-run.

Measured on one L4 (23 GB) + 400 GB disk, for the full 396-timestep span:

| Stage | Wall clock | Peak disk |
|---|---|---|
| `stream_rebuild.py` (ERA5 download + GraphCast + slim + upload) | 268 min | ~50 GB |
| `precompute.py` (SAE encode, GPU) | ~17 min | 2.5 GB |
| `compute_stats.py` (loads the 1,038 MB index array) | ~1 min | — |

GraphCast inference itself is ~5.7 s/timestep; the rest is ERA5 transfer.

If you re-encode, delete `viz/data/activations/` first — `precompute.py` writes
into an existing zarr, so a shorter previous run can leave stale timesteps.

---

## Gotchas that cost real time

1. **`docs/cluster_pipeline.md`'s xarray_jax patch cannot work.** It does
   `import graphcast.xarray_jax` to locate the file, but on jax ≥ 0.7.0 that
   import is exactly what raises (`jax.stages.OutInfo` was removed). Resolve the
   path with `importlib.util.find_spec` without importing — see `setup_venv.sh`.

2. **`requirements.txt` has no netCDF write backend.** `xarray.to_netcdf` fails
   with `No module named 'h5py'`. Install `h5py` and `netCDF4`.

3. **JAX OOMs on a 24 GB card at default settings.** GraphCast 0.25°/37-level
   peaks at 18.3 GiB, but JAX preallocates only 75% of the device (~17.2 GiB on
   a 23 GB L4) and fails by a hair. Set
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.96`. It does fit on 24 GB — the A100 in
   `cluster_pipeline.md` is not actually required. ~5.7 s/timestep on an L4.

4. **Chunk-boundary ERA5 days are written incomplete.** `load_era5()` slices
   inclusively to `chunk_end`, so the final day of each chunk lands with 1 of 4
   timesteps. `generate_activations.py` decides whether to download from file
   *existence* alone, so that stub is accepted permanently and its 06/12/18
   centers silently disappear as `SKIP (missing adjacent day)`.
   `stream_rebuild.py::_prune_partial` deletes short days so they refetch whole.
   Check any activation set for holes before trusting it:

   ```bash
   ls data/activations_raw | sed 's/.*_t//;s/.npy//' | sort | uniq | wc -l
   ```

5. **The original 392-timestep series has a hole at 2021-09-01** (all 4 steps),
   from chaining the `hurricane_ida_week` preset (exclusive end 2021-09-01T00)
   into the season run. `scripts/dependency_graph.py` assumes an evenly spaced
   series, so that gap was absorbed as continuous time in the committed
   lag-correlation results. The rebuilt set fills it: 396 contiguous steps.
   Regenerated figures will differ slightly from the committed PDFs for this
   reason.

6. **Mapper cache is keyed by a grid hash.** `viz/data/graph/mapper_cache/
   mesh2grid_s6_g5ac3c9ae.npz` — the `5ac3c9ae` must match, or the app silently
   rebuilds it (slow, several minutes) on first load. It is derived from the
   0.25° grid `arange(90, -90.001, -0.25) x arange(0, 360, 0.25)`.

---

## Rebuild verification (2026-08-01)

What was checked after the rebuild, and what the numbers should look like:

| Check | Result |
|---|---|
| Activation count / contiguity | 396 files, 2021-08-24T00 → 2021-11-30T18, only 6 h gaps, 2021-09-01 present |
| `SKIP (missing adjacent day)` in the run log | 0 |
| `.npy` sizes | all exactly 41,945,216 B |
| Local vs S3 object counts | 396 / 101 / 101, identical both sides |
| `compute_stats.py` | dead 13, active 4083, **mean freq 0.00781**, high-artifact 1 |
| Mapper cache | `5ac3c9ae` hit, not rebuilt |
| Streamlit `AppTest` | 0 exceptions, 4 tabs, no warnings; Hurricane Ida event selects clean |
| `era5-surface/` overlays | 10 m wind ±24 m/s, 2 m temp −67…+45 °C, mask 0–1 |
| `era5-levels/` | all 11 `ERA5_TARGETS` resolve via `ds[var].sel(level=…)` |

**Mean freq 0.00781 reproduces the original cluster log exactly** — the strongest
single signal that the rebuilt activations match the ones behind the committed
figures.

One expected difference: feature 3243's top timestamps are now 2021-09-06/07
(Hurricane Larry) rather than Ida's 2021-08-28. That is the full-season ranking,
not a regression — the earlier value came from a 20-timestep dry run that only
covered late August.

Streamlit 1.60.0 emits deprecation warnings for `use_container_width` (removal
was slated for 2025-12-31) and `st.components.v1.html`. Both still work; the app
does not need changes to run today, but that first one is now living on borrowed
time and should be migrated to `width='stretch'` / `width='content'`.
