# ERA5 Correlation: Interpreting Unknown SAE Features

**Goal:** Determine what meteorological phenomenon an unknown SAE feature represents by
correlating its activation patterns against ERA5 physical variables.

**When to use:** After running the lagged dependency graph with `--screen_all` and finding
features not in `KNOWN_LABELS` — you have feature IDs but no physical interpretation.

---

## Two Analyses

### A. Temporal Correlation
*"Which physical variables co-move with this feature over time?"*

Reduce everything to one scalar per timestep, then compute Pearson r across T=32 timesteps.

- **Feature side:** `extract_timeseries(..., aggregation="max")` — already in `dependency_graph.py`
- **ERA5 side:** spatial statistic over a bounding box (e.g. max 850 hPa vorticity in Gulf region)
- **Output:** ranked table of `(variable, level, region-statistic) → r`

### B. Spatial Correlation
*"Where on the globe does this feature's activation track a physical field?"*

Pixel-wise Pearson r between the feature's activation map and ERA5 fields across time.

- **Feature side:** interpolate per-node activation → `(T, lat, lon)` using `GridMeshMapper`
- **ERA5 side:** `(T, lat, lon)` field per variable × level
- **Output:** 2D correlation map overlaid on a globe — warm/cool where feature tracks that variable

Spatial correlation is the more interpretable result. Start here.

---

## Step-by-Step

```
1. Load ERA5 for the Ida week
   xr.open_mfdataset("data/era5_daily_nc/era5_2021-08-*.nc")
   → 3D vars: (T=32, level=37, lat=721, lon=1440)
   → surface vars: (T=32, lat=721, lon=1440)

2. Extract feature activation maps
   For each t: load zarr[t], scatter-max onto mesh nodes, apply GridMeshMapper
   → (T=32, lat=721, lon=1440) float32 array

3. Compute pixel-wise Pearson r
   For each (variable, level):
     X = activation_maps[:, :, :]   # (T, lat, lon)
     Y = era5[var, level, :, :, :]  # (T, lat, lon)
     # vectorized: normalize each (T,) pixel vector, then dot product
     X_z = X - X.mean(0); X_z /= (X_z.std(0) + 1e-8)
     Y_z = Y - Y.mean(0); Y_z /= (Y_z.std(0) + 1e-8)
     r_map = (X_z * Y_z).mean(0)   # (lat, lon)

4. Plot the top r-maps
   imshow(r_map, cmap="RdBu_r", vmin=-1, vmax=1) + coastlines
   Annotate with variable name and pressure level
```

---

## Which ERA5 Variables to Check First

| Variable | Level(s) | Diagnostic for |
|---|---|---|
| `mean_sea_level_pressure` | surface | Low-pressure center, storm track |
| `geopotential` | 500, 850 hPa | Pressure troughs, ridge/trough structure |
| `specific_humidity` | 700, 850 hPa | Moisture plumes, moist inflow |
| `u/v_component_of_wind` | 200, 850 hPa | Upper outflow vs. low-level inflow |
| `temperature` | 300 hPa | Warm-core aloft (TC signature) |
| `vertical_velocity` | 500 hPa | Deep convection |

Start with MSLP and 850 hPa geopotential — clearest TC signal, easy to sanity-check
against the storm track.

---

## Practical Considerations

**Derived fields:** Relative vorticity (∂v/∂x − ∂u/∂y at 850 hPa) is a cleaner TC tracer
than wind components individually. Compute from u/v with `np.gradient`.

**Normalize before correlating:** SAE activations are sparse and right-skewed. Consider
`log1p` or rank-transforming before computing r — prevents one large spike from dominating.

**T=32 is small:** Noise floor is ±0.35 at α=0.05 (same caveat as the lag graph). Only
trust |r| ≥ 0.4 as meaningful. Flag top correlations rather than hard-thresholding.

**Region masking:** If the spatial map (from the Streamlit app) already shows the feature
fires in a specific region, compute temporal correlation only over that bounding box.
Much higher signal-to-noise than a global r.

---

## Interpreting the Output

A TC-adjacent feature that lags Feature 3243 by 6h should show:

| ERA5 field | Expected r | Interpretation |
|---|---|---|
| 850 hPa geopotential | negative (low GPH = low pressure) | Low-pressure / vortex |
| 850 hPa specific humidity | positive | Moist inflow |
| 300 hPa temperature anomaly | positive | Warm core aloft |
| 200 hPa divergence | positive | Upper-level outflow |
| MSLP | negative | Surface low deepening |

**Distinguishing similar features:**
- "Moist inflow" vs "warm core": moist inflow feature → strong r with 850 hPa q, weak with
  300 hPa T. Warm core → opposite.
- "Low-pressure" vs "outflow": low-pressure feature → negative r with 850 hPa GPH.
  Outflow → positive r with 200 hPa divergence, near-zero with surface fields.

Once you have a clear candidate label, add it to `KNOWN_LABELS` in `viz/compute_stats.py`
and regenerate the catalog:
```bash
source .venv/bin/activate
python viz/compute_stats.py --sae_id layer8_k32_d4096 --data_dir viz/data
```

