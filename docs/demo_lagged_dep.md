# Demo: Lagged Dependency Graph (Hurricane Ida Week)

**Goal:** Produce a lagged dependency graph centered on Feature 3243 (hurricane-core).  
**Hardware:** A100 or T4 GPU (VS Code / cluster SSH).  
**Data:** Hurricane Ida week — 32 timesteps at 6h resolution (2021-08-24 → 2021-08-31).  
**Time budget:** ~2 hours total (mostly waiting for GraphCast inference).  
**Expected output:** `results/causal/lag_dep_graph.png` + `results/causal/lag_corr_heatmap.png`

---

## Expected output structure

```
              Feature 117
            moist inflow-like
                   |
                   | 6h
                   v

Feature 402 --6h--> Feature 3243 <--12h-- Feature 911
low-pressure-like   hurricane-core        warm-core-like

                   |
                   | 6h
                   v
              Feature 2088
            outflow-like
```

Edges are lagged Pearson correlations (|r| ≥ 0.30), not causal — they show that
feature A's activation predicts feature B's activation τ hours later.

---

## Time budget

| Step | Time |
|---|---|
| Environment setup | 10 min |
| ERA5 download (9 days) | 5–10 min |
| GraphCast inference (32 timesteps) | ~45 min on T4, ~15 min on A100 |
| SAE encoding → Zarr | 2 min |
| Feature statistics | 1 min |
| Lagged dependency graph | < 1 min |
| **Total** | **~65–80 min** |

---

## 0. Prerequisites

```bash
python --version   # 3.10 or 3.11
nvidia-smi         # confirm GPU visible
git clone https://github.com/theodoremacmillan/graphcast-interpretability.git
cd graphcast-interpretability
```

---

## 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Geospatial stack (Linux cluster — system PROJ/GEOS required)
pip install --only-binary=cartopy,pyproj,shapely Cartopy pyproj shapely

# Core requirements (converts SSH → HTTPS for graphcast)
sed 's|git+ssh://git@github.com/|git+https://github.com/|g' requirements.txt \
  | grep -v -E "^(Cartopy|pyproj|shapely)==" \
  > /tmp/req_fixed.txt
pip install -r /tmp/req_fixed.txt

# PyTorch (needed by SAE model.py)
pip install torch

# Viz packages
pip install -r viz/requirements.txt

# Patch graphcast for JAX >= 0.4.25
python - <<'EOF'
import pathlib, re
import graphcast.xarray_jax as m
p = pathlib.Path(m.__file__)
src = p.read_text()
if 'OutInfo' in src:
    p.write_text(re.sub(r',\s*jax\.stages\.OutInfo', '', src))
    print("patched xarray_jax.py")
else:
    print("no patch needed")
EOF

pip install -e .
```

---

## 2. Generate activations — Hurricane Ida week (32 timesteps)

Downloads ERA5 for 9 days, then runs GraphCast with layer-8 activation hooks.
Uses a persistent JAX compile cache so restarts skip recompilation.

```bash
python scripts/generate_activations.py \
    --preset hurricane_ida_week \
    --acts_dir data/activations_raw \
    --era5_dir data/era5_daily \
    --ckpt_cache data/graphcast_cache
```

**Disk:** ~3.5 GB for activations + ~5 GB for ERA5 daily files. Check first:
```bash
df -h .
```

**If interrupted:** re-run the same command. Completed timesteps are skipped automatically.

Expected output:
```
data/activations_raw/
  layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t2021-08-24T00.npy  (~100 MB)
  layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t2021-08-24T06.npy
  ...  (32 files total)
```

---

## 3. SAE encoding → sparse Zarr

Downloads the SAE checkpoint from HuggingFace automatically on first run (~200 MB).

```bash
python viz/precompute.py \
    --acts_dir data/activations_raw \
    --out_dir viz/data
```

Output: `viz/data/activations/layer8_k32_d4096/{indices,values}.zarr` (~60 MB)

---

## 4. Feature statistics

```bash
python viz/compute_stats.py \
    --sae_id layer8_k32_d4096 \
    --data_dir viz/data
```

Output: `viz/data/feature_catalog.parquet`

---

## 5. Lagged dependency graph

```bash
python scripts/causal_graph.py \
    --mode lagged_corr \
    --focal_feature 3243 \
    --seed_features 117,402,911,2088 \
    --tau_max 4 \
    --threshold 0.30 \
    --sae_id layer8_k32_d4096 \
    --data_dir viz/data \
    --out_dir results/causal
```

This computes `corr(feature_i[t], feature_j[t+τ])` for all pairs at τ = 6h, 12h, 18h, 24h,
then draws A → B if `|r| ≥ 0.30` at the peak lag.

**Tuning `--threshold`:** Lower to `0.20` to see more edges; raise to `0.40` for only the
strongest. With 32 timesteps the Pearson noise floor is roughly ±0.35 at α = 0.05 —
treat borderline edges as suggestive rather than confirmed.

Outputs:
- `results/causal/lag_dep_graph.png` — network graph, Feature 3243 at center
- `results/causal/lag_corr_heatmap.png` — cross-correlation matrix by lag
- `results/causal/lag_corr_results.json` — all r-values and edge list
- `results/causal/validation.txt` — expected vs recovered edge summary

---

## 6. (Optional) Visualization atlas

```bash
streamlit run viz/app.py --server.port 8501
```

SSH-tunnel to view locally:
```bash
ssh -L 8501:localhost:8501 user@cluster
```

Then open `http://localhost:8501`.

---

## Interpreting the output

**`lag_dep_graph.png`**
- Orange node = Feature 3243 (focal), placed at center
- Blue nodes = seed features in a ring
- Arrow A → B labeled "6h, r=+0.52" means Feature A predicts Feature B 6h later (r = 0.52)
- Edge thickness encodes |r|

**What to look for:**

| Expected edge | Physical story |
|---|---|
| 117 → 3243 at 6h | Moist inflow organizes before TC core intensifies |
| 402 → 3243 at 6h | Low-pressure precedes vortex deepening |
| 911 → 3243 at 12h | Warm-core development lags slightly behind surface deepening |
| 3243 → 2088 at 6h | TC core drives upper-level divergence / outflow |

**Caveats:**
- With 32 timesteps, only strong signals (|r| ≥ 0.3–0.4) are reliable
- Reverse edges (e.g. 3243 → 117) can appear from bidirectional coupling — not wrong causality
- Shared meteorological forcing can produce spurious edges between unrelated features

---

## Troubleshooting

**OOM during GraphCast inference:**  
The model uses bfloat16 + gradient checkpointing; should fit on 16 GB T4.  
If it OOMs, check `nvidia-smi` for other processes using GPU memory.

**`No module named 'zarr'` / `'torch'`:**
```bash
pip install zarr torch
```

**`sae_registry.json` not found:**  
Run step 3 (`viz/precompute.py`) before step 5.

**Feature 3243 flat (all zeros):**  
Check that the `.npy` files actually cover 2021-08-28 to 2021-08-30 (Ida's landfall window).
If zero everywhere, verify the SAE checkpoint matches the graphcast fork version.

**Graph has no edges:**  
Lower `--threshold` to `0.20` and inspect `lag_corr_heatmap.png` to see raw correlations.
