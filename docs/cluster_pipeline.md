# Cluster Pipeline: Target-Centered Causal Graph

**Goal:** Recover a causal graph centered on Feature 3243 (hurricane-core) from SAE activations.  
**Hardware:** A100 GPU, Linux cluster.  
**Budget:** 24 hours total.  
**Expected output:** `results/causal/causal_graph.png` + `results/causal/results.json`

---

## Expected causal structure

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

The analysis will:
1. Confirm whether these edges are statistically present in the learned representations
2. Recover the lag structure (6h vs 12h)
3. Expand the graph with up to 20 additional features found via MI screening

---

## Time budget

| Step | Wall time (est.) |
|---|---|
| Setup + install | 15 min |
| ERA5 download (2 hurricane seasons) | 30–60 min |
| GraphCast inference — 1,464 timesteps | ~14 h |
| SAE encoding + Zarr preprocessing | 20 min |
| Feature statistics | 10 min |
| MI screening + PCMCI + ANM | 30–60 min |
| **Total** | **~17 h** |

---

## Why 1,464 timesteps

Target-centered PCMCI with ~25 nodes and τ_max=4 needs T/p ≥ 30.  
Two full Atlantic hurricane seasons (Jun–Nov 2020, Jun–Nov 2021) give T=1,464 and T/p≈58 — solid.  
Both seasons were very active (2020: record 30 named storms; 2021: 21 including Ida).

---

## 0. Prerequisites

```bash
# Python 3.10 or 3.11, CUDA 12+
python --version
nvidia-smi

# Clone the repo
git clone https://github.com/theodoremacmillan/graphcast-interpretability.git
cd graphcast-interpretability
```

---

## 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Geospatial stack (needs system PROJ + GEOS)
pip install --only-binary=cartopy,pyproj,shapely Cartopy pyproj shapely

# Main requirements (converts SSH graphcast URL to HTTPS)
sed 's|git+ssh://git@github.com/|git+https://github.com/|g' requirements.txt \
  | grep -v -E "^(Cartopy|pyproj|shapely)==" \
  > /tmp/req_fixed.txt
pip install -r /tmp/req_fixed.txt

# PyTorch (not in requirements.txt but needed by model.py)
pip install torch

# Viz + causal analysis packages
pip install -r viz/requirements.txt
pip install tigramite causal-learn

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

# Install this package
pip install -e .
```

---

## 2. Generate activations — 2 hurricane seasons

Generates 1,464 activation `.npy` files (~650 MB each = ~950 GB total).  
Uses a persistent JAX compile cache so restarts are fast.

```bash
# Hurricane season 2020: June 1 – Nov 30
python scripts/generate_activations.py \
    --preset hurricane_season_2020 \
    --acts_dir data/activations_raw \
    --era5_dir data/era5_daily \
    --ckpt_cache data/graphcast_cache

# Hurricane season 2021: June 1 – Nov 30  (appends to same acts_dir)
python scripts/generate_activations.py \
    --preset hurricane_season_2021 \
    --acts_dir data/activations_raw \
    --era5_dir data/era5_daily \
    --ckpt_cache data/graphcast_cache
```

Or run both back-to-back (unattended):

```bash
for preset in hurricane_season_2020 hurricane_season_2021; do
    python scripts/generate_activations.py \
        --preset $preset \
        --acts_dir data/activations_raw \
        --era5_dir data/era5_daily \
        --ckpt_cache data/graphcast_cache
done
```

**If the job is interrupted**, just re-run the same command — completed timesteps are skipped automatically.

**Disk check before running:**

```bash
df -h .   # needs ~1 TB free for both seasons
ls data/activations_raw | wc -l   # should reach 1464
```

---

## 3. SAE encoding → sparse Zarr

Downloads the SAE checkpoint from HuggingFace automatically.

```bash
python viz/precompute.py \
    --acts_dir data/activations_raw \
    --out_dir viz/data
```

Expected output: `viz/data/activations/layer8_k32_d4096/{indices,values}.zarr`  
Zarr store size: ~2 GB (sparse, compressed).

---

## 4. Feature statistics

```bash
python viz/compute_stats.py \
    --sae_id layer8_k32_d4096 \
    --data_dir viz/data \
    --streaming   # use streaming mode to keep RAM low on cluster
```

Output: `viz/data/feature_catalog.parquet`

---

## 5. Target-centered dependency / causal analysis

The script supports two modes:

### 5a. Lagged dependency graph (fast demo — works with 32 timesteps)

No tigramite needed. Runs in seconds. Good for a 2-day demo.

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

Steps: feature time series extraction → pairwise lagged Pearson cross-correlations
→ thresholded directed graph (A→B if corr(A[t], B[t+τ]) ≥ 0.30) → validation.

Outputs:
- `results/causal/lag_dep_graph.png` — network graph with lag labels
- `results/causal/lag_corr_heatmap.png` — cross-correlation heatmap per lag
- `results/causal/lag_corr_results.json`
- `results/causal/validation.txt`

**Note:** This is a dependency graph, not a causal graph — edges show predictability,
not causation. Spurious correlations from shared forcing are not removed.

### 5b. Full PCMCI+ causal graph (cluster — needs 1,464 timesteps)

```bash
python scripts/causal_graph.py \
    --mode pcmciplus \
    --focal_feature 3243 \
    --seed_features 117,402,911,2088 \
    --n_mi_candidates 20 \
    --tau_max 4 \
    --sae_id layer8_k32_d4096 \
    --data_dir viz/data \
    --out_dir results/causal
```

Steps:
1. **Feature time series extraction** — scalar activation per feature per timestep
2. **MI screening** — rank all 4,096 features by MI with Feature 3243; select top-20
3. **PCMCI+** — causal graph over seed features + MI candidates (~25 nodes)
4. **ANM orientation** — resolve any undirected edges using additive noise models
5. **Validation table** — checks expected edges against recovered graph

Outputs:
- `results/causal/causal_graph.png` — PCMCI+ lag graph
- `results/causal/results.json` — full adjacency matrix, p-values, effect sizes
- `results/causal/validation.txt`

---

## 6. (Optional) Launch the visualization atlas

```bash
pip install streamlit
streamlit run viz/app.py --server.port 8501
```

Then SSH-tunnel from your local machine:
```bash
ssh -L 8501:localhost:8501 user@cluster
```

Open `http://localhost:8501` in your browser.

---

## Interpreting results

**PCMCI+ output:**
- `val_matrix[i, j, tau]` — partial correlation at lag τ between feature i → feature j
- `p_matrix[i, j, tau]` — p-value (significant if < 0.05 after MCI correction)
- Positive val = feature i at time t predicts higher feature j at time t+τ

**Expected edges to validate:**

| Edge | Expected lag | Physical interpretation |
|---|---|---|
| 117 → 3243 | 1 step (6h) | Moist inflow feeds TC core |
| 402 → 3243 | 1 step (6h) | Low-pressure organizes into TC |
| 911 → 3243 | 2 steps (12h) | Warm-core development precedes intensification |
| 3243 → 2088 | 1 step (6h) | TC core drives upper-level outflow |

**Red flags:**
- Edges from 3243 → 117/402/911 (wrong direction) suggest the feature is a proxy for the storm, not a cause
- High artifact score in feature_catalog.parquet means the feature tracks mesh structure, not physics
- Bidirectional edges (3243 ↔ X) are physically plausible for some couplings (e.g., warm core ↔ intensification)

---

## Troubleshooting

**OOM during GraphCast inference:**  
The model uses bfloat16 + gradient checkpointing and should fit on 40 GB A100. If it OOMs, something else is using GPU memory — check `nvidia-smi`.

**Slow PCMCI:**  
With 25 nodes and τ_max=4, PCMCI+ should finish in minutes using `ParCorr`. If using `CMIknn` (nonlinear), it can take hours — switch back to `ParCorr` with `--cond_ind_test parcorr`.

**Feature 3243 barely fires in the data:**  
Check `viz/data/feature_catalog.parquet` — look at `activation_frequency` and `top_timestamps`. If frequency is very low, the Atlantic hurricane season dates are correct but the feature may also fire in the Western Pacific. Consider adding typhoon seasons.

**tigramite import error:**  
```bash
pip install tigramite --upgrade
```

**causal-learn import error:**  
```bash
pip install causal-learn --upgrade
# Note: the package name is causal-learn but imports as causallearn
```
