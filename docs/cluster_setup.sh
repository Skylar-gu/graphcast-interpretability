#!/usr/bin/env bash
# =============================================================================
# cluster_setup.sh — Full environment setup for graphcast-interpretability
#
# Tested on: Ubuntu 22.04 (Jammy), NVIDIA GPU (A100 / T4), Lambda Cloud
#
# Issues this script works around:
#   1. requirements.txt pins Python >=3.11 packages (jax==0.7.0, numpy==2.3.2)
#      but Ubuntu 22.04 ships Python 3.10 by default.
#   2. deadsnakes PPA (common on Lambda) may have 404 failures for python3.11
#      if its cache is stale — removing and reinstalling fixes it.
#   3. requirements.txt uses git+ssh:// URLs which fail without SSH keys;
#      they must be rewritten to git+https://.
#   4. Cartopy/pyproj/shapely must be installed from binary wheels only —
#      building from source requires system PROJ/GEOS headers that may be absent.
#   5. graphcast's xarray_jax.py references jax.stages.OutInfo which was
#      removed in JAX >= 0.4.25. The patch must be applied directly to the file
#      (not via import) because the module fails to import before patching.
#   6. gcsfs is not in requirements.txt but is required by generate_activations.py.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── 0. Python 3.11 ────────────────────────────────────────────────────────────
# Ubuntu 22.04 default is Python 3.10. jax>=0.5 and numpy>=2.3 require >=3.11.
# The deadsnakes PPA sometimes returns 404 for python3.11 if its package list
# is stale. Fix: remove the PPA so apt falls back to the Ubuntu universe repo,
# which carries python3.11 at a slightly older patch version but works fine.

if ! command -v python3.11 &>/dev/null; then
    echo "==> Installing Python 3.11..."
    sudo add-apt-repository --remove ppa:deadsnakes/ppa -y 2>/dev/null || true
    sudo apt-get update -q
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

python3.11 --version   # should print Python 3.11.x

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [ -d ".venv" ]; then
    echo "==> Removing existing .venv..."
    rm -rf .venv
fi

echo "==> Creating .venv with Python 3.11..."
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

# ── 2. Geospatial stack — binary wheels only ──────────────────────────────────
# Cartopy/pyproj/shapely build from source requires system PROJ & GEOS headers.
# On most HPC/cloud nodes these are absent. Always use --only-binary for these.

echo "==> Installing geospatial stack (binary wheels only)..."
pip install --only-binary=cartopy,pyproj,shapely Cartopy pyproj shapely

# ── 3. Core requirements — SSH → HTTPS rewrite ───────────────────────────────
# requirements.txt pins graphcast via git+ssh://git@github.com/... which
# requires a deploy key. Rewrite to git+https:// for keyless installs.
# Also skip the three geo packages already installed above.

echo "==> Installing requirements (SSH→HTTPS, skipping already-installed geo packages)..."
sed 's|git+ssh://git@github.com/|git+https://github.com/|g' requirements.txt \
  | grep -v -E "^(Cartopy|pyproj|shapely)==" \
  > /tmp/req_fixed.txt

pip install -r /tmp/req_fixed.txt

# ── 4. Patch graphcast for JAX >= 0.4.25 ─────────────────────────────────────
# jax.stages.OutInfo was removed. graphcast/xarray_jax.py still references it,
# causing an AttributeError on import.
#
# IMPORTANT: patch the file directly — do NOT use the inline Python script from
# the runbook that does `import graphcast.xarray_jax`. The import itself fails
# before the patch logic can run, so you get ModuleNotFoundError / AttributeError
# instead of a patch.

echo "==> Patching graphcast xarray_jax.py for JAX >= 0.4.25..."
XARRAY_JAX="$(python -c "import site; print(site.getsitepackages()[0])")/graphcast/xarray_jax.py"

if grep -q "OutInfo" "$XARRAY_JAX"; then
    sed -i 's/,\s*jax\.stages\.OutInfo//' "$XARRAY_JAX"
    echo "    patched: removed jax.stages.OutInfo from $XARRAY_JAX"
else
    echo "    no patch needed"
fi

# Verify import works
python -c "import graphcast; print('graphcast import OK')"

# ── 5. PyTorch ────────────────────────────────────────────────────────────────
# Not in requirements.txt but required by src/graphcast_interpretability/sae/model.py

echo "==> Installing torch..."
pip install torch

# ── 6. Viz requirements ───────────────────────────────────────────────────────
echo "==> Installing viz requirements..."
pip install -r viz/requirements.txt

# ── 7. gcsfs — missing from requirements.txt ─────────────────────────────────
# scripts/generate_activations.py imports gcsfs for GCS checkpoint access.
# It is not listed in requirements.txt.

echo "==> Installing gcsfs..."
pip install gcsfs

# ── 8. Local package ─────────────────────────────────────────────────────────
echo "==> Installing local package in editable mode..."
pip install -e .

# ── 9. Smoke test ────────────────────────────────────────────────────────────
echo ""
echo "==> Smoke test..."
python - <<'EOF'
import graphcast;   print("graphcast  OK")
import jax;         print("jax       ", jax.__version__)
import torch;       print("torch     ", torch.__version__)
import xarray;      print("xarray     OK")
import zarr;        print("zarr       OK")
import gcsfs;       print("gcsfs      OK")
EOF

echo ""
echo "==> Setup complete. Activate the environment with:"
echo "    source .venv/bin/activate"
