#!/bin/bash
# -----------------------------------------------------------------------------
# GraphCast Interpretability — macOS local setup
#
# Prerequisites:
#   - Homebrew with: brew install proj geos  (already present if brew list shows them)
#   - Python 3.11 via Homebrew (python@3.11)
#
# Usage:
#   bash setup_mac.sh
#   source .venv/bin/activate
# -----------------------------------------------------------------------------
set -euo pipefail

PYTHON=/opt/homebrew/bin/python3.11
VENV_DIR=".venv"

echo "==> Python: $($PYTHON --version)"
echo "==> PROJ:   $(proj --version 2>&1 | head -1 || echo 'not found')"
echo "==> GEOS:   $(geos-config --version 2>/dev/null || echo 'checking brew...')"

# ── System library check / install ───────────────────────────────────────────
for pkg in proj geos fftw; do
    if ! brew list "$pkg" &>/dev/null; then
        echo "==> Installing $pkg via brew ..."
        brew install "$pkg"
    fi
done

# ── Virtual environment ───────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "==> Removing existing .venv ..."
    rm -rf "$VENV_DIR"
fi

echo "==> Creating $VENV_DIR with $PYTHON ..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip / setuptools / wheel ..."
pip install --upgrade pip setuptools wheel --quiet

# ── Geospatial packages (binary-only; use system PROJ/GEOS from brew) ────────
echo "==> Installing geospatial stack (Cartopy, pyproj, shapely) ..."
pip install --only-binary=cartopy,pyproj,shapely \
    "Cartopy==0.25.0" "pyproj==3.7.2" "shapely==2.1.1" --quiet

# ── Main requirements (replace git+ssh with git+https for graphcast) ─────────
echo "==> Installing main requirements ..."
REQS_FIXED=$(mktemp -t requirements_mac)
sed 's|git+ssh://git@github.com/|git+https://github.com/|g' requirements.txt \
    | grep -v -E "^(Cartopy|pyproj|shapely)==" \
    > "$REQS_FIXED"
pip install -r "$REQS_FIXED" --quiet
rm "$REQS_FIXED"

# ── PyTorch (not in requirements.txt but needed by model.py / precompute.py) ─
echo "==> Installing PyTorch ..."
pip install torch --quiet

# ── Visualization-specific packages ──────────────────────────────────────────
echo "==> Installing viz requirements ..."
pip install -r viz/requirements.txt --quiet

# ── Patch graphcast for JAX 0.7 (OutInfo removed) ────────────────────────────
echo "==> Patching graphcast/xarray_jax.py for JAX 0.7 compatibility ..."
XARRAY_JAX=$(python -c "import graphcast.xarray_jax as m; print(m.__file__)")
sed -i '' \
    's/jax\.Array, jax\.ShapeDtypeStruct, jax\.stages\.ArgInfo, jax\.stages\.OutInfo/jax.Array, jax.ShapeDtypeStruct, jax.stages.ArgInfo/' \
    "$XARRAY_JAX"

# ── Install this package in editable mode ────────────────────────────────────
echo "==> Installing graphcast-interpretability (editable) ..."
pip install -e . --quiet

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "==> Verification"
python - <<'EOF'
imports = [
    ("numpy",              "import numpy as np; print(f'  numpy {np.__version__}')"),
    ("torch",              "import torch; print(f'  torch {torch.__version__}')"),
    ("jax",                "import jax; print(f'  jax   {jax.__version__}')"),
    ("zarr",               "import zarr; print(f'  zarr  {zarr.__version__}')"),
    ("cartopy",            "import cartopy; print(f'  cartopy {cartopy.__version__}')"),
    ("streamlit",          "import streamlit as st; print(f'  streamlit {st.__version__}')"),
    ("plotly",             "import plotly; print(f'  plotly {plotly.__version__}')"),
    ("graphcast_interp",   "from graphcast_interpretability.model import SAE; print('  graphcast_interpretability OK')"),
    ("GridMeshMapper",     "from graphcast_interpretability.interpolate_utils import GridMeshMapper; print('  GridMeshMapper OK')"),
]
failed = []
for name, stmt in imports:
    try:
        exec(stmt)
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        failed.append(name)
if failed:
    print(f"\nFailed: {failed}")
    raise SystemExit(1)
else:
    print("\nAll checks passed.")
EOF

echo ""
echo "Setup complete. To activate later:"
echo "  source .venv/bin/activate"
