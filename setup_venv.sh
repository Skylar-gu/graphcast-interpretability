#!/usr/bin/env bash
# Rebuild the demo environment from scratch (L4 / CUDA 12 wheels).
# Mirrors docs/cluster_pipeline.md §1 but pins GPU JAX instead of the CPU default.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Geospatial stack needs system PROJ/GEOS — binary wheels only.
pip install --only-binary=:all: Cartopy==0.25.0 pyproj==3.7.2 shapely==2.1.1

# Main requirements: the graphcast fork is pinned over SSH; rewrite to HTTPS.
# Drop the geospatial pins (installed above) and CPU jax/jaxlib (GPU below).
sed 's|git+ssh://git@github.com/|git+https://github.com/|g' requirements.txt \
  | grep -v -E "^(Cartopy|pyproj|shapely|jax|jaxlib)==" \
  > /tmp/req_fixed.txt
pip install -r /tmp/req_fixed.txt

# GPU JAX matching the pinned version.
pip install "jax[cuda12]==0.7.0"

# model.py needs torch; SAE encode runs on GPU.
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Viz + S3.
pip install streamlit plotly pyarrow huggingface-hub boto3 h5netcdf gcsfs zarr
# netCDF write backends — xarray.to_netcdf needs one; requirements.txt omits them.
pip install h5py netCDF4

# Patch graphcast for JAX >= 0.4.25 (jax.stages.OutInfo removal).
# NOTE: docs/cluster_pipeline.md does `import graphcast.xarray_jax` here, but that
# import is itself what fails on jax 0.7.0. Resolve the path without importing.
python - <<'EOF'
import pathlib, re, importlib.util
spec = importlib.util.find_spec("graphcast.xarray_jax")
p = pathlib.Path(spec.origin)
src = p.read_text()
if 'OutInfo' in src:
    p.write_text(re.sub(r',\s*jax\.stages\.OutInfo', '', src))
    print("patched xarray_jax.py")
else:
    print("no patch needed")
EOF

pip install -e .

python -c "from graphcast_interpretability.model import load_sae_params_from_torch; print('graphcast_interpretability ok')"
python -c "from graphcast.deep_typed_graph_net import get_activation_manager; print('activation manager ok')"
python -c "import jax; print('jax devices:', jax.devices())"
echo "ENV BUILD OK"
