"""
Offline preprocessing: encode GraphCast activation .npy files with a trained SAE,
storing sparse feature codes in Zarr for fast interactive retrieval.

Usage:
    python viz/precompute.py \
        --acts_dir /path/to/layer8_activations \
        --sae_id layer8_k32_d4096 \
        [--sae_ckpt theodoremacmillan/sae-graphcast-k32-lat4096-lay08 | --sae_local /path/to/ckpt.pt] \
        [--layer 8] [--k_active 32] [--out_dir viz/data]

Outputs (in --out_dir):
    activations/{sae_id}/indices.zarr  — (T, N, K) int16 top-k feature indices
    activations/{sae_id}/values.zarr   — (T, N, K) float32 activation values
    activations/{sae_id}/timestamps.npy
    graph/grid_lat.npy, graph/grid_lon.npy
    sae_registry.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

# Add src/ to path for SAE import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from graphcast_interpretability.model import SAE  # noqa: E402  (triggers device print)


# ── SAE loading ──────────────────────────────────────────────────────────────

def _load_state(ckpt_path: str) -> dict:
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    return state.get("model_state", state)


def build_sae(state: dict, k_active: int, unit_norm_decoder: bool, device: str) -> SAE:
    d_in = state["enc.weight"].shape[1]
    latent = state["enc.weight"].shape[0]
    model = SAE(
        d_in=d_in, latent=latent, k_active=k_active,
        k_aux=max(k_active * 2, 32),
        unit_norm_decoder=unit_norm_decoder,
    )
    model.load_state_dict(state)
    model.eval().to(device)
    return model


# ── Encoding ─────────────────────────────────────────────────────────────────

def load_acts(path) -> np.ndarray:
    """Load a .npy activation file, converting bfloat16 void dtype to float32."""
    import ml_dtypes
    a = np.load(path)
    if a.dtype.kind == 'V' and a.dtype.itemsize == 2:
        a = a.view(ml_dtypes.bfloat16).astype(np.float32)
    return a


def encode_file(acts: np.ndarray, model: SAE, device: str,
                batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k (indices [N,K] int16, values [N,K] float32) for a .npy file."""
    idxs, vals = [], []
    with torch.no_grad():
        for s in range(0, len(acts), batch):
            x = torch.from_numpy(acts[s : s + batch]).float().to(device)
            _, code, _ = model(x)
            topk = code.topk(model.k, dim=-1)
            idxs.append(topk.indices.cpu().numpy().astype(np.int16))
            vals.append(topk.values.cpu().numpy().astype(np.float32))
    return np.concatenate(idxs), np.concatenate(vals)


# ── Filename → timestamp ─────────────────────────────────────────────────────

def parse_timestamp(fname: str) -> str:
    # layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t2021-08-29T00.npy
    return Path(fname).stem.split("_t")[-1]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Precompute sparse SAE codes from activation .npy files")
    p.add_argument("--acts_dir", required=True, help="Directory with layer*.npy activation files")
    p.add_argument("--out_dir", default="viz/data")
    p.add_argument("--sae_id", default="layer8_k32_d4096")
    p.add_argument("--layer", type=int, default=8)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--sae_ckpt", metavar="HF_REPO_ID",
                     help="HuggingFace repo (default: theodoremacmillan/sae-graphcast-k32-lat4096-lay08)")
    src.add_argument("--sae_local", metavar="PATH", help="Local .pt checkpoint path")
    p.add_argument("--k_active", type=int, default=32)
    p.add_argument("--no_unit_norm", dest="unit_norm_decoder", action="store_false", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # ERA5 0.25° standard grid; override if your data uses a different grid
    p.add_argument("--lat_start", type=float, default=90.0)
    p.add_argument("--lat_stop", type=float, default=-90.001)
    p.add_argument("--lat_step", type=float, default=-0.25)
    p.add_argument("--lon_start", type=float, default=0.0)
    p.add_argument("--lon_stop", type=float, default=360.0)
    p.add_argument("--lon_step", type=float, default=0.25)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    acts_dir = Path(args.acts_dir)

    # ── Resolve checkpoint ────────────────────────────────────────────────
    if args.sae_local:
        ckpt_path = args.sae_local
    else:
        repo = args.sae_ckpt or "theodoremacmillan/sae-graphcast-k32-lat4096-lay08"
        print(f"Downloading SAE from HuggingFace: {repo}")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("ERROR: install huggingface_hub:  pip install huggingface-hub")
            sys.exit(1)
        ckpt_path = hf_hub_download(repo, "sae_step0334221_t2300M.pt")
        try:
            cfg_path = hf_hub_download(repo, "config.json")
            cfg = json.loads(Path(cfg_path).read_text())
            args.k_active = cfg.get("k", args.k_active)
            args.unit_norm_decoder = cfg.get("unit_norm_decoder", args.unit_norm_decoder)
        except Exception:
            pass

    # ── Load SAE ─────────────────────────────────────────────────────────
    print(f"Loading SAE (k={args.k_active}, unit_norm={args.unit_norm_decoder}) on {args.device}")
    state = _load_state(ckpt_path)
    model = build_sae(state, args.k_active, args.unit_norm_decoder, args.device)
    d_in = model.enc.weight.shape[1]
    latent = model.enc.weight.shape[0]
    print(f"  d_in={d_in}, latent={latent}")

    # ── Find activation files ─────────────────────────────────────────────
    pattern = f"layer{args.layer:04d}_*.npy"
    act_files = sorted(acts_dir.glob(pattern))
    if not act_files:
        print(f"ERROR: no files matching '{pattern}' in {acts_dir}")
        sys.exit(1)
    print(f"Found {len(act_files)} activation files")

    # Peek at first file
    peek = load_acts(act_files[0])
    if peek.ndim == 3 and peek.shape[1] == 1:
        peek = peek[:, 0, :]
    n_nodes = peek.shape[0]
    assert peek.shape[1] == d_in, (
        f"Activation dim {peek.shape[1]} != SAE d_in {d_in}. "
        "Check --layer or --sae_local."
    )

    # ── Create Zarr stores ────────────────────────────────────────────────
    zarr_root = out_dir / "activations" / args.sae_id
    zarr_root.mkdir(parents=True, exist_ok=True)
    n_time = len(act_files)

    idx_store = zarr.open(
        str(zarr_root / "indices.zarr"), mode="w",
        shape=(n_time, n_nodes, args.k_active),
        chunks=(1, n_nodes, args.k_active),
        dtype=np.int16,
    )
    val_store = zarr.open(
        str(zarr_root / "values.zarr"), mode="w",
        shape=(n_time, n_nodes, args.k_active),
        chunks=(1, n_nodes, args.k_active),
        dtype=np.float32,
    )

    # ── Encode ────────────────────────────────────────────────────────────
    timestamps = []
    for t, f in enumerate(act_files):
        ts = parse_timestamp(f.name)
        print(f"  [{t + 1}/{n_time}] {ts}", flush=True)
        acts = load_acts(f)
        if acts.ndim == 3 and acts.shape[1] == 1:
            acts = acts[:, 0, :]
        idxs, vals = encode_file(acts, model, args.device)
        idx_store[t] = idxs
        val_store[t] = vals
        timestamps.append(ts)

    np.save(str(zarr_root / "timestamps.npy"), np.array(timestamps))

    # ── Save ERA5 grid coordinates ────────────────────────────────────────
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    grid_lat = np.arange(args.lat_start, args.lat_stop, args.lat_step)
    grid_lon = np.arange(args.lon_start, args.lon_stop, args.lon_step)
    np.save(str(graph_dir / "grid_lat.npy"), grid_lat)
    np.save(str(graph_dir / "grid_lon.npy"), grid_lon)
    print(f"Grid: lat {grid_lat[0]}..{grid_lat[-1]} ({len(grid_lat)}), "
          f"lon {grid_lon[0]}..{grid_lon[-1]} ({len(grid_lon)})")

    # ── Update SAE registry ───────────────────────────────────────────────
    reg_path = out_dir / "sae_registry.json"
    registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    registry[args.sae_id] = {
        "sae_id": args.sae_id,
        "layer": args.layer,
        "k_active": args.k_active,
        "latent": latent,
        "d_in": d_in,
        "n_time": n_time,
        "n_nodes": n_nodes,
        "timestamps": timestamps,
    }
    reg_path.write_text(json.dumps(registry, indent=2))

    print(f"\nDone. Outputs in {zarr_root}")
    print(f"Next: python viz/compute_stats.py --sae_id {args.sae_id}")


if __name__ == "__main__":
    main()
