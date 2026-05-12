"""
Offline test of the SAE intervention pipeline.

Three modes:
  ablate  — zero out a feature in one run (default)
  steer   — amplify a feature in one run
  patch   — transplant a feature's codes from run A into run B,
             measure how much B's output shifts toward A

Uses synthetic activations — no GraphCast or cluster needed.
With real .npy files the results reflect actual model behaviour.

Usage:
    python scripts/test_ablation.py                              # ablate F3243
    python scripts/test_ablation.py --feature 117                # ablate F117
    python scripts/test_ablation.py --steer --strength 3.0       # steer F3243
    python scripts/test_ablation.py --patch                      # patch F3243 A→B
    python scripts/test_ablation.py --patch --real_acts A.npy --real_acts_b B.npy
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── SAE loading (reuses model.py) ─────────────────────────────────────────────

HF_REPO   = "theodoremacmillan/sae-graphcast-k32-lat4096-lay08"
HF_FILE   = "sae_step0334221_t2300M.pt"
K_ACTIVE  = 32
N_NODES   = 40962   # icosahedral mesh splits=6
D_IN      = 512     # GraphCast layer-8 hidden dim


def download_sae(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / HF_FILE
    if local.exists():
        print(f"SAE checkpoint cached: {local}")
        return local
    print(f"Downloading SAE from HuggingFace ({HF_REPO})…")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, HF_FILE, local_dir=str(cache_dir))
    print(f"  saved to {path}")
    return Path(path)


def load_sae(ckpt_path: Path):
    """Return (enc_w, dec_w, b_pre) as float32 numpy arrays."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "model_state" in state:
        state = state["model_state"]
    enc_w = state["enc.weight"].t().float().numpy()   # [d_in, latent]
    dec_w = state["dec.weight"].t().float().numpy()   # [latent, d_in]
    b_pre = state["b_pre"].float().numpy()            # [d_in]
    # unit-norm decoder columns
    norms = np.linalg.norm(dec_w, axis=1, keepdims=True).clip(1e-8)
    dec_w = dec_w / norms
    return enc_w, dec_w, b_pre


# ── SAE forward (numpy, no JAX/torch dependency at runtime) ──────────────────

def sae_encode(x: np.ndarray, enc_w, b_pre, k: int) -> np.ndarray:
    """
    x: (N, D) float32 — layer-8 activations, already L2-normalised per node
    Returns codes: (N, latent) float32, top-k sparse
    """
    x_bar = x - b_pre                       # subtract pre-bias
    pre   = np.maximum(0.0, x_bar @ enc_w)  # ReLU encode  (N, latent)
    # top-k mask
    thresh = np.partition(pre, -k, axis=1)[:, -k]   # (N,)
    codes  = pre * (pre >= thresh[:, None])
    return codes


def sae_decode(codes: np.ndarray, dec_w, b_pre) -> np.ndarray:
    """codes: (N, latent) → reconstructed activations (N, D)"""
    return codes @ dec_w + b_pre


def normalise(x: np.ndarray) -> np.ndarray:
    """Per-node L2 normalise (matches SAE.forward)."""
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(1e-6)
    return x / norms


# ── Interventions ─────────────────────────────────────────────────────────────

def ablate_feature(codes: np.ndarray, feature_id: int) -> np.ndarray:
    """Zero out one feature across all nodes."""
    out = codes.copy()
    out[:, feature_id] = 0.0
    return out


def steer_feature(codes: np.ndarray, feature_id: int, strength: float) -> np.ndarray:
    """Add `strength` to one feature's code across all active nodes."""
    out = codes.copy()
    # only steer nodes where at least one code is active (avoid ghost activations)
    active_nodes = codes.any(axis=1)
    out[active_nodes, feature_id] += strength
    return out


def mean_ablate_feature(codes: np.ndarray, feature_id: int) -> np.ndarray:
    """Replace a feature's codes with its dataset mean (fairer baseline than zero)."""
    out = codes.copy()
    mean_val = codes[:, feature_id].mean()
    out[:, feature_id] = mean_val
    return out


def patch_feature(codes_src: np.ndarray, codes_tgt: np.ndarray,
                  feature_id: int) -> np.ndarray:
    """
    Transplant one feature's codes from run A (src) into run B (tgt).

    Only patches nodes where the feature is active in src — nodes where src
    has zero for this feature are left unchanged in tgt. This matches the
    standard activation-patching convention: you're asking "does adding A's
    feature signal into B make B look more like A?" not "does erasing B's
    feature make it look like A?"
    """
    out = codes_tgt.copy()
    out[:, feature_id] = codes_src[:, feature_id]
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def patch_recovery(recon_a: np.ndarray, recon_b: np.ndarray,
                   recon_patched: np.ndarray) -> None:
    """
    Measure how much patching B toward A actually shifts B's output.

    recovery = 1  → patched B is identical to A (full recovery)
    recovery = 0  → patched B is unchanged from B (no effect)
    recovery < 0  → patching moved B further from A (shouldn't happen for a
                    causally relevant feature)

    Uses per-node cosine distance so scale differences don't dominate.
    """
    def cos_dist(x, y):
        num = (x * y).sum(axis=1)
        den = (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)).clip(1e-8)
        return 1.0 - num / den   # 0 = identical, 2 = opposite

    d_ab      = cos_dist(recon_a, recon_b)          # original A↔B distance
    d_patched = cos_dist(recon_a, recon_patched)     # A↔patched-B distance
    recovery  = 1.0 - d_patched / d_ab.clip(1e-8)   # fraction of gap closed

    print(f"\n── Patch recovery ──")
    print(f"  A↔B cosine distance (baseline):   mean={d_ab.mean():.4f}  max={d_ab.max():.4f}")
    print(f"  A↔patched-B cosine distance:      mean={d_patched.mean():.4f}  max={d_patched.max():.4f}")
    print(f"  Recovery (fraction of gap closed): mean={recovery.mean():.4f}  "
          f"[1=full, 0=none, <0=wrong direction]")
    print(f"  Nodes where feature was patched:  "
          f"{int((recon_patched != recon_b).any(axis=1).sum())} / {len(recon_b)}")


def report(label: str, original: np.ndarray, modified: np.ndarray) -> None:
    delta     = modified - original                     # (N, D)
    l2_change = np.linalg.norm(delta, axis=1)           # per-node L2
    cos_sim   = (
        (original * modified).sum(axis=1)
        / (np.linalg.norm(original, axis=1) * np.linalg.norm(modified, axis=1)).clip(1e-8)
    )
    print(f"\n── {label} ──")
    print(f"  L2 change per node:  mean={l2_change.mean():.4f}  max={l2_change.max():.4f}")
    print(f"  Cosine similarity:   mean={cos_sim.mean():.4f}  min={cos_sim.min():.4f}")
    print(f"  Nodes where feature was active: "
          f"{int((l2_change > 0).sum())} / {len(l2_change)}")


def report_codes(label: str, codes_before: np.ndarray, codes_after: np.ndarray,
                 feature_id: int) -> None:
    print(f"\n── Codes: {label} ──")
    print(f"  Feature {feature_id} before: "
          f"n_active={int((codes_before[:, feature_id] > 0).sum())}, "
          f"mean={codes_before[:, feature_id].mean():.4f}, "
          f"max={codes_before[:, feature_id].max():.4f}")
    print(f"  Feature {feature_id} after:  "
          f"n_active={int((codes_after[:, feature_id] > 0).sum())}, "
          f"mean={codes_after[:, feature_id].mean():.4f}, "
          f"max={codes_after[:, feature_id].max():.4f}")
    active_before = int((codes_before > 0).sum(axis=1).mean())
    active_after  = int((codes_after  > 0).sum(axis=1).mean())
    print(f"  Mean active features per node: {active_before} → {active_after}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature",   type=int, default=3243,
                   help="SAE feature to intervene on")
    p.add_argument("--steer",     action="store_true",
                   help="Use steering instead of zero-ablation")
    p.add_argument("--mean_ablate", action="store_true",
                   help="Replace feature with dataset mean (fairer baseline)")
    p.add_argument("--patch",     action="store_true",
                   help="Transplant feature codes from run A into run B")
    p.add_argument("--strength",  type=float, default=2.0,
                   help="Steering strength (only with --steer)")
    p.add_argument("--real_acts", type=str, default=None,
                   help="Path to a real .npy activation file (run A / single run)")
    p.add_argument("--real_acts_b", type=str, default=None,
                   help="Path to a second .npy file (run B, for --patch only)")
    p.add_argument("--cache_dir", default="data/sae_cache")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    # ── Load SAE ──────────────────────────────────────────────────────────
    ckpt_path = download_sae(Path(args.cache_dir))
    enc_w, dec_w, b_pre = load_sae(ckpt_path)
    latent = enc_w.shape[1]
    print(f"\nSAE loaded: d_in={enc_w.shape[0]}, latent={latent}, k={K_ACTIVE}")

    if args.feature >= latent:
        print(f"ERROR: feature {args.feature} out of range (latent={latent})")
        sys.exit(1)

    # ── Activations ───────────────────────────────────────────────────────
    def load_acts(path: str | None, seed_offset: int = 0) -> np.ndarray:
        if path:
            raw = np.load(path).astype(np.float32)
            if raw.ndim == 3:
                raw = raw[:, 0, :]
            acts = normalise(raw)
            print(f"Real activations loaded: {acts.shape}  ({path})")
        else:
            rng = np.random.default_rng(args.seed + seed_offset)
            raw = rng.standard_normal((N_NODES, D_IN)).astype(np.float32)
            acts = normalise(raw)
            print(f"Synthetic activations generated: {acts.shape}  "
                  f"(seed={args.seed + seed_offset})")
        return acts

    # ── Patch mode ────────────────────────────────────────────────────────
    if args.patch:
        print("\n=== Activation patching: transplant F{} from A → B ===".format(args.feature))
        if not args.real_acts_b and args.real_acts:
            print("Note: --real_acts_b not provided; using a second synthetic run as B")

        acts_a = load_acts(args.real_acts,   seed_offset=0)
        acts_b = load_acts(args.real_acts_b, seed_offset=1)

        codes_a = sae_encode(acts_a, enc_w, b_pre, K_ACTIVE)
        codes_b = sae_encode(acts_b, enc_w, b_pre, K_ACTIVE)

        recon_a = sae_decode(codes_a, dec_w, b_pre)
        recon_b = sae_decode(codes_b, dec_w, b_pre)

        codes_patched = patch_feature(codes_a, codes_b, args.feature)
        recon_patched = sae_decode(codes_patched, dec_w, b_pre)

        print(f"\nRun A  — F{args.feature}: "
              f"n_active={int((codes_a[:, args.feature] > 0).sum())}, "
              f"max={codes_a[:, args.feature].max():.4f}")
        print(f"Run B  — F{args.feature}: "
              f"n_active={int((codes_b[:, args.feature] > 0).sum())}, "
              f"max={codes_b[:, args.feature].max():.4f}")
        print(f"Patched B — F{args.feature}: "
              f"n_active={int((codes_patched[:, args.feature] > 0).sum())}, "
              f"max={codes_patched[:, args.feature].max():.4f}")

        patch_recovery(recon_a, recon_b, recon_patched)

        # Sanity: patching with identical runs should give zero recovery gap
        codes_same  = patch_feature(codes_a, codes_a, args.feature)
        recon_same  = sae_decode(codes_same, dec_w, b_pre)
        delta_same  = np.linalg.norm(recon_same - recon_a, axis=1).max()
        print(f"\n── Sanity: patch A into A (should be no change) ──")
        print(f"  Max L2 change: {delta_same:.6f}  (should be 0.0)")

        print("\nDone. Recovery > 0 means patching F{} moves B toward A.".format(args.feature))
        print("With real activations: high recovery on storm timesteps = causal evidence.")
        return

    # ── Single-run modes (ablate / steer) ─────────────────────────────────
    acts  = load_acts(args.real_acts)
    if not args.real_acts:
        print("  (Pass --real_acts <path.npy> to use real GraphCast activations)")

    codes = sae_encode(acts, enc_w, b_pre, K_ACTIVE)
    recon = sae_decode(codes, dec_w, b_pre)

    print(f"\nBaseline reconstruction:")
    print(f"  Global sparsity: {(codes > 0).mean():.4f}  "
          f"(expect ~{K_ACTIVE}/{latent:.0f} = {K_ACTIVE/latent:.4f})")
    l2_recon = np.linalg.norm(acts - recon, axis=1).mean()
    print(f"  Mean L2 reconstruction error: {l2_recon:.4f}")

    if args.steer:
        codes_mod = steer_feature(codes, args.feature, args.strength)
        label = f"Steering F{args.feature} +{args.strength}"
    elif args.mean_ablate:
        codes_mod = mean_ablate_feature(codes, args.feature)
        label = f"Mean-ablation F{args.feature}"
    else:
        codes_mod = ablate_feature(codes, args.feature)
        label = f"Zero-ablation F{args.feature}"

    recon_mod = sae_decode(codes_mod, dec_w, b_pre)

    report_codes(label, codes, codes_mod, args.feature)
    report(f"Activation change after {label}", recon, recon_mod)

    # Sanity: ablating a dead feature should change nothing
    dead_feature = int(np.argmin((codes > 0).sum(axis=0)))
    codes_dead   = ablate_feature(codes, dead_feature)
    recon_dead   = sae_decode(codes_dead, dec_w, b_pre)
    delta_dead   = np.linalg.norm(recon_dead - recon, axis=1).max()
    print(f"\n── Sanity check: ablate least-active feature (F{dead_feature}) ──")
    print(f"  Max L2 change: {delta_dead:.6f}  (should be ~0 or tiny)")

    print("\nDone. If L2 changes above are non-zero and sanity check is near-zero,")
    print("the encode→ablate→decode pipeline is working correctly.")


if __name__ == "__main__":
    main()
