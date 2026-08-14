"""
Rebuild the demo dataset on a disk-constrained box by streaming ERA5.

scripts/generate_activations.py downloads *every* missing ERA5 day up front and
calls ds.load() on a whole monthly chunk (~107 GB resident, ~118 GB on disk for
a 31-day month). Neither fits here. This driver runs the same script over short
center windows so that, at any moment, only a few days of full ERA5 exist:

    for each window:
        generate_activations.py  (downloads its own few days, runs GraphCast)
        era5_slim.py             (full .nc -> surface/ + levels/ subsets)
        upload activations + subsets to S3
        delete full .nc no longer needed by later windows

Resumable: completed timesteps are skipped by generate_activations.py itself,
and already-slimmed / already-uploaded days are skipped here.

Usage:
    python scripts/stream_rebuild.py \
        --center_start 2021-08-24T00 --center_end 2021-12-01T00 \
        --s3 s3://interp-990002768295/graphcast-demo
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent

# GraphCast 0.25°/37-level peaks at ~18.3 GiB. JAX preallocates 75% of the
# device by default, which on a 23 GB L4 is ~17.2 GiB — it OOMs by a hair.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.96")


def _run(cmd: list[str]) -> None:
    print(f"    $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _day(t: np.datetime64) -> str:
    return str(np.datetime64(t, "D"))


def _free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 1e9


def _prune_partial(full_dir: Path) -> None:
    """
    Delete full ERA5 days holding fewer than 4 timesteps.

    load_era5() slices inclusively to chunk_end, so the last day of every chunk
    lands with only the 00:00 step. generate_activations.py decides whether to
    download purely on file existence, so such a stub would be accepted by a
    later window and silently drop that day's 06/12/18 centers via
    three_step_window() -> None ("SKIP (missing adjacent day)"). Removing the
    stub makes the next window refetch the day whole: its chunk spans
    [day-1, day+1], so the day comes back with all 4 steps.
    """
    import xarray as xr
    for nc in sorted(full_dir.glob("era5_*.nc")):
        try:
            with xr.open_dataset(nc) as ds:
                n = ds.sizes.get("time", 0)
        except Exception:
            n = 0                      # unreadable/truncated — refetch it
        if n < 4:
            print(f"    pruning partial {nc.name} ({n}/4 steps)", flush=True)
            nc.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description="Stream-rebuild demo activations + ERA5 subsets")
    p.add_argument("--center_start", default="2021-08-24T00")
    p.add_argument("--center_end", default="2021-12-01T00", help="exclusive")
    p.add_argument("--window_days", type=int, default=5,
                   help="center days per window; ~7 GB RAM per day in ds.load()")
    p.add_argument("--step_hours", type=int, default=6)
    p.add_argument("--acts_dir", default="data/activations_raw")
    p.add_argument("--full_dir", default="data/era5_full")
    p.add_argument("--surface_dir", default="data/era5_daily")
    p.add_argument("--levels_dir", default="data/era5_levels")
    p.add_argument("--ckpt_cache", default="data/graphcast_cache")
    p.add_argument("--s3", default="", help="S3 prefix; empty disables upload")
    p.add_argument("--keep_full", action="store_true",
                   help="do not delete full ERA5 days (needs ~383 GB)")
    a = p.parse_args()

    py = sys.executable
    c_start = np.datetime64(a.center_start)
    c_end = np.datetime64(a.center_end)
    win = np.timedelta64(a.window_days, "D")
    step = np.timedelta64(a.step_hours, "h")

    for d in (a.acts_dir, a.full_dir, a.surface_dir, a.levels_dir):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    n_total = int((c_end - c_start) / step)
    print(f"Rebuilding {n_total} timesteps: {c_start} -> {c_end} (exclusive), "
          f"{a.window_days}-day windows\n", flush=True)

    t_start = time.time()
    cursor = c_start
    w = 0
    while cursor < c_end:
        w += 1
        c0 = cursor
        c1 = min(c0 + win, c_end)

        # A center c needs ERA5 for days of c-6h, c and c+6h. Over [c0, c1) the
        # first center needs c0-6h and the last (c1-6h) needs c1.
        start_day = _day(c0 - step)
        end_day = _day(c1)

        print(f"── window {w}: centers {c0} → {c1}  |  ERA5 {start_day}..{end_day}  "
              f"|  free {_free_gb():.0f} GB", flush=True)

        _prune_partial(ROOT / a.full_dir)

        _run([py, str(ROOT / "scripts" / "generate_activations.py"),
              "--start", start_day, "--end", end_day,
              "--center_start", str(c0), "--center_end", str(c1),
              "--step_hours", str(a.step_hours),
              "--acts_dir", a.acts_dir,
              "--era5_dir", a.full_dir,
              "--ckpt_cache", a.ckpt_cache])

        # Full .nc -> slim subsets (skips days already done). Only complete days:
        # a stub boundary day would otherwise be slimmed at 1/4 steps and then
        # cached forever, since era5_slim.py no-ops when the output exists.
        import xarray as xr
        for nc in sorted((ROOT / a.full_dir).glob("era5_*.nc")):
            with xr.open_dataset(nc) as ds:
                if ds.sizes.get("time", 0) < 4:
                    continue
            _run([py, str(ROOT / "scripts" / "era5_slim.py"),
                  "--in_file", str(nc),
                  "--surface_dir", a.surface_dir,
                  "--levels_dir", a.levels_dir])

        if a.s3:
            for local, remote in ((a.acts_dir, "activations"),
                                  (a.surface_dir, "era5-surface"),
                                  (a.levels_dir, "era5-levels")):
                _run(["aws", "s3", "sync", str(ROOT / local), f"{a.s3}/{remote}/",
                      "--only-show-errors"])

        # Drop full days no longer needed: the next window starts at c1 and its
        # earliest center reaches back to c1 - 6h.
        if not a.keep_full:
            keep_from = _day(c1 - step)
            for nc in sorted((ROOT / a.full_dir).glob("era5_*.nc")):
                day = nc.stem.replace("era5_", "")
                if day < keep_from:
                    nc.unlink()

        n_done = len(list((ROOT / a.acts_dir).glob("layer0008_*.npy")))
        el = time.time() - t_start
        rate = n_done / el if el > 0 else 0
        eta = (n_total - n_done) / rate / 60 if rate > 0 else float("nan")
        print(f"   {n_done}/{n_total} timesteps · {el/60:.1f} min elapsed · "
              f"ETA {eta:.0f} min · free {_free_gb():.0f} GB\n", flush=True)

        cursor = c1

    print(f"Done: {len(list((ROOT / a.acts_dir).glob('layer0008_*.npy')))} activation "
          f"files in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
