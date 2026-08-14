"""
Reduce a full ERA5 daily .nc (~3.8 GB, 37 levels, 14 vars) to the two subsets
the demo actually reads, then let the full file be deleted.

  surface/  → viz/globe_render.py  (10m wind, 2m temperature, land-sea mask)
  levels/   → scripts/era5_feature_correlation.py  (11 (var, level) targets)

Variable names and the `level` coordinate are preserved so both consumers work
unmodified — era5_feature_correlation.py does ds[var].sel(level=...).

Usage:
    python scripts/era5_slim.py --in_file data/era5_full/era5_2021-08-24.nc \
        --surface_dir data/era5_daily --levels_dir data/era5_levels
"""

from __future__ import annotations
import argparse
from pathlib import Path

import xarray as xr

# Read by viz/globe_render.py (_open_era5 → load_wind_for_timestamp /
# load_temperature_for_timestamp). land_sea_mask is static (no time dim).
SURFACE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "land_sea_mask",
]

# Union of levels used by ERA5_TARGETS in scripts/era5_feature_correlation.py:
# GPH 850/500, T 300, q 850/700, U 850/200, V 850/200, omega 500.
LEVELS = [200, 300, 500, 700, 850]

LEVEL_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

# MSLP is a surface field but is an ERA5_TARGET, so it rides along in levels/.
LEVEL_EXTRA_VARS = ["mean_sea_level_pressure"]


def _encoding(ds: xr.Dataset) -> dict:
    """zlib-compress every data variable; geophysical float32 packs ~2x."""
    return {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}


def slim(in_file: Path, surface_dir: Path, levels_dir: Path,
         overwrite: bool = False) -> tuple[Path, Path]:
    surface_dir.mkdir(parents=True, exist_ok=True)
    levels_dir.mkdir(parents=True, exist_ok=True)

    out_surface = surface_dir / in_file.name
    out_levels = levels_dir / in_file.name
    if not overwrite and out_surface.exists() and out_levels.exists():
        return out_surface, out_levels

    ds = xr.open_dataset(in_file)

    have = [v for v in SURFACE_VARS if v in ds.data_vars]
    missing = set(SURFACE_VARS) - set(have)
    if missing:
        raise KeyError(f"{in_file.name}: missing surface vars {sorted(missing)}")
    ds_surf = ds[have]
    ds_surf.to_netcdf(out_surface, encoding=_encoding(ds_surf))

    lv = [v for v in LEVEL_VARS if v in ds.data_vars]
    ds_lev = ds[lv].sel(level=[l for l in LEVELS if l in ds.level.values])
    extra = [v for v in LEVEL_EXTRA_VARS if v in ds.data_vars]
    if extra:
        ds_lev = xr.merge([ds_lev, ds[extra]])
    ds_lev.to_netcdf(out_levels, encoding=_encoding(ds_lev))

    ds.close()
    ds_surf.close()
    ds_lev.close()
    return out_surface, out_levels


def main() -> None:
    p = argparse.ArgumentParser(description="Reduce a full ERA5 daily file to demo subsets")
    p.add_argument("--in_file", required=True)
    p.add_argument("--surface_dir", default="data/era5_daily")
    p.add_argument("--levels_dir", default="data/era5_levels")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    s, l = slim(Path(a.in_file), Path(a.surface_dir), Path(a.levels_dir), a.overwrite)
    print(f"{s}  ({s.stat().st_size / 1e6:.1f} MB)")
    print(f"{l}  ({l.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
