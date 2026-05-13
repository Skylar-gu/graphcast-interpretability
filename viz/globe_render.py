"""
3D rotating globe for SAE feature activations + ERA5 10m wind.
Rendered via globe.gl (WebGL/Three.js) using st.components.v1.html().
"""

from __future__ import annotations
import json
import numpy as np
import matplotlib.cm as cm
import xarray as xr
from pathlib import Path

ERA5_DIR = Path(__file__).parent.parent / "data" / "era5_daily"


# ── ERA5 data loading ─────────────────────────────────────────────────────────

def _open_era5(timestamp: str) -> tuple[xr.Dataset, int]:
    date_str = timestamp[:10]
    nc_path = ERA5_DIR / f"era5_{date_str}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)
    ds = xr.open_dataset(str(nc_path))
    t_str = timestamp[11:13] if len(timestamp) > 10 else "00"
    return ds, int(t_str) // 6


def load_wind_for_timestamp(timestamp: str) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """Return (u10, v10, lat, lon) — lat ascending -90→90, lon 0→360."""
    ds, t_idx = _open_era5(timestamp)
    u10 = ds["10m_u_component_of_wind"].isel(time=t_idx).values.astype(np.float32)
    v10 = ds["10m_v_component_of_wind"].isel(time=t_idx).values.astype(np.float32)
    lat = ds["lat"].values.astype(np.float32)
    lon = ds["lon"].values.astype(np.float32)
    ds.close()
    return u10, v10, lat, lon


def load_temperature_for_timestamp(timestamp: str) -> tuple[np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
    """
    Return (temp_celsius, land_sea_mask, lat, lon).
    land_sea_mask: 1=land, 0=ocean.  lat ascending -90→90.
    """
    ds, t_idx = _open_era5(timestamp)
    temp_k = ds["2m_temperature"].isel(time=t_idx).values.astype(np.float32)
    lsm    = ds["land_sea_mask"].values.astype(np.float32)
    lat    = ds["lat"].values.astype(np.float32)
    lon    = ds["lon"].values.astype(np.float32)
    ds.close()
    return temp_k - 273.15, lsm, lat, lon


# ── Point builders ────────────────────────────────────────────────────────────

def _lon360_to_180(lon: float) -> float:
    return float(lon - 360.0 if lon > 180.0 else lon)


def activation_to_points(
    activation_grid: np.ndarray,   # (nlat, nlon), grid_lat descending 90→-90
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,           # 0→360
    threshold: float = 0.0,
    step: int = 4,
    # unused kwargs kept for call-site compatibility
    vmax: float | None = None,
    colormap: str = "GnBu",
    max_opacity: float = 0.85,
) -> list[dict]:
    """Return [{lat, lng, val}, …] for globe.gl hexBinPointsData."""
    points = []
    for i in range(0, len(grid_lat), step):
        for j in range(0, len(grid_lon), step):
            val = float(activation_grid[i, j])
            if val <= threshold:
                continue
            points.append({
                "lat": float(grid_lat[i]),
                "lng": _lon360_to_180(float(grid_lon[j])),
                "val": val,
            })
    return points


def temperature_to_points(
    temp_c: np.ndarray,     # (nlat, nlon) Celsius, ERA5 lat ascending -90→90
    lsm: np.ndarray,         # (nlat, nlon) land-sea mask
    era5_lat: np.ndarray,
    era5_lon: np.ndarray,    # 0→360
    mode: str = "both",      # "both" | "land" | "sst"
    step: int = 6,
) -> list[dict]:
    """Return [{lat, lng, val}, …] temperature points for hexBinPointsData."""
    points = []
    for i in range(0, len(era5_lat), step):
        for j in range(0, len(era5_lon), step):
            if mode == "land" and lsm[i, j] < 0.5:
                continue
            if mode == "sst" and lsm[i, j] >= 0.5:
                continue
            val = float(temp_c[i, j])
            if np.isnan(val):
                continue
            points.append({
                "lat": float(era5_lat[i]),
                "lng": _lon360_to_180(float(era5_lon[j])),
                "val": val,
            })
    return points


# ── Wind arrows ───────────────────────────────────────────────────────────────

def compute_wind_arrows(
    u10: np.ndarray,
    v10: np.ndarray,
    era5_lat: np.ndarray,
    era5_lon: np.ndarray,
    seed_step_deg: float = 3.0,
    speed_scale: float   = 0.12,
    min_speed: float     = 4.0,
    min_disp_deg: float  = 0.5,
    max_arrow_deg: float = 4.0,
) -> list[dict]:
    """Wind vectors as arcs for globe.gl arcsData, coloured by speed."""
    seed_lats     = np.arange(-80, 81,  seed_step_deg)
    seed_lons_360 = np.arange(0,   360, seed_step_deg)

    def _color(s: float) -> str:
        t = min(1.0, s / 18.0)
        r = int(140 + t * 115)
        g = int(180 + t * 75)
        b = int(220 + t * 35)
        a = 0.12 + t * 0.55
        return f"rgba({r},{g},{b},{a:.2f})"

    arrows = []
    for slat in seed_lats:
        for slon_360 in seed_lons_360:
            i = int(np.argmin(np.abs(era5_lat - slat)))
            j = int(np.argmin(np.abs(era5_lon - slon_360)))
            u = float(u10[i, j])
            v = float(v10[i, j])
            speed = float(np.sqrt(u**2 + v**2))
            if speed < min_speed:
                continue
            cos_lat = max(float(np.cos(np.radians(slat))), 0.05)
            dlat = float(np.clip(v * speed_scale, -max_arrow_deg, max_arrow_deg))
            dlng = float(np.clip(u * speed_scale / cos_lat, -max_arrow_deg, max_arrow_deg))
            if (dlat**2 + dlng**2)**0.5 < min_disp_deg:
                continue
            arrows.append({
                "startLat": float(slat),
                "startLng": _lon360_to_180(float(slon_360)),
                "endLat":   float(np.clip(slat + dlat, -85.0, 85.0)),
                "endLng":   float((_lon360_to_180(slon_360) + dlng + 180) % 360 - 180),
                "color":    _color(speed),
            })
    return arrows


# ── Colorscale helpers ────────────────────────────────────────────────────────

def _colorscale_js(colormap: str, n: int = 16) -> str:
    cmap = cm.get_cmap(colormap)
    stops = [[int(v * 255) for v in cmap(i / (n - 1))[:3]] for i in range(n)]
    return json.dumps(stops)


def _css_gradient(colormap: str, n: int = 10) -> str:
    """Top of bar = vmax (darkest/hottest), bottom = vmin (lightest/coldest)."""
    cmap = cm.get_cmap(colormap)
    parts = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b, _ = cmap(1.0 - t)
        parts.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)}) {int(t*100)}%")
    return f"linear-gradient(to bottom, {', '.join(parts)})"


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_globe_html(
    hex_points:      list[dict],   # [{lat, lng, val}, …]
    wind_arrows:     list[dict],
    feature_id:      int,
    timestamp:       str,
    colormap:        str   = "GnBu",
    vmin:            float = 0.0,
    vmax:            float = 1.0,
    hex_resolution:  int   = 3,
    colorbar_title:  str   = "SAE activation",
    height:          int   = 600,
    # legacy compat
    act_points:      list | None = None,
    threshold:       float = 0.0,
) -> str:
    if act_points is not None:          # backwards-compat shim
        hex_points = act_points

    pts_json  = json.dumps(hex_points)
    wind_json = json.dumps(wind_arrows)
    n_pts     = len(hex_points)
    n_wind    = len(wind_arrows)

    cs_js      = _colorscale_js(colormap)
    css_grad   = _css_gradient(colormap)
    label_top  = f"{vmax:.2f}"
    label_mid  = f"{(vmin + vmax) / 2:.2f}"
    label_bot  = f"{vmin:.2f}"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: 100%; height: {height}px; background: #000010; overflow: hidden; }}
  #globeViz {{ width: 100%; height: {height}px; }}

  #badge {{
    position: absolute; top: 10px; left: 12px; z-index: 10;
    color: #c8daea; font: 12px/1.6 monospace;
    background: rgba(0,0,16,0.65); padding: 5px 10px;
    border-radius: 4px; pointer-events: none;
    border: 1px solid rgba(100,150,200,0.25);
  }}
  #legend {{
    position: absolute; bottom: 14px; left: 12px; z-index: 10;
    color: #a0b8cc; font: 10px/1.5 monospace; pointer-events: none;
  }}
  #err {{
    display: none; position: absolute; top: 50%; left: 50%;
    transform: translate(-50%,-50%); z-index: 20;
    color: #f08080; font: 13px monospace; text-align: center;
    background: rgba(0,0,16,0.85); padding: 16px 24px; border-radius: 6px;
  }}
  #colorbar {{
    position: absolute; right: 18px; top: 5%; height: 90%;
    z-index: 10; display: flex; flex-direction: row;
    align-items: stretch; gap: 5px; pointer-events: none;
  }}
  #cb-bar {{
    width: 14px; border-radius: 3px;
    background: {css_grad};
    border: 1px solid rgba(100,150,200,0.3);
  }}
  #cb-labels {{
    display: flex; flex-direction: column;
    justify-content: space-between;
    color: #c8daea; font: 10px/1 monospace;
  }}
  #cb-title {{
    position: absolute; right: 52px; top: 50%;
    transform: translateY(-50%) rotate(-90deg);
    transform-origin: center center;
    color: #a0b8cc; font: 10px monospace;
    white-space: nowrap; pointer-events: none; z-index: 10;
  }}
</style>
</head>
<body>
<div id="globeViz"></div>

<div id="badge">Feature {feature_id} &nbsp;·&nbsp; {timestamp}</div>

<div id="legend">
  <span style="display:inline-block;width:9px;height:9px;border-radius:2px;
    background:rgba(200,230,255,0.45);margin-right:4px;vertical-align:middle"></span>
  wind (brightness = speed)
</div>

<div id="colorbar">
  <div id="cb-bar"></div>
  <div id="cb-labels">
    <span>{label_top}</span>
    <span>{label_mid}</span>
    <span>{label_bot}</span>
  </div>
</div>
<div id="cb-title">{colorbar_title}</div>

<div id="err"></div>

<script>
const HEX_PTS    = {pts_json};
const WIND_ARROWS = {wind_json};
const VMIN = {vmin};
const VMAX = {vmax};
const CS   = {cs_js};

function sampleCS(t) {{
  t = Math.max(0, Math.min(1, t));
  const idx = t * (CS.length - 1);
  const lo  = Math.floor(idx);
  const hi  = Math.min(CS.length - 1, lo + 1);
  const f   = idx - lo;
  const r   = CS[lo][0] + (CS[hi][0] - CS[lo][0]) * f;
  const g   = CS[lo][1] + (CS[hi][1] - CS[lo][1]) * f;
  const b   = CS[lo][2] + (CS[hi][2] - CS[lo][2]) * f;
  const a   = 0.25 + t * 0.65;
  return `rgba(${{r|0}},${{g|0}},${{b|0}},${{a.toFixed(2)}})`;
}}

function hexColor(d) {{
  const mean = d.sumWeight / Math.max(1, d.points.length);
  const t    = (mean - VMIN) / Math.max(1e-6, VMAX - VMIN);
  return sampleCS(t);
}}

function initGlobe() {{
  const W = window.innerWidth;
  const H = {height};

  const globe = Globe()
    .width(W).height(H)
    (document.getElementById('globeViz'))
    .globeImageUrl('https://unpkg.com/three-globe@2.45.2/example/img/earth-dark.jpg')
    .backgroundImageUrl('https://unpkg.com/three-globe@2.45.2/example/img/night-sky.png')
    .showAtmosphere(true)
    .atmosphereColor('#1a4a7a')
    .atmosphereAltitude(0.13)
    /* ── Wind ── */
    .arcsData(WIND_ARROWS)
    .arcStartLat(d => d.startLat).arcStartLng(d => d.startLng)
    .arcEndLat(d => d.endLat).arcEndLng(d => d.endLng)
    .arcColor(d => d.color)
    .arcAltitude(0.003).arcStroke(0.28)
    .arcDashLength(0.55).arcDashGap(0.45).arcDashAnimateTime(2200)
    /* ── Hex overlay ── */
    .hexBinPointsData(HEX_PTS)
    .hexBinPointLat(d => d.lat)
    .hexBinPointLng(d => d.lng)
    .hexBinPointWeight(d => d.val)
    .hexBinResolution({hex_resolution})
    .hexBinMerge(false)
    .hexAltitude(0.015)
    .hexMargin(0.2)
    .hexTopColor(hexColor)
    .hexSideColor(() => 'rgba(0,0,0,0)');

  globe.controls().autoRotate      = true;
  globe.controls().autoRotateSpeed = 0.32;
  globe.pointOfView({{ lat: 22, lng: -65, altitude: 2.0 }}, 0);

  window.addEventListener('resize', () => globe.width(window.innerWidth));
}}

function onLoadError() {{
  const el = document.getElementById('err');
  el.style.display = 'block';
  el.textContent = 'Could not load globe.gl from CDN.\\nCheck network access and reload.';
}}
</script>
<script src="https://unpkg.com/globe.gl@2.45.3/dist/globe.gl.min.js"
        onload="initGlobe()" onerror="onLoadError()"></script>
</body>
</html>"""
