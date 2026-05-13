"""Geographic map rendering for SAE feature activations."""

from __future__ import annotations
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objects as go

PROJECTION = ccrs.PlateCarree()

# Named region presets: (lon_min, lon_max, lat_min, lat_max)
REGION_PRESETS: dict[str, tuple[float, float, float, float] | None] = {
    "Global": (-180, 180, -90, 90),
    "North Atlantic / Gulf": (-100, -50, 10, 55),
    "North Pacific / US West": (-180, -100, 15, 65),
    "Arctic": (-180, 180, 55, 90),
    "Indo-Pacific": (60, 180, -30, 30),
    "Europe / North Africa": (-15, 50, 25, 75),
    "Custom": None,
}

COLORMAPS = ["GnBu", "Blues", "YlGnBu", "bone", "gray_r", "cividis", "PuBu", "twilight_shifted"]

# Plotly colorscale equivalents (used for the interactive map)
_PLOTLY_CS: dict[str, str] = {
    "GnBu": "GnBu",
    "Blues": "Blues",
    "YlGnBu": "YlGnBu",
    "bone": "Greys",
    "gray_r": "Greys",
    "cividis": "Cividis",
    "PuBu": "PuBu",
    "twilight_shifted": "IceFire",
}

# Dark, dramatic basemap — deep slate ocean, charcoal land
_LAND   = cfeature.NaturalEarthFeature("physical", "land",   "50m",
                                        facecolor="#2e2e2e", edgecolor="none")
_OCEAN  = cfeature.NaturalEarthFeature("physical", "ocean",  "50m",
                                        facecolor="#0d1b2a", edgecolor="none")
_COAST  = cfeature.NaturalEarthFeature("physical", "coastline", "50m",
                                        facecolor="none", edgecolor="#6a8faf",
                                        linewidth=0.7)
_LAKES  = cfeature.NaturalEarthFeature("physical", "lakes",  "50m",
                                        facecolor="#0d1b2a", edgecolor="#6a8faf",
                                        linewidth=0.3)
_RIVERS = cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "50m",
                                        facecolor="none", edgecolor="#3a5f7a",
                                        linewidth=0.3)
_BORDERS = cfeature.NaturalEarthFeature("cultural",
                                         "admin_0_boundary_lines_land", "50m",
                                         facecolor="none", edgecolor="#4a4a4a",
                                         linewidth=0.3, linestyle="--")


def _lon_0_360_to_180(lon: np.ndarray) -> np.ndarray:
    return np.where(lon > 180, lon - 360, lon)


@functools.lru_cache(maxsize=2)
def _geo_land_coast() -> tuple[list, list, list, list]:
    """Return (land_lons, land_lats, coast_lons, coast_lats) as NaN-separated lists.
    Extracted from Natural Earth via cartopy; result is module-level cached."""
    def _extract_polygons(feature):
        lons, lats = [], []
        for geom in feature.geometries():
            parts = list(getattr(geom, 'geoms', [geom]))
            for p in parts:
                try:
                    c = np.array(p.exterior.coords)
                    lons.extend(c[:, 0].tolist() + [None])
                    lats.extend(c[:, 1].tolist() + [None])
                except Exception:
                    pass
        return lons, lats

    def _extract_lines(feature):
        lons, lats = [], []
        for geom in feature.geometries():
            parts = list(getattr(geom, 'geoms', [geom]))
            for p in parts:
                try:
                    c = np.array(p.coords)
                    lons.extend(c[:, 0].tolist() + [None])
                    lats.extend(c[:, 1].tolist() + [None])
                except Exception:
                    pass
        return lons, lats

    land_lons, land_lats = _extract_polygons(
        cfeature.NaturalEarthFeature("physical", "land", "110m", facecolor="none"))
    coast_lons, coast_lats = _extract_lines(
        cfeature.NaturalEarthFeature("physical", "coastline", "50m", facecolor="none"))
    return land_lons, land_lats, coast_lons, coast_lats


def _sort_lon_grid(display_lon, masked):
    """Re-sort columns so lon increases monotonically (needed for imshow extent)."""
    order = np.argsort(display_lon)
    return display_lon[order], masked[:, order]


def render_activation_map(
    activation_grid: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    feature_id: int,
    timestamp: str,
    colormap: str = "YlOrRd",
    threshold: float = 0.0,
    overlay_grid: np.ndarray | None = None,
    overlay_label: str = "",
    figsize: tuple[float, float] = (12, 5.5),
) -> plt.Figure:
    display_lon = _lon_0_360_to_180(grid_lon)

    vmax = float(np.nanmax(activation_grid))
    if vmax <= threshold:
        vmax = threshold + 1e-6

    masked = np.where(activation_grid > threshold, activation_grid, np.nan)

    # Sort lon for imshow (must be monotonically increasing)
    sorted_lon, sorted_masked = _sort_lon_grid(display_lon, masked)

    # lat goes 90→-90; imshow origin='upper' matches that
    lat_min, lat_max = float(grid_lat.min()), float(grid_lat.max())
    lon_min_data, lon_max_data = float(sorted_lon.min()), float(sorted_lon.max())

    fig = plt.figure(figsize=figsize, dpi=150, facecolor="#0a1520")
    ax = fig.add_axes([0.04, 0.06, 0.88, 0.86], projection=PROJECTION)
    ax.set_facecolor("#0d1b2a")
    ax.set_extent(
        [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
        crs=PROJECTION,
    )

    # Base layers
    ax.add_feature(_OCEAN,   zorder=0)
    ax.add_feature(_LAND,    zorder=1)
    ax.add_feature(_LAKES,   zorder=2)
    ax.add_feature(_RIVERS,  zorder=3)
    ax.add_feature(_BORDERS, zorder=4)
    ax.add_feature(_COAST,   zorder=5)

    # Activation heatmap — imshow gives bilinear smoothing for free
    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad(alpha=0)          # NaN (masked) → fully transparent
    norm = mcolors.Normalize(vmin=threshold, vmax=vmax)

    im = ax.imshow(
        sorted_masked,
        origin="upper",
        extent=[lon_min_data, lon_max_data, lat_min, lat_max],
        transform=PROJECTION,
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        alpha=0.80,
        zorder=6,
    )

    # Colorbar — vertical, right side
    cax = fig.add_axes([0.93, 0.12, 0.018, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("SAE activation", fontsize=8, labelpad=6, color="#c8daea")
    cb.ax.tick_params(labelsize=7, colors="#c8daea")
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#c8daea")

    # Contour overlay
    if overlay_grid is not None and not np.all(np.isnan(overlay_grid)):
        ov_lon_sorted, ov_grid_sorted = _sort_lon_grid(display_lon, overlay_grid)
        ov_lon2, ov_lat2 = np.meshgrid(ov_lon_sorted, grid_lat)
        ax.contour(
            ov_lon2, ov_lat2, ov_grid_sorted,
            transform=PROJECTION,
            colors="#003366",
            linewidths=0.6,
            levels=8,
            zorder=7,
            alpha=0.6,
        )
        if overlay_label:
            ax.text(0.01, 0.98, f"Contours: {overlay_label}",
                    transform=ax.transAxes, fontsize=7, color="#003366",
                    va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    ax.set_title(
        f"Feature {feature_id}  ·  {timestamp}",
        fontsize=11, fontweight="semibold", pad=8, color="#c8daea",
    )

    gl = ax.gridlines(linewidth=0.2, color="#2a4a6a", alpha=0.6, linestyle=":")
    gl.top_labels = gl.right_labels = gl.bottom_labels = gl.left_labels = False

    return fig


def render_activation_map_plotly(
    activation_grid: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    feature_id: int,
    timestamp: str,
    colormap: str = "GnBu",
    threshold: float = 0.0,
    vmax: float | None = None,
    height: int = 520,
) -> go.Figure:
    """
    Interactive zoomable activation map using go.Heatmap (no density smoothing).
    Grid cells are rendered accurately at their true positions.
    """
    display_lon = _lon_0_360_to_180(grid_lon)
    masked = np.where(activation_grid > threshold, activation_grid, np.nan)
    sorted_lon, sorted_masked = _sort_lon_grid(display_lon, masked)

    _zmax = float(vmax) if vmax is not None else (
        float(np.nanmax(activation_grid)) if not np.all(np.isnan(activation_grid)) else 1.0)
    colorscale = _PLOTLY_CS.get(colormap, colormap)

    land_lons, land_lats, coast_lons, coast_lats = _geo_land_coast()

    fig = go.Figure()

    # Land fill (behind heatmap — NaN heatmap cells are transparent)
    fig.add_trace(go.Scatter(
        x=land_lons, y=land_lats,
        fill='toself',
        fillcolor='#2e2e2e',
        line=dict(color='#2e2e2e', width=0),
        mode='lines',
        showlegend=False,
        hoverinfo='skip',
    ))

    # Activation heatmap — raw grid, no smoothing
    fig.add_trace(go.Heatmap(
        z=sorted_masked,
        x=sorted_lon.tolist(),
        y=grid_lat.tolist(),
        colorscale=colorscale,
        zmin=float(threshold),
        zmax=_zmax,
        zsmooth=False,
        connectgaps=False,
        opacity=0.85,
        showscale=True,
        colorbar=dict(
            title=dict(text="SAE activation", side="right", font=dict(size=11)),
            thickness=14, len=0.75, x=1.01,
        ),
        hovertemplate="lat: %{y:.2f}°<br>lon: %{x:.2f}°<br>activation: %{z:.3f}<extra></extra>",
    ))

    # Coastlines on top
    fig.add_trace(go.Scatter(
        x=coast_lons, y=coast_lats,
        mode='lines',
        line=dict(color='#6a8faf', width=0.7),
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.update_layout(
        xaxis=dict(
            range=[lon_bounds[0], lon_bounds[1]],
            showgrid=False, zeroline=False,
            showline=False, showticklabels=False, ticks='',
        ),
        yaxis=dict(
            range=[lat_bounds[0], lat_bounds[1]],
            showgrid=False, zeroline=False,
            showline=False, showticklabels=False, ticks='',
        ),
        height=height,
        margin=dict(l=0, r=60, t=36, b=0),
        title=dict(
            text=f"Feature {feature_id}  ·  {timestamp}",
            font=dict(size=13), x=0.02, xanchor="left",
        ),
        paper_bgcolor="#0a1520",
        plot_bgcolor="#0d1b2a",
        font_color="#c8daea",
        dragmode='pan',
    )
    return fig
