"""Geographic map rendering for SAE feature activations using Cartopy."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
