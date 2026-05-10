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

COLORMAPS = ["YlOrRd", "plasma", "viridis", "magma", "Reds", "Blues", "RdBu_r"]


def _lon_0_360_to_180(lon: np.ndarray) -> np.ndarray:
    """Convert 0–360 longitude grid to −180–180 for Cartopy."""
    return np.where(lon > 180, lon - 360, lon)


def render_activation_map(
    activation_grid: np.ndarray,     # (n_lat, n_lon) — raw activation values
    grid_lat: np.ndarray,            # (n_lat,)
    grid_lon: np.ndarray,            # (n_lon,) — may be 0–360 or −180–180
    lon_bounds: tuple[float, float], # display window
    lat_bounds: tuple[float, float],
    feature_id: int,
    timestamp: str,
    colormap: str = "YlOrRd",
    threshold: float = 0.0,
    overlay_grid: np.ndarray | None = None,   # physical field for contour overlay
    overlay_label: str = "",
    figsize: tuple[float, float] = (11, 5),
) -> plt.Figure:
    """
    Render a (n_lat, n_lon) activation grid on a Cartopy geographic map.
    Values at or below `threshold` are masked (shown as ocean/land background).
    """
    # Normalise longitudes to −180–180 for Cartopy's PlateCarree
    display_lon = _lon_0_360_to_180(grid_lon)
    lon2, lat2 = np.meshgrid(display_lon, grid_lat)

    vmax = float(np.nanmax(activation_grid))
    if vmax <= threshold:
        vmax = threshold + 1e-6

    masked = np.where(activation_grid > threshold, activation_grid, np.nan)

    fig = plt.figure(figsize=figsize, dpi=130)
    ax = fig.add_subplot(1, 1, 1, projection=PROJECTION)
    ax.set_extent(
        [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
        crs=PROJECTION,
    )

    ax.add_feature(cfeature.LAND, facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#d8eaf5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#333333", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="#888888",
                   linestyle=":", zorder=3)
    ax.add_feature(cfeature.LAKES, facecolor="#d8eaf5", linewidth=0.3, zorder=1)

    # Feature activation heatmap
    pcm = ax.pcolormesh(
        lon2, lat2, masked,
        transform=PROJECTION,
        cmap=colormap,
        vmin=threshold,
        vmax=vmax,
        shading="auto",
        zorder=2,
        alpha=0.82,
    )
    cb = fig.colorbar(pcm, ax=ax, orientation="horizontal",
                      pad=0.05, fraction=0.035, aspect=40)
    cb.set_label("SAE activation", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Optional physical-field overlay as contours
    if overlay_grid is not None and not np.all(np.isnan(overlay_grid)):
        ov_lon = _lon_0_360_to_180(grid_lon)
        ov_lon2, ov_lat2 = np.meshgrid(ov_lon, grid_lat)
        ax.contour(
            ov_lon2, ov_lat2, overlay_grid,
            transform=PROJECTION,
            colors="navy",
            linewidths=0.7,
            levels=8,
            zorder=4,
            alpha=0.55,
        )
        if overlay_label:
            ax.text(
                0.01, 0.97, f"Contours: {overlay_label}",
                transform=ax.transAxes,
                fontsize=7, color="navy",
                va="top", ha="left",
            )

    ax.set_title(
        f"Feature {feature_id}  ·  {timestamp}",
        fontsize=11, fontweight="semibold", pad=6,
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    fig.tight_layout(pad=1.0)
    return fig
