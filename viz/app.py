"""
GraphCast SAE Feature Atlas — interactive Streamlit visualization tool.

Run with:
    streamlit run viz/app.py

Requires precomputed data in viz/data/ (run precompute.py first).
URL state is encoded so views can be bookmarked and shared.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))           # graphcast_interpretability
sys.path.insert(0, str(Path(__file__).parent))  # viz/

from graphcast_interpretability.interpolate_utils import GridMeshMapper  # noqa: E402
from data_loader import (  # noqa: E402
    load_registry, load_catalog, load_grid,
    get_feature_activation, get_feature_timeseries, get_timestamps,
)
from map_render import render_activation_map, REGION_PRESETS, COLORMAPS  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphCast SAE Feature Atlas",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building mesh→grid interpolation weights (first run only)…")
def get_mapper() -> GridMeshMapper:
    """Build or load cached GridMeshMapper. Expensive first call; fast thereafter."""
    lat, lon = load_grid()
    mapper = GridMeshMapper(
        grid_lat=lat,
        grid_lon=lon,
        splits=6,
        cache_dir=str(DATA_DIR / "graph" / "mapper_cache"),
    )
    mapper.precompute_mesh_to_grid()
    return mapper


@st.cache_data(show_spinner="Interpolating activation to lat/lon grid…")
def cached_regrid(sae_id: str, t_idx: int, feature_id: int, n_nodes: int) -> np.ndarray:
    """Interpolate one (feature, timestep) to a lat/lon grid. Result is cached."""
    mapper = get_mapper()
    mesh_field = get_feature_activation(sae_id, t_idx, feature_id, n_nodes)
    return mapper.apply_mesh_to_grid(mesh_field, method="mean")


@st.cache_data(show_spinner="Computing time series…")
def cached_timeseries(sae_id: str, feature_id: int, n_time: int):
    """Mean activation per timestep for a feature. Cached per (sae_id, feature_id)."""
    return get_feature_timeseries(sae_id, feature_id, n_time)


# ── Load top-level data ───────────────────────────────────────────────────────

registry = load_registry()
catalog = load_catalog()

if not registry:
    st.error(
        "**No preprocessed SAE data found** in `viz/data/`.\n\n"
        "Run the preprocessing pipeline first:\n"
        "```\n"
        "python viz/precompute.py --acts_dir /path/to/layer8_npy_files\n"
        "python viz/compute_stats.py\n"
        "```"
    )
    st.stop()

# ── Read URL state (for shareable links) ──────────────────────────────────────
qp = st.query_params
_default_sae = qp.get("sae_id", list(registry.keys())[0])
_default_feature = int(qp.get("feature_id", 3243))
_default_t = int(qp.get("t_idx", 0))

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — shared controls
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("SAE Feature Atlas")
    st.caption("GraphCast mechanistic interpretability")

    # ── SAE / layer ───────────────────────────────────────────────────────
    st.subheader("Model / SAE")
    sae_options = list(registry.keys())
    sae_id = st.selectbox(
        "SAE Checkpoint",
        options=sae_options,
        index=sae_options.index(_default_sae) if _default_sae in sae_options else 0,
        key="sae_id",
    )
    info = registry[sae_id]
    timestamps = get_timestamps(sae_id)
    st.caption(
        f"Layer {info['layer']} · k={info['k_active']} · "
        f"latent={info['latent']} · {info['n_time']} timesteps"
    )

    st.divider()

    # ── Feature selection ─────────────────────────────────────────────────
    st.subheader("Feature")
    feature_id = st.number_input(
        "Feature ID",
        min_value=0,
        max_value=info["latent"] - 1,
        value=min(_default_feature, info["latent"] - 1),
        step=1,
        key="feature_id_input",
    )

    # Quick-jump to named features
    if not catalog.empty:
        named = catalog[catalog["candidate_label"] != ""][["feature_id", "candidate_label"]]
        if not named.empty:
            options = ["— jump to known feature —"] + [
                f"{int(r.feature_id)}: {r.candidate_label}"
                for _, r in named.iterrows()
            ]
            jump = st.selectbox("Known features", options, index=0)
            if jump != options[0]:
                feature_id = int(jump.split(":")[0])

    st.divider()

    # ── Time ──────────────────────────────────────────────────────────────
    st.subheader("Timestamp")
    t_idx = st.select_slider(
        "Select timestep",
        options=list(range(len(timestamps))),
        value=min(_default_t, len(timestamps) - 1),
        format_func=lambda i: timestamps[i],
        key="t_slider",
    )
    timestamp = timestamps[t_idx]

    st.divider()

    # ── Map view ──────────────────────────────────────────────────────────
    st.subheader("Map view")
    region_name = st.selectbox("Region preset", options=list(REGION_PRESETS.keys()))
    region = REGION_PRESETS[region_name]

    if region is None:
        c1, c2 = st.columns(2)
        with c1:
            lon_min = st.number_input("Lon min", value=-180.0)
            lat_min = st.number_input("Lat min", value=-90.0)
        with c2:
            lon_max = st.number_input("Lon max", value=180.0)
            lat_max = st.number_input("Lat max", value=90.0)
    else:
        lon_min, lon_max, lat_min, lat_max = region

    colormap = st.selectbox("Colormap", COLORMAPS)
    threshold = st.slider("Activation threshold (mask below)", 0.0, 5.0, 0.0, 0.05)

    st.divider()
    st.caption("URL encodes current view for sharing.")

# Write URL state
st.query_params["sae_id"] = sae_id
st.query_params["feature_id"] = str(feature_id)
st.query_params["t_idx"] = str(t_idx)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_inspector, tab_atlas, tab_events = st.tabs(
    ["Feature Inspector", "Feature Atlas", "Event Browser"]
)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — Feature Inspector (main view)
# ───────────────────────────────────────────────────────────────────────────────

with tab_inspector:
    # ── Compute activation grid ────────────────────────────────────────────
    grid_ok = False
    activation_grid = None
    try:
        activation_grid = cached_regrid(sae_id, t_idx, feature_id, info["n_nodes"])
        grid_ok = True
    except Exception as e:
        st.error(f"Interpolation failed: {e}")

    col_map, col_meta = st.columns([3, 1], gap="medium")

    with col_map:
        st.subheader(f"Feature {feature_id}  ·  {timestamp}")
        if grid_ok:
            grid_lat, grid_lon = load_grid()
            fig = render_activation_map(
                activation_grid=activation_grid,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                lon_bounds=(lon_min, lon_max),
                lat_bounds=(lat_min, lat_max),
                feature_id=feature_id,
                timestamp=timestamp,
                colormap=colormap,
                threshold=threshold,
            )
            st.pyplot(fig, use_container_width=True)
            plt = fig  # keep reference to allow tight layout
        else:
            st.empty()

    with col_meta:
        st.subheader("Feature metadata")

        # ── Catalog stats ──────────────────────────────────────────────────
        if not catalog.empty and feature_id in catalog["feature_id"].values:
            row = catalog[catalog["feature_id"] == feature_id].iloc[0]

            if row.get("candidate_label", ""):
                st.info(f"**{row['candidate_label']}**")
            if row.get("dead", False):
                st.warning("Dead feature (never activates in this dataset)")

            st.metric("Activation frequency", f"{row['activation_frequency']:.5f}")
            st.metric("Mean activation", f"{row['mean_activation']:.4f}")
            st.metric("Max activation", f"{row['max_activation']:.3f}")
            st.metric(
                "Artifact score",
                f"{row['artifact_score']:.3f}",
                help="High = consistently fires at same nodes regardless of weather state "
                     "(likely grid-locked artifact).",
            )

            top_ts = row.get("top_timestamps", [])
            if top_ts:
                st.write("**Top activation timestamps:**")
                for ts_ex in top_ts:
                    st.code(ts_ex, language=None)
        else:
            st.info(
                "Feature catalog not available. "
                "Run `python viz/compute_stats.py` to populate metadata."
            )

        # ── Current snapshot stats ─────────────────────────────────────────
        if grid_ok:
            st.divider()
            st.caption("Current snapshot")
            active = int((activation_grid > threshold).sum())
            snap_max = float(np.nanmax(activation_grid))
            st.metric("Active grid cells", f"{active:,}")
            st.metric("Max (this snapshot)", f"{snap_max:.3f}")

    # ── Time series ────────────────────────────────────────────────────────
    st.subheader("Global mean activation over time")
    ts_labels, ts_means = cached_timeseries(sae_id, feature_id, info["n_time"])

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=list(ts_labels),
        y=ts_means.tolist(),
        mode="lines+markers",
        marker=dict(size=5, color="#e06c00"),
        line=dict(color="#e06c00", width=1.5),
        name=f"Feature {feature_id}",
        hovertemplate="<b>%{x}</b><br>Mean activation: %{y:.4f}<extra></extra>",
    ))
    # Vertical line for current selection
    fig_ts.add_vline(
        x=timestamp,
        line_dash="dash",
        line_color="steelblue",
        annotation_text="viewing",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig_ts.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Mean activation (active nodes)",
        height=220,
        margin=dict(l=45, r=15, t=15, b=45),
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig_ts, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Feature Atlas (sortable / filterable catalog)
# ───────────────────────────────────────────────────────────────────────────────

with tab_atlas:
    st.header("Feature Atlas")
    st.caption(
        "Sortable catalog of all SAE features. Click a row to jump to it in the inspector."
    )

    if catalog.empty:
        st.warning(
            "Feature catalog not found. "
            "Run `python viz/compute_stats.py --sae_id` to generate it."
        )
    else:
        # ── Filters ───────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 2, 1, 2])
        with fc1:
            freq_range = st.slider("Activation frequency ≤", 0.0, 1.0, 1.0, 0.0001,
                                   format="%.4f")
        with fc2:
            artifact_max_filter = st.slider("Artifact score ≤", 0.0, 1.0, 1.0, 0.01)
        with fc3:
            min_max_act = st.slider("Min max-activation ≥", 0.0,
                                    float(catalog["max_activation"].max()), 0.0, 0.1)
        with fc4:
            hide_dead = st.checkbox("Hide dead", value=True)
        with fc5:
            label_search = st.text_input("Search label", "")

        df = catalog.copy()
        df = df[df["activation_frequency"] <= freq_range]
        df = df[df["artifact_score"] <= artifact_max_filter]
        df = df[df["max_activation"] >= min_max_act]
        if hide_dead:
            df = df[~df["dead"]]
        if label_search:
            df = df[df["candidate_label"].str.contains(label_search, case=False, na=False)]

        sort_col = st.selectbox(
            "Sort by",
            ["max_activation", "activation_frequency", "artifact_score", "feature_id"],
            index=0,
        )
        ascending = st.checkbox("Ascending", value=False)
        df = df.sort_values(sort_col, ascending=ascending)

        display_cols = [
            "feature_id", "candidate_label",
            "activation_frequency", "mean_activation", "max_activation",
            "artifact_score", "dead",
        ]
        st.dataframe(
            df[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=480,
            column_config={
                "feature_id": st.column_config.NumberColumn("ID", width="small"),
                "activation_frequency": st.column_config.NumberColumn(
                    "Freq", format="%.5f"),
                "mean_activation": st.column_config.NumberColumn(
                    "Mean act", format="%.4f"),
                "max_activation": st.column_config.NumberColumn(
                    "Max act", format="%.3f"),
                "artifact_score": st.column_config.ProgressColumn(
                    "Artifact", min_value=0, max_value=1, format="%.3f"),
                "candidate_label": st.column_config.TextColumn("Label", width="large"),
            },
        )
        st.caption(f"Showing {len(df):,} / {len(catalog):,} features")

        # ── Activation frequency histogram ─────────────────────────────────
        st.subheader("Activation frequency distribution")
        nonzero = df[~df["dead"]]["activation_frequency"]
        if len(nonzero) > 0:
            fig_hist = go.Figure(go.Histogram(
                x=np.log10(nonzero + 1e-9),
                nbinsx=60,
                marker_color="#4a90d9",
                opacity=0.8,
            ))
            fig_hist.update_layout(
                xaxis_title="log₁₀(activation frequency)",
                yaxis_title="# features",
                height=220,
                margin=dict(l=45, r=15, t=15, b=45),
                template="plotly_white",
            )
            st.plotly_chart(fig_hist, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — Event Browser
# ───────────────────────────────────────────────────────────────────────────────

KNOWN_EVENTS: dict[str, dict] = {
    "Hurricane Ida — 2021-08-29": {
        "timestamps": ["2021-08-29T00", "2021-08-29T06", "2021-08-29T12", "2021-08-29T18"],
        "region": "North Atlantic / Gulf",
        "features": [3243],
        "note": (
            "Feature 3243 activates along the hurricane track. "
            "Intervention experiments show causal control over storm intensity."
        ),
    },
    "Pineapple Express — 2019-02-14": {
        "timestamps": ["2019-02-14T00", "2019-02-14T06", "2019-02-14T12"],
        "region": "North Pacific / US West",
        "features": [1820],
        "note": (
            "Feature 1820 tracks the IVT plume from Hawaii to the US West Coast. "
            "Compare against IVT and TPW to distinguish vapor transport from generic moisture."
        ),
    },
}

with tab_events:
    st.header("Event Browser")
    st.caption(
        "Inspect features during known meteorological events. "
        "Timestamps outside your precomputed range will be unavailable."
    )

    event_name = st.selectbox("Event", list(KNOWN_EVENTS.keys()))
    event = KNOWN_EVENTS[event_name]

    st.info(event["note"])

    avail_ts = [t for t in event["timestamps"] if t in timestamps]
    if not avail_ts:
        st.warning(
            f"None of the event timestamps ({', '.join(event['timestamps'])}) "
            "are in your precomputed data.\n\n"
            "Re-run `precompute.py` with an `--acts_dir` that includes these dates."
        )
    else:
        ec1, ec2 = st.columns(2)
        with ec1:
            ev_ts = st.selectbox("Timestamp", avail_ts)
        with ec2:
            ev_feat = st.number_input(
                "Feature ID",
                min_value=0, max_value=info["latent"] - 1,
                value=event["features"][0],
                key="ev_feat",
            )

        ev_t_idx = timestamps.index(ev_ts)
        ev_region = REGION_PRESETS.get(event["region"]) or (-180, 180, -90, 90)
        ev_lon_min, ev_lon_max, ev_lat_min, ev_lat_max = ev_region

        try:
            ag = cached_regrid(sae_id, ev_t_idx, ev_feat, info["n_nodes"])
            grid_lat, grid_lon = load_grid()
            ev_fig = render_activation_map(
                activation_grid=ag,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                lon_bounds=(ev_lon_min, ev_lon_max),
                lat_bounds=(ev_lat_min, ev_lat_max),
                feature_id=ev_feat,
                timestamp=ev_ts,
                colormap="YlOrRd",
            )
            st.pyplot(ev_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render event map: {e}")

    # ── Other features active during this event ────────────────────────────
    if avail_ts and not catalog.empty:
        st.subheader("Top features active during this event")
        st.caption(
            "Features sorted by their max activation in the event timesteps above. "
            "Requires feature_catalog.parquet."
        )
        ev_t_indices = [timestamps.index(t) for t in avail_ts if t in timestamps]
        if ev_t_indices and not catalog.empty:
            # Use top_timestamps overlap as a proxy for event relevance
            def _event_relevance(row):
                return sum(
                    1 for ts_ex in (row.get("top_timestamps") or [])
                    if ts_ex in event["timestamps"]
                )
            ev_cat = catalog.copy()
            ev_cat["event_relevance"] = ev_cat.apply(_event_relevance, axis=1)
            ev_top = (
                ev_cat[ev_cat["event_relevance"] > 0]
                .sort_values("event_relevance", ascending=False)
                .head(15)
            )
            if not ev_top.empty:
                st.dataframe(
                    ev_top[["feature_id", "candidate_label", "event_relevance",
                             "max_activation", "artifact_score"]],
                    use_container_width=True,
                    height=300,
                )
            else:
                st.caption("No features with top-example timestamps overlapping this event.")
