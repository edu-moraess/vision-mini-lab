"""
VISION MINI LAB
Analytics visualizations using Plotly.

Clean dark technical style.
Safe Plotly configuration.
Movement density is spatial density and is NOT thermal data.
"""

from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import plotly.graph_objects as go

from src.events.events import Event
from src.perception.tracker import TrackedObject


# =============================================================================
# SHARED STYLE
# =============================================================================

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,18,22,0.6)",
    font=dict(
        family="Inter, system-ui, sans-serif",
        size=12,
        color="#c8c8d0",
    ),
    margin=dict(
        l=48,
        r=24,
        t=40,
        b=40,
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.15)",
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.15)",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=11),
    ),
)


MODEBAR_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
    ],
}


# =============================================================================
# BASE FIGURE
# =============================================================================

def _base_fig() -> go.Figure:

    fig = go.Figure()

    fig.update_layout(
        **DARK_LAYOUT
    )

    return fig


# =============================================================================
# OBJECT ACTIVITY
# =============================================================================

def object_activity_chart(
    class_counts: Dict[str, int],
    title: str = "OBJECT ACTIVITY",
) -> go.Figure:
    """Active object counts by class."""

    fig = _base_fig()

    if not class_counts:

        fig.add_annotation(
            text="No active objects",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=320,
        )

        return fig

    classes = list(
        class_counts.keys()
    )

    counts = list(
        class_counts.values()
    )

    fig.add_trace(
        go.Bar(
            x=classes,
            y=counts,
            marker_color="#3d8bfd",
            marker_line_width=0,
            hovertemplate=(
                "%{x}: %{y}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=14,
                color="#e0e0e8",
            ),
        ),
        height=320,
        yaxis_title="Count",
        xaxis_title="Class",
    )

    return fig


# =============================================================================
# CONFIDENCE
# =============================================================================

def confidence_chart(
    objects: List[TrackedObject],
    title: str = "CONFIDENCE DISTRIBUTION",
) -> go.Figure:
    """Confidence score per detection."""

    fig = _base_fig()

    if not objects:

        fig.add_annotation(
            text="No detections",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=320,
        )

        return fig

    confs = [
        float(o.confidence)
        for o in objects
    ]

    labels = [
        f"{o.class_name} #{o.track_id}"
        if o.track_id > 0
        else f"{o.class_name} DET"
        for o in objects
    ]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=confs,
            marker_color="#5eead4",
            marker_line_width=0,
            hovertemplate=(
                "%{x}"
                "<br>Conf: %{y:.1%}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=14,
                color="#e0e0e8",
            ),
        ),
        height=320,
        yaxis_title="Confidence",
        yaxis_range=[
            0,
            1.05,
        ],
        xaxis_tickangle=-30,
    )

    return fig


# =============================================================================
# MOTION
# =============================================================================

def motion_chart(
    objects: List[TrackedObject],
    title: str = "MOTION SPEED (px/frame)",
) -> go.Figure:
    """Current speed for tracked objects."""

    fig = _base_fig()

    if not objects:

        fig.add_annotation(
            text="No tracked objects",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=320,
        )

        return fig

    labels = [
        (
            f"{o.class_name} #{o.track_id}"
            if o.track_id > 0
            else f"{o.class_name} DET"
        )
        for o in objects
    ]

    speeds = [
        max(
            0.0,
            float(o.speed),
        )
        for o in objects
    ]

    colors = [
        "#f59e0b"
        if o.state == "MOVING"
        else "#6b7280"
        for o in objects
    ]

    directions = [
        o.direction
        for o in objects
    ]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=speeds,
            marker_color=colors,
            marker_line_width=0,
            hovertemplate=(
                "%{x}"
                "<br>Speed: %{y:.2f} px/frame"
                "<br>%{text}"
                "<extra></extra>"
            ),
            text=directions,
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=14,
                color="#e0e0e8",
            ),
        ),
        height=320,
        yaxis_title="Speed (px/frame)",
        xaxis_tickangle=-30,
    )

    return fig


# =============================================================================
# TRAJECTORY
# =============================================================================

def trajectory_chart(
    trajectories: Dict[
        int,
        List[
            Tuple[float, float]
        ],
    ],
    current_positions: Dict[
        int,
        Tuple[float, float],
    ],
    class_names: Dict[
        int,
        str,
    ],
    selected_id: Optional[int] = None,
    img_shape: Optional[
        Tuple[int, int]
    ] = None,
    title: str = "TRAJECTORY",
) -> go.Figure:
    """Scatter visualization of object trajectories."""

    fig = _base_fig()

    if not trajectories:

        fig.add_annotation(
            text="No trajectories",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=400,
        )

        return fig

    ids = list(
        trajectories.keys()
    )

    if (
        selected_id is not None
        and selected_id in trajectories
    ):

        ids = [
            selected_id
        ]

    elif len(ids) > 12:

        ids = ids[-12:]

    palette = [
        "#3d8bfd",
        "#5eead4",
        "#f59e0b",
        "#a78bfa",
        "#f472b6",
        "#34d399",
        "#fb7185",
        "#38bdf8",
    ]

    for i, track_id in enumerate(ids):

        points = trajectories.get(
            track_id,
            [],
        )

        if not points:

            continue

        xs = [
            point[0]
            for point in points
        ]

        ys = [
            point[1]
            for point in points
        ]

        color = (
            palette[
                i % len(palette)
            ]
        )

        name = (
            f"{class_names.get(track_id, 'obj')}"
            f" #{track_id}"
        )

        # Main trajectory
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                line=dict(
                    color=color,
                    width=2,
                ),
                marker=dict(
                    size=4,
                    color=color,
                ),
                hovertemplate=(
                    f"{name}"
                    "<br>x=%{x:.0f}"
                    " y=%{y:.0f}"
                    "<extra></extra>"
                ),
            )
        )

        # Start position
        fig.add_trace(
            go.Scattergl(
                x=[xs[0]],
                y=[ys[0]],
                mode="markers",
                marker=dict(
                    size=9,
                    color=color,
                    symbol="circle-open",
                    line_width=2,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Current position
        if track_id in current_positions:

            cx, cy = (
                current_positions[
                    track_id
                ]
            )

            fig.add_trace(
                go.Scattergl(
                    x=[cx],
                    y=[cy],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color=color,
                        symbol="diamond",
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"Current {name}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=14,
                color="#e0e0e8",
            ),
        ),
        height=420,
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        yaxis=dict(
            autorange="reversed"
        ),
    )

    if img_shape is not None:

        h, w = img_shape[:2]

        fig.update_xaxes(
            range=[
                0,
                w,
            ]
        )

        fig.update_yaxes(
            range=[
                h,
                0,
            ]
        )

    return fig


# =============================================================================
# EVENT TIMELINE
# =============================================================================

def event_timeline_chart(
    events: List[Event],
    title: str = "EVENT TIMELINE",
) -> go.Figure:
    """Horizontal timeline of recent events."""

    fig = _base_fig()

    if not events:

        fig.add_annotation(
            text="No events recorded",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=320,
        )

        return fig

    t0 = events[0].timestamp

    times = [
        event.timestamp - t0
        for event in events
    ]

    labels = [
        event.event_type
        for event in events
    ]

    texts = [
        (
            f"{event.event_type}"
            f"<br>ID {event.track_id}"
            f" · {event.class_name}"
            f"<br>{event.direction}"
        )
        for event in events
    ]

    color_map = {
        "OBJECT_ENTERED": "#34d399",
        "OBJECT_EXITED": "#f87171",
        "LINE_CROSSED": "#3d8bfd",
        "STARTED_MOVING": "#f59e0b",
        "STOPPED": "#9ca3af",
    }

    colors = [
        color_map.get(
            event.event_type,
            "#a78bfa",
        )
        for event in events
    ]

    fig.add_trace(
        go.Scatter(
            x=times,
            y=labels,
            mode="markers+text",
            marker=dict(
                size=12,
                color=colors,
                symbol="diamond",
            ),
            text=[
                f"#{event.track_id}"
                for event in events
            ],
            textposition="top center",
            hovertext=texts,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=14,
                color="#e0e0e8",
            ),
        ),
        height=340,
        xaxis_title="Time (s)",
        yaxis_title="",
        showlegend=False,
    )

    return fig


# =============================================================================
# THERMAL
# =============================================================================

def thermal_analysis_chart(
    intensity: np.ndarray,
    title: str = "THERMAL INTENSITY DISTRIBUTION",
) -> go.Figure:
    """
    Histogram of relative thermal intensity.

    IMPORTANT:
    These values are relative intensity.
    They are NOT calibrated Celsius temperatures.
    """

    fig = _base_fig()

    if (
        intensity is None
        or intensity.size == 0
    ):

        fig.add_annotation(
            text="No thermal data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=320,
        )

        return fig

    flat = (
        intensity
        .astype(float)
        .ravel()
    )

    flat = flat[
        np.isfinite(flat)
    ]

    if flat.size == 0:

        fig.add_annotation(
            text="No valid thermal data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=320,
        )

        return fig

    fig.add_trace(
        go.Histogram(
            x=flat,
            nbinsx=64,
            marker_color="#f97316",
            opacity=0.85,
            hovertemplate=(
                "Intensity %{x:.1f}"
                "<br>Count %{y}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=14,
                color="#e0e0e8",
            ),
        ),
        height=320,
        xaxis_title="Relative Intensity",
        yaxis_title="Pixel Count",
    )

    return fig


# =============================================================================
# MOVEMENT DENSITY
# =============================================================================

def movement_density_heatmap(
    all_centers: List[
        Tuple[float, float]
    ],
    img_shape: Tuple[int, int],
    bins: int = 40,
    title: str = "MOVEMENT DENSITY",
) -> go.Figure:
    """
    Spatial density of tracked object centers.

    This is NOT a thermal heatmap.

    It represents where object centers have travelled
    through the image.

    The implementation is deliberately defensive because
    Streamlit Cloud may use different Plotly versions.
    """

    fig = _base_fig()

    if img_shape is None:

        fig.add_annotation(
            text="No image dimensions",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=400,
        )

        return fig

    h, w = img_shape[:2]

    # -------------------------------------------------------------------------
    # Validate points
    # -------------------------------------------------------------------------

    valid_points = []

    for center in all_centers:

        try:

            if center is None:
                continue

            x = float(
                center[0]
            )

            y = float(
                center[1]
            )

        except (
            TypeError,
            ValueError,
            IndexError,
        ):

            continue

        if not np.isfinite(x):
            continue

        if not np.isfinite(y):
            continue

        if x < 0 or x > w:
            continue

        if y < 0 or y > h:
            continue

        valid_points.append(
            (
                x,
                y,
            )
        )

    if not valid_points:

        fig.add_annotation(
            text="No movement data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="#666",
            ),
        )

        fig.update_layout(
            title=title,
            height=400,
        )

        return fig

    xs = np.array(
        [
            point[0]
            for point in valid_points
        ],
        dtype=float,
    )

    ys = np.array(
        [
            point[1]
            for point in valid_points
        ],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Safe bin count
    # -------------------------------------------------------------------------

    try:

        bins = int(
            np.clip(
                bins,
                8,
                100,
            )
        )

    except Exception:

        bins = 40

    # -------------------------------------------------------------------------
    # Histogram 2D
    # -------------------------------------------------------------------------

    fig.add_trace(
        go.Histogram2d(
            x=xs,
            y=ys,
            nbinsx=bins,
            nbinsy=bins,
            colorscale="Hot",
            hovertemplate=(
                "x=%{x:.0f}"
                "<br>y=%{y:.0f}"
                "<br>density=%{z}"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="Density",
                thickness=12,
                len=0.7,
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=14,
                color="#e0e0e8",
            ),
        ),
        height=420,
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        xaxis=dict(
            range=[
                0,
                w,
            ],
        ),
        yaxis=dict(
            range=[
                h,
                0,
            ],
            autorange=False,
        ),
    )

    return fig