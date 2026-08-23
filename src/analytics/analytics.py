"""
VISION MINI LAB
Analytics visualizations using Plotly.

Technical, scientific, engineering, minimal, dark.
All charts use apply_chart_theme() for consistency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from src.events.events import Event
from src.perception.tracker import TrackedObject


# =============================================================================
# VISION COLOR PALETTE
# =============================================================================

VISION_COLORS = {
    "background": "#0c0c0f",
    "surface": "#121218",
    "surface_alt": "#1a1a22",
    "border": "#1e1e26",
    "text": "#d0d0d8",
    "text_muted": "#7a7a88",
    "primary": "#3d8bfd",
    "secondary": "#5eead4",
    "success": "#34d399",
    "warning": "#f59e0b",
    "danger": "#f87171",
    "thermal": "#f97316",
}

# Cyclic palette for charts (distinct colors)
CHART_PALETTE = [
    "#3d8bfd", "#5eead4", "#f59e0b", "#a78bfa", "#f472b6",
    "#34d399", "#fb7185", "#38bdf8", "#facc15", "#818cf8",
    "#fb923c", "#c084fc", "#2dd4bf", "#e879f9", "#84cc16",
]


# =============================================================================
# MODEBAR CONFIG
# =============================================================================

MODEBAR_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
    ],
}


# =============================================================================
# CHART THEME
# =============================================================================

def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply the global VISION MINI LAB dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,18,22,0.6)",
        font=dict(
            family="Inter, system-ui, sans-serif",
            size=12,
            color=VISION_COLORS["text"],
        ),
        margin=dict(l=48, r=24, t=48, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(size=11, color=VISION_COLORS["text_muted"]),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(size=11, color=VISION_COLORS["text_muted"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color=VISION_COLORS["text_muted"]),
        ),
        title=dict(
            font=dict(size=14, color=VISION_COLORS["text"]),
            x=0.0,
            xanchor="left",
        ),
        hoverlabel=dict(
            bgcolor=VISION_COLORS["surface_alt"],
            bordercolor=VISION_COLORS["border"],
            font=dict(size=12, color=VISION_COLORS["text"]),
        ),
    )
    return fig


def render_empty_state(fig: go.Figure, message: str) -> go.Figure:
    """Render a professional empty state inside a figure."""
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=VISION_COLORS["text_muted"]),
    )
    apply_chart_theme(fig)
    return fig


# =============================================================================
# OBJECT ACTIVITY
# =============================================================================

def object_activity_chart(
    class_counts: Dict[str, int],
    title: str = "Object Activity",
) -> go.Figure:
    """Objects detected in the current frame, grouped by class."""
    fig = go.Figure()

    if not class_counts:
        render_empty_state(fig, "No objects detected")
        fig.update_layout(title=title, height=300)
        return fig

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(len(classes))]

    fig.add_trace(go.Bar(
        x=classes,
        y=counts,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Class: %{x}<br>Count: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        height=300,
        yaxis_title="Count",
        xaxis_title="",
        showlegend=False,
    )
    apply_chart_theme(fig)
    return fig


# =============================================================================
# CONFIDENCE DISTRIBUTION
# =============================================================================

def confidence_chart(
    objects: List[TrackedObject],
    title: str = "Confidence Distribution",
) -> go.Figure:
    """Confidence distribution of the current detections."""
    fig = go.Figure()

    if not objects:
        render_empty_state(fig, "No detections")
        fig.update_layout(title=title, height=300)
        return fig

    confs = [float(o.confidence) for o in objects]
    labels = [
        f"{o.class_name} {o.display_id}"
        for o in objects
    ]
    colors = [
        CHART_PALETTE[o.class_id % len(CHART_PALETTE)]
        for o in objects
    ]

    fig.add_trace(go.Bar(
        x=labels,
        y=confs,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="%{x}<br>Confidence: %{y:.1%}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        height=300,
        yaxis_title="Confidence",
        yaxis_range=[0, 1.05],
        xaxis_tickangle=-35,
        showlegend=False,
    )
    apply_chart_theme(fig)
    return fig


# =============================================================================
# MOTION SPEED
# =============================================================================

def motion_chart(
    objects: List[TrackedObject],
    title: str = "Motion Speed (px/frame)",
) -> go.Figure:
    """Estimated object displacement in pixels per frame."""
    fig = go.Figure()

    if not objects:
        render_empty_state(fig, "No movement data")
        fig.update_layout(title=title, height=300)
        return fig

    labels = [
        f"{o.class_name} {o.display_id}"
        for o in objects
    ]
    speeds = [max(0.0, float(o.speed)) for o in objects]
    colors = [
        VISION_COLORS["warning"] if o.state == "MOVING" else VISION_COLORS["text_muted"]
        for o in objects
    ]

    fig.add_trace(go.Bar(
        x=labels,
        y=speeds,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="%{x}<br>Speed: %{y:.2f} px/frame<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        height=300,
        yaxis_title="Speed (px/frame)",
        xaxis_tickangle=-35,
        showlegend=False,
    )
    apply_chart_theme(fig)
    return fig


# =============================================================================
# TRAJECTORY
# =============================================================================

def trajectory_chart(
    trajectories: Dict[int, List[Tuple[float, float]]],
    current_positions: Dict[int, Tuple[float, float]],
    class_names: Dict[int, str],
    selected_id: Optional[int] = None,
    img_shape: Optional[Tuple[int, int]] = None,
    title: str = "Trajectory",
) -> go.Figure:
    """Spatial paths of tracked objects with start / current markers."""
    fig = go.Figure()

    if not trajectories:
        render_empty_state(fig, "No trajectory data")
        fig.update_layout(title=title, height=400)
        return fig

    ids = list(trajectories.keys())

    if selected_id is not None and selected_id in trajectories:
        ids = [selected_id]
    elif len(ids) > 12:
        ids = ids[-12:]

    for i, track_id in enumerate(ids):
        points = trajectories.get(track_id, [])
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        name = f"{class_names.get(track_id, 'obj')} #{track_id}"

        # Main trajectory line
        fig.add_trace(go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            line=dict(color=color, width=1.5),
            hovertemplate=f"{name}<br>x=%{{x:.0f}} y=%{{y:.0f}}<extra></extra>",
            showlegend=True,
        ))

        # Start marker (open circle)
        fig.add_trace(go.Scattergl(
            x=[xs[0]],
            y=[ys[0]],
            mode="markers",
            marker=dict(size=8, color=color, symbol="circle-open", line=dict(width=2, color=color)),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Current position (diamond)
        if track_id in current_positions:
            cx, cy = current_positions[track_id]
            fig.add_trace(go.Scattergl(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond"),
                showlegend=False,
                hovertemplate=f"Current {name}<extra></extra>",
            ))

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        yaxis=dict(autorange="reversed"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )

    if img_shape is not None:
        h, w = img_shape[:2]
        fig.update_xaxes(range=[0, w])
        fig.update_yaxes(range=[h, 0], autorange=False)

    apply_chart_theme(fig)
    return fig


# =============================================================================
# EVENT TIMELINE
# =============================================================================

def event_timeline_chart(
    events: List[Event],
    title: str = "Event Timeline",
) -> go.Figure:
    """Temporal event markers."""
    fig = go.Figure()

    if not events:
        render_empty_state(fig, "No events recorded")
        fig.update_layout(title=title, height=300)
        return fig

    t0 = events[0].timestamp
    times = [e.timestamp - t0 for e in events]
    labels = [e.event_type for e in events]

    color_map = {
        "OBJECT_ENTERED": VISION_COLORS["success"],
        "OBJECT_EXITED": VISION_COLORS["danger"],
        "LINE_CROSSED": VISION_COLORS["primary"],
        "STARTED_MOVING": VISION_COLORS["warning"],
        "STOPPED": VISION_COLORS["text_muted"],
    }
    colors = [color_map.get(e.event_type, "#a78bfa") for e in events]

    fig.add_trace(go.Scatter(
        x=times,
        y=labels,
        mode="markers+text",
        marker=dict(size=11, color=colors, symbol="diamond", line=dict(width=1, color="rgba(0,0,0,0.3)")),
        text=[f"#{e.track_id}" for e in events],
        textposition="top center",
        textfont=dict(size=10, color=VISION_COLORS["text_muted"]),
        hovertemplate=(
            "%{text}<br>"
            "Time: %{x:.1f}s<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.update_layout(
        title=title,
        height=320,
        xaxis_title="Time (s)",
        yaxis_title="",
    )
    apply_chart_theme(fig)
    return fig


# =============================================================================
# THERMAL INTENSITY DISTRIBUTION
# =============================================================================

def thermal_analysis_chart(
    intensity: np.ndarray,
    title: str = "Thermal Intensity Distribution",
) -> go.Figure:
    """Relative image intensity distribution. Not calibrated temperature."""
    fig = go.Figure()

    if intensity is None or intensity.size == 0:
        render_empty_state(fig, "No thermal data")
        fig.update_layout(title=title, height=300)
        return fig

    flat = intensity.astype(float).ravel()
    flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        render_empty_state(fig, "No valid thermal data")
        fig.update_layout(title=title, height=300)
        return fig

    fig.add_trace(go.Histogram(
        x=flat,
        nbinsx=64,
        marker_color=VISION_COLORS["thermal"],
        opacity=0.85,
        hovertemplate="Intensity: %{x:.1f}<br>Count: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        height=300,
        xaxis_title="Relative Intensity",
        yaxis_title="Pixel Count",
        showlegend=False,
    )
    apply_chart_theme(fig)
    return fig


# =============================================================================
# MOVEMENT DENSITY
# =============================================================================

def movement_density_heatmap(
    all_centers: List[Tuple[float, float]],
    img_shape: Tuple[int, int],
    bins: int = 40,
    title: str = "Movement Density",
) -> go.Figure:
    """Spatial concentration of tracked object positions."""
    fig = go.Figure()

    if img_shape is None or len(img_shape) < 2:
        render_empty_state(fig, "No image dimensions")
        fig.update_layout(title=title, height=400)
        return fig

    h, w = img_shape[:2]

    # Validate and filter points
    valid_points = []
    for center in all_centers:
        try:
            if center is None:
                continue
            x = float(center[0])
            y = float(center[1])
        except (TypeError, ValueError, IndexError):
            continue

        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < 0 or x > w or y < 0 or y > h:
            continue
        valid_points.append((x, y))

    if not valid_points:
        render_empty_state(fig, "No movement data")
        fig.update_layout(title=title, height=400)
        return fig

    xs = np.array([p[0] for p in valid_points], dtype=float)
    ys = np.array([p[1] for p in valid_points], dtype=float)

    try:
        bins = int(np.clip(bins, 8, 100))
    except Exception:
        bins = 40

    fig.add_trace(go.Histogram2d(
        x=xs,
        y=ys,
        nbinsx=bins,
        nbinsy=bins,
        colorscale="Hot",
        hovertemplate="x=%{x:.0f}<br>y=%{y:.0f}<br>density=%{z}<extra></extra>",
        colorbar=dict(
            title=dict(text="Density", font=dict(size=11, color=VISION_COLORS["text_muted"])),
            thickness=12,
            len=0.7,
            tickfont=dict(size=10, color=VISION_COLORS["text_muted"]),
        ),
    ))

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        xaxis=dict(range=[0, w]),
        yaxis=dict(range=[h, 0], autorange=False),
        showlegend=False,
    )
    apply_chart_theme(fig)
    return fig
