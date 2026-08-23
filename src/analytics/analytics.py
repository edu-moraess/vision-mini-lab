"""
Analytics visualizations using Plotly.
Clean dark technical style, modebar transparent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.events.events import Event
from src.perception.tracker import TrackedObject


# Shared layout for dark technical look
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,18,22,0.6)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#c8c8d0"),
    margin=dict(l=48, r=24, t=40, b=40),
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
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _base_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**DARK_LAYOUT)
    return fig


def object_activity_chart(
    class_counts: Dict[str, int],
    title: str = "OBJECT ACTIVITY",
) -> go.Figure:
    """Bar chart of active object counts by class."""
    fig = _base_fig()
    if not class_counts:
        fig.add_annotation(
            text="No active objects",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(title=title, height=320)
        return fig

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    fig.add_trace(
        go.Bar(
            x=classes,
            y=counts,
            marker_color="#3d8bfd",
            marker_line_width=0,
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e8")),
        height=320,
        yaxis_title="Count",
        xaxis_title="Class",
    )
    return fig


def confidence_chart(
    objects: List[TrackedObject],
    title: str = "CONFIDENCE DISTRIBUTION",
) -> go.Figure:
    """Histogram / bar of confidence scores."""
    fig = _base_fig()
    if not objects:
        fig.add_annotation(
            text="No detections",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(title=title, height=320)
        return fig

    confs = [o.confidence for o in objects]
    labels = [f"{o.class_name} #{o.track_id}" for o in objects]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=confs,
            marker_color="#5eead4",
            marker_line_width=0,
            hovertemplate="%{x}<br>Conf: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e8")),
        height=320,
        yaxis_title="Confidence",
        yaxis_range=[0, 1.05],
        xaxis_tickangle=-30,
    )
    return fig


def motion_chart(
    objects: List[TrackedObject],
    title: str = "MOTION SPEED (px/frame)",
) -> go.Figure:
    """Bar chart of current speed per tracked object."""
    fig = _base_fig()
    if not objects:
        fig.add_annotation(
            text="No tracked objects",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(title=title, height=320)
        return fig

    labels = [f"{o.class_name} #{o.track_id}" for o in objects]
    speeds = [o.speed for o in objects]
    colors = ["#f59e0b" if o.state == "MOVING" else "#6b7280" for o in objects]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=speeds,
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="%{x}<br>Speed: %{y:.2f} px/frame<br>%{text}<extra></extra>",
            text=[o.direction for o in objects],
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e8")),
        height=320,
        yaxis_title="Speed (px/frame)",
        xaxis_tickangle=-30,
    )
    return fig


def trajectory_chart(
    trajectories: Dict[int, List[Tuple[float, float]]],
    current_positions: Dict[int, Tuple[float, float]],
    class_names: Dict[int, str],
    selected_id: Optional[int] = None,
    img_shape: Optional[Tuple[int, int]] = None,
    title: str = "TRAJECTORY",
) -> go.Figure:
    """Scatter plot of trajectories."""
    fig = _base_fig()

    if not trajectories:
        fig.add_annotation(
            text="No trajectories",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(title=title, height=400)
        return fig

    # Limit number of tracks shown for clarity
    ids = list(trajectories.keys())
    if selected_id is not None and selected_id in trajectories:
        ids = [selected_id]
    elif len(ids) > 12:
        ids = ids[-12:]

    palette = [
        "#3d8bfd", "#5eead4", "#f59e0b", "#a78bfa",
        "#f472b6", "#34d399", "#fb7185", "#38bdf8",
    ]

    for i, tid in enumerate(ids):
        pts = trajectories[tid]
        if len(pts) < 1:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = palette[i % len(palette)]
        name = f"{class_names.get(tid, 'obj')} #{tid}"

        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                hovertemplate=f"{name}<br>x=%{{x:.0f}} y=%{{y:.0f}}<extra></extra>",
            )
        )
        # start marker
        fig.add_trace(
            go.Scattergl(
                x=[xs[0]],
                y=[ys[0]],
                mode="markers",
                marker=dict(size=9, color=color, symbol="circle-open", line_width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # current position
        if tid in current_positions:
            cx, cy = current_positions[tid]
            fig.add_trace(
                go.Scattergl(
                    x=[cx],
                    y=[cy],
                    mode="markers",
                    marker=dict(size=11, color=color, symbol="diamond"),
                    showlegend=False,
                    hovertemplate=f"Current {name}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e8")),
        height=420,
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        yaxis=dict(autorange="reversed"),  # image coordinates
    )
    if img_shape is not None:
        h, w = img_shape[:2]
        fig.update_xaxes(range=[0, w])
        fig.update_yaxes(range=[h, 0])

    return fig


def event_timeline_chart(
    events: List[Event],
    title: str = "EVENT TIMELINE",
) -> go.Figure:
    """Horizontal timeline of recent events."""
    fig = _base_fig()
    if not events:
        fig.add_annotation(
            text="No events recorded",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(title=title, height=320)
        return fig

    # Use relative time from first event
    t0 = events[0].timestamp
    times = [e.timestamp - t0 for e in events]
    labels = [f"{e.event_type}" for e in events]
    texts = [
        f"{e.event_type}<br>ID {e.track_id} · {e.class_name}<br>{e.direction}"
        for e in events
    ]

    color_map = {
        "OBJECT_ENTERED": "#34d399",
        "OBJECT_EXITED": "#f87171",
        "LINE_CROSSED": "#3d8bfd",
        "STARTED_MOVING": "#f59e0b",
        "STOPPED": "#9ca3af",
    }
    colors = [color_map.get(e.event_type, "#a78bfa") for e in events]

    fig.add_trace(
        go.Scatter(
            x=times,
            y=labels,
            mode="markers+text",
            marker=dict(size=12, color=colors, symbol="diamond"),
            text=[f"#{e.track_id}" for e in events],
            textposition="top center",
            hovertext=texts,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e8")),
        height=340,
        xaxis_title="Time (s)",
        yaxis_title="",
        showlegend=False,
    )
    return fig


def thermal_analysis_chart(
    intensity: np.ndarray,
    title: str = "THERMAL INTENSITY DISTRIBUTION",
) -> go.Figure:
    """Histogram of relative thermal intensity."""
    fig = _base_fig()
    if intensity is None or intensity.size == 0:
        fig.add_annotation(
            text="No thermal data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(title=title, height=320)
        return fig

    flat = intensity.ravel()
    fig.add_trace(
        go.Histogram(
            x=flat,
            nbinsx=64,
            marker_color="#f97316",
            opacity=0.85,
            hovertemplate="Intensity %{x:.1f}<br>Count %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e8")),
        height=320,
        xaxis_title="Relative Intensity",
        yaxis_title="Pixel Count",
    )
    return fig


def movement_density_heatmap(
    all_centers: List[Tuple[float, float]],
    img_shape: Tuple[int, int],
    bins: int = 40,
    title: str = "MOVEMENT DENSITY",
) -> go.Figure:
    """
    2D histogram of trajectory centers → density heatmap.
    Conceptually separate from thermal intensity.
    """
    fig = _base_fig()
    h, w = img_shape[:2]

    if not all_centers:
        fig.add_annotation(
            text="No movement data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(title=title, height=400)
        return fig

    xs = [c[0] for c in all_centers]
    ys = [c[1] for c in all_centers]

    fig.add_trace(
        go.Histogram2d(
            x=xs,
            y=ys,
            nbinsx=bins,
            nbinsy=bins,
            colorscale="Hot",
            hovertemplate="x=%{x:.0f}<br>y=%{y:.0f}<br>density=%{z}<extra></extra>",
            colorbar=dict(
                title="Density",
                titleside="right",
                thickness=12,
                len=0.7,
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e0e8")),
        height=420,
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        yaxis=dict(autorange="reversed", range=[h, 0]),
        xaxis=dict(range=[0, w]),
    )
    return fig
