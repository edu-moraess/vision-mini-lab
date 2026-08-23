"""
VISION MINI LAB
Real-Time Visual Intelligence

Compact experimental station for computer vision perception.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from src.analytics.analytics import (
    MODEBAR_CONFIG,
    confidence_chart,
    event_timeline_chart,
    motion_chart,
    movement_density_heatmap,
    object_activity_chart,
    thermal_analysis_chart,
    trajectory_chart,
)
from src.events.events import EventEngine
from src.motion.motion import MotionAnalyzer
from src.perception.detector import Detector
from src.perception.tracker import Tracker, TrackedObject
from src.thermal.thermal import process_thermal, compute_roi_stats
from src.utils.image import load_image_from_bytes, resize_keep_aspect, to_bgr

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision-mini-lab")

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VISION MINI LAB",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Dark technical CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
    }

    .stApp {
        background-color: #0c0c0f;
        color: #d0d0d8;
    }

    h1, h2, h3, h4 {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em;
        color: #e8e8f0 !important;
    }

    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        color: #f0f0f8;
        margin-bottom: 0.15rem;
    }

    .sub-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.18em;
        color: #6b6b78;
        margin-bottom: 1.5rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #101014;
        border-right: 1px solid #1e1e26;
    }

    .stButton > button {
        background-color: #1a1a22;
        color: #c8c8d4;
        border: 1px solid #2a2a34;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }

    .stButton > button:hover {
        border-color: #3d8bfd;
        color: #e0e0ec;
    }

    div[data-testid="stMetric"] {
        background-color: #121218;
        border: 1px solid #1e1e26;
        border-radius: 4px;
        padding: 0.6rem 0.8rem;
    }

    div[data-testid="stMetric"] label {
        color: #7a7a88 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem !important;
        letter-spacing: 0.08em;
    }

    .block-divider {
        border: none;
        border-top: 1px solid #1e1e26;
        margin: 1.2rem 0;
    }

    .info-box {
        background-color: #121218;
        border: 1px solid #1e1e26;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #a0a0b0;
        line-height: 1.55;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached model loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_name: str = "yolov8s.pt") -> Optional[YOLO]:
    try:
        model = YOLO(model_name)
        return model
    except Exception as exc:
        logger.exception("YOLO load failed: %s", exc)
        return None


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def draw_overlays(
    image_rgb: np.ndarray,
    objects: List[TrackedObject],
    roi: Optional[Tuple[int, int, int, int]] = None,
    line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    show_motion: bool = False,
) -> np.ndarray:
    """Minimal discrete overlays."""
    out = image_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ROI
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 140, 220), 1, cv2.LINE_AA)

    # Virtual line
    if line is not None:
        (lx1, ly1), (lx2, ly2) = line
        cv2.line(out, (lx1, ly1), (lx2, ly2), (220, 160, 60), 1, cv2.LINE_AA)

    for obj in objects:
        x1, y1, x2, y2 = map(int, obj.bbox)
        # discrete box
        color = (90, 180, 255) if obj.state == "MOVING" else (140, 140, 150)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        label = f"{obj.class_name.upper()} #{obj.track_id}"
        conf_str = f"{obj.confidence:.0%}"
        text = f"{label}  {conf_str}"
        if show_motion and obj.state == "MOVING":
            text += f"  {obj.speed:.1f}px"

        (tw, th), _ = cv2.getTextSize(text, font, 0.42, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (20, 20, 28), -1)
        cv2.putText(
            out, text, (x1 + 2, y1 - 4),
            font, 0.42, (210, 210, 220), 1, cv2.LINE_AA,
        )

    return out


def draw_trajectories(
    image_rgb: np.ndarray,
    tracker: Tracker,
    objects: List[TrackedObject],
) -> np.ndarray:
    out = image_rgb.copy()
    for obj in objects:
        pts = tracker.get_trajectory(obj.track_id)
        if len(pts) < 2:
            continue
        pts_i = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [pts_i], False, (60, 120, 200), 1, cv2.LINE_AA)
    return out


# -----------------------------------------------------------------------------
# Session state init
# -----------------------------------------------------------------------------
def init_state():
    defaults = {
        "model_loaded": False,
        "tracker": None,
        "motion": MotionAnalyzer(),
        "events": EventEngine(),
        "all_centers": [],  # for movement density
        "selected_track": None,
        "frame_count": 0,
        "last_fps": 0.0,
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### CONTROLS")
    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)

    conf = st.slider("Confidence", 0.10, 0.90, 0.35, 0.05)
    iou = st.slider("IoU", 0.20, 0.80, 0.45, 0.05)

    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### VIEW")
    view_mode = st.radio(
        "Display Mode",
        ["RGB", "THERMAL", "OVERLAY"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    colormap = st.selectbox("Colormap", ["inferno", "magma", "turbo"], index=0)
    thermal_opacity = st.slider("Thermal Opacity", 0.1, 0.9, 0.5, 0.1)

    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### OVERLAYS")
    show_tracks = st.checkbox("Trajectories", value=True)
    show_motion_label = st.checkbox("Motion Labels", value=False)

    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### ROI")
    enable_roi = st.checkbox("Enable ROI", value=False)
    if enable_roi:
        roi_x1 = st.number_input("ROI X1", 0, 4000, 100, 10)
        roi_y1 = st.number_input("ROI Y1", 0, 4000, 100, 10)
        roi_x2 = st.number_input("ROI X2", 0, 4000, 500, 10)
        roi_y2 = st.number_input("ROI Y2", 0, 4000, 400, 10)
        current_roi = (int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2))
    else:
        current_roi = None

    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### LINE CROSSING")
    enable_line = st.checkbox("Enable Line", value=False)
    if enable_line:
        lx1 = st.number_input("Line X1", 0, 4000, 200, 10)
        ly1 = st.number_input("Line Y1", 0, 4000, 300, 10)
        lx2 = st.number_input("Line X2", 0, 4000, 600, 10)
        ly2 = st.number_input("Line Y2", 0, 4000, 300, 10)
        current_line = ((int(lx1), int(ly1)), (int(lx2), int(ly2)))
    else:
        current_line = None

    st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
    if st.button("RESET STATE"):
        st.session_state.motion.reset()
        st.session_state.events.clear()
        st.session_state.all_centers = []
        st.session_state.frame_count = 0
        if st.session_state.tracker is not None:
            st.session_state.tracker.reset()
        st.rerun()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">VISION MINI LAB</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">REAL-TIME VISUAL INTELLIGENCE</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
model = load_yolo_model("yolov8s.pt")
if model is None:
    st.error("YOLO não pôde ser carregado.")
    st.stop()

if st.session_state.tracker is None:
    st.session_state.tracker = Tracker(model, conf=conf, iou=iou)
else:
    st.session_state.tracker.set_thresholds(conf, iou)

tracker: Tracker = st.session_state.tracker
motion: MotionAnalyzer = st.session_state.motion
events: EventEngine = st.session_state.events

events.set_roi(current_roi)
if current_line is not None:
    events.set_line(current_line[0], current_line[1])
else:
    events.set_line(None, None)


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
st.markdown("### INPUT")
input_type = st.radio(
    "Source",
    ["Image", "Video"],
    horizontal=True,
    label_visibility="collapsed",
)

uploaded = None
if input_type == "Image":
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"],
        label_visibility="collapsed",
    )
else:
    uploaded = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )

st.markdown('<hr class="block-divider">', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
display_image = None
objects: List[TrackedObject] = []
intensity_map = None
thermal_stats = None
img_shape = None

if uploaded is not None:
    if input_type == "Image":
        data = uploaded.getvalue()
        image = load_image_from_bytes(data)
        if image is None:
            st.error("Não foi possível carregar esta imagem.")
        else:
            image, _ = resize_keep_aspect(image, max_side=1280)
            img_shape = image.shape

            t0 = time.perf_counter()
            objects = tracker.update(image)
            objects = motion.update(objects)
            events.update(objects)
            elapsed = time.perf_counter() - t0
            st.session_state.last_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            st.session_state.frame_count += 1

            # collect centers for density
            for o in objects:
                st.session_state.all_centers.append(o.center)
            # limit memory
            if len(st.session_state.all_centers) > 5000:
                st.session_state.all_centers = st.session_state.all_centers[-5000:]

            # thermal
            mode_map = {"RGB": "rgb", "THERMAL": "thermal", "OVERLAY": "overlay"}
            display_image, intensity_map, thermal_stats = process_thermal(
                image,
                mode=mode_map[view_mode],
                colormap=colormap,
                opacity=thermal_opacity,
            )

            if show_tracks:
                display_image = draw_trajectories(display_image, tracker, objects)
            display_image = draw_overlays(
                display_image, objects, current_roi, current_line, show_motion_label
            )

    else:  # Video
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                st.error("Não foi possível processar este vídeo.")
            else:
                fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps_v if fps_v > 0 else 0

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("FPS", f"{fps_v:.1f}")
                col_m2.metric("Resolution", f"{width}×{height}")
                col_m3.metric("Frames", f"{total_frames}")
                col_m4.metric("Duration", f"{duration:.1f}s")

                # Process limited frames for stability
                max_process = min(total_frames, 120)
                frame_idx = st.slider("Frame", 0, max(0, max_process - 1), 0)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_bgr = cap.read()
                if not ret:
                    st.error("Não foi possível ler o frame.")
                else:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb, _ = resize_keep_aspect(frame_rgb, max_side=1280)
                    img_shape = frame_rgb.shape

                    t0 = time.perf_counter()
                    objects = tracker.update(frame_rgb)
                    objects = motion.update(objects)
                    events.update(objects, frame_time=frame_idx / fps_v)
                    elapsed = time.perf_counter() - t0
                    st.session_state.last_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    st.session_state.frame_count = frame_idx

                    for o in objects:
                        st.session_state.all_centers.append(o.center)
                    if len(st.session_state.all_centers) > 5000:
                        st.session_state.all_centers = st.session_state.all_centers[-5000:]

                    mode_map = {"RGB": "rgb", "THERMAL": "thermal", "OVERLAY": "overlay"}
                    display_image, intensity_map, thermal_stats = process_thermal(
                        frame_rgb,
                        mode=mode_map[view_mode],
                        colormap=colormap,
                        opacity=thermal_opacity,
                    )

                    if show_tracks:
                        display_image = draw_trajectories(display_image, tracker, objects)
                    display_image = draw_overlays(
                        display_image, objects, current_roi, current_line, show_motion_label
                    )

            cap.release()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# LIVE VIEW + OBJECT INTELLIGENCE
# -----------------------------------------------------------------------------
col_view, col_info = st.columns([1.6, 1.0])

with col_view:
    st.markdown("### LIVE VIEW")
    if display_image is not None:
        st.image(display_image, use_container_width=True, channels="RGB")
    else:
        st.markdown(
            '<div class="info-box">Aguardando entrada de imagem ou vídeo.</div>',
            unsafe_allow_html=True,
        )

with col_info:
    st.markdown("### OBJECT INTELLIGENCE")
    n_active = len(objects)
    n_tracked = len(tracker.histories)
    inside = events.count_inside_roi(objects)

    m1, m2 = st.columns(2)
    m1.metric("Active Objects", n_active)
    m2.metric("Tracked", n_tracked)
    m3, m4 = st.columns(2)
    m3.metric("FPS", f"{st.session_state.last_fps:.1f}")
    m4.metric("Inside ROI", inside)

    st.markdown("")
    if objects:
        # object selector
        options = {f"{o.class_name} #{o.track_id}": o.track_id for o in objects}
        sel_label = st.selectbox(
            "Selected Object",
            list(options.keys()),
            label_visibility="collapsed",
        )
        st.session_state.selected_track = options.get(sel_label)

        sel = next(
            (o for o in objects if o.track_id == st.session_state.selected_track),
            None,
        )
        if sel:
            st.markdown(
                f"""
                <div class="info-box">
                ID {sel.track_id}<br>
                Class · {sel.class_name}<br>
                Confidence · {sel.confidence:.1%}<br>
                State · {sel.state}<br>
                Direction · {sel.direction}<br>
                Speed · {sel.speed:.2f} px/frame<br>
                Center · ({sel.center[0]:.0f}, {sel.center[1]:.0f})
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="info-box">Nenhum objeto ativo.</div>',
            unsafe_allow_html=True,
        )

    # Thermal stats
    if thermal_stats is not None:
        st.markdown("#### THERMAL INTENSITY")
        st.markdown(
            f"""
            <div class="info-box">
            Mean · {thermal_stats.mean:.1f}<br>
            Max · {thermal_stats.maximum:.1f}<br>
            Min · {thermal_stats.minimum:.1f}<br>
            Std · {thermal_stats.std:.1f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if current_roi is not None and intensity_map is not None:
            roi_stats = compute_roi_stats(intensity_map, current_roi)
            if roi_stats:
                st.markdown("#### THERMAL ROI")
                st.markdown(
                    f"""
                    <div class="info-box">
                    Mean · {roi_stats.mean:.1f}<br>
                    Max · {roi_stats.maximum:.1f}<br>
                    Min · {roi_stats.minimum:.1f}<br>
                    Variance · {roi_stats.variance:.1f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# ANALYTICS
# -----------------------------------------------------------------------------
st.markdown('<hr class="block-divider">', unsafe_allow_html=True)
st.markdown("### ANALYTICS")

if objects or st.session_state.all_centers:
    # 1. Object Activity
    class_counts: Dict[str, int] = defaultdict(int)
    for o in objects:
        class_counts[o.class_name] += 1
    fig1 = object_activity_chart(dict(class_counts))
    st.plotly_chart(fig1, use_container_width=True, config=MODEBAR_CONFIG)
    st.caption("Contagem de objetos ativos por classe no frame atual.")

    # 2. Confidence
    fig2 = confidence_chart(objects)
    st.plotly_chart(fig2, use_container_width=True, config=MODEBAR_CONFIG)
    st.caption("Distribuição de confiança das detecções atuais.")

    # 3. Motion
    fig3 = motion_chart(objects)
    st.plotly_chart(fig3, use_container_width=True, config=MODEBAR_CONFIG)
    st.caption("Velocidade instantânea em pixels por frame. Unidade: px/frame.")

    # 4. Trajectory
    trajectories = {
        tid: tracker.get_trajectory(tid) for tid in tracker.histories
    }
    current_pos = {o.track_id: o.center for o in objects}
    class_names = {o.track_id: o.class_name for o in objects}
    # also from histories for lost ones
    for tid in trajectories:
        if tid not in class_names:
            class_names[tid] = "obj"

    fig4 = trajectory_chart(
        trajectories,
        current_pos,
        class_names,
        selected_id=st.session_state.selected_track,
        img_shape=img_shape,
    )
    st.plotly_chart(fig4, use_container_width=True, config=MODEBAR_CONFIG)
    st.caption("Trajetórias recentes (últimos ~60 pontos). Eixo Y invertido (coordenadas de imagem).")

    # 5. Event Timeline
    recent_events = events.get_recent(80)
    fig5 = event_timeline_chart(recent_events)
    st.plotly_chart(fig5, use_container_width=True, config=MODEBAR_CONFIG)
    st.caption("Linha do tempo de eventos detectados (enter/exit, line cross, motion).")

    # 6. Thermal Analysis
    if intensity_map is not None:
        fig6 = thermal_analysis_chart(intensity_map)
        st.plotly_chart(fig6, use_container_width=True, config=MODEBAR_CONFIG)
        st.caption(
            "Distribuição de intensidade térmica relativa. "
            "Não representa temperatura calibrada (°C / K)."
        )

    # 7. Movement Density Heatmap
    if st.session_state.all_centers and img_shape is not None:
        fig7 = movement_density_heatmap(
            st.session_state.all_centers,
            img_shape,
        )
        st.plotly_chart(fig7, use_container_width=True, config=MODEBAR_CONFIG)
        st.caption(
            "Densidade espacial de trajetórias (centros dos objetos). "
            "Conceitualmente distinto do mapa térmico."
        )
else:
    st.markdown(
        '<div class="info-box">Carregue uma imagem ou vídeo para gerar analytics.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# EVENT TIMELINE (text)
# -----------------------------------------------------------------------------
st.markdown('<hr class="block-divider">', unsafe_allow_html=True)
st.markdown("### EVENT TIMELINE")

recent = events.get_recent(30)
if recent:
    lines = []
    t0 = recent[0].timestamp
    for e in reversed(recent):
        t_rel = e.timestamp - t0
        dir_str = f" · {e.direction}" if e.direction else ""
        lines.append(
            f"{t_rel:06.1f}s  {e.event_type:<16}  ID {e.track_id:<4}  {e.class_name}{dir_str}"
        )
    st.code("\n".join(lines), language=None)
else:
    st.markdown(
        '<div class="info-box">Nenhum evento registrado ainda.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Footer note
# -----------------------------------------------------------------------------
st.markdown('<hr class="block-divider">', unsafe_allow_html=True)
st.markdown(
    """
<div class="info-box">
VISION MINI LAB · Relative Thermal Intensity ≠ calibrated temperature ·
Speed = pixels/frame · Physical metrics require geometric calibration.
</div>
""",
    unsafe_allow_html=True,
)
