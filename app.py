"""
VISION MINI LAB
Real-Time Visual Intelligence

Computer vision laboratory:
- YOLO object detection
- Optional persistent tracking
- Motion analysis
- Event detection
- Relative thermal intensity
- Trajectory visualization
- Streamlit analytics
"""

from __future__ import annotations

import logging
import os
import tempfile
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
from src.perception.tracker import Tracker, TrackedObject
from src.thermal.thermal import (
    compute_roi_stats,
    process_thermal,
)
from src.utils.image import (
    load_image_from_bytes,
    resize_keep_aspect,
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision-mini-lab")


# =============================================================================
# PAGE
# =============================================================================

st.set_page_config(
    page_title="VISION MINI LAB",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap'
);

html,
body,
[class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

.stApp {
    background-color: #0c0c0f;
    color: #d0d0d8;
}

h1,
h2,
h3,
h4 {
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

.detection-status {
    background-color: #121218;
    border: 1px solid #1e1e26;
    border-radius: 4px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(
    model_name: str = "yolov8s.pt",
) -> Optional[YOLO]:

    try:
        logger.info("Loading YOLO model: %s", model_name)

        model = YOLO(model_name)

        logger.info("YOLO model loaded successfully")

        return model

    except Exception as exc:
        logger.exception(
            "YOLO model loading failed: %s",
            exc,
        )

        return None


# =============================================================================
# DRAWING
# =============================================================================

def draw_overlays(
    image_rgb: np.ndarray,
    objects: List[TrackedObject],
    roi: Optional[
        Tuple[int, int, int, int]
    ] = None,
    line: Optional[
        Tuple[
            Tuple[int, int],
            Tuple[int, int],
        ]
    ] = None,
    show_motion: bool = False,
) -> np.ndarray:

    """
    Draw clean computer-vision overlays.

    Important:
    Detection objects are rendered even when
    no tracking ID is available.
    """

    out = image_rgb.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # -------------------------------------------------------------------------
    # ROI
    # -------------------------------------------------------------------------

    if roi is not None:

        x1, y1, x2, y2 = roi

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (80, 140, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            "ROI",
            (x1 + 5, y1 + 18),
            font,
            0.45,
            (100, 160, 230),
            1,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------------
    # Virtual line
    # -------------------------------------------------------------------------

    if line is not None:

        (lx1, ly1), (lx2, ly2) = line

        cv2.line(
            out,
            (lx1, ly1),
            (lx2, ly2),
            (220, 160, 60),
            2,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------------
    # Objects
    # -------------------------------------------------------------------------

    for obj in objects:

        x1, y1, x2, y2 = map(
            int,
            obj.bbox,
        )

        # Tracked object
        if obj.track_id > 0:

            box_color = (
                90,
                180,
                255,
            )

            identifier = f"#{obj.track_id}"

        # Detection without tracking
        else:

            box_color = (
                160,
                160,
                170,
            )

            identifier = "DET"

        # Bounding box
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            box_color,
            2,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------------------
        # Label
        # ---------------------------------------------------------------------

        label = (
            f"{obj.class_name.upper()} "
            f"{identifier} "
            f"{obj.confidence:.0%}"
        )

        if (
            show_motion
            and obj.track_id > 0
            and obj.state == "MOVING"
        ):
            label += (
                f"  {obj.direction}"
                f"  {obj.speed:.1f}px/f"
            )

        (
            text_width,
            text_height,
        ), _ = cv2.getTextSize(
            label,
            font,
            0.48,
            1,
        )

        # Keep label inside frame
        label_y = max(
            y1,
            text_height + 10,
        )

        # Label background
        cv2.rectangle(
            out,
            (
                x1,
                label_y - text_height - 9,
            ),
            (
                x1 + text_width + 10,
                label_y,
            ),
            (15, 15, 20),
            -1,
        )

        # Label text
        cv2.putText(
            out,
            label,
            (
                x1 + 5,
                label_y - 5,
            ),
            font,
            0.48,
            (235, 235, 240),
            1,
            cv2.LINE_AA,
        )

    return out


# =============================================================================
# TRAJECTORIES
# =============================================================================

def draw_trajectories(
    image_rgb: np.ndarray,
    tracker: Tracker,
    objects: List[TrackedObject],
) -> np.ndarray:

    out = image_rgb.copy()

    for obj in objects:

        # Do not draw trajectories for
        # temporary detection-only IDs.
        if obj.track_id <= 0:
            continue

        points = tracker.get_trajectory(
            obj.track_id
        )

        if len(points) < 2:
            continue

        points_i = np.array(
            points,
            dtype=np.int32,
        )

        cv2.polylines(
            out,
            [points_i],
            False,
            (60, 120, 200),
            1,
            cv2.LINE_AA,
        )

    return out


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state() -> None:

    defaults = {
        "tracker": None,
        "motion": MotionAnalyzer(),
        "events": EventEngine(),
        "all_centers": [],
        "selected_track": None,
        "frame_count": 0,
        "last_fps": 0.0,
        "processing": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


init_state()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown("### CONTROLS")

    st.markdown(
        '<div class="block-divider"></div>',
        unsafe_allow_html=True,
    )

    conf = st.slider(
        "Confidence",
        0.10,
        0.90,
        0.35,
        0.05,
    )

    iou = st.slider(
        "IoU",
        0.20,
        0.80,
        0.45,
        0.05,
    )

    st.markdown(
        '<div class="block-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### VIEW")

    view_mode = st.radio(
        "Display Mode",
        [
            "RGB",
            "THERMAL",
            "OVERLAY",
        ],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    colormap = st.selectbox(
        "Colormap",
        [
            "inferno",
            "magma",
            "turbo",
        ],
        index=0,
    )

    thermal_opacity = st.slider(
        "Thermal Opacity",
        0.1,
        0.9,
        0.5,
        0.1,
    )

    st.markdown(
        '<div class="block-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### OVERLAYS")

    show_tracks = st.checkbox(
        "Trajectories",
        value=True,
    )

    show_motion_label = st.checkbox(
        "Motion Labels",
        value=False,
    )

    st.markdown(
        '<div class="block-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### ROI")

    enable_roi = st.checkbox(
        "Enable ROI",
        value=False,
    )

    if enable_roi:

        roi_x1 = st.number_input(
            "ROI X1",
            0,
            4000,
            100,
            10,
        )

        roi_y1 = st.number_input(
            "ROI Y1",
            0,
            4000,
            100,
            10,
        )

        roi_x2 = st.number_input(
            "ROI X2",
            0,
            4000,
            500,
            10,
        )

        roi_y2 = st.number_input(
            "ROI Y2",
            0,
            4000,
            400,
            10,
        )

        current_roi = (
            int(roi_x1),
            int(roi_y1),
            int(roi_x2),
            int(roi_y2),
        )

    else:

        current_roi = None

    st.markdown(
        '<div class="block-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### LINE CROSSING")

    enable_line = st.checkbox(
        "Enable Line",
        value=False,
    )

    if enable_line:

        lx1 = st.number_input(
            "Line X1",
            0,
            4000,
            200,
            10,
        )

        ly1 = st.number_input(
            "Line Y1",
            0,
            4000,
            300,
            10,
        )

        lx2 = st.number_input(
            "Line X2",
            0,
            4000,
            600,
            10,
        )

        ly2 = st.number_input(
            "Line Y2",
            0,
            4000,
            300,
            10,
        )

        current_line = (
            (int(lx1), int(ly1)),
            (int(lx2), int(ly2)),
        )

    else:

        current_line = None

    st.markdown(
        '<div class="block-divider"></div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "RESET STATE",
        use_container_width=True,
    ):

        st.session_state.motion.reset()

        st.session_state.events.clear()

        st.session_state.all_centers = []

        st.session_state.frame_count = 0

        st.session_state.selected_track = None

        if (
            st.session_state.tracker
            is not None
        ):
            st.session_state.tracker.reset()

        st.rerun()


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">VISION MINI LAB</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">REAL-TIME VISUAL INTELLIGENCE</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL
# =============================================================================

model = load_yolo_model(
    "yolov8s.pt"
)

if model is None:

    st.error(
        "YOLO não pôde ser carregado. "
        "Verifique a instalação do Ultralytics."
    )

    st.stop()


# =============================================================================
# TRACKER
# =============================================================================

if (
    st.session_state.tracker
    is None
):

    st.session_state.tracker = Tracker(
        model,
        conf=conf,
        iou=iou,
    )

else:

    st.session_state.tracker.set_thresholds(
        conf,
        iou,
    )


tracker: Tracker = (
    st.session_state.tracker
)

motion: MotionAnalyzer = (
    st.session_state.motion
)

events: EventEngine = (
    st.session_state.events
)


# =============================================================================
# EVENT CONFIG
# =============================================================================

events.set_roi(
    current_roi
)

if current_line is not None:

    events.set_line(
        current_line[0],
        current_line[1],
    )

else:

    events.set_line(
        None,
        None,
    )


# =============================================================================
# INPUT
# =============================================================================

st.markdown("### INPUT")

input_type = st.radio(
    "Source",
    [
        "Image",
        "Video",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

uploaded = None

if input_type == "Image":

    uploaded = st.file_uploader(
        "Upload Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
            "tiff",
            "tif",
        ],
        label_visibility="collapsed",
    )

else:

    uploaded = st.file_uploader(
        "Upload Video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm",
        ],
        label_visibility="collapsed",
    )


st.markdown(
    '<hr class="block-divider">',
    unsafe_allow_html=True,
)


# =============================================================================
# VARIABLES
# =============================================================================

display_image = None

objects: List[TrackedObject] = []

intensity_map = None

thermal_stats = None

img_shape = None


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

if (
    uploaded is not None
    and input_type == "Image"
):

    data = uploaded.getvalue()

    image = load_image_from_bytes(
        data
    )

    if image is None:

        st.error(
            "Não foi possível carregar esta imagem."
        )

    else:

        image, _ = resize_keep_aspect(
            image,
            max_side=1280,
        )

        img_shape = image.shape

        start_time = (
            time.perf_counter()
        )

        # -------------------------------------------------------------
        # YOLO + TRACKING
        # -------------------------------------------------------------

        objects = tracker.update(
            image
        )

        # -------------------------------------------------------------
        # MOTION
        # -------------------------------------------------------------

        try:

            objects = motion.update(
                objects
            )

        except Exception as exc:

            logger.exception(
                "Motion analysis failed: %s",
                exc,
            )

        # -------------------------------------------------------------
        # EVENTS
        # -------------------------------------------------------------

        try:

            events.update(
                objects
            )

        except Exception as exc:

            logger.exception(
                "Event engine failed: %s",
                exc,
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        st.session_state.last_fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )

        st.session_state.frame_count += 1

        # -------------------------------------------------------------
        # CENTERS
        # -------------------------------------------------------------

        for obj in objects:

            st.session_state.all_centers.append(
                obj.center
            )

        if len(
            st.session_state.all_centers
        ) > 5000:

            st.session_state.all_centers = (
                st.session_state.all_centers[
                    -5000:
                ]
            )

        # -------------------------------------------------------------
        # THERMAL
        # -------------------------------------------------------------

        mode_map = {
            "RGB": "rgb",
            "THERMAL": "thermal",
            "OVERLAY": "overlay",
        }

        (
            display_image,
            intensity_map,
            thermal_stats,
        ) = process_thermal(
            image,
            mode=mode_map[
                view_mode
            ],
            colormap=colormap,
            opacity=thermal_opacity,
        )

        # -------------------------------------------------------------
        # TRAJECTORIES
        # -------------------------------------------------------------

        if show_tracks:

            display_image = (
                draw_trajectories(
                    display_image,
                    tracker,
                    objects,
                )
            )

        # -------------------------------------------------------------
        # BOUNDING BOXES
        # -------------------------------------------------------------

        display_image = draw_overlays(
            display_image,
            objects,
            current_roi,
            current_line,
            show_motion_label,
        )


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

elif (
    uploaded is not None
    and input_type == "Video"
):

    tmp_path = None

    try:

        suffix = os.path.splitext(
            uploaded.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:

            tmp.write(
                uploaded.getvalue()
            )

            tmp_path = tmp.name

        cap = cv2.VideoCapture(
            tmp_path
        )

        if not cap.isOpened():

            st.error(
                "Não foi possível abrir este vídeo."
            )

        else:

            fps_video = (
                cap.get(
                    cv2.CAP_PROP_FPS
                )
                or 25.0
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            duration = (
                total_frames / fps_video
                if fps_video > 0
                else 0
            )

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "VIDEO FPS",
                f"{fps_video:.1f}",
            )

            c2.metric(
                "RESOLUTION",
                f"{width}×{height}",
            )

            c3.metric(
                "FRAMES",
                f"{total_frames}",
            )

            c4.metric(
                "DURATION",
                f"{duration:.1f}s",
            )

            # ---------------------------------------------------------
            # Frame selector
            # ---------------------------------------------------------

            max_frames = min(
                total_frames,
                300,
            )

            frame_idx = st.slider(
                "Frame",
                0,
                max(
                    0,
                    max_frames - 1,
                ),
                0,
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_idx,
            )

            ret, frame_bgr = (
                cap.read()
            )

            if not ret:

                st.error(
                    "Não foi possível ler este frame."
                )

            else:

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                frame_rgb, _ = (
                    resize_keep_aspect(
                        frame_rgb,
                        max_side=1280,
                    )
                )

                img_shape = (
                    frame_rgb.shape
                )

                start_time = (
                    time.perf_counter()
                )

                # -----------------------------------------------------
                # YOLO
                # -----------------------------------------------------

                objects = tracker.update(
                    frame_rgb
                )

                # -----------------------------------------------------
                # Motion
                # -----------------------------------------------------

                try:

                    objects = motion.update(
                        objects
                    )

                except Exception as exc:

                    logger.exception(
                        "Motion error: %s",
                        exc,
                    )

                # -----------------------------------------------------
                # Events
                # -----------------------------------------------------

                try:

                    events.update(
                        objects,
                        frame_time=(
                            frame_idx
                            / fps_video
                        ),
                    )

                except Exception as exc:

                    logger.exception(
                        "Event error: %s",
                        exc,
                    )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                st.session_state.last_fps = (
                    1.0 / elapsed
                    if elapsed > 0
                    else 0.0
                )

                st.session_state.frame_count = (
                    frame_idx
                )

                # -----------------------------------------------------
                # Centers
                # -----------------------------------------------------

                for obj in objects:

                    st.session_state.all_centers.append(
                        obj.center
                    )

                if len(
                    st.session_state.all_centers
                ) > 5000:

                    st.session_state.all_centers = (
                        st.session_state.all_centers[
                            -5000:
                        ]
                    )

                # -----------------------------------------------------
                # Thermal
                # -----------------------------------------------------

                mode_map = {
                    "RGB": "rgb",
                    "THERMAL": "thermal",
                    "OVERLAY": "overlay",
                }

                (
                    display_image,
                    intensity_map,
                    thermal_stats,
                ) = process_thermal(
                    frame_rgb,
                    mode=mode_map[
                        view_mode
                    ],
                    colormap=colormap,
                    opacity=thermal_opacity,
                )

                # -----------------------------------------------------
                # Trajectories
                # -----------------------------------------------------

                if show_tracks:

                    display_image = (
                        draw_trajectories(
                            display_image,
                            tracker,
                            objects,
                        )
                    )

                # -----------------------------------------------------
                # Bounding boxes
                # -----------------------------------------------------

                display_image = draw_overlays(
                    display_image,
                    objects,
                    current_roi,
                    current_line,
                    show_motion_label,
                )

            cap.release()

    except Exception as exc:

        logger.exception(
            "Video processing failed: %s",
            exc,
        )

        st.error(
            f"Erro no processamento do vídeo: {exc}"
        )

    finally:

        if tmp_path is not None:

            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# =============================================================================
# LIVE VIEW
# =============================================================================

col_view, col_info = st.columns(
    [1.6, 1.0]
)


with col_view:

    st.markdown(
        "### LIVE VIEW"
    )

    if display_image is not None:

        st.image(
            display_image,
            width="stretch",
            channels="RGB",
        )

    else:

        st.markdown(
            """
            <div class="info-box">
            Aguardando entrada de imagem ou vídeo.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# OBJECT INTELLIGENCE
# =============================================================================

with col_info:

    st.markdown(
        "### OBJECT INTELLIGENCE"
    )

    active_objects = len(
        objects
    )

    tracked_objects = len(
        tracker.histories
    )

    try:

        inside_roi = (
            events.count_inside_roi(
                objects
            )
        )

    except Exception:

        inside_roi = 0

    m1, m2 = st.columns(2)

    m1.metric(
        "Active Objects",
        active_objects,
    )

    m2.metric(
        "Tracked",
        tracked_objects,
    )

    m3, m4 = st.columns(2)

    m3.metric(
        "FPS",
        f"{st.session_state.last_fps:.1f}",
    )

    m4.metric(
        "Inside ROI",
        inside_roi,
    )

    # -------------------------------------------------------------
    # Detection status
    # -------------------------------------------------------------

    if objects:

        st.markdown(
            f"""
            <div class="detection-status">
            DETECTION ENGINE · {len(objects)} OBJECT(S)
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="detection-status">
            DETECTION ENGINE · NO OBJECTS
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Object selector
    # -------------------------------------------------------------

    if objects:

        options = {}

        for obj in objects:

            if obj.track_id > 0:

                label = (
                    f"{obj.class_name} "
                    f"#{obj.track_id}"
                )

            else:

                label = (
                    f"{obj.class_name} "
                    f"[DET]"
                )

            options[label] = (
                obj.track_id
            )

        selected_label = st.selectbox(
            "Selected Object",
            list(options.keys()),
            label_visibility="collapsed",
        )

        selected_id = options.get(
            selected_label
        )

        st.session_state.selected_track = (
            selected_id
        )

        selected_obj = next(
            (
                obj
                for obj in objects
                if obj.track_id
                == selected_id
            ),
            None,
        )

        if selected_obj is not None:

            if selected_obj.track_id > 0:

                id_text = (
                    f"#{selected_obj.track_id}"
                )

            else:

                id_text = "Detection only"

            st.markdown(
                f"""
                <div class="info-box">
                ID · {id_text}<br>
                Class · {selected_obj.class_name}<br>
                Confidence · {selected_obj.confidence:.1%}<br>
                State · {selected_obj.state}<br>
                Direction · {selected_obj.direction}<br>
                Speed · {selected_obj.speed:.2f} px/frame<br>
                Center · (
                {selected_obj.center[0]:.0f},
                {selected_obj.center[1]:.0f}
                )
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.markdown(
            """
            <div class="info-box">
            Nenhum objeto ativo.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Thermal
    # -------------------------------------------------------------

    if thermal_stats is not None:

        st.markdown(
            "#### THERMAL INTENSITY"
        )

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

        if (
            current_roi is not None
            and intensity_map is not None
        ):

            roi_stats = (
                compute_roi_stats(
                    intensity_map,
                    current_roi,
                )
            )

            if roi_stats:

                st.markdown(
                    "#### THERMAL ROI"
                )

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


# =============================================================================
# ANALYTICS
# =============================================================================

st.markdown(
    '<hr class="block-divider">',
    unsafe_allow_html=True,
)

st.markdown(
    "### ANALYTICS"
)


if (
    objects
    or st.session_state.all_centers
):

    # -------------------------------------------------------------------------
    # Object activity
    # -------------------------------------------------------------------------

    class_counts: Dict[
        str,
        int,
    ] = defaultdict(int)

    for obj in objects:

        class_counts[
            obj.class_name
        ] += 1

    if class_counts:

        fig1 = object_activity_chart(
            dict(class_counts)
        )

        st.plotly_chart(
            fig1,
            width="stretch",
            config=MODEBAR_CONFIG,
        )

        st.caption(
            "Objetos ativos por classe no frame atual."
        )

    # -------------------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------------------

    if objects:

        fig2 = confidence_chart(
            objects
        )

        st.plotly_chart(
            fig2,
            width="stretch",
            config=MODEBAR_CONFIG,
        )

        st.caption(
            "Distribuição de confiança das detecções."
        )

    # -------------------------------------------------------------------------
    # Motion
    # -------------------------------------------------------------------------

    if objects:

        fig3 = motion_chart(
            objects
        )

        st.plotly_chart(
            fig3,
            width="stretch",
            config=MODEBAR_CONFIG,
        )

        st.caption(
            "Velocidade instantânea em pixels/frame."
        )

    # -------------------------------------------------------------------------
    # Trajectory
    # -------------------------------------------------------------------------

    trajectories = {
        track_id:
        tracker.get_trajectory(
            track_id
        )
        for track_id
        in tracker.histories
    }

    current_positions = {
        obj.track_id:
        obj.center
        for obj in objects
        if obj.track_id > 0
    }

    class_names = {
        obj.track_id:
        obj.class_name
        for obj in objects
        if obj.track_id > 0
    }

    for track_id in trajectories:

        if track_id not in class_names:

            class_names[
                track_id
            ] = "obj"

    if trajectories:

        fig4 = trajectory_chart(
            trajectories,
            current_positions,
            class_names,
            selected_id=(
                st.session_state
                .selected_track
            ),
            img_shape=img_shape,
        )

        st.plotly_chart(
            fig4,
            width="stretch",
            config=MODEBAR_CONFIG,
        )

        st.caption(
            "Trajetórias recentes dos objetos rastreados."
        )

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    recent_events = (
        events.get_recent(80)
    )

    if recent_events:

        fig5 = event_timeline_chart(
            recent_events
        )

        st.plotly_chart(
            fig5,
            width="stretch",
            config=MODEBAR_CONFIG,
        )

        st.caption(
            "Linha do tempo de eventos detectados."
        )

    # -------------------------------------------------------------------------
    # Thermal
    # -------------------------------------------------------------------------

    if intensity_map is not None:

        fig6 = thermal_analysis_chart(
            intensity_map
        )

        st.plotly_chart(
            fig6,
            width="stretch",
            config=MODEBAR_CONFIG,
        )

        st.caption(
            "Intensidade relativa. "
            "Não representa temperatura calibrada."
        )

    # -------------------------------------------------------------------------
    # Movement density
    # -------------------------------------------------------------------------

    if (
        st.session_state.all_centers
        and img_shape is not None
    ):

        fig7 = movement_density_heatmap(
            st.session_state.all_centers,
            img_shape,
        )

        st.plotly_chart(
            fig7,
            width="stretch",
            config=MODEBAR_CONFIG,
        )

        st.caption(
            "Densidade espacial das trajetórias."
        )

else:

    st.markdown(
        """
        <div class="info-box">
        Carregue uma imagem ou vídeo para gerar analytics.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# EVENT TIMELINE
# =============================================================================

st.markdown(
    '<hr class="block-divider">',
    unsafe_allow_html=True,
)

st.markdown(
    "### EVENT TIMELINE"
)

recent = events.get_recent(
    30
)

if recent:

    lines = []

    t0 = recent[0].timestamp

    for event in reversed(
        recent
    ):

        t_rel = (
            event.timestamp
            - t0
        )

        direction = (
            f" · {event.direction}"
            if event.direction
            else ""
        )

        lines.append(
            f"{t_rel:06.1f}s  "
            f"{event.event_type:<16}  "
            f"ID {event.track_id:<4}  "
            f"{event.class_name}"
            f"{direction}"
        )

    st.code(
        "\n".join(lines),
        language=None,
    )

else:

    st.markdown(
        """
        <div class="info-box">
        Nenhum evento registrado ainda.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    '<hr class="block-divider">',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    VISION MINI LAB · Relative Thermal Intensity ≠ calibrated temperature ·
    Speed = pixels/frame · Physical metrics require geometric calibration.
    </div>
    """,
    unsafe_allow_html=True,
)