import base64
import datetime
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import pandas as pd
from src import charts

from src.detector import YoloDetector
from src.metrics import MetricsAggregator, compute_box_metrics
from src.video import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    decode_image_bytes,
    get_video_metadata,
    save_uploaded_file,
)
from src.processor import FrameProcessor
from src.report import generate_report
from src.export import export_detections_csv, export_report_json
from src.visualization import draw_grid, draw_centers, draw_trajectory, create_scene_map, draw_roi, draw_line
from src.temporal import TemporalAnalyzer
from src.spatial import RegionOfInterest, LineCrossingDetector
from src.motion import MotionAnalyzer
from src.events import EventEngine
from src.tracking import compute_tracking_metrics, build_tracked_objects

st.set_page_config(
    page_title="VISION MINI LAB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Estado da aplicação
# -----------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "video_status": "idle",
        "video_path": None,
        "video_meta": None,
        "video_frame_index": 0,
        "video_upload_token": None,
        "video_error": None,
        "camera_status": "idle",
        "camera_error": None,
        "latest_frame": None,
        "capture_message": None,
        "groq_message": None,
        "metrics": MetricsAggregator(),
        "processor": None,
        "video_running": False,
        "temporal_analyzer": TemporalAnalyzer(max_history=300),
        "show_confidence_legend": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

init_state()

# -----------------------------------------------------------------------------
# Detector YOLO
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_detector():
    try:
        model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        detector = YoloDetector(model_name=model_name)
        return detector, None
    except Exception as exc:
        return None, str(exc)

detector, detector_error = load_detector()

# -----------------------------------------------------------------------------
# Groq (opcional)
# -----------------------------------------------------------------------------

def get_groq_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key if key else None

def describe_with_groq(image_bgr: np.ndarray, prompt: str):
    api_key = get_groq_api_key()
    if not api_key:
        return None, "DISABLED"
    try:
        import requests
    except Exception as exc:
        return None, f"requests não instalado: {exc}"
    try:
        img = image_bgr
        h, w = img.shape[:2]
        max_side = 768
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None, "Falha ao codificar imagem."
        b64 = base64.b64encode(buffer).decode('utf-8')
        payload = {
            "model": os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "300")),
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"], None
    except Exception as exc:
        return None, str(exc)

def run_groq_on_latest():
    frame = st.session_state.get("latest_frame")
    if frame is None:
        st.session_state.groq_message = "Sem frame."
        return
    prompt = (
        "Descreva a cena em português, com foco em objetos visíveis e disposição "
        "espacial. Não invente medições. Não use unidades físicas como metros ou km/h."
    )
    text, err = describe_with_groq(frame, prompt)
    if err:
        st.session_state.groq_message = f"GROQ ERROR: {err}"
    else:
        st.session_state.groq_message = text

# -----------------------------------------------------------------------------
# Funções auxiliares de desenho (versão simplificada)
# -----------------------------------------------------------------------------

def get_confidence_color(confidence: float):
    if confidence >= 0.7:
        return (0, 255, 0)   # verde
    elif confidence >= 0.4:
        return (0, 255, 255) # amarelo
    else:
        return (0, 0, 255)   # vermelho

def draw_detections(
    frame_bgr: np.ndarray,
    detections,
    show_details: bool = True,
    draw_tracks: bool = False,
    track_history: dict = None,
    draw_centers: bool = True,
    show_confidence_legend: bool = True,
) -> np.ndarray:
    img = frame_bgr.copy()
    h_img, w_img = img.shape[:2]

    if draw_tracks and track_history:
        for tid, history in track_history.items():
            if len(history) > 1:
                color = (255, 255, 255)
                pts = np.array([(int(x), int(y)) for x, y in history], dtype=np.int32)
                cv2.polylines(img, [pts], False, color, 2)

    for det in detections:
        color = get_confidence_color(det.confidence)

        x1 = max(0, int(det.x1))
        y1 = max(0, int(det.y1))
        x2 = min(w_img - 1, int(det.x2))
        y2 = min(h_img - 1, int(det.y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if draw_centers:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)
            cv2.circle(img, (cx, cy), 2, color, -1)

        label_parts = []
        if det.track_id is not None:
            label_parts.append(f"ID {det.track_id}")
        label_parts.append(det.label.upper())
        label_parts.append(f"{det.confidence:.0%}")
        label_text = " · ".join(label_parts)

        text_x = max(5, x1)
        text_y = y1 - 8
        if text_y < 15:
            text_y = min(h_img - 5, y2 + 18)

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (text_x - 2, text_y - th - 4), (text_x + tw + 2, text_y + 2), (0, 0, 0), -1)
        cv2.putText(img, label_text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if show_confidence_legend:
        legend_x = w_img - 160
        legend_y = 20
        cv2.rectangle(img, (legend_x, legend_y), (legend_x + 140, legend_y + 90), (50, 50, 50), -1)
        cv2.rectangle(img, (legend_x, legend_y), (legend_x + 140, legend_y + 90), (200, 200, 200), 1)

        items = [
            ("HIGH (>=70%)", (0, 255, 0)),
            ("MEDIUM (40-70%)", (0, 255, 255)),
            ("LOW (<40%)", (0, 0, 255)),
        ]
        for i, (text, col) in enumerate(items):
            y = legend_y + 20 + i * 25
            cv2.circle(img, (legend_x + 15, y + 2), 6, col, -1)
            cv2.putText(img, text, (legend_x + 30, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return img

# -----------------------------------------------------------------------------
# Funções para exibir relatório formatado
# -----------------------------------------------------------------------------

def display_report_section(report: dict) -> None:
    """Exibe o relatório técnico completo de forma legível."""
    st.markdown("### 1. EXECUTIVE SUMMARY")
    st.markdown(f"> {report['executive_summary']['summary']}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Objects", report['executive_summary']['total_objects'])
    col2.metric("Classes", report['executive_summary']['unique_classes'])
    col3.metric("Dominant", report['executive_summary']['dominant_class'] or "N/A")
    col4.metric("Avg Confidence", f"{report['executive_summary']['avg_confidence']:.1%}")
    col5.metric("Quality", report['executive_summary']['quality_level'])

    st.markdown("### 2. DETECTION ANALYSIS")
    da = report['detection_analysis']
    conf_stats = da['confidence_stats']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{conf_stats['mean']:.1%}")
    c2.metric("Median", f"{conf_stats['median']:.1%}")
    c3.metric("Min", f"{conf_stats['min']:.1%}")
    c4.metric("Max", f"{conf_stats['max']:.1%}")
    st.caption(f"Std Dev: {conf_stats['std']:.2f}")

    st.markdown("**Confidence Distribution:**")
    for range_name, data in da['confidence_ranges'].items():
        st.progress(data['percentage'] / 100, text=f"{range_name.title()}: {data['count']} ({data['percentage']:.0f}%)")

    st.markdown("### 3. SPATIAL ANALYSIS")
    sa = report['spatial_analysis']
    st.markdown(f"> {sa['interpretation']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Most Occupied", sa['most_occupied_region'] or "N/A")
    c2.metric("Density", f"{sa['density_per_million_pixels']:.1f} obj/Mpx")
    c3.metric("Centroid", f"({sa['centroid']['x']:.0f}, {sa['centroid']['y']:.0f})")

    st.markdown("### 4. IMAGE OCCUPANCY")
    occ = report['image_occupancy']
    c1, c2, c3 = st.columns(3)
    c1.metric("Coverage (Union)", f"{occ['union_coverage_ratio']:.1%}")
    c2.metric("Mean Area", f"{occ['mean_area']:.0f} px²")
    c3.metric("Largest", f"{occ['largest_detection']} ({occ['max_area']:.0f} px²)")
    c4, c5, c6 = st.columns(3)
    c4.metric("Median Area", f"{occ['median_area']:.0f} px²")
    c5.metric("Smallest", f"{occ['smallest_detection']} ({occ['min_area']:.0f} px²)")
    c6.metric("Std Area", f"{occ['std_area']:.0f} px²")

    st.markdown("### 5. CLASS ANALYSIS")
    ca = report['class_analysis']
    if ca:
        for label, data in ca.items():
            with st.expander(f"{label} ({data['count']} objects)"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Count", data['count'])
                c2.metric("Mean Confidence", f"{data['mean_confidence']:.1%}")
                c3.metric("Mean Area", f"{data['mean_area']:.0f} px²")
                c4.metric("Dominant Region", data['dominant_region'] or "N/A")
    else:
        st.caption("Nenhuma classe detectada.")

    st.markdown("### 6. DETECTION RANKING")
    rank_df = pd.DataFrame(report['detection_ranking'])
    if not rank_df.empty:
        st.dataframe(
            rank_df.style.format({
                'confidence': '{:.0%}',
                'relative_area': '{:.2%}',
                'aspect_ratio': '{:.2f}',
                'distance_to_center': '{:.3f}',
            }),
            use_container_width=True,
            height=300
        )
    else:
        st.caption("Nenhuma detecção para classificar.")

    st.markdown("### 7. LOW-CONFIDENCE REVIEW")
    low = report['low_confidence_review']
    if low:
        for item in low[:10]:
            st.warning(f"Detection #{item['id']}: {item['class']} — Confidence: {item['confidence']:.0%} — Region: {item['region']}")
        if len(low) > 10:
            st.caption(f"... e mais {len(low) - 10} detecções com baixa confiança.")
    else:
        st.success("Nenhuma detecção com baixa confiança.")

    st.markdown("### 8. OVERLAP ANALYSIS")
    oa = report['overlap_analysis']
    st.markdown(f"> {oa['interpretation']}")
    c1, c2 = st.columns(2)
    c1.metric("Max IoU", f"{oa['max_iou']:.2f}")
    c2.metric("Pairs > 0.3", oa['pairs_above_threshold'])
    if oa['pairs']:
        with st.expander(f"View {len(oa['pairs'])} overlapping pairs"):
            for pair in oa['pairs'][:10]:
                st.caption(f"#{pair['id1']} {pair['class1']} ↔ #{pair['id2']} {pair['class2']} — IoU: {pair['iou']:.2f} ({pair['level']})")

    st.markdown("### 9. SCENE PROFILE")
    sp = report['scene_profile']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects", sp['objects'])
    c2.metric("Classes", sp['classes'])
    c3.metric("Density", sp['density'])
    c4.metric("Distribution", sp['distribution'])
    st.markdown(f"> {report['scene_interpretation']}")

    st.markdown("### 10. QUALITY ASSESSMENT")
    qa = report['quality_assessment']
    quality_color = "🟢" if qa['level'] == "HIGH" else "🟡" if qa['level'] == "MEDIUM" else "🔴"
    st.markdown(f"{quality_color} **Score:** {qa['score']:.2f} — **Level:** {qa['level']}")
    st.markdown(f"> {qa['interpretation']}")

    st.markdown("### 11. LIMITATIONS")
    for lim in report['limitations']:
        st.caption(f"• {lim}")

# -----------------------------------------------------------------------------
# ROI / Line Crossing — construção a partir da configuração da sidebar
# -----------------------------------------------------------------------------

def build_roi_from_state(width: int, height: int) -> Optional[RegionOfInterest]:
    if not st.session_state.get("roi_enabled") or not width or not height:
        return None
    x1 = st.session_state.get("roi_x1", 25) / 100.0 * width
    y1 = st.session_state.get("roi_y1", 25) / 100.0 * height
    x2 = st.session_state.get("roi_x2", 75) / 100.0 * width
    y2 = st.session_state.get("roi_y2", 75) / 100.0 * height
    return RegionOfInterest(x1, y1, x2, y2, name="ROI")

def build_line_from_state(width: int, height: int) -> Optional[LineCrossingDetector]:
    if not st.session_state.get("line_enabled") or not width or not height:
        return None
    x1 = st.session_state.get("line_x1", 10) / 100.0 * width
    y1 = st.session_state.get("line_y1", 50) / 100.0 * height
    x2 = st.session_state.get("line_x2", 90) / 100.0 * width
    y2 = st.session_state.get("line_y2", 50) / 100.0 * height
    return LineCrossingDetector(x1, y1, x2, y2, name="LINE")

# -----------------------------------------------------------------------------
# Callbacks (mantidos iguais)
# -----------------------------------------------------------------------------

def start_video():
    if st.session_state.processor:
        st.session_state.processor.stop()
    try:
        meta = st.session_state.video_meta
        roi = build_roi_from_state(meta.width, meta.height) if meta else None
        line = build_line_from_state(meta.width, meta.height) if meta else None
        proc = FrameProcessor(
            video_path=str(st.session_state.video_path),
            detector=detector,
            conf_threshold=st.session_state.conf_threshold,
            sample_every=st.session_state.sample_every,
            roi=roi,
            line_detector=line,
        )
        proc.start()
        st.session_state.processor = proc
        st.session_state.video_status = "running"
        st.session_state.video_error = None
        st.session_state.temporal_analyzer = TemporalAnalyzer(max_history=300)
    except Exception as exc:
        st.session_state.video_error = f"Erro: {exc}"
        st.session_state.video_status = "idle"

def pause_video():
    if st.session_state.processor:
        st.session_state.processor.pause()
        st.session_state.video_status = "paused"

def resume_video():
    if st.session_state.processor:
        st.session_state.processor.resume()
        st.session_state.video_status = "running"

def stop_video():
    if st.session_state.processor:
        st.session_state.processor.stop()
        st.session_state.processor = None
    st.session_state.video_status = "stopped"
    st.session_state.video_frame_index = 0

def start_camera():
    st.session_state.camera_status = "running"
    st.session_state.camera_error = None
    st.session_state.metrics.reset()
    st.session_state.temporal_analyzer = TemporalAnalyzer(max_history=300)
    if detector is not None:
        detector.reset_tracker()

def stop_camera():
    st.session_state.camera_status = "stopped"

def capture_current_frame():
    frame = st.session_state.get("latest_frame")
    if frame is None:
        st.session_state.capture_message = "Nenhum frame."
        return
    capture_dir = Path("data/captures")
    capture_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.datetime.now().strftime("capture_%Y%m%d_%H%M%S")
    candidate = capture_dir / f"{base}.jpg"
    counter = 1
    while candidate.exists():
        candidate = capture_dir / f"{base}_{counter}.jpg"
        counter += 1
    try:
        ok = cv2.imwrite(str(candidate), frame)
        st.session_state.capture_message = f"Salva em {candidate}" if ok else "Falha."
    except Exception as exc:
        st.session_state.capture_message = f"Erro: {exc}"

# -----------------------------------------------------------------------------
# Câmera
# -----------------------------------------------------------------------------

def run_camera_loop(conf_threshold, sample_every, tracking_enabled, show_details, draw_centers, show_legend):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state.camera_error = "Webcam indisponível."
        st.session_state.camera_status = "idle"
        return
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    intelligence_placeholder = st.empty()
    frame_count = 0
    last_detections = []
    last_read = time.perf_counter()
    source_fps = None
    loop_start = time.perf_counter()

    # Estado local do pipeline de inteligência (recriado a cada sessão de
    # câmera — o loop é um único bloco contínuo, sem rerun por frame).
    track_history: dict = {}
    motion_analyzer = MotionAnalyzer()
    event_engine = EventEngine()
    previously_active_ids: set = set()
    roi = None
    line_detector = None

    try:
        while cap.isOpened():
            if st.session_state.camera_status != "running":
                break
            t_start = time.perf_counter()
            ret, frame_bgr = cap.read()
            decode_ms = (time.perf_counter() - t_start) * 1000.0
            if not ret:
                st.session_state.camera_error = "Falha ao ler webcam."
                break
            frame_count += 1
            h, w = frame_bgr.shape[:2]

            if roi is None:
                roi = build_roi_from_state(w, h)
            if line_detector is None:
                line_detector = build_line_from_state(w, h)

            now = time.perf_counter()
            interval = now - last_read
            if frame_count > 1 and interval > 0:
                instant_fps = 1.0 / interval
                if source_fps is None:
                    source_fps = instant_fps
                else:
                    source_fps = 0.85 * source_fps + 0.15 * instant_fps
            last_read = now
            do_infer = detector is not None and (frame_count % sample_every == 0 or frame_count == 1)
            infer_ms = None
            if do_infer:
                t_infer = time.perf_counter()
                try:
                    if tracking_enabled:
                        last_detections = detector.track(frame_bgr, conf=conf_threshold, persist=True)
                    else:
                        last_detections = detector.detect(frame_bgr, conf=conf_threshold)
                except Exception as exc:
                    st.session_state.camera_error = f"Erro: {exc}"
                    last_detections = []
                infer_ms = (time.perf_counter() - t_infer) * 1000.0

            # --- Motion / ROI / Line Crossing / Events (só fazem sentido
            # com tracking ativo, já que dependem de track_id estável) -----
            tracking_metrics = {}
            motion_metrics = {}
            roi_snapshot = None
            line_crossings = []
            timestamp = time.time()
            if tracking_enabled and last_detections:
                for det in last_detections:
                    if det.track_id is not None:
                        cx, cy = (det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2
                        history = track_history.setdefault(det.track_id, [])
                        history.append((cx, cy))
                        if len(history) > 10:
                            history.pop(0)
                tracking_metrics = compute_tracking_metrics(track_history, last_detections)

                active_ids = [d.track_id for d in last_detections if d.track_id is not None]
                current_active = set(active_ids)
                for tid in current_active - previously_active_ids:
                    label = next((d.label for d in last_detections if d.track_id == tid), None)
                    event_engine.emit("OBJECT_ENTERED", object_id=tid, class_name=label, frame_index=frame_count, timestamp=timestamp)
                for tid in previously_active_ids - current_active:
                    event_engine.emit("OBJECT_EXITED", object_id=tid, frame_index=frame_count, timestamp=timestamp)
                previously_active_ids = current_active

                for tid, tm in tracking_metrics.items():
                    m = motion_analyzer.update(tid, tm["dx"], tm["dy"], tm["displacement_px"])
                    motion_metrics[tid] = m
                    if m["transition_event"]:
                        label = next((d.label for d in last_detections if d.track_id == tid), None)
                        event_engine.emit(m["transition_event"], object_id=tid, class_name=label, frame_index=frame_count, timestamp=timestamp)
                motion_analyzer.prune(active_ids)

                tracked_objects = build_tracked_objects(last_detections)
                if roi is not None:
                    roi_snapshot = roi.update(tracked_objects, timestamp=timestamp, event_engine=event_engine, frame_index=frame_count)
                if line_detector is not None:
                    line_crossings = line_detector.update(tracked_objects, timestamp=timestamp, event_engine=event_engine, frame_index=frame_count)
                    line_detector.prune(active_ids)

                time_sec = now - loop_start
                st.session_state.temporal_analyzer.record_detections(frame_count, time_sec, last_detections, motion_metrics)
                if roi_snapshot is not None:
                    st.session_state.temporal_analyzer.record_roi_snapshot(frame_count, time_sec, roi_snapshot)
                if line_crossings:
                    st.session_state.temporal_analyzer.record_line_crossings(line_crossings)

            annotated = draw_detections(
                frame_bgr,
                last_detections,
                show_details=show_details,
                draw_centers=draw_centers,
                show_confidence_legend=show_legend,
            )
            if roi is not None:
                annotated = draw_roi(annotated, roi)
            if line_detector is not None:
                annotated = draw_line(annotated, line_detector)

            st.session_state.latest_frame = annotated
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="auto", use_container_width=True)
            total_ms = (time.perf_counter() - t_start) * 1000.0
            st.session_state.metrics.note_frame_displayed(
                frame_index=frame_count,
                time_sec=now - loop_start,
                resolution=(w, h),
                source_fps=source_fps,
            )
            if do_infer:
                st.session_state.metrics.note_inference(
                    last_detections,
                    decode_ms=decode_ms,
                    infer_ms=infer_ms,
                    total_ms=total_ms,
                )
            with metrics_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Frame", frame_count)
                c2.metric("Objetos", len(last_detections))
                c3.metric("Confiança", f"{st.session_state.metrics.last_avg_confidence:.0%}" if st.session_state.metrics.last_avg_conf_count else "--")

            if tracking_enabled and (roi_snapshot is not None or line_detector is not None or motion_metrics):
                with intelligence_placeholder.container():
                    status_counts = {}
                    for m in motion_metrics.values():
                        status_counts[m["status"]] = status_counts.get(m["status"], 0) + 1
                    cols = st.columns(4)
                    cols[0].metric("Parados", status_counts.get("PARADO", 0))
                    cols[1].metric("Movendo", status_counts.get("MOVENDO", 0))
                    cols[2].metric("Rápidos", status_counts.get("MOVIMENTO_RAPIDO", 0))
                    if roi_snapshot is not None:
                        cols[3].metric("Ocupação ROI", roi_snapshot["occupancy"])
                    if line_detector is not None:
                        st.caption(
                            f"Linha: {line_detector.crossing_counts['forward']} A→B · "
                            f"{line_detector.crossing_counts['backward']} B→A"
                        )
    finally:
        cap.release()
        if st.session_state.camera_status == "running":
            st.session_state.camera_status = "idle"

# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------

# Cabeçalho refinado
st.markdown(
    """
    <div style="margin-bottom: -10px;">
        <h1 style="font-size: 2.5rem; font-weight: 600; margin-bottom: 0;">VISION MINI LAB</h1>
        <p style="font-size: 1rem; color: #666; margin-top: 0;">Computer Vision Scene Analysis — detection, spatial & temporal metrics</p>
        <hr style="margin-top: 0.5rem; margin-bottom: 1rem; border: 0; border-top: 2px solid #eee;">
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar minimalista
with st.sidebar:
    st.subheader("⚙️ Configurações")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        key="conf_threshold",
    )
    sample_every = st.selectbox(
        "Sample every (frames)",
        options=[1, 2, 3, 5, 10],
        index=0,
        key="sample_every",
    )
    tracking_enabled = st.checkbox("Tracking", value=True, key="tracking_enabled")
    draw_centers = st.checkbox("Mostrar centros", value=True, key="draw_centers")
    show_trajectory = st.checkbox("Mostrar trajetórias", value=False, key="show_trajectory")
    show_legend = st.checkbox("Mostrar legenda de confiança", value=True, key="show_legend")
    st.divider()

    st.subheader("Zona de Interesse (ROI)")
    st.caption("Aplica-se aos modos Vídeo e Câmera (requer Tracking ativo).")
    roi_enabled = st.checkbox("Ativar ROI", value=False, key="roi_enabled")
    if roi_enabled:
        rc1, rc2 = st.columns(2)
        with rc1:
            st.slider("ROI X1 (%)", 0, 100, 25, key="roi_x1")
            st.slider("ROI Y1 (%)", 0, 100, 25, key="roi_y1")
        with rc2:
            st.slider("ROI X2 (%)", 0, 100, 75, key="roi_x2")
            st.slider("ROI Y2 (%)", 0, 100, 75, key="roi_y2")

    st.subheader("Linha de Contagem")
    line_enabled = st.checkbox("Ativar linha de contagem", value=False, key="line_enabled")
    if line_enabled:
        lc1, lc2 = st.columns(2)
        with lc1:
            st.slider("Linha X1 (%)", 0, 100, 10, key="line_x1")
            st.slider("Linha Y1 (%)", 0, 100, 50, key="line_y1")
        with lc2:
            st.slider("Linha X2 (%)", 0, 100, 90, key="line_x2")
            st.slider("Linha Y2 (%)", 0, 100, 50, key="line_y2")

    st.divider()
    st.button("📸 CAPTURE FRAME", on_click=capture_current_frame, use_container_width=True)
    if st.session_state.get("capture_message"):
        st.caption(st.session_state.capture_message)
    st.divider()
    groq_key = get_groq_api_key()
    st.caption(f"🧠 GROQ: {'✅ ENABLED' if groq_key else '❌ DISABLED'}")
    if groq_key:
        st.button("🔍 Descrever frame", on_click=run_groq_on_latest, use_container_width=True)
    if st.session_state.get("groq_message"):
        st.markdown(f"**GROQ:** {st.session_state.groq_message}")
    st.caption("🚀 ARQTECH: NOT TRAINED")
    if detector_error:
        st.warning(f"⚠️ YOLO indisponível: {detector_error}")
    if detector is not None and detector.last_tracking_error:
        st.warning(f"⚠️ Tracking indisponível: {detector.last_tracking_error}")

tab_image, tab_video, tab_camera, tab_metrics = st.tabs(["📷 IMAGEM", "🎬 VÍDEO", "📹 CÂMERA", "📊 ANALYTICS"])

# =============================================================================
# ABA IMAGEM (simplificada com gráficos e relatório)
# =============================================================================
with tab_image:
    uploaded_image = st.file_uploader(
        "Carregar imagem",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader",
    )
    if uploaded_image is not None:
        img = decode_image_bytes(uploaded_image.getvalue())
        if img is None:
            st.error("❌ Imagem inválida.")
        else:
            h, w = img.shape[:2]

            with st.sidebar:
                st.divider()
                st.subheader("🖼️ Visualização")
                show_grid = st.checkbox("Mostrar grade espacial", value=False, key="show_grid")

            if detector is not None:
                try:
                    with st.spinner("🔍 Executando YOLO..."):
                        detections = detector.detect(img, conf=conf_threshold)

                    # Anotação simplificada
                    annotated = draw_detections(
                        img,
                        detections,
                        draw_centers=draw_centers,
                        show_confidence_legend=show_legend,
                    )

                    if show_grid:
                        annotated = draw_grid(annotated)

                    st.session_state.latest_frame = annotated

                    col_original, col_result = st.columns(2)
                    with col_original:
                        st.markdown("**🖼️ Original**")
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="auto", use_container_width=True)
                    with col_result:
                        st.markdown("**🎯 YOLO**")
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="auto", use_container_width=True)

                    if detections:
                        # Gerar relatório uma única vez
                        report = generate_report(detections, w, h, conf_threshold)

                        # ============================================================
                        # RESUMO DA DETECÇÃO (melhorado com 8 métricas)
                        # ============================================================
                        st.divider()
                        st.subheader("📊 Resumo da Detecção")

                        confs = [d.confidence for d in detections]
                        avg_conf = np.mean(confs) if confs else 0.0
                        classes_set = set(d.label for d in detections)
                        class_counts = {}
                        for d in detections:
                            class_counts[d.label] = class_counts.get(d.label, 0) + 1
                        dominant_class = max(class_counts, key=class_counts.get) if class_counts else None

                        # Métricas do relatório
                        occupancy = report['executive_summary']['union_coverage'] * 100
                        density = report['executive_summary']['density']
                        high_conf_count = sum(1 for c in confs if c >= 0.7)

                        # Primeira linha
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Objetos", len(detections))
                        col2.metric("Classes", len(classes_set))
                        col3.metric("Confiança média", f"{avg_conf:.1%}")
                        if avg_conf >= 0.7:
                            quality = "🔵 Alta"
                        elif avg_conf >= 0.4:
                            quality = "🟡 Média"
                        else:
                            quality = "🔴 Baixa"
                        col4.metric("Qualidade", quality)

                        # Segunda linha
                        col5, col6, col7, col8 = st.columns(4)
                        col5.metric("Classe dominante", dominant_class or "N/A")
                        col6.metric("Ocupação", f"{occupancy:.1f}%")
                        col7.metric("Densidade", f"{density:.1f} obj/Mpx")
                        col8.metric("Alta confiança", f"{high_conf_count} ({high_conf_count/len(detections):.0%})")

                        # ============================================================
                        # TABELA DE DETECÇÕES
                        # ============================================================
                        st.subheader("📋 Detecções")
                        df_det = pd.DataFrame([
                            {
                                "ID": i+1,
                                "Classe": d.label.upper(),
                                "Confiança": f"{d.confidence:.0%}",
                                "Área (px²)": compute_box_metrics(d.x1, d.y1, d.x2, d.y2)["area"],
                            }
                            for i, d in enumerate(detections)
                        ])
                        st.dataframe(df_det, use_container_width=True, hide_index=True)

                        # ============================================================
                        # GRÁFICOS ANALÍTICOS (Plotly — hover, zoom, pan)
                        # ============================================================
                        with st.expander("📊 Ver gráficos analíticos (opcional)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(charts.confidence_histogram(confs), use_container_width=True)
                            with col2:
                                st.plotly_chart(charts.class_bar_chart(class_counts), use_container_width=True)

                        # ============================================================
                        # RELATÓRIO TÉCNICO COMPLETO (formatado, dentro de expansor)
                        # ============================================================
                        with st.expander("📄 Ver relatório técnico completo (opcional)"):
                            display_report_section(report)

                        # ============================================================
                        # EXPORTAÇÃO
                        # ============================================================
                        st.divider()
                        st.subheader("📤 Exportar")

                        export_col1, export_col2 = st.columns(2)
                        with export_col1:
                            csv_data = df_det.to_csv(index=False)
                            st.download_button(
                                label="📥 Baixar CSV",
                                data=csv_data,
                                file_name=f"detections_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        with export_col2:
                            json_data = export_report_json(report)
                            st.download_button(
                                label="📥 Baixar JSON",
                                data=json_data,
                                file_name=f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True,
                            )

                    else:
                        st.info("🔍 Nenhum objeto detectado.")

                except Exception as exc:
                    st.error(f"❌ Erro: {exc}")
            else:
                st.warning("⚠️ Detector YOLO indisponível.")

# =============================================================================
# ABA VÍDEO (mantida, com a mesma lógica simplificada)
# =============================================================================
with tab_video:
    uploaded_video = st.file_uploader(
        "Carregar vídeo",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )
    if uploaded_video is not None:
        token = (uploaded_video.name, uploaded_video.size)
        if st.session_state.video_upload_token != token or st.session_state.video_meta is None:
            try:
                old_path = st.session_state.video_path
                if old_path and Path(old_path).exists():
                    Path(old_path).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                saved_path = save_uploaded_file(uploaded_video, ALLOWED_VIDEO_EXTENSIONS, target_dir="outputs/uploads")
                meta = get_video_metadata(saved_path)
                st.session_state.video_path = str(saved_path)
                st.session_state.video_meta = meta
                st.session_state.video_upload_token = token
                st.session_state.video_status = "idle"
                st.session_state.video_frame_index = 0
                st.session_state.video_error = None
                st.session_state.metrics.reset()
                st.session_state.temporal_analyzer = TemporalAnalyzer(max_history=300)
            except Exception as exc:
                st.error(f"❌ Erro ao carregar: {exc}")

        meta = st.session_state.video_meta
        if meta is None:
            st.info("📥 Carregando...")
        elif not meta.ok:
            st.error(meta.error)
        else:
            st.caption(f"**Arquivo:** {uploaded_video.name}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Duração", f"{meta.duration_sec:.2f}s" if meta.duration_sec else "--")
            c2.metric("Resolução", f"{meta.width}x{meta.height}")
            c3.metric("FPS", f"{meta.fps:.1f}" if meta.fps else "--")
            c4.metric("Frames", meta.frame_count if meta.frame_count else "--")
            c5.metric("Tamanho", f"{meta.file_size_bytes / (1024*1024):.2f} MB")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("▶ INICIAR", use_container_width=True):
                    start_video()
            with col2:
                if st.button("⏸ PAUSAR", use_container_width=True):
                    pause_video()
            with col3:
                if st.button("▶ RETOMAR", use_container_width=True):
                    resume_video()
            with col4:
                if st.button("⏹ PARAR", use_container_width=True):
                    stop_video()

            if st.session_state.video_error:
                st.error(st.session_state.video_error)

            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            temporal_placeholder = st.empty()
            status_placeholder = st.empty()

            if st.session_state.video_status in ("running", "paused"):
                processor = st.session_state.processor
                if processor:
                    status = processor.get_status()
                    latest = processor.get_latest_result()
                    if latest:
                        detections = latest.get("detections", [])
                        tracking_metrics = latest.get("tracking_metrics", {})
                        motion_metrics = latest.get("motion_metrics", {})
                        roi_snapshot = latest.get("roi_snapshot")
                        line_crossings = latest.get("line_crossings") or []
                        new_events = latest.get("new_events") or []
                        meta = st.session_state.video_meta
                        time_sec = latest["frame_index"] / (meta.fps if meta and meta.fps else 30.0)

                        st.session_state.temporal_analyzer.push_frame(
                            latest["frame_index"],
                            detections,
                            tracking_metrics
                        )
                        st.session_state.temporal_analyzer.record_detections(
                            latest["frame_index"], time_sec, detections, motion_metrics
                        )
                        if roi_snapshot is not None:
                            st.session_state.temporal_analyzer.record_roi_snapshot(
                                latest["frame_index"], time_sec, roi_snapshot
                            )
                        if line_crossings:
                            st.session_state.temporal_analyzer.record_line_crossings(line_crossings)
                        if new_events:
                            st.session_state.temporal_analyzer.record_events(new_events)

                        track_history = {}
                        for tid, data in tracking_metrics.items():
                            if "history" in data:
                                track_history[tid] = data["history"]

                        annotated = draw_detections(
                            latest["frame"],
                            detections,
                            draw_centers=draw_centers,
                            draw_tracks=show_trajectory,
                            track_history=track_history if show_trajectory else None,
                            show_confidence_legend=show_legend,
                        )

                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="auto", use_container_width=True)

                        m = latest["metrics"]

                        with metrics_placeholder.container():
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("Frame", latest["frame_index"])
                            c2.metric("Objetos", m["object_count"])
                            c3.metric("Confiança", f"{m['confidence']:.0%}")
                            c4.metric("Status", st.session_state.video_status)
                            c5.metric("Tracking", f"{len(tracking_metrics)} ativos")

                            if m.get("classes"):
                                st.caption(" | ".join(f"{k} {v}" for k, v in m["classes"].items()))

                            if motion_metrics or roi_snapshot is not None or st.session_state.get("line_enabled"):
                                status_counts = {}
                                for mm in motion_metrics.values():
                                    status_counts[mm["status"]] = status_counts.get(mm["status"], 0) + 1
                                ic1, ic2, ic3, ic4 = st.columns(4)
                                ic1.metric("Parados", status_counts.get("PARADO", 0))
                                ic2.metric("Movendo", status_counts.get("MOVENDO", 0))
                                ic3.metric("Rápidos", status_counts.get("MOVIMENTO_RAPIDO", 0))
                                if roi_snapshot is not None:
                                    ic4.metric("Ocupação ROI", roi_snapshot["occupancy"])
                                if st.session_state.processor and st.session_state.processor.line_detector:
                                    counts = st.session_state.processor.line_detector.crossing_counts
                                    st.caption(f"Linha: {counts['forward']} A→B · {counts['backward']} B→A")

                            if tracking_metrics:
                                with st.expander("📊 Detalhes do Tracking", expanded=False):
                                    for tid, data in tracking_metrics.items():
                                        mm = motion_metrics.get(tid, {})
                                        extra = f" | {mm.get('status', '')} ({mm.get('direction', data['direction'])})" if mm else ""
                                        st.write(f"**ID {tid}**: {data['direction']} | Deslocamento: {data['displacement_px']:.1f} px/frame{extra}")
                                        st.caption(f"ΔX: {data['dx']:.1f}  ΔY: {data['dy']:.1f}")

                            if new_events:
                                with st.expander(f"🧭 Eventos recentes ({len(new_events)} neste frame)", expanded=False):
                                    for ev in new_events[-10:]:
                                        st.caption(f"{ev['event_type']} — ID {ev['object_id']} ({ev.get('class_name') or '—'})")


                        with temporal_placeholder.container():
                            temporal_stats = st.session_state.temporal_analyzer.get_stats()
                            if temporal_stats["total_frames"] > 0:
                                st.subheader("⏱️ Análise Temporal")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Frames processados", temporal_stats["total_frames"])
                                col2.metric("Média objetos/frame", f"{temporal_stats['avg_objects_per_frame']:.1f}")
                                col3.metric("Média tracks ativos", f"{temporal_stats['avg_tracks']:.1f}")
                                col4.metric("Frames com tracking", temporal_stats["frames_with_tracking"])

                                hist = st.session_state.temporal_analyzer.get_object_count_history()
                                if len(hist) > 1:
                                    st.caption("Evolução do número de objetos por frame")
                                    st.line_chart(hist)

                    status_placeholder.caption(
                        f"Fila frames: {status.get('queue_size', 0)} | "
                        f"Fila resultados: {status.get('result_queue_size', 0)} | "
                        f"Erro: {status.get('error') or 'Nenhum'}"
                    )
                    time.sleep(0.05)
                    st.rerun()
                else:
                    st.info("⏳ Processador não inicializado.")
            elif st.session_state.video_status == "stopped":
                st.info("⏹ Vídeo parado.")
            elif st.session_state.video_status == "finished":
                st.success("✅ Processamento concluído.")
            else:
                st.info("⏳ Aguardando início.")

# =============================================================================
# ABA CÂMERA (mantida)
# =============================================================================
with tab_camera:
    b1, b2 = st.columns(2)
    b1.button("📸 START CAMERA", on_click=start_camera, use_container_width=True)
    b2.button("⏹ STOP CAMERA", on_click=stop_camera, use_container_width=True)
    if st.session_state.camera_error:
        st.error(st.session_state.camera_error)
    if st.session_state.camera_status == "running":
        run_camera_loop(conf_threshold, sample_every, tracking_enabled, show_details=False, draw_centers=draw_centers, show_legend=show_legend)
    elif st.session_state.camera_status == "stopped":
        st.info("⏹ Câmera parada.")
    else:
        st.info("📹 Câmera inativa.")

# =============================================================================
# ABA ANALYTICS (reconstruída em Plotly: Overview, Objects, Motion,
# Trajectories, Heatmap, Events)
# =============================================================================
with tab_metrics:
    st.subheader("📊 Métricas da Sessão")
    metrics = st.session_state.metrics
    if metrics.frames_analyzed == 0 and metrics.last_frame_index is None:
        st.info("📭 Nenhuma análise executada.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames analisados", metrics.frames_analyzed)
        c2.metric("Total de objetos", metrics.total_objects)
        c3.metric("Confiança média", f"{metrics.avg_confidence:.1%}" if metrics.conf_count else "--")
        c4.metric("FPS processamento", f"{metrics.processing_fps:.1f}" if metrics.processing_fps else "--")
        if metrics.class_counts:
            st.caption("Objetos por classe (acumulado):")
            st.caption(" | ".join(f"{k} {v}" for k, v in metrics.class_counts.items()))

    st.divider()
    st.subheader("Analytics")

    ta = st.session_state.temporal_analyzer
    records = ta.get_detection_dataframe_records() if ta else []

    temporal_stats = ta.get_stats() if ta else {}
    if temporal_stats and temporal_stats["total_frames"] > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames processados", temporal_stats["total_frames"])
        c2.metric("Média objetos/frame", f"{temporal_stats['avg_objects_per_frame']:.1f}")
        c3.metric("Máx. objetos", temporal_stats["max_objects"])
        c4.metric("Mín. objetos", temporal_stats["min_objects"])

    if not records:
        st.info("Sem dados de detecção nesta sessão. Rode Vídeo ou Câmera (com Tracking ativo) para popular o Analytics.")
    else:
        df = pd.DataFrame(records)

        section = st.radio(
            "Seção",
            ["Overview", "Objects", "Motion", "Trajectories", "Heatmap", "Events"],
            horizontal=True,
            key="analytics_section",
        )

        video_meta = st.session_state.get("video_meta")
        frame_width = video_meta.width if video_meta else None
        frame_height = video_meta.height if video_meta else None

        if section == "Overview":
            by_frame = df.groupby("frame_index")
            counts = by_frame.size()
            times = by_frame["time_sec"].first()
            st.plotly_chart(
                charts.object_activity_overview(times.values, counts.values),
                use_container_width=True,
            )
            st.plotly_chart(
                charts.confidence_over_time(df["time_sec"], df["confidence"], df["class_name"]),
                use_container_width=True,
            )

        elif section == "Objects":
            obj_summary = df.dropna(subset=["track_id"]).groupby("track_id").agg(
                classe=("class_name", "first"),
                confianca_media=("confidence", "mean"),
                status_atual=("status", "last"),
                direcao_atual=("direction", "last"),
                velocidade_media=("speed", "mean"),
                frames_em_cena=("frame_index", "nunique"),
            ).reset_index()
            if obj_summary.empty:
                st.info("Nenhum objeto com ID de tracking nesta sessão.")
            else:
                obj_summary["track_id"] = obj_summary["track_id"].astype(int)
                st.dataframe(
                    obj_summary.style.format({
                        "confianca_media": "{:.0%}",
                        "velocidade_media": "{:.1f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
                ids = sorted(df["track_id"].dropna().astype(int).unique().tolist())
                selected_id = st.selectbox("Selecionar objeto (ID)", options=["Todos"] + ids, key="analytics_object_id")
                filtered = df if selected_id == "Todos" else df[df["track_id"] == selected_id]
                st.plotly_chart(
                    charts.confidence_over_time(filtered["time_sec"], filtered["confidence"], filtered["class_name"]),
                    use_container_width=True,
                )

        elif section == "Motion":
            motion_df = df.dropna(subset=["speed"])
            if motion_df.empty:
                st.info("Sem dados de Motion Intelligence ainda (requer Tracking ativo).")
            else:
                agg = motion_df.groupby("frame_index").agg(
                    time_sec=("time_sec", "first"),
                    avg_speed=("speed", "mean"),
                    moving=("status", lambda s: (s != "PARADO").sum()),
                ).reset_index()
                st.plotly_chart(
                    charts.motion_intensity(agg["time_sec"], agg["avg_speed"], agg["moving"]),
                    use_container_width=True,
                )
                st.plotly_chart(charts.speed_distribution(motion_df["speed"]), use_container_width=True)

        elif section == "Trajectories":
            trajectories = {tid: ta.get_trajectory(tid) for tid in ta.trajectory_points.keys()}
            if not trajectories:
                st.info("Nenhuma trajetória registrada ainda.")
            else:
                ids = sorted(trajectories.keys())
                selected = st.multiselect(
                    "Objetos", options=ids, default=ids[: min(5, len(ids))], key="analytics_trajectory_ids"
                )
                subset = {tid: trajectories[tid] for tid in selected} if selected else trajectories
                st.plotly_chart(
                    charts.trajectory_plot(subset, image_width=frame_width, image_height=frame_height),
                    use_container_width=True,
                )

        elif section == "Heatmap":
            points = ta.get_heatmap_points()
            if len(points) < 20:
                st.info("Dados insuficientes para o Movement Heatmap ainda — poucos pontos de movimento registrados.")
            else:
                st.plotly_chart(
                    charts.movement_heatmap(points, image_width=frame_width, image_height=frame_height),
                    use_container_width=True,
                )

        elif section == "Events":
            if not ta.event_log:
                st.info("Nenhum evento registrado ainda (ROI, Line Crossing e Motion emitem eventos).")
            else:
                st.plotly_chart(charts.event_timeline(ta.event_log), use_container_width=True)

                if ta.roi_history:
                    st.plotly_chart(charts.roi_occupancy_chart(ta.roi_history), use_container_width=True)
                    peak = max(r["occupancy"] for r in ta.roi_history)
                    avg = sum(r["occupancy"] for r in ta.roi_history) / len(ta.roi_history)
                    c1, c2 = st.columns(2)
                    c1.metric("Pico de ocupação (ROI)", peak)
                    c2.metric("Ocupação média (ROI)", f"{avg:.1f}")

                if ta.line_crossing_log:
                    st.plotly_chart(charts.line_crossing_chart(ta.line_crossing_log), use_container_width=True)
                elif st.session_state.get("line_enabled"):
                    st.caption("Nenhum cruzamento de linha detectado ainda.")

    st.divider()
    st.subheader("🚀 ARQTECH")
    st.write("**STATUS:** NOT TRAINED")
    st.caption("ARQTECH será o modelo experimental futuro.")
    st.caption("YOLO: EXTERNAL BASELINE")
    st.caption(f"🧠 GROQ: {'ENABLED' if get_groq_api_key() else 'DISABLED'}")