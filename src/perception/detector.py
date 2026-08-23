"""
VISION MINI LAB
YOLO detector wrapper + advanced detection pipeline.

Includes:
- Multi-scale detection
- Tiling with overlap
- Deduplication (class-aware, IoU + containment)
- Adaptive confidence filtering
- Small object handling
- Light low-light enhancement
- Full pipeline statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# =============================================================================
# VISION COLOR PALETTE (shared across modules)
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

# Stable colors for track IDs (cyclic palette of 20 distinct colors)
ID_COLORS = [
    (90, 180, 255),    # 0: blue
    (52, 211, 153),    # 1: emerald
    (245, 158, 11),    # 2: amber
    (167, 139, 250),   # 3: violet
    (244, 114, 182),   # 4: pink
    (56, 189, 248),    # 5: sky
    (251, 113, 133),   # 6: rose
    (163, 230, 53),    # 7: lime
    (232, 121, 249),   # 8: fuchsia
    (45, 212, 191),    # 9: teal
    (250, 204, 21),    # 10: yellow
    (129, 140, 248),   # 11: indigo
    (251, 146, 60),    # 12: orange
    (192, 132, 252),   # 13: purple
    (94, 234, 212),    # 14: cyan
    (217, 70, 239),    # 15: magenta
    (132, 204, 22),    # 16: green
    (239, 68, 68),     # 17: red
    (148, 163, 184),   # 18: slate
    (203, 213, 225),   # 19: gray
]


def get_id_color(track_id: int) -> Tuple[int, int, int]:
    """Return a stable RGB color for a given track_id."""
    if track_id <= 0:
        return (160, 160, 170)  # gray for detection-only
    return ID_COLORS[(track_id - 1) % len(ID_COLORS)]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    small_object: bool = False
    source: str = "pass1"
    track_id: Optional[int] = None


@dataclass
class DetectionConfig:
    conf: float = 0.30
    iou: float = 0.45
    max_det: int = 50
    imgsz: int = 960
    augment: bool = False
    smart_second_pass: bool = True
    tile_overlap: float = 0.15
    small_object_threshold: float = 0.02
    tracking_conf: float = 0.40
    enable_adaptive_conf: bool = True
    enhance_low_light: bool = True
    max_tiles: int = 4
    temporal_consistency_frames: int = 3
    temporal_consistency_boost: float = 0.05


@dataclass
class PipelineStats:
    """Statistics collected at each pipeline stage."""
    raw: int = 0
    after_conf_filter: int = 0
    after_nms: int = 0
    tile_merge: int = 0
    small_objects: int = 0
    final: int = 0
    tracked: int = 0
    inference_ms: float = 0.0


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_name: str = "yolo11s.pt") -> Optional[YOLO]:
    """Load YOLO model with error handling."""
    try:
        logger.info("Loading YOLO model: %s", model_name)
        model = YOLO(model_name)
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as exc:
        logger.exception("YOLO model loading failed: %s", exc)
        return None


# =============================================================================
# PREPROCESSING
# =============================================================================

def estimate_brightness_contrast(image_rgb: np.ndarray) -> Tuple[float, float]:
    """Returns (brightness, contrast) from grayscale image."""
    if image_rgb is None or image_rgb.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(gray.mean()), float(gray.std())


def preprocess_image(image_rgb: np.ndarray, enhance: bool = False) -> np.ndarray:
    """
    Light preprocessing. Optionally enhances low-light or low-contrast images.
    """
    if image_rgb is None or image_rgb.size == 0:
        return image_rgb

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if not enhance:
        return image_rgb

    brightness, contrast = estimate_brightness_contrast(image_rgb)

    if brightness < 60 or contrast < 35:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        lab_eq = cv2.merge((l_eq, a_channel, b_channel))
        image_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return image_rgb


# =============================================================================
# YOLO INFERENCE HELPERS
# =============================================================================

def run_yolo_detection(
    model: YOLO,
    image_rgb: np.ndarray,
    config: DetectionConfig,
    use_tracking: bool = False,
    track_persist: bool = True,
    verbose: bool = False,
) -> List[Detection]:
    """Execute YOLO (track or predict) and extract detections."""
    detections: List[Detection] = []

    if model is None or image_rgb is None or image_rgb.size == 0:
        return detections

    try:
        if use_tracking:
            results = model.track(
                source=image_rgb,
                conf=config.conf,
                iou=config.iou,
                max_det=config.max_det,
                imgsz=config.imgsz,
                persist=track_persist,
                augment=config.augment,
                verbose=verbose,
            )
        else:
            results = model.predict(
                source=image_rgb,
                conf=config.conf,
                iou=config.iou,
                max_det=config.max_det,
                imgsz=config.imgsz,
                augment=config.augment,
                verbose=verbose,
            )
    except Exception as exc:
        logger.exception("YOLO inference failed: %s", exc)
        return detections

    if not results:
        return detections

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes
    names = result.names or {}

    ids = None
    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)

    for i in range(len(boxes)):
        try:
            cls_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            track_id = None
            if ids is not None:
                track_id = int(ids[i])

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=names.get(cls_id, f"class_{cls_id}"),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    track_id=track_id,
                    source="pass1",
                )
            )
        except Exception:
            logger.exception("Failed to parse detection %d", i)
            continue

    return detections


# =============================================================================
# TILING
# =============================================================================

def tile_image(
    image_rgb: np.ndarray,
    tile_size: int = 2,
    overlap: float = 0.15,
    max_tiles: int = 4,
) -> List[Dict[str, Any]]:
    """Divide image into tiles with overlap."""
    if image_rgb is None or image_rgb.size == 0:
        return []

    h, w = image_rgb.shape[:2]

    if tile_size * tile_size > max_tiles:
        tile_size = int(np.sqrt(max_tiles))
        if tile_size < 1:
            tile_size = 1

    tile_h = h // tile_size
    tile_w = w // tile_size

    if tile_h < 32 or tile_w < 32:
        # Image too small for meaningful tiling
        return []

    overlap_h = int(tile_h * overlap)
    overlap_w = int(tile_w * overlap)

    tiles = []

    for i in range(tile_size):
        for j in range(tile_size):
            x_start = max(0, j * tile_w - (overlap_w if j > 0 else 0))
            x_end = min(w, (j + 1) * tile_w + (overlap_w if j < tile_size - 1 else 0))
            y_start = max(0, i * tile_h - (overlap_h if i > 0 else 0))
            y_end = min(h, (i + 1) * tile_h + (overlap_h if i < tile_size - 1 else 0))

            tile = image_rgb[y_start:y_end, x_start:x_end]
            if tile.size == 0:
                continue

            tiles.append({
                "tile": tile,
                "x_offset": x_start,
                "y_offset": y_start,
                "original_shape": (h, w),
            })

    return tiles


def detect_tiles(
    model: YOLO,
    image_rgb: np.ndarray,
    config: DetectionConfig,
) -> List[Detection]:
    """Run detection on image tiles and return detections in global coordinates."""
    all_detections: List[Detection] = []

    if model is None or image_rgb is None or image_rgb.size == 0:
        return all_detections

    tiles = tile_image(
        image_rgb,
        tile_size=2,
        overlap=config.tile_overlap,
        max_tiles=config.max_tiles,
    )

    for tile_info in tiles:
        try:
            tile_dets = run_yolo_detection(
                model,
                tile_info["tile"],
                config,
                use_tracking=False,  # No tracking on tiles
            )

            for det in tile_dets:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (
                    x1 + tile_info["x_offset"],
                    y1 + tile_info["y_offset"],
                    x2 + tile_info["x_offset"],
                    y2 + tile_info["y_offset"],
                )
                det.center = (
                    (det.bbox[0] + det.bbox[2]) / 2.0,
                    (det.bbox[1] + det.bbox[3]) / 2.0,
                )
                det.source = "tile"
                all_detections.append(det)
        except Exception as exc:
            logger.exception("Tile detection failed: %s", exc)
            continue

    return all_detections


# =============================================================================
# IOU / MERGING / DEDUPLICATION
# =============================================================================

def iou_bbox(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Intersection over Union."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _intersection_area(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def _box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def merge_detections(
    detections: List[Detection],
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.70,
) -> List[Detection]:
    """
    Class-aware deduplication.
    Same class + (IoU > threshold OR containment > threshold) = duplicate.
    Keeps the one with higher confidence.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []

    for det in sorted_dets:
        duplicate = False

        for existing in kept:
            if existing.class_id != det.class_id:
                continue

            iou = iou_bbox(existing.bbox, det.bbox)
            inter_area = _intersection_area(existing.bbox, det.bbox)
            area_a = _box_area(existing.bbox)
            area_b = _box_area(det.bbox)
            min_area = min(area_a, area_b)
            containment = inter_area / min_area if min_area > 0 else 0.0

            # Center distance normalized by average box size
            ca = existing.center
            cb = det.center
            dx = ca[0] - cb[0]
            dy = ca[1] - cb[1]
            avg_w = ((existing.bbox[2] - existing.bbox[0]) + (det.bbox[2] - det.bbox[0])) / 2.0
            avg_h = ((existing.bbox[3] - existing.bbox[1]) + (det.bbox[3] - det.bbox[1])) / 2.0
            if avg_w > 0 and avg_h > 0:
                norm_dist = (abs(dx) / avg_w) + (abs(dy) / avg_h)
            else:
                norm_dist = float("inf")

            if (
                iou > iou_threshold
                or containment > containment_threshold
                or (iou > 0.30 and norm_dist < 0.5)
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


# =============================================================================
# SMALL OBJECTS
# =============================================================================

def analyze_small_objects(
    detections: List[Detection],
    img_shape: Tuple[int, ...],
    small_object_threshold: float = 0.02,
) -> int:
    """Mark small detections. Returns count of small objects."""
    if not detections or not img_shape:
        return 0

    h, w = img_shape[:2]
    image_area = float(h * w)
    count = 0

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        relative_area = bbox_area / image_area if image_area > 0 else 0.0
        det.small_object = relative_area < small_object_threshold
        if det.small_object:
            count += 1

    return count


# =============================================================================
# ADAPTIVE CONFIDENCE FILTERING
# =============================================================================

def adaptive_confidence_filter(
    detections: List[Detection],
    base_conf: float,
    small_object_threshold: float = 0.02,
) -> List[Detection]:
    """
    Adaptive confidence filter:
    - conf >= base_conf: accepted
    - base_conf * 0.7 <= conf < base_conf: accepted if small_object
    - conf < base_conf * 0.7: rejected
    """
    filtered = []
    lower_bound = base_conf * 0.7

    for det in detections:
        conf = det.confidence
        if conf >= base_conf:
            filtered.append(det)
        elif conf >= lower_bound and det.small_object:
            filtered.append(det)
        # below lower_bound: discarded

    return filtered


# =============================================================================
# COVERAGE ANALYSIS
# =============================================================================

def compute_coverage(detections: List[Detection], img_shape: Tuple[int, ...]) -> float:
    """Compute fraction of image area covered by detection bounding boxes."""
    if not detections or not img_shape:
        return 0.0

    h, w = img_shape[:2]
    image_area = float(h * w)
    if image_area <= 0:
        return 0.0

    total_bbox_area = 0.0
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        total_bbox_area += area

    return min(1.0, total_bbox_area / image_area)


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_full_pipeline(
    model: YOLO,
    image_rgb: np.ndarray,
    config: DetectionConfig,
    use_tracking: bool = True,
    temporal_buffer: Optional[Dict[int, List[float]]] = None,
) -> Tuple[List[Detection], PipelineStats]:
    """
    Complete detection pipeline:
    1) Preprocessing
    2) Pass 1 (full image, with tracking if enabled)
    3) Analyze small objects
    4) Smart second pass (tiling) if needed
    5) Merge detections from all sources
    6) Re-analyze small objects
    7) Adaptive confidence filter
    8) Temporal consistency (video only)
    9) Limit to max_det

    Returns (detections, stats).
    """
    stats = PipelineStats()

    if image_rgb is None or image_rgb.size == 0:
        return [], stats

    processed = preprocess_image(image_rgb, enhance=config.enhance_low_light)
    img_shape = processed.shape

    # -------------------------------------------------------------------------
    # Pass 1: Full image
    # -------------------------------------------------------------------------
    detections = run_yolo_detection(
        model,
        processed,
        config,
        use_tracking=use_tracking,
    )
    stats.raw = len(detections)

    # -------------------------------------------------------------------------
    # Confidence filter (pass 1)
    # -------------------------------------------------------------------------
    if config.enable_adaptive_conf:
        detections = adaptive_confidence_filter(
            detections,
            config.conf,
            config.small_object_threshold,
        )
    else:
        detections = [d for d in detections if d.confidence >= config.conf]
    stats.after_conf_filter = len(detections)

    # -------------------------------------------------------------------------
    # Small object analysis
    # -------------------------------------------------------------------------
    small_count = analyze_small_objects(detections, img_shape, config.small_object_threshold)

    # -------------------------------------------------------------------------
    # Smart second pass decision
    # -------------------------------------------------------------------------
    coverage = compute_coverage(detections, img_shape)
    small_ratio = small_count / max(len(detections), 1)

    need_second_pass = (
        config.smart_second_pass
        and (
            len(detections) < 3
            or coverage < 0.10
            or small_ratio > 0.5
        )
    )

    tile_dets: List[Detection] = []
    if need_second_pass:
        try:
            tile_dets = detect_tiles(model, processed, config)
            # Filter tile detections
            if config.enable_adaptive_conf:
                tile_dets = adaptive_confidence_filter(
                    tile_dets,
                    config.conf,
                    config.small_object_threshold,
                )
            else:
                tile_dets = [d for d in tile_dets if d.confidence >= config.conf]
        except Exception as exc:
            logger.exception("Smart second pass failed: %s", exc)
            tile_dets = []

    # -------------------------------------------------------------------------
    # Merge full + tile detections
    # -------------------------------------------------------------------------
    all_dets = detections + tile_dets
    stats.tile_merge = len(all_dets)

    merged = merge_detections(
        all_dets,
        iou_threshold=config.iou,
        containment_threshold=0.70,
    )
    stats.after_nms = len(merged)

    # -------------------------------------------------------------------------
    # Re-analyze small objects
    # -------------------------------------------------------------------------
    stats.small_objects = analyze_small_objects(
        merged, img_shape, config.small_object_threshold
    )

    # -------------------------------------------------------------------------
    # Final adaptive confidence filter
    # -------------------------------------------------------------------------
    if config.enable_adaptive_conf:
        merged = adaptive_confidence_filter(
            merged,
            config.conf,
            config.small_object_threshold,
        )
    else:
        merged = [d for d in merged if d.confidence >= config.conf]

    # -------------------------------------------------------------------------
    # Temporal consistency (video only — requires temporal_buffer)
    # -------------------------------------------------------------------------
    if temporal_buffer is not None and config.temporal_consistency_frames > 0:
        promoted: List[Detection] = []
        for det in merged:
            if det.confidence >= config.conf:
                promoted.append(det)
                continue

            # Track temporal consistency by a pseudo-key (class_id + rough position)
            # For tracked objects, use track_id; for detections, use position hash
            if det.track_id is not None and det.track_id > 0:
                key = det.track_id
            else:
                # Hash based on class and quantized position
                key = f"{det.class_id}_{int(det.center[0] // 20)}_{int(det.center[1] // 20)}"

            history = temporal_buffer.get(key, [])
            history.append(det.confidence)
            if len(history) > config.temporal_consistency_frames:
                history = history[-config.temporal_consistency_frames:]
            temporal_buffer[key] = history

            # Promote if consistently detected across frames
            if len(history) >= config.temporal_consistency_frames:
                avg_conf = sum(history) / len(history)
                if avg_conf >= config.conf - config.temporal_consistency_boost:
                    det.confidence = min(0.99, avg_conf + config.temporal_consistency_boost)
                    promoted.append(det)
            else:
                promoted.append(det)  # keep it for now, may be promoted later
        merged = promoted

    # -------------------------------------------------------------------------
    # Sort by confidence and limit
    # -------------------------------------------------------------------------
    merged.sort(key=lambda d: d.confidence, reverse=True)
    merged = merged[: config.max_det]
    stats.final = len(merged)

    return merged, stats
