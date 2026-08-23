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
    conf: float = 0.25
    iou: float = 0.50
    max_det: int = 50
    imgsz: int = 960
    augment: bool = False
    smart_second_pass: bool = True
    tile_overlap: float = 0.20
    small_object_threshold: float = 0.02
    tracking_conf: float = 0.40
    enable_adaptive_conf: bool = True
    enhance_low_light: bool = True
    max_tiles: int = 4


# =============================================================================
# DETECTOR WRAPPER
# =============================================================================

class Detector:
    """Wrapper around Ultralytics YOLO for detection with advanced features."""

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        max_det: int = 50,
        imgsz: int = 960,
        augment: bool = False,
    ):
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.imgsz = imgsz
        self.augment = augment
        self._model: Optional[YOLO] = None

    def load(self) -> bool:
        """Load model. Returns True on success."""
        try:
            self._model = YOLO(self.model_name)
            logger.info("YOLO model loaded: %s", self.model_name)
            return True
        except Exception as exc:
            logger.exception("Failed to load YOLO model: %s", exc)
            self._model = None
            return False

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def set_thresholds(self, conf: float, iou: float, max_det: Optional[int] = None) -> None:
        self.conf = float(np.clip(conf, 0.05, 0.95))
        self.iou = float(np.clip(iou, 0.1, 0.9))
        if max_det is not None:
            self.max_det = int(np.clip(max_det, 1, 100))

    def detect(self, image_rgb: np.ndarray) -> List[Detection]:
        """
        Run detection on RGB image (single pass).
        Returns list of Detection objects.
        """
        if not self.is_ready:
            return []

        try:
            results = self._model.predict(
                source=image_rgb,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
                augment=self.augment,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("Detection failed: %s", exc)
            return []

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        names = result.names or {}
        boxes = result.boxes

        for i in range(len(boxes)):
            try:
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(float, xyxy)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                name = names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                    )
                )
            except Exception:
                continue

        return detections


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
# TILING
# =============================================================================

def tile_image(
    image_rgb: np.ndarray,
    tile_size: int = 2,
    overlap: float = 0.20,
    max_tiles: int = 4,
) -> List[Dict[str, Any]]:
    """Divide a imagem em tiles com overlap."""
    if image_rgb is None or image_rgb.size == 0:
        return []

    h, w = image_rgb.shape[:2]

    if tile_size * tile_size > max_tiles:
        tile_size = int(np.sqrt(max_tiles))

    tile_h = h // tile_size
    tile_w = w // tile_size

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

            tiles.append(
                {
                    "tile": tile,
                    "x_offset": x_start,
                    "y_offset": y_start,
                    "original_shape": (h, w),
                }
            )

    return tiles


# =============================================================================
# IOU / MERGING
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
    iou_threshold: float = 0.40,
    containment_threshold: float = 0.70,
) -> List[Detection]:
    """
    Deduplicação class-aware aprimorada.
    - Mesma classe
    - IoU > iou_threshold OR
      containment > containment_threshold (caixa quase dentro da outra)
    - Considera também proximidade entre centros
    Mantém a de maior confiança.
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

            # 1) IoU tradicional
            iou = iou_bbox(existing.bbox, det.bbox)

            # 2) Contenção (interseção / área da menor caixa)
            inter_area = _intersection_area(existing.bbox, det.bbox)
            area_a = _box_area(existing.bbox)
            area_b = _box_area(det.bbox)
            min_area = min(area_a, area_b)
            containment = inter_area / min_area if min_area > 0 else 0.0

            # 3) Distância entre centros relativa ao tamanho médio
            ca = existing.center
            cb = det.center
            dx = ca[0] - cb[0]
            dy = ca[1] - cb[1]
            avg_w = ((existing.bbox[2] - existing.bbox[0]) + (det.bbox[2] - det.bbox[0])) / 2.0
            avg_h = ((existing.bbox[3] - existing.bbox[1]) + (det.bbox[3] - det.bbox[1])) / 2.0
            if avg_w == 0 or avg_h == 0:
                continue
            norm_dist = (abs(dx) / avg_w) + (abs(dy) / avg_h)

            # Considera duplicata se:
            # - IoU alto
            # - Ou uma caixa está quase contida na outra (containment alto)
            # - Ou IoU moderado E centros muito próximos
            if (
                iou > iou_threshold
                or containment > containment_threshold
                or (iou > 0.3 and norm_dist < 0.5)
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


# =============================================================================
# SMALL OBJECTS / ADAPTIVE CONFIDENCE
# =============================================================================

def analyze_small_objects(
    detections: List[Detection],
    img_shape: Tuple[int, int, int],
    small_object_threshold: float = 0.02,
) -> None:
    """Marca detecções pequenas."""
    h, w = img_shape[:2]
    image_area = float(h * w)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        relative_area = bbox_area / image_area if image_area > 0 else 0.0
        det.small_object = relative_area < small_object_threshold


def adaptive_confidence_filter(
    detections: List[Detection],
    base_conf: float,
    small_object_threshold: float = 0.02,
) -> List[Detection]:
    """
    Filtro adaptativo:
    - conf >= 0.60 : aceito
    - 0.30 <= conf < 0.60 : aceito
    - 0.20 <= conf < 0.30 : aceito somente se small_object
    - conf < 0.20 : rejeitado
    """
    filtered = []
    for det in detections:
        conf = det.confidence
        if conf >= 0.60:
            filtered.append(det)
        elif conf >= 0.30:
            filtered.append(det)
        elif conf >= 0.20 and det.small_object:
            filtered.append(det)
        # below 0.20 discarded
    return filtered


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
    """Executa YOLO (track ou predict) e extrai detecções."""
    detections: List[Detection] = []

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
    if result.boxes is None:
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
            logger.exception("Failed to parse detection")

    return detections


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_multi_scale_detection(
    model: YOLO,
    image_rgb: np.ndarray,
    config: DetectionConfig,
    use_tracking: bool = True,
) -> List[Detection]:
    """
    Pipeline completo:
    1) Pré-processamento opcional
    2) Passada 1 (imagem inteira, tracking ou predict)
    3) Análise de objetos pequenos
    4) Smart second pass (tiling) se necessário
    5) Merge das detecções
    6) Análise de objetos pequenos novamente
    7) Filtro adaptativo de confiança
    8) Limite de max_det
    """
    if image_rgb is None or image_rgb.size == 0:
        return []

    processed = preprocess_image(image_rgb, enhance=config.enhance_low_light)
    img_shape = processed.shape

    detections = run_yolo_detection(
        model,
        processed,
        config,
        use_tracking=use_tracking,
    )

    analyze_small_objects(detections, img_shape, config.small_object_threshold)

    # Smart second pass
    if config.smart_second_pass and len(detections) < 3:
        try:
            tiles = tile_image(
                processed,
                tile_size=2,
                overlap=config.tile_overlap,
                max_tiles=config.max_tiles,
            )

            for tile_info in tiles:
                tile_dets = run_yolo_detection(
                    model,
                    tile_info["tile"],
                    config,
                    use_tracking=False,
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

                detections.extend(tile_dets)

        except Exception as exc:
            logger.exception("Smart second pass failed: %s", exc)

    # Deduplicação aprimorada
    detections = merge_detections(
        detections,
        iou_threshold=0.40,
        containment_threshold=0.70,
    )

    # Reanálise de pequenos objetos
    analyze_small_objects(detections, img_shape, config.small_object_threshold)

    # Filtro adaptativo
    if config.enable_adaptive_conf:
        detections = adaptive_confidence_filter(
            detections,
            config.conf,
            config.small_object_threshold,
        )
    else:
        detections = [d for d in detections if d.confidence >= config.conf]

    # Limita ao máximo
    detections.sort(key=lambda d: d.confidence, reverse=True)
    detections = detections[: config.max_det]

    return detections