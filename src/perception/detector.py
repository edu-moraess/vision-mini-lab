"""
YOLO detector wrapper using Ultralytics.
Model is loaded once via Streamlit cache_resource.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]


class Detector:
    """Thin wrapper around Ultralytics YOLO for detection only."""

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        conf: float = 0.35,
        iou: float = 0.45,
    ):
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
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

    def detect(self, image_rgb: np.ndarray) -> List[Detection]:
        """
        Run detection on RGB image.
        Returns list of Detection objects.
        """
        if not self.is_ready:
            return []

        try:
            results = self._model.predict(
                source=image_rgb,
                conf=self.conf,
                iou=self.iou,
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

    def set_thresholds(self, conf: float, iou: float) -> None:
        self.conf = float(np.clip(conf, 0.05, 0.95))
        self.iou = float(np.clip(iou, 0.1, 0.9))
