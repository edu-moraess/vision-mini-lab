"""
Object detection + tracking using Ultralytics.

Detection is always preserved.
Tracking is optional: if Ultralytics cannot assign an ID,
the detected object is still returned with track_id=-1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]

    dx: float = 0.0
    dy: float = 0.0
    speed: float = 0.0
    direction: str = "STATIONARY"
    state: str = "STATIONARY"


@dataclass
class TrackHistory:
    points: List[Tuple[float, float]] = field(default_factory=list)
    max_len: int = 60

    def add(self, center: Tuple[float, float]) -> None:
        self.points.append(center)

        if len(self.points) > self.max_len:
            self.points = self.points[-self.max_len:]


class Tracker:
    """
    YOLO detection with optional persistent tracking.

    Important:
    - Every YOLO detection is returned.
    - A missing tracking ID never removes a detection.
    """

    def __init__(
        self,
        model: YOLO,
        conf: float = 0.35,
        iou: float = 0.45,
        history_len: int = 60,
    ):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.history_len = history_len

        self.histories: Dict[int, TrackHistory] = {}
        self._prev_centers: Dict[int, Tuple[float, float]] = {}

        # Synthetic IDs for detections without tracker IDs.
        self._next_detection_id = -1

    def update(self, image_rgb: np.ndarray) -> List[TrackedObject]:
        """
        Run YOLO tracking.

        If tracking IDs exist, they are preserved.
        If tracking IDs do not exist, detections are still returned.
        """

        try:
            results = self.model.track(
                source=image_rgb,
                conf=self.conf,
                iou=self.iou,
                persist=True,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("Tracking failed: %s", exc)

            # Fallback: plain detection.
            return self._detect_without_tracking(image_rgb)

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes
        names = result.names or {}

        tracked: List[TrackedObject] = []

        has_ids = boxes.id is not None

        ids = None
        if has_ids:
            ids = boxes.id.cpu().numpy().astype(int)

        current_ids = set()

        for i in range(len(boxes)):
            try:
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                xyxy = boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = map(float, xyxy)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                class_name = names.get(
                    cls_id,
                    f"class_{cls_id}",
                )

                # Real tracker ID when available.
                if ids is not None:
                    track_id = int(ids[i])
                else:
                    # Negative IDs mean detection without persistent tracking.
                    track_id = self._next_detection_id
                    self._next_detection_id -= 1

                obj = TrackedObject(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                )

                tracked.append(obj)

                if track_id > 0:
                    current_ids.add(track_id)

                    if track_id not in self.histories:
                        self.histories[track_id] = TrackHistory(
                            max_len=self.history_len
                        )

                    self.histories[track_id].add(
                        (cx, cy)
                    )

            except Exception:
                logger.exception(
                    "Failed to parse detection."
                )
                continue

        # Remove histories for tracks no longer visible.
        lost = [
            track_id
            for track_id in self.histories
            if track_id not in current_ids
        ]

        for track_id in lost:
            del self.histories[track_id]
            self._prev_centers.pop(track_id, None)

        return tracked

    def _detect_without_tracking(
        self,
        image_rgb: np.ndarray,
    ) -> List[TrackedObject]:
        """
        Emergency fallback using YOLO detection only.
        """

        try:
            results = self.model.predict(
                source=image_rgb,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception as exc:
            logger.exception(
                "Detection fallback failed: %s",
                exc,
            )
            return []

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes
        names = result.names or {}

        detections = []

        for i in range(len(boxes)):
            try:
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                xyxy = boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = map(float, xyxy)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                detections.append(
                    TrackedObject(
                        track_id=self._next_detection_id,
                        class_id=cls_id,
                        class_name=names.get(
                            cls_id,
                            f"class_{cls_id}",
                        ),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                    )
                )

                self._next_detection_id -= 1

            except Exception:
                continue

        return detections

    def get_trajectory(
        self,
        track_id: int,
    ) -> List[Tuple[float, float]]:

        hist = self.histories.get(track_id)

        if hist is None:
            return []

        return list(hist.points)

    def set_thresholds(
        self,
        conf: float,
        iou: float,
    ) -> None:

        self.conf = float(
            np.clip(conf, 0.05, 0.95)
        )

        self.iou = float(
            np.clip(iou, 0.1, 0.9)
        )

    def reset(self) -> None:
        self.histories.clear()
        self._prev_centers.clear()
        self._next_detection_id = -1