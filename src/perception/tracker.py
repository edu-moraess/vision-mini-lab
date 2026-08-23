"""
VISION MINI LAB
Detection + Tracking Engine

Design goals:
- YOLO detection is always preserved.
- Tracking is optional.
- Detection confidence filtering.
- Maximum detections per frame.
- Temporal confirmation to reduce flickering.
- Stable tracking IDs when available.
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

    # Temporal confidence
    consecutive_frames: int = 1
    confirmed: bool = False


@dataclass
class TrackHistory:
    points: List[Tuple[float, float]] = field(
        default_factory=list
    )

    max_len: int = 60

    def add(
        self,
        center: Tuple[float, float],
    ) -> None:

        self.points.append(center)

        if len(self.points) > self.max_len:
            self.points = self.points[-self.max_len:]


class Tracker:

    def __init__(
        self,
        model: YOLO,
        conf: float = 0.50,
        iou: float = 0.50,
        max_det: int = 20,
        confirmation_frames: int = 2,
        history_len: int = 60,
    ):

        self.model = model

        self.conf = float(conf)
        self.iou = float(iou)

        self.max_det = int(max_det)

        self.confirmation_frames = int(
            confirmation_frames
        )

        self.history_len = int(
            history_len
        )

        self.histories: Dict[
            int,
            TrackHistory,
        ] = {}

        self._prev_centers: Dict[
            int,
            Tuple[float, float],
        ] = {}

        self._confirmation_counts: Dict[
            int,
            int,
        ] = {}

        self._next_detection_id = -1

    # -------------------------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------------------------

    def update(
        self,
        image_rgb: np.ndarray,
    ) -> List[TrackedObject]:

        try:

            results = self.model.track(
                source=image_rgb,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                persist=True,
                verbose=False,
            )

        except Exception as exc:

            logger.exception(
                "Tracking failed: %s",
                exc,
            )

            return self._detect_without_tracking(
                image_rgb
            )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes

        names = result.names or {}

        tracked: List[
            TrackedObject
        ] = []

        ids = None

        if boxes.id is not None:

            ids = (
                boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

        current_ids = set()

        for i in range(
            len(boxes)
        ):

            try:

                cls_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                if confidence < self.conf:
                    continue

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    float,
                    xyxy,
                )

                cx = (
                    x1 + x2
                ) / 2.0

                cy = (
                    y1 + y2
                ) / 2.0

                class_name = names.get(
                    cls_id,
                    f"class_{cls_id}",
                )

                # -------------------------------------------------------------
                # ID
                # -------------------------------------------------------------

                if ids is not None:

                    track_id = int(
                        ids[i]
                    )

                else:

                    track_id = (
                        self._next_detection_id
                    )

                    self._next_detection_id -= 1

                # -------------------------------------------------------------
                # Temporal confirmation
                # -------------------------------------------------------------

                if track_id > 0:

                    count = (
                        self._confirmation_counts.get(
                            track_id,
                            0,
                        )
                        + 1
                    )

                    self._confirmation_counts[
                        track_id
                    ] = count

                    confirmed = (
                        count
                        >= self.confirmation_frames
                    )

                else:

                    # Detection-only objects are
                    # immediately visible.
                    confirmed = True

                # -------------------------------------------------------------
                # History
                # -------------------------------------------------------------

                if track_id > 0:

                    current_ids.add(
                        track_id
                    )

                    if (
                        track_id
                        not in self.histories
                    ):

                        self.histories[
                            track_id
                        ] = TrackHistory(
                            max_len=self.history_len
                        )

                    self.histories[
                        track_id
                    ].add(
                        (
                            cx,
                            cy,
                        )
                    )

                obj = TrackedObject(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    center=(
                        cx,
                        cy,
                    ),
                    consecutive_frames=(
                        self._confirmation_counts.get(
                            track_id,
                            1,
                        )
                    ),
                    confirmed=confirmed,
                )

                tracked.append(obj)

            except Exception:

                logger.exception(
                    "Failed to parse detection"
                )

        # ---------------------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------------------

        lost_ids = [
            track_id
            for track_id in self.histories
            if track_id not in current_ids
        ]

        for track_id in lost_ids:

            del self.histories[
                track_id
            ]

            self._prev_centers.pop(
                track_id,
                None,
            )

            self._confirmation_counts.pop(
                track_id,
                None,
            )

        return tracked

    # -------------------------------------------------------------------------
    # DETECTION FALLBACK
    # -------------------------------------------------------------------------

    def _detect_without_tracking(
        self,
        image_rgb: np.ndarray,
    ) -> List[TrackedObject]:

        try:

            results = self.model.predict(
                source=image_rgb,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
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

        for i in range(
            len(boxes)
        ):

            try:

                cls_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                if confidence < self.conf:
                    continue

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    float,
                    xyxy,
                )

                cx = (
                    x1 + x2
                ) / 2.0

                cy = (
                    y1 + y2
                ) / 2.0

                detections.append(
                    TrackedObject(
                        track_id=(
                            self._next_detection_id
                        ),
                        class_id=cls_id,
                        class_name=names.get(
                            cls_id,
                            f"class_{cls_id}",
                        ),
                        confidence=confidence,
                        bbox=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        center=(
                            cx,
                            cy,
                        ),
                        confirmed=True,
                    )
                )

                self._next_detection_id -= 1

            except Exception:

                logger.exception(
                    "Fallback detection parsing failed"
                )

        return detections

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------

    def set_thresholds(
        self,
        conf: float,
        iou: float,
        max_det: int | None = None,
    ) -> None:

        self.conf = float(
            np.clip(
                conf,
                0.05,
                0.95,
            )
        )

        self.iou = float(
            np.clip(
                iou,
                0.10,
                0.90,
            )
        )

        if max_det is not None:

            self.max_det = int(
                np.clip(
                    max_det,
                    1,
                    100,
                )
            )

    # -------------------------------------------------------------------------
    # TRAJECTORY
    # -------------------------------------------------------------------------

    def get_trajectory(
        self,
        track_id: int,
    ) -> List[
        Tuple[float, float]
    ]:

        history = (
            self.histories.get(
                track_id
            )
        )

        if history is None:
            return []

        return list(
            history.points
        )

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------

    def reset(self) -> None:

        self.histories.clear()

        self._prev_centers.clear()

        self._confirmation_counts.clear()

        self._next_detection_id = -1