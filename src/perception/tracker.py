"""
VISION MINI LAB
Detection + Tracking Engine

Design goals:
- YOLO detection is always preserved.
- Tracking is an additional layer, not a gate.
- Detection confidence filtering is separate from tracking confidence.
- Temporal confirmation to reduce flickering.
- Stable tracking IDs when available.
- Real pipeline statistics.
- Temporal consistency buffer for video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
from ultralytics import YOLO

from src.perception.detector import (
    Detection,
    DetectionConfig,
    PipelineStats,
    run_full_pipeline,
    get_id_color,
)

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

    consecutive_frames: int = 1
    confirmed: bool = False
    small_object: bool = False

    @property
    def id_color(self) -> Tuple[int, int, int]:
        return get_id_color(self.track_id)

    @property
    def display_id(self) -> str:
        if self.track_id > 0:
            return f"#{self.track_id}"
        return "DET"


@dataclass
class TrackHistory:
    points: List[Tuple[float, float]] = field(default_factory=list)
    max_len: int = 60

    def add(self, center: Tuple[float, float]) -> None:
        self.points.append(center)
        if len(self.points) > self.max_len:
            self.points = self.points[-self.max_len:]


class Tracker:

    def __init__(
        self,
        model: YOLO,
        config: Optional[DetectionConfig] = None,
        confirmation_frames: int = 2,
        history_len: int = 60,
    ):
        self.model = model

        if config is None:
            config = DetectionConfig()

        self.config = config
        self.confirmation_frames = int(confirmation_frames)
        self.history_len = int(history_len)

        self.histories: Dict[int, TrackHistory] = {}
        self._prev_centers: Dict[int, Tuple[float, float]] = {}
        self._confirmation_counts: Dict[int, int] = {}
        self._next_detection_id = -1
        self._temporal_buffer: Dict = {}

        # Real pipeline statistics
        self.last_stats = PipelineStats()

    # -------------------------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------------------------

    def update(self, image_rgb: np.ndarray, is_video: bool = False) -> List[TrackedObject]:
        """
        Execute the complete detection/tracking pipeline.
        Never drops detections due to tracking failure.
        """
        if image_rgb is None or image_rgb.size == 0:
            self.last_stats = PipelineStats()
            return []

        temporal_buffer = self._temporal_buffer if is_video else None

        detections, stats = run_full_pipeline(
            self.model,
            image_rgb,
            self.config,
            use_tracking=True,
            temporal_buffer=temporal_buffer,
        )

        self.last_stats = stats

        tracked: List[TrackedObject] = []
        current_ids = set()
        raw_count = len(detections)
        small_count = sum(1 for d in detections if d.small_object)

        for det in detections:
            track_id = det.track_id

            # Tracking confidence gate: only confirmed tracks get persistent IDs
            if (
                track_id is not None
                and track_id > 0
                and det.confidence >= self.config.tracking_conf
            ):
                count = self._confirmation_counts.get(track_id, 0) + 1
                self._confirmation_counts[track_id] = count
                confirmed = count >= self.confirmation_frames
            else:
                # No tracking ID or below tracking confidence → detection-only ID
                track_id = self._next_detection_id
                self._next_detection_id -= 1
                confirmed = True  # detection-only visible immediately

            if track_id > 0:
                current_ids.add(track_id)
                if track_id not in self.histories:
                    self.histories[track_id] = TrackHistory(max_len=self.history_len)
                self.histories[track_id].add(det.center)

            obj = TrackedObject(
                track_id=track_id,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                center=det.center,
                consecutive_frames=self._confirmation_counts.get(track_id, 1),
                confirmed=confirmed,
                small_object=det.small_object,
            )

            tracked.append(obj)

        # Clean up lost trajectories
        lost_ids = [tid for tid in self.histories if tid not in current_ids]
        for tid in lost_ids:
            del self.histories[tid]
            self._prev_centers.pop(tid, None)
            self._confirmation_counts.pop(tid, None)

        # Update stats with final counts
        self.last_stats.small_objects = small_count
        self.last_stats.tracked = sum(1 for o in tracked if o.track_id > 0)

        return tracked

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------

    def set_config(self, config: DetectionConfig) -> None:
        self.config = config

    def set_thresholds(
        self,
        conf: float,
        iou: float,
        max_det: Optional[int] = None,
    ) -> None:
        self.config.conf = float(np.clip(conf, 0.05, 0.95))
        self.config.iou = float(np.clip(iou, 0.10, 0.90))
        if max_det is not None:
            self.config.max_det = int(np.clip(max_det, 1, 100))

    # -------------------------------------------------------------------------
    # TRAJECTORY
    # -------------------------------------------------------------------------

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        history = self.histories.get(track_id)
        if history is None:
            return []
        return list(history.points)

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        self.histories.clear()
        self._prev_centers.clear()
        self._confirmation_counts.clear()
        self._next_detection_id = -1
        self._temporal_buffer.clear()
        self.last_stats = PipelineStats()
