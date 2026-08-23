"""
Object tracking using Ultralytics built-in tracker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    # motion fields filled later
    dx: float = 0.0
    dy: float = 0.0
    speed: float = 0.0
    direction: str = "STATIONARY"
    state: str = "STATIONARY"  # STATIONARY | MOVING


@dataclass
class TrackHistory:
    """Short trajectory history per track_id."""
    points: List[Tuple[float, float]] = field(default_factory=list)
    max_len: int = 60

    def add(self, center: Tuple[float, float]) -> None:
        self.points.append(center)
        if len(self.points) > self.max_len:
            self.points = self.points[-self.max_len :]


class Tracker:
    """
    Wrapper that runs YOLO track() and maintains short histories.
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

    def update(self, image_rgb: np.ndarray) -> List[TrackedObject]:
        """
        Run tracking on a single frame.
        Returns list of TrackedObject with current detections.
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
            return []

        tracked: List[TrackedObject] = []
        if not results:
            return tracked

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return tracked

        names = result.names or {}
        boxes = result.boxes
        ids = boxes.id.cpu().numpy().astype(int)

        current_ids = set()

        for i, tid in enumerate(ids):
            try:
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(float, xyxy)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                name = names.get(cls_id, f"class_{cls_id}")

                obj = TrackedObject(
                    track_id=int(tid),
                    class_id=cls_id,
                    class_name=name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                )
                tracked.append(obj)
                current_ids.add(int(tid))

                # update history
                if tid not in self.histories:
                    self.histories[tid] = TrackHistory(max_len=self.history_len)
                self.histories[tid].add((cx, cy))

            except Exception:
                continue

        # prune histories of lost tracks
        lost = [k for k in self.histories if k not in current_ids]
        for k in lost:
            del self.histories[k]
            self._prev_centers.pop(k, None)

        return tracked

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        hist = self.histories.get(track_id)
        if hist is None:
            return []
        return list(hist.points)

    def set_thresholds(self, conf: float, iou: float) -> None:
        self.conf = float(np.clip(conf, 0.05, 0.95))
        self.iou = float(np.clip(iou, 0.1, 0.9))

    def reset(self) -> None:
        self.histories.clear()
        self._prev_centers.clear()
