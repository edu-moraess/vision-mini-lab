"""
Event detection: enter/exit ROI, line crossing, motion state changes.
Simple in-memory event log with debounce.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.perception.tracker import TrackedObject


@dataclass
class Event:
    timestamp: float
    event_type: str
    track_id: int
    class_name: str
    direction: str = ""
    extra: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event": self.event_type,
            "track_id": self.track_id,
            "class": self.class_name,
            "direction": self.direction,
            "extra": self.extra,
        }


class EventEngine:
    """
    Detects:
    - OBJECT_ENTERED / OBJECT_EXITED (rectangular ROI)
    - LINE_CROSSED (single virtual line)
    - STARTED_MOVING / STOPPED
    """

    def __init__(
        self,
        max_events: int = 200,
        line_debounce_sec: float = 1.0,
    ):
        self.max_events = max_events
        self.line_debounce_sec = line_debounce_sec
        self.events: List[Event] = []

        # ROI state: track_id -> inside
        self._inside_roi: Dict[int, bool] = {}
        # Motion state
        self._was_moving: Dict[int, bool] = {}
        # Line crossing: last cross time per track
        self._last_line_cross: Dict[int, float] = {}
        # Previous side of the line per track (-1 left/above, +1 right/below)
        self._prev_side: Dict[int, int] = {}

        # Geometry (set externally)
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    def set_roi(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        self.roi = roi

    def set_line(
        self,
        p1: Optional[Tuple[int, int]],
        p2: Optional[Tuple[int, int]],
    ) -> None:
        if p1 is None or p2 is None:
            self.line = None
        else:
            self.line = (p1, p2)

    def _point_in_roi(self, cx: float, cy: float) -> bool:
        if self.roi is None:
            return False
        x1, y1, x2, y2 = self.roi
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _side_of_line(self, cx: float, cy: float) -> int:
        """Return +1 or -1 depending on which side of the line the point is."""
        if self.line is None:
            return 0
        (x1, y1), (x2, y2) = self.line
        cross = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
        if cross >= 0:
            return 1
        return -1

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def update(self, objects: List[TrackedObject], frame_time: Optional[float] = None) -> List[Event]:
        """
        Process current frame objects and emit new events.
        Returns only the newly generated events this call.
        Ignores detection-only objects (track_id <= 0) for event generation
        to avoid spam from temporary detections.
        """
        now = frame_time if frame_time is not None else time.time()
        new_events: List[Event] = []
        current_ids = set()

        for obj in objects:
            tid = obj.track_id
            current_ids.add(tid)
            cx, cy = obj.center

            # Skip event generation for detection-only objects
            if tid <= 0:
                continue

            # --- ROI enter / exit ---
            if self.roi is not None:
                inside = self._point_in_roi(cx, cy)
                was_inside = self._inside_roi.get(tid, False)
                if inside and not was_inside:
                    ev = Event(
                        timestamp=now,
                        event_type="OBJECT_ENTERED",
                        track_id=tid,
                        class_name=obj.class_name,
                    )
                    self._emit(ev)
                    new_events.append(ev)
                elif not inside and was_inside:
                    ev = Event(
                        timestamp=now,
                        event_type="OBJECT_EXITED",
                        track_id=tid,
                        class_name=obj.class_name,
                    )
                    self._emit(ev)
                    new_events.append(ev)
                self._inside_roi[tid] = inside

            # --- Motion state changes ---
            is_moving = obj.state == "MOVING"
            was_moving = self._was_moving.get(tid, False)
            if is_moving and not was_moving:
                ev = Event(
                    timestamp=now,
                    event_type="STARTED_MOVING",
                    track_id=tid,
                    class_name=obj.class_name,
                    direction=obj.direction,
                )
                self._emit(ev)
                new_events.append(ev)
            elif not is_moving and was_moving:
                ev = Event(
                    timestamp=now,
                    event_type="STOPPED",
                    track_id=tid,
                    class_name=obj.class_name,
                )
                self._emit(ev)
                new_events.append(ev)
            self._was_moving[tid] = is_moving

            # --- Line crossing ---
            if self.line is not None:
                side = self._side_of_line(cx, cy)
                prev_side = self._prev_side.get(tid)
                if prev_side is not None and side != 0 and prev_side != 0 and side != prev_side:
                    last = self._last_line_cross.get(tid, 0.0)
                    if (now - last) >= self.line_debounce_sec:
                        direction = obj.direction if obj.direction != "STATIONARY" else "UNKNOWN"
                        ev = Event(
                            timestamp=now,
                            event_type="LINE_CROSSED",
                            track_id=tid,
                            class_name=obj.class_name,
                            direction=direction,
                        )
                        self._emit(ev)
                        new_events.append(ev)
                        self._last_line_cross[tid] = now
                self._prev_side[tid] = side

        # Prune lost tracks
        for store in (self._inside_roi, self._was_moving, self._prev_side):
            lost = [k for k in store if k not in current_ids]
            for k in lost:
                del store[k]

        return new_events

    def get_recent(self, n: int = 50) -> List[Event]:
        return self.events[-n:]

    def clear(self) -> None:
        self.events.clear()
        self._inside_roi.clear()
        self._was_moving.clear()
        self._last_line_cross.clear()
        self._prev_side.clear()

    def count_inside_roi(self, objects: List[TrackedObject]) -> int:
        if self.roi is None:
            return 0
        return sum(1 for o in objects if self._point_in_roi(*o.center))
