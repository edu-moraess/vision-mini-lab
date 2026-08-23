"""
Motion analysis: displacement, speed (px/frame), direction, state.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.perception.tracker import TrackedObject


MIN_MOVE_PX = 2.5
SMOOTH_ALPHA = 0.35

DIRECTION_MAP = [
    (0, "RIGHT"),
    (45, "UP-RIGHT"),
    (90, "UP"),
    (135, "UP-LEFT"),
    (180, "LEFT"),
    (225, "DOWN-LEFT"),
    (270, "DOWN"),
    (315, "DOWN-RIGHT"),
]


def _angle_to_direction(angle_deg: float) -> str:
    """Map angle in degrees [0, 360) to 8-way direction."""
    angle = angle_deg % 360.0
    best = "RIGHT"
    best_diff = 360.0
    for ref, name in DIRECTION_MAP:
        diff = min(abs(angle - ref), 360.0 - abs(angle - ref))
        if diff < best_diff:
            best_diff = diff
            best = name
    return best


class MotionAnalyzer:
    """
    Computes per-track motion metrics from consecutive centers.
    Only processes objects with valid track IDs for meaningful motion.
    """

    def __init__(self, min_move: float = MIN_MOVE_PX, alpha: float = SMOOTH_ALPHA):
        self.min_move = min_move
        self.alpha = alpha
        self._prev_centers: Dict[int, Tuple[float, float]] = {}
        self._smoothed_speed: Dict[int, float] = {}

    def update(self, objects: List[TrackedObject]) -> List[TrackedObject]:
        """
        Enrich TrackedObject list with dx, dy, speed, direction, state.
        Returns the same list (mutated in place).
        """
        current_ids = set()

        for obj in objects:
            tid = obj.track_id
            current_ids.add(tid)
            cx, cy = obj.center

            prev = self._prev_centers.get(tid)
            if prev is None:
                obj.dx = 0.0
                obj.dy = 0.0
                obj.speed = 0.0
                obj.direction = "STATIONARY"
                obj.state = "STATIONARY"
            else:
                dx = cx - prev[0]
                dy = cy - prev[1]
                raw_speed = math.sqrt(dx * dx + dy * dy)

                # EMA smoothing
                prev_s = self._smoothed_speed.get(tid, raw_speed)
                speed = self.alpha * raw_speed + (1.0 - self.alpha) * prev_s
                self._smoothed_speed[tid] = speed

                obj.dx = dx
                obj.dy = dy
                obj.speed = speed

                if speed < self.min_move:
                    obj.direction = "STATIONARY"
                    obj.state = "STATIONARY"
                else:
                    angle = math.degrees(math.atan2(-dy, dx))
                    if angle < 0:
                        angle += 360.0
                    obj.direction = _angle_to_direction(angle)
                    obj.state = "MOVING"

            self._prev_centers[tid] = (cx, cy)

        # Prune lost tracks
        lost = [k for k in self._prev_centers if k not in current_ids]
        for k in lost:
            del self._prev_centers[k]
            self._smoothed_speed.pop(k, None)

        return objects

    def reset(self) -> None:
        self._prev_centers.clear()
        self._smoothed_speed.clear()
