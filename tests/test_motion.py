"""Tests for motion analysis."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.motion.motion import MotionAnalyzer, _angle_to_direction
from src.perception.tracker import TrackedObject


def _obj(tid, cx, cy, cls="person"):
    return TrackedObject(
        track_id=tid,
        class_id=0,
        class_name=cls,
        confidence=0.9,
        bbox=(cx - 10, cy - 10, cx + 10, cy + 10),
        center=(cx, cy),
    )


def test_stationary_first_frame():
    ma = MotionAnalyzer(min_move=2.5)
    objs = [_obj(1, 100, 100)]
    out = ma.update(objs)
    assert out[0].state == "STATIONARY"
    assert out[0].speed == 0.0
    assert out[0].direction == "STATIONARY"


def test_moving_right():
    ma = MotionAnalyzer(min_move=2.5, alpha=1.0)
    ma.update([_obj(1, 100, 100)])
    out = ma.update([_obj(1, 120, 100)])
    assert out[0].state == "MOVING"
    assert out[0].dx == pytest.approx(20.0)
    assert out[0].dy == pytest.approx(0.0)
    assert out[0].speed == pytest.approx(20.0)
    assert out[0].direction == "RIGHT"


def test_moving_up():
    ma = MotionAnalyzer(min_move=2.5, alpha=1.0)
    ma.update([_obj(1, 100, 100)])
    # image y increases downward → negative dy = UP
    out = ma.update([_obj(1, 100, 80)])
    assert out[0].direction == "UP"
    assert out[0].state == "MOVING"


def test_small_displacement_ignored():
    ma = MotionAnalyzer(min_move=5.0, alpha=1.0)
    ma.update([_obj(1, 100, 100)])
    out = ma.update([_obj(1, 102, 100)])
    assert out[0].state == "STATIONARY"


def test_direction_map():
    assert _angle_to_direction(0) == "RIGHT"
    assert _angle_to_direction(90) == "UP"
    assert _angle_to_direction(180) == "LEFT"
    assert _angle_to_direction(270) == "DOWN"


def test_reset():
    ma = MotionAnalyzer()
    ma.update([_obj(1, 10, 10)])
    ma.reset()
    out = ma.update([_obj(1, 50, 50)])
    assert out[0].speed == 0.0
