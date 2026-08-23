"""Tests for event engine."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.events.events import EventEngine
from src.perception.tracker import TrackedObject


def _obj(tid, cx, cy, state="STATIONARY", direction="STATIONARY"):
    o = TrackedObject(
        track_id=tid,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(cx - 5, cy - 5, cx + 5, cy + 5),
        center=(cx, cy),
    )
    o.state = state
    o.direction = direction
    return o


def test_roi_enter_exit():
    eng = EventEngine()
    eng.set_roi((50, 50, 150, 150))

    # outside
    eng.update([_obj(1, 10, 10)])
    assert len(eng.events) == 0

    # enter
    new = eng.update([_obj(1, 100, 100)])
    assert any(e.event_type == "OBJECT_ENTERED" for e in new)

    # stay
    eng.update([_obj(1, 110, 110)])
    # exit
    new = eng.update([_obj(1, 10, 10)])
    assert any(e.event_type == "OBJECT_EXITED" for e in new)


def test_line_crossing():
    eng = EventEngine(line_debounce_sec=0.0)
    eng.set_line((0, 100), (200, 100))  # horizontal line

    # start above
    eng.update([_obj(1, 50, 50, state="MOVING", direction="DOWN")])
    # cross below
    new = eng.update([_obj(1, 50, 150, state="MOVING", direction="DOWN")])
    assert any(e.event_type == "LINE_CROSSED" for e in new)


def test_motion_events():
    eng = EventEngine()
    eng.update([_obj(1, 0, 0, state="STATIONARY")])
    new = eng.update([_obj(1, 0, 0, state="MOVING", direction="RIGHT")])
    assert any(e.event_type == "STARTED_MOVING" for e in new)

    new = eng.update([_obj(1, 0, 0, state="STATIONARY")])
    assert any(e.event_type == "STOPPED" for e in new)


def test_max_events():
    eng = EventEngine(max_events=5)
    eng.set_roi((0, 0, 1000, 1000))
    for i in range(10):
        eng.update([_obj(i, 50, 50)])
        eng.update([])  # force exit on next? simplify
    assert len(eng.events) <= 5


def test_count_inside():
    eng = EventEngine()
    eng.set_roi((0, 0, 100, 100))
    objs = [_obj(1, 50, 50), _obj(2, 200, 200)]
    assert eng.count_inside_roi(objs) == 1
