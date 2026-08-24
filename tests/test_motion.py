import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.motion import (
    MotionAnalyzer,
    classify_direction_8way,
    classify_speed,
    STATUS_STATIONARY,
    STATUS_MOVING,
    STATUS_FAST,
    EVENT_STARTED_MOVING,
    EVENT_STOPPED,
)


def test_classify_direction_cardinal():
    assert classify_direction_8way(10, 0) == "DIREITA"
    assert classify_direction_8way(-10, 0) == "ESQUERDA"
    assert classify_direction_8way(0, 10) == "BAIXO"
    assert classify_direction_8way(0, -10) == "CIMA"


def test_classify_direction_diagonal():
    assert classify_direction_8way(10, 10) == "BAIXO_DIREITA"
    assert classify_direction_8way(-10, 10) == "BAIXO_ESQUERDA"
    assert classify_direction_8way(-10, -10) == "CIMA_ESQUERDA"
    assert classify_direction_8way(10, -10) == "CIMA_DIREITA"


def test_classify_direction_noise_is_stationary():
    assert classify_direction_8way(0.1, 0.1, min_displacement=0.75) == STATUS_STATIONARY


def test_classify_speed_thresholds():
    assert classify_speed(0.1, stationary_threshold=0.75, fast_threshold=12.0) == STATUS_STATIONARY
    assert classify_speed(5.0, stationary_threshold=0.75, fast_threshold=12.0) == STATUS_MOVING
    assert classify_speed(15.0, stationary_threshold=0.75, fast_threshold=12.0) == STATUS_FAST


def test_motion_analyzer_smoothing_reduces_noise_oscillation():
    analyzer = MotionAnalyzer(smoothing_alpha=0.3, stationary_threshold=2.0, fast_threshold=20.0)
    # Um único frame com deslocamento momentâneo não deve, sozinho, levar a
    # velocidade suavizada acima do limiar (a suavização amortece o pico).
    analyzer.update(1, dx=0, dy=0, displacement_px=0.0)
    result = analyzer.update(1, dx=8, dy=0, displacement_px=8.0)
    assert result["speed_smoothed"] < 8.0


def test_motion_analyzer_transition_events():
    analyzer = MotionAnalyzer(smoothing_alpha=1.0, stationary_threshold=1.0, fast_threshold=20.0)
    r1 = analyzer.update(1, dx=0, dy=0, displacement_px=0.0)
    assert r1["status"] == STATUS_STATIONARY
    assert r1["transition_event"] is None

    r2 = analyzer.update(1, dx=5, dy=0, displacement_px=5.0)
    assert r2["status"] == STATUS_MOVING
    assert r2["transition_event"] == EVENT_STARTED_MOVING

    r3 = analyzer.update(1, dx=0, dy=0, displacement_px=0.0)
    assert r3["status"] == STATUS_STATIONARY
    assert r3["transition_event"] == EVENT_STOPPED


def test_motion_analyzer_prune_removes_inactive():
    analyzer = MotionAnalyzer()
    analyzer.update(1, dx=5, dy=0, displacement_px=5.0)
    analyzer.update(2, dx=5, dy=0, displacement_px=5.0)
    analyzer.prune(active_track_ids=[1])
    assert 1 in analyzer._smoothed_speed
    assert 2 not in analyzer._smoothed_speed
