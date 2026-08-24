import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking import compute_tracking_metrics
from src.detector import Detection


def test_compute_tracking_metrics_single():
    track_history = {
        1: [(100, 100), (110, 105)]
    }
    detections = [Detection(0, "person", 0.9, 0, 0, 10, 10, track_id=1)]
    metrics = compute_tracking_metrics(track_history, detections)

    assert 1 in metrics
    assert metrics[1]["dx"] == 10.0
    assert metrics[1]["dy"] == 5.0
    assert metrics[1]["displacement_px"] > 0
    assert metrics[1]["direction"] in ["DIREITA", "CIMA", "BAIXO", "ESQUERDA", "PARADO"]


def test_compute_tracking_metrics_static():
    track_history = {
        1: [(100, 100), (100, 100)]
    }
    detections = [Detection(0, "person", 0.9, 0, 0, 10, 10, track_id=1)]
    metrics = compute_tracking_metrics(track_history, detections)

    assert metrics[1]["displacement_px"] == 0.0
    assert metrics[1]["direction"] == "PARADO"


def test_compute_tracking_metrics_insufficient_history():
    track_history = {
        1: [(100, 100)]
    }
    detections = [Detection(0, "person", 0.9, 0, 0, 10, 10, track_id=1)]
    metrics = compute_tracking_metrics(track_history, detections)

    assert metrics[1]["displacement_px"] == 0.0
    assert metrics[1]["direction"] == "PARADO"


def test_compute_tracking_metrics_no_track_id():
    track_history = {}
    detections = [Detection(0, "person", 0.9, 0, 0, 10, 10, track_id=None)]
    metrics = compute_tracking_metrics(track_history, detections)
    assert metrics == {}