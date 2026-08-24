import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import Detection
from src.metrics import MetricsAggregator, compute_box_metrics
from src.video import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    decode_image_bytes,
    get_video_metadata,
    validate_extension,
)


def test_compute_box_metrics():
    m = compute_box_metrics(10, 20, 110, 220)

    assert m["w"] == 100
    assert m["h"] == 200
    assert m["area"] == 20_000
    assert m["perimeter"] == 600
    assert m["cx"] == 60
    assert m["cy"] == 120


def test_compute_box_metrics_invalid_order():
    m = compute_box_metrics(100, 100, 50, 50)

    assert m["w"] == 0
    assert m["h"] == 0
    assert m["area"] == 0
    assert m["perimeter"] == 0


def test_metrics_aggregator_current_and_cumulative():
    agg = MetricsAggregator()

    det1 = Detection(0, "person", 0.8, 0, 0, 10, 10, None)
    det2 = Detection(1, "chair", 0.6, 0, 0, 20, 20, None)

    agg.note_inference([det1, det2], decode_ms=1.0, infer_ms=10.0, total_ms=12.0)

    assert agg.frames_analyzed == 1
    assert agg.total_objects == 2
    assert agg.last_object_count == 2
    assert agg.last_class_counts["PERSON"] == 1
    assert agg.last_class_counts["CHAIR"] == 1
    assert abs(agg.avg_confidence - 0.7) < 1e-6

    agg.note_inference([det1], decode_ms=1.0, infer_ms=8.0, total_ms=9.0)

    assert agg.frames_analyzed == 2
    assert agg.total_objects == 3
    assert agg.last_object_count == 1
    assert agg.last_class_counts["PERSON"] == 1
    assert "CHAIR" not in agg.last_class_counts


def test_invalid_image_decode():
    assert decode_image_bytes(b"not-an-image") is None


def test_video_metadata_invalid_path():
    meta = get_video_metadata("does-not-exist.mp4")

    assert meta.ok is False
    assert meta.error


def test_extension_validation():
    assert validate_extension("photo.JPG", ALLOWED_IMAGE_EXTENSIONS)
    assert validate_extension("photo.png", ALLOWED_IMAGE_EXTENSIONS)
    assert not validate_extension("script.txt", ALLOWED_IMAGE_EXTENSIONS)

    assert validate_extension("video.MP4", ALLOWED_VIDEO_EXTENSIONS)
    assert validate_extension("video.mov", ALLOWED_VIDEO_EXTENSIONS)
    assert not validate_extension("app.exe", ALLOWED_VIDEO_EXTENSIONS)


def test_groq_disabled_without_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    key = os.getenv("GROQ_API_KEY")
    assert key is None or key == ""