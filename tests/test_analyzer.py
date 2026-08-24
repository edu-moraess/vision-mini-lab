import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyzer import (
    classify_size,
    compute_aspect_ratio,
    enrich_detection,
    estimate_relative_distance,
    filter_detections,
    summarize_analysis,
)
from src.detector import Detection


def test_classify_size():
    assert classify_size(999) == "MUITO_PEQUENO"
    assert classify_size(1000) == "PEQUENO"
    assert classify_size(4999) == "PEQUENO"
    assert classify_size(5000) == "MEDIO"
    assert classify_size(11999) == "MEDIO"
    assert classify_size(12000) == "GRANDE"
    assert classify_size(20000) == "GRANDE"
    assert classify_size(20001) == "MUITO_GRANDE"


def test_estimate_relative_distance():
    assert estimate_relative_distance(180) == 1.0
    assert estimate_relative_distance(90) == 0.5
    assert estimate_relative_distance(360) == 2.0
    assert estimate_relative_distance(0) == 0.0
    assert estimate_relative_distance(100, 0) == 0.0


def test_compute_aspect_ratio():
    assert compute_aspect_ratio(200, 100) == 2.0
    assert compute_aspect_ratio(100, 200) == 0.5
    assert compute_aspect_ratio(100, 0) == 0.0


def test_enrich_detection():
    det = Detection(0, "person", 0.9, 100, 200, 300, 400)
    enriched = enrich_detection(det, 640, 480)

    assert enriched["width_px"] == 200
    assert enriched["height_px"] == 200
    assert enriched["area_px2"] == 40000
    assert enriched["size_category"] == "MUITO_GRANDE"
    assert enriched["aspect_ratio"] == 1.0
    assert enriched["relative_distance"] == 200 / 180
    assert 0.3 < enriched["position_x_rel"] < 0.7
    assert 0.3 < enriched["position_y_rel"] < 0.7
    assert enriched["is_near_center"] is True


def test_filter_detections():
    det1 = Detection(0, "person", 0.9, 0, 0, 100, 200)
    det2 = Detection(1, "chair", 0.8, 0, 0, 50, 50)
    det3 = Detection(0, "person", 0.7, 0, 0, 10, 10)

    # Filtro por área mínima
    result = filter_detections([det1, det2, det3], 640, 480, min_area=1000)
    assert len(result) == 2  # det1 e det2
    assert result[0]["detection"].label == "person"

    # Filtro por classe
    result = filter_detections([det1, det2], 640, 480, only_classes=["person"])
    assert len(result) == 1
    assert result[0]["detection"].label == "person"

    # Ordenação por distância relativa
    det_far = Detection(0, "person", 0.9, 0, 0, 10, 10)
    det_near = Detection(0, "person", 0.9, 0, 0, 100, 200)
    result = filter_detections([det_far, det_near], 640, 480)
    assert result[0]["detection"] == det_near  # maior altura → mais próximo


def test_summarize_analysis():
    det1 = Detection(0, "person", 0.9, 0, 0, 100, 200)
    det2 = Detection(1, "chair", 0.8, 0, 0, 50, 50)

    enriched = filter_detections([det1, det2], 640, 480)
    summary = summarize_analysis(enriched)

    assert summary["total"] == 2
    assert summary["classes"]["PERSON"] == 1
    assert summary["classes"]["CHAIR"] == 1
    assert summary["size_categories"]["GRANDE"] == 1
    assert summary["size_categories"]["PEQUENO"] == 1
    assert summary["avg_area"] > 0
    assert summary["near_center_count"] == 2

    # Lista vazia
    empty_summary = summarize_analysis([])
    assert empty_summary["total"] == 0
    assert empty_summary["avg_area"] == 0.0