import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.detector import Detection
from src.report import generate_report
from src.quality import compute_quality_score


def test_generate_report_empty():
    report = generate_report([], 640, 480, 0.25)
    assert report['executive_summary']['total_objects'] == 0
    assert report['executive_summary']['summary'] == "Nenhum objeto detectado. O relatório está vazio."


def test_generate_report_single_detection():
    detections = [Detection(0, "person", 0.85, 100, 100, 200, 200)]
    report = generate_report(detections, 640, 480, 0.25)
    assert report['executive_summary']['total_objects'] == 1
    assert report['executive_summary']['unique_classes'] == 1
    assert report['detection_analysis']['total'] == 1
    assert report['scene_profile']['objects'] == 1


def test_generate_report_multiple_classes():
    detections = [
        Detection(0, "person", 0.85, 100, 100, 200, 200),
        Detection(1, "chair", 0.75, 300, 300, 400, 400),
        Detection(0, "person", 0.90, 500, 500, 600, 600),
    ]
    report = generate_report(detections, 640, 480, 0.25)
    assert report['executive_summary']['total_objects'] == 3
    assert report['executive_summary']['unique_classes'] == 2
    assert report['executive_summary']['dominant_class'] == "person"
    assert report['class_analysis']['person']['count'] == 2
    assert report['class_analysis']['chair']['count'] == 1


def test_quality_score():
    confidences = [0.95, 0.85, 0.75, 0.65, 0.55]
    score, level = compute_quality_score(
        confidences=confidences,
        total_detections=5,
        unique_classes=2,
        coverage_ratio=0.3,
        union_coverage=0.25,
        density=30.0,
    )
    assert 0.0 <= score <= 1.0
    assert level in ["HIGH", "MEDIUM", "LOW"]


def test_quality_score_empty():
    score, level = compute_quality_score(
        confidences=[],
        total_detections=0,
        unique_classes=0,
        coverage_ratio=0.0,
        union_coverage=0.0,
        density=0.0,
    )
    assert score == 0.0
    assert level == "LOW"


def test_spatial_analysis_interpretation():
    detections = [
        Detection(0, "person", 0.85, 10, 10, 100, 100),
        Detection(1, "chair", 0.75, 300, 10, 400, 100),
        Detection(0, "person", 0.90, 10, 300, 100, 400),
    ]
    report = generate_report(detections, 640, 480, 0.25)
    sa = report['spatial_analysis']
    assert sa['most_occupied_region'] is not None
    assert sa['interpretation'] is not None
    assert "região" in sa['interpretation'].lower() or "central" in sa['interpretation'].lower()


def test_overlap_analysis():
    # Dois objetos sobrepostos
    detections = [
        Detection(0, "person", 0.85, 100, 100, 200, 200),
        Detection(1, "chair", 0.75, 150, 150, 250, 250),
    ]
    report = generate_report(detections, 640, 480, 0.25)
    oa = report['overlap_analysis']
    assert oa['max_iou'] > 0.0
    assert len(oa['pairs']) > 0
    assert oa['interpretation'] is not None


def test_scene_profile():
    detections = [
        Detection(0, "person", 0.85, 100, 100, 200, 200),
        Detection(1, "chair", 0.75, 300, 300, 400, 400),
    ]
    report = generate_report(detections, 640, 480, 0.25)
    sp = report['scene_profile']
    assert sp['objects'] == 2
    assert sp['classes'] == 2
    assert sp['density'] in ["ALTA", "MÉDIA", "BAIXA"]
    assert sp['distribution'] in ["CENTRALIZADA", "PERIFÉRICA", "DIVERSA"]
    assert report['scene_interpretation'] is not None