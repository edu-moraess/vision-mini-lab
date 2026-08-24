import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.spatial import get_spatial_region, compute_centroid, compute_density, get_region_counts
from src.detector import Detection


def test_get_spatial_region():
    region = get_spatial_region(320, 240, 640, 480)
    assert region == "CENTRO"

    region = get_spatial_region(50, 50, 640, 480)
    assert region == "SUPERIOR_ESQUERDA"

    region = get_spatial_region(600, 450, 640, 480)
    assert region == "INFERIOR_DIREITA"


def test_compute_centroid():
    detections = [
        {"center_x": 100, "center_y": 100},
        {"center_x": 200, "center_y": 200},
        {"center_x": 300, "center_y": 300},
    ]
    cx, cy = compute_centroid(detections)
    assert cx == 200.0
    assert cy == 200.0


def test_compute_centroid_empty():
    cx, cy = compute_centroid([])
    assert cx == 0.0
    assert cy == 0.0


def test_compute_density():
    detections = [{"center_x": 0, "center_y": 0}] * 10
    density = compute_density(detections, 640, 480)
    expected = 10 / (640 * 480 / 1_000_000)
    assert abs(density - expected) < 1e-6


def test_compute_density_empty():
    density = compute_density([], 640, 480)
    assert density == 0.0


def test_get_region_counts():
    det1 = Detection(0, "person", 0.9, 0, 0, 100, 100)
    det2 = Detection(1, "chair", 0.8, 300, 200, 400, 300)
    det3 = Detection(0, "person", 0.7, 500, 400, 600, 500)

    enriched = [
        {"center_x": 50, "center_y": 50, "detection": det1},
        {"center_x": 350, "center_y": 250, "detection": det2},
        {"center_x": 550, "center_y": 450, "detection": det3},
    ]

    counts = get_region_counts(enriched, 640, 480)
    assert counts["SUPERIOR_ESQUERDA"] == 1
    assert counts["CENTRO"] == 1
    assert counts["INFERIOR_DIREITA"] == 1