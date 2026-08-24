import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyzer import classify_size, estimate_relative_distance, compute_aspect_ratio, get_confidence_range
from src.spatial import get_spatial_region, compute_centroid, compute_density
from src.report import compute_iou
from src.detector import Detection

def test_classify_size():
    assert classify_size(999) == "MUITO_PEQUENO"
    assert classify_size(1000) == "PEQUENO"
    assert classify_size(5000) == "MEDIO"
    assert classify_size(12000) == "GRANDE"
    assert classify_size(20001) == "MUITO_GRANDE"

def test_relative_distance():
    assert estimate_relative_distance(180) == 1.0
    assert estimate_relative_distance(90) == 0.5
    assert estimate_relative_distance(0) == 0.0

def test_aspect_ratio():
    assert compute_aspect_ratio(200, 100) == 2.0
    assert compute_aspect_ratio(100, 0) == 0.0

def test_confidence_range():
    assert get_confidence_range(0.92) == "MUITO_ALTA"
    assert get_confidence_range(0.80) == "ALTA"
    assert get_confidence_range(0.60) == "MODERADA"
    assert get_confidence_range(0.30) == "BAIXA"

def test_spatial_region():
    # centro na região central
    region = get_spatial_region(320, 240, 640, 480)
    assert region == "CENTRO"
    # canto superior esquerdo
    region = get_spatial_region(50, 50, 640, 480)
    assert region == "SUPERIOR_ESQUERDA"

def test_centroid():
    det1 = Detection(0, "person", 0.9, 0, 0, 100, 200)
    det2 = Detection(0, "person", 0.8, 100, 100, 300, 400)
    enriched = [
        {"center_x": 50, "center_y": 100},
        {"center_x": 200, "center_y": 250},
    ]
    cx, cy = compute_centroid(enriched)
    assert cx == 125.0
    assert cy == 175.0

def test_density():
    enriched = [{"center_x": 0, "center_y": 0}] * 10
    density = compute_density(enriched, 640, 480)
    expected = 10 / (640*480/1_000_000)
    assert abs(density - expected) < 1e-6

def test_iou():
    det1 = Detection(0, "a", 0.9, 0, 0, 100, 100)
    det2 = Detection(0, "b", 0.8, 50, 50, 150, 150)
    iou = compute_iou(det1, det2)
    # Intersecção: 50x50 = 2500, área1 = 10000, área2 = 10000, union = 17500, iou = 2500/17500 ≈ 0.142857
    assert abs(iou - 0.142857) < 1e-6
    # Sem sobreposição
    det3 = Detection(0, "c", 0.7, 200, 200, 300, 300)
    assert compute_iou(det1, det3) == 0.0