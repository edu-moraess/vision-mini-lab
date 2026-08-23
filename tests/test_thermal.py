"""Tests for thermal processing."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.thermal.thermal import (
    to_intensity,
    apply_colormap,
    overlay_thermal,
    compute_stats,
    compute_roi_stats,
    process_thermal,
)


def test_to_intensity_rgb():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = [100, 150, 200]
    intens = to_intensity(img)
    assert intens.shape == (10, 10)
    assert intens.dtype == np.float32


def test_to_intensity_gray():
    img = np.full((8, 8), 50, dtype=np.uint8)
    intens = to_intensity(img)
    assert intens.shape == (8, 8)


def test_apply_colormap():
    intens = np.linspace(0, 255, 100).reshape(10, 10).astype(np.float32)
    colored = apply_colormap(intens, "inferno")
    assert colored.shape == (10, 10, 3)
    assert colored.dtype == np.uint8


def test_overlay():
    rgb = np.full((20, 20, 3), 50, dtype=np.uint8)
    thermal = np.full((20, 20, 3), 200, dtype=np.uint8)
    out = overlay_thermal(rgb, thermal, opacity=0.5)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_compute_stats():
    intens = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    stats = compute_stats(intens)
    assert stats.mean == pytest.approx(25.0)
    assert stats.maximum == pytest.approx(40.0)
    assert stats.minimum == pytest.approx(10.0)


def test_roi_stats():
    intens = np.arange(100, dtype=np.float32).reshape(10, 10)
    stats = compute_roi_stats(intens, (2, 2, 5, 5))
    assert stats is not None
    assert stats.mean > 0


def test_roi_invalid():
    intens = np.zeros((10, 10), dtype=np.float32)
    assert compute_roi_stats(intens, (8, 8, 3, 3)) is None


def test_process_thermal_modes():
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    for mode in ("rgb", "thermal", "overlay"):
        display, intens, stats = process_thermal(img, mode=mode)
        assert display.shape[2] == 3
        assert intens.shape == (32, 32)
        assert stats.mean >= 0
