"""Tests for image loading and normalization."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.image import (
    load_image_from_bytes,
    validate_image_array,
    resize_keep_aspect,
)


def _make_image(mode: str, size=(64, 48), color=None) -> bytes:
    if mode == "L":
        img = Image.new("L", size, color=128)
    elif mode == "RGBA":
        img = Image.new("RGBA", size, color=(30, 60, 90, 200))
    elif mode == "P":
        img = Image.new("P", size)
        img.putpalette([i % 256 for i in range(768)])
    else:
        img = Image.new("RGB", size, color=color or (10, 20, 30))
    buf = io.BytesIO()
    fmt = "PNG" if mode in ("RGBA", "P", "L") else "JPEG"
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_load_rgb_jpeg():
    data = _make_image("RGB")
    arr = load_image_from_bytes(data)
    assert arr is not None
    assert validate_image_array(arr)
    assert arr.shape[2] == 3
    assert arr.dtype == np.uint8


def test_load_png_rgba():
    data = _make_image("RGBA")
    arr = load_image_from_bytes(data)
    assert arr is not None
    assert validate_image_array(arr)


def test_load_grayscale():
    data = _make_image("L")
    arr = load_image_from_bytes(data)
    assert arr is not None
    assert validate_image_array(arr)


def test_load_palette():
    data = _make_image("P")
    arr = load_image_from_bytes(data)
    assert arr is not None
    assert validate_image_array(arr)


def test_load_webp():
    img = Image.new("RGB", (32, 32), (5, 10, 15))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    arr = load_image_from_bytes(buf.getvalue())
    assert arr is not None
    assert validate_image_array(arr)


def test_load_bmp():
    img = Image.new("RGB", (32, 32), (1, 2, 3))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    arr = load_image_from_bytes(buf.getvalue())
    assert arr is not None


def test_load_tiff():
    img = Image.new("RGB", (32, 32), (7, 8, 9))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    arr = load_image_from_bytes(buf.getvalue())
    assert arr is not None


def test_invalid_bytes():
    arr = load_image_from_bytes(b"not an image")
    assert arr is None


def test_empty_bytes():
    arr = load_image_from_bytes(b"")
    assert arr is None


def test_resize_keep_aspect():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    out, scale = resize_keep_aspect(img, max_side=200)
    assert out.shape[1] == 200
    assert abs(scale - 0.5) < 1e-6
