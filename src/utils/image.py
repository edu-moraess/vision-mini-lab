"""
Image loading, validation and normalization utilities.
All images are converted to RGB uint8 H×W×3.
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


def load_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """
    Load image from raw bytes, validate, correct EXIF orientation,
    convert to RGB uint8 ndarray (H, W, 3).

    Returns None on failure.
    """
    if not data:
        logger.warning("Empty image bytes received")
        return None

    try:
        pil_img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        logger.warning("Unidentified image format")
        return None
    except Exception as exc:
        logger.exception("Failed to open image: %s", exc)
        return None

    try:
        # Correct EXIF orientation when present
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        # EXIF missing or malformed – continue without correction
        pass

    try:
        # Normalize mode
        if pil_img.mode == "RGBA":
            background = Image.new("RGB", pil_img.size, (0, 0, 0))
            background.paste(pil_img, mask=pil_img.split()[3])
            pil_img = background
        elif pil_img.mode == "P":
            pil_img = pil_img.convert("RGBA")
            background = Image.new("RGB", pil_img.size, (0, 0, 0))
            if pil_img.mode == "RGBA":
                background.paste(pil_img, mask=pil_img.split()[3])
            else:
                background.paste(pil_img)
            pil_img = background
        elif pil_img.mode == "L":
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        arr = np.asarray(pil_img, dtype=np.uint8)

        if arr.ndim != 3 or arr.shape[2] != 3:
            logger.warning("Unexpected array shape after conversion: %s", arr.shape)
            return None

        return arr
    except Exception as exc:
        logger.exception("Image normalization failed: %s", exc)
        return None


def load_image_from_path(path: str) -> Optional[np.ndarray]:
    """Load image from filesystem path."""
    try:
        with open(path, "rb") as f:
            return load_image_from_bytes(f.read())
    except Exception as exp:
        logger.exception("Failed to read image path %s: %s", path, exp)
        return None


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR for OpenCV operations."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_keep_aspect(
    image: np.ndarray,
    max_side: int = 1280,
) -> Tuple[np.ndarray, float]:
    """
    Resize image so that the longest side equals max_side.
    Returns resized image and scale factor (new / original).
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, scale


def validate_image_array(arr: np.ndarray) -> bool:
    """Basic sanity check for RGB uint8 array."""
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    if arr.dtype != np.uint8:
        return False
    return True
