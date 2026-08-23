"""
Thermal visualization module.
Operates on relative intensity derived from grayscale / single-channel data.
Does NOT invent physical temperature units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": cv2.COLORMAP_TURBO,
}


@dataclass
class ThermalStats:
    mean: float
    maximum: float
    minimum: float
    std: float
    variance: float


def to_intensity(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to relative intensity map (float32 0-255 range).
    Uses luminance approximation.
    """
    if image_rgb.ndim == 2:
        return image_rgb.astype(np.float32)
    # ITU-R BT.601 luminance
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    intensity = 0.299 * r + 0.587 * g + 0.114 * b
    return intensity


def apply_colormap(
    intensity: np.ndarray,
    colormap: str = "inferno",
) -> np.ndarray:
    """
    Apply OpenCV colormap to intensity map.
    Returns RGB uint8 image.
    """
    cmap = COLORMAPS.get(colormap.lower(), cv2.COLORMAP_INFERNO)
    # Normalize to 0-255 uint8
    norm = np.clip(intensity, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(norm, cmap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def overlay_thermal(
    rgb: np.ndarray,
    thermal_rgb: np.ndarray,
    opacity: float = 0.5,
) -> np.ndarray:
    """
    Alpha blend thermal visualization over RGB.
    opacity in [0, 1].
    """
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if rgb.shape != thermal_rgb.shape:
        thermal_rgb = cv2.resize(
            thermal_rgb,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    blended = cv2.addWeighted(
        rgb.astype(np.float32),
        1.0 - opacity,
        thermal_rgb.astype(np.float32),
        opacity,
        0.0,
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_stats(intensity: np.ndarray) -> ThermalStats:
    """Global statistics on intensity map."""
    flat = intensity.ravel()
    return ThermalStats(
        mean=float(np.mean(flat)),
        maximum=float(np.max(flat)),
        minimum=float(np.min(flat)),
        std=float(np.std(flat)),
        variance=float(np.var(flat)),
    )


def compute_roi_stats(
    intensity: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Optional[ThermalStats]:
    """
    ROI = (x1, y1, x2, y2) in pixel coordinates.
    Returns None if ROI is invalid.
    """
    h, w = intensity.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    region = intensity[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return compute_stats(region)


def process_thermal(
    image_rgb: np.ndarray,
    mode: str = "thermal",
    colormap: str = "inferno",
    opacity: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, ThermalStats]:
    """
    Full thermal pipeline.
    mode: "rgb" | "thermal" | "overlay"
    Returns: (display_image, intensity_map, global_stats)
    """
    intensity = to_intensity(image_rgb)
    stats = compute_stats(intensity)
    thermal_rgb = apply_colormap(intensity, colormap)

    mode = mode.lower()
    if mode == "rgb":
        display = image_rgb.copy()
    elif mode == "overlay":
        display = overlay_thermal(image_rgb, thermal_rgb, opacity)
    else:
        display = thermal_rgb

    return display, intensity, stats
