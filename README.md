# VISION MINI LAB

**Real-Time Visual Intelligence**

A compact experimental station for computer vision perception.

Built for stability, clarity and professional visual analysis — not as a tutorial demo.

---

## Overview

VISION MINI LAB is a focused laboratory for real-time visual intelligence. It combines object detection, multi-object tracking, motion analysis, relative thermal visualization, event detection and interactive spatial analytics into a single, coherent Streamlit application.

**Design principles**

- Stability over feature count
- Quality over complexity
- Clarity over visual noise
- Controlled dependencies

---

## Features

| Module | Capability |
|--------|------------|
| **Detection** | YOLOv8s (Ultralytics) with confidence / IoU control |
| **Tracking** | Built-in Ultralytics tracker with persistent IDs |
| **Motion** | dx, dy, speed (px/frame), 8-way direction, STATIONARY / MOVING |
| **Trajectory** | Short per-object history (≈ 60 points) |
| **ROI** | Single rectangular region — Objects Inside count |
| **Line Crossing** | Single virtual line with debounce |
| **Events** | OBJECT_ENTERED, OBJECT_EXITED, LINE_CROSSED, STARTED_MOVING, STOPPED |
| **Thermal** | Relative intensity visualization (Inferno / Magma / Turbo) + overlay |
| **Analytics** | Object activity, confidence, motion, trajectory, event timeline, thermal distribution, movement density heatmap |
| **Input** | Image (JPG, PNG, WEBP, BMP, TIFF) and Video (MP4 and common codecs) |

---

## Architecture

```
IMAGE / VIDEO
      ↓
YOLO DETECTION
      ↓
OBJECT TRACKING
      ↓
MOTION ANALYSIS
      ↓
THERMAL VISUALIZATION
      ↓
EVENT DETECTION
      ↓
VISUAL ANALYTICS
```

```
vision-mini-lab/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── perception/
│   │   ├── detector.py     # YOLO wrapper
│   │   └── tracker.py      # Ultralytics track + history
│   ├── motion/
│   │   └── motion.py       # speed, direction, state
│   ├── thermal/
│   │   └── thermal.py      # intensity, colormap, overlay, stats
│   ├── events/
│   │   └── events.py       # ROI, line, motion events
│   ├── analytics/
│   │   └── analytics.py    # Plotly charts
│   └── utils/
│       └── image.py        # load, EXIF, normalize, resize
└── tests/
    ├── test_image.py
    ├── test_motion.py
    ├── test_thermal.py
    └── test_events.py
```

---

## Installation

```bash
git clone https://github.com/edu-moraess/vision-mini-lab.git
cd vision-mini-lab

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> Uses `opencv-python-headless` only. Do not install `opencv-python` alongside it.

---

## Usage

```bash
streamlit run app.py
```

1. Upload an image or video.
2. Adjust confidence / IoU in the sidebar.
3. Toggle RGB / THERMAL / OVERLAY view.
4. Optionally enable ROI and a virtual line.
5. Inspect Object Intelligence, analytics charts and the event timeline.

---

## Computer Vision Pipeline

1. **Image normalization** — Pillow loads bytes → EXIF orientation correction → RGB uint8 H×W×3.
2. **Detection / Tracking** — YOLOv8s with Ultralytics `track(persist=True)`.
3. **Motion** — Centroid displacement between frames. Speed = √(dx² + dy²) in **pixels per frame**. Direction quantized to 8 cardinal/intercardinal bins. Small displacements are ignored.
4. **Events** — Stateless geometry checks + simple debounce for line crossings. Events are kept in a bounded in-memory list.
5. **Rendering** — Minimal overlays (ID, class, confidence). Optional short trajectory polylines.

---

## Thermal Vision

Thermal processing is **image-based relative intensity**, not radiometric temperature.

- Intensity is derived from luminance (no deep-learning thermal model).
- Colormaps: Inferno (default), Magma, Turbo.
- Modes: RGB · THERMAL · OVERLAY (with adjustable opacity).
- Statistics reported as **Relative Intensity** (mean, max, min, std, variance).
- **Never** presented as °C or Kelvin without proper radiometric calibration.

---

## Analytics

Charts are rendered vertically (one below the other) with Plotly:

1. **Object Activity** — counts by class
2. **Confidence** — per-object confidence bars
3. **Motion** — speed (px/frame) and direction
4. **Trajectory** — spatial paths with start / current markers
5. **Event Timeline** — temporal event markers
6. **Thermal Intensity Distribution** — histogram of relative intensity
7. **Movement Density** — 2D histogram of trajectory centers (conceptually separate from thermal)

Modebar is configured for a clean technical appearance (transparent background, no logo).

---

## Testing

```bash
pytest tests/ -v
```

Coverage focuses on:

- Image loader (JPG, PNG, WEBP, BMP, TIFF, grayscale, RGBA)
- Motion (dx, dy, speed, direction, stationary threshold)
- Thermal (normalization, statistics, overlay, ROI)
- Events (ROI enter/exit, line crossing, motion state changes)

---

## Limitations

- Speed is expressed in **pixels/frame**. Converting to m/s or km/h requires camera calibration and known geometry.
- Thermal values are **relative intensity**, not physical temperature.
- Video processing is frame-selectable (not continuous live playback) for stability on Streamlit Cloud.
- Single rectangular ROI and single virtual line only.
- Trajectory history is intentionally short (≈ 60 points) to bound memory.

---

## Roadmap (Future)

The following are **not** implemented in this version:

- Depth estimation
- Multi-camera support
- Advanced sensor fusion
- Custom model training
- Instance segmentation
- Pose estimation
- Edge hardware deployment
- Mobile / AR clients
- LLM / agent integration

Architecture is modular enough to accept future video sources (USB camera, thermal camera, edge device) without rewriting the core pipeline.

---

## License

MIT License — see [LICENSE](LICENSE).

---

*VISION MINI LAB — small in surface area, deliberate in engineering.*
