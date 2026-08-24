import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spatial import RegionOfInterest, LineCrossingDetector
from src.events import EventEngine, ROI_ENTER, ROI_EXIT, LINE_CROSSED


def test_roi_enter_and_exit_with_dwell_time():
    engine = EventEngine()
    roi = RegionOfInterest(100, 100, 200, 200, name="ZONA")

    # Fora da ROI.
    snap = roi.update(
        [{"track_id": 1, "class_name": "person", "center_x": 10, "center_y": 10}],
        timestamp=0.0, event_engine=engine, frame_index=1,
    )
    assert snap["occupancy"] == 0

    # Entra na ROI.
    snap = roi.update(
        [{"track_id": 1, "class_name": "person", "center_x": 150, "center_y": 150}],
        timestamp=1.0, event_engine=engine, frame_index=2,
    )
    assert snap["occupancy"] == 1
    assert snap["entered_ids"] == [1]
    assert engine.counts_by_type().get(ROI_ENTER) == 1

    # Sai da ROI (some da lista de objetos rastreados).
    snap = roi.update([], timestamp=3.0, event_engine=engine, frame_index=3)
    assert snap["occupancy"] == 0
    assert snap["exited_ids"] == [1]
    assert engine.counts_by_type().get(ROI_EXIT) == 1
    assert abs(snap["average_dwell_time"] - 2.0) < 1e-9


def test_roi_ignores_objects_without_track_id():
    roi = RegionOfInterest(0, 0, 100, 100)
    snap = roi.update([{"track_id": None, "class_name": "person", "center_x": 50, "center_y": 50}], timestamp=0.0)
    assert snap["occupancy"] == 0


def test_line_crossing_counts_and_debounce():
    engine = EventEngine()
    line = LineCrossingDetector(0, 100, 200, 100, name="L1", debounce_frames=5)

    # Acima da linha.
    crossings = line.update(
        [{"track_id": 5, "class_name": "car", "center_x": 50, "center_y": 50}],
        timestamp=0.0, event_engine=engine, frame_index=1,
    )
    assert crossings == []

    # Cruza para baixo da linha -> conta 1 cruzamento.
    crossings = line.update(
        [{"track_id": 5, "class_name": "car", "center_x": 50, "center_y": 150}],
        timestamp=1.0, event_engine=engine, frame_index=2,
    )
    assert len(crossings) == 1
    assert engine.counts_by_type().get(LINE_CROSSED) == 1

    # Continua do mesmo lado -> não conta de novo.
    crossings = line.update(
        [{"track_id": 5, "class_name": "car", "center_x": 50, "center_y": 160}],
        timestamp=2.0, event_engine=engine, frame_index=3,
    )
    assert crossings == []


def test_line_crossing_prune_removes_inactive():
    line = LineCrossingDetector(0, 100, 200, 100)
    line.update([{"track_id": 1, "class_name": "car", "center_x": 50, "center_y": 50}], timestamp=0.0, frame_index=1)
    line.prune(active_track_ids=[])
    assert line._last_side == {}
