import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.events import EventEngine, ROI_ENTER, ROI_EXIT, LINE_CROSSED


def test_emit_and_count():
    engine = EventEngine()
    engine.emit(ROI_ENTER, object_id=1, class_name="person", frame_index=10)
    engine.emit(LINE_CROSSED, object_id=2, class_name="car", frame_index=11, metadata={"direction": "A_PARA_B"})

    assert len(engine) == 2
    assert engine.counts_by_type() == {ROI_ENTER: 1, LINE_CROSSED: 1}


def test_get_events_filters():
    engine = EventEngine()
    engine.emit(ROI_ENTER, object_id=1, frame_index=1)
    engine.emit(ROI_EXIT, object_id=1, frame_index=2)
    engine.emit(ROI_ENTER, object_id=2, frame_index=3)

    only_enter = engine.get_events(event_type=ROI_ENTER)
    assert len(only_enter) == 2

    only_obj1 = engine.get_events(object_id=1)
    assert len(only_obj1) == 2

    limited = engine.get_events(limit=1)
    assert len(limited) == 1


def test_history_is_bounded():
    engine = EventEngine(max_history=5)
    for i in range(10):
        engine.emit(ROI_ENTER, object_id=i, frame_index=i)
    assert len(engine) == 5


def test_reset_clears_state():
    engine = EventEngine()
    engine.emit(ROI_ENTER, object_id=1)
    engine.reset()
    assert len(engine) == 0
    assert engine.counts_by_type() == {}


def test_to_dict_roundtrip():
    engine = EventEngine()
    engine.emit(LINE_CROSSED, object_id=7, class_name="person", metadata={"direction": "A_PARA_B"})
    as_list = engine.to_list()
    assert as_list[0]["object_id"] == 7
    assert as_list[0]["metadata"]["direction"] == "A_PARA_B"
