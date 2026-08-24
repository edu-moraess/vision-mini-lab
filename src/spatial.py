import numpy as np
from typing import Dict, List, Optional, Tuple

from src.events import LINE_CROSSED, ROI_ENTER, ROI_EXIT

def get_spatial_region(center_x: float, center_y: float, image_width: int, image_height: int) -> str:
    if image_width <= 0 or image_height <= 0:
        return "DESCONHECIDO"
    x = center_x / image_width
    y = center_y / image_height
    row = 0 if y < 1/3 else (1 if y < 2/3 else 2)
    col = 0 if x < 1/3 else (1 if x < 2/3 else 2)
    regions = [
        ["SUPERIOR_ESQUERDA", "SUPERIOR_CENTRO", "SUPERIOR_DIREITA"],
        ["CENTRO_ESQUERDA", "CENTRO", "CENTRO_DIREITA"],
        ["INFERIOR_ESQUERDA", "INFERIOR_CENTRO", "INFERIOR_DIREITA"],
    ]
    return regions[row][col]

def compute_centroid(detections: List[Dict]) -> Tuple[float, float]:
    if not detections:
        return (0.0, 0.0)
    cx = sum(item["center_x"] for item in detections) / len(detections)
    cy = sum(item["center_y"] for item in detections) / len(detections)
    return (cx, cy)

def compute_density(detections: List[Dict], image_width: int, image_height: int) -> float:
    if image_width <= 0 or image_height <= 0 or not detections:
        return 0.0
    area_pixels = image_width * image_height
    return len(detections) / (area_pixels / 1_000_000)

def get_region_counts(detections: List[Dict], image_width: int, image_height: int) -> Dict[str, int]:
    counts = {}
    for item in detections:
        region = get_spatial_region(item["center_x"], item["center_y"], image_width, image_height)
        counts[region] = counts.get(region, 0) + 1
    return counts

def compute_union_coverage(detections: List[Dict], image_width: int, image_height: int) -> float:
    """
    Calcula a área da união das bounding boxes.
    Útil para evitar dupla contagem em regiões sobrepostas.
    """
    if not detections or image_width <= 0 or image_height <= 0:
        return 0.0

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for item in detections:
        det = item["detection"]
        x1 = max(0, int(det.x1))
        y1 = max(0, int(det.y1))
        x2 = min(image_width, int(det.x2))
        y2 = min(image_height, int(det.y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    union_area = np.sum(mask)
    return union_area / (image_width * image_height)


class RegionOfInterest:
    """Região retangular de interesse (ROI), em coordenadas de pixel.

    Calcula, quadro a quadro, quais objetos estão dentro, quem entrou, quem
    saiu, a ocupação atual e o tempo médio de permanência (dwell time).
    Não implementa um editor geométrico complexo — a ROI é um retângulo
    simples, suficiente para esta versão.
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float, name: str = "ROI"):
        self.x1, self.x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        self.y1, self.y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        self.name = name
        self._inside_since: Dict[int, float] = {}
        self._completed_dwell_times: List[float] = []

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def update(
        self,
        tracked_objects: List[Dict],
        timestamp: float,
        event_engine=None,
        frame_index: Optional[int] = None,
    ) -> Dict:
        """Atualiza a ROI para o frame atual e retorna um snapshot.

        ``tracked_objects``: lista de dicts com ao menos track_id, class_name,
        center_x, center_y. Objetos sem track_id são ignorados (a ROI
        depende de identidade estável para calcular entradas/saídas).
        """
        current_inside: Dict[int, Optional[str]] = {}
        for obj in tracked_objects:
            tid = obj.get("track_id")
            if tid is None:
                continue
            if self.contains(obj["center_x"], obj["center_y"]):
                current_inside[tid] = obj.get("class_name")

        previous_ids = set(self._inside_since.keys())
        current_ids = set(current_inside.keys())
        entered_ids = current_ids - previous_ids
        exited_ids = previous_ids - current_ids

        new_events: List[Dict] = []

        for tid in entered_ids:
            self._inside_since[tid] = timestamp
            class_name = current_inside.get(tid)
            metadata = {"roi": self.name}
            if event_engine is not None:
                event_engine.emit(
                    ROI_ENTER, object_id=tid, class_name=class_name,
                    frame_index=frame_index, timestamp=timestamp, metadata=metadata,
                )
            new_events.append({
                "event_type": ROI_ENTER, "object_id": tid, "class_name": class_name,
                "timestamp": timestamp, "frame_index": frame_index, "metadata": metadata,
            })

        for tid in exited_ids:
            entry_time = self._inside_since.pop(tid, None)
            if entry_time is not None:
                self._completed_dwell_times.append(timestamp - entry_time)
            metadata = {"roi": self.name}
            if event_engine is not None:
                event_engine.emit(
                    ROI_EXIT, object_id=tid, class_name=None,
                    frame_index=frame_index, timestamp=timestamp, metadata=metadata,
                )
            new_events.append({
                "event_type": ROI_EXIT, "object_id": tid, "class_name": None,
                "timestamp": timestamp, "frame_index": frame_index, "metadata": metadata,
            })

        avg_dwell = (
            sum(self._completed_dwell_times) / len(self._completed_dwell_times)
            if self._completed_dwell_times
            else 0.0
        )

        return {
            "inside_ids": sorted(current_ids),
            "entered_ids": sorted(entered_ids),
            "exited_ids": sorted(exited_ids),
            "occupancy": len(current_ids),
            "average_dwell_time": avg_dwell,
            "events": new_events,
        }

    def reset(self) -> None:
        self._inside_since.clear()
        self._completed_dwell_times.clear()


class LineCrossingDetector:
    """Detecta cruzamentos de uma linha virtual por objetos rastreados.

    Usa o sinal do produto vetorial entre o vetor da linha e o vetor até o
    objeto para determinar de que lado ele está. Um cruzamento é contado
    apenas quando o lado muda E o objeto respeita um debounce mínimo de
    frames, evitando contagem duplicada por ruído bem em cima da linha.
    """

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        name: str = "LINE",
        forward_label: str = "A_PARA_B",
        backward_label: str = "B_PARA_A",
        debounce_frames: int = 10,
    ):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.name = name
        self.forward_label = forward_label
        self.backward_label = backward_label
        self.debounce_frames = debounce_frames
        self._last_side: Dict[int, int] = {}
        self._last_cross_frame: Dict[int, int] = {}
        self.crossing_counts = {"forward": 0, "backward": 0}

    def _side(self, x: float, y: float) -> int:
        val = (self.x2 - self.x1) * (y - self.y1) - (self.y2 - self.y1) * (x - self.x1)
        if val > 1e-6:
            return 1
        if val < -1e-6:
            return -1
        return 0

    def update(
        self,
        tracked_objects: List[Dict],
        timestamp: float,
        event_engine=None,
        frame_index: Optional[int] = None,
    ) -> List[Dict]:
        """Verifica cruzamentos no frame atual e retorna a lista de ocorrências."""
        crossings: List[Dict] = []
        current_frame = frame_index if frame_index is not None else 0

        for obj in tracked_objects:
            tid = obj.get("track_id")
            if tid is None:
                continue
            side = self._side(obj["center_x"], obj["center_y"])
            if side == 0:
                continue

            previous_side = self._last_side.get(tid)
            self._last_side[tid] = side
            if previous_side is None or previous_side == side:
                continue

            last_cross = self._last_cross_frame.get(tid, -10_000)
            if current_frame - last_cross < self.debounce_frames:
                continue
            self._last_cross_frame[tid] = current_frame

            direction = self.forward_label if side > 0 else self.backward_label
            if side > 0:
                self.crossing_counts["forward"] += 1
            else:
                self.crossing_counts["backward"] += 1

            crossing = {
                "track_id": tid,
                "class_name": obj.get("class_name"),
                "direction": direction,
                "timestamp": timestamp,
                "frame_index": frame_index,
            }
            crossings.append(crossing)

            if event_engine is not None:
                event_engine.emit(
                    LINE_CROSSED,
                    object_id=tid,
                    class_name=obj.get("class_name"),
                    frame_index=frame_index,
                    metadata={"line": self.name, "direction": direction},
                )

        return crossings

    def prune(self, active_track_ids) -> None:
        """Remove estado de objetos que não estão mais ativos (evita crescimento indefinido)."""
        active = set(active_track_ids)
        for tid in list(self._last_side.keys()):
            if tid not in active:
                self._last_side.pop(tid, None)
                self._last_cross_frame.pop(tid, None)

    def reset(self) -> None:
        self._last_side.clear()
        self._last_cross_frame.clear()
        self.crossing_counts = {"forward": 0, "backward": 0}