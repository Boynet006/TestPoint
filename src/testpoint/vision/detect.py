"""Person detection on its own, for the run cameras (no pose needed there).

Uses the same YOLOX person detector that rtmlib's ``Body`` loads for the
exercise stations, so no extra model: ``mode`` picks the preset
("lightweight", "balanced", "performance") and the ONNX file is shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def feet(self) -> tuple[float, float]:
        """Bottom centre: where the runner touches the ground."""
        return ((self.x0 + self.x1) / 2, self.y1)

    def iou(self, other: Box) -> float:
        ix = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, w: int, h: int) -> Box:
        return Box(max(0.0, self.x0), max(0.0, self.y0), min(float(w), self.x1), min(float(h), self.y1))


class YOLOXPersonDetector:
    def __init__(self, device: str = "cpu", mode: str = "balanced", min_height: float = 40.0) -> None:
        import onnxruntime as ort
        from rtmlib import Body
        from rtmlib.tools.object_detection import YOLOX

        if device.startswith("cuda"):
            ort.preload_dlls()
        preset = Body.MODE[mode]
        self.model = YOLOX(preset["det"], model_input_size=tuple(preset["det_input_size"]),
                           backend="onnxruntime", device=device)
        self.min_height = min_height
        if device.startswith("cuda") and "CUDAExecutionProvider" not in self.model.session.get_providers():
            raise RuntimeError(f"requested device {device!r} but ONNX Runtime is using "
                               f"{self.model.session.get_providers()}; follow the GPU setup steps in README.md")

    def detect(self, frame_bgr: np.ndarray) -> list[Box]:
        out = self.model(frame_bgr)
        boxes = out[0] if isinstance(out, tuple) else out
        h, w = frame_bgr.shape[:2]
        result = []
        for b in np.asarray(boxes).reshape(-1, 4):
            box = Box(*(float(v) for v in b)).clip(w, h)
            if box.height >= self.min_height and box.width > 0:
                result.append(box)
        return result
