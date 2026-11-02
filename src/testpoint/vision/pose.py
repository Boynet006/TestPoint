"""Pose estimation backends. The rest of the pipeline only sees ``PoseBackend``."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .keypoints import Person


class PoseBackend(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Person]:
        """Return all people found in the frame."""
        ...


class RTMLibBackend:
    """RTMPose (detector + pose) through rtmlib on ONNX Runtime.

    ``device`` is "cpu" or "cuda". ``mode`` is "lightweight", "balanced" or
    "performance" (rtmlib's model presets). Models are downloaded on first use.

    Finding people takes far longer than reading their pose (68 against 10 ms a frame on the
    station PC), and a person hardly moves from one frame to the next: the detector runs every
    ``detect_every`` frames (1 = always) and in between the boxes follow last frame's keypoints.
    """

    BOX_MARGIN = 0.25
    """How much larger than last frame's keypoints the box is, so the person stays inside it."""
    MIN_POINTS = 6
    """A person with fewer confident keypoints is lost: the detector looks again at once."""

    def __init__(self, device: str = "cpu", mode: str = "balanced", detect_every: int = 5) -> None:
        import onnxruntime as ort
        from rtmlib import Body  # imported lazily: heavy and optional for tests

        if device.startswith("cuda"):
            # onnxruntime-gpu finds the CUDA/cuDNN DLLs shipped by the nvidia-* pip
            # packages only after this call; without it ORT silently falls back to CPU.
            ort.preload_dlls()
        self.device = device
        self.mode = mode
        self.detect_every = max(1, int(detect_every))
        self._since_detect = 0
        self._boxes: np.ndarray | None = None
        self._body = Body(to_openpose=False, mode=mode, backend="onnxruntime", device=device)
        if device.startswith("cuda") and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError(
                f"requested device {device!r} but ONNX Runtime is using {self.providers}; "
                "follow the GPU setup steps in README.md"
            )

    @property
    def providers(self) -> list[str]:
        """Execution providers actually used by the detector and pose sessions."""
        out: list[str] = []
        for model in (self._body.det_model, self._body.pose_model):
            for prov in model.session.get_providers():
                if prov not in out:
                    out.append(prov)
        return out

    def detect(self, frame_bgr: np.ndarray) -> list[Person]:
        self._since_detect += 1
        if self._boxes is None or not len(self._boxes) or self._since_detect >= self.detect_every:
            self._boxes = np.asarray(self._body.det_model(frame_bgr))
            self._since_detect = 0
        if not len(self._boxes):
            return []
        keypoints, scores = self._body.pose_model(frame_bgr, bboxes=self._boxes)
        self._boxes = self._follow(keypoints, scores, frame_bgr.shape)
        people = []
        for xy, conf in zip(keypoints, scores):
            people.append(Person(xy=np.asarray(xy), conf=np.asarray(conf)))
        return people


    def _follow(self, keypoints: np.ndarray, scores: np.ndarray, shape: tuple) -> np.ndarray | None:
        """Next frame's boxes from this frame's keypoints; None when someone was lost."""
        h, w = shape[:2]
        boxes = []
        for xy, conf in zip(keypoints, scores):
            good = np.asarray(xy)[np.asarray(conf) >= 0.3]
            if len(good) < self.MIN_POINTS:
                return None
            x0, y0 = good.min(axis=0)
            x1, y1 = good.max(axis=0)
            mx, my = (x1 - x0) * self.BOX_MARGIN, (y1 - y0) * self.BOX_MARGIN
            mx, my = max(mx, 0.04 * w), max(my, 0.04 * h)    # a person seen edge-on is a thin box
            boxes.append([max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)])
        return np.asarray(boxes, dtype=np.float32)


def make_backend(name: str = "rtmlib", **kwargs) -> PoseBackend:
    if name == "rtmlib":
        return RTMLibBackend(**kwargs)
    raise ValueError(f"unknown pose backend {name!r}")
