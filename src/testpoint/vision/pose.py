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
    """

    def __init__(self, device: str = "cpu", mode: str = "balanced") -> None:
        from rtmlib import Body  # imported lazily: heavy and optional for tests

        self._body = Body(to_openpose=False, mode=mode, backend="onnxruntime", device=device)

    def detect(self, frame_bgr: np.ndarray) -> list[Person]:
        keypoints, scores = self._body(frame_bgr)
        people = []
        for xy, conf in zip(keypoints, scores):
            people.append(Person(xy=np.asarray(xy), conf=np.asarray(conf)))
        return people


def make_backend(name: str = "rtmlib", **kwargs) -> PoseBackend:
    if name == "rtmlib":
        return RTMLibBackend(**kwargs)
    raise ValueError(f"unknown pose backend {name!r}")
