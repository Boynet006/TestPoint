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
        import onnxruntime as ort
        from rtmlib import Body  # imported lazily: heavy and optional for tests

        if device.startswith("cuda"):
            # onnxruntime-gpu finds the CUDA/cuDNN DLLs shipped by the nvidia-* pip
            # packages only after this call; without it ORT silently falls back to CPU.
            ort.preload_dlls()
        self.device = device
        self.mode = mode
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
        keypoints, scores = self._body(frame_bgr)
        people = []
        for xy, conf in zip(keypoints, scores):
            people.append(Person(xy=np.asarray(xy), conf=np.asarray(conf)))
        return people


def make_backend(name: str = "rtmlib", **kwargs) -> PoseBackend:
    if name == "rtmlib":
        return RTMLibBackend(**kwargs)
    raise ValueError(f"unknown pose backend {name!r}")
