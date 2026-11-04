"""The face models on the GPU: YuNet (detection) and SFace (recognition) on ONNX Runtime CUDA.

The same two models as ``api.faces.FaceEngine`` (OpenCV DNN, CPU), so templates enrolled with
either compare with each other. On the GPU a 640x640 detection takes a few milliseconds whatever
the camera's resolution: a big frame is looked at whole (letterboxed to 640) and, when that finds
no face, in overlapping 640 tiles of a 1280-wide copy, as the CPU engine looks at 640 then 1280.
Faces are aligned with OpenCV's own ``alignCrop`` (the reference 5-point alignment), so the
embedding is the same as the CPU engine's up to rounding.

``api.faces.get_engine`` picks this engine when ONNX Runtime has CUDA, else the CPU one.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

SIZE = 640                      # YuNet's input (the ONNX graph is fixed at 1x3x640x640)
STRIDES = (8, 16, 32)
SCORE, NMS = 0.8, 0.3           # as the CPU engine (cv2.FaceDetectorYN.create(..., 0.8, 0.3))


def cuda_available() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def _session(path: Path, device: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3          # the models carry harmless initializer warnings
    if device == "cuda":
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()           # the CUDA / cuDNN DLLs inside the nvidia pip packages
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    s = ort.InferenceSession(str(path), opts, providers=providers)
    if device == "cuda" and "CUDAExecutionProvider" not in s.get_providers():
        raise RuntimeError("ONNX Runtime fell back to the CPU")
    return s


class YuNetOrt:
    """YuNet on ONNX Runtime: rows like cv2.FaceDetectorYN (x, y, w, h, 5 landmarks x/y, score)."""

    def __init__(self, path: Path, device: str = "cuda") -> None:
        self.s = _session(path, device)
        self.inp = self.s.get_inputs()[0].name
        self.out = [o.name for o in self.s.get_outputs()]
        # anchor grid per stride (cell centres in input pixels)
        self.grid = {}
        for st in STRIDES:
            n = SIZE // st
            r, c = np.divmod(np.arange(n * n), n)
            self.grid[st] = (c.astype(np.float32), r.astype(np.float32))

    def _run(self, square: np.ndarray) -> np.ndarray:
        """Faces in one 640x640 BGR image, in its own pixels."""
        blob = square.astype(np.float32).transpose(2, 0, 1)[None]
        outs = dict(zip(self.out, self.s.run(self.out, {self.inp: blob})))
        rows = []
        for st in STRIDES:
            cls = outs[f"cls_{st}"][0, :, 0]
            obj = outs[f"obj_{st}"][0, :, 0]
            score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            keep = score >= SCORE
            if not keep.any():
                continue
            bb = outs[f"bbox_{st}"][0][keep]
            kp = outs[f"kps_{st}"][0][keep]
            cx, cy = self.grid[st][0][keep], self.grid[st][1][keep]
            x = (cx + bb[:, 0]) * st
            y = (cy + bb[:, 1]) * st
            w = np.exp(bb[:, 2]) * st
            h = np.exp(bb[:, 3]) * st
            k = np.empty((len(x), 10), np.float32)
            k[:, 0::2] = (kp[:, 0::2] + cx[:, None]) * st
            k[:, 1::2] = (kp[:, 1::2] + cy[:, None]) * st
            rows.append(np.column_stack([x - w / 2, y - h / 2, w, h, k, score[keep]]).astype(np.float32))
        return np.concatenate(rows) if rows else np.zeros((0, 15), np.float32)

    def _view(self, img: np.ndarray, x0: int, y0: int, scale: float, out: list) -> None:
        """Detect in ``img`` (already scaled by ``scale``) placed at (x0, y0) on a 640 canvas; map back."""
        h, w = img.shape[:2]
        canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
        canvas[:h, :w] = img
        f = self._run(canvas)
        if len(f):
            _place(f, x0, y0, scale)
            out.append(f)

    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        found: list[np.ndarray] = []
        s = min(1.0, SIZE / max(h, w))
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        self._view(small, 0, 0, s, found)
        if not found and max(h, w) > SIZE:
            # small faces in a big frame: overlapping 640 tiles of a copy at most 1280 wide
            s2 = min(1.0, 2 * SIZE / max(h, w))
            big = cv2.resize(img, (round(w * s2), round(h * s2)), interpolation=cv2.INTER_AREA) if s2 < 1 else img
            bh, bw = big.shape[:2]
            for ty in _starts(bh):
                for tx in _starts(bw):
                    tile = big[ty:ty + SIZE, tx:tx + SIZE]
                    f = []
                    self._view(tile, 0, 0, 1.0, f)
                    for a in f:
                        _place(a, tx, ty, s2)
                        found.append(a)
        if not found:
            return np.zeros((0, 15), np.float32)
        f = np.concatenate(found)
        keep = cv2.dnn.NMSBoxes(f[:, :4].tolist(), f[:, 14].tolist(), SCORE, NMS)
        return f[np.array(keep, dtype=int).reshape(-1)] if len(keep) else np.zeros((0, 15), np.float32)


X_COLS, Y_COLS, SIZE_COLS = [0, 4, 6, 8, 10, 12], [1, 5, 7, 9, 11, 13], [2, 3]


def _place(f: np.ndarray, x0: float, y0: float, scale: float) -> None:
    """Rows found in a tile or a scaled copy, back in the picture's own pixels: the box corner and the
    landmarks move by the tile's origin, the box's width and height only scale (adding the tile's origin
    to them made a 47-pixel face 721 pixels wide)."""
    f[:, X_COLS] = (f[:, X_COLS] + x0) / scale
    f[:, Y_COLS] = (f[:, Y_COLS] + y0) / scale
    f[:, SIZE_COLS] = f[:, SIZE_COLS] / scale


def _starts(n: int) -> list[int]:
    """Tile origins covering ``n`` pixels with 640 tiles overlapping by at least a quarter."""
    if n <= SIZE:
        return [0]
    k = math.ceil((n - SIZE) / (SIZE * 0.75)) + 1
    return [round(i * (n - SIZE) / (k - 1)) for i in range(k)]


class SFaceOrt:
    """SFace on ONNX Runtime: a 128-d embedding per aligned 112x112 face (BGR, as alignCrop gives)."""

    def __init__(self, path: Path, device: str = "cuda") -> None:
        self.s = _session(path, device)
        self.inp = self.s.get_inputs()[0].name

    def features(self, aligned: list[np.ndarray]) -> np.ndarray:
        """Embeddings for several aligned faces (one run each: the graph has batch 1)."""
        out = np.zeros((len(aligned), 128), np.float32)
        for i, a in enumerate(aligned):
            blob = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None]
            out[i] = self.s.run(None, {self.inp: blob})[0].reshape(-1)
        return out
