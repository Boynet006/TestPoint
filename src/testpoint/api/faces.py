"""Face check-in: detect the face in a frame and match it against today's list.

Models (OpenCV Zoo, run with OpenCV's DNN, fully offline):
  * YuNet 2023mar  - face detection   (MIT)
  * SFace 2021dec  - face recognition (Apache-2.0), 128-d embedding,
                     cosine similarity >= 0.363 means "same person" per the authors.

    python -m testpoint.api.faces --download data/models/face   # fetch + verify the models

Matching rule: the best candidate must clear ``threshold`` and beat the
runner-up by ``margin``; otherwise the examiner gets "not sure" and types the
bib. A wrong person checked in is worse than an extra 5 seconds of typing.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}
DETECTOR, RECOGNIZER = MODELS

DEFAULT_THRESHOLD = 0.363
DEFAULT_MARGIN = 0.05
MIN_FACE_PX = 40


class FaceError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


@dataclass
class Face:
    box: tuple[int, int, int, int]   # x, y, w, h in image pixels
    score: float
    embedding: np.ndarray            # (128,) float32, L2-normalised
    crop_jpeg: bytes                 # aligned 112x112 face, for the enrolment photo


@dataclass
class Match:
    candidate_id: int | None
    score: float                     # best cosine similarity
    runner_up: float
    confident: bool
    ranking: list[tuple[int, float]]  # (candidate_id, score), best first


def decode_image(data: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError("bad_image", "the file is not an image")
    return img


def normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def match(probe: np.ndarray, gallery: dict[int, list[np.ndarray]],
          threshold: float = DEFAULT_THRESHOLD, margin: float = DEFAULT_MARGIN) -> Match:
    """Best gallery identity for ``probe``. Each identity may have several templates (max wins)."""
    probe = normalise(probe)
    scores = []
    for cid, templates in gallery.items():
        if templates:
            scores.append((cid, max(float(np.dot(probe, normalise(t))) for t in templates)))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        return Match(None, 0.0, 0.0, False, [])
    best_id, best = scores[0]
    runner = scores[1][1] if len(scores) > 1 else -1.0
    confident = best >= threshold and (best - runner) >= margin
    return Match(best_id if confident else None, round(best, 3), round(runner, 3), confident,
                 [(c, round(s, 3)) for c, s in scores[:3]])


class FaceEngine:
    def __init__(self, model_dir: str | Path) -> None:
        import cv2

        self.model_dir = Path(model_dir)
        for name in MODELS:
            if not (self.model_dir / name).is_file():
                raise FaceError("models_missing",
                                f"face model {name} not found in {self.model_dir}; "
                                f"run: python -m testpoint.api.faces --download {self.model_dir}")
        self._detector = cv2.FaceDetectorYN.create(str(self.model_dir / DETECTOR), "", (320, 320),
                                                   0.8, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(self.model_dir / RECOGNIZER), "")

    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(img)
        return np.zeros((0, 15), np.float32) if faces is None else faces

    def face_from_bytes(self, data: bytes, require_single: bool = False) -> Face:
        return self.largest_face(decode_image(data), require_single)

    def largest_face(self, img: np.ndarray, require_single: bool = False) -> Face:
        import cv2

        # YuNet works best around 640 px. Try that first (close-up portraits, where the face
        # fills the frame, are missed at full size), then 1280 px for small faces in big frames.
        h, w = img.shape[:2]
        faces = np.zeros((0, 15), np.float32)
        for target in (640.0, 1280.0):
            scale = min(1.0, target / max(h, w))
            small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
            found = self.detect(small)
            if len(found):
                faces = found.copy()
                faces[:, :14] /= scale
                break
            if scale == 1.0:
                break
        if len(faces) == 0:
            raise FaceError("no_face", "no face found in the picture")
        big = [f for f in faces if min(f[2], f[3]) >= MIN_FACE_PX]
        if not big:
            raise FaceError("face_too_small", "the face is too small; move closer to the camera")
        if require_single and len(big) > 1:
            raise FaceError("multiple_faces", "more than one face in the picture")
        f = max(big, key=lambda r: r[2] * r[3])
        aligned = self._recognizer.alignCrop(img, f)
        emb = normalise(self._recognizer.feature(aligned))
        ok, jpg = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
        x, y, bw, bh = (int(round(v)) for v in f[:4])
        return Face((x, y, bw, bh), round(float(f[14]), 3), emb, jpg.tobytes() if ok else b"")


_engine: FaceEngine | None = None


def get_engine(model_dir: str | Path | None = None) -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine(model_dir or os.environ.get("TESTPOINT_FACE_MODELS", "data/models/face"))
    return _engine


def download(model_dir: str | Path) -> None:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in MODELS.items():
        path = d / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            print(f"ok      {name}")
            continue
        print(f"fetch   {name}")
        data = urllib.request.urlopen(url, timeout=120).read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise SystemExit(f"checksum mismatch for {name}: {got} (expected {sha})")
        path.write_bytes(data)
        print(f"saved   {name} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--download":
        download(sys.argv[2])
    else:
        raise SystemExit("usage: python -m testpoint.api.faces --download DIR")
