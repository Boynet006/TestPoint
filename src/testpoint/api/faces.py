"""Face check-in: detect the face in a frame and match it against today's list.

Models (OpenCV Zoo, run with OpenCV's DNN, fully offline):
  * YuNet 2023mar  - face detection   (MIT)
  * SFace 2021dec  - face recognition (Apache-2.0), 128-d embedding,
                     cosine similarity >= 0.363 means "same person" per the authors.
  * AdaFace IR-101 - face recognition, 512-d (code MIT; trained on WebFace12M, whose licence
                     the authors ask to follow), used instead of SFace when its file is there
                     (scripts/build_adaface.py makes it). Far better on poor pictures: on LFW
                     shrunk and blurred as the finish camera sees a runner, 96.9 % right against
                     SFace's 90.3 %, and 88 % of true matches found at 1 wrong in 1000 (SFace 46 %).
                     Its scores are put on SFace's scale (``common_score``), so every threshold,
                     setting and score shown keeps its meaning, at the same rate of wrong matches.

    python -m testpoint.api.faces --download data/models/face   # fetch + verify the models

Matching rule: the best candidate must clear ``threshold`` and beat the
runner-up by ``margin``; otherwise the examiner gets "not sure" and types the
bib. A wrong person checked in is worse than an extra 5 seconds of typing.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
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
ADAFACE = "face_recognition_adaface_ir101.onnx"

# AdaFace score -> the SFace score with the same rate of wrong matches (different people), measured
# on all of LFW as the cameras compare: a clean photo against one shrunk, blurred and compressed as
# the finish camera sees a runner (13,000 faces, 3 million pairs of different people). Above
# SFace 0.55, where wrong matches no longer happen, the same rate of right matches instead.
ADA_TO_SFACE = [(-1.0, -1.0), (0.095, 0.20), (0.128, 0.25), (0.166, 0.30), (0.192, 0.33), (0.223, 0.363),
                (0.264, 0.40), (0.366, 0.45), (0.50, 0.50), (0.60, 0.60), (1.0, 1.0)]


def common_score(raw: float, dim: int) -> float:
    """A similarity on SFace's scale, whichever model made the embeddings (by their size)."""
    if dim != 512:
        return raw
    ys, xs = zip(*ADA_TO_SFACE)
    return float(np.interp(raw, ys, xs))

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
    crop_jpeg: bytes                 # the photo people see: head and shoulders, 4:5 (see _largest_face)
    yaw: float = 0.0                 # head turn (head_yaw): about 0 facing the camera
    quality: float | None = None     # how good a template this picture makes (see ``quality``), 0..1; None = not measured
    issues: tuple[str, ...] = ()     # what lowers it: too_small, blurry, too_dark, too_bright, low_contrast, unclear


@dataclass
class Match:
    candidate_id: int | None
    score: float                     # best cosine similarity
    runner_up: float
    confident: bool
    ranking: list[tuple[int, float]]  # (candidate_id, score), best first


def head_yaw(row: np.ndarray) -> float:
    """Head turn from a YuNet detection row (x, y, w, h, right eye, left eye, nose, mouth...):
    how far the nose tip sits from the midpoint between the eyes, in eye distances, in the
    raw (not mirrored) image. About 0 looking at the camera; positive when the nose is to the
    right in the image, which is the person turning to their own left."""
    rex, rey, lex, ley, nx = (float(v) for v in row[4:9])
    eye_dist = ((lex - rex) ** 2 + (ley - rey) ** 2) ** 0.5
    if eye_dist <= 1e-6:
        return 0.0
    return (nx - (rex + lex) / 2) / eye_dist


def head_yaw_box(row: np.ndarray) -> float:
    """Head turn from where the nose tip sits across the face box: 0 in the middle, towards
    +-0.5 at the edges. Same sign as ``head_yaw``. Keeps growing on strong turns, where the
    far eye is hidden and the eye-based measure under-reads."""
    x, w, nx = float(row[0]), float(row[2]), float(row[8])
    return (nx - (x + w / 2)) / w if w > 0 else 0.0


def _ramp(v: float, lo: float, hi: float) -> float:
    """0 at ``lo`` (or below), 1 at ``hi`` (or above), straight in between."""
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


QUALITY_OK = 0.5
"""Below this an enrolment picture is refused with its issues (and the person asked again)."""


def quality(aligned: np.ndarray, row: np.ndarray) -> dict:
    """How good a recognition template a face makes, 0..1, and what lowers it. The geometric mean
    of: size (40 px refused, 100 px full marks), sharpness (Laplacian variance of the aligned
    112-px face), exposure (mean light 70-190), contrast, and the detector's confidence. A head
    turn is not counted here: enrolment asks for turns on purpose (see head_yaw)."""
    import cv2

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    side = float(min(row[2], row[3]))
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean, std = float(gray.mean()), float(gray.std())
    parts = {
        "too_small": _ramp(side, 40, 100),
        "blurry": _ramp(sharp, 15, 90),
        "too_dark": _ramp(mean, 35, 70),
        "too_bright": 1.0 - _ramp(mean, 190, 235),
        "low_contrast": _ramp(std, 12, 30),
        "unclear": _ramp(float(row[14]), 0.75, 0.9),
    }
    score = float(np.prod([max(v, 1e-3) for v in parts.values()]) ** (1 / len(parts)))
    issues = tuple(k for k, v in parts.items() if v < 0.5)
    return {"score": round(score, 3), "issues": issues, "sharpness": round(sharp, 1), "light": round(mean, 1)}


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
    """Best gallery identity for ``probe``. Each identity may have several templates (max wins).
    Templates made by another model (another size) are left out; scores are on SFace's scale."""
    probe = normalise(probe)
    scores = []
    for cid, templates in gallery.items():
        same = [t for t in templates if np.asarray(t).size == probe.size]
        if same:
            raw = max(float(np.dot(probe, normalise(t))) for t in same)
            scores.append((cid, common_score(raw, probe.size)))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        return Match(None, 0.0, 0.0, False, [])
    best_id, best = scores[0]
    runner = scores[1][1] if len(scores) > 1 else -1.0
    confident = best >= threshold and (best - runner) >= margin
    return Match(best_id if confident else None, round(best, 3), round(runner, 3), confident,
                 [(c, round(s, 3)) for c, s in scores[:3]])


class FaceEngine:
    """Detection and recognition. ``device`` "auto": on the GPU (ONNX Runtime CUDA, see
    vision/face_gpu.py) when there is one, else OpenCV on the CPU; "cpu" / "cuda" to choose."""

    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        import cv2

        self.model_dir = Path(model_dir)
        for name in MODELS:
            if not (self.model_dir / name).is_file():
                raise FaceError("models_missing",
                                f"face model {name} not found in {self.model_dir}; "
                                f"run: python -m testpoint.api.faces --download {self.model_dir}")
        self._detector = cv2.FaceDetectorYN.create(str(self.model_dir / DETECTOR), "", (320, 320),
                                                   0.8, 0.3, 5000)
        # also the reference 5-point alignment (alignCrop) for the GPU engine
        self._recognizer = cv2.FaceRecognizerSF.create(str(self.model_dir / RECOGNIZER), "")
        self._gpu_det = self._gpu_rec = self._ada = None
        self.device = "cpu"
        # the stronger recognition model when it is there (TESTPOINT_FACE_MODEL=sface keeps SFace)
        want = os.environ.get("TESTPOINT_FACE_MODEL", "auto").lower()
        self.model = "adaface" if want in ("auto", "adaface") and (self.model_dir / ADAFACE).is_file() else "sface"
        self.dim = 512 if self.model == "adaface" else 128
        if device != "cpu":
            from ..vision import face_gpu
            if face_gpu.cuda_available():
                try:
                    self._gpu_det = face_gpu.YuNetOrt(self.model_dir / DETECTOR)
                    self._gpu_rec = face_gpu.SFaceOrt(self.model_dir / RECOGNIZER)
                    self.device = "cuda"
                except Exception as e:      # a broken CUDA install: the CPU still works
                    if device == "cuda":
                        raise
                    print(f"face models on the CPU: the GPU could not be used ({e})")
            elif device == "cuda":
                raise FaceError("no_gpu", "ONNX Runtime has no CUDA here")
        if self.model == "adaface":
            from ..vision import face_gpu
            self._ada = face_gpu.AdaFaceOrt(self.model_dir / ADAFACE, "cuda" if self.device == "cuda" else "cpu")
        # The OpenCV detector keeps per-call state (input size): one caller at a time.
        self._lock = threading.Lock()

    SMALL_PX = 400 * 400
    """A crop this small (a head or body crop from a camera) is found faster on the CPU: 1.5 ms
    against 7.5 on the GPU for 200x200 (measured), where starting the GPU costs more than the work."""

    RETRY_SCALES = (0.5, 0.35)
    """No face found: look again in smaller copies. YuNet finds faces of about 30-40 px best; a soft
    face of 70-100 px (a head crop from a camera at night, in infrared) is found only once halved."""

    def detect(self, img: np.ndarray) -> np.ndarray:
        faces = self._detect_once(img)
        if len(faces) or min(img.shape[:2]) < 120:
            return faces
        import cv2

        for scale in self.RETRY_SCALES:
            small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            faces = self._detect_once(small)
            if len(faces):
                faces = faces.copy()
                faces[:, :14] /= scale        # back in this picture's pixels (box, size and landmarks)
                return faces
        return faces

    def _detect_once(self, img: np.ndarray) -> np.ndarray:
        if self._gpu_det is not None and img.shape[0] * img.shape[1] > self.SMALL_PX:
            return self._gpu_det.detect(img)
        h, w = img.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(img)
        return np.zeros((0, 15), np.float32) if faces is None else faces

    def embed(self, img: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """L2-normalised embeddings (512-d AdaFace, else 128-d SFace) of the faces ``rows``
        (detector rows) of ``img``."""
        aligned = [self._recognizer.alignCrop(img, r) for r in rows]
        if self._ada is not None:
            feats = self._ada.features(aligned)
        elif self._gpu_rec is not None:
            feats = self._gpu_rec.features(aligned)
        else:
            feats = np.array([self._recognizer.feature(a).reshape(-1) for a in aligned], np.float32).reshape(-1, 128)
        n = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.where(n > 0, n, 1)

    def pose_from_bytes(self, data: bytes) -> dict:
        """Detection only, for the live guide while enrolling: the largest face's box and head
        turn. Much cheaper than ``face_from_bytes`` (no recognition)."""
        import cv2

        img = decode_image(data)
        h, w = img.shape[:2]
        scale = min(1.0, 640.0 / max(h, w))
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
        with self._lock:
            faces = self.detect(small).copy()
        if len(faces) == 0:
            return {"face": False, "faces": 0, "w": w, "h": h}
        faces[:, :14] /= scale
        f = max(faces, key=lambda r: r[2] * r[3])
        x, y, bw, bh = (int(round(v)) for v in f[:4])
        return {"face": True, "faces": int(len(faces)), "box": [x, y, bw, bh], "w": w, "h": h,
                "yaw": round(head_yaw(f), 3), "yaw_box": round(head_yaw_box(f), 3),
                "too_small": bool(min(bw, bh) < MIN_FACE_PX)}

    def face_from_bytes(self, data: bytes, require_single: bool = False) -> Face:
        return self.largest_face(decode_image(data), require_single)

    def largest_face(self, img: np.ndarray, require_single: bool = False) -> Face:
        with self._lock:
            return self._largest_face(img, require_single)

    def _largest_face(self, img: np.ndarray, require_single: bool = False) -> Face:
        import cv2

        # YuNet works best around 640 px. Try that first (close-up portraits, where the face
        # fills the frame, are missed at full size), then 1280 px for small faces in big frames.
        # (The GPU engine does both passes itself.)
        h, w = img.shape[:2]
        faces = np.zeros((0, 15), np.float32)
        for target in ((0,) if self._gpu_det is not None else (640.0, 1280.0)):
            scale = min(1.0, target / max(h, w)) if target else 1.0
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
        emb = self.embed(img, np.array([f]))[0]
        q = quality(aligned, f)
        x, y, bw, bh = (int(round(v)) for v in f[:4])
        # The photo people see (thumbnails, the ID card, the finish-line check) is head and
        # shoulders from the picture itself, upright, 4:5, not the 112-px aligned crop the
        # recogniser works on.
        ih, iw = img.shape[:2]
        ph = bh * 2.0
        pw = ph * 0.8
        cx, cy = x + bw / 2, y + bh * 0.6
        x0, x1 = int(max(0, cx - pw / 2)), int(min(iw, cx + pw / 2))
        y0, y1 = int(max(0, cy - ph / 2)), int(min(ih, cy + ph / 2))
        portrait = img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else aligned
        if portrait.shape[1] > 400:
            portrait = cv2.resize(portrait, (400, int(portrait.shape[0] * 400 / portrait.shape[1])), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", portrait, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Face((x, y, bw, bh), round(float(f[14]), 3), emb, jpg.tobytes() if ok else b"",
                    yaw=round(head_yaw(f), 3), quality=q["score"], issues=q["issues"])


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
        # A few tries: a fresh Docker Desktop often fails DNS for its first seconds.
        for attempt in range(1, 6):
            try:
                data = urllib.request.urlopen(url, timeout=120).read()
                break
            except OSError as e:
                if attempt == 5:
                    raise
                print(f"retry   {name} in {2 * attempt} s ({e})")
                time.sleep(2 * attempt)
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
