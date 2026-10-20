"""Reading the bib number of a tracked runner.

The torso crop of several frames is read with OCR and the digit strings are
voted: the number seen most often, weighted by OCR confidence, wins, and only
when its share is high enough. One bad frame therefore does not produce a
wrong bib; an unreadable bib produces ``None`` and the crossing goes to the
examiner's review queue, which is the safe failure.

The OCR engine is RapidOCR (ONNX Runtime, so it shares the runtime we already
ship); ``BibVote`` itself has no dependency and is what the tests exercise.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .detect import Box

DIGITS = re.compile(r"(?<!\d)\d{1,5}(?!\d)")   # whole digit runs only: "123456" is not a bib

Reading = tuple[str, float]
"""A digit string and the OCR confidence 0..1."""


@dataclass
class BibResult:
    bib: str | None
    confidence: float
    readings: list[Reading]
    """Every digit string seen, with its confidence, for the audit trail."""


class BibVote:
    def __init__(self, min_share: float = 0.5, min_readings: int = 1, min_digits: int = 1, max_digits: int = 5) -> None:
        self.min_share = min_share
        self.min_readings = min_readings
        self.min_digits = min_digits
        self.max_digits = max_digits

    def decide(self, readings: Iterable[Reading]) -> BibResult:
        votes: dict[str, float] = defaultdict(float)
        kept: list[Reading] = []
        for text, conf in readings:
            for m in DIGITS.findall(text):
                if self.min_digits <= len(m) <= self.max_digits:
                    votes[m] += conf
                    kept.append((m, conf))
        if not votes or len(kept) < self.min_readings:
            return BibResult(None, 0.0, kept)
        # All readings come from one runner, so a shorter string that is the start or end of a
        # longer one is that number with a digit hidden (a hand, a fold): it supports the longer one.
        for short in [k for k in votes if any(k != o and (o.startswith(k) or o.endswith(k)) for o in votes)]:
            longer = [o for o in votes if o != short and (o.startswith(short) or o.endswith(short))]
            if len(longer) == 1:
                votes[longer[0]] += votes.pop(short)
        total = sum(votes.values())
        best, weight = max(votes.items(), key=lambda kv: kv[1])
        share = weight / total
        if share < self.min_share:
            return BibResult(None, round(share, 2), kept)
        return BibResult(best, round(share, 2), kept)


def torso_crop(frame: np.ndarray, box: Box, top: float = 0.15, bottom: float = 0.65, pad: float = 0.1) -> np.ndarray:
    """The part of the person box where a bib is pinned (chest to hips), a little wider."""
    h, w = frame.shape[:2]
    x0 = max(0, int(box.x0 - pad * box.width))
    x1 = min(w, int(box.x1 + pad * box.width))
    y0 = max(0, int(box.y0 + top * box.height))
    y1 = min(h, int(box.y0 + bottom * box.height))
    return frame[y0:y1, x0:x1]


class RapidOCRReader:
    """Digit strings in an image crop with RapidOCR. Returns ``(text, confidence)`` pairs."""

    def __init__(self, min_side: int = 64) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.ocr = RapidOCR()
        self.min_side = min_side

    def read(self, crop: np.ndarray) -> list[Reading]:
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return []
        h, w = crop.shape[:2]
        if min(h, w) < self.min_side:            # OCR wants text a few dozen pixels tall
            s = self.min_side / min(h, w)
            import cv2
            crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        result, _ = self.ocr(crop)
        return [(str(text), float(conf)) for _box, text, conf in (result or [])]


class BibReader:
    """Reads the bib of a track from the crops it was given, when asked."""

    def __init__(self, read: Callable[[np.ndarray], list[Reading]], vote: BibVote | None = None,
                 max_crops: int = 8) -> None:
        self.read = read
        self.vote = vote or BibVote()
        self.max_crops = max_crops
        self._crops: dict[int, list[tuple[float, np.ndarray]]] = {}

    def observe(self, track_id: int, frame: np.ndarray, box: Box) -> None:
        """Keep the largest recent crops of this runner (closest to the camera reads best)."""
        crops = self._crops.setdefault(track_id, [])
        crops.append((box.area, torso_crop(frame, box).copy()))
        crops.sort(key=lambda c: c[0], reverse=True)
        del crops[self.max_crops:]

    def take(self, track_id: int) -> list[np.ndarray]:
        """Hand over (and forget) the crops kept for this track; cheap, for the frame loop."""
        return [crop for _area, crop in self._crops.pop(track_id, [])]

    def read_crops(self, crops: list[np.ndarray]) -> BibResult:
        """OCR and vote; the slow part, which can run off the frame loop."""
        readings: list[Reading] = []
        for crop in crops:
            readings.extend(self.read(crop))
        return self.vote.decide(readings)

    def decide(self, track_id: int) -> BibResult:
        return self.read_crops(self.take(track_id))

    def forget(self, live_ids: set[int]) -> None:
        for tid in list(self._crops):
            if tid not in live_ids:
                del self._crops[tid]
