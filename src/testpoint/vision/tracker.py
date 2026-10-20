"""A small IoU tracker for a finish chute: a handful of runners, mostly one at a time.

Greedy IoU matching, highest overlap first; a track survives ``max_missing``
frames without a detection and is then dropped. Each track keeps its recent
positions, which is what the finish line needs to interpolate the crossing
instant. Deliberately not ByteTrack: nothing to install, easy to reason about,
and the chute is a controlled scene. Swap it when the clip library says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detect import Box


@dataclass
class Track:
    id: int
    box: Box
    t: float
    history: list[tuple[float, Box]] = field(default_factory=list)
    """``(t, box)`` for the frames where the track was seen, newest last."""
    missing: int = 0
    hits: int = 1

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 15, keep_history: int = 90) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.keep_history = keep_history
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list[Box], t: float) -> list[Track]:
        """Match the frame's boxes to the tracks. Returns the tracks seen in this frame."""
        pairs = sorted(((tr.box.iou(b), ti, bi) for ti, tr in enumerate(self.tracks)
                        for bi, b in enumerate(boxes)), reverse=True)
        used_t: set[int] = set()
        used_b: set[int] = set()
        seen: list[Track] = []
        for iou, ti, bi in pairs:
            if iou < self.iou_threshold:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            tr = self.tracks[ti]
            tr.box, tr.t, tr.missing, tr.hits = boxes[bi], t, 0, tr.hits + 1
            tr.history.append((t, boxes[bi]))
            del tr.history[:-self.keep_history]
            seen.append(tr)
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                tr = Track(self._next_id, b, t, [(t, b)])
                self._next_id += 1
                self.tracks.append(tr)
                seen.append(tr)
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t and tr not in seen:
                tr.missing += 1
        self.tracks = [tr for tr in self.tracks if tr.missing <= self.max_missing]
        return seen
