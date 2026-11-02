"""The virtual finish line: when a tracked runner's feet cross it, and at what instant.

The line is a segment in normalized image coordinates (0..1, x right, y down),
so the same setting works for the main and sub stream of a camera. A crossing
is a change of side of the track's anchor point (bottom centre of the box)
between two consecutive frames; the crossing instant is interpolated between
the two frame times by the signed distances, which is what gives better than
frame-rate timing. ``direction`` restricts which way counts: "down" (towards a
camera looking back up the course), "up", "left", "right" or "any".
"""

from __future__ import annotations

from dataclasses import dataclass

from .detect import Box
from .tracker import Track


@dataclass(frozen=True)
class CrossEvent:
    track_id: int
    t: float
    """Interpolated crossing instant, in the frame clock."""
    t_before: float
    t_after: float
    box: Box
    """The runner's box in the first frame past the line."""


class FinishLine:
    def __init__(self, line: tuple[float, float, float, float] = (0.0, 0.85, 1.0, 0.85),
                 direction: str = "down") -> None:
        x0, y0, x1, y1 = line
        for v in line:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"line coordinates must be within 0..1, got {line}")
        if (x0, y0) == (x1, y1):
            raise ValueError("line has zero length")
        if direction not in ("down", "up", "left", "right", "any"):
            raise ValueError(f"unknown direction {direction!r}")
        self.line = line
        self.direction = direction
        self._crossed: set[int] = set()

    def signed_distance(self, x: float, y: float, w: int, h: int) -> float:
        """Signed distance from the line, in normalized units; positive on the side
        the normal points to (for a horizontal line: below it)."""
        x0, y0, x1, y1 = self.line
        px, py = x / w, y / h
        dx, dy = x1 - x0, y1 - y0
        n = (dx * dx + dy * dy) ** 0.5
        # normal (-dy, dx) points "down" for a left-to-right horizontal line
        return ((px - x0) * (-dy) + (py - y0) * dx) / n

    def _direction_ok(self, before: float, after: float) -> bool:
        x0, y0, x1, y1 = self.line
        horizontal = abs(x1 - x0) >= abs(y1 - y0)
        if self.direction == "any":
            return True
        positive = after > before
        if horizontal:
            # normal points down when the line runs left to right, up otherwise
            down = positive if x1 >= x0 else not positive
            return down if self.direction == "down" else (not down if self.direction == "up" else False)
        # vertical line: normal points left when the line runs top to bottom
        left = positive if y1 >= y0 else not positive
        return left if self.direction == "left" else (not left if self.direction == "right" else False)

    MIN_APPROACH = 0.05
    """A crossing needs the person to have been at least this far before the line (share of the
    picture), and MIN_TRAVEL is how far they must have moved: someone sitting or shifting near the
    line (their box edge wobbling across it) is not a runner crossing."""
    MIN_TRAVEL = 0.06

    def check(self, tracks: list[Track], w: int, h: int) -> list[CrossEvent]:
        """Crossings completed in this frame, one per track at most, each track once."""
        events = []
        for tr in tracks:
            if tr.id in self._crossed or len(tr.history) < 2 or not tr.confirmed:
                continue
            (t0, b0), (t1, b1) = tr.history[-2], tr.history[-1]
            d0 = self.signed_distance(*b0.feet, w, h)
            d1 = self.signed_distance(*b1.feet, w, h)
            if d0 == 0 or (d0 > 0) == (d1 > 0):
                continue
            if not self._direction_ok(d0, d1):
                continue
            past = [self.signed_distance(*b.feet, w, h) for _t, b in tr.history[:-1]]
            before = [d for d in past if (d > 0) == (d0 > 0)]
            if not before or max(abs(d) for d in before) < self.MIN_APPROACH:
                continue                 # never really before the line: hovering on it
            if max(past + [d1]) - min(past + [d1]) < self.MIN_TRAVEL:
                continue                 # hardly moved: sitting or standing near the line
            frac = abs(d0) / (abs(d0) + abs(d1))
            self._crossed.add(tr.id)
            events.append(CrossEvent(tr.id, t0 + frac * (t1 - t0), t0, t1, b1))
        return events

    def forget(self, track_ids: set[int]) -> None:
        self._crossed &= track_ids

    def to_pixels(self, w: int, h: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.line
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def parse_line(text: str) -> tuple[float, float, float, float]:
    """``"x0,y0,x1,y1"`` in 0..1."""
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError("line needs four numbers: x0,y0,x1,y1")
    return parts[0], parts[1], parts[2], parts[3]
