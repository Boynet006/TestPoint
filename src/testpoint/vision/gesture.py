"""Gestures from the pose keypoints (COCO-17): a raised hand, and who stands at the mark."""

from __future__ import annotations

import numpy as np

from .keypoints import Person

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10
L_HIP, R_HIP = 11, 12
MIN_CONF = 0.4
RAISE = 0.15
"""A hand counts as raised when the wrist is above the shoulder by this share of the torso."""


def _ok(p: Person, *idx: int) -> bool:
    return all(float(p.conf[i]) >= MIN_CONF for i in idx)


def torso(p: Person) -> float | None:
    """Shoulder-to-hip length (pixels), the person's own scale; None when not seen."""
    if not _ok(p, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        return None
    sh = (p.xy[L_SHOULDER] + p.xy[R_SHOULDER]) / 2
    hip = (p.xy[L_HIP] + p.xy[R_HIP]) / 2
    return float(np.linalg.norm(sh - hip)) or None


def hand_raised(p: Person) -> str | None:
    """"left" or "right" (the person's own hand) when one hand is clearly up, "both" when both are,
    None otherwise."""
    t = torso(p)
    if t is None:
        return None
    up = []
    for side, w, s in (("left", L_WRIST, L_SHOULDER), ("right", R_WRIST, R_SHOULDER)):
        if _ok(p, w, s) and p.xy[s][1] - p.xy[w][1] > RAISE * t:     # image y grows downwards
            up.append(side)
    return up[0] if len(up) == 1 else "both" if up else None


POINT_REACH = 1.4
"""The cursor's reach either side of the shoulders, in torso lengths: an easy sweep of the arm
covers the screen, without stretching."""


def pointer(p: Person) -> tuple[float, float] | None:
    """Where a raised hand points on the screen the person faces: (x, y), each 0..1, as in a
    mirror (their right hand moved to their right moves it right). Measured from the person's own
    shoulders, so it works wherever they stand. The higher hand points; None while both hands
    are down (below the middle of the chest)."""
    t = torso(p)
    if t is None:
        return None
    cx, cy = (p.xy[L_SHOULDER] + p.xy[R_SHOULDER]) / 2
    up = [p.xy[w] for w in (L_WRIST, R_WRIST) if float(p.conf[w]) >= MIN_CONF and p.xy[w][1] < cy + 0.5 * t]
    if not up:
        return None
    wx, wy = min(up, key=lambda xy: xy[1])
    x = 0.5 - (wx - cx) / (2 * POINT_REACH * t)       # image x grows to the person's left
    y = (wy - (cy - 1.2 * t)) / (1.7 * t)             # above the head: 0, the middle of the chest: 1
    return round(min(1.0, max(0.0, float(x))), 3), round(min(1.0, max(0.0, float(y))), 3)


def box(p: Person, min_conf: float = 0.3) -> tuple[float, float, float, float] | None:
    """Bounding box (x0, y0, x1, y1) of the keypoints seen."""
    seen = p.xy[p.conf >= min_conf]
    if len(seen) < 4:
        return None
    x0, y0 = seen.min(axis=0)
    x1, y1 = seen.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def head_box(p: Person, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
    """A generous crop around the head (nose, eyes, ears) for face detection, or None."""
    pts = [p.xy[i] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if float(p.conf[i]) >= MIN_CONF]
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    cx, cy = pts.mean(axis=0)
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 8.0)
    half = span * 2.2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half * 1.1))
    x1, y1 = int(min(frame_w, cx + half)), int(min(frame_h, cy + half * 1.2))
    return (x0, y0, x1, y1) if x1 - x0 > 16 and y1 - y0 > 16 else None


def at_the_mark(people: list[Person], frame_h: int, min_height: float = 0.25) -> Person | None:
    """The person standing at the start mark: the largest one in view (nearest the camera), when
    tall enough in the picture to be at the mark rather than walking past behind."""
    best, best_h = None, 0.0
    for p in people:
        b = box(p)
        if b is None:
            continue
        h = b[3] - b[1]
        if h > best_h:
            best, best_h = p, h
    return best if best is not None and best_h >= min_height * frame_h else None
