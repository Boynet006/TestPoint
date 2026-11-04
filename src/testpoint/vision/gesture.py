"""Gestures from the pose keypoints (COCO-17): a raised hand, and who stands at the mark."""

from __future__ import annotations

import numpy as np

from .keypoints import L_ANKLE, R_ANKLE, Person

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


def scale(p: Person) -> float | None:
    """The person's own size in pixels: the torso, or, close to the camera with the hips out of
    the picture, the shoulders (a torso is about 1.3 shoulder widths)."""
    t = torso(p)
    if t is None and _ok(p, L_SHOULDER, R_SHOULDER):
        t = 1.3 * float(np.linalg.norm(p.xy[L_SHOULDER] - p.xy[R_SHOULDER])) or None
    return t


def hand_raised(p: Person) -> str | None:
    """"left" or "right" (the person's own hand) when one hand is clearly up, "both" when both are,
    None otherwise."""
    t = scale(p)
    if t is None:
        return None
    up = []
    for side, w, s in (("left", L_WRIST, L_SHOULDER), ("right", R_WRIST, R_SHOULDER)):
        if _ok(p, w, s) and p.xy[s][1] - p.xy[w][1] > RAISE * t:     # image y grows downwards
            up.append(side)
    return up[0] if len(up) == 1 else "both" if up else None


def pointer(p: Person, size: tuple[int, int]) -> tuple[float, float] | None:
    """Where a raised hand is on the screen that shows the camera's picture as a mirror: (x, y),
    each 0..1 of the picture (``size``: its width and height). The cursor sits on the hand the
    person sees there, so left and right are never the wrong way round. A hand points when it is
    held out to the side at about shoulder height or above; a hand at the face or the chest (scratching
    the chin, adjusting headphones) does not. The higher of two points; None otherwise."""
    t = scale(p)
    if t is None or not _ok(p, L_SHOULDER, R_SHOULDER):
        return None
    cx = float(p.xy[L_SHOULDER][0] + p.xy[R_SHOULDER][0]) / 2
    cy = float(p.xy[L_SHOULDER][1] + p.xy[R_SHOULDER][1]) / 2
    sw = abs(float(p.xy[L_SHOULDER][0] - p.xy[R_SHOULDER][0])) or t / 1.3
    up = [p.xy[w] for w in (L_WRIST, R_WRIST) if float(p.conf[w]) >= MIN_CONF
          and p.xy[w][1] < cy + 0.2 * t and abs(float(p.xy[w][0]) - cx) > 0.6 * sw]
    if not up:
        return None
    wx, wy = min(up, key=lambda xy: xy[1])
    w, h = size
    return round(min(1.0, max(0.0, 1.0 - float(wx) / w)), 3), round(min(1.0, max(0.0, float(wy) / h)), 3)


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


LINE_BAND = 0.25
"""With a start line: how far behind it (share of the picture's height) the feet may be to stand
at it; further back is someone waiting or walking past."""
OVER_MARGIN = 0.015
"""Feet this far past the line (share of the picture's height) are over it: the ankle keypoint
sits a little above the sole, and toes on the line are allowed."""


def feet_y(p: Person, frame_h: int, min_conf: float = 0.3) -> float | None:
    """How far down the picture the feet are (0 top, 1 bottom): the lower ankle."""
    ys = [p.xy[i][1] for i in (L_ANKLE, R_ANKLE) if p.conf[i] >= min_conf]
    return max(ys) / frame_h if ys else None


def over_line(p: Person, frame_h: int, line_y: float, direction: str = "down") -> bool:
    """A foot past the start line, on the side the runners set off to."""
    y = feet_y(p, frame_h)
    if y is None:
        return False
    return y > line_y + OVER_MARGIN if direction == "down" else y < line_y - OVER_MARGIN


def near_line(p: Person, frame_h: int, line_y: float, direction: str = "down") -> bool:
    """Standing at the start line: the feet behind it, but not far behind (or just over it: a
    false start is still the runner at the mark)."""
    y = feet_y(p, frame_h)
    if y is None:
        return False
    behind = line_y - y if direction == "down" else y - line_y
    return -LINE_BAND / 2 <= behind <= LINE_BAND


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
