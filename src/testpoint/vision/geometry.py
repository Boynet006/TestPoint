"""Small 2-D geometry helpers on image coordinates (x right, y down)."""

from __future__ import annotations

import math

import numpy as np


def angle_deg(a, b, c) -> float:
    """Angle at point ``b`` formed by segments b->a and b->c, in degrees 0..180.

    A straight limb gives 180, a right angle gives 90.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na == 0 or nc == 0:
        return 0.0
    cosang = float(np.dot(ba, bc) / (na * nc))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def elevation_deg(p_from, p_to) -> float:
    """Signed angle of the vector p_from -> p_to above the horizontal, in degrees.

    Positive when ``p_to`` is higher on screen than ``p_from`` (smaller y). The sign
    of x is ignored so the value is the same whether the person faces left or right.
    Range is -90..90.
    """
    p_from = np.asarray(p_from, dtype=np.float64)
    p_to = np.asarray(p_to, dtype=np.float64)
    dx = abs(float(p_to[0] - p_from[0]))
    dy = float(p_from[1] - p_to[1])  # up on screen is positive
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def distance(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def point_line_distance(p, a, b) -> float:
    """Perpendicular distance from point ``p`` to the infinite line through ``a`` and ``b``.

    Falls back to the plain distance to ``a`` when ``a`` and ``b`` coincide.
    """
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n == 0:
        return float(np.linalg.norm(p - a))
    ap = p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    return float(abs(cross) / n)
