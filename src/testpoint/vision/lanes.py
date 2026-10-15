"""Lane zones: rectangles in the image, each with its own candidate and counter.

A station camera sees 3-4 mats side by side. The administrator draws one
rectangle per mat (Advanced mode) and the engine assigns every detected person
to the lane that contains their hip midpoint. People outside all lanes
(examiners, the next candidate waiting) are ignored.

Rectangles are stored in normalized image coordinates (0..1, x right, y down)
so the same lane file works for the main stream and the sub stream of a camera.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .keypoints import L_HIP, R_HIP, Person


@dataclass(frozen=True)
class Lane:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for v in (self.x0, self.y0, self.x1, self.y1):
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"lane {self.name!r}: coordinates must be within 0..1, got {v}")
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(f"lane {self.name!r}: rectangle is empty ({self.rect})")

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def contains(self, x: float, y: float) -> bool:
        """True if the normalized point lies inside the rectangle (borders included)."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x0 * width), round(self.y0 * height),
            round(self.x1 * width), round(self.y1 * height),
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "rect": [self.x0, self.y0, self.x1, self.y1]}

    @classmethod
    def from_dict(cls, d: dict, default_name: str = "") -> Lane:
        if "rect" in d:
            x0, y0, x1, y1 = (float(v) for v in d["rect"])
        else:
            x0, y0, x1, y1 = (float(d[k]) for k in ("x0", "y0", "x1", "y1"))
        return cls(str(d.get("name", default_name)), x0, y0, x1, y1)


def whole_frame(name: str = "1") -> Lane:
    return Lane(name, 0.0, 0.0, 1.0, 1.0)


def equal_strips(n: int) -> list[Lane]:
    """``n`` vertical strips of equal width across the frame, named "1".."n" left to right."""
    if n < 1:
        raise ValueError("need at least one lane")
    return [Lane(str(i + 1), i / n, 0.0, (i + 1) / n, 1.0) for i in range(n)]


def load_lanes(path: str | Path) -> list[Lane]:
    """Read lanes from JSON: ``{"lanes": [{"name": "1", "rect": [x0, y0, x1, y1]}, ...]}``
    (a bare list of the same objects is also accepted). Coordinates are 0..1."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["lanes"] if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a non-empty list of lanes")
    lanes = [Lane.from_dict(d, default_name=str(i + 1)) for i, d in enumerate(items)]
    names = [lane.name for lane in lanes]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: lane names must be unique, got {names}")
    return lanes


def save_lanes(lanes: list[Lane], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({"lanes": [lane.to_dict() for lane in lanes]}, indent=2), encoding="utf-8"
    )


def anchor_point(person: Person, min_conf: float = 0.3) -> tuple[float, float] | None:
    """Pixel point used to place a person in a lane: the hip midpoint, or the centre of
    the keypoint bounding box when the hips are not confident. None if nothing is visible."""
    if person.visible(L_HIP, R_HIP, min_conf=min_conf):
        mid = (person.point(L_HIP) + person.point(R_HIP)) / 2.0
        return (float(mid[0]), float(mid[1]))
    x0, y0, x1, y1 = person.bbox(min_conf)
    if x1 <= x0 and y1 <= y0:
        return None
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class LaneAssigner:
    """Map detected people to lanes. When several people fall into one lane the largest
    detection wins (the candidate on the mat is closer to the camera than a bystander)."""

    def __init__(self, lanes: list[Lane], min_conf: float = 0.3, min_visible: int = 4) -> None:
        if not lanes:
            raise ValueError("need at least one lane")
        self.lanes = list(lanes)
        self.min_conf = min_conf
        self.min_visible = min_visible

    def lane_for_point(self, x_norm: float, y_norm: float) -> Lane | None:
        for lane in self.lanes:
            if lane.contains(x_norm, y_norm):
                return lane
        return None

    def assign(self, people: list[Person], width: int, height: int) -> dict[str, Person]:
        """Return ``{lane name: person}`` for the lanes that have someone in them."""
        best: dict[str, tuple[float, Person]] = {}
        for p in people:
            if int((p.conf >= self.min_conf).sum()) < self.min_visible:
                continue
            anchor = anchor_point(p, self.min_conf)
            if anchor is None:
                continue
            lane = self.lane_for_point(anchor[0] / width, anchor[1] / height)
            if lane is None:
                continue
            area = p.area(self.min_conf)
            if lane.name not in best or area > best[lane.name][0]:
                best[lane.name] = (area, p)
        return {name: person for name, (_, person) in best.items()}
