"""COCO-17 keypoint layout and the Person container used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE = 0
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

NUM_KEYPOINTS = 17

# Pairs of keypoints joined when drawing the skeleton.
SKELETON = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
]

LEFT_SIDE = {
    "shoulder": L_SHOULDER, "elbow": L_ELBOW, "wrist": L_WRIST,
    "hip": L_HIP, "knee": L_KNEE, "ankle": L_ANKLE,
}
RIGHT_SIDE = {
    "shoulder": R_SHOULDER, "elbow": R_ELBOW, "wrist": R_WRIST,
    "hip": R_HIP, "knee": R_KNEE, "ankle": R_ANKLE,
}


@dataclass
class Person:
    """One detected person: 17 keypoints in pixel coordinates plus confidences."""

    xy: np.ndarray    # shape (17, 2), float, image pixels (x right, y down)
    conf: np.ndarray  # shape (17,), 0..1

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float32).reshape(NUM_KEYPOINTS, 2)
        self.conf = np.asarray(self.conf, dtype=np.float32).reshape(NUM_KEYPOINTS)

    def point(self, index: int) -> np.ndarray:
        return self.xy[index]

    def visible(self, *indices: int, min_conf: float = 0.3) -> bool:
        return all(self.conf[i] >= min_conf for i in indices)

    def mean_conf(self) -> float:
        return float(self.conf.mean())

    def bbox(self, min_conf: float = 0.3) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) around the confident keypoints; zeros if none."""
        pts = self.xy[self.conf >= min_conf]
        if len(pts) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return (float(x0), float(y0), float(x1), float(y1))

    def area(self, min_conf: float = 0.3) -> float:
        x0, y0, x1, y1 = self.bbox(min_conf)
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def best_side(self, *joints: str, min_conf: float = 0.3) -> dict[str, int] | None:
        """Return the index map of the side (left/right) whose requested joints are
        more confident, or None if neither side has all joints visible."""
        candidates = []
        for side in (LEFT_SIDE, RIGHT_SIDE):
            idx = [side[j] for j in joints]
            if all(self.conf[i] >= min_conf for i in idx):
                candidates.append((float(self.conf[idx].mean()), side))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]
