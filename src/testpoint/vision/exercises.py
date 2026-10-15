"""Exercise rule sets: turn a Person into a movement metric plus form faults.

Thresholds are deliberately plain numbers in dataclasses so that they can be
stored as a versioned RuleSet and tuned in the Advanced UI later. Values here
are starting points; they must be calibrated on real footage (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .counter import RepCounterConfig
from .geometry import angle_deg, distance, elevation_deg, point_line_distance
from .keypoints import Person


@dataclass
class FrameMetrics:
    metric: float | None
    """Movement metric for the counter; None when the person is not measurable."""
    faults: list[str] = field(default_factory=list)
    debug: dict[str, float] = field(default_factory=dict)


class ExerciseAnalyzer(Protocol):
    name: str

    def counter_config(self) -> RepCounterConfig: ...

    def analyze(self, person: Person) -> FrameMetrics: ...


# --------------------------------------------------------------------------- push-ups

@dataclass
class PushupRules:
    down_elbow_deg: float = 95.0
    """Elbow angle below which the bottom position is reached."""
    up_elbow_deg: float = 155.0
    """Elbow angle above which arms count as extended."""
    min_body_line_deg: float = 160.0
    """Shoulder-hip-ankle angle below which hips are sagging or piked."""
    knee_ground_max_ratio: float = 0.10
    """Knee height above the ground line (wrist-ankle line, or the ankle level when the
    wrist is not visible) as a fraction of the hip-ankle length. Below this, in the
    down position, the knee is on the ground ("knee_on_ground")."""
    rest_in_down_s: float = 3.0
    """Longest time allowed away from the top position before the rep is rejected for
    resting ("rest_in_down")."""
    min_conf: float = 0.3
    min_work_frames: int = 2
    min_rest_frames: int = 2


class PushupAnalyzer:
    name = "pushup"

    def __init__(self, rules: PushupRules | None = None) -> None:
        self.rules = rules or PushupRules()

    def counter_config(self) -> RepCounterConfig:
        return RepCounterConfig(
            enter_work=self.rules.down_elbow_deg,
            enter_rest=self.rules.up_elbow_deg,
            work_is_below=True,
            min_work_frames=self.rules.min_work_frames,
            min_rest_frames=self.rules.min_rest_frames,
            max_work_seconds=self.rules.rest_in_down_s,
            work_timeout_fault="rest_in_down",
        )

    def analyze(self, person: Person) -> FrameMetrics:
        side = person.best_side("shoulder", "elbow", "wrist", min_conf=self.rules.min_conf)
        if side is None:
            return FrameMetrics(metric=None)
        elbow = angle_deg(
            person.point(side["shoulder"]), person.point(side["elbow"]), person.point(side["wrist"])
        )
        faults: list[str] = []
        debug = {"elbow_deg": elbow}

        body_side = person.best_side("shoulder", "hip", "ankle", min_conf=self.rules.min_conf)
        if body_side is not None:
            body_line = angle_deg(
                person.point(body_side["shoulder"]),
                person.point(body_side["hip"]),
                person.point(body_side["ankle"]),
            )
            debug["body_line_deg"] = body_line
            if body_line < self.rules.min_body_line_deg:
                faults.append("body_not_straight")

        leg_side = person.best_side("hip", "knee", "ankle", min_conf=self.rules.min_conf)
        if leg_side is not None:
            ratio = self._knee_ground_ratio(person, leg_side)
            if ratio is not None:
                debug["knee_ground_ratio"] = ratio
                if elbow <= self.rules.down_elbow_deg and ratio < self.rules.knee_ground_max_ratio:
                    faults.append("knee_on_ground")

        return FrameMetrics(metric=elbow, faults=faults, debug=debug)

    def _knee_ground_ratio(self, person: Person, side: dict[str, int]) -> float | None:
        """Knee distance from the ground line divided by the hip-ankle length (scale-free)."""
        hip = person.point(side["hip"])
        knee = person.point(side["knee"])
        ankle = person.point(side["ankle"])
        body_len = distance(hip, ankle)
        if body_len <= 0:
            return None
        if person.visible(side["wrist"], min_conf=self.rules.min_conf):
            ground_a, ground_b = person.point(side["wrist"]), ankle
        else:
            # Only the feet are known to be on the floor: use a horizontal through the ankle.
            ground_a, ground_b = ankle, (ankle[0] + 1.0, ankle[1])
        return point_line_distance(knee, ground_a, ground_b) / body_len


# --------------------------------------------------------------------------- crunches

@dataclass
class CrunchRules:
    up_torso_deg: float = 35.0
    """Torso elevation (hip->shoulder above horizontal) that counts as crunched up."""
    down_torso_deg: float = 15.0
    """Torso elevation below which the candidate is back on the floor."""
    knee_min_deg: float = 45.0
    knee_max_deg: float = 135.0
    """Knee angle must stay within this range (legs bent, feet on the floor)."""
    min_conf: float = 0.3
    min_work_frames: int = 2
    min_rest_frames: int = 2


class CrunchAnalyzer:
    name = "crunch"

    def __init__(self, rules: CrunchRules | None = None) -> None:
        self.rules = rules or CrunchRules()

    def counter_config(self) -> RepCounterConfig:
        return RepCounterConfig(
            enter_work=self.rules.up_torso_deg,
            enter_rest=self.rules.down_torso_deg,
            work_is_below=False,
            min_work_frames=self.rules.min_work_frames,
            min_rest_frames=self.rules.min_rest_frames,
        )

    def analyze(self, person: Person) -> FrameMetrics:
        side = person.best_side("shoulder", "hip", min_conf=self.rules.min_conf)
        if side is None:
            return FrameMetrics(metric=None)
        torso = elevation_deg(person.point(side["hip"]), person.point(side["shoulder"]))
        faults: list[str] = []
        debug = {"torso_deg": torso}

        leg_side = person.best_side("hip", "knee", "ankle", min_conf=self.rules.min_conf)
        if leg_side is not None:
            knee = angle_deg(
                person.point(leg_side["hip"]),
                person.point(leg_side["knee"]),
                person.point(leg_side["ankle"]),
            )
            debug["knee_deg"] = knee
            if not (self.rules.knee_min_deg <= knee <= self.rules.knee_max_deg):
                faults.append("knees_not_bent")

        return FrameMetrics(metric=torso, faults=faults, debug=debug)


ANALYZERS: dict[str, type] = {
    PushupAnalyzer.name: PushupAnalyzer,
    CrunchAnalyzer.name: CrunchAnalyzer,
}


def make_analyzer(name: str) -> ExerciseAnalyzer:
    try:
        return ANALYZERS[name]()
    except KeyError:
        raise ValueError(f"unknown exercise {name!r}; choose from {sorted(ANALYZERS)}") from None
