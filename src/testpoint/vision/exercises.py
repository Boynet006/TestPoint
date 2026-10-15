"""Exercise rule sets: turn a Person into a movement metric plus form faults.

Thresholds are deliberately plain numbers in dataclasses so that they can be
stored as a versioned RuleSet and tuned in the Advanced UI later. Values here
are starting points; they must be calibrated on real footage (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .counter import RepCounterConfig
from .geometry import angle_deg, elevation_deg
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

        return FrameMetrics(metric=elbow, faults=faults, debug=debug)


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
