"""Exercise rule sets: turn a Person into a movement metric plus form faults.

Thresholds are deliberately plain numbers in dataclasses so that they can be
stored as a versioned RuleSet and tuned in the Advanced UI later. Values here
are starting points; they must be calibrated on real footage (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Protocol

from .counter import RepCounterConfig
from .geometry import angle_deg, distance, elevation_deg
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
    min_body_line_deg: float = 150.0
    """Shoulder-hip-ankle angle below which hips are sagging or piked ("body_not_straight").
    Calibrated 2026-10-19 on public clips: clean reps never stay below 150 for 0.05 s,
    even from an oblique camera (docs/CALIBRATION.md)."""
    max_knee_lift: float = -0.3
    """Knee height above the ankle as a fraction of the hip's height above the ankle.
    On the toes the knee sits about half way (0.1 .. 0.45 on the clips); on the knees it
    is as low as the ankle or lower (-0.6 .. -0.8). Below this, with a bent knee, the knee
    is on the ground ("knee_on_ground")."""
    min_knee_drop: float = 0.15
    """With the feet raised to hip height or above, how far below both hip and ankle the
    knee must be (as a fraction of the thigh length) to count as on the ground."""
    max_knee_deg: float = 165.0
    """Hip-knee-ankle angle; the knee-lift test only applies when the knee is bent below this."""
    fault_min_s: float = 0.2
    """A form fault must last this long to reject the rep (pose jitter lasts a frame or two)."""
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
            min_fault_seconds=self.rules.fault_min_s,
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
            hip, knee, ankle = (person.point(leg_side[k]) for k in ("hip", "knee", "ankle"))
            knee_deg = angle_deg(hip, knee, ankle)
            debug["knee_deg"] = knee_deg
            lift = knee_lift(hip, knee, ankle)
            if lift is not None:
                debug["knee_lift"] = lift
                knee_down = lift < self.rules.max_knee_lift
            else:
                # Feet raised to hip height or above (a common knee push-up): the knee on the
                # floor is then clearly lower than both the hip and the ankle.
                drop = float(knee[1] - max(hip[1], ankle[1])) / max(distance(hip, knee), 1e-6)
                debug["knee_drop"] = drop
                knee_down = drop > self.rules.min_knee_drop
            if knee_down and knee_deg < self.rules.max_knee_deg:
                faults.append("knee_on_ground")

        return FrameMetrics(metric=elbow, faults=faults, debug=debug)


def knee_lift(hip, knee, ankle, min_rise: float = 0.05) -> float | None:
    """Knee height above the ankle divided by the hip's height above the ankle (image y
    grows downwards). About 0.5 for a straight body on the toes, 0 or below with the knee
    on the floor. None when the hip is barely above the ankle (the ratio is meaningless)."""
    rise = float(ankle[1] - hip[1])
    if rise < min_rise * distance(hip, ankle):
        return None
    return float(ankle[1] - knee[1]) / rise


# --------------------------------------------------------------------------- crunches

@dataclass
class CrunchRules:
    """Torso elevation is the hip->shoulder line above horizontal, measured from the
    candidate's own lying level (``floor_window_s``), so the camera angle does not
    matter. Calibrated 2026-10-19 on public clips (docs/CALIBRATION.md): a crunch with
    the shoulder blades off the floor raises the torso 10-15 degrees; a full sit-up 60+.
    For a full sit-up test set ``up_torso_deg`` to about 60."""

    up_torso_deg: float = 7.0
    """Rise above the lying level that counts as crunched up."""
    down_torso_deg: float = 4.0
    """Rise below which the candidate is back on the floor. Some candidates do not lie all
    the way back between crunches; 3 merged two crunches into one on the clips."""
    min_rep_s: float = 0.4
    """Shorter movements are keypoint noise (0.2-0.4 s); real crunches take 0.8-1.4 s."""
    floor_window_s: float = 6.0
    """Seconds over which the lying level is taken (a low percentile of the elevation)."""
    max_lying_deg: float = 40.0
    """Absolute torso elevation above which the candidate is not lying (sitting, getting
    up): never the rest position. Lying measures 0 from the side, up to 33 from a 3/4 view."""
    knee_min_deg: float = 20.0
    knee_max_deg: float = 135.0
    """Knee angle must stay within this range (legs bent, feet on the floor). The fault
    worth catching is straight legs; feet tucked close in measure 30-40 degrees."""
    fault_min_s: float = 0.2
    """A fault must last this long to reject the rep (pose jitter)."""
    arm_s: float = 1.0
    """Lying still this long before counting starts (RULES.md); lying down is not a rep."""
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
            floor_window_s=self.rules.floor_window_s,
            rest_limit=self.rules.max_lying_deg,
            arm_seconds=self.rules.arm_s,
            min_rep_seconds=self.rules.min_rep_s,
            min_fault_seconds=self.rules.fault_min_s,
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


# The rules a station may tune (Settings > Exercise station), with the range each may take:
# (lowest, highest, step). Anything else stays as calibrated.
TUNABLE: dict[str, dict[str, tuple[float, float, float]]] = {
    "pushup": {
        "down_elbow_deg": (60, 130, 1), "up_elbow_deg": (120, 175, 1), "min_body_line_deg": (120, 175, 1),
        "max_knee_lift": (-1.0, 0.5, 0.05), "max_knee_deg": (120, 180, 1), "min_knee_drop": (0.0, 1.0, 0.05),
        "rest_in_down_s": (1, 10, 0.5), "fault_min_s": (0.0, 1.0, 0.05),
    },
    "crunch": {
        "up_torso_deg": (3, 60, 0.5), "down_torso_deg": (1, 50, 0.5), "knee_min_deg": (0, 90, 1),
        "knee_max_deg": (90, 180, 1), "max_lying_deg": (20, 70, 1), "min_rep_s": (0.1, 2.0, 0.1),
        "fault_min_s": (0.0, 1.0, 0.05),
    },
}
RULES_CLASS = {"pushup": PushupRules, "crunch": CrunchRules}


def tuning_defaults() -> dict:
    """Every tunable rule with its calibrated value and its range, for the settings page."""
    return {ex: {k: {"default": getattr(RULES_CLASS[ex](), k), "min": lo, "max": hi, "step": st}
                 for k, (lo, hi, st) in keys.items()} for ex, keys in TUNABLE.items()}


def check_tuning(tuning: dict) -> dict:
    """Validate ``{exercise: {rule: value}}``: known rules, in range, and a rep that can happen
    (bottom below straight arms, back down below the rise, knee range the right way round).
    Returns it cleaned (floats). Raises ValueError with a readable reason."""
    out: dict[str, dict[str, float]] = {}
    for ex, values in (tuning or {}).items():
        if ex not in TUNABLE:
            raise ValueError(f"unknown exercise {ex!r}")
        clean = {}
        for k, v in (values or {}).items():
            if k not in TUNABLE[ex]:
                raise ValueError(f"{ex}: {k} cannot be tuned")
            lo, hi, _ = TUNABLE[ex][k]
            v = float(v)
            if not lo <= v <= hi:
                raise ValueError(f"{ex}: {k} must be between {lo:g} and {hi:g}")
            clean[k] = v
        r = replace(RULES_CLASS[ex](), **clean)
        if ex == "pushup" and r.up_elbow_deg - r.down_elbow_deg < 20:
            raise ValueError("push-ups: straight arms must be at least 20° above the bottom")
        if ex == "crunch" and r.down_torso_deg >= r.up_torso_deg:
            raise ValueError("crunches: back down must be lower than the rise")
        if ex == "crunch" and r.knee_min_deg >= r.knee_max_deg:
            raise ValueError("crunches: the knee range is the wrong way round")
        if clean:
            out[ex] = clean
    return out


def make_analyzer(name: str, tuning: dict[str, float] | None = None) -> ExerciseAnalyzer:
    """The analyzer for ``name``; ``tuning`` (from the station's settings) replaces calibrated rules."""
    if name not in ANALYZERS:
        raise ValueError(f"unknown exercise {name!r}; choose from {sorted(ANALYZERS)}")
    if tuning:
        rules = replace(RULES_CLASS[name](), **check_tuning({name: tuning}).get(name, {}))
        return ANALYZERS[name](rules)
    return ANALYZERS[name]()
