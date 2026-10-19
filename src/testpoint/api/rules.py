"""Day rules for the station app.

Every number is a placeholder until the official regulation arrives (see
docs/RULES.md and docs/TEST-DAY.md). They live in one dataclass so that the
Advanced screen can edit and version them later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

EVENTS = ("pushup", "crunch")


@dataclass(frozen=True)
class DayRules:
    bmi_min: float = 18.5
    bmi_max: float = 30.0
    run_pass_s: float = 16 * 60
    """A run time at or under this passes (placeholder 16:00)."""
    run_cutoff_s: float = 25 * 60
    """Runners without a finish after this are DNF when the run is closed."""
    pushup_pass: int = 25
    crunch_pass: int = 25
    exercise_time_s: int = 120
    version: str = "v1-placeholder"

    def exercise_pass(self, event: str) -> int:
        if event == "pushup":
            return self.pushup_pass
        if event == "crunch":
            return self.crunch_pass
        raise ValueError(f"unknown event {event!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def bmi(weight_kg: float, height_cm: float) -> float:
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("weight and height must be positive")
    m = height_cm / 100.0
    return round(weight_kg / (m * m), 1)
