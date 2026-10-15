"""Generic repetition counter: a hysteresis state machine over one scalar metric.

The metric is whatever an exercise analyzer decides represents the movement
(elbow angle for push-ups, torso elevation for crunches). The counter does not
know the exercise; it only knows two thresholds and which side of them is the
"work" position.

States
------
WAITING  nobody counted yet; the candidate must first hold the REST position
         for ``min_rest_frames`` so that getting into position never counts.
REST     the top / lying position.
WORK     the bottom / crunched position.

A rep is REST -> WORK -> REST. Faults reported from the moment the candidate
leaves the rest position until the rep completes make the rep *rejected*: it
is not counted, and the reason is kept, so the examiner can see why.
``t_start`` of a rep is the last frame fully at rest before the descent.

Time-based fault: when ``max_work_seconds`` is set, a candidate who stays away
from the rest position (resting at the bottom, or half-way up) for longer than
that gets ``work_timeout_fault`` added to the current rep, which is then
rejected when it completes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    WAITING = "waiting"
    REST = "rest"
    WORK = "work"


@dataclass
class RepCounterConfig:
    enter_work: float
    """Metric threshold that must be crossed to enter the WORK phase."""
    enter_rest: float
    """Metric threshold that must be crossed to return to REST (completes a rep)."""
    work_is_below: bool = True
    """True if WORK means metric < enter_work (e.g. elbow angle), False if metric > enter_work."""
    min_work_frames: int = 2
    min_rest_frames: int = 2
    max_missing_frames: int = 30
    """Frames without a metric before the counter resets to WAITING."""
    max_work_seconds: float | None = None
    """Longest time allowed in the WORK phase before the rep is faulted; None disables."""
    work_timeout_fault: str = "rest_in_down"
    """Fault name recorded when ``max_work_seconds`` is exceeded."""

    def __post_init__(self) -> None:
        if self.work_is_below and not self.enter_work < self.enter_rest:
            raise ValueError("with work_is_below, enter_work must be < enter_rest")
        if not self.work_is_below and not self.enter_work > self.enter_rest:
            raise ValueError("without work_is_below, enter_work must be > enter_rest")
        if self.max_work_seconds is not None and self.max_work_seconds <= 0:
            raise ValueError("max_work_seconds must be positive or None")

    def in_work(self, metric: float) -> bool:
        return metric < self.enter_work if self.work_is_below else metric > self.enter_work

    def in_rest(self, metric: float) -> bool:
        return metric > self.enter_rest if self.work_is_below else metric < self.enter_rest


@dataclass
class RepEvent:
    index: int            # 1-based index over all attempts (valid + rejected)
    valid: bool
    faults: list[str]
    t_start: float
    t_end: float
    count_after: int      # valid count after this event


@dataclass
class RepCounter:
    cfg: RepCounterConfig
    phase: Phase = Phase.WAITING
    count: int = 0
    rejected: int = 0
    events: list[RepEvent] = field(default_factory=list)
    _work_streak: int = 0
    _rest_streak: int = 0
    _missing: int = 0
    _rep_start: float = 0.0
    _work_since: float = 0.0
    _work_entered: float = 0.0
    _faults: set[str] = field(default_factory=set)

    @property
    def attempts(self) -> int:
        return self.count + self.rejected

    @property
    def pending_faults(self) -> list[str]:
        """Faults collected so far for the rep in progress (shown live on the overlay)."""
        return sorted(self._faults)

    def work_seconds(self, t: float) -> float:
        """Time spent in the WORK phase so far at time ``t`` (0 outside WORK)."""
        return max(0.0, t - self._work_entered) if self.phase is Phase.WORK else 0.0

    def reset(self) -> None:
        self.phase = Phase.WAITING
        self._work_streak = self._rest_streak = self._missing = 0
        self._faults.clear()

    def update(self, metric: float | None, t: float, faults: Iterable[str] = ()) -> RepEvent | None:
        """Feed one frame. Returns a RepEvent when a rep completes (valid or rejected)."""
        if metric is None:
            self._missing += 1
            if self._missing > self.cfg.max_missing_frames:
                self.reset()
            return None
        self._missing = 0

        in_work = self.cfg.in_work(metric)
        in_rest = self.cfg.in_rest(metric)

        if self.phase is Phase.WAITING:
            self._rest_streak = self._rest_streak + 1 if in_rest else 0
            if self._rest_streak >= self.cfg.min_rest_frames:
                self.phase = Phase.REST
                self._rest_streak = 0
            return None

        if self.phase is Phase.REST:
            # Faults seen while leaving the rest position (the descent) belong to the
            # coming rep; faults seen while fully at rest are discarded.
            if in_rest:
                self._faults.clear()
                self._rep_start = t
            else:
                self._faults.update(faults)
            if in_work:
                if self._work_streak == 0:
                    self._work_since = t
                self._work_streak += 1
            else:
                self._work_streak = 0
            if self._work_streak >= self.cfg.min_work_frames:
                self.phase = Phase.WORK
                self._work_entered = self._work_since
                self._work_streak = 0
            return None

        # Phase.WORK
        self._faults.update(faults)
        if (
            self.cfg.max_work_seconds is not None
            and t - self._work_entered > self.cfg.max_work_seconds
        ):
            self._faults.add(self.cfg.work_timeout_fault)
        self._rest_streak = self._rest_streak + 1 if in_rest else 0
        if self._rest_streak < self.cfg.min_rest_frames:
            return None

        self.phase = Phase.REST
        self._rest_streak = 0
        valid = not self._faults
        if valid:
            self.count += 1
        else:
            self.rejected += 1
        event = RepEvent(
            index=self.attempts,
            valid=valid,
            faults=sorted(self._faults),
            t_start=self._rep_start,
            t_end=t,
            count_after=self.count,
        )
        self.events.append(event)
        self._faults = set()
        return event
