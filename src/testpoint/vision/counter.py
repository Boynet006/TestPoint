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

Relative metric: with ``floor_window_s`` set, the thresholds are measured from the
candidate's own rest level instead of from zero. The rest level is a low percentile
of the metric over the last ``floor_window_s`` seconds (a high one when WORK is
below). This is for crunches: the torso rises only 10-15 degrees above lying, and
the lying level itself moves with the camera angle (0 from the side, 18-33 from a
3/4 view), so no fixed threshold works for every camera.

Persistent faults: with ``min_fault_seconds`` > 0 a fault only counts once it has
been reported on consecutive frames for that long. Pose estimates jitter, and a
single bad frame (a hand in front of the hip, a mis-placed knee) must not reject
a clean rep; a real fault (sagging hips, knees down) lasts much longer.

Time-based fault: when ``max_work_seconds`` is set, a candidate who stays away
from the rest position (resting at the bottom, or half-way up) for longer than
that gets ``work_timeout_fault`` added to the current rep, which is then
rejected when it completes.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


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
    min_fault_seconds: float = 0.0
    """How long a fault must be reported without a break before it counts (0 = at once)."""
    floor_window_s: float | None = None
    """Measure the metric from the candidate's rest level over this many seconds (None = absolute)."""
    floor_percentile: float = 10.0
    """Which percentile of the window is the rest level (mirrored when WORK is below)."""
    min_rep_seconds: float = 0.0
    """A movement shorter than this (rest to rest) is ignored: not counted, not rejected.
    Keypoint noise makes 0.2-0.4 s blips; a real crunch takes 0.8-1.4 s."""
    arm_seconds: float = 0.0
    """How long the rest position must be held before counting starts (RULES.md: 1 s).
    Getting into position (lying down, a plank) is never a rep. 0 = ``min_rest_frames`` only.
    With a relative metric: how many seconds of rest-position data the rest level needs
    (a candidate who crunches without pausing never holds still for a second)."""
    drop_work: float | None = None
    """A second way to see the bottom of a rep: the body's ``height`` has fallen this far from
    its top level (push-ups: the elbow angle cannot be read from every camera). None = off."""
    drop_rest: float = 0.05
    """With ``drop_work``: the height is back within this of its top level, the rep is done."""
    drop_window_s: float = 6.0
    """With ``drop_work``: seconds over which the top level is taken."""
    auto_share: float = 0.0
    """Auto-calibration: over the person's first ``auto_reps`` reps, "back at the rest position"
    becomes this share of how far they really move (0 = off). Only that follows the person; the
    depth that counts stays the same for everyone."""
    auto_reps: int = 3
    auto_rest_min: float = 0.0
    """The learned rest level is never below this ..."""
    auto_rest_gap: float = 0.0
    """... nor closer than this to the depth that counts."""
    rest_limit: float | None = None
    """Absolute bound the raw metric must be on the rest side of to count as the rest
    position (crunches: lying, not sitting). Frames beyond it never arm the counter, never
    complete a rep and are left out of the rest level. None = no bound."""

    def __post_init__(self) -> None:
        if self.work_is_below and not self.enter_work < self.enter_rest:
            raise ValueError("with work_is_below, enter_work must be < enter_rest")
        if not self.work_is_below and not self.enter_work > self.enter_rest:
            raise ValueError("without work_is_below, enter_work must be > enter_rest")
        if self.max_work_seconds is not None and self.max_work_seconds <= 0:
            raise ValueError("max_work_seconds must be positive or None")
        if self.min_fault_seconds < 0:
            raise ValueError("min_fault_seconds cannot be negative")
        if self.floor_window_s is not None and self.floor_window_s <= 0:
            raise ValueError("floor_window_s must be positive or None")

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
    _fault_since: dict[str, float] = field(default_factory=dict)
    """When each fault reported on the current frame was first seen without a break."""
    _rest_since: float | None = None
    _history: deque = field(default_factory=deque)
    _heights: deque = field(default_factory=deque)
    depth: float | None = None
    """How far from the rest level the person is now (shoulders' drop, or the torso's rise)."""
    _peak: float = 0.0
    _ranges: list = field(default_factory=list)
    _learned: float | None = None
    """``(t, metric)`` over the last ``floor_window_s`` seconds, for the rest level."""
    floor: float | None = None
    """The current rest level when the metric is relative (shown on the overlay)."""

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

    def restart(self) -> RepCounter:
        """A fresh counter for a new attempt that keeps the rest-level history: the person
        lying on the mat when the examiner presses Start is already measured, so the first
        crunch after Start counts even without a pause."""
        fresh = RepCounter(self.cfg)
        fresh._history = deque(self._history)
        fresh.floor = self.floor
        return fresh

    def reset(self) -> None:
        self.phase = Phase.WAITING
        self._rest_since = None
        self._work_streak = self._rest_streak = self._missing = 0
        self._faults.clear()
        self._fault_since.clear()
        self.depth, self._peak, self._learned = None, 0.0, None
        self._ranges = []

    def _work_level(self) -> float | None:
        """The depth that counts, in the units of ``depth``."""
        if self.cfg.drop_work is not None:
            return self.cfg.drop_work
        return self.cfg.enter_work if self.cfg.floor_window_s is not None and not self.cfg.work_is_below else None

    def _rest_level(self) -> float | None:
        """Back at the rest position, in the units of ``depth``: learned from the person once known."""
        work = self._work_level()
        if work is None:
            return None
        fixed = self.cfg.drop_rest if self.cfg.drop_work is not None else self.cfg.enter_rest
        if self._learned is None or self.cfg.auto_share <= 0:
            return fixed
        return max(self.cfg.auto_rest_min, min(self._learned, work - self.cfg.auto_rest_gap))

    def _learn(self) -> None:
        if self.cfg.auto_share > 0 and self._peak > 0 and len(self._ranges) < self.cfg.auto_reps:
            self._ranges.append(self._peak)
            self._learned = self.cfg.auto_share * float(np.median(self._ranges))
        self._peak = 0.0

    def calibration(self) -> dict | None:
        """What the screen shows of the auto-calibration; None when the exercise has no depth."""
        work = self._work_level()
        if work is None:
            return None
        n = len(self._ranges)
        state = "off" if self.cfg.auto_share <= 0 else ("done" if n >= self.cfg.auto_reps else "learning")
        return {"state": state, "reps": n, "need": self.cfg.auto_reps, "work": round(work, 3),
                "rest": round(self._rest_level(), 3),
                "fixed_rest": round(self.cfg.drop_rest if self.cfg.drop_work is not None else self.cfg.enter_rest, 3),
                "now": None if self.depth is None else round(self.depth, 3),
                "range": round(float(np.median(self._ranges)), 3) if n else None,
                "unit": "body" if self.cfg.drop_work is not None else "deg"}

    def _persistent(self, faults: Iterable[str], t: float) -> set[str]:
        """The frame's faults that have lasted ``min_fault_seconds`` (all of them when 0)."""
        now = set(faults)
        self._fault_since = {f: self._fault_since.get(f, t) for f in now}
        return {f for f, since in self._fault_since.items() if t - since >= self.cfg.min_fault_seconds}

    def _restable(self, raw: float) -> bool:
        """Whether the raw metric is on the rest side of ``rest_limit`` (always without one)."""
        lim = self.cfg.rest_limit
        if lim is None:
            return True
        return raw > lim if self.cfg.work_is_below else raw < lim

    def _relative(self, metric: float, t: float) -> float:
        """The metric measured from the rest level over the last ``floor_window_s`` seconds."""
        h = self._history
        if self._restable(metric):
            h.append((t, metric))
        while h and h[0][0] < t - self.cfg.floor_window_s:
            h.popleft()
        if h:
            q = self.cfg.floor_percentile if not self.cfg.work_is_below else 100 - self.cfg.floor_percentile
            self.floor = float(np.percentile([m for _t, m in h], q))
        return metric - (self.floor if self.floor is not None else metric)

    def _drop(self, height: float | None, t: float) -> float | None:
        """How far the height is below its top level over the last ``drop_window_s`` seconds."""
        if self.cfg.drop_work is None or height is None:
            return None
        h = self._heights
        h.append((t, height))
        while h and h[0][0] < t - self.cfg.drop_window_s:
            h.popleft()
        return float(np.percentile([v for _t, v in h], 90)) - height

    def update(self, metric: float | None, t: float, faults: Iterable[str] = (),
               height: float | None = None) -> RepEvent | None:
        """Feed one frame. Returns a RepEvent when a rep completes (valid or rejected).
        ``height``: the body's height for ``drop_work``, when the exercise uses it."""
        if metric is None:
            self._missing += 1
            self._fault_since.clear()
            if self._missing > self.cfg.max_missing_frames:
                self.reset()
                self._history.clear()
                self._heights.clear()
            return None
        self._missing = 0
        faults = self._persistent(faults, t)
        restable = self._restable(metric)
        if self.cfg.floor_window_s is not None:
            metric = self._relative(metric, t)

        in_work = self.cfg.in_work(metric)
        relative_rise = self.cfg.floor_window_s is not None and not self.cfg.work_is_below
        in_rest = (metric < self._rest_level() if relative_rise else self.cfg.in_rest(metric)) and restable
        drop = self._drop(height, t)
        if drop is not None:
            in_work = in_work or drop >= self.cfg.drop_work
            in_rest = (in_rest or drop <= self._rest_level()) and not in_work
        self.depth = drop if drop is not None else (metric if relative_rise else None)
        if self.depth is not None and self.phase is not Phase.WAITING:
            self._peak = max(self._peak, self.depth)

        if self.phase is Phase.WAITING:
            self._rest_streak = self._rest_streak + 1 if in_rest else 0
            if in_rest and self._rest_since is None:
                self._rest_since = t
            elif not in_rest:
                self._rest_since = None
            if self.cfg.floor_window_s is not None:
                # Relative metric: arm once the rest level rests on arm_seconds of rest-position
                # data. Crunching without a pause still arms; the first frames after lying
                # down (which skew the level) do not count on their own.
                h = self._history
                held = bool(h) and h[-1][0] - h[0][0] >= self.cfg.arm_seconds
            else:
                held = self._rest_since is not None and t - self._rest_since >= self.cfg.arm_seconds
            if self._rest_streak >= self.cfg.min_rest_frames and held:
                self.phase = Phase.REST
                self._rest_streak = 0
            return None

        if self.phase is Phase.REST:
            # Faults seen while leaving the rest position (the descent) belong to the
            # coming rep; faults seen while fully at rest are discarded.
            if in_rest:
                self._faults.clear()
                self._rep_start = t
                self._peak = 0.0
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
        if t - self._rep_start < self.cfg.min_rep_seconds:
            self._faults = set()
            self._peak = 0.0
            return None
        self._learn()
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
        self._rep_start = t      # the next rep starts here, even without a rest frame in between
        return event
