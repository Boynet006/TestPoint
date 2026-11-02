"""Sessions: one camera, one exercise, one counter per lane.

``MultiLaneSession`` runs one pose detection per frame, assigns the detected
people to lanes and feeds each lane's candidate to that lane's ``RepCounter``.
``StationSession`` is the single-candidate wrapper (one lane = the whole frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .counter import Phase, RepCounter, RepEvent
from .exercises import ExerciseAnalyzer, FrameMetrics
from .keypoints import Person
from .lanes import Lane, LaneAssigner, whole_frame
from .pose import PoseBackend


@dataclass
class FrameResult:
    t: float
    person: Person | None
    metrics: FrameMetrics
    event: RepEvent | None
    count: int
    rejected: int
    phase: Phase
    lane: str = ""
    pending_faults: list[str] = field(default_factory=list)
    """Faults already collected for the rep in progress (live feedback)."""
    work_seconds: float = 0.0
    """How long the candidate has been away from the rest position."""


@dataclass
class MultiLaneResult:
    t: float
    lanes: dict[str, FrameResult]
    """Per-lane result, keyed by lane name, in lane order."""
    people: list[Person]
    """Everything the pose backend found, assigned or not (for the overlay)."""

    @property
    def events(self) -> list[tuple[str, RepEvent]]:
        return [(name, r.event) for name, r in self.lanes.items() if r.event is not None]


def select_candidate(
    people: list[Person], min_conf: float = 0.3, min_visible: int = 4
) -> Person | None:
    """Pick the person to score: the largest detection with enough confident keypoints."""
    best = None
    best_area = 0.0
    for p in people:
        if int((p.conf >= min_conf).sum()) < min_visible:
            continue
        a = p.area()
        if a > best_area:
            best, best_area = p, a
    return best


class MultiLaneSession:
    def __init__(
        self,
        backend: PoseBackend,
        analyzer: ExerciseAnalyzer,
        lanes: list[Lane],
        min_conf: float = 0.3,
        min_visible: int = 4,
    ) -> None:
        self.backend = backend
        self.analyzer = analyzer
        self.lanes = list(lanes)
        self.assigner = LaneAssigner(self.lanes, min_conf=min_conf, min_visible=min_visible)
        self.counters: dict[str, RepCounter] = {
            lane.name: RepCounter(analyzer.counter_config()) for lane in self.lanes
        }

    def retune(self, tuning: dict[str, float]):
        """New counting rules while counting (Settings or the station screen's tune panel): the
        analyzer's rules and every lane's thresholds change at once; counts and reps so far stay.
        Returns the new counter config."""
        from dataclasses import replace

        from .exercises import RULES_CLASS, check_tuning
        name = self.analyzer.name
        clean = check_tuning({name: tuning or {}}).get(name, {})
        self.analyzer.rules = replace(RULES_CLASS[name](), **clean)
        cfg = self.analyzer.counter_config()
        for counter in self.counters.values():
            counter.cfg = cfg
        return cfg

    def process(self, frame_bgr: np.ndarray, t: float) -> MultiLaneResult:
        h, w = frame_bgr.shape[:2]
        people = self.backend.detect(frame_bgr)
        assigned = self.assigner.assign(people, w, h)
        results: dict[str, FrameResult] = {}
        for lane in self.lanes:
            counter = self.counters[lane.name]
            person = assigned.get(lane.name)
            metrics = (
                self.analyzer.analyze(person) if person is not None else FrameMetrics(metric=None)
            )
            event = counter.update(metrics.metric, t, metrics.faults, metrics.height)
            results[lane.name] = FrameResult(
                t=t,
                person=person,
                metrics=metrics,
                event=event,
                count=counter.count,
                rejected=counter.rejected,
                phase=counter.phase,
                lane=lane.name,
                pending_faults=counter.pending_faults,
                work_seconds=counter.work_seconds(t),
            )
        return MultiLaneResult(t=t, lanes=results, people=people)

    def summary(self) -> dict:
        return {
            lane.name: {
                "count": c.count,
                "rejected": c.rejected,
                "events": [e.__dict__ for e in c.events],
            }
            for lane in self.lanes
            for c in (self.counters[lane.name],)
        }


class StationSession:
    """One candidate: the whole frame is a single lane."""

    LANE = "1"

    def __init__(self, backend: PoseBackend, analyzer: ExerciseAnalyzer) -> None:
        self._multi = MultiLaneSession(backend, analyzer, [whole_frame(self.LANE)])
        self.backend = backend
        self.analyzer = analyzer

    @property
    def counter(self) -> RepCounter:
        return self._multi.counters[self.LANE]

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        return self._multi.process(frame_bgr, t).lanes[self.LANE]
