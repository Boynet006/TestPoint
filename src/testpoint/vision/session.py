"""StationSession: one camera, one exercise, one candidate at a time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .counter import Phase, RepCounter, RepEvent
from .exercises import ExerciseAnalyzer, FrameMetrics
from .keypoints import Person
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


def select_candidate(
    people: list[Person], min_conf: float = 0.3, min_visible: int = 4
) -> Person | None:
    """Pick the person to score: the largest detection with enough confident keypoints.

    Phase 2 replaces this with a station zone drawn by the administrator.
    """
    best = None
    best_area = 0.0
    for p in people:
        if int((p.conf >= min_conf).sum()) < min_visible:
            continue
        a = p.area()
        if a > best_area:
            best, best_area = p, a
    return best


class StationSession:
    def __init__(self, backend: PoseBackend, analyzer: ExerciseAnalyzer) -> None:
        self.backend = backend
        self.analyzer = analyzer
        self.counter = RepCounter(analyzer.counter_config())

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        people = self.backend.detect(frame_bgr)
        person = select_candidate(people)
        metrics = self.analyzer.analyze(person) if person is not None else FrameMetrics(metric=None)
        event = self.counter.update(metrics.metric, t, metrics.faults)
        return FrameResult(
            t=t,
            person=person,
            metrics=metrics,
            event=event,
            count=self.counter.count,
            rejected=self.counter.rejected,
            phase=self.counter.phase,
        )
