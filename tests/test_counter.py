import math

import pytest

from testpoint.vision.counter import Phase, RepCounter, RepCounterConfig


def pushup_cfg(**kw):
    base = dict(enter_work=90.0, enter_rest=155.0, work_is_below=True,
                min_work_frames=2, min_rest_frames=2)
    base.update(kw)
    return RepCounterConfig(**base)


def run(counter, metrics, faults_at=None):
    """Feed a metric sequence at 30 fps; faults_at maps frame index -> faults."""
    faults_at = faults_at or {}
    events = []
    for i, m in enumerate(metrics):
        ev = counter.update(m, i / 30.0, faults_at.get(i, ()))
        if ev:
            events.append(ev)
    return events


def elbow_reps(n, frames_per_rep=30, top=170.0, bottom=70.0):
    """Smooth elbow angle going top -> bottom -> top, n times, starting at top."""
    seq = [top] * 10
    for _ in range(n):
        for k in range(frames_per_rep):
            phase = 2 * math.pi * k / frames_per_rep
            seq.append(top - (top - bottom) * (1 - math.cos(phase)) / 2)
    seq += [top] * 10
    return seq


def test_counts_clean_reps():
    c = RepCounter(pushup_cfg())
    events = run(c, elbow_reps(5))
    assert c.count == 5
    assert c.rejected == 0
    assert [e.index for e in events] == [1, 2, 3, 4, 5]
    assert all(e.valid for e in events)
    assert events[-1].count_after == 5


def test_jitter_around_threshold_does_not_double_count():
    c = RepCounter(pushup_cfg())
    seq = [170.0] * 5
    # go down and hover noisily right around the work threshold
    seq += [92, 88, 91, 87, 93, 86, 89, 90, 88]
    seq += [120, 150, 158, 170, 170]
    run(c, seq)
    assert c.count == 1


def test_getting_into_position_from_the_floor_does_not_count():
    c = RepCounter(pushup_cfg())
    # candidate starts lying down (elbows bent) and pushes up once to the start position
    seq = [60.0] * 20 + [100, 140, 170, 170, 170]
    run(c, seq)
    assert c.count == 0
    assert c.phase is Phase.REST


def test_half_depth_rep_is_not_counted():
    c = RepCounter(pushup_cfg())
    seq = [170.0] * 5 + [140, 120, 110, 105, 110, 130, 160, 170, 170]
    run(c, seq)
    assert c.count == 0
    assert c.rejected == 0


def test_fault_during_rep_rejects_it_with_reason():
    c = RepCounter(pushup_cfg())
    seq = elbow_reps(1)
    events = run(c, seq, faults_at={20: ["body_not_straight"]})
    assert c.count == 0
    assert c.rejected == 1
    assert events[0].valid is False
    assert events[0].faults == ["body_not_straight"]


def test_person_lost_resets_to_waiting():
    c = RepCounter(pushup_cfg(max_missing_frames=5))
    run(c, [170.0] * 5)
    assert c.phase is Phase.REST
    run(c, [None] * 6)
    assert c.phase is Phase.WAITING


def test_work_above_direction_for_crunches():
    cfg = RepCounterConfig(enter_work=35.0, enter_rest=15.0, work_is_below=False,
                           min_work_frames=2, min_rest_frames=2)
    c = RepCounter(cfg)
    seq = [5.0] * 5 + [20, 30, 40, 45, 40, 30, 20, 10, 5, 5] * 3
    run(c, seq)
    assert c.count == 3


def test_config_validation():
    with pytest.raises(ValueError):
        RepCounterConfig(enter_work=160.0, enter_rest=90.0, work_is_below=True)
    with pytest.raises(ValueError):
        RepCounterConfig(enter_work=10.0, enter_rest=30.0, work_is_below=False)
