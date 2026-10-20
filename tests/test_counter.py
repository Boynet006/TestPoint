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


def test_resting_in_the_down_position_rejects_the_rep():
    c = RepCounter(pushup_cfg(max_work_seconds=3.0))
    seq = [170.0] * 5 + [120, 95, 80, 70] + [70.0] * 120 + [100, 140, 160, 170, 170]
    events = run(c, seq)        # 120 frames at 30 fps = 4 s at the bottom
    assert c.count == 0
    assert c.rejected == 1
    assert events[0].faults == ["rest_in_down"]


def test_rest_half_way_up_also_counts_as_resting():
    c = RepCounter(pushup_cfg(max_work_seconds=3.0))
    seq = [170.0] * 5 + [120, 95, 80, 70, 70, 110, 120] + [120.0] * 100 + [160, 170, 170]
    run(c, seq)
    assert c.rejected == 1


def test_quick_rep_within_the_limit_is_valid_and_timer_is_exposed():
    c = RepCounter(pushup_cfg(max_work_seconds=3.0))
    seq = [170.0] * 5 + [120, 95, 80, 70] + [70.0] * 30 + [100, 140, 160, 170, 170]
    run(c, seq[:20])
    assert c.phase is Phase.WORK
    assert 0.3 < c.work_seconds(19 / 30.0) < 0.6
    run(c, seq[20:])
    assert c.count == 1 and c.rejected == 0
    assert c.pending_faults == []


def test_max_work_seconds_validation():
    with pytest.raises(ValueError):
        pushup_cfg(max_work_seconds=0)


# ------------------------------------------------- crunch features (CALIBRATION.md)

def crunch_cfg(**kw):
    base = dict(enter_work=7.0, enter_rest=4.0, work_is_below=False, min_work_frames=2,
                min_rest_frames=2, floor_window_s=6.0, rest_limit=40.0)
    base.update(kw)
    return RepCounterConfig(**base)


def crunch_reps(n, lying, rise=12.0, frames_per_rep=36, pause=6):
    """Torso elevation: lying level, then n crunches rising ``rise`` above it."""
    seq = [lying] * 45
    for _ in range(n):
        for k in range(frames_per_rep):
            seq.append(lying + rise * (1 - math.cos(2 * math.pi * k / frames_per_rep)) / 2)
        seq += [lying] * pause
    return seq


def test_relative_floor_counts_the_same_from_any_camera_angle():
    # side view lies at 0 degrees, a 3/4 view at 25: the same crunches count the same
    for lying in (0.0, 25.0):
        c = RepCounter(crunch_cfg())
        assert len(run(c, crunch_reps(5, lying))) == 5 and c.count == 5
        assert c.floor == pytest.approx(lying, abs=0.5)
    # with absolute thresholds the 3/4 view never returns to rest: nothing counts
    c = RepCounter(crunch_cfg(enter_work=35.0, enter_rest=15.0, floor_window_s=None))
    run(c, crunch_reps(5, 25.0))
    assert c.count == 0


def test_small_rise_is_not_a_crunch():
    c = RepCounter(crunch_cfg())
    run(c, crunch_reps(5, 0.0, rise=5.0))       # jitter-sized movement
    assert c.count == 0 and c.rejected == 0


def test_sitting_is_never_the_rest_position():
    # sitting up at 70 degrees and wobbling is not a series of crunches
    c = RepCounter(crunch_cfg())
    seq = [70.0 + (10.0 if (i // 20) % 2 else 0.0) for i in range(300)]
    run(c, seq)
    assert c.phase is Phase.WAITING and c.count == 0
    # lying down afterwards arms it, and real crunches count
    run(c, crunch_reps(3, 0.0))
    assert c.count == 3


def test_arming_needs_the_rest_position_held():
    # 0.5 s of lying is not enough with arm_seconds=1: the first movement is ignored
    c = RepCounter(crunch_cfg(arm_seconds=1.0))
    seq = [0.0] * 15 + [12.0] * 20 + [0.0] * 40 + [12.0] * 20 + [0.0] * 20
    run(c, seq)
    assert c.count == 1


def test_movements_shorter_than_min_rep_seconds_are_ignored():
    c = RepCounter(crunch_cfg(min_rep_seconds=0.4))
    seq = [0.0] * 60 + [12.0] * 4 + [0.0] * 30 + crunch_reps(1, 0.0)[45:]
    events = run(c, seq)
    assert c.count == 1 and c.rejected == 0 and len(events) == 1


def test_crunch_config_validation():
    with pytest.raises(ValueError):
        crunch_cfg(floor_window_s=0)


def test_restart_keeps_the_rest_level_so_the_first_crunch_after_start_counts():
    old = RepCounter(crunch_cfg(arm_seconds=1.0))
    before = crunch_reps(3, 20.0)
    run(old, before)                               # free-running before Start, lying at 20
    new = old.restart()
    assert new.count == 0 and new.phase is Phase.WAITING and new.floor == pytest.approx(old.floor)
    # crunching starts at once after Start: no still second, the first one still counts
    seq = crunch_reps(2, 20.0)[45:]
    t0 = len(before) / 30.0                        # same clock: Start is pressed right after
    for i, m in enumerate(seq):
        new.update(m, t0 + i / 30.0)
    assert new.count == 2
    # a counter started from nothing loses that first crunch
    blank = RepCounter(crunch_cfg(arm_seconds=1.0))
    for i, m in enumerate(seq):
        blank.update(m, t0 + i / 30.0)
    assert blank.count < 2


def test_next_rep_starts_where_the_last_one_ended():
    # back to rest only for the minimum frames, never fully "in rest" afterwards: the second
    # rep's start must not repeat the first rep's start
    c = RepCounter(crunch_cfg())
    seq = [0.0] * 60 + [12.0] * 15 + [2.0] * 2 + [5.0] * 3 + [12.0] * 15 + [0.0] * 10
    events = run(c, seq)
    assert len(events) == 2 and events[1].t_start >= events[0].t_end
