import numpy as np
import pytest

from testpoint.vision import keypoints as K
from testpoint.vision.exercises import CrunchAnalyzer, PushupAnalyzer, make_analyzer
from testpoint.vision.keypoints import Person
from testpoint.vision.session import select_candidate


def person_from(points: dict[int, tuple[float, float]], conf: float = 0.9) -> Person:
    xy = np.zeros((17, 2), dtype=np.float32)
    c = np.zeros(17, dtype=np.float32)
    for i, p in points.items():
        xy[i] = p
        c[i] = conf
    return Person(xy, c)


def test_pushup_top_position_is_extended_and_straight():
    # side view, facing left; y grows downward
    p = person_from({
        K.L_SHOULDER: (100, 100), K.L_ELBOW: (100, 160), K.L_WRIST: (100, 220),
        K.L_HIP: (200, 110), K.L_ANKLE: (330, 123),
    })
    m = PushupAnalyzer().analyze(p)
    assert m.metric is not None and m.metric > 170
    assert m.faults == []


def test_pushup_bottom_position_with_sagging_hips_is_faulted():
    p = person_from({
        K.L_SHOULDER: (100, 180), K.L_ELBOW: (150, 200), K.L_WRIST: (110, 220),
        K.L_HIP: (200, 230), K.L_ANKLE: (330, 190),
    })
    m = PushupAnalyzer().analyze(p)
    assert m.metric is not None and m.metric < 95
    assert "body_not_straight" in m.faults


def test_pushup_without_arm_keypoints_is_unmeasurable():
    p = person_from({K.L_HIP: (0, 0), K.L_ANKLE: (10, 0)})
    assert PushupAnalyzer().analyze(p).metric is None


def test_crunch_lying_then_up():
    lying = person_from({
        K.R_HIP: (200, 300), K.R_SHOULDER: (100, 295),
        K.R_KNEE: (250, 250), K.R_ANKLE: (300, 300),
    })
    up = person_from({
        K.R_HIP: (200, 300), K.R_SHOULDER: (130, 230),
        K.R_KNEE: (250, 250), K.R_ANKLE: (300, 300),
    })
    a = CrunchAnalyzer()
    assert a.analyze(lying).metric < 15
    assert a.analyze(up).metric > 35
    assert a.analyze(up).faults == []


def test_crunch_with_straight_legs_is_faulted():
    up = person_from({
        K.R_HIP: (200, 300), K.R_SHOULDER: (130, 230),
        K.R_KNEE: (260, 300), K.R_ANKLE: (320, 300),
    })
    assert "knees_not_bent" in CrunchAnalyzer().analyze(up).faults


def test_select_largest_confident_person():
    def body(scale, conf=0.9):
        return person_from({
            K.L_SHOULDER: (0, 0), K.L_HIP: (scale, scale),
            K.L_KNEE: (scale * 1.5, scale), K.L_ANKLE: (scale * 2, scale),
        }, conf=conf)

    small, big, ghost = body(10), body(100), body(500, conf=0.1)
    two_points_only = person_from({K.L_SHOULDER: (0, 0), K.L_HIP: (900, 900)})
    assert select_candidate([small, big, ghost, two_points_only]) is big
    assert select_candidate([]) is None


def test_make_analyzer_names():
    assert make_analyzer("pushup").name == "pushup"
    assert make_analyzer("crunch").name == "crunch"


def _pushup_low(knee: tuple[float, float], ankle: tuple[float, float]) -> Person:
    # side view facing left, y grows downwards, elbows bent (bottom of the rep)
    return person_from({
        K.L_SHOULDER: (100, 180), K.L_ELBOW: (150, 200), K.L_WRIST: (110, 220),
        K.L_HIP: (200, 190), K.L_KNEE: knee, K.L_ANKLE: ankle,
    })


def test_straight_body_on_the_toes_is_not_a_knee_fault():
    # knee half way between hip and ankle heights: knee lift ~0.5, knee straight
    m = PushupAnalyzer().analyze(_pushup_low(knee=(265, 200), ankle=(330, 210)))
    assert m.debug["knee_lift"] == pytest.approx(0.5)
    assert m.debug["knee_deg"] > 170
    assert "knee_on_ground" not in m.faults and "body_not_straight" not in m.faults


def test_knee_push_up_with_shins_on_the_floor_is_faulted():
    # knee lower than the ankle (shin on the floor, toes tucked): lift about -0.7, as measured on clips
    m = PushupAnalyzer().analyze(_pushup_low(knee=(255, 228), ankle=(330, 212)))
    assert m.debug["knee_lift"] < -0.3 and m.debug["knee_deg"] < 165
    assert "knee_on_ground" in m.faults


def test_knee_push_up_with_feet_raised_is_faulted():
    # ankle above the hip: the lift ratio is undefined, the knee is below hip and ankle
    m = PushupAnalyzer().analyze(_pushup_low(knee=(250, 225), ankle=(300, 170)))
    assert "knee_lift" not in m.debug and m.debug["knee_drop"] > 0.15
    assert "knee_on_ground" in m.faults


def test_low_knee_with_a_straight_leg_is_not_a_knee_fault():
    # the knee-lift test only applies to a bent knee (camera angle can put a straight knee low)
    m = PushupAnalyzer().analyze(_pushup_low(knee=(265, 214), ankle=(330, 238)))
    assert m.debug["knee_deg"] > 165
    assert "knee_on_ground" not in m.faults


def test_one_frame_fault_does_not_reject_but_a_sustained_one_does():
    from testpoint.vision.counter import RepCounter

    a = PushupAnalyzer()
    cfg = a.counter_config()
    assert cfg.min_fault_seconds == 0.2

    def run(fault_frames: int) -> RepCounter:
        c = RepCounter(cfg)
        t = 0.0
        seq = [170] * 3 + [120] * 3 + [80] * 6 + [120] * 3 + [170] * 3
        for i, angle in enumerate(seq):
            faults = ["body_not_straight"] if 6 <= i < 6 + fault_frames else []
            c.update(angle, t, faults)
            t += 1 / 30
        return c

    assert (run(1).count, run(1).rejected) == (1, 0)      # a single bad frame (pose jitter)
    assert (run(5).count, run(5).rejected) == (1, 0)      # 0.13 s
    c = run(8)                                             # 0.23 s: a real fault
    assert (c.count, c.rejected) == (0, 1) and c.events[0].faults == ["body_not_straight"]


def test_pushup_counter_config_carries_rest_limit():
    cfg = PushupAnalyzer().counter_config()
    assert cfg.max_work_seconds == 3.0
    assert cfg.work_timeout_fault == "rest_in_down"



def test_tuned_analyzer_uses_the_new_rules():
    from testpoint.vision.exercises import make_analyzer
    a = make_analyzer("pushup", {"down_elbow_deg": 88})
    assert a.rules.down_elbow_deg == 88 and a.counter_config().enter_work == 88
    assert make_analyzer("crunch").rules.up_torso_deg == 15
