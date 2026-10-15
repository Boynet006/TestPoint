import numpy as np

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
