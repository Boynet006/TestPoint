"""Real recordings from the station camera, replayed through the real session and counter. The
true counts are known: 10 push-ups and 10 crunches, with getting into position, walking away and
sitting at a desk around them. The files hold the poses only, no pictures."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from testpoint.vision import keypoints as K
from testpoint.vision.exercises import PushupAnalyzer, make_analyzer
from testpoint.vision.keypoints import Person
from testpoint.vision.lanes import whole_frame
from testpoint.vision.session import MultiLaneSession

DATA = Path(__file__).parent / "data"


class Recorded:
    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def detect(self, _frame):
        people = self.frames[self.i]
        self.i += 1
        return [Person(np.array(xy, float), np.array(conf, float)) for xy, conf in people]


def replay(name: str, exercise: str):
    rec = json.loads(gzip.open(DATA / name, "rt", encoding="utf-8").read())
    session = MultiLaneSession(Recorded(rec["frames"]), make_analyzer(exercise), [whole_frame("S")])
    frame = np.zeros((int(rec["size"][1]), int(rec["size"][0]), 3), np.uint8)
    for i in range(len(rec["frames"])):
        session.process(frame, i * rec["dt"])
    return session.counters["S"]


@pytest.mark.parametrize(("name", "exercise"), [("pushup-10.json.gz", "pushup"), ("crunch-10.json.gz", "crunch")])
def test_every_real_rep_counts_and_nothing_else(name, exercise):
    counter = replay(name, exercise)
    assert (counter.count, counter.rejected) == (10, 0)


def _person(shoulder, hip, knee, ankle, elbow=(0, 0), wrist=(0, 0)) -> Person:
    xy = np.zeros((17, 2))
    conf = np.zeros(17)
    for idx, point in ((K.L_SHOULDER, shoulder), (K.L_ELBOW, elbow), (K.L_WRIST, wrist), (K.L_HIP, hip),
                       (K.L_KNEE, knee), (K.L_ANKLE, ankle)):
        xy[idx] = point
        conf[idx] = 0.9
    return Person(xy, conf)


def test_sitting_or_standing_is_not_measured():
    # torso upright, arm straight: without the check this looks like the top of a push-up
    m = PushupAnalyzer().analyze(_person(shoulder=(100, 50), hip=(105, 200), knee=(110, 300), ankle=(115, 400),
                                         elbow=(100, 100), wrist=(100, 150)))
    assert m.metric is None


def test_crouching_is_not_measured():
    # torso low but the knee sharply bent: getting up or down
    m = PushupAnalyzer().analyze(_person(shoulder=(100, 150), hip=(200, 170), knee=(120, 230), ankle=(210, 240),
                                         elbow=(100, 200), wrist=(100, 250)))
    assert m.metric is None


def test_shoulders_coming_down_count_when_the_elbow_cannot_be_read():
    # the elbow keeps reading 150 degrees; only the shoulders show the reps
    analyzer = PushupAnalyzer()
    session_counter = make_analyzer("pushup")
    from testpoint.vision.counter import RepCounter

    counter = RepCounter(session_counter.counter_config())
    t = 0.0
    for _ in range(3):
        for height in [0.38] * 8 + [0.22] * 8:
            counter.update(150.0 if height < 0.3 else 170.0, t, (), height)
            t += 0.06
    counter.update(170.0, t, (), 0.38)
    counter.update(170.0, t + 0.06, (), 0.38)
    assert counter.count == 3
    assert analyzer.rules.min_drop == pytest.approx(0.12)


def test_a_small_shoulder_movement_is_not_a_rep():
    from testpoint.vision.counter import RepCounter

    counter = RepCounter(make_analyzer("pushup").counter_config())
    t = 0.0
    for _ in range(4):
        for height in [0.38] * 8 + [0.31] * 8:        # 7 % of the body: well under the limit
            counter.update(170.0, t, (), height)
            t += 0.06
    assert (counter.count, counter.rejected) == (0, 0)
