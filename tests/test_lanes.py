import itertools
import json
import math

import numpy as np
import pytest

from testpoint.vision import keypoints as K
from testpoint.vision.exercises import PushupAnalyzer
from testpoint.vision.keypoints import Person
from testpoint.vision.lanes import (
    Lane,
    LaneAssigner,
    anchor_point,
    equal_strips,
    load_lanes,
    save_lanes,
    whole_frame,
)
from testpoint.vision.session import MultiLaneSession, StationSession


def person_from(points: dict[int, tuple[float, float]], conf: float = 0.9) -> Person:
    xy = np.zeros((17, 2), dtype=np.float32)
    c = np.zeros(17, dtype=np.float32)
    for i, p in points.items():
        xy[i] = p
        c[i] = conf
    return Person(xy, c)


def pushup_person(xc: float, elbow_deg: float, scale: float = 1.0) -> Person:
    """Side-view push-up body whose hip midpoint is at x = xc + 100 * scale."""
    th = math.radians(elbow_deg)
    s = scale
    return person_from({
        K.L_SHOULDER: (xc, 100 * s),
        K.L_ELBOW: (xc, 160 * s),
        K.L_WRIST: (xc + 60 * s * math.sin(th), 160 * s - 60 * s * math.cos(th)),
        K.L_HIP: (xc + 100 * s, 110 * s),
        K.R_HIP: (xc + 100 * s, 112 * s),
        K.L_ANKLE: (xc + 230 * s, 123 * s),
    })


# ----------------------------------------------------------------------------- Lane

def test_lane_contains_and_pixels():
    lane = Lane("a", 0.25, 0.0, 0.5, 1.0)
    assert lane.contains(0.3, 0.5)
    assert lane.contains(0.25, 0.0)      # borders included
    assert not lane.contains(0.6, 0.5)
    assert lane.to_pixels(1920, 1080) == (480, 0, 960, 1080)


def test_lane_validation():
    with pytest.raises(ValueError):
        Lane("x", 0.5, 0.0, 0.2, 1.0)   # x1 < x0
    with pytest.raises(ValueError):
        Lane("x", 0.0, 0.0, 1.5, 1.0)   # outside 0..1


def test_equal_strips_tile_the_frame():
    lanes = equal_strips(3)
    assert [lane.name for lane in lanes] == ["1", "2", "3"]
    assert lanes[0].x0 == 0.0 and lanes[-1].x1 == 1.0
    for a, b in itertools.pairwise(lanes):
        assert a.x1 == b.x0
    with pytest.raises(ValueError):
        equal_strips(0)


def test_load_and_save_lanes(tmp_path):
    path = tmp_path / "lanes.json"
    path.write_text(json.dumps({"lanes": [
        {"name": "left", "rect": [0.0, 0.1, 0.4, 0.9]},
        {"name": "right", "x0": 0.5, "y0": 0.1, "x1": 1.0, "y1": 0.9},
    ]}))
    lanes = load_lanes(path)
    assert [lane.name for lane in lanes] == ["left", "right"]
    assert lanes[1].rect == (0.5, 0.1, 1.0, 0.9)

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"rect": [0, 0, 0.5, 1]}, {"rect": [0.5, 0, 1, 1]}]))
    assert [lane.name for lane in load_lanes(bare)] == ["1", "2"]

    out = tmp_path / "out.json"
    save_lanes(lanes, out)
    assert load_lanes(out) == lanes

    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([{"name": "a", "rect": [0, 0, 0.5, 1]},
                               {"name": "a", "rect": [0.5, 0, 1, 1]}]))
    with pytest.raises(ValueError):
        load_lanes(dup)


# ----------------------------------------------------------------------- assignment

def test_anchor_is_hip_midpoint_with_bbox_fallback():
    p = person_from({K.L_HIP: (100, 200), K.R_HIP: (120, 220), K.NOSE: (0, 0)})
    assert anchor_point(p) == (110.0, 210.0)

    no_hips = person_from({K.NOSE: (0, 0), K.L_ANKLE: (200, 400)})
    assert anchor_point(no_hips) == (100.0, 200.0)

    one_hip = person_from({K.L_HIP: (100, 200), K.NOSE: (0, 0)})
    assert anchor_point(one_hip) == (50.0, 100.0)

    assert anchor_point(person_from({})) is None


def test_people_are_assigned_by_hip_midpoint():
    lanes = equal_strips(3)
    assigner = LaneAssigner(lanes)
    a = pushup_person(50, 170)     # hips at x=150 -> lane 1 of 900 px
    b = pushup_person(650, 170)    # hips at x=750 -> lane 3
    assigned = assigner.assign([b, a], width=900, height=300)
    assert assigned == {"1": a, "3": b}


def test_people_outside_all_lanes_are_ignored():
    lanes = [Lane("mat", 0.0, 0.5, 1.0, 1.0)]      # bottom half only
    top = pushup_person(50, 170)                   # hips at y=111 of 300 -> top half
    assert LaneAssigner(lanes).assign([top], 900, 300) == {}
    # the same person lower in the frame lands in the lane
    low = pushup_person(50, 170)
    low.xy[:, 1] += 150
    assert LaneAssigner(lanes).assign([low], 900, 300) == {"mat": low}


def test_largest_person_wins_a_shared_lane_and_ghosts_are_dropped():
    lanes = [whole_frame("all")]
    small = pushup_person(50, 170, scale=0.5)
    big = pushup_person(50, 170, scale=1.0)
    ghost = pushup_person(50, 170, scale=2.0)
    ghost.conf[:] = 0.1
    two_points = person_from({K.L_HIP: (10, 10), K.R_HIP: (20, 20)})
    assigned = LaneAssigner(lanes).assign([small, ghost, big, two_points], 900, 300)
    assert assigned == {"all": big}


# -------------------------------------------------------------------------- session

class ScriptedBackend:
    """Pose backend that returns a prepared list of people per frame."""

    def __init__(self, frames: list[list[Person]]) -> None:
        self.frames = frames
        self.calls = 0

    def detect(self, frame_bgr):
        people = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return people


def elbow_track(n_reps: int, frames_per_rep: int = 30, top: float = 170.0, bottom: float = 70.0):
    seq = [top] * 10
    for _ in range(n_reps):
        for k in range(frames_per_rep):
            seq.append(top - (top - bottom) * (1 - math.cos(2 * math.pi * k / frames_per_rep)) / 2)
    return seq + [top] * 10


def test_multilane_session_counts_each_lane_independently():
    fast = elbow_track(4, frames_per_rep=30)
    slow = elbow_track(2, frames_per_rep=60)
    n = max(len(fast), len(slow))
    fast += [170.0] * (n - len(fast))
    slow += [170.0] * (n - len(slow))
    frames = []
    for i in range(n):
        frames.append([
            pushup_person(50, fast[i]),      # lane 1
            pushup_person(650, slow[i]),     # lane 3
            pushup_person(1000, 170),        # outside the frame width -> ignored
        ])
    session = MultiLaneSession(ScriptedBackend(frames), PushupAnalyzer(), equal_strips(3))
    frame = np.zeros((300, 900, 3), dtype=np.uint8)
    last = None
    for i in range(n):
        last = session.process(frame, i / 30.0)
    assert session.backend.calls == n            # one detection per frame
    assert list(last.lanes) == ["1", "2", "3"]
    assert last.lanes["1"].count == 4
    assert last.lanes["2"].count == 0 and last.lanes["2"].person is None
    assert last.lanes["3"].count == 2
    assert session.summary()["1"]["count"] == 4
    assert len(session.summary()["3"]["events"]) == 2


def test_station_session_is_a_single_whole_frame_lane():
    track = elbow_track(3)
    frames = [[pushup_person(50, m)] for m in track]
    session = StationSession(ScriptedBackend(frames), PushupAnalyzer())
    frame = np.zeros((300, 900, 3), dtype=np.uint8)
    res = None
    for i in range(len(track)):
        res = session.process(frame, i / 30.0)
    assert res.count == 3
    assert session.counter.count == 3
    assert res.lane == "1"
