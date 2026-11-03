"""Finish camera: tracker, line crossing with interpolation, bib vote, the pipeline on a
scripted detector, and the app's crossing endpoint. No model, no camera."""

from datetime import timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import app as app_mod
from testpoint.api import models
from testpoint.api.app import create_app
from testpoint.finishcam import Crossing, FinishCam
from testpoint.vision.bib import BibReader, BibVote, torso_crop
from testpoint.vision.detect import Box
from testpoint.vision.line import FinishLine, parse_line
from testpoint.vision.tracker import IoUTracker
from runhelp import start_lined_up


def box(x, y, w=40, h=100):
    return Box(x, y, x + w, y + h)


# ------------------------------------------------------------------- tracker

def test_tracker_follows_moving_runner_and_drops_lost_tracks():
    tr = IoUTracker(iou_threshold=0.3, max_missing=2)
    a = tr.update([box(100, 100)], 0.0)[0]
    assert a.id == 1 and not a.confirmed
    b = tr.update([box(104, 110)], 0.1)[0]
    assert b is a and a.confirmed and a.hits == 2 and len(a.history) == 2
    # a second runner far away is a new track
    seen = tr.update([box(108, 120), box(400, 100)], 0.2)
    assert sorted(t.id for t in seen) == [1, 2]
    # runner 1 disappears: kept for max_missing frames, then dropped
    tr.update([box(400, 110)], 0.3)
    tr.update([box(400, 120)], 0.4)
    assert [t.id for t in tr.tracks] == [1, 2]
    tr.update([box(400, 130)], 0.5)
    assert [t.id for t in tr.tracks] == [2]


# ---------------------------------------------------------------------- line

def test_line_crossing_is_interpolated_and_counted_once():
    line = FinishLine((0, 0.5, 1, 0.5), "down")       # horizontal line at mid height
    tr = IoUTracker()
    w, h = 1000, 1000
    # coming from y=400, feet at y=460 then 520: the line (500) is crossed 2/3 of the way between the frames
    tr.update([Box(100, 100, 140, 400)], 0.7)
    tr.update([Box(100, 160, 140, 460)], 1.0)
    tr.update([Box(100, 220, 140, 520)], 1.3)
    events = line.check(tr.tracks, w, h)
    assert len(events) == 1
    ev = events[0]
    assert ev.track_id == 1 and ev.t == pytest.approx(1.2) and (ev.t_before, ev.t_after) == (1.0, 1.3)
    # moving on past the line: no second crossing for the same track
    tr.update([Box(100, 300, 140, 600)], 1.6)
    assert line.check(tr.tracks, w, h) == []
    # a new track crossing upwards is ignored with direction "down"
    tr2 = IoUTracker()
    tr2.update([Box(500, 320, 540, 620)], 1.7)
    tr2.update([Box(500, 260, 540, 560)], 2.0)
    tr2.update([Box(500, 180, 540, 480)], 2.3)
    assert line.check(tr2.tracks, w, h) == []
    assert FinishLine((0, 0.5, 1, 0.5), "any").check(tr2.tracks, w, h)[0].t == pytest.approx(2.225)


def test_line_first_frame_and_unconfirmed_tracks_do_not_cross():
    line = FinishLine((0, 0.5, 1, 0.5), "down")
    tr = IoUTracker()
    tr.update([Box(0, 250, 40, 550)], 0.0)              # one frame only, already past the line
    assert line.check(tr.tracks, 100, 1000) == []
    tr.update([Box(0, 270, 40, 570)], 0.1)
    assert line.check(tr.tracks, 100, 1000) == []       # never seen before the line


def test_someone_sitting_or_shifting_on_the_line_is_not_a_crossing():
    """A person at a desk under the line: their box edge wobbles across it without coming from
    anywhere. Only someone who came from before the line and moved through it counts."""
    line = FinishLine((0, 0.5, 1, 0.5), "down")
    tr = IoUTracker()
    for i, y in enumerate((490, 497, 503, 495, 506, 499, 508)):
        tr.update([Box(100, y - 300, 140, y)], i * 0.1)
        assert line.check(tr.tracks, 1000, 1000) == []


def test_vertical_line_directions():
    line = FinishLine((0.5, 0, 0.5, 1), "right")
    tr = IoUTracker()
    tr.update([Box(360, 0, 520, 100)], -0.5)            # coming from x=440
    tr.update([Box(400, 0, 560, 100)], 0.0)
    tr.update([Box(440, 0, 600, 100)], 0.5)             # feet x 480 -> 520 across x=500
    ev = line.check(tr.tracks, 1000, 1000)
    assert len(ev) == 1 and ev[0].t == pytest.approx(0.25)
    assert parse_line("0,0.85,1,0.85") == (0, 0.85, 1, 0.85)
    with pytest.raises(ValueError):
        FinishLine((0, 0, 0, 0))
    with pytest.raises(ValueError):
        parse_line("1,2,3")


# ----------------------------------------------------------------------- bibs

def test_bib_vote_takes_the_consistent_number_or_nothing():
    v = BibVote(min_share=0.5)
    r = v.decide([("128", 0.9), ("128", 0.8), ("126", 0.4), ("RUN", 0.9)])
    assert (r.bib, r.confidence) == ("128", 0.81) and len(r.readings) == 3
    assert v.decide([]).bib is None
    assert v.decide([("abc", 0.9)]).bib is None
    split = v.decide([("12", 0.5), ("13", 0.5), ("14", 0.5)])
    assert split.bib is None and split.confidence == 0.33
    assert BibVote(max_digits=3).decide([("123456", 0.9)]).bib is None
    assert v.decide([("No 42 ok", 0.7)]).bib == "42"            # digits inside other text
    # a digit hidden by a hand: "3" is the start of "30", so it votes for 30
    r = v.decide([("3", 0.56), ("3", 0.67), ("30", 0.68), ("40", 0.6), ("3", 0.65)])
    assert r.bib == "30" and r.confidence == 0.81
    # ambiguous: "1" could be 12 or 21, so it stays on its own
    assert v.decide([("1", 0.9), ("12", 0.5), ("21", 0.5)]).bib is None


def test_bib_reader_keeps_the_largest_crops_and_reads_at_decision():
    seen = []

    def fake_read(crop):
        seen.append(crop.shape)
        return [("77", 0.9)]

    rd = BibReader(fake_read, BibVote(), max_crops=2)
    frame = np.zeros((400, 400, 3), np.uint8)
    rd.observe(1, frame, Box(0, 0, 40, 100))      # small
    rd.observe(1, frame, Box(0, 0, 80, 200))      # large
    rd.observe(1, frame, Box(0, 0, 60, 150))      # medium -> small one dropped
    res = rd.decide(1)
    assert res.bib == "77" and len(seen) == 2
    assert sorted(s[0] for s in seen) == [75, 100]   # torso: 15%..65% of 150 and 200
    assert rd.decide(1).bib is None                  # crops consumed
    assert torso_crop(frame, Box(0, 0, 100, 100)).shape == (50, 110, 3)


# ------------------------------------------------------------------- pipeline

class ScriptedDetector:
    def __init__(self, frames):
        self.frames = list(frames)

    def detect(self, frame):
        return self.frames.pop(0)


def test_finishcam_pipeline_reports_a_crossing_with_bib(tmp_path):
    frames = [[Box(100, 100, 140, 400)], [Box(100, 160, 140, 460)], [Box(100, 220, 140, 520)], [Box(100, 280, 140, 580)]]
    cam = FinishCam(ScriptedDetector(frames), FinishLine((0, 0.5, 1, 0.5), "down"),
                    BibReader(lambda crop: [("21", 0.9)]), log=lambda _: None)
    img = np.zeros((1000, 1000, 3), np.uint8)
    out = []
    for i in range(4):
        out.extend(cam.process(img, i * 0.1))
    assert len(out) == 1 and out[0].bib == "21" and out[0].t == pytest.approx(0.1 + 0.1 * 2 / 3)
    payload = out[0].payload(out[0].wall + 0.4)
    assert payload["bib"] == "21" and payload["latency_s"] == 0.4 and payload["readings"][0]["text"] == "21"
    assert payload["clip"] is None and "face" not in payload          # a confident bib: no face sent


def test_finishcam_sends_the_face_when_no_bib_is_read():
    frames = [[Box(100, 100, 300, 400)], [Box(100, 160, 300, 460)], [Box(100, 220, 300, 520)], [Box(100, 280, 300, 580)]]
    cam = FinishCam(ScriptedDetector(frames), FinishLine((0, 0.5, 1, 0.5), "down"),
                    BibReader(lambda crop: []), log=lambda _: None)
    img = np.full((1000, 1000, 3), 128, np.uint8)
    out = []
    for i in range(4):
        out.extend(cam.process(img, i * 0.1))
    # several snapshots, spaced in time, the newest last
    assert len(out) == 1 and out[0].bib is None and 1 <= len(out[0].face_jpegs) <= FinishCam.FACE_SNAPSHOTS
    assert all(j[:2] == bytes([0xFF, 0xD8]) for j in out[0].face_jpegs)
    import base64
    assert all(base64.b64decode(f)[:2] == bytes([0xFF, 0xD8]) for f in out[0].payload()["faces"])


# ------------------------------------------------------------------------ app

@pytest.fixture()
def client(monkeypatch):
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", tz="Asia/Dubai", web_dir="does-not-exist", today=lambda: "2026-10-20")
    with TestClient(app) as c:
        yield c


def ok(r, code=200):
    assert r.status_code == code, r.text
    return r.json()


def test_camera_crossing_lands_as_result_or_review(client, monkeypatch):
    ok(client.post("/api/entries/import", content=b"external_id,name,bib,gender,dob\nC-1,Ahmed Hassan,11,M,2000-01-01\nC-2,Mariam Ali,12,M,2000-01-01\n"))
    for bib in (11, 12):
        e = ok(client.post("/api/checkin", json={"station": "weigh", "query": str(bib)}))["entry"]
        ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
        ok(client.post(f"/api/entries/{e['id']}/line-up"))
    t0 = app_mod.S.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    ok(start_lined_up(client))
    # a crossing timed before anyone started: nobody was out, so it is not a runner (ignored)
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(seconds=30.0))
    r = client.post("/api/engine/crossing", json={"bib": "12", "latency_s": 59})
    assert r.status_code == 201 and r.json() == {"ignored": True, "reason": "nobody_running"}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(seconds=900.0))

    # known running bib, seen 0.8 s before the report: a result stamped at the true instant
    c = ok(client.post("/api/engine/crossing", json={"bib": "11", "confidence": 0.81, "latency_s": 0.8,
                                                     "clip": "run/000899200-t3.mp4"}), 201)
    assert c["time_s"] == 899.2 and c["source"] == "camera" and c["confidence"] == 0.81
    assert c["clip"] == "run/000899200-t3.mp4" and not c["needs_review"]
    assert c["runner"] == {"bib": "11", "name": "Ahmed Hassan"}          # who it was, for the camera's picture
    # unreadable bib: review queue, clip kept
    c = ok(client.post("/api/engine/crossing", json={"bib": None, "confidence": 0.0, "clip": "run/x.mp4"}), 201)
    assert c["needs_review"] and c["note_code"] == "no_bib_read" and c["clip"] == "run/x.mp4"
    # bib not on the list: review
    c = ok(client.post("/api/engine/crossing", json={"bib": "999", "confidence": 0.6}), 201)
    assert c["needs_review"] and c["note_code"] == "bib_not_on_list"
    rows = ok(client.get("/api/run/crossings"))
    assert [r["source"] for r in rows] == ["camera"] * 3


def test_crossing_clip_includes_the_second_after_the_line(tmp_path):
    from testpoint.evidence import EvidenceWriter
    from testpoint.vision.lanes import whole_frame

    ev = EvidenceWriter(tmp_path, [whole_frame("1")], fps=10, keep_s=4.0, pad_s=0.0, ffmpeg=None)
    boxes = [[Box(100, 100 + 60 * i, 140, 400 + 60 * i)] for i in range(4)] + [[] for _ in range(16)]
    got = []
    cam = FinishCam(ScriptedDetector(boxes), FinishLine((0, 0.5, 1, 0.5), "down"), None,
                    evidence=ev, on_crossing=got.append, log=lambda _: None)
    img = np.zeros((1000, 1000, 3), np.uint8)
    for i in range(20):
        cam.process(img, i * 0.1)
    cam.close()
    ev.close()
    assert len(got) == 1 and got[0].clip.startswith("run/")
    cap = cv2.VideoCapture(str(tmp_path / got[0].clip))
    n = 0
    while cap.read()[0]:
        n += 1
    # crossing at ~0.167 s: clip from 0 (no frames before) to ~1.167 s -> frames 0.0 .. 1.1
    assert n == 12


def test_async_ocr_reports_after_the_frame_loop_moves_on():
    import threading

    gate = threading.Event()
    got = []

    def slow_read(crop):
        gate.wait(2)
        return [("21", 0.9)]

    frames = [[Box(100, 100, 140, 400)], [Box(100, 160, 140, 460)], [Box(100, 220, 140, 520)], [Box(100, 280, 140, 580)]]
    cam = FinishCam(ScriptedDetector(frames), FinishLine((0, 0.5, 1, 0.5), "down"), BibReader(slow_read),
                    on_crossing=got.append, log=lambda _: None, ocr_async=True)
    img = np.zeros((1000, 1000, 3), np.uint8)
    for i in range(4):
        assert cam.process(img, i * 0.1) == []        # never blocked by the OCR
    assert got == []
    gate.set()
    cam.close()
    assert len(got) == 1 and got[0].bib == "21" and got[0].t == pytest.approx(0.1 + 0.1 * 2 / 3)
    assert got[0].payload(got[0].wall)["latency_s"] == 0.0     # stamped when the line was crossed


def test_crossing_payload_defaults():
    c = Crossing(t=12.5, bib=None, confidence=0.0, readings=[])
    p = c.payload(c.wall + 2)
    assert p == {"bib": None, "confidence": 0.0, "readings": [], "latency_s": 2.0, "t": 12.5, "clip": None, "track": 0}


def test_the_finish_keeps_the_clearest_faces_of_the_approach():
    """With a face finder: frames without a face are skipped, the clearest faces are kept."""
    import numpy as np
    sizes = iter([None, 50, 30, 90, None, 70, 60, 120, 45])   # face size found in each look (None: no face)

    def finder(img):
        s = next(sizes)
        return None if s is None else (10.0, 10.0, float(s), float(s), 0.9)

    cam = FinishCam(detector=None, line=FinishLine(), reader=None, face_finder=finder)
    frame = np.zeros((1000, 1000, 3), np.uint8)
    for i in range(9):
        cam._keep_head(1, frame, Box(100, 100, 500, 900), t=i * 0.2)
    kept = [q for q, _img in cam._heads[1]]
    assert len(kept) == FinishCam.FACE_SNAPSHOTS
    assert kept == sorted(kept, reverse=True) and kept[0] == 120 * 0.9   # 30 px is too small to keep
    assert all(q >= 45 * 0.9 for q in kept)


def test_a_test_video_plays_at_its_speed_while_the_analysis_takes_the_newest_frame():
    import threading
    from testpoint.finishcam import Analyser
    seen, gate = [], threading.Event()

    def slow(frame, t, jumped):
        seen.append((t, jumped))
        gate.wait(2)                         # the analysis is busy with frame 0
    a = Analyser(slow); a.start()
    a.offer(None, 0.0)
    assert a.first.wait(0.01) is False       # the first frame waits for the analysis (warm-up)
    for k in range(1, 5):
        a.offer(None, k * 0.04, jumped=(k == 2))   # meanwhile the video moves on, a seek at frame 2
    gate.set(); a.first.wait(2)
    for _ in range(500):                     # generous: a busy machine (cameras running) is slow to switch threads
        if len(seen) == 2:
            break
        threading.Event().wait(0.01)
    a.stop(); a.join(2)
    assert seen == [(0.0, False), (0.16, True)]   # the newest frame next, and the jump not lost
