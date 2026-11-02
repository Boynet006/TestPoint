"""Evidence video: full recording per attempt and per-rep clips cut from the ring buffer.

With ffmpeg on PATH the files are H.264 (browser-playable); without it the writer falls
back to OpenCV's mp4v. The frame-count tests pass either way, the codec tests skip."""

import cv2
import numpy as np
import pytest
from test_engine import PokeCounter, ScriptedSession, ScriptedSource, rep, result

from testpoint import evidence
from testpoint.engine import Reporter, run
from testpoint.evidence import EvidenceWriter, find_ffmpeg
from testpoint.vision.counter import RepCounterConfig
from testpoint.vision.lanes import equal_strips, whole_frame

CFG = RepCounterConfig(enter_work=95, enter_rest=155)


def frame(i, w=96, h=48):
    f = np.zeros((h, w, 3), np.uint8)
    f[:, :, 1] = i % 256
    return f


def n_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def size(path):
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def fourcc(path):
    cap = cv2.VideoCapture(str(path))
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return bytes(code >> (8 * i) & 255 for i in range(4)).decode()


def write_one_clip(tmp_path, w=48, h=48, **kw):
    ev = EvidenceWriter(tmp_path, [whole_frame("1")], fps=10, keep_s=1.0, pad_s=0, day="d", **kw)
    ev.start("1", 1)
    for i in range(5):
        ev.push(frame(i, w, h), i / 10)
    ev.clip("1", {"index": 1, "t_start": 0.0, "t_end": 0.4})
    ev.close()
    return tmp_path / "d/1/rep-001.mp4", ev


def test_full_recording_and_rep_clips_per_lane(tmp_path):
    lanes = equal_strips(2)                       # two 48 px wide strips of a 96 px frame
    ev = EvidenceWriter(tmp_path, lanes, fps=10, keep_s=1.0, pad_s=0.1, day="2026-10-20")
    assert ev.clip("1", {"index": 1, "t_start": 0, "t_end": 0.5}) is None     # nobody running
    ev.start("1", 17)
    for i in range(21):                           # 2.1 s at 10 fps
        ev.push(frame(i), i / 10)
    # rep from 1.2 s to 1.9 s, cut as it ends -> with 0.1 s padding frames 11..20 (10 frames)
    clip = ev.clip("1", {"index": 2, "t_start": 1.2, "t_end": 1.9})
    assert clip == "2026-10-20/17/rep-002.mp4"
    assert ev.clip("2", {"index": 1, "t_start": 1.2, "t_end": 1.9}) is None  # lane 2 idle
    for i in range(21, 30):
        ev.push(frame(i), i / 10)
    ev.start("2", 18)
    ev.push(frame(30), 3.0)
    ev.stop("1")
    ev.push(frame(31), 3.1)                       # lane 1 stopped: not in its recording
    ev.close()

    assert n_frames(tmp_path / "2026-10-20/17/full.mp4") == 31
    assert size(tmp_path / "2026-10-20/17/full.mp4") == (48, 48)
    assert n_frames(tmp_path / "2026-10-20/17/rep-002.mp4") == 10
    assert n_frames(tmp_path / "2026-10-20/18/full.mp4") == 2
    assert not (tmp_path / "2026-10-20/18/rep-001.mp4").exists()


def test_ring_buffer_is_bounded_and_old_frames_are_gone(tmp_path):
    ev = EvidenceWriter(tmp_path, [whole_frame("1")], fps=10, keep_s=0.5, pad_s=0, day="d")
    ev.start("1", 1)
    for i in range(20):
        ev.push(frame(i), i / 10)
    assert len(ev._ring) == 6                     # 0.5 s window at 10 fps, inclusive
    assert ev.clip("1", {"index": 1, "t_start": 0.0, "t_end": 0.3}) == "d/1/rep-001.mp4"
    ev.close()
    assert not (tmp_path / "d/1/rep-001.mp4").exists()   # nothing left in the window: no file


def test_run_loop_records_the_attempt_and_names_the_clip(tmp_path):
    ev = EvidenceWriter(tmp_path, [whole_frame("1")], fps=10, day="d")
    rp = Reporter("pushup", "clip.mp4", ["1"], clip_for=ev.clip)
    rp.on_reply({"lanes": {"1": {"attempt_id": 7}}, "acked": {}})
    ses = ScriptedSession([result(0.0, **{"1": {}}),
                           result(0.1, **{"1": {"count": 1, "event": rep(1, t_start=0.0, t_end=0.1)}}),
                           result(0.2, **{"1": {"count": 1}})])
    assert run(ses, ScriptedSource(3), rp, PokeCounter(), CFG, log=lambda _: None, evidence=ev) == 3
    assert rp.snapshot()["lanes"]["1"]["events"][0]["clip"] == "d/7/rep-001.mp4"
    # The app ends the attempt: the recording closes, later frames are not written.
    rp.on_reply({"lanes": {"1": {"attempt_id": None}}, "acked": {"1": 1}})
    ses.results = [result(0.3, **{"1": {}})]
    run(ses, ScriptedSource(1), rp, PokeCounter(), CFG, log=lambda _: None, evidence=ev)
    ev.close()
    assert n_frames(tmp_path / "d/7/full.mp4") == 3
    assert n_frames(tmp_path / "d/7/rep-001.mp4") == 2


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not on PATH")
def test_with_ffmpeg_clips_are_h264_with_moov_first(tmp_path):
    logs = []
    path, ev = write_one_clip(tmp_path, log=logs.append)
    assert ev.ffmpeg and any("H.264" in m and "ffmpeg" in m for m in logs)
    assert fourcc(path) == "h264"
    assert n_frames(path) == 5
    data = path.read_bytes()
    assert 0 < data.find(b"moov") < data.find(b"mdat")      # +faststart: header before media
    assert fourcc(tmp_path / "d/1/full.mp4") == "h264"


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not on PATH")
def test_odd_lane_sizes_are_padded_to_even(tmp_path):
    path, _ = write_one_clip(tmp_path, w=47, h=45)
    assert size(path) == (48, 46)
    assert n_frames(path) == 5


def test_without_ffmpeg_falls_back_to_mp4v(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "find_ffmpeg", lambda: None)
    logs = []
    path, ev = write_one_clip(tmp_path, log=logs.append)
    assert ev.ffmpeg is None and any("ffmpeg not found" in m for m in logs)
    assert fourcc(path) == "FMP4"
    assert n_frames(path) == 5


def test_broken_ffmpeg_path_falls_back_to_mp4v(tmp_path):
    logs = []
    path, _ = write_one_clip(tmp_path, ffmpeg=str(tmp_path / "no-such-ffmpeg"), log=logs.append)
    assert any("falling back to mp4v" in m for m in logs)
    assert fourcc(path) == "FMP4"
    assert n_frames(path) == 5


def test_clip_at_cuts_from_the_ring_without_a_recording(tmp_path):
    ev = EvidenceWriter(tmp_path, [whole_frame("1")], fps=10, keep_s=2.0, pad_s=0.0, day="d")
    for i in range(20):
        ev.push(frame(i), i / 10)
    assert ev.clip_at("run", "000001200-t1", 1.0, 1.5) == "run/000001200-t1.mp4"
    assert ev.clip_at("run", "none", 5.0, 6.0) is None          # nothing in that window
    ev.close()
    assert n_frames(tmp_path / "run/000001200-t1.mp4") == 6      # t = 1.0 .. 1.5 inclusive
    assert size(tmp_path / "run/000001200-t1.mp4") == (96, 48)   # whole frame, no crop


def test_recording_plays_at_real_speed_when_frames_come_slower(tmp_path):
    ev = EvidenceWriter(tmp_path, [whole_frame("1")], fps=20, keep_s=5.0, pad_s=0, day="d")
    folder = ev.start("1", 7)
    frame = np.zeros((48, 64, 3), np.uint8)
    for i in range(11):             # 10 frames a second for one second
        ev.push(frame, i * 0.1)
    ev.close()
    cap = cv2.VideoCapture(str(folder / "full.mp4"))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 21      # 1 s at 20 fps, plus the first frame


def test_a_number_used_again_keeps_the_earlier_clips(tmp_path):
    ev = EvidenceWriter(tmp_path, [whole_frame("1")], fps=10, day="d")
    frame = np.zeros((48, 64, 3), np.uint8)
    folder = ev.start("1", 1)
    ev.push(frame, 0.0)
    ev.stop("1")
    ev.close()
    ev2 = EvidenceWriter(tmp_path, [whole_frame("1")], fps=10, day="d")
    again = ev2.start("1", 1)
    ev2.push(frame, 0.0)
    ev2.close()
    assert again == folder and (folder / "full.mp4").exists()
    assert (folder.with_name("1-redone-1") / "full.mp4").exists()
