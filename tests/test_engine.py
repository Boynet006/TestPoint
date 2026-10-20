"""The engine's report side (docs/ENGINE-LINK.md): events kept until acked, counters reset
on a new attempt, the sender keeps going when the app is down. No camera, no model."""

import threading
import urllib.error

import numpy as np

from testpoint.engine import Reporter, Sender, run
from testpoint.vision.counter import Phase, RepCounter, RepCounterConfig, RepEvent
from testpoint.vision.exercises import FrameMetrics
from testpoint.vision.session import FrameResult, MultiLaneResult

CFG = RepCounterConfig(enter_work=95, enter_rest=155)


def result(t, **lanes):
    """One MultiLaneResult; ``lanes`` maps name -> {"count": , "rejected": , "event": , "phase": }."""
    out = {}
    for name, v in lanes.items():
        out[name] = FrameResult(t=t, person=None, metrics=FrameMetrics(metric=None),
                                event=v.get("event"), count=v.get("count", 0),
                                rejected=v.get("rejected", 0), phase=v.get("phase", Phase.REST),
                                lane=name, pending_faults=v.get("faults", []),
                                work_seconds=v.get("work", 0.0))
    return MultiLaneResult(t=t, lanes=out, people=[])


def rep(index, valid=True, faults=(), t_start=0.0, t_end=1.0):
    return RepEvent(index=index, valid=valid, faults=list(faults), t_start=t_start, t_end=t_end,
                    count_after=index)


class FakeSession:
    def __init__(self, names):
        self.counters = {n: RepCounter(CFG) for n in names}


def test_report_keeps_events_until_acked_and_resets_on_new_attempt():
    rp = Reporter("pushup", "0", ["1", "2"])
    s = FakeSession(["1", "2"])
    c1 = s.counters["1"]

    new = rp.observe(result(1.0, **{"1": {"count": 1, "event": rep(1, t_start=0.2, t_end=0.9), "phase": Phase.WORK, "faults": ["knee_down"], "work": 1.25},
                                    "2": {}}), fps=30.04)
    assert new == [("1", {"index": 1, "valid": True, "faults": [], "t_start": 0.2, "t_end": 0.9})]
    snap = rp.snapshot()
    assert snap["station"] == "pushup" and snap["t"] == 1.0 and snap["fps"] == 30.0 and snap["source"] == "0"
    assert snap["lanes"]["1"] == {"attempt_id": None, "count": 1, "rejected": 0, "phase": "work",
                                  "person": False, "pending_faults": ["knee_down"], "work_seconds": 1.25,
                                  "events": [{"index": 1, "valid": True, "faults": [], "t_start": 0.2, "t_end": 0.9}]}
    assert snap["lanes"]["2"]["events"] == []

    # No ack: the event is resent with the next one.
    rp.observe(result(2.0, **{"1": {"count": 1, "rejected": 1, "event": rep(2, valid=False, faults=["hips_sag"])}}), 30)
    assert [e["index"] for e in rp.snapshot()["lanes"]["1"]["events"]] == [1, 2]
    rp.on_reply({"lanes": {"1": {"attempt_id": None}, "2": {"attempt_id": None}}, "acked": {"1": 1}})
    assert [e["index"] for e in rp.snapshot()["lanes"]["1"]["events"]] == [2]
    assert rp.sync(s, CFG) == [] and s.counters["1"] is c1          # nothing changed

    # The examiner started a test on lane 1: fresh counter, old events dropped.
    rp.on_reply({"lanes": {"1": {"attempt_id": 17}, "2": {"attempt_id": None}}, "acked": {}})
    assert rp.sync(s, CFG) == [("1", 17)]
    assert s.counters["1"] is not c1 and s.counters["1"].count == 0 and s.counters["1"].phase is Phase.WAITING
    lane = rp.snapshot()["lanes"]["1"]
    assert lane["attempt_id"] == 17 and lane["count"] == 0 and lane["events"] == [] and lane["phase"] == "waiting"
    assert rp.sync(s, CFG) == []                                      # applied once

    # Test over: counting continues for the overlay but for nobody.
    c2 = s.counters["1"]
    rp.on_reply({"lanes": {"1": {"attempt_id": None}}, "acked": {}})
    assert rp.sync(s, CFG) == [("1", None)] and s.counters["1"] is not c2
    # A lane the app did not mention (or an unknown one) is left alone.
    rp.on_reply({"lanes": {"9": {"attempt_id": 3}}, "acked": {"9": 4}})
    assert rp.sync(s, CFG) == []


def test_pending_events_are_capped():
    rp = Reporter("crunch", "0", ["1"], max_pending=3)
    for i in range(1, 6):
        rp.observe(result(float(i), **{"1": {"count": i, "event": rep(i)}}), 30)
    assert [e["index"] for e in rp.snapshot()["lanes"]["1"]["events"]] == [3, 4, 5]


class FlakyTransport:
    url = "http://app/api/engine/report"

    def __init__(self, failures=0):
        self.failures = failures
        self.reports = []
        self.got = threading.Event()

    def post(self, report):
        if self.failures:
            self.failures -= 1
            raise urllib.error.URLError("connection refused")
        self.reports.append(report)
        self.got.set()
        return {"lanes": {"1": {"attempt_id": 5}}, "acked": {"1": max((e["index"] for e in report["lanes"]["1"]["events"]), default=0)}}


def test_sender_retries_logs_once_and_applies_the_reply():
    rp = Reporter("pushup", "0", ["1"])
    tr = FlakyTransport(failures=2)
    logs = []
    sd = Sender(rp, tr, interval=0.01, log=logs.append)
    assert sd.send_once() is False and sd.send_once() is False
    assert len(logs) == 1 and "unreachable" in logs[0] and sd.online is False
    rp.observe(result(1.0, **{"1": {"count": 1, "event": rep(1)}}), 30)
    assert sd.send_once() is True and sd.online is True and logs[-1] == "app connected"
    assert tr.reports[0]["lanes"]["1"]["events"][0]["index"] == 1
    assert rp.snapshot()["lanes"]["1"]["events"] == []                # acked
    s = FakeSession(["1"])
    assert rp.sync(s, CFG) == [("1", 5)]

    # As a thread: poke() sends at once, stop() ends it.
    tr2 = FlakyTransport()
    sd2 = Sender(rp, tr2, interval=60, log=logs.append)
    sd2.start()
    sd2.poke()
    assert tr2.got.wait(2) and len(tr2.reports) == 1
    sd2.stop()
    sd2.join(2)
    assert not sd2.is_alive()


class ScriptedSource:
    is_live = False
    fps = 10.0

    def __init__(self, n):
        self.frames = [np.zeros((4, 4, 3), np.uint8)] * n

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def release(self):
        pass


class ScriptedSession(FakeSession):
    def __init__(self, results):
        super().__init__(["1"])
        self.results = list(results)
        self.times = []

    def process(self, frame, t):
        self.times.append(t)
        return self.results.pop(0)


class PokeCounter:
    pokes = 0

    def poke(self):
        self.pokes += 1


def test_run_loop_syncs_before_each_frame_and_pokes_on_events():
    rp = Reporter("pushup", "clip.mp4", ["1"])
    rp.on_reply({"lanes": {"1": {"attempt_id": 2}}, "acked": {}})
    ses = ScriptedSession([result(0.0, **{"1": {}}),
                           result(0.1, **{"1": {"count": 1, "event": rep(1)}}),
                           result(0.2, **{"1": {"count": 1}})])
    c0 = ses.counters["1"]
    logs = []
    sender = PokeCounter()
    assert run(ses, ScriptedSource(3), rp, sender, CFG, log=logs.append) == 3
    assert ses.times == [0.0, 0.1, 0.2]                               # the file's own clock
    assert ses.counters["1"] is not c0                                # reset applied before frame 1
    assert sender.pokes == 1
    assert logs[0] == "lane 1: counting for attempt 2" and "rep 1: VALID" in logs[1]
    assert rp.snapshot()["lanes"]["1"]["attempt_id"] == 2
