"""Station engine: count reps on one camera and report to the station app.

    testpoint-engine --source 0 --exercise pushup --device cuda --app http://127.0.0.1:8000
    testpoint-engine --source rtsp://... --exercise crunch --lanes 3 --app http://station:8080

The frame loop is the demo CLI's (``MultiLaneSession`` per lane) without the
preview window. A background thread POSTs a report to ``/api/engine/report``
every ``interval`` seconds and at once after a rep; the reply says which
attempt each lane is counting for, and a change replaces that lane's counter
(docs/ENGINE-LINK.md). Reports never block the frame loop: the sender takes a
snapshot, and resets are applied by the frame thread between two frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

# Must be in the environment before OpenCV loads its FFmpeg plugin (see vision/sources.py).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

from .evidence import EvidenceWriter
from .preview import FramePublisher
from .vision.counter import RepCounter, RepCounterConfig
from .vision.exercises import ANALYZERS, make_analyzer
from .vision.lanes import Lane, equal_strips, load_lanes
from .vision.pose import make_backend
from .vision.session import MultiLaneResult, MultiLaneSession
from .vision.sources import SourceError, iter_frames, open_source

REPORT_PATH = "/api/engine/report"
MAX_PENDING = 500
"""Unacknowledged events kept per lane while the app is unreachable (oldest dropped)."""


@dataclass
class LaneState:
    attempt_id: int | None = None
    """The attempt the app told us to count for (None: nobody on the mat)."""
    count: int = 0
    rejected: int = 0
    phase: str = "waiting"
    person: bool = False
    pending_faults: list[str] = field(default_factory=list)
    work_seconds: float = 0.0
    depth: float | None = None
    """How far into the rep the person is: 0 at the rest position, 1 at the depth that counts
    (the counter's thresholds); None when not measurable. For the station screen's gauge."""
    events: list[dict] = field(default_factory=list)
    """Completed reps not yet acknowledged by the app, oldest first."""


def rep_depth(metric: float | None, cfg: RepCounterConfig | None) -> float | None:
    """The movement metric as 0 (rest threshold) .. 1 (work threshold), up to 1.25 past it."""
    if metric is None or cfg is None:
        return None
    span = (cfg.enter_rest - cfg.enter_work) if cfg.work_is_below else (cfg.enter_work - cfg.enter_rest)
    if span <= 0:
        return None
    d = (cfg.enter_rest - metric) / span if cfg.work_is_below else (metric - cfg.enter_rest) / span
    return round(max(0.0, min(1.25, d)), 3)


class Reporter:
    """Per-lane bookkeeping between the frame thread and the sender thread.

    ``observe`` (frame thread) records the latest lane values and new rep events;
    ``snapshot`` (sender) builds the report; ``on_reply`` (sender) stores the acks
    and the attempts the app wants; ``sync`` (frame thread) applies the attempt
    changes to the session's counters so the frame loop is the only writer there.
    """

    def __init__(self, station: str, source: str, lane_names: list[str],
                 max_pending: int = MAX_PENDING,
                 clip_for: Callable[[str, dict], str | None] | None = None,
                 counter: RepCounterConfig | None = None) -> None:
        self.station = station
        self.source = source
        self.lanes = {name: LaneState() for name in lane_names}
        self.max_pending = max_pending
        self.clip_for = clip_for
        """Cuts the evidence clip for ``(lane, event)`` and returns its path, if recording."""
        self.counter = counter
        """The exercise's counter thresholds, for the live depth (rep_depth)."""
        self.t = 0.0
        self.fps = 0.0
        self._wanted: dict[str, int | None] | None = None
        self._lock = threading.Lock()

    def observe(self, res: MultiLaneResult, fps: float) -> list[tuple[str, dict]]:
        """Record one frame's results. Returns the new events as ``(lane, event)``."""
        new = []
        with self._lock:
            self.t, self.fps = res.t, fps
            for name, r in res.lanes.items():
                lane = self.lanes[name]
                lane.count, lane.rejected = r.count, r.rejected
                lane.phase = r.phase.value
                lane.person = r.person is not None
                lane.pending_faults = list(r.pending_faults)
                lane.work_seconds = round(r.work_seconds, 2)
                lane.depth = rep_depth(r.metrics.metric if r.person is not None else None, self.counter)
                if r.event is not None:
                    ev = {"index": r.event.index, "valid": r.event.valid,
                          "faults": list(r.event.faults),
                          "t_start": round(r.event.t_start, 3), "t_end": round(r.event.t_end, 3)}
                    if self.clip_for is not None:
                        ev["clip"] = self.clip_for(name, ev)
                    lane.events.append(ev)
                    del lane.events[:-self.max_pending]
                    new.append((name, ev))
        return new

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "station": self.station, "t": round(self.t, 3), "fps": round(self.fps, 1),
                "source": self.source,
                "lanes": {
                    name: {"attempt_id": lane.attempt_id, "count": lane.count,
                           "rejected": lane.rejected, "phase": lane.phase, "person": lane.person,
                           "pending_faults": lane.pending_faults,
                           "work_seconds": lane.work_seconds, "depth": lane.depth, "events": list(lane.events)}
                    for name, lane in self.lanes.items()
                },
            }

    def on_reply(self, reply: dict) -> None:
        with self._lock:
            for name, index in (reply.get("acked") or {}).items():
                lane = self.lanes.get(name)
                if lane is not None:
                    lane.events = [e for e in lane.events if e["index"] > int(index)]
            wanted = reply.get("lanes") or {}
            self._wanted = {name: (wanted.get(name) or {}).get("attempt_id")
                            for name in self.lanes if name in wanted}

    def sync(self, session: MultiLaneSession, cfg: RepCounterConfig) -> list[tuple[str, int | None]]:
        """Apply the app's wishes: a lane whose attempt changed gets a fresh counter.
        Returns the lanes that were reset with their new attempt id."""
        with self._lock:
            wanted, self._wanted = self._wanted, None
            if not wanted:
                return []
            reset = []
            for name, attempt_id in wanted.items():
                lane = self.lanes[name]
                if attempt_id == lane.attempt_id:
                    continue
                old = session.counters.get(name)
                session.counters[name] = old.restart() if old is not None and old.cfg == cfg else RepCounter(cfg)
                lane.attempt_id = attempt_id
                lane.count = lane.rejected = 0
                lane.phase, lane.pending_faults, lane.work_seconds = "waiting", [], 0.0
                lane.events = []          # they belonged to the previous attempt
                reset.append((name, attempt_id))
            return reset


class HttpTransport:
    """POST a report to the app with the standard library (the engine image has no extra deps)."""

    def __init__(self, app_url: str, timeout_s: float = 2.0) -> None:
        self.url = app_url.rstrip("/") + REPORT_PATH
        self.timeout_s = timeout_s

    def post(self, report: dict) -> dict:
        req = urllib.request.Request(
            self.url, data=json.dumps(report).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode() or "{}")


class Sender(threading.Thread):
    """Reports every ``interval`` seconds, or at once after ``poke()``; one request at a time.
    A failed request is logged once, and once more when the app answers again."""

    def __init__(self, reporter: Reporter, transport, interval: float = 0.25,
                 log: Callable[[str], None] = print) -> None:
        super().__init__(name="engine-sender", daemon=True)
        self.reporter = reporter
        self.transport = transport
        self.interval = interval
        self.log = log
        self.online: bool | None = None
        self.sent = 0
        self._wake = threading.Event()
        self._stopping = threading.Event()

    def poke(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()

    def send_once(self) -> bool:
        try:
            reply = self.transport.post(self.reporter.snapshot())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
            if self.online is not False:
                self.log(f"app unreachable at {getattr(self.transport, 'url', '?')}: "
                         f"{getattr(e, 'reason', e)} (counting continues, retrying)")
            self.online = False
            return False
        if self.online is not True:
            self.log("app connected")
        self.online = True
        self.sent += 1
        self.reporter.on_reply(reply)
        return True

    def run(self) -> None:
        while not self._stopping.is_set():
            self._wake.wait(self.interval)
            self._wake.clear()
            if self._stopping.is_set():
                break
            self.send_once()


def build_lanes(args: argparse.Namespace) -> list[Lane]:
    if args.lanes_file:
        return load_lanes(args.lanes_file)
    return equal_strips(args.lanes)


def run(session: MultiLaneSession, source, reporter: Reporter, sender: Sender,
        cfg: RepCounterConfig, log: Callable[[str], None] = print,
        should_stop: Callable[[], bool] = lambda: False,
        evidence: EvidenceWriter | None = None, preview: FramePublisher | None = None,
        realtime: bool = False) -> int:
    """The frame loop. Returns the number of frames processed. ``realtime``: a video file is
    played at its own speed (demos: reps arrive as they happen, the lane timer stays honest)."""
    frames = 0
    fps = 0.0
    last = time.monotonic()
    for frame, t in iter_frames(source, realtime=realtime):
        for name, attempt_id in reporter.sync(session, cfg):
            log(f"lane {name}: " + (f"counting for attempt {attempt_id}" if attempt_id else "free"))
            if evidence is not None:
                evidence.stop(name)
                if attempt_id is not None:
                    evidence.start(name, attempt_id)
        if evidence is not None:
            evidence.push(frame, t)       # before the counter sees it: the clip ends on this frame
        res = session.process(frame, t)
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        new = reporter.observe(res, fps)
        for name, ev in new:
            verdict = "VALID" if ev["valid"] else f"REJECTED {ev['faults']}"
            log(f"[{t:7.2f}s] lane {name} rep {ev['index']}: {verdict} -> count {res.lanes[name].count}")
        if new:
            sender.poke()
        if preview is not None:
            preview.offer(frame, (res, fps))
        frames += 1
        if should_stop():
            break
    return frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TestPoint station engine: count reps and report to the app")
    p.add_argument("--exercise", choices=sorted(ANALYZERS), required=True)
    p.add_argument("--source", default="0",
                   help="camera index (0, 1, ...), video file path or rtsp:// URL")
    p.add_argument("--app", default="http://127.0.0.1:8000", help="station app URL")
    p.add_argument("--station", default=None,
                   help="station name the lanes belong to (default: the exercise)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    p.add_argument("--lanes", type=int, default=1, metavar="N",
                   help="split the frame into N equal vertical lanes (default 1 = whole frame)")
    p.add_argument("--lanes-file", metavar="LANES.JSON",
                   help="drawn lane rectangles in 0..1 image coordinates; overrides --lanes")
    p.add_argument("--interval", type=float, default=0.25, metavar="S",
                   help="seconds between reports to the app (default 0.25)")
    p.add_argument("--evidence", default="data/evidence", metavar="DIR",
                   help="folder for attempt recordings and per-rep clips (default data/evidence; "
                        "'' = no evidence video)")
    p.add_argument("--connect-timeout", type=float, default=10.0, metavar="S",
                   help="RTSP open/read timeout in seconds (default 10)")
    p.add_argument("--connect-attempts", type=int, default=3, metavar="N",
                   help="RTSP connection attempts before giving up, 0 = for ever (default 3)")
    p.add_argument("--realtime", action="store_true",
                   help="play a video file at its own speed, as a live camera would (demos, training)")
    p.add_argument("--no-preview", action="store_true",
                   help="do not send the live camera picture to the station screen")
    args = p.parse_args(argv)

    if args.lanes < 1:
        p.error("--lanes must be at least 1")
    try:
        lanes = build_lanes(args)
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"bad lanes: {e}") from None
    try:
        source = open_source(args.source, realtime=args.realtime, timeout_s=args.connect_timeout,
                             max_attempts=args.connect_attempts)
    except SourceError as e:
        raise SystemExit(f"error: {e}") from None
    try:
        backend = make_backend("rtmlib", device=args.device, mode=args.mode)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        source.release()
        raise SystemExit(f"error: could not start the pose backend: {e}") from None

    analyzer = make_analyzer(args.exercise)
    session = MultiLaneSession(backend, analyzer, lanes)
    station = args.station or args.exercise
    display = getattr(source, "display_url", args.source)
    evidence = EvidenceWriter(args.evidence, lanes, fps=source.fps) if args.evidence else None
    reporter = Reporter(station, display, [lane.name for lane in lanes],
                        clip_for=evidence.clip if evidence else None, counter=analyzer.counter_config())
    sender = Sender(reporter, HttpTransport(args.app), interval=args.interval)
    print(f"station {station}  lanes {[lane.name for lane in lanes]}  source {display}  app {args.app}"
          + (f"  evidence {evidence.root}" if evidence else ""))
    preview = None
    if not args.no_preview:
        from .cli import draw_hud, draw_lanes, draw_person

        def draw(img, state):
            res, fps = state
            assigned = {id(r.person) for r in res.lanes.values() if r.person is not None}
            for person in res.people:
                draw_person(img, person, (0, 200, 0) if id(person) in assigned else (140, 140, 140))
            draw_lanes(img, lanes, res)
            draw_hud(img, args.exercise, fps, "")

        preview = FramePublisher(args.app, station, draw)
        preview.tap(source)
        preview.start()
    sender.start()
    try:
        run(session, source, reporter, sender, analyzer.counter_config(), evidence=evidence, preview=preview,
            realtime=args.realtime)
    except SourceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        sender.stop()
        if preview is not None:
            preview.stop()
        source.release()
        if evidence is not None:
            evidence.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
