"""Demo CLI: count reps on a webcam, a video file or a live RTSP camera, per lane.

    testpoint-demo --exercise pushup --source 0 --show
    testpoint-demo --exercise crunch --source clip.mp4 --out annotated.mp4
    testpoint-demo --exercise pushup --source rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101 --lanes 3 --show
    testpoint-demo --exercise pushup --source rtsp://... --lanes-file lanes.json --device cuda --show
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Must be in the environment before OpenCV loads its FFmpeg plugin (see vision/sources.py).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

from .vision.exercises import ANALYZERS, make_analyzer
from .vision.keypoints import SKELETON, Person
from .vision.lanes import Lane, equal_strips, load_lanes
from .vision.pose import make_backend
from .vision.session import MultiLaneResult, MultiLaneSession
from .vision.sources import SourceError, open_source

GREEN = (0, 200, 0)
GREY = (140, 140, 140)
RED = (0, 0, 230)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
LANE_COLOURS = [(255, 180, 0), (0, 200, 255), (200, 0, 255), (0, 255, 160), (255, 80, 160)]


def draw_person(frame: np.ndarray, person: Person, colour=GREEN, min_conf: float = 0.3) -> None:
    for a, b in SKELETON:
        if person.conf[a] >= min_conf and person.conf[b] >= min_conf:
            pa = tuple(int(v) for v in person.xy[a])
            pb = tuple(int(v) for v in person.xy[b])
            cv2.line(frame, pa, pb, colour, 2)
    for i in range(len(person.xy)):
        if person.conf[i] >= min_conf:
            cv2.circle(frame, tuple(int(v) for v in person.xy[i]), 4, YELLOW, -1)


def draw_lanes(frame: np.ndarray, lanes: list[Lane], res: MultiLaneResult) -> None:
    """One rectangle per lane with its name, count, phase and live faults."""
    h, w = frame.shape[:2]
    for i, lane in enumerate(lanes):
        colour = LANE_COLOURS[i % len(LANE_COLOURS)]
        x0, y0, x1, y1 = lane.to_pixels(w, h)
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), colour, 2)
        r = res.lanes[lane.name]
        x, y = x0 + 8, y0 + 40
        cv2.putText(frame, f"L{lane.name}  {r.count}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3)
        y += 26
        status = f"rej {r.rejected}  {r.phase.value}"
        if r.work_seconds > 0:
            status += f"  {r.work_seconds:.1f}s"
        cv2.putText(frame, status, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        y += 24
        for k, v in r.metrics.debug.items():
            cv2.putText(frame, f"{k} {v:.2f}" if abs(v) < 2 else f"{k} {v:.0f}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
            y += 22
        for f in sorted(set(r.metrics.faults) | set(r.pending_faults)):
            cv2.putText(frame, f"FAULT {f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED, 2)
            y += 26
        if r.person is None:
            cv2.putText(frame, "no person", (x, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)


def draw_hud(frame: np.ndarray, exercise: str, fps: float, source_note: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"{exercise.upper()}  {fps:.0f} fps  {source_note}", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)


def build_lanes(args: argparse.Namespace) -> list[Lane]:
    if args.lanes_file:
        return load_lanes(args.lanes_file)
    return equal_strips(args.lanes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TestPoint rep-counting demo")
    p.add_argument("--exercise", choices=sorted(ANALYZERS), required=True)
    p.add_argument("--source", default="0",
                   help="camera index (0, 1, ...), video file path or rtsp:// URL")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    p.add_argument("--lanes", type=int, default=1, metavar="N",
                   help="split the frame into N equal vertical lanes (default 1 = whole frame)")
    p.add_argument("--lanes-file", metavar="LANES.JSON",
                   help='drawn lane rectangles: {"lanes": [{"name": "1", "rect": [x0,y0,x1,y1]}]}'
                        " in 0..1 image coordinates; overrides --lanes")
    p.add_argument("--connect-timeout", type=float, default=10.0, metavar="S",
                   help="RTSP open/read timeout in seconds (default 10)")
    p.add_argument("--connect-attempts", type=int, default=3, metavar="N",
                   help="RTSP connection attempts before giving up, 0 = for ever (default 3)")
    p.add_argument("--show", action="store_true", help="open a preview window (press q to quit)")
    p.add_argument("--out", help="write an annotated video to this path")
    args = p.parse_args(argv)

    if args.lanes < 1:
        p.error("--lanes must be at least 1")
    try:
        lanes = build_lanes(args)
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"bad lanes: {e}") from None

    try:
        source = open_source(
            args.source, timeout_s=args.connect_timeout, max_attempts=args.connect_attempts
        )
    except SourceError as e:
        raise SystemExit(f"error: {e}") from None

    try:
        backend = make_backend("rtmlib", device=args.device, mode=args.mode)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        source.release()
        raise SystemExit(f"error: could not start the pose backend: {e}") from None
    session = MultiLaneSession(backend, make_analyzer(args.exercise), lanes)

    writer = None
    src_fps = source.fps or 30.0
    frame_idx = 0
    t0 = time.monotonic()
    last = t0
    lane_names = [lane.name for lane in lanes]
    print(f"lanes: {lane_names}  source: {getattr(source, 'display_url', args.source)}")

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            # Video files use their own clock; live cameras use wall time.
            t = frame_idx / src_fps if not source.is_live else time.monotonic() - t0
            res = session.process(frame, t)
            for name, ev in res.events:
                verdict = "VALID" if ev.valid else f"REJECTED {ev.faults}"
                print(f"[{t:7.2f}s] lane {name} rep {ev.index}: {verdict} "
                      f"-> count {res.lanes[name].count}")

            now = time.monotonic()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            if args.show or args.out:
                assigned = {id(r.person) for r in res.lanes.values() if r.person is not None}
                for person in res.people:
                    draw_person(frame, person, GREEN if id(person) in assigned else GREY)
                draw_lanes(frame, lanes, res)
                note = f"reconnects {source.reconnects}" if hasattr(source, "reconnects") else ""
                draw_hud(frame, args.exercise, fps, note)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
                    )
                writer.write(frame)
            if args.show:
                cv2.imshow("TestPoint", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    except SourceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    summary = {
        "exercise": args.exercise,
        "source": getattr(source, "display_url", args.source),
        "frames": frame_idx,
        "lanes": session.summary(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
