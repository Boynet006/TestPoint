"""Demo CLI: count reps on a webcam or a video file with an on-screen overlay.

    testpoint-demo --exercise pushup --source 0 --show
    testpoint-demo --exercise crunch --source clip.mp4 --out annotated.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from .vision.exercises import ANALYZERS, make_analyzer
from .vision.keypoints import SKELETON, Person
from .vision.pose import make_backend
from .vision.session import FrameResult, StationSession

GREEN = (0, 200, 0)
RED = (0, 0, 230)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)


def draw_person(frame: np.ndarray, person: Person, min_conf: float = 0.3) -> None:
    for a, b in SKELETON:
        if person.conf[a] >= min_conf and person.conf[b] >= min_conf:
            pa = tuple(int(v) for v in person.xy[a])
            pb = tuple(int(v) for v in person.xy[b])
            cv2.line(frame, pa, pb, GREEN, 2)
    for i in range(len(person.xy)):
        if person.conf[i] >= min_conf:
            cv2.circle(frame, tuple(int(v) for v in person.xy[i]), 4, YELLOW, -1)


def draw_hud(frame: np.ndarray, res: FrameResult, exercise: str, fps: float) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"{exercise.upper()}  count {res.count}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 3)
    cv2.putText(frame, f"rejected {res.rejected}  phase {res.phase.value}  {fps:.0f} fps",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    y = 100
    for k, v in res.metrics.debug.items():
        cv2.putText(frame, f"{k}: {v:.0f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        y += 28
    for f in res.metrics.faults:
        cv2.putText(frame, f"FAULT: {f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
        y += 30
    if res.person is None:
        cv2.putText(frame, "no person", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)


def open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source {source!r}")
    return cap


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TestPoint rep-counting demo")
    p.add_argument("--exercise", choices=sorted(ANALYZERS), required=True)
    p.add_argument("--source", default="0", help="camera index (0, 1, ...) or video file path")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    p.add_argument("--show", action="store_true", help="open a preview window (press q to quit)")
    p.add_argument("--out", help="write an annotated video to this path")
    args = p.parse_args(argv)

    backend = make_backend("rtmlib", device=args.device, mode=args.mode)
    session = StationSession(backend, make_analyzer(args.exercise))
    cap = open_source(args.source)

    writer = None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_file = not args.source.isdigit()
    frame_idx = 0
    t0 = time.perf_counter()
    last = t0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Video files use their own clock; live cameras use wall time.
        t = frame_idx / src_fps if is_file else time.perf_counter() - t0
        res = session.process(frame, t)
        if res.event is not None:
            verdict = "VALID" if res.event.valid else f"REJECTED {res.event.faults}"
            print(f"[{t:7.2f}s] rep {res.event.index}: {verdict} -> count {res.count}")

        now = time.perf_counter()
        fps = 1.0 / max(now - last, 1e-6)
        last = now
        if args.show or args.out:
            if res.person is not None:
                draw_person(frame, res.person)
            draw_hud(frame, res, args.exercise, fps)
        if args.out:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(frame)
        if args.show:
            cv2.imshow("TestPoint", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    summary = {
        "exercise": args.exercise,
        "frames": frame_idx,
        "count": session.counter.count,
        "rejected": session.counter.rejected,
        "events": [e.__dict__ for e in session.counter.events],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
