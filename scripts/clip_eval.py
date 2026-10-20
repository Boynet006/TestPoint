"""Calibrate the rep counter on recorded clips.

Pose is the slow part, so it runs once per clip and the keypoints are cached;
counting is then replayed from the cache in a fraction of a second, which is
what makes threshold tuning practical.

    python scripts/clip_eval.py extract data/clips/pushup/*.mp4 --device cuda
    python scripts/clip_eval.py score data/clips/pushup/*.mp4 --exercise pushup --truth truth.json
    python scripts/clip_eval.py score clip.mp4 --exercise pushup --set down_elbow_deg=100 --plot out.png

``extract`` writes ``<clip>.pose.npz`` next to each clip: per frame the time
and the 17 keypoints + confidences of the candidate (largest confident person),
NaN when nobody was found. ``--counter-ocr X0,Y0,X1,Y1`` also reads an
on-screen rep counter in that region (0..1 coordinates) every half second, as
ground truth, into ``<clip>.counter.json``.

``score`` replays the cache through the exercise analyzer and the rep counter
and prints valid / rejected counts, the faults, and the error against the truth
(``--truth`` JSON ``{"clip name": count}``, or the OCR counter file). ``--plot``
draws the metric curve with the thresholds and every rep (green valid, red
rejected) for a human to check against the video.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import itertools

from testpoint.vision.counter import RepCounter
from testpoint.vision.exercises import CrunchAnalyzer, CrunchRules, PushupAnalyzer, PushupRules
from testpoint.vision.keypoints import Person
from testpoint.vision.session import select_candidate


def cache_path(clip: Path) -> Path:
    return clip.with_suffix(".pose.npz")


def counter_path(clip: Path) -> Path:
    return clip.with_suffix(".counter.json")


# ----------------------------------------------------------------------- extract

def extract(clips: list[Path], device: str, mode: str, counter_box: tuple[float, ...] | None) -> None:
    from testpoint.vision.pose import make_backend

    backend = make_backend("rtmlib", device=device, mode=mode)
    ocr = None
    if counter_box:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ts, xy, conf, readings = [], [], [], []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            p = select_candidate(backend.detect(frame))
            ts.append(t)
            if p is None:
                xy.append(np.full((17, 2), np.nan, np.float32))
                conf.append(np.zeros(17, np.float32))
            else:
                xy.append(p.xy)
                conf.append(p.conf)
            if ocr is not None and i % max(1, int(fps / 2)) == 0:
                h, w = frame.shape[:2]
                x0, y0, x1, y1 = counter_box
                crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
                res, _ = ocr(crop)
                nums = [int(m) for _b, txt, _c in (res or []) for m in re.findall(r"\d+", txt)]
                if nums:
                    readings.append((round(t, 2), max(nums)))
            i += 1
        cap.release()
        np.savez_compressed(cache_path(clip), t=np.array(ts, np.float32), xy=np.array(xy), conf=np.array(conf),
                            fps=np.float32(fps))
        msg = f"{clip.name}: {i} frames, person in {sum(c.max() > 0 for c in conf)}"
        if ocr is not None:
            counter_path(clip).write_text(json.dumps(readings))
            msg += f", counter readings {len(readings)} (last {readings[-1] if readings else None})"
        print(msg, flush=True)


# ------------------------------------------------------------------------- score

RULES = {"pushup": (PushupAnalyzer, PushupRules), "crunch": (CrunchAnalyzer, CrunchRules)}


def make_rules(exercise: str, overrides: list[str]):
    _, rules_cls = RULES[exercise]
    fields = {f.name: f.type for f in dataclasses.fields(rules_cls)}
    kw = {}
    for item in overrides:
        k, v = item.split("=", 1)
        if k not in fields:
            raise SystemExit(f"unknown rule {k!r}; known: {sorted(fields)}")
        kw[k] = int(v) if fields[k] in (int, "int") else float(v)
    return rules_cls(**kw)


def replay(clip: Path, exercise: str, rules) -> dict:
    data = np.load(cache_path(clip))
    analyzer = RULES[exercise][0](rules)
    counter = RepCounter(analyzer.counter_config())
    metric, faults_seen = [], Counter()
    for t, xy, conf in zip(data["t"], data["xy"], data["conf"]):
        if conf.max() <= 0:
            counter.update(None, float(t))
            metric.append(np.nan)
            continue
        m = analyzer.analyze(Person(xy, conf))
        counter.update(m.metric, float(t), m.faults)
        if m.metric is not None and counter.floor is not None:
            metric.append(m.metric - counter.floor)       # what the thresholds are compared with
        else:
            metric.append(np.nan if m.metric is None else m.metric)
        faults_seen.update(m.faults)
    return {"t": data["t"], "metric": np.array(metric, np.float32), "counter": counter,
            "frame_faults": faults_seen, "config": analyzer.counter_config()}


def truth_for(clip: Path, truth: dict) -> int | None:
    if clip.name in truth:
        return int(truth[clip.name])
    cp = counter_path(clip)
    if cp.is_file():
        readings = json.loads(cp.read_text())
        # the counter only goes up: the largest value read twice in a row is trusted
        vals = [v for _t, v in readings]
        stable = [a for a, b in itertools.pairwise(vals) if a == b]
        return max(stable) if stable else None
    return None


def plot(res: dict, path: Path, title: str, w: int = 1600, h: int = 420) -> None:
    t, m, cfg, counter = res["t"], res["metric"], res["config"], res["counter"]
    img = np.full((h, w, 3), 255, np.uint8)
    if len(t) == 0:
        cv2.imwrite(str(path), img)
        return
    t_max = float(t[-1]) or 1.0
    finite = m[np.isfinite(m)]
    lo = min(float(finite.min()) if len(finite) else 0.0, cfg.enter_work, cfg.enter_rest) - 10
    hi = max(float(finite.max()) if len(finite) else 180.0, cfg.enter_work, cfg.enter_rest) + 10
    pad = 50

    def px(tt, v):
        return int(pad + (tt / t_max) * (w - 2 * pad)), int(h - pad - (v - lo) / (hi - lo) * (h - 2 * pad))

    for v, col in ((cfg.enter_work, (0, 140, 255)), (cfg.enter_rest, (200, 120, 0))):
        y = px(0, v)[1]
        cv2.line(img, (pad, y), (w - pad, y), col, 1)
        cv2.putText(img, f"{v:g}", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    for ev in counter.events:
        x0, x1 = px(ev.t_start, lo)[0], px(ev.t_end, lo)[0]
        cv2.rectangle(img, (x0, pad), (max(x1, x0 + 1), h - pad), (200, 245, 200) if ev.valid else (210, 210, 255), -1)
    prev = None
    for tt, v in zip(t, m):
        if not np.isfinite(v):
            prev = None
            continue
        p = px(float(tt), float(v))
        if prev is not None:
            cv2.line(img, prev, p, (40, 40, 40), 1, cv2.LINE_AA)
        prev = p
    for s in range(0, int(t_max) + 1, 5):
        x = px(s, lo)[0]
        cv2.line(img, (x, h - pad), (x, h - pad + 5), (0, 0, 0), 1)
        cv2.putText(img, f"{s}s", (x - 8, h - pad + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, title, (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def score(clips: list[Path], exercise: str, overrides: list[str], truth_file: str | None,
          plot_dir: str | None) -> int:
    rules = make_rules(exercise, overrides)
    truth = json.loads(Path(truth_file).read_text()) if truth_file else {}
    total_err, n_truth = 0, 0
    print(f"rules: {rules}")
    for clip in clips:
        if not cache_path(clip).is_file():
            print(f"{clip.name}: no pose cache, run extract first")
            continue
        res = replay(clip, exercise, rules)
        c = res["counter"]
        tr = truth_for(clip, truth)
        rej = Counter(f for e in c.events if not e.valid for f in e.faults)
        line = f"{clip.name:22s} valid {c.count:3d}  rejected {c.rejected:3d}  {dict(rej) or ''}"
        if tr is not None:
            err = c.count - tr
            total_err += abs(err)
            n_truth += 1
            line += f"   truth {tr:3d}  error {err:+d}"
        print(line)
        if plot_dir:
            Path(plot_dir).mkdir(parents=True, exist_ok=True)
            plot(res, Path(plot_dir) / f"{clip.stem}.png",
                 f"{clip.name}  valid {c.count}  rejected {c.rejected}" + (f"  truth {tr}" if tr is not None else ""))
    if n_truth:
        print(f"total absolute error {total_err} over {n_truth} clips with truth")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("clips", nargs="+", type=Path)
    e.add_argument("--device", default="cpu")
    e.add_argument("--mode", default="balanced")
    e.add_argument("--counter-ocr", metavar="X0,Y0,X1,Y1")
    s = sub.add_parser("score")
    s.add_argument("clips", nargs="+", type=Path)
    s.add_argument("--exercise", default="pushup", choices=sorted(RULES))
    s.add_argument("--set", action="append", default=[], metavar="RULE=VALUE")
    s.add_argument("--truth", help='JSON {"clip.mp4": count}')
    s.add_argument("--plot", metavar="DIR", help="write one curve image per clip here")
    args = p.parse_args(argv)
    if args.cmd == "extract":
        box = tuple(float(v) for v in args.counter_ocr.split(",")) if args.counter_ocr else None
        extract(args.clips, args.device, args.mode, box)
        return 0
    return score(args.clips, args.exercise, args.set, args.truth, args.plot)


if __name__ == "__main__":
    sys.exit(main())
