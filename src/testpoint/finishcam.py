"""Finish-line camera: detect runners, track them, time the crossing, read the bib,
report each crossing to the station app.

    testpoint-finishcam --source rtsp://... --line 0,0.85,1,0.85 --device cuda --app http://127.0.0.1:8000
    testpoint-finishcam --source clip.mp4 --crop 0.21,0,1,1 --out annotated.mp4 --no-app

Per frame: person detector (``vision/detect.py``) -> IoU tracker -> finish line
(``vision/line.py``, crossing instant interpolated between frames). The torso
crops of each track are kept while it approaches; at the crossing they are read
by OCR and voted (``vision/bib.py``). The crossing is POSTed to
``/api/engine/crossing`` with the bib (or none), the OCR confidence, how long
ago it happened (so the server stamps the true instant) and the evidence clip
path. The app does the rest with its existing finish logic: a known running
bib is a result, anything else is a review-queue entry with the clip.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

from .applink import ssl_context
from .evidence import EvidenceWriter
from .preview import FramePublisher
from .vision.bib import BibReader, BibResult, BibVote
from .vision.detect import Box
from .vision.lanes import whole_frame
from .vision.line import CrossEvent, FinishLine, parse_line
from .vision.sources import SourceError, VideoPlayer, iter_frames, open_source
from .vision.tracker import IoUTracker, Track

CROSSING_PATH = "/api/engine/crossing"


@dataclass
class Crossing:
    t: float
    bib: str | None
    confidence: float
    readings: list[tuple[str, float]]
    clip: str | None = None
    track_id: int = 0
    wall: float = field(default_factory=time.time)
    """Wall-clock time the crossing was detected (the frame after the line)."""
    face_jpegs: list[bytes] = field(default_factory=list, repr=False)
    """Snapshots of the runner's upper body just before the line, always sent: the app recognises
    the face among the runners still out, and a bib read that disagrees with the face goes to
    the examiner (the pictures are not stored)."""

    def payload(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        return {"bib": self.bib, "confidence": self.confidence,
                "readings": [{"text": s, "conf": round(c, 3)} for s, c in self.readings],
                "latency_s": round(max(0.0, now - self.wall), 3), "t": round(self.t, 3),
                "clip": self.clip, "track": self.track_id,
                **({"faces": [base64.b64encode(j).decode("ascii") for j in self.face_jpegs]} if self.face_jpegs else {})}


class CrossingPoster:
    """Sends crossings to the app from a thread, in order, retrying while unreachable."""

    def __init__(self, app_url: str, timeout_s: float = 3.0, log: Callable[[str], None] = print,
                 on_reply: Callable[[Crossing, dict], None] | None = None) -> None:
        self.url = app_url.rstrip("/") + CROSSING_PATH
        self.on_reply = on_reply
        """Told what the app made of each crossing (who it was), for the picture."""
        self.timeout_s = timeout_s
        self.log = log
        self.queue: list[Crossing] = []
        self.sent: list[dict] = []
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="finishcam-poster", daemon=True)
        self._thread.start()

    def post(self, crossing: Crossing) -> None:
        with self._cv:
            self.queue.append(crossing)
            self._cv.notify()

    def _send(self, c: Crossing) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(c.payload()).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=ssl_context(self.url)) as resp:
            return json.loads(resp.read().decode() or "{}")

    def _run(self) -> None:
        backoff = 1.0
        while True:
            with self._cv:
                while not self.queue and not self._stop:
                    self._cv.wait()
                if self._stop and not self.queue:
                    return
                c = self.queue[0]
            try:
                reply = self._send(c)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                self.log(f"crossing refused by the app ({e.code}): {body[:200]}")
                reply = {"error": e.code, "detail": body[:200]}
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.log(f"app unreachable at {self.url}: {getattr(e, 'reason', e)}; retrying in {backoff:g} s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            with self._cv:
                self.queue.pop(0)
                self.sent.append(reply)
            if self.on_reply is not None:
                try:
                    self.on_reply(c, reply)
                except Exception:          # noqa: BLE001 - the picture's label must never stop the reports
                    pass

    def close(self, timeout_s: float = 10.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout_s)


class FinishCam:
    """The per-frame pipeline, independent of the video source and of the app."""

    def __init__(self, detector, line: FinishLine, reader: BibReader | None,
                 tracker: IoUTracker | None = None, evidence: EvidenceWriter | None = None,
                 evidence_folder: str = "run", on_crossing: Callable[[Crossing], None] | None = None,
                 log: Callable[[str], None] = print, ocr_async: bool = False, faces: bool = True,
                 face_finder: Callable[[np.ndarray], tuple | None] | None = None) -> None:
        """``ocr_async``: read bibs on a worker thread, so a crossing never stalls the frame
        loop (OCR of eight crops takes about a second, and a runner close behind must still
        be tracked). The crossing is then reported when its bib is decided, with its true
        detection time; ``process`` returns only the crossings decided synchronously."""
        self.detector = detector
        self.line = line
        self.reader = reader
        self.tracker = tracker or IoUTracker()
        self.evidence = evidence
        self.evidence_folder = evidence_folder
        self.on_crossing = on_crossing
        self.log = log
        self.crossings: list[Crossing] = []
        self.last_boxes: list[Box] = []
        self.last_tracks: list[Track] = []
        self._ocr = ThreadPoolExecutor(1, thread_name_prefix="bib-ocr") if ocr_async else None
        self._lock = threading.Lock()
        self._pending_clips: list[tuple[float, float, str]] = []
        """``(t_from, t_to, name)``: clips to cut once the frame clock passes ``t_to``."""
        self.faces = faces
        self.face_finder = face_finder
        """Finds a face in an upper-body snapshot: (x, y, w, h, score) or None. With it, only
        snapshots that show a face are kept, the best ones over the whole approach (the face
        models on the GPU); without it, the last upper-body snapshots before the line."""
        self._heads: dict[int, list[tuple[float, np.ndarray]]] = {}
        """Per track, the face snapshots kept: (time or quality, picture)."""
        self._head_t: dict[int, float] = {}
        self.names: dict[int, str] = {}
        """Per track, who the app said crossed (bib and name), or "to review"; shown on the picture."""

    CLIP_BEFORE_S = 1.5
    CLIP_AFTER_S = 1.0

    def process(self, frame: np.ndarray, t: float) -> list[Crossing]:
        h, w = frame.shape[:2]
        if self.evidence is not None:
            self.evidence.push(frame, t)
            # A crossing clip is cut once the frames after the line are in the buffer.
            due = [c for c in self._pending_clips if t >= c[1]]
            for t0, t1, name in due:
                self.evidence.clip_at(self.evidence_folder, name, t0, t1)
            self._pending_clips = [c for c in self._pending_clips if t < c[1]]
        boxes = self.detector.detect(frame)
        tracks = self.tracker.update(boxes, t)
        self.last_boxes, self.last_tracks = boxes, tracks
        if self.reader is not None:
            for tr in tracks:
                self.reader.observe(tr.id, frame, tr.box)
        if self.faces:
            for tr in tracks:
                self._keep_head(tr.id, frame, tr.box, t)
        out = []
        for ev in self.line.check(tracks, w, h):
            c = self._crossing(ev)
            if c is not None:
                out.append(c)
        live = {tr.id for tr in self.tracker.tracks}
        self.line.forget(live)
        for tid in [t for t in self._heads if t not in live]:
            del self._heads[tid]
        for tid in [t for t in self._head_t if t not in live]:
            del self._head_t[tid]
        if self.reader is not None:
            self.reader.forget(live)
        return out

    UPPER_SHARE = 0.7
    """The part of a person box kept for the face: the upper body, so a raised arm (which makes
    the box taller) does not push the face out of the snapshot; the face detector finds it."""
    FACE_SNAPSHOTS = 5
    FACE_SPACING_S = 0.3
    """Snapshots kept per runner, at least this far apart: the last moments before the line, from
    different angles (one where they look down or an arm covers the face is not the only one)."""

    FACE_LOOK_S = 0.12
    """With a face finder: how often each runner's face is looked for while they approach."""
    MIN_FACE_PX = 40

    def _keep_head(self, track_id: int, frame: np.ndarray, box, t: float = 0.0) -> None:
        shots = self._heads.setdefault(track_id, [])
        last = self._head_t.get(track_id)
        if last is not None and t - last < (self.FACE_LOOK_S if self.face_finder else self.FACE_SPACING_S):
            return
        self._head_t[track_id] = t
        h, w = frame.shape[:2]
        x0, x1 = max(0, int(box.x0 - 0.1 * box.width)), min(w, int(box.x1 + 0.1 * box.width))
        y0, y1 = max(0, int(box.y0 - 0.05 * box.height)), min(h, int(box.y0 + self.UPPER_SHARE * box.height))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return
        crop = frame[y0:y1, x0:x1]
        if self.face_finder is not None:
            found = self.face_finder(crop)
            if not found:
                return
            fx, fy, fw, fh, score = found
            if min(fw, fh) < self.MIN_FACE_PX:
                return
            # the head with room around it, at most 320 wide: plenty for recognition, small to send
            m = 0.7 * max(fw, fh)
            hx0, hy0 = max(0, int(fx - m)), max(0, int(fy - m))
            hx1, hy1 = min(crop.shape[1], int(fx + fw + m)), min(crop.shape[0], int(fy + fh + m))
            head = crop[hy0:hy1, hx0:hx1]
            if head.shape[1] > 320:
                k = 320 / head.shape[1]
                head = cv2.resize(head, (320, max(1, int(head.shape[0] * k))), interpolation=cv2.INTER_AREA)
            shots.append((min(fw, fh) * float(score), head.copy()))
            shots.sort(key=lambda s: -s[0])                   # the clearest faces first
            del shots[self.FACE_SNAPSHOTS:]
            return
        if crop.shape[1] > 480:
            k = 480 / crop.shape[1]
            crop = cv2.resize(crop, (480, int(crop.shape[0] * k)))
        shots.append((t, crop.copy()))
        del shots[:-self.FACE_SNAPSHOTS]

    def _crossing(self, ev: CrossEvent) -> Crossing | None:
        wall = time.time()
        crops = self.reader.take(ev.track_id) if self.reader is not None else []
        heads = [crop for _t, crop in self._heads.pop(ev.track_id, [])]
        clip = None
        if self.evidence is not None:
            name = f"{int(ev.t * 1000):09d}-t{ev.track_id}"
            self._pending_clips.append((ev.t - self.CLIP_BEFORE_S, ev.t + self.CLIP_AFTER_S, name))
            clip = f"{self.evidence_folder}/{name}.mp4"      # written about a second from now
        if self._ocr is not None:
            self._ocr.submit(self._decide, ev, crops, clip, wall, heads)
            return None
        return self._decide(ev, crops, clip, wall, heads)

    def _decide(self, ev: CrossEvent, crops: list[np.ndarray], clip: str | None, wall: float,
                heads: list[np.ndarray] | None = None) -> Crossing:
        result = self.reader.read_crops(crops) if self.reader is not None else BibResult(None, 0.0, [])
        faces = []
        if heads:
            for head in heads:
                ok, buf = cv2.imencode(".jpg", head, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    faces.append(buf.tobytes())
        c = Crossing(t=ev.t, bib=result.bib, confidence=result.confidence, readings=result.readings,
                     clip=clip, track_id=ev.track_id, wall=wall, face_jpegs=faces)
        with self._lock:
            self.crossings.append(c)
        self.log(f"[{ev.t:8.3f}s] crossing track {ev.track_id}: bib {c.bib or '?'} "
                 f"(confidence {c.confidence}, {len(c.readings)} readings)")
        if self.on_crossing is not None:
            self.on_crossing(c)
        return c

    def close(self) -> None:
        """Cut the clips still waiting (the source ended) and wait for bibs still being read."""
        if self.evidence is not None:
            for t0, t1, name in self._pending_clips:
                self.evidence.clip_at(self.evidence_folder, name, t0, t1)
            self._pending_clips = []
        if self._ocr is not None:
            self._ocr.shutdown(wait=True)


class Analyser(threading.Thread):
    """A test video played as a live camera is: the picture moves at the video's own speed
    (every frame shown) while the analysis takes the newest frame whenever it is free, instead
    of the video crawling at the analysis' rate. The first frame waits for the analysis (the
    models warm up), so a crossing at the start of a clip is not lost."""

    def __init__(self, process: Callable[[np.ndarray, float, bool], None]) -> None:
        super().__init__(name="finish-analysis", daemon=True)
        self.process = process
        self.fps = 0.0
        self.error: BaseException | None = None
        self.first = threading.Event()
        self._latest: tuple[np.ndarray, float, bool] | None = None
        self._jumped = False
        self._cv = threading.Condition()
        self._halt = False

    def offer(self, frame: np.ndarray, t: float, jumped: bool = False) -> None:
        with self._cv:
            self._jumped = self._jumped or jumped    # a seek between two analysed frames still counts
            self._latest = (frame, t, self._jumped)
            self._cv.notify()

    def stop(self) -> None:
        with self._cv:
            self._halt = True
            self._cv.notify()

    def run(self) -> None:
        last = time.monotonic()
        while True:
            with self._cv:
                while self._latest is None and not self._halt:
                    self._cv.wait()
                if self._halt:
                    return
                (frame, t, jumped), self._latest, self._jumped = self._latest, None, False
            try:
                self.process(frame, t, jumped)
            except BaseException as e:   # noqa: BLE001 - the frame loop reports it
                self.error = e
                self.first.set()
                return
            now = time.monotonic()
            inst = 1.0 / max(now - last, 1e-6)
            self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
            last = now
            self.first.set()


def draw(frame: np.ndarray, cam: FinishCam, fps: float) -> None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = cam.line.to_pixels(w, h)
    cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 230), 2)
    for tr in cam.last_tracks:
        b = tr.box
        crossed = tr.id in cam.line._crossed
        colour = (0, 200, 0) if crossed else (255, 180, 0)      # green: counted; blue: followed, not crossed yet
        cv2.rectangle(frame, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), colour, 2)
        who = cam.names.get(tr.id)
        label = f"#{tr.id} " + (f"crossed - {who}" if crossed and who else "crossed" if crossed else "not crossed yet")
        # readable on the screen: the picture is sent at most 960 wide, so the text grows with the camera
        scale = max(0.5, min(1.6, w / 1700))
        thick = max(1, int(scale * 1.6))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = int(b.x0), max(th + 8, int(b.y0) - 8)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 10, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, label, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)
    y = 20
    for c in cam.crossings[-4:]:
        cv2.putText(frame, f"{c.t:7.2f}s  bib {c.bib or '?'}  {c.confidence:.2f}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 20
    cv2.putText(frame, f"{fps:.0f} fps", (w - 70, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def parse_crop(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 4 or not all(0 <= v <= 1 for v in parts) or parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise ValueError("crop needs x0,y0,x1,y1 in 0..1 with x0<x1 and y0<y1")
    return parts[0], parts[1], parts[2], parts[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TestPoint finish-line camera")
    p.add_argument("--source", default="0", help="camera index, video file or rtsp:// URL")
    p.add_argument("--line", default="0,0.85,1,0.85", help="finish line x0,y0,x1,y1 in 0..1 (default 0,0.85,1,0.85)")
    p.add_argument("--direction", default="down", choices=["down", "up", "left", "right", "any"],
                   help="which way a crossing counts (default down: runners come towards the camera)")
    p.add_argument("--crop", default=None, metavar="X0,Y0,X1,Y1",
                   help="use only this part of the frame, in 0..1 (a feed with side panels)")
    p.add_argument("--app", default="http://127.0.0.1:8000")
    p.add_argument("--no-app", action="store_true", help="do not report crossings (clip analysis)")
    p.add_argument("--no-ocr", action="store_true", help="skip bib reading")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    p.add_argument("--min-share", type=float, default=0.5, help="OCR vote share needed to trust a bib (default 0.5)")
    p.add_argument("--evidence", default="data/evidence", help="evidence folder ('' = none)")
    p.add_argument("--out", help="write an annotated video here")
    p.add_argument("--show", action="store_true", help="preview window (q quits)")
    p.add_argument("--station", default="finish", help="screen this camera shows on (default finish)")
    p.add_argument("--no-preview", action="store_true", help="do not send the live picture to the app")
    p.add_argument("--realtime", action="store_true",
                   help="play a video file at its own speed, as a live camera would (demos, training)")
    p.add_argument("--loop", action="store_true",
                   help="with --realtime: play the video on repeat (test the counting on a short clip)")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--connect-attempts", type=int, default=3)
    args = p.parse_args(argv)

    try:
        line = FinishLine(parse_line(args.line), args.direction)
        crop = parse_crop(args.crop)
    except ValueError as e:
        p.error(str(e))
    try:
        source = open_source(args.source, realtime=args.realtime, loop=args.loop, every_frame=True, timeout_s=args.connect_timeout,
                             max_attempts=args.connect_attempts)
    except SourceError as e:
        raise SystemExit(f"error: {e}") from None
    try:
        from .vision.detect import YOLOXPersonDetector
        detector = YOLOXPersonDetector(device=args.device, mode=args.mode)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        source.release()
        raise SystemExit(f"error: could not start the person detector: {e}") from None
    reader = None
    if not args.no_ocr:
        try:
            from .vision.bib import RapidOCRReader
            reader = BibReader(RapidOCRReader().read, BibVote(min_share=args.min_share))
        except ImportError:
            print("bib OCR disabled: pip install rapidocr-onnxruntime", file=sys.stderr)

    evidence = EvidenceWriter(args.evidence, [whole_frame("1")], fps=source.fps, keep_s=4.0, pad_s=0.0) if args.evidence else None
    names: dict[int, str] = {}

    def heard(c: Crossing, reply: dict) -> None:
        r = reply.get("runner")
        names[c.track_id] = f"{r.get('bib') or ''} {r.get('name') or ''}".strip() if r else "to review" if reply.get("needs_review") else ""
    poster = None if args.no_app else CrossingPoster(args.app, on_reply=heard)
    finder = None
    try:
        from .api.faces import get_engine
        eng = get_engine()

        def finder(img):
            rows = eng.detect(img)
            if len(rows) == 0:
                return None
            r = max(rows, key=lambda r: r[2] * r[3])
            return float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[14])
        print(f"faces looked for on {eng.device}")
    except Exception as e:          # noqa: BLE001 - no face models: upper-body snapshots as before
        print(f"face snapshots without a face finder ({e})")
    cam = FinishCam(detector, line, reader, evidence=evidence,
                    on_crossing=(poster.post if poster else None), ocr_async=True, face_finder=finder)
    cam.names = names
    print(f"finish cam: source {getattr(source, 'display_url', args.source)}  line {args.line} {args.direction}"
          f"  crop {args.crop or 'full'}  app {'off' if args.no_app else args.app}  ocr {'off' if reader is None else 'on'}")

    preview = None
    if poster is not None and not args.no_preview:
        preview = FramePublisher(args.app, args.station, lambda img, fps_now: draw(img, cam, fps_now), max_width=1920, quality=85)
        preview.tap(source, None if crop is None else lambda f: f[int(crop[1] * f.shape[0]):int(crop[3] * f.shape[0]),
                                                                    int(crop[0] * f.shape[1]):int(crop[2] * f.shape[1])])
        if isinstance(source, VideoPlayer):
            preview.attach_player(source)    # the finish screen's player controls this test video
        preview.start()
    writer = None
    fps = 0.0
    last = time.monotonic()
    analyser = None
    if args.realtime and not source.is_live and not (args.out or args.show):
        def analyse(frame, t, jumped):
            if jumped:
                cam.tracker.tracks.clear()   # a seek or a repeat: no one is followed across the jump (ids go on)
            cam.process(frame, t)
            if preview is not None:
                preview.offer(frame, analyser.fps)
        analyser = Analyser(analyse)
        shown, said = 0, time.monotonic()
        analyser.start()
        if preview is not None:
            preview.live = True              # the frames come from the loop below, every one shown
    try:
        # realtime: a crossing in a video is reported when it happens on screen
        for frame, t in iter_frames(source, realtime=args.realtime):
            if crop is not None:
                h, w = frame.shape[:2]
                frame = frame[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
            jumped = bool(getattr(source, "take_jump", None) and source.take_jump())
            if analyser is not None:
                shown += 1
                if time.monotonic() - said >= 10.0:   # in the camera's messages: the picture and the analysis rates
                    print(f"finish cam: {shown / (time.monotonic() - said):.0f} pictures a second shown, {analyser.fps:.0f} analysed", flush=True)
                    shown, said = 0, time.monotonic()
                if preview is not None:
                    preview.offer_frame(frame)
                analyser.offer(frame, t, jumped)
                if not analyser.first.is_set():
                    analyser.first.wait()    # the models warm up on the first frame; the clip waits
                if analyser.error is not None:
                    raise analyser.error
                continue
            if jumped:
                cam.tracker.tracks.clear()
            cam.process(frame, t)
            now = time.monotonic()
            inst = 1.0 / max(now - last, 1e-6)
            fps = inst if not fps else 0.9 * fps + 0.1 * inst
            last = now
            if preview is not None:
                preview.offer(frame, fps)
            if args.out or args.show:
                draw(frame, cam, fps)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), source.fps or 30.0, (w, h))
                writer.write(np.ascontiguousarray(frame))
            if args.show:
                cv2.imshow("TestPoint finish", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except SourceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if analyser is not None:
            analyser.stop()
            analyser.join(timeout=5.0)
        source.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        cam.close()
        if preview is not None:
            preview.stop()
        if poster is not None:
            poster.close()
        if evidence is not None:
            evidence.close()
    # The end-of-run summary, readable in the camera log: no face pictures (the app has them).
    summary = [{k: v for k, v in c.payload(c.wall).items() if k != "faces"} for c in cam.crossings]
    print(f"finish cam stopped: {len(summary)} crossing(s)")
    if summary:
        print(json.dumps({"crossings": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
