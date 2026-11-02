"""Start-line camera: recognises the runner at the mark and sees their raised hand.

Reads the start camera (an IP camera on the station PC), and a few times a second:
  * finds the person standing at the mark (pose on the GPU: the largest person in view),
  * reads which hand they hold up (wrist above the shoulder),
  * looks at their face (the face models on the GPU) and sends its embedding,
and reports that to the app (``/api/engine/start/obs``), which recognises them and runs the
start (api/startline.py): "raise your LEFT hand", 3-2-1-GO. The picture goes to the start page
too (``/api/engine/frame/start``), with the person at the mark outlined.

    python -m testpoint.startcam --source rtsp://user:pass@<ip>:554/Streaming/Channels/101 --app http://127.0.0.1:8000

(The camera service starts it from the start page; see testpoint.camhost.)
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from .applink import ssl_context
from .preview import FramePublisher
from .vision.sources import SourceError, open_source, redact_url

STATION = "start"
OBS_PATH = "/api/engine/start/obs"


class ObsPoster(threading.Thread):
    """Sends the newest observation to the app, one request at a time (a slow answer never
    holds up the camera; an older observation not sent yet is simply replaced)."""

    def __init__(self, app_url: str, timeout_s: float = 2.0) -> None:
        super().__init__(name="start-obs", daemon=True)
        self.url = app_url.rstrip("/") + OBS_PATH
        self.timeout_s = timeout_s
        self._obs: dict | None = None
        self._cv = threading.Condition()
        self._stop = False
        self.sent = self.failed = 0
        self.last_reply: dict = {}

    def offer(self, obs: dict) -> None:
        with self._cv:
            self._obs = obs
            self._cv.notify()

    def run(self) -> None:
        while True:
            with self._cv:
                while self._obs is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                obs, self._obs = self._obs, None
            req = urllib.request.Request(self.url, data=json.dumps(obs).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s, context=ssl_context(self.url)) as r:
                    self.last_reply = json.loads(r.read() or b"{}")
                self.sent += 1
            except (urllib.error.URLError, OSError, ValueError) as e:
                if self.failed == 0 or self.failed % 50 == 0:
                    print(f"start cam: the app did not take an observation ({e})", flush=True)
                self.failed += 1

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()


class Analyst:
    """Pose and face on one frame: the person at the mark, their raised hand, their face."""

    def __init__(self, device: str, face_every_s: float = 0.25) -> None:
        from .api.faces import get_engine
        from .vision.pose import make_backend
        self.pose = make_backend("rtmlib", device=device, mode="balanced")
        self.faces = get_engine()
        self.face_every_s = face_every_s
        self._last_face = 0.0

    def look(self, frame: np.ndarray, now: float) -> dict:
        from .api.faces import FaceError
        from .vision.gesture import at_the_mark, box, hand_raised, head_box
        h, w = frame.shape[:2]
        p = at_the_mark(self.pose.detect(frame), h)
        obs: dict = {"present": p is not None}
        draw: dict = {"box": None, "hand": None, "face": None}
        if p is None:
            return {"obs": obs, "draw": draw}
        draw["box"] = box(p)
        obs["hand"] = draw["hand"] = hand_raised(p)
        if now - self._last_face >= self.face_every_s:
            hb = head_box(p, w, h)
            if hb is not None:
                self._last_face = now
                x0, y0, x1, y1 = hb
                try:
                    f = self.faces.largest_face(frame[y0:y1, x0:x1])
                    obs["emb"] = base64.b64encode(np.asarray(f.embedding, np.float32).tobytes()).decode()
                    obs["quality"] = f.quality
                    fx, fy, fw, fh = f.box
                    draw["face"] = (x0 + fx, y0 + fy, x0 + fx + fw, y0 + fy + fh)
                except FaceError:
                    pass
        return {"obs": obs, "draw": draw}


def draw(img: np.ndarray, d: dict) -> None:
    """The person at the mark outlined (green while a hand is up), their face boxed."""
    if not d:
        return
    if d.get("box"):
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        col = (80, 220, 80) if d.get("hand") else (0, 200, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 3)
        if d.get("hand"):
            cv2.putText(img, f"{d['hand'].upper()} HAND", (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    if d.get("face"):
        x0, y0, x1, y1 = (int(v) for v in d["face"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 2)


class StartClips:
    """A short video of every start: from a few seconds before the 3-2-1 until just after GO, for
    checking a start later. Files go to ``<evidence>/start/``. Pictures are kept small: the ring
    holds several seconds of them in memory."""

    AFTER_S = 3.0
    BEFORE_S = 4.0

    def __init__(self, evidence, width: int = 1280) -> None:
        self.evidence, self.width = evidence, width
        self.began: float | None = None
        self.entry: int | None = None
        self.until: float | None = None
        self.saved = 0

    def push(self, frame: np.ndarray, now: float) -> None:
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        self.evidence.push(frame, now)
        if self.until is not None and now >= self.until:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.evidence.clip_at("start", f"{stamp}-entry{self.entry}", self.began - self.BEFORE_S, now)
            self.saved += 1
            self.began = self.until = None

    def state(self, reply: dict, now: float) -> None:
        """The app's answer: a countdown begins the clip, the start ends it soon after."""
        state = reply.get("state")
        if state == "countdown" and self.began is None:
            self.began, self.entry = now, reply.get("entry_id")
        elif state == "started" and self.began is not None and self.until is None:
            self.until = now + self.AFTER_S
        elif state == "idle" and self.until is None:
            self.began = None           # the countdown was cancelled: nothing to keep


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="TestPoint start-line camera (recognise, raised hand, picture)")
    p.add_argument("--source", required=True, help="webcam index, video file or rtsp:// address")
    p.add_argument("--app", required=True, help="station app address, e.g. http://127.0.0.1:8000")
    p.add_argument("--station", default=STATION, help="screen this camera shows on (default start)")
    p.add_argument("--fps", type=float, default=10.0, help="pictures sent to the screen a second (default 10)")
    p.add_argument("--analyse-fps", type=float, default=12.0, help="frames looked at a second (default 12)")
    p.add_argument("--max-width", type=int, default=1920, help="widest picture sent (default 1920)")
    p.add_argument("--device", default="auto", help="cuda, cpu or auto (default)")
    p.add_argument("--no-analyse", action="store_true", help="only relay the picture (no recognition)")
    p.add_argument("--realtime", action="store_true", help="play a video file at its own speed")
    p.add_argument("--loop", action="store_true", help="play a video file on repeat")
    p.add_argument("--evidence", default="data/evidence", help="evidence folder ('' = no videos of the starts)")
    args = p.parse_args(argv)

    src = open_source(args.source, realtime=args.realtime, loop=args.loop)
    pub = FramePublisher(args.app, args.station, draw=draw, interval=1 / args.fps, max_width=args.max_width, quality=88)
    pub.start()
    analyst = poster = None
    if not args.no_analyse:
        device = args.device
        if device == "auto":
            from .vision.face_gpu import cuda_available
            device = "cuda" if cuda_available() else "cpu"
        try:
            analyst = Analyst(device)
            poster = ObsPoster(args.app)
            poster.start()
            print(f"start cam: recognition on {device} (faces on {analyst.faces.device})", flush=True)
        except Exception as e:      # noqa: BLE001 - no models / no GPU: the picture still goes to the screen
            print(f"start cam: recognition off ({e})", flush=True)
    clips = None
    if poster is not None and args.evidence:
        from .evidence import EvidenceWriter
        from .vision.lanes import whole_frame
        clips = StartClips(EvidenceWriter(args.evidence, [whole_frame("1")], fps=args.analyse_fps, keep_s=10.0, pad_s=0.0))
    print(f"start cam: source {redact_url(args.source)}  app {args.app}  station {args.station}", flush=True)
    frames, t0, last_look, state = 0, time.monotonic(), 0.0, {}
    try:
        while True:
            frame = src.read()
            if frame is None:
                print("start cam: the video ended", flush=True)
                break
            now = time.monotonic()
            if analyst is not None and now - last_look >= 1 / args.analyse_fps:
                last_look = now
                r = analyst.look(frame, now)
                state = r["draw"]
                poster.offer(r["obs"])
                if clips is not None:
                    clips.state(poster.last_reply, now)
                    clips.push(frame, now)
            pub.offer(frame, state)
            frames += 1
            if frames == 1:
                h, w = frame.shape[:2]
                print(f"start cam: picture {w}x{h}", flush=True)
            if now - t0 >= 60:
                extra = f", {poster.sent} looks sent" if poster else ""
                print(f"start cam: {frames / (now - t0):.0f} fps from the camera, {pub.sent} pictures sent{extra}", flush=True)
                frames, t0 = 0, now
    except SourceError as e:
        print(f"start cam: {e}", flush=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        pass
    finally:
        if poster:
            poster.stop()
        if clips is not None:
            clips.evidence.close()
        pub.stop()
        src.release()


if __name__ == "__main__":
    main()
