"""Camera service: runs on the station PC, next to the cameras and the GPU.

The station app (in Docker, without camera access or the vision libraries) asks this service to
start or stop the camera programs with the source the examiner picks on the page: a webcam on
this PC, a video file (demo, training) or an IP camera address.

- the finish camera (testpoint.finishcam), from the finish page;
- the exercise-station engine (testpoint.engine), from the exercise station page, with the
  exercise and the lanes drawn on its picture;
- the start-line camera (testpoint.startcam), from the start page: an IP camera's picture for
  face check-in at the line.

It listens on 127.0.0.1 only: the app reaches it through Docker's host.docker.internal, nothing
else on the network can. One process of each kind at a time; output goes to data/logs/.

    python -m testpoint.camhost            (Start TestPoint.cmd starts it for you)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
VIDEO_GROUPS = ("run", "pushup", "crunch")
DEFAULT_VIDEO_ROOTS = [ROOT / "data" / "clips", Path.home() / "Desktop" / "TestPoint test videos"]
APP_URL = re.compile(r"^https?://(127\.0\.0\.1|localhost)(:\d{1,5})?$")
CAMERA_URL = re.compile(r"^(rtsp|rtsps|http|https)://\S+$")


def video_roots() -> list[Path]:
    extra = os.environ.get("TESTPOINT_VIDEO_DIRS")
    roots = [Path(d) for d in extra.split(os.pathsep)] if extra else DEFAULT_VIDEO_ROOTS
    return [d for d in roots if d.is_dir()]


def list_videos() -> list[dict]:
    """Videos in each root's run / pushup / crunch folder; the same clip in two roots once."""
    out, seen = [], set()
    for root in video_roots():
        for group in VIDEO_GROUPS:
            d = root / group
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in VIDEO_EXT:
                    continue
                key = (group, f.name, f.stat().st_size)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"path": str(f), "group": group, "label": f"{group}/{f.name}", "mb": round(f.stat().st_size / 1e6, 1)})
    return out


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PICTURE = re.compile(r"[A-Za-z0-9+/=]{120,}")   # base64 picture data


def clean(line: str, limit: int = 300) -> str:
    """A log line as text: some libraries write UTF-16 or colour codes into the shared log. A line
    of picture data (base64) means nothing to read and is dropped (empty); others are cut to ``limit``."""
    text = ANSI.sub("", line.replace("\x00", "")).strip()
    if PICTURE.search(text):
        return ""
    return text if len(text) <= limit else text[:limit] + " …"


def list_cameras(max_index: int = 4) -> list[dict]:
    """Webcams that open right now. A camera another program holds (Chrome, say) does not."""
    import cv2

    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        if ok:
            found.append({"index": i, "label": f"Camera {i + 1}", "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]})
        cap.release()
    return found


def gpu_available() -> bool:
    try:
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        return False


class SourceIn(BaseModel):
    kind: str = Field(pattern="^(webcam|video|url)$")
    source: str
    app_url: str
    """Where the camera reports: the station app as this PC sees it."""
    loop: bool = False
    """A video file: play it on repeat."""


class StartIn(SourceIn):
    """The finish camera."""
    line_y: float = Field(default=0.85, ge=0.05, le=0.95)
    """The finish line across the picture, as a share of its height from the top."""
    direction: str = Field(default="down", pattern="^(down|up|any)$")


class StationIn(SourceIn):
    """The exercise-station engine: one camera watching side-by-side lanes."""
    tuning: dict[str, float] = {}
    """Counting rules to replace for this exercise (the station's settings; the engine checks them)."""
    exercise: str = Field(pattern="^(pushup|crunch)$")
    lanes: int = Field(default=3, ge=1, le=4)
    dividers: list[float] = []
    """Lane boundaries across the picture (0..1 from the left), one fewer than the lanes;
    empty = equal strips."""


def check_source(cfg: SourceIn) -> None:
    if not APP_URL.match(cfg.app_url):
        raise HTTPException(400, "the app address must be 127.0.0.1 or localhost on this PC")
    if cfg.kind == "webcam" and not cfg.source.isdigit():
        raise HTTPException(400, "a webcam is chosen by its number")
    if cfg.kind == "video" and cfg.source not in {v["path"] for v in list_videos()}:
        raise HTTPException(400, "pick one of the listed videos")
    if cfg.kind == "url" and not CAMERA_URL.match(cfg.source):
        raise HTTPException(400, "a camera address starts with rtsp:// or http://")


def lane_rects(lanes: int, dividers: list[float]) -> list[dict]:
    if not dividers:
        dividers = [i / lanes for i in range(1, lanes)]
    if len(dividers) != lanes - 1 or any(not 0.02 <= d <= 0.98 for d in dividers) or dividers != sorted(dividers) \
            or any(b - a < 0.05 for a, b in zip([0.0, *dividers], [*dividers, 1.0])):
        raise HTTPException(400, "lane boundaries must be in order, at least 5% apart")
    xs = [0.0, *dividers, 1.0]
    return [{"name": str(i + 1), "rect": [round(xs[i], 4), 0.0, round(xs[i + 1], 4), 1.0]} for i in range(lanes)]


class CamProcess:
    """One camera program, started and stopped on request."""

    def __init__(self, name: str, log_dir: Path, runner=subprocess.Popen):
        self.name = name
        self.log_dir = log_dir
        self.log = log_dir / f"{name}.log"
        self.runner = runner
        self.proc = None
        self.config: dict | None = None
        self.since: float | None = None
        self.lock = threading.Lock()

    def start(self, cmd: list[str], config: dict) -> dict:
        with self.lock:
            self._stop()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            out = open(self.log, "w", encoding="utf-8", errors="replace")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            self.proc = self.runner(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, creationflags=flags, env=env)
            out.close()
            self.config = config
            self.since = time.time()
        return self.status()

    def _stop(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    def stop(self) -> dict:
        with self.lock:
            self._stop()
            self.config = None
        return self.status()

    def status(self) -> dict:
        p = self.proc
        running = p is not None and p.poll() is None
        tail = []
        if self.log.exists():
            lines = [clean(ln) for ln in self.log.read_text(encoding="utf-8", errors="replace").splitlines()]
            tail = [ln for ln in lines if ln and ":onnxruntime:" not in ln and not ln.startswith("onnxruntime::")][-6:]   # GPU-library chatter
        return {"running": running, "config": self.config, "since": self.since,
                "exit_code": None if p is None or running else p.poll(), "log": tail}


def device() -> str:
    return "cuda" if gpu_available() else "cpu"


def app_address(url: str) -> str:
    """The app as the camera processes reach it. "localhost" becomes 127.0.0.1: on Windows it
    tries IPv6 first, which Docker Desktop forwards far more slowly (about 40 ms more per
    picture or report on a kept-open connection)."""
    return re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", url)


def finish_command(cfg: StartIn) -> list[str]:
    y = round(cfg.line_y, 3)
    cmd = [sys.executable, "-m", "testpoint.finishcam", "--source", cfg.source,
           "--line", f"0,{y},1,{y}", "--direction", cfg.direction, "--device", device(),
           "--app", app_address(cfg.app_url)]
    if cfg.kind == "video":
        cmd.append("--realtime")       # play at its own speed, as a camera would
        if cfg.loop:
            cmd.append("--loop")
    return cmd


def start_command(cfg: SourceIn) -> list[str]:
    cmd = [sys.executable, "-m", "testpoint.startcam", "--source", cfg.source, "--app", app_address(cfg.app_url)]
    if cfg.kind == "video":
        cmd.append("--realtime")
        if cfg.loop:
            cmd.append("--loop")
    return cmd


def station_command(cfg: StationIn, lanes_file: Path) -> list[str]:
    cmd = [sys.executable, "-m", "testpoint.engine", "--exercise", cfg.exercise, "--source", cfg.source,
           "--lanes-file", str(lanes_file), "--device", device(), "--app", app_address(cfg.app_url)]
    if cfg.kind == "video":
        cmd.append("--realtime")
        if cfg.loop:
            cmd.append("--loop")
    if cfg.tuning:
        cmd += ["--tuning", json.dumps(cfg.tuning, sort_keys=True)]
    return cmd


def create_service(log_dir: Path | None = None, runner=subprocess.Popen) -> FastAPI:
    app = FastAPI(title="TestPoint camera service")
    logs = log_dir or ROOT / "data" / "logs"
    finish = CamProcess("finishcam", logs, runner=runner)
    station = CamProcess("station", logs, runner=runner)
    startcam = CamProcess("startcam", logs, runner=runner)
    cache: dict = {"at": 0.0, "cameras": []}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/sources")
    def sources(refresh: bool = False) -> dict:
        # Opening every webcam takes a moment: remember the answer for half a minute.
        if refresh or time.time() - cache["at"] > 30:
            try:
                cache["cameras"] = list_cameras()
            except ImportError:
                cache["cameras"] = []
            cache["at"] = time.time()
        return {"cameras": cache["cameras"], "videos": list_videos(), "gpu": gpu_available()}

    @app.get("/finish")
    def finish_status() -> dict:
        return finish.status()

    @app.post("/finish/start")
    def finish_start(body: StartIn) -> dict:
        check_source(body)
        return finish.start(finish_command(body), body.model_dump())

    @app.post("/finish/stop")
    def finish_stop() -> dict:
        return finish.stop()

    @app.get("/station")
    def station_status() -> dict:
        return station.status()

    @app.post("/station/start")
    def station_start(body: StationIn) -> dict:
        check_source(body)
        rects = lane_rects(body.lanes, body.dividers)
        logs.mkdir(parents=True, exist_ok=True)
        lanes_file = logs / "station_lanes.json"
        lanes_file.write_text(json.dumps({"lanes": rects}, indent=2), encoding="utf-8")
        return station.start(station_command(body, lanes_file), {**body.model_dump(), "rects": rects})

    @app.post("/station/stop")
    def station_stop() -> dict:
        return station.stop()

    # the start-line camera: relays an IP camera's picture for face check-in at the line
    @app.get("/startcam")
    def startcam_status() -> dict:
        return startcam.status()

    @app.post("/startcam/start")
    def startcam_start(body: SourceIn) -> dict:
        check_source(body)
        return startcam.start(start_command(body), body.model_dump())

    @app.post("/startcam/stop")
    def startcam_stop() -> dict:
        return startcam.stop()

    app.state.cam = finish
    app.state.station = station
    app.state.startcam = startcam
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="TestPoint camera service (finish camera and exercise-station engine)")
    p.add_argument("--port", type=int, default=int(os.environ.get("TESTPOINT_CAMHOST_PORT", "8099")))
    args = p.parse_args(argv)
    uvicorn.run(create_service(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
