"""Camera service: runs on the station PC, next to the cameras and the GPU.

The station app (in Docker, without camera access or the vision libraries) asks this service to
start or stop the finish camera (testpoint.finishcam) with the source the examiner picks on the
finish page: a webcam on this PC, a video file (demo, training) or an IP camera address.

It listens on 127.0.0.1 only: the app reaches it through Docker's host.docker.internal, nothing
else on the network can. One finish camera at a time; its output goes to data/logs/finishcam.log.

    python -m testpoint.camhost            (Start TestPoint.cmd starts it for you)
"""

from __future__ import annotations

import argparse
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
DEFAULT_VIDEO_DIRS = [ROOT / "data" / "clips" / "run", Path.home() / "Desktop" / "TestPoint test videos" / "run"]
APP_URL = re.compile(r"^http://(127\.0\.0\.1|localhost)(:\d{1,5})?$")
CAMERA_URL = re.compile(r"^(rtsp|rtsps|http|https)://\S+$")


def video_dirs() -> list[Path]:
    extra = os.environ.get("TESTPOINT_VIDEO_DIRS")
    dirs = [Path(d) for d in extra.split(os.pathsep)] if extra else DEFAULT_VIDEO_DIRS
    return [d for d in dirs if d.is_dir()]


def list_videos() -> list[dict]:
    out, seen = [], set()
    for d in video_dirs():
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            key = (f.name, f.stat().st_size)      # the same clip in two folders (a desktop copy) once
            if key in seen:
                continue
            seen.add(key)
            out.append({"path": str(f), "label": f"{d.name}/{f.name}", "mb": round(f.stat().st_size / 1e6, 1)})
    return out


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean(line: str) -> str:
    """A log line as text: some libraries write UTF-16 or colour codes into the shared log."""
    return ANSI.sub("", line.replace("\x00", "")).strip()


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


class StartIn(BaseModel):
    kind: str = Field(pattern="^(webcam|video|url)$")
    source: str
    line_y: float = Field(default=0.85, ge=0.05, le=0.95)
    """The finish line across the picture, as a share of its height from the top."""
    direction: str = Field(default="down", pattern="^(down|up|any)$")
    app_url: str
    """Where the camera reports crossings: the station app as this PC sees it."""


class FinishCamProcess:
    """The one finish-camera process, started and stopped on request."""

    def __init__(self, log_dir: Path, runner=subprocess.Popen):
        self.log = log_dir / "finishcam.log"
        self.runner = runner
        self.proc = None
        self.config: dict | None = None
        self.since: float | None = None
        self.lock = threading.Lock()

    def command(self, cfg: StartIn, device: str) -> list[str]:
        y = round(cfg.line_y, 3)
        cmd = [sys.executable, "-m", "testpoint.finishcam", "--source", cfg.source,
               "--line", f"0,{y},1,{y}", "--direction", cfg.direction, "--device", device, "--app", cfg.app_url]
        if cfg.kind == "video":
            cmd.append("--realtime")       # play at its own speed, as a camera would
        return cmd

    def start(self, cfg: StartIn) -> dict:
        if not APP_URL.match(cfg.app_url):
            raise HTTPException(400, "the app address must be http://127.0.0.1 or localhost")
        if cfg.kind == "webcam" and not cfg.source.isdigit():
            raise HTTPException(400, "a webcam is chosen by its number")
        if cfg.kind == "video" and cfg.source not in {v["path"] for v in list_videos()}:
            raise HTTPException(400, "pick one of the listed videos")
        if cfg.kind == "url" and not CAMERA_URL.match(cfg.source):
            raise HTTPException(400, "a camera address starts with rtsp:// or http://")
        with self.lock:
            self._stop()
            self.log.parent.mkdir(parents=True, exist_ok=True)
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            out = open(self.log, "w", encoding="utf-8", errors="replace")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            self.proc = self.runner(self.command(cfg, "cuda" if gpu_available() else "cpu"), cwd=str(ROOT),
                                    stdout=out, stderr=subprocess.STDOUT, creationflags=flags, env=env)
            out.close()
            self.config = cfg.model_dump()
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


def create_service(log_dir: Path | None = None, runner=subprocess.Popen) -> FastAPI:
    app = FastAPI(title="TestPoint camera service")
    cam = FinishCamProcess(log_dir or ROOT / "data" / "logs", runner=runner)
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
        return cam.status()

    @app.post("/finish/start")
    def finish_start(body: StartIn) -> dict:
        return cam.start(body)

    @app.post("/finish/stop")
    def finish_stop() -> dict:
        return cam.stop()

    app.state.cam = cam
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="TestPoint camera service (finish camera control)")
    p.add_argument("--port", type=int, default=int(os.environ.get("TESTPOINT_CAMHOST_PORT", "8099")))
    args = p.parse_args(argv)
    uvicorn.run(create_service(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
