"""Evidence video for the station engine: one full recording per attempt and one short
clip per rep, cropped to the lane (docs/ENGINE-LINK.md).

Files go under ``<root>/<day>/<attempt_id>/`` as ``full.mp4`` and ``rep-NNN.mp4``.
The frame thread only crops and queues; a worker thread does the encoding, so a
slow disk never stalls the pose loop. Clips are cut from a ring buffer of the
last ``keep_s`` seconds, from ``t_start - pad_s`` to ``t_end + pad_s``.

Encoding: the review screen plays clips in a browser ``<video>`` element, which
needs H.264 in MP4 (yuv420p, ``moov`` at the front). OpenCV's own writer only
produces that with an OpenH264 DLL it usually lacks, so frames are piped to an
``ffmpeg`` subprocess (``libx264``) found on PATH or via ``TESTPOINT_FFMPEG``.
Without ffmpeg the writer falls back to OpenCV's ``mp4v`` (MPEG-4 part 2), which
browsers refuse; a log line says which one is in use.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from .vision.lanes import Lane

FOURCC = "mp4v"                     # OpenCV fallback when ffmpeg is missing
FFMPEG_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart"]


def find_ffmpeg() -> str | None:
    """Path of the ffmpeg executable: ``TESTPOINT_FFMPEG`` if set, else the first on PATH."""
    env = os.environ.get("TESTPOINT_FFMPEG")
    if env:
        return env if Path(env).is_file() else shutil.which(env)
    return shutil.which("ffmpeg")


class FfmpegWriter:
    """Pipes raw BGR frames to ffmpeg, which writes browser-playable H.264/MP4.

    Same ``write``/``release``/``isOpened`` surface as ``cv2.VideoWriter``. Odd sizes are
    padded by one pixel (yuv420p needs even dimensions)."""

    def __init__(self, ffmpeg: str, path: Path, fps: float, size: tuple[int, int],
                 log: Callable[[str], None] = print) -> None:
        w, h = size
        self.path, self.size, self.log = path, size, log
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
               "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", *FFMPEG_ARGS, str(path)]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc: subprocess.Popen | None = None
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError as e:
            log(f"evidence: could not start ffmpeg for {path}: {e}")

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as e:
            self.log(f"evidence: ffmpeg stopped taking frames for {self.path}: {e}")
            self._abandon()

    def release(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            rc = self._proc.wait(timeout=60)
            if rc != 0:
                self.log(f"evidence: ffmpeg exited with {rc} writing {self.path}")
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log(f"evidence: ffmpeg did not finish {self.path}: {e}")
            self._abandon()
        self._proc = None

    def _abandon(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


@dataclass
class Recording:
    attempt_id: int
    folder: Path
    writer: cv2.VideoWriter | FfmpegWriter | None = None
    t0: float | None = None
    written: int = 0


class EvidenceWriter:
    def __init__(self, root: str | Path, lanes: list[Lane], fps: float = 30.0,
                 keep_s: float = 4.0, pad_s: float = 0.5, day: str | None = None,
                 log: Callable[[str], None] = print, ffmpeg: str | None = None) -> None:
        """``ffmpeg``: path of the executable; ``None`` looks it up (``find_ffmpeg``) and
        falls back to OpenCV's mp4v writer when there is none."""
        self.root = Path(root)
        self.lanes = {lane.name: lane for lane in lanes}
        self.fps = fps if fps and fps > 0 else 30.0
        self.keep_s = keep_s
        self.pad_s = pad_s
        self.day = day or date.today().isoformat()
        self.log = log
        self.ffmpeg = ffmpeg or find_ffmpeg()
        if self.ffmpeg:
            log(f"evidence: H.264 clips via ffmpeg at {self.ffmpeg}")
        else:
            log("evidence: ffmpeg not found (PATH or TESTPOINT_FFMPEG); writing mp4v with OpenCV, "
                "clips will not play in the browser review screen")
        self._ring: deque[tuple[float, np.ndarray]] = deque()
        self._active: dict[str, Recording] = {}
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="evidence-writer", daemon=True)
        self._worker.start()

    # ----------------------------------------------------------------- frame thread
    def start(self, lane: str, attempt_id: int) -> Path:
        """Begin ``full.mp4`` for an attempt on this lane (ends the previous one, if any)."""
        self.stop(lane)
        folder = self.root / self.day / str(attempt_id)
        if folder.is_dir() and any(folder.iterdir()):      # the number came free (a result was redone): keep those clips
            n = 1
            while (old := folder.with_name(f"{attempt_id}-redone-{n}")).exists():
                n += 1
            folder.rename(old)
        folder.mkdir(parents=True, exist_ok=True)
        self._active[lane] = Recording(attempt_id, folder)
        return folder

    def stop(self, lane: str) -> None:
        rec = self._active.pop(lane, None)
        if rec is not None:
            self._jobs.put(("close", rec))

    def push(self, frame: np.ndarray, t: float) -> None:
        """Remember the frame for clips and append it to every running recording."""
        self._ring.append((t, frame))
        while self._ring and self._ring[0][0] < t - self.keep_s:
            self._ring.popleft()
        for lane, rec in self._active.items():
            # The camera's frames reach us slower than its nominal rate: repeat them so that the
            # recording plays at real speed (and skip when they come faster).
            rec.t0 = t if rec.t0 is None else rec.t0
            due = round((t - rec.t0) * self.fps) + 1
            for _ in range(due - rec.written):
                self._jobs.put(("frame", rec, self._crop(frame, lane)))
            rec.written = max(rec.written, due)

    def clip(self, lane: str, event: dict) -> str | None:
        """Cut the clip for a finished rep. Returns its path relative to ``root``
        (what the report carries), or None when the lane has no attempt running."""
        rec = self._active.get(lane)
        if rec is None:
            return None
        lo, hi = event["t_start"] - self.pad_s, event["t_end"] + self.pad_s
        frames = self._paced([(t, self._crop(f, lane)) for t, f in self._ring if lo <= t <= hi])
        path = rec.folder / f"rep-{int(event['index']):03d}.mp4"
        self._jobs.put(("clip", path, frames))
        return path.relative_to(self.root).as_posix()

    def clip_at(self, folder: str, name: str, t0: float, t1: float, lane: str | None = None) -> str | None:
        """Cut a clip of the ring buffer between ``t0`` and ``t1`` (plus padding) into
        ``<root>/<folder>/<name>.mp4`` without a running recording: a finish-line
        crossing, for instance. ``lane`` crops to that lane; None keeps the whole frame.
        Returns the path relative to ``root``, or None when the window holds no frame."""
        lo, hi = t0 - self.pad_s, t1 + self.pad_s
        frames = self._paced([(t, self._crop(f, lane) if lane else f) for t, f in self._ring if lo <= t <= hi])
        if not frames:
            return None
        path = self.root / folder / f"{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs.put(("clip", path, frames))
        return path.relative_to(self.root).as_posix()

    def close(self, timeout_s: float = 30.0) -> None:
        for lane in list(self._active):
            self.stop(lane)
        self._jobs.put(("quit",))
        self._worker.join(timeout_s)

    def _paced(self, timed: list[tuple[float, np.ndarray]]) -> list[np.ndarray]:
        """The frames at ``fps``, so that a clip plays at real speed whatever rate they came at."""
        out: list[np.ndarray] = []
        for t, f in timed:
            due = round((t - timed[0][0]) * self.fps) + 1
            out.extend([f] * (due - len(out)))
        return out

    def _crop(self, frame: np.ndarray, lane: str) -> np.ndarray:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.lanes[lane].to_pixels(w, h)
        return frame[y0:y1, x0:x1]

    # ---------------------------------------------------------------- worker thread
    def _open(self, path: Path, frame: np.ndarray) -> cv2.VideoWriter | FfmpegWriter:
        h, w = frame.shape[:2]
        if self.ffmpeg:
            writer = FfmpegWriter(self.ffmpeg, path, self.fps, (w, h), self.log)
            if writer.isOpened():
                return writer
            self.log(f"evidence: falling back to {FOURCC} for {path}")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*FOURCC), self.fps, (w, h))
        if not writer.isOpened():
            self.log(f"evidence: could not open {path}")
        return writer

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            kind = job[0]
            if kind == "quit":
                return
            if kind == "frame":
                _, rec, frame = job
                if rec.writer is None:
                    rec.writer = self._open(rec.folder / "full.mp4", frame)
                rec.writer.write(np.ascontiguousarray(frame))
            elif kind == "clip":
                _, path, frames = job
                if not frames:
                    continue
                writer = self._open(path, frames[0])
                for f in frames:
                    writer.write(np.ascontiguousarray(f))
                writer.release()
            elif kind == "close":
                rec = job[1]
                if rec.writer is not None:
                    rec.writer.release()
                    rec.writer = None
