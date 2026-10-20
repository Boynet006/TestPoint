"""Frame sources for the demo CLI: a webcam index, a video file or a live RTSP stream.

Hikvision cameras are read over RTSP with FFmpeg. TCP transport is forced
through ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` (UDP drops packets on a busy switch
and produces grey, smeared frames); the variable has to be in the environment
before OpenCV loads its FFmpeg plugin, hence the ``setdefault`` above the import.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np


class SourceError(RuntimeError):
    """The source could not be opened or was lost and could not be recovered."""


def is_rtsp(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtsps://"))


def redact_url(url: str) -> str:
    """Hide the password in ``scheme://user:password@host`` for logs and messages."""
    return re.sub(r"(://[^/@:]+:)[^@/]*@", r"\1***@", url)


class FileSource:
    """A video file or a local webcam index."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.is_live = source.isdigit()
        self.cap = cv2.VideoCapture(int(source)) if self.is_live else cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SourceError(f"could not open video source {source!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class RtspSource:
    """A live RTSP stream with reconnect.

    Opening and reading use OpenCV's FFmpeg timeouts so that a dead camera is
    noticed after ``timeout_s`` instead of FFmpeg's 30 s default. Every connection
    attempt (initial or after a lost stream) is retried ``max_attempts`` times
    with exponential backoff (1, 2, 4 ... ``max_backoff_s`` seconds); 0 retries
    for ever. When the budget is used up ``SourceError`` is raised.
    """

    is_live = True

    def __init__(
        self,
        url: str,
        timeout_s: float = 10.0,
        max_attempts: int = 3,
        max_backoff_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        capture_factory: Callable[[str], cv2.VideoCapture] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep
        self._capture_factory = capture_factory or self._open_capture
        self._log = log
        self.cap: cv2.VideoCapture | None = None
        self.reconnects = 0
        self.fps = 0.0
        self.connect()

    @property
    def display_url(self) -> str:
        return redact_url(self.url)

    def _open_capture(self, url: str) -> cv2.VideoCapture:
        timeout_ms = int(self.timeout_s * 1000)
        return cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
            ],
        )

    def connect(self) -> None:
        self.release()
        attempt = 0
        backoff = 1.0
        while True:
            attempt += 1
            self._log(f"connecting to {self.display_url} (attempt {attempt})")
            cap = self._capture_factory(self.url)
            if cap.isOpened():
                self.cap = cap
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                # Keep the latest frame only: counting must not lag behind the camera.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._log(f"connected to {self.display_url}")
                return
            cap.release()
            if self.max_attempts and attempt >= self.max_attempts:
                raise SourceError(
                    f"could not connect to {self.display_url} after {attempt} attempts "
                    f"(timeout {self.timeout_s:g} s each). Check the camera IP, the RTSP "
                    "port (554), the user/password and that the PC is on the camera LAN."
                )
            self._log(f"connection failed, retrying in {backoff:g} s")
            self._sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)

    def read(self) -> np.ndarray | None:
        """Return the next frame, reconnecting if the stream drops. Raises SourceError
        when reconnecting fails; never returns None (a live stream has no end)."""
        while True:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if ok and frame is not None:
                return frame
            self.reconnects += 1
            self._log(f"stream lost from {self.display_url}, reconnecting")
            self.connect()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_source(source: str, **rtsp_options) -> FileSource | RtspSource:
    if is_rtsp(source):
        return RtspSource(source, **rtsp_options)
    return FileSource(source)


def iter_frames(source: FileSource | RtspSource, clock: Callable[[], float] = time.monotonic):
    """Yield ``(frame, t)`` until the source ends. Video files use their own clock
    (frame index / fps, so a clip scores the same every run); live cameras use
    wall time, which is what the evidence and the rep times are stamped with."""
    fps = source.fps or 30.0
    t0 = clock()
    idx = 0
    while True:
        frame = source.read()
        if frame is None:
            return
        yield frame, (clock() - t0 if source.is_live else idx / fps)
        idx += 1
