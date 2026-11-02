"""Frame sources for the demo CLI: a webcam index, a video file or a live RTSP stream.

Hikvision cameras are read over RTSP with FFmpeg. TCP transport is forced
through ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` (UDP drops packets on a busy switch
and produces grey, smeared frames); the variable has to be in the environment
before OpenCV loads its FFmpeg plugin, hence the ``setdefault`` above the import.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket
import sys
import threading
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


def rtsp_login_check(url: str, timeout_s: float = 5.0) -> bool | None:
    """Ask the camera once (RTSP DESCRIBE, digest or basic auth) whether it accepts the
    user/password in ``url``. False: it answered 401, a wrong password, and retrying would
    only get the camera to lock itself (Hikvision locks after a few failures). True: it
    accepted. None: no answer or no password in the URL, so the caller just tries to open."""
    m = re.match(r"rtsps?://([^:/@]+):([^@/]*)@([^/:]+)(?::(\d+))?", url, re.I)
    if not m:
        return None
    user, pw, host, port = m.group(1), m.group(2), m.group(3), int(m.group(4) or 554)
    clean = re.sub(r"://[^@/]*@", "://", url)

    def ask(sock: socket.socket, cseq: int, auth: str = "") -> tuple[int, str]:
        sock.sendall((f"DESCRIBE {clean} RTSP/1.0\r\nCSeq: {cseq}\r\nAccept: application/sdp\r\n"
                      f"{auth}\r\n").encode())
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        head = data.decode("latin-1")
        code = re.match(r"RTSP/1\.0 (\d+)", head)
        return (int(code.group(1)) if code else 0), head

    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            code, head = ask(sock, 1)
            if code != 401:
                return None if code == 0 else True
            ch = re.search(r"WWW-Authenticate:\s*(Digest|Basic)\s*(.*)", head, re.I)
            if not ch:
                return None
            if ch.group(1).lower() == "basic":
                import base64
                auth = "Authorization: Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode() + "\r\n"
            else:
                f = dict(re.findall(r'(\w+)="([^"]*)"', ch.group(2)))
                md5 = lambda t: hashlib.md5(t.encode()).hexdigest()  # noqa: E731
                resp = md5(f"{md5(f'{user}:{f.get('realm', '')}:{pw}')}:{f.get('nonce', '')}:{md5(f'DESCRIBE:{clean}')}")
                auth = (f'Authorization: Digest username="{user}", realm="{f.get("realm", "")}", '
                        f'nonce="{f.get("nonce", "")}", uri="{clean}", response="{resp}"\r\n')
            code, _ = ask(sock, 2, auth)
            return False if code == 401 else (None if code == 0 else True)
    except OSError:
        return None


class FileSource:
    """A video file or a local webcam index."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.is_live = source.isdigit()
        if self.is_live and sys.platform == "win32":
            # DirectShow opens at once and adds less delay than Windows' default (Media Foundation),
            # and MJPG lets USB webcams send 720p at full frame rate. A camera without MJPG keeps
            # its default size: raw 720p would cost it frame rate (often 20 a second or fewer).
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)
            if int(self.cap.get(cv2.CAP_PROP_FOURCC)) == mjpg:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.cap = cv2.VideoCapture(int(source)) if self.is_live else cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SourceError(f"could not open video source {source!r}")
        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def skip(self) -> bool:
        """Move past one frame without decoding it (a video falling behind real time)."""
        return bool(self.cap.grab())

    def rewind(self) -> bool:
        """Back to the first frame (a video played on repeat)."""
        return bool(self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0))

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
        self._login_check = rtsp_login_check if capture_factory is None else (lambda url: None)
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
        if self._login_check(self.url) is False:
            raise SourceError(
                f"the camera at {self.display_url} rejected the user/password (401 Unauthorized). "
                "Not retrying, so the camera does not lock itself: correct the password in "
                "Settings -> Cameras, then start the camera again."
            )
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


class LatestFrame:
    """A live camera read on its own thread, keeping only the newest frame.

    A camera sends frames at its own pace (25-30 a second). When the analysis is slower, the
    frames it has not taken pile up in the driver's or FFmpeg's buffer and the analysis works
    further and further in the past: counts and crossings arrive late, the picture lags. Here
    every ``read()`` returns the newest frame and the ones in between are dropped (counted in
    ``dropped``), so the delay stays one frame however slow the analysis is."""

    is_live = True

    def __init__(self, inner) -> None:
        self.inner = inner
        self.fps = inner.fps
        self.dropped = 0
        self.on_frame: Callable[[np.ndarray], None] | None = None
        """Called on the reader thread with every frame the camera sends (the live picture);
        must be quick and must not keep the thread waiting."""
        self._frame = None
        self._seq = 0
        self._given = 0
        self._error: Exception | None = None
        self._done = False
        self._stop = False
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    @property
    def display_url(self) -> str:
        return getattr(self.inner, "display_url", getattr(self.inner, "source", ""))

    def _run(self) -> None:
        while not self._stop:
            try:
                frame = self.inner.read()
            except Exception as e:          # a lost camera that could not reconnect
                with self._cv:
                    self._error, self._done = e, True
                    self._cv.notify_all()
                return
            with self._cv:
                if frame is None:
                    self._done = True
                    self._cv.notify_all()
                    return
                if self._seq > self._given:
                    self.dropped += 1       # the previous newest was never taken
                self._frame, self._seq = frame, self._seq + 1
                self._cv.notify_all()
            tap = self.on_frame
            if tap is not None:
                try:
                    tap(frame)
                except Exception:           # the picture must never stop the analysis
                    pass

    def read(self) -> np.ndarray | None:
        with self._cv:
            self._cv.wait_for(lambda: self._seq > self._given or self._done)
            if self._seq > self._given:
                self._given = self._seq
                return self._frame
            if self._error is not None:
                raise self._error
            return None

    def release(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self.inner.release()


class RealtimeVideo:
    """A video file that hands out its frames at its own speed, as a camera would: ``read()``
    waits for the next frame's time and, if decoding fell behind, skips frames without
    decoding them. Wrapped in ``LatestFrame`` a demo video behaves exactly like a live camera:
    the picture shows every frame, the analysis takes the newest."""

    is_live = False

    def __init__(self, inner: FileSource, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, loop: bool = False) -> None:
        self.inner = inner
        self.source = inner.source
        self.fps = inner.fps or 30.0
        self.clock, self.sleep = clock, sleep
        self.loop = loop
        """Start again from the beginning at the end (a short demo clip keeps counting)."""
        self.loops = 0
        self._t0: float | None = None
        self._idx = 0

    def read(self) -> np.ndarray | None:
        frame = self._next()
        if frame is None and self.loop and self._idx > 1 and self.inner.rewind():
            self.loops += 1
            self._t0, self._idx = None, 0
            frame = self._next()
        return frame

    def _next(self) -> np.ndarray | None:
        if self._t0 is None:
            self._t0 = self.clock()
        due = (self.clock() - self._t0) * self.fps
        while self._idx + 1 < due:
            if not self.inner.skip():
                return None
            self._idx += 1
        wait = self._t0 + self._idx / self.fps - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._idx += 1
        return self.inner.read()

    def release(self) -> None:
        self.inner.release()


def open_source(source: str, latest: bool = True, realtime: bool = False, loop: bool = False, **rtsp_options):
    """A webcam index, a video file or an RTSP address. Live cameras come with only their newest
    frame (``LatestFrame``) unless ``latest`` is False. ``realtime``: a video file is played like
    a camera (``RealtimeVideo`` in a ``LatestFrame``); ``loop``: on repeat (with ``realtime``)."""
    src = RtspSource(source, **rtsp_options) if is_rtsp(source) else FileSource(source)
    if src.is_live:
        return LatestFrame(src) if latest else src
    return LatestFrame(RealtimeVideo(src, loop=loop)) if realtime else src


def iter_frames(source, clock: Callable[[], float] = time.monotonic, realtime: bool = False,
                sleep: Callable[[float], None] = time.sleep):
    """Yield ``(frame, t)`` until the source ends. Video files use their own clock
    (frame index / fps, so a clip scores the same every run); live cameras use
    wall time, which is what the evidence and the rep times are stamped with.

    ``realtime``: a video file plays at its own speed, as a camera would: it waits when the
    analysis is ahead and skips frames (without decoding them) when it is behind, so it never
    drifts into slow motion."""
    fps = source.fps or 30.0
    t0 = clock()
    idx = 0
    pace = realtime and not source.is_live
    while True:
        if pace:
            due = (clock() - t0) * fps           # the frame that should be on screen now
            while idx + 1 < due and hasattr(source, "skip"):
                if not source.skip():
                    return
                idx += 1
            wait = t0 + idx / fps - clock()
            if wait > 0:
                sleep(wait)
        frame = source.read()
        if frame is None:
            return
        yield frame, (clock() - t0 if source.is_live else idx / fps)
        idx += 1
