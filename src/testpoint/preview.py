"""Live camera preview for the station screens: the engine's annotated picture.

The frame loop hands over its latest frame and analysis state (no copy, no
encoding) with ``offer``; a background thread draws the overlay on a copy,
encodes a JPEG (at most ``max_width`` wide) and POSTs it to
``/api/engine/frame/<station>`` as soon as it is ready. Only the newest frame is
ever sent, one request at a time, so a slow network lowers the preview rate but
never the counting rate. The app streams the newest picture to the screens.

With a live camera, ``tap`` feeds the picture straight from the camera instead:
every frame it sends is shown, with the latest analysis drawn on it, so the
picture moves at the camera's own speed whatever the analysis costs (the
standard split of video and analytics overlay).
"""

from __future__ import annotations

import http.client
import socket
import threading
import time
import urllib.error
import urllib.parse
from collections.abc import Callable

import cv2
import numpy as np

from .applink import ssl_context

FRAME_PATH = "/api/engine/frame/"
_NO_STATE = object()


class FramePublisher(threading.Thread):
    def __init__(self, app_url: str, station: str, draw: Callable[[np.ndarray, object], None],
                 interval: float = 1 / 30, max_width: int = 960, quality: int = 75,
                 timeout_s: float = 2.0, post: Callable[[bytes], None] | None = None) -> None:
        super().__init__(name="preview-publisher", daemon=True)
        self.url = app_url.rstrip("/") + FRAME_PATH + station
        self.draw = draw
        self.interval = interval
        self.max_width = max_width
        self.quality = quality
        self.timeout_s = timeout_s
        self._post = post or self._http_post
        self._conn: http.client.HTTPConnection | None = None
        self._frame: np.ndarray | None = None
        self._held: np.ndarray | None = None
        """The last picture sent: sent again now and then while no new one comes (a paused test
        video), so the page keeps showing it, with the latest counts drawn on it."""
        self.keepalive_s = 1.0
        self.watchers: int | None = None
        """Screens showing this camera now (the app's answer to each picture); None until known.
        Nobody watching: one picture a second, enough for the screens to know the camera is there."""
        self.idle_interval = 1.0
        self._state: object = _NO_STATE
        self.live = False
        """True once ``tap`` feeds the picture from the camera: ``offer`` then brings only the state."""
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._new = threading.Event()
        self.sent = 0
        self.failed = 0

    def offer(self, frame: np.ndarray, state: object) -> None:
        """Called by the frame loop for every analysed frame: just keeps a reference to the newest."""
        with self._lock:
            self._state = state
            if self.live:
                return                  # the camera brings the picture
            self._frame = frame
        self._new.set()

    def offer_frame(self, frame: np.ndarray) -> None:
        """A picture straight from the camera (``tap``): shown with the latest state drawn on it."""
        with self._lock:
            self._frame = frame
        self._new.set()

    def tap(self, source, prepare: Callable[[np.ndarray], np.ndarray] | None = None) -> bool:
        """Show every frame of a live camera (``LatestFrame``) instead of only the analysed ones.
        ``prepare`` gets each frame first (the finish camera's crop). Videos and other sources
        are left as they are; returns whether the picture now comes from the camera."""
        if not getattr(source, "is_live", False) or not hasattr(source, "on_frame"):
            return False
        self.live = True
        source.on_frame = self.offer_frame if prepare is None else (lambda f: self.offer_frame(prepare(f)))
        return True

    def render(self, again: bool = False) -> bytes | None:
        with self._lock:
            frame, self._frame = self._frame, None
            if frame is None and again:
                frame = self._held
            if frame is not None:
                self._held = frame
            state = self._state
        if frame is None:
            return None
        img = frame.copy()
        if state is not _NO_STATE:      # before the first analysis: the bare picture
            self.draw(img, state)
        h, w = img.shape[:2]
        if w > self.max_width:
            img = cv2.resize(img, (self.max_width, int(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    def _http_post(self, jpeg: bytes) -> None:
        # One connection kept open (a new one per picture costs a TCP set-up each time); a
        # connection the app closed meanwhile is opened again once.
        u = urllib.parse.urlsplit(self.url)
        for attempt in (0, 1):
            if self._conn is None:
                if u.scheme == "https":
                    self._conn = http.client.HTTPSConnection(u.hostname, u.port, timeout=self.timeout_s,
                                                             context=ssl_context(self.url))
                else:
                    self._conn = http.client.HTTPConnection(u.hostname, u.port, timeout=self.timeout_s)
            try:
                if self._conn.sock is None:
                    self._conn.connect()
                    # no Nagle: a picture leaves at once instead of waiting ~40 ms for an ACK
                    self._conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._conn.request("POST", u.path, body=jpeg, headers={"Content-Type": "image/jpeg"})
                resp = self._conn.getresponse()
                resp.read()
                w = resp.getheader("X-Watchers")
                self.watchers = int(w) if w is not None and w.isdigit() else None
                if resp.status >= 400:
                    raise OSError(f"app answered {resp.status}")
                return
            except (http.client.HTTPException, OSError):
                self._conn.close()
                self._conn = None
                if attempt:
                    raise

    def publish_once(self, again: bool = False) -> bool:
        jpeg = self.render(again)
        if jpeg is None:
            return False
        try:
            self._post(jpeg)
        except (urllib.error.URLError, OSError, ValueError):
            self.failed += 1      # the report sender already logs an unreachable app
            return False
        self.sent += 1
        return True

    def run(self) -> None:
        # Send each frame as soon as it is ready, at most once per ``interval`` (30 a second):
        # the picture on the page is as fresh as the analysis, without flooding the network.
        last = 0.0
        while not self._stopping.is_set():
            if not self._new.wait(0.5):
                if time.monotonic() - last >= self.keepalive_s:
                    last = time.monotonic()
                    self.publish_once(again=True)
                continue
            interval = self.idle_interval if self.watchers == 0 else self.interval   # nobody watching: no need to encode 30 a second
            wait = last + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)        # precise; Event.wait moves in ~16 ms steps on Windows
            if self._stopping.is_set():
                break
            self._new.clear()
            last = time.monotonic()
            self.publish_once()

    def stop(self) -> None:
        self._stopping.set()
        self._new.set()
