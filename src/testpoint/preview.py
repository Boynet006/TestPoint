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

import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

import cv2
import numpy as np

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
        self._frame: np.ndarray | None = None
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

    def render(self) -> bytes | None:
        with self._lock:
            frame, self._frame = self._frame, None
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
        req = urllib.request.Request(self.url, data=jpeg, method="POST",
                                     headers={"Content-Type": "image/jpeg"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            resp.read()

    def publish_once(self) -> bool:
        jpeg = self.render()
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
                continue
            wait = last + self.interval - time.monotonic()
            if wait > 0 and self._stopping.wait(wait):
                break
            self._new.clear()
            last = time.monotonic()
            self.publish_once()

    def stop(self) -> None:
        self._stopping.set()
        self._new.set()
