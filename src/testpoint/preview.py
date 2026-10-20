"""Live camera preview for the station screens: the engine's annotated picture.

The frame loop hands over its latest frame (no copy, no encoding) with
``offer``; a background thread draws the overlay on a copy, encodes a JPEG
(at most ``max_width`` wide) and POSTs it to ``/api/engine/frame/<station>``
every ``interval`` seconds. Only the newest frame is ever sent, one request at
a time, so a slow network lowers the preview rate but never the counting rate.
The app keeps the last picture per station in memory and the tablet polls it.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from collections.abc import Callable

import cv2
import numpy as np

FRAME_PATH = "/api/engine/frame/"


class FramePublisher(threading.Thread):
    def __init__(self, app_url: str, station: str, draw: Callable[[np.ndarray, object], None],
                 interval: float = 0.3, max_width: int = 960, quality: int = 75,
                 timeout_s: float = 2.0, post: Callable[[bytes], None] | None = None) -> None:
        super().__init__(name="preview-publisher", daemon=True)
        self.url = app_url.rstrip("/") + FRAME_PATH + station
        self.draw = draw
        self.interval = interval
        self.max_width = max_width
        self.quality = quality
        self.timeout_s = timeout_s
        self._post = post or self._http_post
        self._latest: tuple[np.ndarray, object] | None = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self.sent = 0
        self.failed = 0

    def offer(self, frame: np.ndarray, state: object) -> None:
        """Called by the frame loop for every frame: just keeps a reference to the newest one."""
        with self._lock:
            self._latest = (frame, state)

    def render(self) -> bytes | None:
        with self._lock:
            latest, self._latest = self._latest, None
        if latest is None:
            return None
        frame, state = latest
        img = frame.copy()
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
        while not self._stopping.wait(self.interval):
            self.publish_once()

    def stop(self) -> None:
        self._stopping.set()
