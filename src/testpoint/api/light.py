"""The finish camera's shutter, following the daylight (fairness: every runner's finish picture as
sharp as the light allows, morning, noon or dusk, indoors or out).

A runner crosses the line at 5-8 m/s: at 1/250 s the body moves 2-3 cm while the picture is taken
(a smeared face), at 1/1000 s under 1 cm. A faster shutter takes in less light, which the camera
makes up with its iris and gain until it cannot: the picture then gets dark and grainy. So:

- a dark picture (its median brightness under ``DARK`` for a minute) -> one step slower;
- every ``PROBE_EVERY_S`` while not at the fastest: try one step faster for a minute and keep it only
  if the picture stays as bright (85 %) and not much grainier (noise up to 35 % more); else back.

Brightness and noise come from the finish camera's live picture that already reaches the app (no
extra connection to the camera). Only the finish: at the start line and the exercise station
people stand still or move slowly, and 1/250 s gives them the cleanest picture.
The shutter is set over the camera's ISAPI; its password is read from the database, never shown."""
from __future__ import annotations

import re
import statistics
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable

import cv2
import numpy as np

LADDER = ("1/1000", "1/500", "1/250")
"""The finish camera's shutters, fastest first."""
FIXED = ("1/1000", "1/500", "1/250")
DARK = 60.0
"""A picture darker than this (mean grey level, 0-255) is short of light."""
SAMPLE_EVERY_S = 10.0
WINDOW = 6
"""Samples judged together (a minute): one person in dark clothes passing by does not move it."""
SETTLE_S = 20.0
"""After a change, the camera's exposure settles before the picture is judged."""
PROBE_EVERY_S = 20 * 60.0


def measure(jpeg: bytes) -> tuple[float, float] | None:
    """(brightness, noise) of a picture: mean grey level of the whole picture (what the camera's
    exposure aims at) and Immerkaer's noise estimate on its middle at full resolution (grain shows
    there; the edges hold the labels)."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    h, w = img.shape
    mid = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4].astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(mid, -1, k)[1:-1, 1:-1]
    noise = float(np.abs(r).mean() * np.sqrt(np.pi / 2) / 6)
    return float(img.mean()), noise


class ShutterControl:
    """Decides the finish camera's shutter from its pictures; no camera access (testable)."""

    def __init__(self, level: int = len(LADDER) - 1) -> None:
        self.level = level
        self.samples: deque[tuple[float, float]] = deque(maxlen=WINDOW)
        self.changed_at = float("-inf")
        self.next_probe = float("-inf")
        self.probe: tuple[int, float, float] | None = None
        """While trying a faster shutter: (the level before, its brightness, its noise)."""
        self.last: tuple[float, float] | None = None

    @property
    def shutter(self) -> str:
        return LADDER[self.level]

    def _median(self) -> tuple[float, float]:
        return (statistics.median(s[0] for s in self.samples), statistics.median(s[1] for s in self.samples))

    def _go(self, level: int, now: float) -> int:
        self.level, self.changed_at = level, now
        self.samples.clear()
        return level

    def step(self, now: float, sample: tuple[float, float] | None, busy: bool = False) -> int | None:
        """One picture's measure; returns the new level when the shutter must change. ``busy``: someone
        is running: no trial of a faster shutter then (every runner's finish picture is taken with a
        shutter already proven), and one under way is undone at once."""
        if busy and self.probe is not None:
            before = self.probe[0]
            self.probe = None
            return self._go(before, now)
        if sample is None or now - self.changed_at < SETTLE_S:
            return None
        self.samples.append(sample)
        self.last = sample
        if len(self.samples) < WINDOW:
            return None
        bright, noise = self._median()
        if self.probe is not None:
            before, b0, n0 = self.probe
            self.probe = None
            self.next_probe = now + PROBE_EVERY_S
            if bright >= max(DARK, 0.85 * b0) and noise <= 1.35 * max(n0, 0.5):
                return None              # as good a picture with a faster shutter: keep it
            return self._go(before, now)
        if bright < DARK and self.level < len(LADDER) - 1:
            self.next_probe = now + PROBE_EVERY_S
            return self._go(self.level + 1, now)
        if self.level > 0 and not busy and now >= self.next_probe and bright >= DARK + 10:
            self.probe = (self.level, bright, noise)
            return self._go(self.level - 1, now)
        return None


def _opener(url: str):
    m = re.match(r"rtsps?://([^:/@]+):([^@/]*)@([^/:]+)", url)
    if not m:
        raise ValueError("not an IP camera address")
    user, pw, host = m.groups()
    base = f"http://{host}"
    pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    pm.add_password(None, base, user, pw)
    return base, urllib.request.build_opener(urllib.request.HTTPDigestAuthHandler(pm))


def read_shutter(url: str) -> str | None:
    base, op = _opener(url)
    x = op.open(base + "/ISAPI/Image/channels/1", timeout=5).read().decode()
    m = re.search(r"<ShutterLevel>([^<]*)", x)
    return m.group(1) if m else None


def set_shutter(url: str, shutter: str) -> bool:
    base, op = _opener(url)
    path = base + "/ISAPI/Image/channels/1"
    x = op.open(path, timeout=5).read().decode()
    x, n = re.subn(r"(<ShutterLevel>)[^<]*(</ShutterLevel>)", rf"\g<1>{shutter}\g<2>", x, count=1)
    if not n:
        return False
    r = op.open(urllib.request.Request(path, data=x.encode(), method="PUT",
                                       headers={"Content-Type": "application/xml"}), timeout=10).read().decode()
    return "<statusString>OK" in r


class FinishLight(threading.Thread):
    """Runs ``ShutterControl`` on the finish camera's live picture, in the app."""

    def __init__(self, picture: Callable[[], bytes | None], camera_url: Callable[[], str | None],
                 mode: Callable[[], str], busy: Callable[[], bool] = lambda: False,
                 log: Callable[[str], None] = print) -> None:
        super().__init__(name="finish-light", daemon=True)
        self.picture, self.camera_url, self.mode, self.busy, self.log = picture, camera_url, mode, busy, log
        self.ctl = ShutterControl()
        self.applied: str | None = None
        self.error: str | None = None
        self._stop = threading.Event()

    def status(self) -> dict:
        b = self.ctl.last
        return {"mode": self.mode(), "shutter": self.applied, "trying": self.ctl.probe is not None,
                "brightness": round(b[0]) if b else None, "noise": round(b[1], 1) if b else None,
                "error": self.error}

    def stop(self) -> None:
        self._stop.set()

    def _apply(self, url: str, shutter: str) -> None:
        if self.applied == shutter:
            return
        if set_shutter(url, shutter):
            self.log(f"finish camera: shutter {self.applied or '?'} -> {shutter}")
            self.applied = shutter
            self.error = None
        else:
            self.error = "camera refused the shutter"

    def run(self) -> None:
        while not self._stop.wait(SAMPLE_EVERY_S):
            try:
                url = self.camera_url()
                if not url:
                    continue
                if self.applied is None:
                    self.applied = read_shutter(url)
                    if self.applied in LADDER:
                        self.ctl.level = LADDER.index(self.applied)
                mode = self.mode()
                if mode in FIXED:
                    self.ctl.probe = None
                    self._apply(url, mode)
                    continue
                if self.ctl.probe is None and self.applied in LADDER:
                    self.ctl.level = LADDER.index(self.applied)   # back from a fixed shutter
                jpeg = self.picture()
                if self.ctl.step(time.monotonic(), measure(jpeg) if jpeg else None, self.busy()) is not None:
                    self._apply(url, self.ctl.shutter)
            except Exception as e:   # noqa: BLE001 - a camera away for a moment must not stop the thread
                self.error = type(e).__name__
