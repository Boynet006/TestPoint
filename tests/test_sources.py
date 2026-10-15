import numpy as np
import pytest

from testpoint.vision.sources import RtspSource, SourceError, is_rtsp, redact_url

URL = "rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101"


class FakeCapture:
    def __init__(self, frames: int, opened: bool = True) -> None:
        self.opened = opened
        self.frames = frames
        self.released = False
        self.props = {}

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 25.0

    def set(self, prop, value):
        self.props[prop] = value

    def read(self):
        if self.frames <= 0:
            return False, None
        self.frames -= 1
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_is_rtsp_and_redaction():
    assert is_rtsp(URL) and is_rtsp("RTSPS://x")
    assert not is_rtsp("clip.mp4") and not is_rtsp("0")
    assert redact_url(URL) == "rtsp://admin:***@192.168.1.64:554/Streaming/Channels/101"
    assert redact_url("rtsp://192.168.1.64/stream") == "rtsp://192.168.1.64/stream"


def test_connect_failure_backs_off_then_fails_cleanly():
    sleeps, logs, caps = [], [], []

    def factory(url):
        cap = FakeCapture(0, opened=False)
        caps.append(cap)
        return cap

    with pytest.raises(SourceError) as info:
        RtspSource(URL, timeout_s=5, max_attempts=3, sleep=sleeps.append,
                   capture_factory=factory, log=logs.append)
    assert len(caps) == 3
    assert sleeps == [1.0, 2.0]
    assert "after 3 attempts" in str(info.value)
    assert "secret" not in str(info.value)
    assert all(c.released for c in caps)


def test_read_reconnects_after_stream_loss():
    sleeps = []
    caps = [FakeCapture(2), FakeCapture(0, opened=False), FakeCapture(3)]

    def factory(url):
        return caps.pop(0)

    src = RtspSource(URL, max_attempts=0, sleep=sleeps.append, capture_factory=factory,
                     log=lambda m: None)
    assert src.fps == 25.0
    got = [src.read() for _ in range(5)]
    assert all(f is not None for f in got)
    assert src.reconnects == 1
    assert sleeps == [1.0]          # one failed attempt during the reconnect, then success
    src.release()
