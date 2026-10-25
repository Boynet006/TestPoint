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



class _Live:
    """A camera that delivers frames faster than they are taken."""
    is_live = True
    fps = 30.0

    def __init__(self, n):
        self.n, self.i, self.released = n, 0, False

    def read(self):
        import time as _t
        _t.sleep(0.002)
        if self.i >= self.n:
            return None
        self.i += 1
        return self.i

    def release(self):
        self.released = True


def test_latest_frame_drops_what_the_analysis_could_not_take():
    import time as _t

    from testpoint.vision.sources import LatestFrame
    src = LatestFrame(_Live(200))
    first = src.read()
    _t.sleep(0.15)                      # a slow analysis step: the camera moved on meanwhile
    second = src.read()
    assert second - first > 10 and src.dropped > 10   # the newest frame, not the next in line
    while src.read() is not None:
        pass
    src.release()
    assert src.inner.released


class _Video:
    is_live = False
    fps = 10.0

    def __init__(self, n):
        self.n, self.i, self.skipped = n, 0, 0

    def read(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return self.i - 1

    def skip(self):
        if self.i >= self.n:
            return False
        self.i += 1
        self.skipped += 1
        return True


def test_realtime_video_skips_frames_when_the_analysis_is_slow():
    from testpoint.vision.sources import iter_frames
    now = {"t": 0.0}
    clock = lambda: now["t"]
    slept = []
    vid = _Video(100)
    seen = []
    for frame, t in iter_frames(vid, clock=clock, realtime=True, sleep=lambda s: slept.append(s)):
        seen.append(t)
        now["t"] += 0.35                # each frame takes 0.35 s to analyse: 3.5 video frames
        if len(seen) == 5:
            break
    # it keeps to the video's own clock: about one frame in 3-4 is analysed, the rest skipped
    assert vid.skipped >= 9 and abs(seen[-1] - 4 * 0.35) < 0.2


def test_realtime_video_waits_when_the_analysis_is_fast():
    from testpoint.vision.sources import iter_frames
    now = {"t": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        now["t"] += s

    vid = _Video(5)
    out = [t for _f, t in iter_frames(vid, clock=lambda: now["t"], realtime=True, sleep=sleep)]
    assert out == [0.0, 0.1, 0.2, 0.3, 0.4] and vid.skipped == 0 and all(abs(s - 0.1) < 1e-9 for s in slept)



def test_realtime_video_plays_like_a_camera():
    from testpoint.vision.sources import RealtimeVideo
    now = {"t": 0.0}

    def sleep(s):
        now["t"] += s

    class Inner(_Video):
        source = "clip.mp4"

    vid = RealtimeVideo(Inner(100), clock=lambda: now["t"], sleep=sleep)
    assert [vid.read() for _ in range(3)] == [0, 1, 2] and abs(now["t"] - 0.2) < 1e-9   # paced at 10 fps
    now["t"] += 0.55                                    # the reader was held up: skip, don't fall behind
    assert vid.read() == 7 and vid.inner.skipped == 4
