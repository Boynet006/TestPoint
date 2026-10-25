"""Live camera preview: the publisher sends only the newest frame, the app serves the latest picture."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import models
from testpoint.api.app import FRAME_STALE_S, create_app
from testpoint.preview import FramePublisher


def test_publisher_sends_newest_frame_drawn_and_downscaled():
    sent = []
    drawn = []
    pub = FramePublisher("http://app", "pushup", lambda img, st: drawn.append(st) or img.__setitem__((0, 0), 255),
                         max_width=320, post=sent.append)
    assert pub.url == "http://app/api/engine/frame/pushup"
    assert pub.publish_once() is False                       # nothing offered yet
    a = np.zeros((480, 640, 3), np.uint8)
    pub.offer(a, "old")
    pub.offer(a, "new")                                       # only the newest is kept
    assert pub.publish_once() is True and drawn == ["new"]
    img = cv2.imdecode(np.frombuffer(sent[0], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (240, 320, 3)
    assert a[0, 0, 0] == 0                                    # drawn on a copy, not the engine's frame
    assert pub.publish_once() is False                       # no new frame since


def test_publisher_counts_failures_without_raising():
    def boom(_):
        raise OSError("refused")
    pub = FramePublisher("http://app", "finish", lambda img, st: None, post=boom)
    pub.offer(np.zeros((10, 10, 3), np.uint8), None)
    assert pub.publish_once() is False and pub.failed == 1


class Clock:
    t = 50.0

    def __call__(self):
        return self.t


@pytest.fixture()
def app_client(monkeypatch):
    clock = Clock()
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", web_dir="does-not-exist", today=lambda: "2026-10-20", clock=clock)
    with TestClient(app) as c:
        yield c, clock


def test_app_serves_latest_frame_until_stale(app_client):
    c, clock = app_client
    jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    assert c.get("/api/engine/frame/pushup.jpg").status_code == 404
    assert c.post("/api/engine/frame/pushup", content=jpeg, headers={"content-type": "image/jpeg"}).status_code == 204
    r = c.get("/api/engine/frame/pushup.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content == jpeg
    assert c.get("/api/engine/frame/finish.jpg").status_code == 404      # per station
    clock.t += FRAME_STALE_S + 0.1
    assert c.get("/api/engine/frame/pushup.jpg").status_code == 404      # engine stopped
    assert c.post("/api/engine/frame/pushup", content=b"not a jpeg").status_code == 400
    assert c.post("/api/engine/frame/Bad!", content=jpeg).status_code in (400, 404)



def test_app_streams_frames_as_they_arrive(app_client):
    c, clock = app_client
    jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    c.post("/api/engine/frame/finish", content=jpeg, headers={"content-type": "image/jpeg"})
    # the stream ends by itself once the camera's picture is stale: make it stale after the first frame
    import threading
    t = threading.Timer(0.2, lambda: setattr(clock, "t", clock.t + 10))
    t.start()
    with c.stream("GET", "/api/engine/stream/finish") as r:
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        body = b"".join(r.iter_bytes())
    assert body.startswith(b"--frame") and jpeg in body
    assert c.get("/api/engine/stream/Bad!").status_code in (400, 404)


def test_publisher_sends_each_new_frame_soon_but_capped():
    import time
    sent = []
    pub = FramePublisher("http://app", "finish", lambda img, st: None, interval=0.05, post=sent.append)
    pub.start()
    try:
        t0 = time.monotonic()
        pub.offer(np.zeros((20, 20, 3), np.uint8), None)
        while not sent and time.monotonic() - t0 < 1:
            time.sleep(0.005)
        assert sent and time.monotonic() - t0 < 0.2            # sent at once, not after a fixed tick
        for _ in range(20):                                     # a burst of frames: at most one per 50 ms
            pub.offer(np.zeros((20, 20, 3), np.uint8), None)
            time.sleep(0.005)
        time.sleep(0.12)
        assert len(sent) <= 1 + 4
    finally:
        pub.stop()



def test_live_camera_picture_shows_every_frame_with_the_latest_overlay():
    from testpoint.vision.sources import LatestFrame

    class Cam:
        is_live, fps = True, 30.0

        def __init__(self):
            self.i = 0

        def read(self):
            import time
            time.sleep(0.002)
            self.i += 1
            return None if self.i > 50 else np.full((10, 10, 3), self.i, np.uint8)

        def release(self):
            pass

    drawn = []
    sent = []
    pub = FramePublisher("http://app", "finish", lambda img, st: drawn.append((int(img[0, 0, 0]), st)),
                         post=sent.append)
    src = LatestFrame(Cam())
    assert pub.tap(src, prepare=lambda f: f[:5])
    pub.offer(np.zeros((10, 10, 3), np.uint8), "analysis 1")   # the analysis brings only its state
    while src.read() is not None:
        pass
    assert pub.publish_once()
    assert drawn[-1] == (50, "analysis 1")                     # the newest camera frame, the latest overlay
    src.release()


def test_tap_leaves_videos_alone():
    class Video:
        is_live = False

    pub = FramePublisher("http://app", "finish", lambda img, st: None, post=lambda b: None)
    assert not pub.tap(Video()) and not pub.live



def test_publisher_keeps_one_connection_to_the_app():
    import http.server
    import threading

    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(self.client_address[1])
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        pub = FramePublisher(f"http://127.0.0.1:{srv.server_port}", "finish", lambda img, st: None)
        for _ in range(3):
            pub.offer(np.zeros((10, 10, 3), np.uint8), None)
            assert pub.publish_once()
        assert len(seen) == 3 and len(set(seen)) == 1     # the same client port: one connection
    finally:
        srv.shutdown()
