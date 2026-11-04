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


def test_the_last_picture_is_sent_again_while_none_comes():
    import numpy as np

    from testpoint.preview import FramePublisher
    sent = []
    pub = FramePublisher("http://127.0.0.1:1", "pushup", lambda img, state: None, post=sent.append)
    assert not pub.publish_once(again=True)            # nothing seen yet: nothing to repeat
    pub.offer(np.zeros((4, 4, 3), np.uint8), "state")
    assert pub.publish_once() and not pub.publish_once()
    assert pub.publish_once(again=True) and len(sent) == 2


def test_the_app_tells_the_camera_how_many_watch_and_measures_its_rate(tmp_path):
    from fastapi.testclient import TestClient

    from testpoint.api.app import create_app
    now = [100.0]
    app = create_app(db_url=f"sqlite:///{tmp_path / 'a.db'}", web_dir="does-not-exist", clock=lambda: now[0])
    c = TestClient(app)
    jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    for i in range(6):
        now[0] = 100.0 + i * 0.25
        r = c.post("/api/engine/frame/finish", content=jpeg, headers={"content-type": "image/jpeg"})
        assert r.status_code == 204 and r.headers["X-Watchers"] == "0"      # nobody watching yet
    live = c.get("/api/engine/live").json()["finish"]
    assert live["live"] and live["watchers"] == 0 and live["fps"] == 3.0


def test_nobody_watching_the_camera_sends_one_picture_a_second():
    pub = FramePublisher("http://app", "finish", lambda img, s: None, post=lambda b: None)
    pub.watchers = 0
    assert pub.idle_interval == 1.0 and pub.watchers == 0


def test_the_finish_screens_player_reaches_the_cameras_test_video(tmp_path):
    import json
    app = create_app(db_url=f"sqlite:///{tmp_path / 'a.db'}", web_dir="does-not-exist")
    c = TestClient(app)
    jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    status = {"t": 1.5, "duration": 10.0, "paused": False, "speed": 1.0}
    post = lambda: c.post("/api/engine/frame/finish", content=jpeg, headers={"content-type": "image/jpeg", "X-Player": json.dumps(status)})
    assert "X-Player-Cmd" not in post().headers                        # nothing asked yet
    assert c.get("/api/engine/live").json()["finish"]["player"] == status   # the screen sees where the video is
    assert c.post("/api/player/finish", json={"action": "pause"}).status_code == 200
    cmd = json.loads(post().headers["X-Player-Cmd"])
    assert cmd["action"] == "pause"
    # the camera applies a command once
    seen = []

    class P:
        def status(self): return status
        def control(self, cmd): seen.append(cmd["action"])
    pub = FramePublisher("http://app", "finish", lambda img, s: None, post=lambda b: None)
    pub.attach_player(P())
    pub._player_cmd(json.dumps(cmd)); pub._player_cmd(json.dumps(cmd))
    assert seen == ["pause"] and pub.keepalive_s == 0.25


def test_the_station_video_cannot_go_back_while_a_test_is_counted(tmp_path):
    app = create_app(db_url=f"sqlite:///{tmp_path / 'a.db'}", web_dir="does-not-exist")
    c = TestClient(app)
    report = lambda attempt: c.post("/api/engine/report", json={"station": "exercise", "player": {"t": 5.0, "paused": False},
                                                                "lanes": {"S": {"person": True, "attempt_id": attempt}}})
    report(None)
    assert c.post("/api/player/station", json={"action": "seek", "t": 1.0}).status_code == 200   # nobody counted: free
    report(7)
    r = c.post("/api/player/station", json={"action": "seek", "t": 1.0})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "no_back_in_test"   # its reps would count again
    assert c.post("/api/player/station", json={"action": "step", "n": -1}).status_code == 409
    assert c.post("/api/player/station", json={"action": "seek", "t": 8.0}).status_code == 200    # forward is fine
    assert c.post("/api/player/station", json={"action": "pause"}).status_code == 200


def test_a_page_can_ask_whether_its_stream_is_still_open():
    from fastapi.testclient import TestClient

    from testpoint.api.app import create_app

    with TestClient(create_app("sqlite://")) as c:
        assert c.get("/api/engine/stream-alive", params={"key": "123"}).json() == {"alive": False}
