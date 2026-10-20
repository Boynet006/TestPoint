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
