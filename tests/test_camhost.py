"""Camera service: the finish page picks a source, the service starts the finish camera."""

from __future__ import annotations

import threading

import pytest
import uvicorn
from fastapi.testclient import TestClient

from testpoint import camhost


class FakeProc:
    def __init__(self, cmd, **kw):
        self.cmd, self.kw, self.code = cmd, kw, None
        FakeProc.last = self

    def poll(self):
        return self.code

    def terminate(self):
        self.code = 0

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    vids = tmp_path / "run"
    vids.mkdir()
    (vids / "finish.mp4").write_bytes(b"x")
    monkeypatch.setenv("TESTPOINT_VIDEO_DIRS", str(vids))
    monkeypatch.setattr(camhost, "list_cameras", lambda max_index=4: [{"index": 0, "label": "Camera 1", "size": [1280, 720]}])
    return TestClient(camhost.create_service(log_dir=tmp_path / "logs", runner=FakeProc)), vids


def test_sources_and_start_stop(svc):
    c, vids = svc
    src = c.get("/sources").json()
    assert src["cameras"][0]["index"] == 0 and src["videos"][0]["label"] == "run/finish.mp4"
    video = src["videos"][0]["path"]
    r = c.post("/finish/start", json={"kind": "video", "source": video, "line_y": 0.7, "direction": "down",
                                       "app_url": "http://127.0.0.1:8080"})
    assert r.status_code == 200 and r.json()["running"]
    cmd = FakeProc.last.cmd
    assert cmd[1:3] == ["-m", "testpoint.finishcam"] and "--realtime" in cmd
    assert cmd[cmd.index("--line") + 1] == "0,0.7,1,0.7" and cmd[cmd.index("--app") + 1] == "http://127.0.0.1:8080"
    first = FakeProc.last
    # a webcam replaces it (one finish camera at a time)
    assert c.post("/finish/start", json={"kind": "webcam", "source": "0", "app_url": "http://localhost:8001"}).json()["running"]
    assert first.code == 0 and "--realtime" not in FakeProc.last.cmd
    assert c.post("/finish/stop").json()["running"] is False


def test_refuses_what_it_should_not_run(svc):
    c, _ = svc
    app = "http://127.0.0.1:8080"
    assert c.post("/finish/start", json={"kind": "video", "source": "C:/Windows/notepad.exe", "app_url": app}).status_code == 400
    assert c.post("/finish/start", json={"kind": "webcam", "source": "0; rm", "app_url": app}).status_code == 400
    assert c.post("/finish/start", json={"kind": "url", "source": "file:///etc/passwd", "app_url": app}).status_code == 400
    assert c.post("/finish/start", json={"kind": "webcam", "source": "0", "app_url": "http://evil.example"}).status_code == 400
    assert c.post("/finish/start", json={"kind": "url", "source": "rtsp://192.168.1.64:554/Streaming/Channels/101",
                                          "app_url": app}).json()["running"]


def test_app_reports_an_offline_camera_service(tmp_path):
    from testpoint.api.app import create_app
    app = create_app(db_url=f"sqlite:///{tmp_path / 'a.db'}", web_dir="none", camhost_url="http://127.0.0.1:9")
    with TestClient(app) as c:
        assert c.get("/api/finishcam").json() == {"available": False, "running": False, "reason": "camera_service_offline"}
        r = c.post("/api/finishcam/start", json={"kind": "webcam", "source": "0", "app_url": "http://127.0.0.1:8080"})
        assert r.status_code == 503 and r.json()["detail"]["code"] == "camera_service_offline"


def test_app_talks_to_the_camera_service(svc, tmp_path):
    from testpoint.api.app import create_app
    service = camhost.create_service(log_dir=tmp_path / "logs2", runner=FakeProc)
    server = uvicorn.Server(uvicorn.Config(service, host="127.0.0.1", port=8197, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        import time
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        app = create_app(db_url=f"sqlite:///{tmp_path / 'b.db'}", web_dir="none", camhost_url="http://127.0.0.1:8197")
        with TestClient(app) as c:
            assert c.get("/api/finishcam").json()["available"] is True
            assert c.get("/api/finishcam/sources").json()["cameras"][0]["index"] == 0
            r = c.post("/api/finishcam/start", json={"kind": "webcam", "source": "0", "app_url": "http://127.0.0.1:8080"})
            assert r.status_code == 200 and r.json()["running"]
            bad = c.post("/api/finishcam/start", json={"kind": "url", "source": "ftp://x", "app_url": "http://127.0.0.1:8080"})
            assert bad.status_code == 400 and bad.json()["detail"]["code"] == "camera_refused"
            assert c.post("/api/finishcam/stop").json()["running"] is False
    finally:
        server.should_exit = True
        th.join(timeout=5)
