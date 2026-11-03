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
    (tmp_path / "pushup").mkdir()
    (tmp_path / "pushup" / "lanes.mp4").write_bytes(b"xy")
    monkeypatch.setenv("TESTPOINT_VIDEO_DIRS", str(tmp_path))
    monkeypatch.setattr(camhost, "list_cameras", lambda max_index=4: [{"index": 0, "label": "Camera 1", "size": [1280, 720]}])
    return TestClient(camhost.create_service(log_dir=tmp_path / "logs", runner=FakeProc)), vids


def test_sources_and_start_stop(svc):
    c, vids = svc
    src = c.get("/sources").json()
    assert src["cameras"][0]["index"] == 0
    labels = {v["label"]: v for v in src["videos"]}
    assert set(labels) == {"run/finish.mp4", "pushup/lanes.mp4"} and labels["pushup/lanes.mp4"]["group"] == "pushup"
    video = labels["run/finish.mp4"]["path"]
    r = c.post("/finish/start", json={"kind": "video", "source": video, "line_y": 0.7, "direction": "down",
                                       "app_url": "http://127.0.0.1:8080"})
    assert r.status_code == 200 and r.json()["running"]
    cmd = FakeProc.last.cmd
    assert cmd[1:3] == ["-m", "testpoint.finishcam"] and "--realtime" in cmd
    assert cmd[cmd.index("--line") + 1] == "0,0.7,1,0.7" and cmd[cmd.index("--app") + 1] == "http://127.0.0.1:8080"
    first = FakeProc.last
    # asked again with the same settings: left running, not restarted
    since = c.get("/finish").json()["since"]
    again = c.post("/finish/start", json={"kind": "video", "source": video, "line_y": 0.7, "direction": "down",
                                           "app_url": "http://127.0.0.1:8080"}).json()
    assert FakeProc.last is first and again["since"] == since
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



def test_station_engine_with_lanes_drawn_on_the_picture(svc, tmp_path):
    c, _ = svc
    video = next(v["path"] for v in c.get("/sources").json()["videos"] if v["group"] == "pushup")
    r = c.post("/station/start", json={"kind": "video", "source": video, "exercise": "pushup", "lanes": 3,
                                        "dividers": [0.3, 0.62], "app_url": "http://127.0.0.1:8080"}).json()
    assert r["running"] and [x["rect"] for x in r["config"]["rects"]] == [[0.0, 0.0, 0.3, 1.0], [0.3, 0.0, 0.62, 1.0], [0.62, 0.0, 1.0, 1.0]]
    cmd = FakeProc.last.cmd
    assert cmd[1:3] == ["-m", "testpoint.engine"] and cmd[cmd.index("--exercise") + 1] == "pushup" and "--realtime" in cmd
    import json, pathlib
    lanes = json.loads(pathlib.Path(cmd[cmd.index("--lanes-file") + 1]).read_text())["lanes"]
    assert [x["name"] for x in lanes] == ["1", "2", "3"]
    # equal strips when no dividers are given; bad dividers refused
    r = c.post("/station/start", json={"kind": "webcam", "source": "0", "exercise": "crunch", "lanes": 2, "app_url": "http://127.0.0.1:8080"}).json()
    assert [x["rect"][2] for x in r["config"]["rects"]] == [0.5, 1.0]
    bad = c.post("/station/start", json={"kind": "webcam", "source": "0", "exercise": "crunch", "lanes": 3,
                                          "dividers": [0.6, 0.3], "app_url": "http://127.0.0.1:8080"})
    assert bad.status_code == 400
    # the finish camera is a separate process: both can run (on different cameras)
    assert c.get("/finish").json()["running"] is False and c.get("/station").json()["running"]
    assert c.post("/station/stop").json()["running"] is False


def test_camera_processes_reach_the_app_over_ipv4():
    from testpoint.camhost import app_address
    assert app_address("http://localhost:8080") == "http://127.0.0.1:8080"
    assert app_address("http://localhost") == "http://127.0.0.1"
    assert app_address("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert app_address("http://localhostile:8080") == "http://localhostile:8080"


def test_log_lines_are_kept_short():
    from testpoint.camhost import clean
    assert clean("\x1b[31mok\x1b[0m") == "ok"
    assert clean('"/9j/' + "A" * 5000 + '",') == ""                 # picture data: dropped
    long = clean("word " * 200)
    assert len(long) < 310 and long.endswith("…")


def test_video_on_repeat_is_passed_to_the_camera():
    from pathlib import Path

    from testpoint.camhost import StartIn, StationIn, finish_command, station_command
    f = finish_command(StartIn(kind="video", source="clip.mp4", app_url="http://127.0.0.1:8080", loop=True))
    assert "--realtime" in f and "--loop" in f
    s = station_command(StationIn(kind="video", source="clip.mp4", app_url="http://127.0.0.1:8080", exercise="pushup"), Path("l.json"))
    assert "--realtime" in s and "--loop" not in s
    w = finish_command(StartIn(kind="webcam", source="0", app_url="http://127.0.0.1:8080", loop=True))
    assert "--loop" not in w



def test_station_tuning_goes_to_the_engine():
    from pathlib import Path

    from testpoint.camhost import StationIn, station_command
    cmd = station_command(StationIn(kind="webcam", source="0", app_url="http://127.0.0.1:8080", exercise="pushup",
                                    tuning={"down_elbow_deg": 90}), Path("l.json"))
    assert cmd[cmd.index("--tuning") + 1] == '{"down_elbow_deg": 90.0}'


def test_start_line_ip_camera(svc):
    c, _ = svc
    url = "rtsp://admin:pw@192.168.250.241:554/Streaming/Channels/101"
    r = c.post("/startcam/start", json={"kind": "url", "source": url, "app_url": "http://localhost:8000"}).json()
    assert r["running"] and r["config"]["source"] == url
    cmd = FakeProc.last.cmd
    assert cmd[1:3] == ["-m", "testpoint.startcam"] and cmd[cmd.index("--source") + 1] == url
    assert cmd[cmd.index("--app") + 1] == "http://127.0.0.1:8000" and "--realtime" not in cmd
    # its own process: the finish camera and the station keep running beside it
    assert c.get("/finish").json()["running"] is False and c.get("/startcam").json()["running"]
    assert c.post("/startcam/start", json={"kind": "url", "source": "file:///etc/passwd",
                                           "app_url": "http://127.0.0.1:8000"}).status_code == 400
    assert c.post("/startcam/stop").json()["running"] is False


def test_app_reports_an_offline_start_camera_service(tmp_path):
    from testpoint.api.app import create_app
    app = create_app(db_url=f"sqlite:///{tmp_path / 'c.db'}", web_dir="none", camhost_url="http://127.0.0.1:9")
    with TestClient(app) as c:
        assert c.get("/api/startcam").json() == {"available": False, "running": False, "reason": "camera_service_offline"}
        r = c.post("/api/startcam/start", json={"kind": "url", "source": "rtsp://x", "app_url": "http://127.0.0.1:8000"})
        assert r.status_code == 503


def test_camera_processes_skip_the_certificate_check_only_on_this_pc():
    import ssl

    from testpoint.applink import ssl_context
    assert ssl_context("http://127.0.0.1:8000/api/engine/report") is None
    assert ssl_context("https://192.168.250.149:8000/api") is None          # another address: verified
    for url in ("https://127.0.0.1:8000/api", "https://localhost:8443/x"):
        ctx = ssl_context(url)
        assert ctx is not None and ctx.verify_mode == ssl.CERT_NONE and not ctx.check_hostname


def test_https_app_address_is_accepted():
    from testpoint.camhost import APP_URL
    assert APP_URL.match("https://127.0.0.1:8000") and APP_URL.match("http://localhost:8080")
    assert not APP_URL.match("https://192.168.250.149:8000")


def test_the_app_tells_cameras_its_own_loopback_address(svc, tmp_path, monkeypatch):
    """A page opened on another computer sends that computer's view of the address; the camera
    processes on the station PC get the app's loopback address with the request's scheme and port."""
    from testpoint.api.app import create_app
    monkeypatch.setenv("TESTPOINT_CERT_DIR", str(tmp_path / "no-certs"))
    service = camhost.create_service(log_dir=tmp_path / "logs3", runner=FakeProc)
    server = uvicorn.Server(uvicorn.Config(service, host="127.0.0.1", port=8198, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        import time
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        app = create_app(db_url=f"sqlite:///{tmp_path / 'd.db'}", web_dir="none", camhost_url="http://127.0.0.1:8198")
        with TestClient(app, base_url="https://192.168.250.149:8000") as c:
            r = c.post("/api/startcam/start", json={"kind": "url", "source": "rtsp://cam/101",
                                                    "app_url": "https://192.168.250.149:8000"})
            assert r.status_code == 200
            cmd = FakeProc.last.cmd
            assert cmd[cmd.index("--app") + 1] == "https://127.0.0.1:8000"
            assert c.get("/testpoint-ca.crt").status_code == 404           # no certificate made here
    finally:
        server.should_exit = True
        th.join(timeout=5)


def test_the_combined_station_runs_one_engine_for_both_exercises(svc):
    c, _ = svc
    r = c.post("/station/start", json={"kind": "url", "source": "rtsp://cam/101", "exercise": "auto",
                                       "app_url": "http://127.0.0.1:8000"}).json()
    assert r["running"] and r["config"]["rects"] == [{"name": "S", "rect": [0.0, 0.0, 1.0, 1.0]}]
    cmd = FakeProc.last.cmd
    assert cmd[cmd.index("--station") + 1] == "exercise" and cmd[cmd.index("--exercise") + 1] == "pushup"


def test_camera_addresses_saved_in_settings_never_show_the_password(svc, tmp_path, monkeypatch):
    from testpoint.api.app import create_app
    monkeypatch.setenv("TESTPOINT_CERT_DIR", str(tmp_path / "no-certs"))
    service = camhost.create_service(log_dir=tmp_path / "logs4", runner=FakeProc)
    server = uvicorn.Server(uvicorn.Config(service, host="127.0.0.1", port=8196, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        import time
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        app = create_app(db_url=f"sqlite:///{tmp_path / 'e.db'}", web_dir="none", camhost_url="http://127.0.0.1:8196")
        with TestClient(app) as c:
            url = "rtsp://admin:example-pw@192.168.250.241:554/Streaming/Channels/101"
            assert c.get("/api/settings/cameras").json()["start"] == {"set": False, "address": None}
            r = c.put("/api/settings/cameras", json={"start": url}).json()
            assert r["start"] == {"set": True, "address": "rtsp://admin:***@192.168.250.241:554/Streaming/Channels/101"}
            assert "example-pw" not in c.get("/api/settings").text and "example-pw" not in c.get("/api/settings/cameras").text
            assert c.put("/api/settings/cameras", json={"finish": "ftp://x"}).status_code == 400
            # the station camera too (its name must not clash with anything in the audit record)
            assert c.put("/api/settings/cameras", json={"station": url.replace(".241", ".243")}).json()["station"]["set"] is True
            # "saved": the server starts the real address; the screen gets it without the password
            st = c.post("/api/startcam/start", json={"kind": "saved", "app_url": "http://127.0.0.1:8000"}).json()
            assert st["running"] and "example-pw" not in str(st) and "***" in st["config"]["source"]
            assert FakeProc.last.cmd[FakeProc.last.cmd.index("--source") + 1] == url
            assert "example-pw" not in c.get("/api/startcam").text
            # a camera picked by name from the dropdown: the finish camera, its address put in on the server
            c.put("/api/settings/cameras", json={"finish": url.replace(".241", ".242")})
            assert c.post("/api/finishcam/start", json={"kind": "saved", "source": "saved:finish", "app_url": "x"}).json()["running"]
            assert ".242" in FakeProc.last.cmd[FakeProc.last.cmd.index("--source") + 1]
            assert c.post("/api/startcam/start", json={"kind": "saved", "source": "saved:kitchen", "app_url": "x"}).status_code == 400
            # not saved (removed again): said so
            c.put("/api/settings/cameras", json={"finish": ""})
            r = c.post("/api/finishcam/start", json={"kind": "saved", "app_url": "http://127.0.0.1:8000"})
            assert r.status_code == 409 and r.json()["detail"]["code"] == "no_saved_camera"
            # an empty address removes it; the admin PIN guards the change
            assert c.put("/api/settings/cameras", json={"start": ""}).json()["start"]["set"] is False
            c.put("/api/settings/admin-pin", json={"pin": "2468"})
            assert c.put("/api/settings/cameras", json={"start": url}).status_code == 403
    finally:
        server.should_exit = True
        th.join(timeout=5)



def test_weigh_in_ip_camera_only_relays_its_picture():
    from testpoint.camhost import SourceIn, weigh_command
    cmd = weigh_command(SourceIn(kind="url", source="rtsp://u:p@10.0.0.5/s", app_url="http://127.0.0.1:8080"))
    assert cmd[1:3] == ["-m", "testpoint.startcam"]
    assert cmd[cmd.index("--station") + 1] == "weigh" and "--no-analyse" in cmd and cmd[cmd.index("--evidence") + 1] == ""


def test_one_webcam_serves_one_camera_program(svc):
    c, _ = svc
    assert c.post("/startcam/start", json={"kind": "webcam", "source": "0", "app_url": "http://127.0.0.1:8080"}).json()["running"]
    r = c.post("/station/start", json={"kind": "webcam", "source": "0", "exercise": "auto", "lanes": 1, "app_url": "http://127.0.0.1:8080"}).json()
    assert r["running"] and r["freed"] == ["start"]           # the start camera let go of the webcam
    assert c.get("/startcam").json()["running"] is False and c.get("/startcam").json()["freed_by"] == "station"
    r = c.post("/finish/start", json={"kind": "webcam", "source": "1", "line_y": 0.8, "direction": "down", "app_url": "http://127.0.0.1:8080"}).json()
    assert "freed" not in r and c.get("/station").json()["running"]   # another webcam: both run
