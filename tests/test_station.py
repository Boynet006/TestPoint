"""The exercise station run by its camera: recognise, get into position, 3-2-1, count, done."""

from testpoint.api.station import StationFlow, StationLook, next_event


def flow():
    return StationFlow(countdown_s=3.0, time_s=120.0)


def recognise(f, t, who=(7, "pushup", None)):
    f.observe(t, StationLook(True, who))
    return f.observe(t + 0.3, StationLook(True, who))


def test_the_next_test_is_chosen_for_them():
    assert next_event(set()) == "pushup"
    assert next_event({"pushup"}) == "crunch"
    assert next_event({"pushup", "crunch"}) is None


def test_recognised_position_countdown_testing_done():
    f = flow()
    assert recognise(f, 0.0) == []
    assert f.state == "position" and f.watching == "pushup"
    f.observe(1.0, StationLook(True, ready=True))
    f.observe(1.5, StationLook(True, ready=False))       # broke the posture: start again
    f.observe(2.0, StationLook(True, ready=True))
    f.observe(3.1, StationLook(True, ready=True))
    assert f.state == "countdown" and f.go_at == 6.1
    f.observe(5.0, StationLook(True))
    acts = f.observe(6.2, StationLook(True))
    assert acts == [("start", 7, "pushup")] and f.state == "testing"
    f.started(55)
    for t in range(7, 126, 2):
        assert f.observe(float(t), StationLook(True)) == []
    acts = f.observe(126.2, StationLook(True))
    assert acts == [("finish", 55)] and f.state == "done"
    v = f.view(127.0)
    assert v["attempt_id"] == 55 and v["ends_at"] == 126.1
    f.observe(130.0, StationLook(True))
    f.observe(134.5, StationLook(True))
    assert f.state == "idle"
    # the person just tested is not picked up again at once; someone else is
    assert recognise(f, 135.0) == [] and f.state == "idle"
    recognise(f, 136.0, (8, "crunch", None))
    assert f.state == "position" and f.watching == "crunch"


def test_refused_when_nothing_to_do_here():
    f = flow()
    recognise(f, 0.0, (9, None, "has_not_run"))
    v = f.view(0.5)
    assert v["state"] == "refused" and v["reason"] == "has_not_run"
    f.observe(6.0, StationLook(True))
    assert f.state == "idle"


def test_walking_away_before_go():
    f = flow()
    recognise(f, 0.0)
    f.observe(2.0, StationLook(False))
    f.observe(5.5, StationLook(False))
    assert f.state == "idle" and f.reason == "stepped_away"
    # during the countdown: back to getting into position
    recognise(f, 6.0)
    f.observe(7.0, StationLook(True, ready=True))
    f.observe(8.1, StationLook(True, ready=True))
    assert f.state == "countdown"
    f.observe(9.0, StationLook(False))
    f.observe(10.5, StationLook(False))
    assert f.state == "position" and f.reason == "left_position"


def test_the_engine_watches_the_test_being_done():
    f = flow()
    assert f.watching is None
    recognise(f, 0.0, (7, "crunch", None))
    assert f.watching == "crunch"


def test_the_station_camera_runs_pushups_then_crunches(monkeypatch):
    """Through the API: face -> push-ups -> start posture -> 3-2-1 -> counted -> saved at the time;
    back later -> crunches."""
    import base64
    from datetime import timedelta

    import numpy as np
    from fastapi.testclient import TestClient
    from runhelp import start_lined_up
    from sqlalchemy.pool import StaticPool
    from test_faces import A, FakeEngine

    from testpoint.api import app as app_mod
    from testpoint.api import models
    from testpoint.api.app import create_app

    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: "2026-10-20", face_engine=lambda: FakeEngine())
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    emb = base64.b64encode(np.asarray(A, np.float32).tobytes()).decode()
    with TestClient(app) as c:
        c.post("/api/entries/import", content=b"external_id,name,bib,gender,dob\nC-1,Alpha,11,M,2000-01-01\n")
        e = c.get("/api/entries").json()[0]["id"]
        c.post(f"/api/entries/{e}/face", content=b"face:A")
        c.post(f"/api/entries/{e}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True})
        c.post(f"/api/entries/{e}/line-up")
        start_lined_up(c)
        clock["t"] = t0 + timedelta(minutes=14)
        c.post("/api/run/finish", json={"bib": "11"})

        def report(dt, **lane):
            clock["t"] = t0 + timedelta(minutes=20, seconds=dt)
            r = c.post("/api/engine/report", json={"station": "exercise", "lanes": {"S": {"person": True, **lane}}})
            assert r.status_code == 200, r.text
            return r.json()

        report(0.0, face=emb)
        r = report(0.3, face=emb)
        assert r["lanes"]["S"] == {"attempt_id": None, "exercise": "pushup"}
        v = c.get("/api/stationflow").json()
        assert v["state"] == "position" and v["event"] == "pushup" and v["entry"]["bib"] == "11"
        report(1.0, exercise="pushup", ready=True)
        report(2.1, exercise="pushup", ready=True)
        assert c.get("/api/stationflow").json()["state"] == "countdown"
        report(4.0, exercise="pushup")
        r = report(5.2, exercise="pushup")                     # GO at 5.1: the attempt starts
        att = r["lanes"]["S"]["attempt_id"]
        assert att and c.get("/api/stationflow").json()["state"] == "testing"
        report(60.0, exercise="pushup", attempt_id=att, count=31,
               events=[{"index": 1, "valid": True, "t_start": 1, "t_end": 2}])
        # time up (120 s after GO): saved with the camera's count, even with no report then
        clock["t"] = t0 + timedelta(minutes=20, seconds=125.2)
        v = c.get("/api/stationflow").json()
        assert v["state"] == "done" and v["result"]["count"] == 31 and v["result"]["status"] == "finished"
        assert c.get(f"/api/entries/{e}").json()["events"]["pushup"]["count"] == 31
        # back after a break: crunches this time
        clock["t"] = t0 + timedelta(minutes=30)
        report(600.0, face=emb)
        r = report(600.3, face=emb)
        assert r["lanes"]["S"]["exercise"] == "crunch"
        assert c.get("/api/stationflow").json()["event"] == "crunch"


def test_the_examiner_can_name_the_person_and_start_without_the_posture():
    f = StationFlow(countdown_s=3.0, time_s=120.0)
    f.manual(100.0, 7, "crunch")
    assert (f.state, f.entry_id, f.event) == ("position", 7, "crunch")
    f.go_now(101.0)
    assert f.state == "countdown" and f.go_at == 104.0 and f.entry_id == 7
    assert f.tick(104.0) == [("start", 7, "crunch")] and f.state == "testing"
