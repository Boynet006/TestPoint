"""The start line's camera logic: recognise, raise the hand, 3-2-1-GO, one runner at a time."""

import numpy as np

from testpoint.api.startline import Look, StartLine
from testpoint.vision.gesture import at_the_mark, hand_raised
from testpoint.vision.keypoints import Person


def line():
    return StartLine(countdown_s=3.0)


ME = (7, True, None)


def test_recognised_then_hand_then_countdown_then_go():
    s = line()
    assert s.observe(0.0, Look(True, ME)) == []                      # one look is not enough
    assert s.observe(0.25, Look(True, ME)) == [("line_up", 7)]       # 2 of 3: recognised, lined up
    assert s.state == "raise_hand"
    # either hand (or both) held 0.5 s: start, GO 3 s later
    assert s.observe(1.0, Look(True, None, "right")) == []
    assert s.observe(1.5, Look(True, None, "left")) == [("start", 7)]
    v = s.view(1.6)
    assert v["state"] == "countdown" and v["go_at"] == 4.5 and v["entry_id"] == 7
    s.observe(3.0, Look(True))
    s.observe(4.6, Look(True))
    assert s.state == "started"
    s.observe(6.0, Look(True))
    s.observe(7.5, Look(True))
    assert s.state == "idle"
    # the runner who just left is not recognised again meanwhile
    assert s.observe(7.6, Look(True, ME)) == [] and s.observe(7.7, Look(True, ME)) == []
    # someone else is
    other = (8, True, None)
    s.observe(7.8, Look(True, other))
    assert s.observe(7.9, Look(True, other)) == [("line_up", 8)]


def test_one_wrong_look_does_not_recognise_anyone():
    s = line()
    s.observe(0.0, Look(True, ME))
    s.observe(0.2, Look(True, (9, True, None)))
    assert s.state == "idle"
    s.observe(0.4, Look(True, None))
    assert s.observe(0.6, Look(True, ME)) == [] and s.state == "idle"   # votes: 9, None, 7


def test_refused_when_they_may_not_start():
    s = line()
    s.observe(0.0, Look(True, (7, False, "already_ran")))
    s.observe(0.2, Look(True, (7, False, "already_ran")))
    v = s.view(0.3)
    assert v["state"] == "refused" and v["reason"] == "already_ran"
    s.observe(4.3, Look(True))
    assert s.state == "idle"


def test_stepping_away_or_no_hand_goes_back_to_waiting():
    s = line()
    s.observe(0.0, Look(True, ME))
    s.observe(0.2, Look(True, ME))
    s.observe(1.0, Look(False))
    s.observe(2.3, Look(False))
    assert s.state == "idle" and s.reason == "stepped_away"
    s.observe(3.0, Look(True, ME))
    s.observe(3.2, Look(True, ME))
    for t in np.arange(3.4, 18.5, 0.5):
        s.observe(float(t), Look(True))
    assert s.state == "idle" and s.reason == "timeout"


def test_stepping_away_during_the_countdown_cancels_the_start():
    s = line()
    s.observe(0.0, Look(True, ME))
    s.observe(0.2, Look(True, ME))
    s.observe(0.4, Look(True, None, "right"))
    assert s.observe(0.9, Look(True, None, "right")) == [("start", 7)]
    s.observe(1.5, Look(False))
    assert s.observe(2.5, Look(False)) == [("cancel", 7)] and s.state == "idle"


def test_the_camera_going_quiet_resets():
    s = line()
    s.observe(0.0, Look(True, ME))
    s.observe(0.2, Look(True, ME))
    assert s.view(4.0)["state"] == "idle" and not s.view(4.0)["camera"]


def person(points):
    xy = np.zeros((17, 2), np.float32)
    conf = np.zeros(17, np.float32)
    for i, (x, y) in points.items():
        xy[i] = (x, y)
        conf[i] = 0.9
    return Person(xy=xy, conf=conf)


BODY = {0: (100, 50), 1: (95, 45), 2: (105, 45), 5: (80, 100), 6: (120, 100), 11: (85, 200), 12: (115, 200),
        9: (75, 180), 10: (125, 180), 15: (85, 380), 16: (115, 380)}


def test_hand_raised_is_the_persons_own_side():
    assert hand_raised(person(BODY)) is None                                  # arms down
    assert hand_raised(person({**BODY, 9: (70, 60)})) == "left"               # left wrist above the shoulder
    assert hand_raised(person({**BODY, 10: (130, 40)})) == "right"
    assert hand_raised(person({**BODY, 9: (70, 60), 10: (130, 40)})) == "both"
    assert hand_raised(person({**BODY, 9: (70, 95)})) is None                 # barely above: not raised


def test_the_person_at_the_mark_is_the_one_nearest():
    near = person(BODY)                                                       # 335 px tall
    far = person({i: (x / 3 + 400, y / 3 + 100) for i, (x, y) in BODY.items()})
    assert at_the_mark([far, near], frame_h=1080) is near
    assert at_the_mark([far], frame_h=1080) is None                           # too small: walking past behind


def test_the_start_camera_lines_up_and_starts_the_runner(monkeypatch):
    """Through the API: looks from the camera -> lined up -> raised hand -> running from GO."""
    import base64
    from datetime import timedelta

    from fastapi.testclient import TestClient
    from sqlalchemy.pool import StaticPool
    from test_faces import A, B, FakeEngine

    from testpoint.api import app as app_mod
    from testpoint.api import models
    from testpoint.api.app import create_app

    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: "2026-10-20", face_engine=lambda: FakeEngine())
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    emb = lambda v: base64.b64encode(np.asarray(v, np.float32).tobytes()).decode()
    with TestClient(app) as c:
        c.post("/api/entries/import", content=b"external_id,name,bib,gender,dob\nC-1,Alpha,11,M,2000-01-01\nC-2,Bravo,12,M,2000-01-01\n")
        ids = {e["candidate"]["bib"]: e["id"] for e in c.get("/api/entries").json()}
        c.post(f"/api/entries/{ids['11']}/face", content=b"face:A")
        c.post(f"/api/entries/{ids['12']}/face", content=b"face:B")
        c.post(f"/api/entries/{ids['11']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True})

        def look(dt, **kw):
            clock["t"] = t0 + timedelta(seconds=dt)
            r = c.post("/api/engine/start/obs", json={"present": True, **kw})
            assert r.status_code == 200, r.text
            return r.json()

        assert look(0.0, emb=emb(A))["state"] == "idle"
        v = look(0.25, emb=emb(A))
        assert v["state"] == "raise_hand" and v["entry"]["bib"] == "11"
        assert c.get(f"/api/entries/{ids['11']}").json()["status"] == "lined_up"
        look(0.75, hand="both")
        v = look(1.3, hand="right")
        assert v["state"] == "countdown" and v["go_at"].startswith((t0 + timedelta(seconds=4.3)).isoformat()[:19])
        e = c.get(f"/api/entries/{ids['11']}").json()
        assert e["status"] == "running" and e["run_started_at"] == v["go_at"]
        assert c.get("/api/startline").json()["state"] == "countdown"
        # Bravo is not weighed in: recognised, but refused with the reason
        for dt in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
            look(dt)
        look(8.2, emb=emb(B))
        v = look(8.4, emb=emb(B))
        assert v["state"] == "refused" and v["reason"] == "not_weighed" and v["entry"]["bib"] == "12"


def test_the_examiner_starting_the_called_runner_shows_their_countdown_not_a_refusal(monkeypatch):
    import base64
    from datetime import timedelta

    from fastapi.testclient import TestClient
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

        def look(dt, **kw):
            clock["t"] = t0 + timedelta(seconds=dt)
            return c.post("/api/engine/start/obs", json={"present": True, **kw}).json()

        look(0.0, emb=emb)
        assert look(0.25, emb=emb)["state"] == "raise_hand"
        clock["t"] = t0 + timedelta(seconds=1)
        assert c.post(f"/api/entries/{e}/start", json={"countdown_s": 3}).status_code == 200   # the examiner presses Start
        v = look(1.25, emb=emb)
        assert v["state"] == "started" and v["entry_id"] == e and v["go_at"]   # their 3-2-1, not "still running"
        for dt in (2.0, 4.0, 6.0, 9.0, 12.0, 20.0):                          # still in front of the camera after GO
            assert look(dt, emb=emb)["state"] in ("started", "idle")
