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


def test_a_hand_raised_over_the_start_line_does_not_start():
    s = line()
    s.observe(0.0, Look(True, ME))
    s.observe(0.25, Look(True, ME))
    assert s.observe(0.5, Look(True, None, "right", over=True)) == []
    assert s.observe(1.2, Look(True, None, "right", over=True)) == [] and s.state == "raise_hand"
    assert s.view(1.2)["over"] is True                               # the screen says: behind the line
    s.observe(1.3, Look(True, None, "right"))                        # stepped back: the hand counts
    assert s.observe(1.9, Look(True, None, "right")) == [("start", 7)]


def test_over_the_start_line_before_go_is_a_false_start():
    s = line()
    s.observe(0.0, Look(True, ME))
    s.observe(0.25, Look(True, ME))
    s.observe(0.5, Look(True, None, "both"))
    assert s.observe(1.0, Look(True, None, "both")) == [("start", 7)]   # GO at 4.0
    assert s.observe(2.0, Look(True, None, None, over=True)) == [("cancel", 7)]
    assert s.state == "idle" and s.reason == "false_start"
    # over the line after GO is simply running
    s = line()
    s.observe(0.0, Look(True, ME)); s.observe(0.25, Look(True, ME))
    s.observe(0.5, Look(True, None, "both")); s.observe(1.0, Look(True, None, "both"))
    s.observe(2.5, Look(True)); s.observe(3.9, Look(True))
    assert s.observe(4.2, Look(True, None, None, over=True)) == [] and s.state == "started"


def test_with_a_start_line_the_clock_starts_when_the_runner_crosses_it():
    s = StartLine(countdown_s=0.0)
    lk = lambda **kw: Look(True, line=True, **kw)
    s.observe(0.0, lk(match=ME)); s.observe(0.25, lk(match=ME))
    s.observe(0.5, lk(hand="right"))
    assert s.observe(1.0, lk(hand="right")) == [("start", 7)]          # GO now
    assert s.observe(1.3, lk()) == []                                   # still behind the line
    assert s.observe(1.6, lk(over=True)) == [("crossed", 7)]            # the clock runs from here
    assert s.observe(1.8, lk(over=True)) == []                          # once
    # not seen crossing within CROSS_WAIT_S: the clock stays from GO
    s = StartLine(countdown_s=0.0)
    s.observe(0.0, lk(match=ME)); s.observe(0.25, lk(match=ME))
    s.observe(0.5, lk(hand="right")); s.observe(1.0, lk(hand="right"))
    acts = [a for t in range(2, 14) for a in s.observe(float(t), lk())]
    assert acts == [("cross_missed", 7)]
    # without a line, nothing is awaited
    s = StartLine(countdown_s=0.0)
    s.observe(0.0, Look(True, ME)); s.observe(0.25, Look(True, ME))
    s.observe(0.5, Look(True, None, "right")); s.observe(1.0, Look(True, None, "right"))
    assert s.observe(1.5, Look(True, over=True)) == [] and s.crossing is None


def test_the_start_line_picks_the_runner_at_it():
    from testpoint.vision.gesture import near_line, over_line

    def standing(feet_y):
        xy = np.zeros((17, 2), np.float32)
        xy[:, 1] = feet_y - 300
        xy[15:17, 1] = feet_y
        return Person(xy, np.full(17, 0.9, np.float32))

    h = 1000
    assert near_line(standing(820), h, 0.85) and not over_line(standing(820), h, 0.85)   # just behind
    assert type(over_line(standing(900), h, 0.85)) is bool       # sent as JSON by the camera
    assert over_line(standing(900), h, 0.85) and near_line(standing(900), h, 0.85)       # a foot over: still the runner
    assert not near_line(standing(500), h, 0.85)                                         # far back: someone waiting
    # running up the picture: behind is below the line
    assert near_line(standing(880), h, 0.85, "up") and over_line(standing(800), h, 0.85, "up")


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
    near = {5: (80, 100), 6: (120, 100), 9: (75, 180), 10: (125, 180)}       # at a desk: no hips in the picture
    assert hand_raised(person({**near, 10: (130, 40)})) == "right"


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
        c.put("/api/settings", json={"start_countdown_s": 3})   # with a 3-2-1 (the default is none)
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


def test_a_runner_started_by_the_camera_leaves_their_start_line_face_as_a_template(monkeypatch):
    """The finish camera sees faces like the start camera does (small, often infrared): the face
    seen at the start, once the runner raised the hand and was started, is kept."""
    import base64
    from datetime import timedelta

    from fastapi.testclient import TestClient
    from sqlalchemy.pool import StaticPool
    from test_faces import A, FakeEngine, vec

    from testpoint.api import app as app_mod
    from testpoint.api import models
    from testpoint.api.app import create_app

    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: "2026-10-20", face_engine=lambda: FakeEngine())
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    venue = base64.b64encode(np.asarray(vec(77, 0.8, A), np.float32).tobytes()).decode()   # A, less alike
    crop = base64.b64encode(b"\xff\xd8start").decode()
    with TestClient(app) as c:
        c.post("/api/entries/import", content=b"external_id,name,bib,gender,dob\nC-1,Alpha,11,M,2000-01-01\n")
        e = c.get("/api/entries").json()[0]["id"]
        cid = c.get(f"/api/entries/{e}").json()["candidate"]["id"]
        c.post(f"/api/entries/{e}/face", content=b"face:A")
        c.post(f"/api/entries/{e}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True})

        def look(dt, **kw):
            clock["t"] = t0 + timedelta(seconds=dt)
            return c.post("/api/engine/start/obs", json={"present": True, **kw}).json()

        look(0.0, emb=venue, quality=0.5, crop=crop)               # infrared, small: still kept
        assert look(0.25, emb=venue, quality=0.5, crop=crop)["state"] == "raise_hand"
        assert len(c.get(f"/api/candidates/{cid}/faces").json()["faces"]) == 1    # not before the start
        look(0.75, hand="right")
        look(1.3, hand="right")
        assert c.get(f"/api/entries/{e}").json()["status"] == "running"
        faces = c.get(f"/api/candidates/{cid}/faces").json()["faces"]
        assert len(faces) == 2
        assert c.get(f"/api/entries/{e}").json()["faces"] == 1     # the count: today's registration photos


def test_the_start_camera_keeps_several_pictures_taken_while_the_runner_waits(monkeypatch):
    import base64
    from datetime import timedelta

    from fastapi.testclient import TestClient
    from sqlalchemy.pool import StaticPool
    from test_faces import A, FakeEngine, vec

    from testpoint.api import app as app_mod
    from testpoint.api import models
    from testpoint.api.app import create_app

    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: "2026-10-20", face_engine=lambda: FakeEngine())
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    pic = lambda seed: base64.b64encode(np.asarray(vec(seed, 0.8, A), np.float32).tobytes()).decode()
    crop = base64.b64encode(b"\xff\xd8start").decode()
    with TestClient(app) as c:
        c.post("/api/entries/import", content=b"external_id,name,bib,gender,dob\nC-1,Alpha,11,M,2000-01-01\n")
        e = c.get("/api/entries").json()[0]["id"]
        cid = c.get(f"/api/entries/{e}").json()["candidate"]["id"]
        c.post(f"/api/entries/{e}/face", content=b"face:A")
        c.post(f"/api/entries/{e}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True})

        def look(dt, **kw):
            clock["t"] = t0 + timedelta(seconds=dt)
            return c.post("/api/engine/start/obs", json={"present": True, **kw}).json()

        look(0.0, emb=pic(1), quality=0.5, crop=crop)
        assert look(0.2, emb=pic(2), quality=0.5, crop=crop)["state"] == "raise_hand"
        for k, dt in enumerate((0.4, 0.6, 0.8, 1.0, 1.2, 1.4)):          # waiting, looking around
            look(dt, emb=pic(10 + k), quality=0.5, crop=crop)
        look(1.6, hand="right")
        look(2.2, hand="right")
        assert c.get(f"/api/entries/{e}").json()["status"] == "running"
        assert len(c.get(f"/api/candidates/{cid}/faces").json()["faces"]) == 1 + 5   # at most 5 kept


def test_the_start_line_sends_someone_without_todays_photos_back_to_the_desk(monkeypatch):
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
    day = {"d": "2026-10-20"}
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: day["d"], face_engine=lambda: FakeEngine())
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    emb = base64.b64encode(np.asarray(A, np.float32).tobytes()).decode()
    csv = b"external_id,name,bib,gender,dob\nC-1,Alpha,11,M,2000-01-01\n"
    with TestClient(app) as c:
        c.post("/api/entries/import", content=csv)
        c.post(f"/api/entries/{c.get('/api/entries').json()[0]['id']}/face", content=b"face:A")
        day["d"] = "2026-10-27"                                   # a week later: last week's photos only
        c.post("/api/entries/import", content=csv)
        e = c.get("/api/entries").json()[0]["id"]
        c.post(f"/api/entries/{e}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True})

        def look(dt, **kw):
            clock["t"] = t0 + timedelta(seconds=dt)
            return c.post("/api/engine/start/obs", json={"present": True, **kw}).json()

        look(0.0, emb=emb)
        v = look(0.25, emb=emb)
        assert v["state"] == "refused" and v["reason"] == "no_photos_today"
        c.post(f"/api/entries/{e}/face", content=b"face:A")      # today's photo taken at the desk
        for dt in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            look(dt)
        look(10.0, emb=emb)
        assert look(10.25, emb=emb)["state"] == "raise_hand"


def test_the_run_is_timed_from_the_start_line_crossing(monkeypatch):
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
            return c.post("/api/engine/start/obs", json={"present": True, "line": True, **kw}).json()

        look(0.0, emb=emb); look(0.25, emb=emb)
        look(0.5, hand="right"); look(1.0, hand="right")              # GO (no countdown)
        go = c.get(f"/api/entries/{e}").json()["run_started_at"]
        look(1.4); look(2.2, over=True)                                # crosses 1.2 s after GO
        started = c.get(f"/api/entries/{e}").json()["run_started_at"]
        assert started > go and started.startswith((t0 + timedelta(seconds=2.2)).isoformat()[:21])
