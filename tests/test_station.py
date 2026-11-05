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
    assert acts == [("finish", 55, None)] and f.state == "done"
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
        c.put("/api/settings", json={"start_countdown_s": 3})   # with a 3-2-1 (the default is none)
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
        v = c.get("/api/stationflow").json()
        # both tests to do: they choose with a raised hand; held on push-ups (the left) 1.5 s picks it
        assert v["state"] == "choose" and v["choices"] == ["pushup", "crunch"] and v["entry"]["bib"] == "11"
        report(0.6, hand=[0.2, 0.5])
        v = c.get("/api/stationflow").json()
        assert v["hover"] == "pushup" and v["hand"] == [0.2, 0.5]
        r = report(2.2, hand=[0.2, 0.5])
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


def under_way(f, attempt=55):
    recognise(f, 0.0)
    f.go_now(1.0)
    f.observe(4.1, StationLook(True))
    f.started(attempt)
    assert f.state == "testing"


def test_the_examiner_ends_the_test_early_and_the_count_stands():
    f = flow()
    under_way(f)
    assert f.end_now(30.0) == [("finish", 55, "examiner")]
    assert f.state == "done" and f.view(30.5)["ended"] == "examiner"
    assert f.end_now(31.0) == []                      # nothing left to end


def test_the_test_ends_when_they_leave_the_mat():
    f = flow()
    under_way(f)
    f.observe(10.0, StationLook(False, count=3))
    assert f.observe(14.0, StationLook(False, count=3)) == []      # 4 s away: still on
    f.observe(14.5, StationLook(True, count=3))                    # back in time
    f.observe(20.0, StationLook(False, count=3))
    assert f.observe(25.1, StationLook(False, count=3)) == [("finish", 55, "left_mat")]


def test_the_test_ends_when_no_rep_comes_and_a_pause_does_not_count():
    f = StationFlow(countdown_s=3.0, time_s=120.0, idle_s=20.0)
    under_way(f)
    f.observe(10.0, StationLook(True, count=1))
    assert f.observe(29.0, StationLook(True, count=1)) == []
    f.observe(29.5, StationLook(True, count=2))                     # a rep: the 20 s start again
    f.hold(45.0)                                                    # a test video was paused
    assert f.observe(60.0, StationLook(True, count=2)) == []
    assert f.observe(65.1, StationLook(True, count=2)) == [("finish", 55, "no_reps")]
    off = flow()                                                    # idle_s 0: only the time ends it
    under_way(off)
    assert off.observe(100.0, StationLook(True, count=0)) == []



def test_no_rep_yet_is_not_stopping():
    f = StationFlow(countdown_s=3.0, time_s=120.0, idle_s=20.0)
    under_way(f)
    assert f.observe(60.0, StationLook(True, count=0)) == []        # getting down on the mat: still on
    f.observe(61.0, StationLook(True, count=1))
    assert f.observe(81.5, StationLook(True, count=1)) == [("finish", 55, "no_reps")]



def test_standing_up_during_the_test_ends_it_with_the_count_so_far():
    f = StationFlow(countdown_s=3.0, time_s=120.0, idle_s=20.0, out_s=3.0)
    under_way(f)
    f.observe(30.0, StationLook(True, count=20))
    # a slip out of position (the pose's jitter, a wobble) and back: the test goes on
    assert f.observe(31.0, StationLook(True, count=20, out=True)) == []
    assert f.view(32.0)["out_left"] == 2.0                          # the screen warns
    assert f.observe(32.5, StationLook(True, count=20)) == []
    assert f.view(32.6)["out_left"] is None
    # stood up and stayed up: over after out_s, with the 20 reps
    f.observe(40.0, StationLook(True, count=20, out=True))
    assert f.observe(42.0, StationLook(True, count=20, out=True)) == []
    assert f.observe(43.0, StationLook(True, count=20, out=True)) == [("finish", 55, "left_position")]
    assert f.state == "done" and f.view(43.5)["ended"] == "left_position"
    # off (0): only the other rules end it
    off = StationFlow(countdown_s=3.0, time_s=120.0, out_s=0.0)
    under_way(off)
    assert off.observe(20.0, StationLook(True, count=5, out=True)) == []
    assert off.observe(40.0, StationLook(True, count=5, out=True)) == []


def test_a_face_seen_but_not_recognised_is_said_on_the_screen():
    f = StationFlow()
    assert f.view(0.0)["reason"] is None
    f.unknown_face(10.0)
    assert f.view(11.0)["reason"] == "face_not_recognised"
    assert f.view(15.0)["reason"] is None                            # for a few seconds only


def test_resting_in_the_start_position_is_not_stopping():
    f = StationFlow(countdown_s=3.0, time_s=120.0, idle_s=20.0)
    under_way(f)
    f.observe(10.0, StationLook(True, count=3))
    for t in range(12, 60, 2):                                       # arms straight, getting their breath
        assert f.observe(float(t), StationLook(True, ready=True, count=3)) == []
    assert f.observe(80.5, StationLook(True, ready=False, count=3)) == [("finish", 55, "no_reps")]


def test_choosing_the_first_test_with_a_raised_hand():
    from testpoint.api.station import CHOOSE_DWELL_S, CHOOSE_TIMEOUT_S
    me = (7, "choose", None)
    f = StationFlow()
    f.observe(0.0, StationLook(True, me)); f.observe(0.25, StationLook(True, me))
    assert f.state == "choose" and f.watching is None
    f.observe(0.5, StationLook(True, hand=(0.8, 0.4)))          # on crunches
    f.observe(1.0, StationLook(True, hand=(0.5, 0.4)))          # the gap in the middle: starts again
    f.observe(1.2, StationLook(True, hand=(0.8, 0.4)))
    assert f.view(1.2)["hover"] == "crunch" and f.state == "choose"
    f.observe(1.2 + CHOOSE_DWELL_S, StationLook(True, hand=(0.85, 0.5)))
    assert f.state == "position" and f.event == "crunch"
    # no choice in time: push-ups, as without choosing
    g = StationFlow()
    g.observe(0.0, StationLook(True, me)); g.observe(0.25, StationLook(True, me))
    for k in range(1, int(CHOOSE_TIMEOUT_S) + 2):
        g.observe(0.25 + k, StationLook(True))
    assert g.state == "position" and g.event == "pushup"
    # the examiner taps it; stepping away goes back to waiting
    h = StationFlow()
    h.observe(0.0, StationLook(True, me)); h.observe(0.25, StationLook(True, me))
    assert h.choose(0.5, "crunch") and h.event == "crunch" and not h.choose(0.6, "pushup")
    k = StationFlow()
    k.observe(0.0, StationLook(True, me)); k.observe(0.25, StationLook(True, me))
    k.observe(6.0, StationLook(False))
    assert k.state == "idle"


def test_a_raised_hand_points_as_in_a_mirror():
    import numpy as np

    from testpoint.vision.gesture import pointer
    from testpoint.vision.keypoints import Person

    def person(points):
        xy, conf = np.zeros((17, 2), np.float32), np.zeros(17, np.float32)
        for i, (x, y) in points.items():
            xy[i], conf[i] = (x, y), 0.9
        return Person(xy=xy, conf=conf)
    body = {5: (120, 100), 6: (80, 100), 11: (115, 200), 12: (85, 200)}   # facing the camera: their left is image right
    arms_down = person({**body, 9: (125, 230), 10: (75, 230)})
    assert pointer(arms_down, (200, 300)) is None
    x, y = pointer(person({**body, 9: (125, 230), 10: (20, 60)}), (200, 300))           # right hand up and out to their right
    assert x > 0.6 and y < 0.5
    x, _ = pointer(person({**body, 9: (180, 60), 10: (75, 230)}), (200, 300))           # left hand out to their left
    assert x < 0.4
    assert pointer(person({**body, 9: (125, 230), 10: (95, 70)}), (200, 300)) is None   # a hand at the chin: not pointing
    near = {5: (120, 100), 6: (80, 100)}                                     # hips out of the picture: the shoulders give the scale
    x, y = pointer(person({**near, 10: (30, 70)}), (200, 300))
    assert x > 0.6 and y < 0.6


def test_a_test_left_running_when_the_app_restarts_goes_on_or_ends(monkeypatch):
    """The app restarted during a camera-run test: it carries on, or (its time up, the count may miss
    reps) is cancelled to be done again. Left running, nobody could finish it and the person could
    never be tested again."""
    from datetime import timedelta

    from fastapi.testclient import TestClient
    from runhelp import start_lined_up
    from sqlalchemy.pool import StaticPool
    from test_faces import FakeEngine

    from testpoint.api import app as app_mod
    from testpoint.api import models
    from testpoint.api.app import create_app

    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    for later, want in ((30, "testing"), (400, "cancelled")):
        app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: "2026-10-20", face_engine=lambda: FakeEngine())
        t0 = models.utcnow()
        clock = {"t": t0}
        monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
        with TestClient(app) as c:
            c.post("/api/entries/import", content=b"external_id,name,bib,gender,dob\nC-1,Alpha,11,M,2000-01-01\n")
            e = c.get("/api/entries").json()[0]["id"]
            c.post(f"/api/entries/{e}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True})
            c.post(f"/api/entries/{e}/line-up")
            start_lined_up(c)
            clock["t"] = t0 + timedelta(minutes=14)
            c.post("/api/run/finish", json={"bib": "11"})
            clock["t"] = t0 + timedelta(minutes=20)
            a = c.post("/api/attempts", json={"entry_id": e, "event": "pushup", "lane": "S"}).json()   # the camera's test, then the app restarts
            clock["t"] = t0 + timedelta(seconds=later)   # the attempt was stamped with the real clock (about t0)
            v = c.get("/api/stationflow").json()
            finished = {x["id"] for x in c.get("/api/attempts").json()}
            if want == "testing":
                assert v["state"] == "testing" and v["attempt_id"] == a["id"] and a["id"] not in finished
            else:
                assert v["state"] == "idle" and a["id"] not in finished
                assert c.post("/api/attempts", json={"entry_id": e, "event": "pushup", "lane": "S"}).status_code == 201   # free to test again


def test_getting_into_a_start_position_while_choosing_chooses_that_test():
    from testpoint.api.station import READY_HOLD_S
    me = (7, "choose", None)
    f = StationFlow(countdown_s=3.0)
    f.observe(0.0, StationLook(True, me)); f.observe(0.25, StationLook(True, me))
    assert f.state == "choose"
    f.observe(1.0, StationLook(True, ready_ex="pushup"))
    f.observe(1.0 + READY_HOLD_S, StationLook(True, ready_ex="pushup"))
    assert f.state == "position" and f.event == "pushup"
    f.observe(1.1 + READY_HOLD_S, StationLook(True, ready=True))      # still in position: the 3-2-1 at once
    assert f.state == "countdown"


def test_without_a_countdown_the_count_starts_as_soon_as_they_are_ready():
    from testpoint.api.station import READY_HOLD_S
    f = StationFlow(countdown_s=0.0)
    f.observe(0.0, StationLook(True, (7, "pushup", None))); f.observe(0.25, StationLook(True, (7, "pushup", None)))
    assert f.state == "position"
    f.observe(1.0, StationLook(True, ready=True))
    acts = f.observe(1.0 + READY_HOLD_S, StationLook(True, ready=True))
    acts += f.observe(1.1 + READY_HOLD_S, StationLook(True, ready=True))
    assert ("start", 7, "pushup") in acts and f.state == "testing" and f.go_at == 1.0 + READY_HOLD_S   # GO = ready
