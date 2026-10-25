"""The engine -> app link (docs/ENGINE-LINK.md): reports, reps, acks and the live lanes picture."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import app as app_mod
from testpoint.api import models
from testpoint.api.app import ENGINE_STALE_S, create_app

CSV = """external_id,name,bib,gender,dob
C-1,Ahmed Hassan,11,M,2000-01-01
C-2,Mariam Ali,12,M,2000-01-01
"""


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def client(monkeypatch, clock):
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine",
                        lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", tz="Asia/Dubai", web_dir="does-not-exist",
                     today=lambda: "2026-10-20", clock=clock)
    with TestClient(app) as c:
        yield c


def ok(r, code=200):
    assert r.status_code == code, r.text
    return r.json()


def err(r, code, reason):
    assert r.status_code == code, r.text
    assert r.json()["detail"]["code"] == reason, r.text


@pytest.fixture()
def passed(client, monkeypatch):
    """Two candidates through the run, so the push-up station may take them. Returns entry ids by bib."""
    ok(client.post("/api/entries/import", content=CSV.encode()))
    ids = {}
    for bib in (11, 12):
        e = ok(client.post("/api/checkin", json={"station": "weigh", "query": str(bib)}))["entry"]
        ids[bib] = e["id"]
        ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
        ok(client.post(f"/api/entries/{e['id']}/line-up"))
    t0 = app_mod.S.utcnow()
    ok(client.post("/api/run/start"))
    for bib in (11, 12):
        monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=14))
        ok(client.post("/api/run/finish", json={"bib": str(bib)}), 201)
    monkeypatch.undo()
    ok(client.post("/api/run/close"))
    return ids


def report(client, lanes, station="pushup", **extra):
    body = {"station": station, "t": 1.0, "fps": 30.0, "source": "0", "lanes": lanes, **extra}
    return ok(client.post("/api/engine/report", json=body))


def event(index, valid=True, faults=(), t_start=0.0, t_end=1.0, clip=None):
    return {"index": index, "valid": valid, "faults": list(faults),
            "t_start": t_start, "t_end": t_end, "clip": clip}


def test_report_attaches_reps_to_the_running_attempt(client, passed):
    # Nobody is being tested yet: the reply says so, nothing is stored.
    r = report(client, {"1": {"count": 3}, "2": {}})
    assert r == {"lanes": {"1": {"attempt_id": None}, "2": {"attempt_id": None}}, "acked": {}}

    att = ok(client.post("/api/attempts", json={"entry_id": passed[11], "event": "pushup", "lane": "1"}), 201)
    r = report(client, {"1": {"count": 0}, "2": {}})
    assert r["lanes"]["1"]["attempt_id"] == att["id"]

    r = report(client, {"1": {"attempt_id": att["id"], "count": 2, "rejected": 1, "phase": "rest",
                              "events": [event(1, t_start=0.5, t_end=1.4),
                                         event(2, valid=False, faults=["knee_down"], t_start=1.5, t_end=2.6,
                                               clip="2026-10-20/1/rep-002.mp4"),
                                         event(3, t_start=2.7, t_end=3.5)]},
                        "2": {}})
    assert r["acked"] == {"1": 3}
    reps = ok(client.get(f"/api/attempts/{att['id']}/reps"))
    assert [(x["index"], x["valid"], x["faults"]) for x in reps] == [(1, True, []), (2, False, ["knee_down"]), (3, True, [])]
    assert reps[1]["clip"] == "2026-10-20/1/rep-002.mp4" and reps[1]["t_end"] == 2.6

    # The same events again (the engine never saw the ack): nothing is duplicated.
    report(client, {"1": {"attempt_id": att["id"], "count": 3, "rejected": 1,
                          "events": [event(2, valid=False, faults=["knee_down"]), event(3), event(4)]}})
    assert [x["index"] for x in ok(client.get(f"/api/attempts/{att['id']}/reps"))] == [1, 2, 3, 4]

    # The examiner saves a corrected count: both numbers are kept.
    done = ok(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 3, "rejected": 1}))
    assert (done["count"], done["auto_count"], done["auto_rejected"]) == (3, 3, 1)
    # ... and the lane is free again from the engine's point of view.
    assert report(client, {"1": {"attempt_id": att["id"], "count": 5}})["lanes"]["1"]["attempt_id"] is None
    assert len(ok(client.get(f"/api/attempts/{att['id']}/reps"))) == 4


def test_events_for_another_attempt_are_dropped_but_acked(client, passed):
    att = ok(client.post("/api/attempts", json={"entry_id": passed[11], "event": "pushup", "lane": "1"}), 201)
    # The engine has not heard of the attempt yet (attempt_id missing / stale).
    r = report(client, {"1": {"attempt_id": None, "count": 7, "events": [event(1), event(2)]}})
    assert r["acked"] == {"1": 2} and r["lanes"]["1"]["attempt_id"] == att["id"]
    assert ok(client.get(f"/api/attempts/{att['id']}/reps")) == []
    r = report(client, {"1": {"attempt_id": att["id"] + 100, "count": 7, "events": [event(1)]}})
    assert r["acked"] == {"1": 1}
    assert ok(client.get(f"/api/attempts/{att['id']}/reps")) == []
    assert ok(client.get(f"/api/entries/{passed[11]}"))["events"]["pushup"]["count"] is None

    # Cancelled: the lane is empty for the engine, late events are dropped.
    assert client.post(f"/api/attempts/{att['id']}/cancel").status_code == 204
    r = report(client, {"1": {"attempt_id": att["id"], "count": 1, "events": [event(1)]}})
    assert r["lanes"]["1"]["attempt_id"] is None and r["acked"] == {"1": 1}
    assert ok(client.get(f"/api/attempts/{att['id']}/reps")) == []


def test_lanes_are_per_station(client, passed):
    a = ok(client.post("/api/attempts", json={"entry_id": passed[11], "event": "pushup", "lane": "1"}), 201)
    b = ok(client.post("/api/attempts", json={"entry_id": passed[12], "event": "crunch", "lane": "1"}), 201)
    assert report(client, {"1": {}}, station="pushup")["lanes"]["1"]["attempt_id"] == a["id"]
    assert report(client, {"1": {}}, station="crunch")["lanes"]["1"]["attempt_id"] == b["id"]
    err(client.post("/api/engine/report", json={"station": "weigh", "lanes": {}}), 400, "unknown_station")
    err(client.post("/api/engine/report", json={"station": "pushup", "lanes": {"1": {"count": -1}}}), 422, "bad_input")
    err(client.get("/api/lanes", params={"station": "nope"}), 400, "unknown_station")


def test_live_lanes_picture_and_staleness(client, passed, clock):
    assert ok(client.get("/api/lanes", params={"station": "pushup"})) == {
        "station": "pushup", "engine": {"online": False, "age_s": None, "fps": None, "source": None}, "lanes": {}}

    att = ok(client.post("/api/attempts", json={"entry_id": passed[11], "event": "pushup", "lane": "2"}), 201)
    report(client, {"1": {"count": 1, "phase": "rest", "person": True},
                    "2": {"attempt_id": att["id"], "count": 4, "rejected": 1, "phase": "work", "person": True,
                          "pending_faults": ["hips_sag"], "work_seconds": 1.3, "depth": 0.6}})
    clock.t += 0.3
    live = ok(client.get("/api/lanes", params={"station": "pushup"}))
    assert live["engine"] == {"online": True, "age_s": 0.3, "fps": 30.0, "source": "0"}
    assert live["lanes"]["2"] == {"count": 4, "rejected": 1, "phase": "work", "person": True,
                                  "pending_faults": ["hips_sag"], "work_seconds": 1.3, "depth": 0.6,
                                  "attempt_id": att["id"], "current": True}
    assert live["lanes"]["1"]["current"] is False and live["lanes"]["1"]["attempt_id"] is None
    assert ok(client.get("/api/lanes", params={"station": "crunch"}))["engine"]["online"] is False

    # The engine stopped reporting: offline, last picture still shown, nothing marked current.
    clock.t += ENGINE_STALE_S + 1
    live = ok(client.get("/api/lanes", params={"station": "pushup"}))
    assert live["engine"]["online"] is False and live["lanes"]["2"]["count"] == 4
    # A new attempt on the lane: the stale report is not for it.
    assert client.post(f"/api/attempts/{att['id']}/cancel").status_code == 204
    att2 = ok(client.post("/api/attempts", json={"entry_id": passed[11], "event": "pushup", "lane": "2"}), 201)
    report(client, {"2": {"attempt_id": att["id"], "count": 4}})
    live = ok(client.get("/api/lanes", params={"station": "pushup"}))
    assert live["engine"]["online"] is True and live["lanes"]["2"]["current"] is False
    assert report(client, {"2": {}})["lanes"]["2"]["attempt_id"] == att2["id"]


def test_existing_database_gets_the_new_columns(tmp_path):
    """A database from before the engine link: attempt rows have no auto_* columns yet."""
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE attempt (id INTEGER PRIMARY KEY, entry_id INTEGER, event VARCHAR, lane VARCHAR,
            started_at DATETIME, ended_at DATETIME, count INTEGER, rejected INTEGER, passed BOOLEAN,
            status VARCHAR);
    """)
    con.commit()
    con.close()
    models.make_sessionmaker(f"sqlite:///{path}")
    con = sqlite3.connect(path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(attempt)")}
    assert {"auto_count", "auto_rejected"} <= cols
    assert con.execute("SELECT name FROM sqlite_master WHERE name='rep'").fetchone()



def test_reset_today_with_an_open_attempt_needs_a_yes(client, passed):
    ok(client.post("/api/attempts", json={"entry_id": passed[11], "event": "pushup", "lane": "1"}), 201)
    today = ok(client.get("/api/state"))["date"]
    r = client.post("/api/day/reset", json={"confirm": today})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "in_progress" and r.json()["detail"]["open"]
    done = ok(client.post("/api/day/reset", json={"confirm": today, "stop_running": True}))
    assert done["attempts"] == 1
    assert all(e["status"] == "expected" for e in ok(client.get("/api/entries")))
