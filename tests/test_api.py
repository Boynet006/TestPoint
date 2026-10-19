"""End-to-end test of the candidate journey through the HTTP API (in-memory SQLite)."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import app as app_mod
from testpoint.api import models
from testpoint.api.app import create_app

CSV = """external_id,name,name_ar,gender,dob,bib
C-1,Ahmed Hassan,أحمد حسن,M,1999-04-12,11
C-2,Mariam Ali,مريم علي,F,2001-11-02,12
C-3,Sami Omar,,M,1998-07-30,13
C-4,Lina Khalil,,F,2000-01-19,14
C-5,Omar Saleh,,M,1997-09-05,15
"""


@pytest.fixture()
def client(monkeypatch):
    # one shared in-memory database for every connection of the test
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine",
                        lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", tz="Asia/Dubai", web_dir="does-not-exist",
                     today=lambda: "2026-10-20")
    with TestClient(app) as c:
        yield c


def ok(r, code=200):
    assert r.status_code == code, r.text
    return r.json()


def err(r, code, reason):
    assert r.status_code == code, r.text
    assert r.json()["detail"]["code"] == reason, r.text


def entry_by_bib(client, bib):
    return ok(client.post("/api/checkin", json={"station": "weigh", "query": str(bib)}))["entry"]


def test_import_and_list(client):
    res = ok(client.post("/api/entries/import", content=CSV.encode("utf-8"),
                         headers={"content-type": "text/csv"}))
    assert res == {"added": 5, "updated": 0, "unchanged": 0, "errors": []}
    rows = ok(client.get("/api/entries"))
    assert [r["candidate"]["bib"] for r in rows] == ["11", "12", "13", "14", "15"]
    assert rows[0]["candidate"]["name_ar"] == "أحمد حسن"
    assert ok(client.get("/api/state"))["counts"]["expected"] == 5
    # duplicate bib is refused, re-importing the same people is harmless
    err(client.post("/api/entries", json={"external_id": "C-9", "name": "X", "bib": "11"}), 409, "bib_in_use")
    assert ok(client.post("/api/entries/import", content=CSV.encode()))["unchanged"] == 5
    assert ok(client.get("/api/state"))["total"] == 5


def test_unknown_person_is_refused_at_checkin(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    err(client.post("/api/checkin", json={"station": "weigh", "query": "999"}), 404, "not_on_todays_list")


def test_full_day(client, monkeypatch):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    ids = {b: entry_by_bib(client, b)["id"] for b in (11, 12, 13, 14, 15)}

    # exercise stations refuse before the run
    c = ok(client.post("/api/checkin", json={"station": "pushup", "query": "11"}))
    assert c["allowed"] is False and c["reason"] == "not_weighed"

    # weigh-in: 11, 12, 13, 15 accepted; 14 BMI too high -> rejected; override needs a reason
    for b in (11, 12, 13, 15):
        ok(client.post(f"/api/entries/{ids[b]}/weigh-in",
                       json={"weight_kg": 72, "height_cm": 176, "accept": True}))
    err(client.post(f"/api/entries/{ids[14]}/weigh-in",
                    json={"weight_kg": 110, "height_cm": 170, "accept": True}), 409, "bmi_out_of_range")
    e14 = ok(client.post(f"/api/entries/{ids[14]}/weigh-in",
                         json={"weight_kg": 110, "height_cm": 170, "accept": False}))
    assert e14["status"] == "rejected" and e14["bmi"] == 38.1
    err(client.post(f"/api/entries/{ids[11]}/weigh-in",
                    json={"weight_kg": 72, "height_cm": 176, "accept": True}), 409, "already_weighed")

    # line-up: 15 does not show up; rejected 14 is refused
    for b in (11, 12, 13):
        ok(client.post(f"/api/entries/{ids[b]}/line-up"))
    err(client.post(f"/api/entries/{ids[14]}/line-up"), 409, "rejected_at_weigh_in")

    # mass start, with a controllable clock
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    ok(client.post("/api/run/start"))
    err(client.post("/api/run/start"), 409, "run_already_started")
    st = ok(client.get("/api/state"))["counts"]
    assert st["running"] == 3 and st["absent"] == 1 and st["rejected"] == 1
    err(client.post(f"/api/entries/{ids[15]}/line-up"), 409, "run_already_started")

    def cross(bib, minutes):
        monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=minutes))
        return ok(client.post("/api/run/finish", json={"bib": bib}), 201)

    a = cross("11", 14.2)
    assert a["time_s"] == 852.0 and not a["needs_review"]
    unknown = cross(None, 15.0)                    # bib hidden -> review queue
    assert unknown["needs_review"] and unknown["note"] == "no bib read"
    ghost = cross("14", 15.5)                      # rejected candidate's bib -> review
    assert ghost["needs_review"]
    cross("13", 17.0)                              # over 16:00 -> fail
    assert ok(client.get("/api/state"))["review_queue"] == 2

    # examiner assigns the hidden-bib crossing to 12
    r = ok(client.post(f"/api/run/crossings/{unknown['id']}/resolve", json={"entry_id": ids[12]}))
    assert r["entry_id"] == ids[12] and not r["needs_review"]
    statuses = {b: ok(client.get(f"/api/entries/{ids[b]}"))["status"] for b in ids}
    assert statuses == {11: "run_passed", 12: "run_passed", 13: "run_failed",
                        14: "rejected", 15: "absent"}
    err(client.post("/api/run/close"), 409, "review_pending")    # the bib-14 crossing is still open
    assert client.delete(f"/api/run/crossings/{ghost['id']}").status_code == 204
    ok(client.post("/api/run/close"))

    # stations: failed / absent refused, passed accepted once per event
    for b, reason in ((13, "run_not_passed"), (15, "missed_the_run"), (14, "rejected_at_weigh_in")):
        err(client.post("/api/attempts", json={"entry_id": ids[b], "event": "pushup"}), 409, reason)
    att = ok(client.post("/api/attempts", json={"entry_id": ids[11], "event": "pushup", "lane": "1"}), 201)
    err(client.post("/api/attempts", json={"entry_id": ids[11], "event": "crunch"}), 409, "attempt_in_progress")
    done = ok(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 31, "rejected": 2}))
    assert done["passed"] is True
    err(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 1}), 409, "attempt_not_running")
    err(client.post("/api/attempts", json={"entry_id": ids[11], "event": "pushup"}), 409, "event_already_done")
    att2 = ok(client.post("/api/attempts", json={"entry_id": ids[11], "event": "crunch", "lane": "2"}), 201)
    assert ok(client.post(f"/api/attempts/{att2['id']}/finish", json={"count": 10}))["passed"] is False
    e11 = ok(client.get(f"/api/entries/{ids[11]}"))
    assert e11["status"] == "complete"
    assert e11["events"]["pushup"]["count"] == 31 and e11["events"]["crunch"]["passed"] is False

    # a cancelled attempt can be retried
    att3 = ok(client.post("/api/attempts", json={"entry_id": ids[12], "event": "pushup"}), 201)
    assert client.post(f"/api/attempts/{att3['id']}/cancel").status_code == 204
    ok(client.post("/api/attempts", json={"entry_id": ids[12], "event": "pushup"}), 201)


def test_runner_not_running_cannot_be_dnf_before_close(client, monkeypatch):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    ok(client.post("/api/run/start"))
    ok(client.post("/api/run/close"))
    assert ok(client.get(f"/api/entries/{e['id']}"))["status"] == "dnf"
    err(client.post("/api/run/close"), 409, "run_already_closed")
