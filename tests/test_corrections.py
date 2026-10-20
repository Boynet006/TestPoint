"""Correcting examiner mistakes: every step of the day can be undone safely."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import app as app_mod
from testpoint.api import models
from testpoint.api.app import create_app

CSV = "external_id,name,bib,gender,dob\nC-1,Alpha,11,M,2000-01-01\nC-2,Bravo,12,M,2000-01-01\nC-3,Charlie,13,M,2000-01-01\n"
W = {"weight_kg": 70, "height_cm": 175, "accept": True}


@pytest.fixture()
def client(monkeypatch):
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: "2026-10-20",
                     face_engine=lambda: None)
    with TestClient(app) as c:
        c.post("/api/entries/import", content=CSV.encode())
        yield c


def ids(c):
    return {e["candidate"]["bib"]: e["id"] for e in c.get("/api/entries").json()}


def status(c, i):
    return c.get(f"/api/entries/{i}").json()["status"]


def code(r):
    return r.json()["detail"]["code"]


def test_undo_line_up_before_start_only(client):
    i = ids(client)
    client.post(f"/api/entries/{i['11']}/weigh-in", json=W)
    client.post(f"/api/entries/{i['11']}/line-up")
    assert client.post(f"/api/entries/{i['11']}/line-up/undo").json()["status"] == "accepted"
    assert code(client.post(f"/api/entries/{i['11']}/line-up/undo")) == "not_lined_up"
    client.post(f"/api/entries/{i['11']}/line-up")
    client.post("/api/run/start")
    assert code(client.post(f"/api/entries/{i['11']}/line-up/undo")) == "not_lined_up"   # running now


def test_undo_weigh_in_then_redo(client):
    i = ids(client)
    client.post(f"/api/entries/{i['12']}/weigh-in", json={"weight_kg": 170, "height_cm": 175, "accept": False})
    r = client.post(f"/api/entries/{i['12']}/weigh-in/undo", json={"reason": "typed 170 instead of 70"})
    assert r.json()["status"] == "expected" and r.json()["bmi"] is None
    assert client.post(f"/api/entries/{i['12']}/weigh-in", json=W).json()["status"] == "accepted"
    client.post(f"/api/entries/{i['12']}/line-up")
    assert code(client.post(f"/api/entries/{i['12']}/weigh-in/undo")) == "remove_from_line_first"


def test_remove_candidate_only_while_untouched(client):
    i = ids(client)
    assert client.delete(f"/api/entries/{i['13']}").status_code == 204
    assert "13" not in ids(client)
    client.post(f"/api/entries/{i['11']}/weigh-in", json=W)
    assert code(client.delete(f"/api/entries/{i['11']}")) == "cannot_remove"


def test_cancel_accidental_start(client, monkeypatch):
    i = ids(client)
    for b in ("11", "12"):
        client.post(f"/api/entries/{i[b]}/weigh-in", json=W)
    client.post(f"/api/entries/{i['11']}/line-up")          # 12 accepted but not at the line
    client.post("/api/run/start")
    assert status(client, i["11"]) == "running" and status(client, i["12"]) == "accepted"
    client.post(f"/api/entries/{i['12']}/line-up")          # 12 lines up for wave 2 meanwhile
    assert client.post("/api/run/start/cancel").status_code == 200
    # both are back in wave 1, waiting for the real start
    assert status(client, i["11"]) == "lined_up" and status(client, i["12"]) == "lined_up"
    assert [w["number"] for w in client.get("/api/state").json()["waves"]] == [1]
    assert client.get("/api/state").json()["run_started_at"] is None
    # too late after 2 minutes
    client.post("/api/run/start")
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=3))
    assert code(client.post("/api/run/start/cancel")) == "too_late_to_cancel"
    # an administrator can still call it off while nobody of the wave has finished
    client.put("/api/settings/admin-pin", json={"pin": "2468"})
    assert code(client.post("/api/run/start/cancel", params={"late": True})) == "admin_pin_required"
    r = client.post("/api/run/start/cancel", params={"late": True}, headers={"X-Admin-Pin": "2468"})
    assert r.status_code == 200 and status(client, i["11"]) == "lined_up"


def test_cancel_start_refused_once_someone_finished(client):
    i = ids(client)
    client.post(f"/api/entries/{i['11']}/weigh-in", json=W)
    client.post(f"/api/entries/{i['11']}/line-up")
    client.post("/api/run/start")
    client.post("/api/run/finish", json={"bib": "11"})
    assert code(client.post("/api/run/start/cancel")) == "finishers_exist"


def test_wrong_bib_at_finish_can_be_unassigned_and_reassigned(client):
    i = ids(client)
    for b in ("11", "12"):
        client.post(f"/api/entries/{i[b]}/weigh-in", json=W)
        client.post(f"/api/entries/{i[b]}/line-up")
    client.post("/api/run/start")
    wrong = client.post("/api/run/finish", json={"bib": "12"}).json()   # it was actually 11
    assert status(client, i["12"]) == "run_passed"
    r = client.post(f"/api/run/crossings/{wrong['id']}/unassign").json()
    assert r["needs_review"] and r["entry_id"] is None
    assert status(client, i["12"]) == "running"
    client.post(f"/api/run/crossings/{wrong['id']}/resolve", json={"entry_id": i["11"]})
    assert status(client, i["11"]) == "run_passed"


def test_discard_and_close_rules(client):
    i = ids(client)
    client.post(f"/api/entries/{i['11']}/weigh-in", json=W)
    client.post(f"/api/entries/{i['11']}/line-up")
    client.post("/api/run/start")
    ghost = client.post("/api/run/finish", json={"bib": None}).json()
    assert code(client.post("/api/run/close")) == "review_pending"       # cannot close with an open review
    assert client.delete(f"/api/run/crossings/{ghost['id']}").status_code == 204
    assert client.post("/api/run/close").status_code == 200
    assert status(client, i["11"]) == "dnf"


def test_unassign_refused_after_exercise_results(client):
    i = ids(client)
    client.post(f"/api/entries/{i['11']}/weigh-in", json=W)
    client.post(f"/api/entries/{i['11']}/line-up")
    client.post("/api/run/start")
    c = client.post("/api/run/finish", json={"bib": "11"}).json()
    a = client.post("/api/attempts", json={"entry_id": i["11"], "event": "pushup"}).json()
    client.post(f"/api/attempts/{a['id']}/finish", json={"count": 30})
    assert code(client.post(f"/api/run/crossings/{c['id']}/unassign")) == "results_exist"


def test_review_notes_carry_a_code_for_translation(client):
    i = ids(client)
    client.post(f"/api/entries/{i['11']}/weigh-in", json=W)
    client.post(f"/api/entries/{i['11']}/line-up")
    client.post("/api/run/start")
    no_bib = client.post("/api/run/finish", json={"bib": None}).json()
    stranger = client.post("/api/run/finish", json={"bib": "999"}).json()
    assert (no_bib["note_code"], stranger["note_code"], stranger["note_bib"]) == ("no_bib_read", "bib_not_on_list", "999")
    ok = client.post("/api/run/finish", json={"bib": "11"}).json()
    again = client.post("/api/run/finish", json={"bib": "11"}).json()
    assert (again["note_code"], again["note_status"]) == ("bib_not_running", "run_passed")
    r = client.post(f"/api/run/crossings/{ok['id']}/unassign").json()
    assert (r["note_code"], r["note_bib"]) == ("unassigned_by_examiner", "11")
