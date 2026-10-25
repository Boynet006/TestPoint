"""The register: everyone saved, on a test day or not."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import models
from testpoint.api.app import create_app

from test_faces import FakeEngine


@pytest.fixture
def ctx(monkeypatch):
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    day = {"d": "2026-10-20"}
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: day["d"], face_engine=FakeEngine)
    with TestClient(app) as c:
        yield c, day


def reg(c, **kw):
    return c.get("/api/candidates", params=kw).json()


def test_register_without_a_test_day_then_add_to_today(ctx):
    c, _ = ctx
    r = c.post("/api/candidates", json={"external_id": "784-1", "name": "Ali Saeed", "name_ar": "علي سعيد", "dob": "2000-05-01"})
    assert r.status_code == 201 and r.json()["outcome"] == "added"
    c.post("/api/candidates", json={"external_id": "784-2", "name": "Omar Khalid", "bib": "7"})
    assert c.get("/api/entries").json() == []                         # nobody is on today's test
    page = reg(c)
    assert page["total"] == 2 and [x["name"] for x in page["rows"]] == ["Ali Saeed", "Omar Khalid"]
    assert reg(c, q="omar")["total"] == 1 and reg(c, q="علي")["total"] == 1 and reg(c, q="784-2")["total"] == 1
    # the same ID with another name is a different person: refused
    assert c.post("/api/candidates", json={"external_id": "784-1", "name": "Someone Else"}).status_code == 409
    ids = [x["id"] for x in page["rows"]]
    r = c.post("/api/candidates/today", json={"ids": ids + ids[:1]}).json()
    assert r["added"] == 2 and r["already"] == 0 and r["errors"] == []
    today = {e["candidate"]["external_id"]: e for e in c.get("/api/entries").json()}
    assert today["784-2"]["candidate"]["bib"] == "7" and today["784-1"]["candidate"]["bib"]   # usual bib kept, else the next
    assert c.post("/api/candidates/today", json={"ids": ids}).json()["already"] == 2
    assert reg(c, show="today")["total"] == 2 and reg(c, show="not_today")["total"] == 0


def test_face_enrolled_in_the_register_is_recognised_at_weigh_in(ctx):
    c, _ = ctx
    cid = c.post("/api/candidates", json={"external_id": "784-1", "name": "Ali"}).json()["id"]
    assert c.post(f"/api/candidates/{cid}/face", content=b"face:A").json() == {"faces": 1}
    assert c.get(f"/api/candidates/{cid}/face.jpg").status_code == 200
    r = c.post("/api/face/identify?station=weigh", content=b"face:A").json()
    assert r["entry"] is None and r["reason"] == "not_on_todays_list" and r["known"]["external_id"] == "784-1"
    assert c.get("/api/candidates/lookup", params={"q": "784-1"}).json()["name"] == "Ali"
    assert c.get("/api/candidates/lookup", params={"q": "nobody"}).status_code == 404
    assert reg(c, show="no_face")["total"] == 0


def test_history_edit_and_csv_round_trip(ctx):
    c, day = ctx
    c.post("/api/entries", json={"external_id": "A-1", "name": "Alpha", "bib": "11", "gender": "M", "dob": "1999-01-01"})
    day["d"] = "2026-10-27"
    cid = reg(c)["rows"][0]["id"]
    h = c.get(f"/api/candidates/{cid}").json()
    assert [d["date"] for d in h["days"]] == ["2026-10-20"] and h["days"][0]["today"] is False
    assert c.patch(f"/api/candidates/{cid}", json={"name_ar": "ألفا"}).json()["name_ar"] == "ألفا"
    imp = c.post("/api/candidates/import", content="external_id,name,unit\nB-1,=Bravo,admin\nA-1,Alpha,\n".encode())
    assert imp.json()["added"] == 1 and imp.json()["unchanged"] == 1
    assert c.get("/api/entries").json() == []                         # the import put nobody on a test day
    csv = c.get("/api/candidates/export.csv")
    assert csv.headers["content-type"].startswith("text/csv") and csv.text.startswith("\ufeffexternal_id,name")
    assert "'=Bravo" in csv.text                                       # never a formula in Excel


def test_removal_from_today_keeps_the_person_and_delete_erases(ctx):
    c, _ = ctx
    e = c.post("/api/entries", json={"external_id": "A-1", "name": "Alpha", "bib": "11", "gender": "M", "dob": "1999-01-01"}).json()
    c.post(f"/api/entries/{e['id']}/face", content=b"face:A")
    assert c.delete(f"/api/entries/{e['id']}").status_code == 204
    row = reg(c)["rows"][0]
    assert row["external_id"] == "A-1" and row["faces"] == 1 and row["today_entry"] is None
    c.post("/api/candidates/today", json={"ids": [row["id"]]})
    entry = c.get("/api/entries").json()[0]
    assert c.post(f"/api/entries/{entry['id']}/weigh-in", json={"weight_kg": 80, "height_cm": 180, "accept": True}).status_code == 200
    assert c.get(f"/api/candidates/{row['id']}").json()["days"][0]["status"] == "accepted"
    assert c.delete(f"/api/candidates/{row['id']}").json()["detail"]["code"] == "has_results"   # results: a second, explicit yes
    assert c.delete(f"/api/candidates/{row['id']}?delete_results=true").status_code == 204
    assert reg(c)["total"] == 0 and c.get("/api/entries").json() == []
    assert c.post("/api/face/identify?station=weigh", content=b"face:A").json().get("known") is None


def test_register_changes_need_the_admin_pin(ctx):
    c, _ = ctx
    cid = c.post("/api/candidates", json={"external_id": "A-1", "name": "Alpha"}).json()["id"]
    assert c.put("/api/settings/admin-pin", json={"pin": "1234"}).status_code == 200
    assert c.patch(f"/api/candidates/{cid}", json={"name_ar": "x"}).status_code == 403
    assert c.delete(f"/api/candidates/{cid}").status_code == 403
    assert c.get("/api/candidates/export.csv").status_code == 403
    pin = {"X-Admin-Pin": "1234"}
    assert c.get("/api/candidates/export.csv", headers=pin).status_code == 200
    assert c.delete(f"/api/candidates/{cid}", headers=pin).status_code == 204
