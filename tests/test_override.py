"""Examiner override of a camera-counted result: the reason rule, the review list and evidence files."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from testpoint.api import app as app_mod
from testpoint.api import models
from testpoint.api.app import create_app
from testpoint.api.models import Attempt, AuditLog

CSV = """external_id,name,bib,gender,dob
C-1,Ahmed Hassan,11,M,2000-01-01
C-2,Mariam Ali,12,M,2000-01-01
"""


@pytest.fixture()
def engines(monkeypatch):
    """Every engine the app creates, so a test can read the audit log directly."""
    made = []
    real = models.create_engine

    def make(url, **kw):
        e = real(url, poolclass=StaticPool, **kw)
        made.append(e)
        return e

    monkeypatch.setattr(models, "create_engine", make)
    return made


@pytest.fixture()
def evidence(tmp_path):
    d = tmp_path / "evidence"
    d.mkdir()
    return d


@pytest.fixture()
def client(engines, evidence):
    app = create_app(db_url="sqlite:///:memory:", tz="Asia/Dubai", web_dir="does-not-exist",
                     today=lambda: "2026-10-20", evidence_dir=evidence)
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
        ok(client.post(f"/api/entries/{e['id']}/weigh-in",
                       json={"weight_kg": 70, "height_cm": 175, "accept": True}))
        ok(client.post(f"/api/entries/{e['id']}/line-up"))
    t0 = app_mod.S.utcnow()
    ok(client.post("/api/run/start"))
    for bib in (11, 12):
        monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=14))
        ok(client.post("/api/run/finish", json={"bib": str(bib)}), 201)
    monkeypatch.undo()
    ok(client.post("/api/run/close"))
    return ids


def report(client, lanes, station="pushup"):
    return ok(client.post("/api/engine/report",
                          json={"station": station, "t": 1.0, "fps": 30.0, "source": "0", "lanes": lanes}))


def event(index, valid=True, faults=(), t_start=0.0, t_end=1.0, clip=None):
    return {"index": index, "valid": valid, "faults": list(faults),
            "t_start": t_start, "t_end": t_end, "clip": clip}


def start(client, entry_id, lane="1", event_name="pushup"):
    return ok(client.post("/api/attempts", json={"entry_id": entry_id, "event": event_name, "lane": lane}), 201)


def last_finish_audit(engines) -> dict:
    import json

    with Session(engines[0]) as db:
        row = db.scalar(select(AuditLog).where(AuditLog.action == "finish_attempt")
                        .order_by(AuditLog.id.desc()))
        return json.loads(row.detail)


def test_without_a_camera_count_any_number_saves(client, passed):
    att = start(client, passed[11])
    done = ok(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 20, "rejected": 1}))
    assert done["status"] == "finished" and done["auto_count"] is None
    assert done["override_reason"] is None


def test_matching_count_needs_no_reason(client, passed, engines):
    att = start(client, passed[11])
    report(client, {"1": {"attempt_id": att["id"], "count": 5, "rejected": 1}})
    done = ok(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 5, "rejected": 1}))
    assert done["auto_count"] == 5 and done["override_reason"] is None
    detail = last_finish_audit(engines)
    assert detail["auto_count"] == 5 and detail["override_reason"] is None


def test_different_count_needs_a_reason(client, passed, engines):
    att = start(client, passed[11])
    report(client, {"1": {"attempt_id": att["id"], "count": 5, "rejected": 1,
                          "events": [event(1, clip="2026-10-20/1/rep-001.mp4"),
                                     event(2, valid=False, faults=["knee_on_ground"], t_start=1.5, t_end=2.5)]}})
    err(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 7, "rejected": 1}),
        409, "override_reason_required")
    err(client.post(f"/api/attempts/{att['id']}/finish",
                    json={"count": 7, "rejected": 1, "override_reason": "   "}),
        409, "override_reason_required")
    # refused: still running, nothing to review yet
    assert ok(client.get("/api/attempts")) == []
    done = ok(client.post(f"/api/attempts/{att['id']}/finish",
                          json={"count": 7, "rejected": 1, "override_reason": "camera missed two reps"}))
    assert done["count"] == 7 and done["auto_count"] == 5
    assert done["override_reason"] == "camera missed two reps"
    detail = last_finish_audit(engines)
    assert detail["count"] == 7 and detail["auto_count"] == 5
    assert detail["override_reason"] == "camera missed two reps"
    with Session(engines[0]) as db:
        assert db.get(Attempt, att["id"]).override_reason == "camera missed two reps"

    # the review list: saved vs camera counts, the reason, the reps with their clips
    rows = ok(client.get("/api/attempts"))
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate"]["bib"] == "11" and row["candidate"]["name"] == "Ahmed Hassan"
    assert (row["count"], row["auto_count"], row["override_reason"]) == (7, 5, "camera missed two reps")
    assert [r["index"] for r in row["reps"]] == [1, 2]
    assert row["reps"][0]["clip"] == "2026-10-20/1/rep-001.mp4"
    assert row["reps"][1] == {"index": 2, "valid": False, "faults": ["knee_on_ground"],
                              "t_start": 1.5, "t_end": 2.5, "clip": None}


def test_review_list_is_newest_first_and_filters_by_event(client, passed):
    a = start(client, passed[11])
    ok(client.post(f"/api/attempts/{a['id']}/finish", json={"count": 3}))
    b = start(client, passed[12], lane="2", event_name="crunch")
    ok(client.post(f"/api/attempts/{b['id']}/finish", json={"count": 4}))
    c = start(client, passed[11], lane="3", event_name="crunch")      # cancelled: never listed
    assert client.post(f"/api/attempts/{c['id']}/cancel").status_code == 204
    assert [r["id"] for r in ok(client.get("/api/attempts"))] == [b["id"], a["id"]]
    assert [r["id"] for r in ok(client.get("/api/attempts?event=pushup"))] == [a["id"]]
    err(client.get("/api/attempts?event=weigh"), 400, "unknown_event")


def test_evidence_files_are_served_from_the_evidence_folder_only(client, evidence, tmp_path):
    clip = evidence / "2026-10-20" / "1" / "rep-001.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"not really mp4")
    secret = tmp_path / "secret.txt"
    secret.write_text("keep out")

    r = client.get("/api/evidence/2026-10-20/1/rep-001.mp4")
    assert r.status_code == 200 and r.content == b"not really mp4"
    assert r.headers["content-type"].startswith("video/mp4")
    err(client.get("/api/evidence/2026-10-20/1/rep-002.mp4"), 404, "not_found")
    err(client.get("/api/evidence/2026-10-20/1"), 404, "not_found")          # a folder, not a file
    for bad in ("../secret.txt", "..%2Fsecret.txt", "2026-10-20/../../secret.txt",
                "%2e%2e/secret.txt", str(secret), str(secret).replace("\\", "/")):
        r = client.get(f"/api/evidence/{bad}")
        assert r.status_code == 404, bad
        assert b"keep out" not in r.content, bad


def test_existing_database_gets_the_override_reason_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE attempt (id INTEGER PRIMARY KEY, entry_id INTEGER, event VARCHAR, lane VARCHAR,
            started_at DATETIME, ended_at DATETIME, count INTEGER, rejected INTEGER, passed BOOLEAN,
            auto_count INTEGER, auto_rejected INTEGER, status VARCHAR);
    """)
    con.commit()
    con.close()
    models.make_sessionmaker(f"sqlite:///{path}")
    con = sqlite3.connect(path)
    assert "override_reason" in {r[1] for r in con.execute("PRAGMA table_info(attempt)")}
