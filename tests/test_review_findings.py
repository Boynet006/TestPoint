"""Regression tests for the issues found in the 2026-10-19 code review."""

import threading
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import app as app_mod
from testpoint.api import journey as J
from testpoint.api import models
from testpoint.api.app import create_app

W = {"weight_kg": 70, "height_cm": 175, "accept": True}


def make(monkeypatch, today="2026-10-20"):
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    box = {"today": today}
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: box["today"],
                     face_engine=lambda: None)
    return app, box


@pytest.fixture()
def client(monkeypatch):
    app, _ = make(monkeypatch)
    with TestClient(app) as c:
        yield c


def code(r):
    return r.json()["detail"]["code"]


def entry_id(c, q):
    return c.post("/api/checkin", json={"station": "weigh", "query": q}).json()["entry"]["id"]


def test_bib_kept_from_another_day_cannot_clash(monkeypatch):
    app, box = make(monkeypatch, "2026-10-20")
    with TestClient(app) as c:
        c.post("/api/entries", json={"external_id": "P-1", "name": "Pat", "bib": "21"})
        box["today"] = "2026-10-21"
        c.post("/api/entries", json={"external_id": "NEW-1", "name": "Newcomer", "bib": "21"})
        r = c.post("/api/entries", json={"external_id": "P-1", "name": "Pat"})   # no bib: would keep 21
        assert r.status_code == 409 and code(r) == "bib_in_use"


def test_csv_trailing_comma_and_blank_lines_do_not_break_the_import(client):
    csv = "external_id,name,bib\nX-1,Ali,1,\n\nX-2,Badr,2\nX-3,Carl,3,EXTRA\n"
    r = client.post("/api/entries/import", content=csv.encode()).json()
    assert r["added"] == 2
    assert [e["code"] for e in r["errors"]] == ["extra_columns"]


def test_csv_arabic_windows_encoding(client):
    csv = "external_id,name,name_ar,bib\nA-1,Ahmed,أحمد,5\n".encode("cp1256")
    assert client.post("/api/entries/import", content=csv).json()["added"] == 1
    e = client.post("/api/checkin", json={"station": "weigh", "query": "5"}).json()["entry"]
    assert e["candidate"]["name_ar"] == "أحمد"


def test_csv_duplicate_rows_and_counts(client):
    csv = "external_id,name,bib\nD-1,Huda,6\nD-1,Huda,6\n"
    r = client.post("/api/entries/import", content=csv.encode()).json()
    assert r["added"] == 1 and [e["code"] for e in r["errors"]] == ["duplicate_row"]
    r = client.post("/api/entries/import", content=csv.encode()).json()
    assert r["added"] == 0 and r["unchanged"] == 1


def test_existing_id_with_another_name_is_refused_not_renamed(client):
    client.post("/api/entries", json={"external_id": "C-7", "name": "Real Person", "bib": "7"})
    r = client.post("/api/entries", json={"external_id": "c-7", "name": "Typo Person"})
    assert r.status_code == 409 and code(r) == "external_id_in_use"
    e = client.post("/api/checkin", json={"station": "weigh", "query": "C-7"}).json()["entry"]
    assert e["candidate"]["name"] == "Real Person"


def test_ambiguous_query_is_refused(client):
    client.post("/api/entries", json={"external_id": "12", "name": "Id Twelve", "bib": "40"})
    client.post("/api/entries", json={"external_id": "Z-9", "name": "Bib Twelve", "bib": "12"})
    r = client.post("/api/checkin", json={"station": "weigh", "query": "12"})
    assert r.status_code == 409 and code(r) == "ambiguous_query"


def test_bib_normalisation_and_gender_dob(client):
    r = client.post("/api/entries", json={"external_id": "N-1", "name": "Zero", "bib": "011", "gender": "m"})
    assert r.status_code == 201 and r.json()["candidate"]["bib"] == "11" and r.json()["candidate"]["gender"] == "M"
    assert client.post("/api/checkin", json={"station": "weigh", "query": "0011"}).json()["entry"]["candidate"]["name"] == "Zero"
    assert code(client.post("/api/entries", json={"external_id": "N-2", "name": "X", "gender": "q"})) == "bad_gender"
    assert code(client.post("/api/entries", json={"external_id": "N-3", "name": "X", "dob": "31/12/1999"})) == "bad_dob"


def test_input_errors_have_a_code(client):
    client.post("/api/entries", json={"external_id": "V-1", "name": "Val", "bib": "1"})
    r = client.post(f"/api/entries/{entry_id(client, '1')}/weigh-in", json={"weight_kg": 450, "height_cm": 175, "accept": True})
    assert r.status_code == 422 and code(r) == "bad_input" and "weight_kg" in r.json()["detail"]["message"]
    assert code(client.post("/api/checkin", json={"station": "weigh", "query": "  "})) == "empty_query"


def test_rejecting_inside_the_bmi_range_needs_a_reason(client):
    client.post("/api/entries", json={"external_id": "R-1", "name": "Rae", "bib": "1"})
    i = entry_id(client, "1")
    assert code(client.post(f"/api/entries/{i}/weigh-in", json=W | {"accept": False})) == "reason_required"
    r = client.post(f"/api/entries/{i}/weigh-in", json=W | {"accept": False, "override_reason": "injured knee"})
    assert r.json()["status"] == "rejected"


def setup_runner(c, bib="1"):
    c.post("/api/entries", json={"external_id": f"E-{bib}", "name": f"Runner {bib}", "bib": bib})
    i = entry_id(c, bib)
    c.post(f"/api/entries/{i}/weigh-in", json=W)
    c.post(f"/api/entries/{i}/line-up")
    return i


def test_one_running_attempt_even_with_parallel_requests(client):
    i = setup_runner(client)
    client.post("/api/run/start")
    client.post("/api/run/finish", json={"bib": "1"})
    results = []
    def go():
        results.append(client.post("/api/attempts", json={"entry_id": i, "event": "pushup"}).status_code)
    threads = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert sorted(results).count(201) == 1 and results.count(409) == 5


def test_finish_after_cutoff_is_dnf_and_closed_run_takes_no_finishes(client, monkeypatch):
    i = setup_runner(client)
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    client.post("/api/run/start")
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=26))
    client.post("/api/run/finish", json={"bib": "1"})
    assert client.get(f"/api/entries/{i}").json()["status"] == "dnf"
    client.post("/api/run/close")
    assert code(client.post("/api/run/finish", json={"bib": "1"})) == "run_already_closed"


def test_open_run_keeps_its_day_past_midnight(monkeypatch):
    app, box = make(monkeypatch, "2026-10-20")
    with TestClient(app) as c:
        setup_runner(c)
        c.post("/api/run/start")
        box["today"] = "2026-10-21"                       # midnight passes during the run
        assert c.get("/api/state").json()["date"] == "2026-10-20"
        assert c.post("/api/run/finish", json={"bib": "1"}).status_code == 201
        c.post("/api/run/close")
        assert c.get("/api/state").json()["date"] == "2026-10-21"


def test_upload_size_limit(client):
    r = client.post("/api/entries/import", content=b"x" * (16 * 1024 * 1024))
    assert r.status_code == 413 and code(r) == "too_large"


def test_after_finish_rule():
    assert J.after_finish(900, 960, 1500) == J.RUN_PASSED
    assert J.after_finish(1000, 960, 1500) == J.RUN_FAILED
    assert J.after_finish(1600, 960, 1500) == J.DNF
