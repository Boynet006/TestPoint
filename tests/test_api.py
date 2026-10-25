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

    # line-up: 15 is late (runs in the next wave); rejected 14 is refused
    for b in (11, 12, 13):
        ok(client.post(f"/api/entries/{ids[b]}/line-up"))
    err(client.post(f"/api/entries/{ids[14]}/line-up"), 409, "rejected_at_weigh_in")

    # wave 1 starts, with a controllable clock
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    assert ok(client.post("/api/run/start"))["wave"] == 1
    err(client.post("/api/run/start"), 409, "nobody_lined_up")
    st = ok(client.get("/api/state"))
    assert st["counts"]["running"] == 3 and st["counts"]["accepted"] == 1 and st["counts"]["rejected"] == 1
    assert [(w["number"], w["running"]) for w in st["waves"]] == [(1, 3)]
    # the late one lines up for wave 2 while wave 1 runs
    assert ok(client.post(f"/api/entries/{ids[15]}/line-up"))["wave"] == 2

    def cross(bib, minutes):
        monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=minutes))
        return ok(client.post("/api/run/finish", json={"bib": bib}), 201)

    a = cross("11", 14.2)
    assert a["time_s"] == 852.0 and not a["needs_review"]
    unknown = cross(None, 15.0)                    # bib hidden -> review queue
    assert unknown["needs_review"] and unknown["note"] == "no bib read"
    ghost = cross("14", 15.5)                      # rejected candidate's bib -> review
    assert ghost["needs_review"]
    cross("13", 19.0)                              # age 28: 60 % needs 18:30 -> fail
    assert ok(client.get("/api/state"))["review_queue"] == 2

    # examiner assigns the hidden-bib crossing to 12
    r = ok(client.post(f"/api/run/crossings/{unknown['id']}/resolve", json={"entry_id": ids[12]}))
    assert r["entry_id"] == ids[12] and not r["needs_review"]
    statuses = {b: ok(client.get(f"/api/entries/{ids[b]}"))["status"] for b in ids}
    # 12 is a woman: no women's table yet, the run is recorded without a verdict
    assert statuses == {11: "run_passed", 12: "run_done", 13: "run_failed",
                        14: "rejected", 15: "lined_up"}
    e11 = ok(client.get(f"/api/entries/{ids[11]}"))
    assert e11["standard"] == {"age": 27, "band": "23-30", "level": 60} and e11["run_grade"] == 100
    err(client.post("/api/run/close"), 409, "review_pending")    # the bib-14 crossing is still open
    assert client.delete(f"/api/run/crossings/{ghost['id']}").status_code == 204
    ok(client.post("/api/run/close"))

    # stations: everyone who ran takes all three tests; not yet run / rejected are refused
    for b, reason in ((15, "has_not_run"), (14, "rejected_at_weigh_in")):
        err(client.post("/api/attempts", json={"entry_id": ids[b], "event": "pushup"}), 409, reason)
    assert ok(client.post("/api/checkin", json={"station": "pushup", "query": "13"}))["allowed"] is True
    att = ok(client.post("/api/attempts", json={"entry_id": ids[11], "event": "pushup", "lane": "1"}), 201)
    err(client.post("/api/attempts", json={"entry_id": ids[11], "event": "crunch"}), 409, "attempt_in_progress")
    done = ok(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 45, "rejected": 2}))
    assert done["passed"] is True and done["grade"] == 70          # 42 <= 45 < 58
    err(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 1}), 409, "attempt_not_running")
    err(client.post("/api/attempts", json={"entry_id": ids[11], "event": "pushup"}), 409, "event_already_done")
    att2 = ok(client.post("/api/attempts", json={"entry_id": ids[11], "event": "crunch", "lane": "2"}), 201)
    assert ok(client.post(f"/api/attempts/{att2['id']}/finish", json={"count": 10}))["passed"] is False
    e11 = ok(client.get(f"/api/entries/{ids[11]}"))
    assert e11["status"] == "complete"
    assert e11["events"]["pushup"]["count"] == 45 and e11["events"]["crunch"]["passed"] is False
    assert e11["events"]["crunch"]["grade"] == 0 and e11["verdict"] == "fail"
    assert ok(client.get(f"/api/entries/{ids[13]}"))["verdict"] == "fail"       # run below 60 %
    assert ok(client.get(f"/api/entries/{ids[12]}"))["verdict"] == "no_standard"

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


def test_delete_someone_with_results_only_when_asked(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    ok(client.post("/api/run/start"))
    ok(client.post("/api/run/finish", json={"bib": "11"}), 201)
    att = ok(client.post("/api/attempts", json={"entry_id": e["id"], "event": "pushup", "lane": "1"}), 201)
    ok(client.post("/api/engine/report", json={"station": "pushup", "lanes": {"1": {
        "attempt_id": att["id"], "count": 1, "events": [{"index": 1, "valid": True, "t_start": 0, "t_end": 1}]}}}))
    ok(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 1, "rejected": 0}))
    # results exist: a plain remove is refused
    err(client.delete(f"/api/entries/{e['id']}"), 409, "cannot_remove")
    # with delete_results: gone, and so is the finish crossing
    assert client.delete(f"/api/entries/{e['id']}?delete_results=true").status_code == 204
    assert all(r["candidate"]["bib"] != "11" for r in ok(client.get("/api/entries")))
    assert ok(client.get("/api/run/crossings")) == []
    # the same person can be added again from scratch
    assert client.post("/api/entries", json={"external_id": "C-1", "name": "Ahmed Hassan", "bib": "11"}).status_code == 201
    assert entry_by_bib(client, 11)["status"] == "expected"


def test_bibs_are_assigned_automatically_when_asked(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))            # bibs 11..15
    # walk-in with no bib: the next free one
    a = ok(client.post("/api/entries", json={"external_id": "W-1", "name": "Walk In", "auto_bib": True}), 201)
    assert a["candidate"]["bib"] == "16"
    # a bib that is taken today: refused without auto_bib, replaced by the next one with it
    err(client.post("/api/entries", json={"external_id": "W-2", "name": "Second", "bib": "11"}), 409, "bib_in_use")
    b = ok(client.post("/api/entries", json={"external_id": "W-2", "name": "Second", "bib": "11", "auto_bib": True}), 201)
    assert b["candidate"]["bib"] == "17"
    # a free bib that was asked for is kept
    c = ok(client.post("/api/entries", json={"external_id": "W-3", "name": "Third", "bib": "40", "auto_bib": True}), 201)
    assert c["candidate"]["bib"] == "40"
    assert ok(client.post("/api/entries", json={"external_id": "W-4", "name": "Fourth", "auto_bib": True}), 201)["candidate"]["bib"] == "41"
    # without auto_bib and without a bib nothing is invented (CSV lists may assign bibs later)
    assert ok(client.post("/api/entries", json={"external_id": "W-5", "name": "Fifth"}), 201)["candidate"]["bib"] is None


def test_station_settings_defaults_update_and_audit(client):
    assert ok(client.get("/api/settings")) == {
        "auto_identify": True, "camera_background": "off", "camera_bg_places": ["weigh", "start", "photo"], "backdrop_color": "#8a949e", "face_match": "normal", "face_required": False, "walk_ins": True,
        "default_gender": "M", "default_unit": "combat", "bib_start": 1, "lanes": 3, "wave_size": 50, "start_countdown_s": 10, "cancel_window_s": 120, "site_name": "", "default_language": "en", "theme": "night",
        "admin_pin_set": False, "background_image": None}
    s = ok(client.put("/api/settings", json={"auto_identify": False, "bib_start": 500}))
    assert s["auto_identify"] is False and s["bib_start"] == 500 and s["default_gender"] == "M"
    assert ok(client.get("/api/settings"))["bib_start"] == 500          # stored, not just echoed
    err(client.put("/api/settings", json={"default_gender": "X"}), 422, "bad_input")
    err(client.put("/api/settings", json={"bib_start": 0}), 422, "bad_input")
    # automatic bibs start at bib_start
    ok(client.post("/api/entries/import", content=CSV.encode()))        # bibs 11..15
    assert ok(client.post("/api/entries", json={"external_id": "W-1", "name": "Walk In", "auto_bib": True}), 201)["candidate"]["bib"] == "500"
    assert ok(client.post("/api/entries", json={"external_id": "W-2", "name": "Walk Two", "auto_bib": True}), 201)["candidate"]["bib"] == "501"


def test_rules_change_applies_today_until_a_result_then_next_day(client):
    r0 = ok(client.get("/api/rules"))
    assert r0["bmi_max"] == 30 and r0["men"][0] == {"age_min": 17, "age_max": 22, "run_s": [1030, 980, 830],
                                                    "pushup": [34, 40, 56], "crunch": [40, 45, 59]}
    assert r0["unit_level"] == {"combat": 60, "admin": 60, "special": 70} and r0["women"] == []
    err(client.put("/api/rules", json={"bmi_min": 31}), 400, "bad_rules")             # min above max
    bad = [dict(b) for b in r0["men"]]; bad[0]["pushup"] = [40, 34, 56]
    err(client.put("/api/rules", json={"men": bad}), 400, "bad_rules")                # 70 % below 60 %
    bad = [dict(b) for b in r0["men"]]; bad[1]["age_min"] = 20
    err(client.put("/api/rules", json={"men": bad}), 400, "bad_rules")                # overlapping bands
    err(client.put("/api/rules", json={"unit_level": {"combat": 65, "admin": 60, "special": 70}}), 400, "bad_rules")
    r = ok(client.put("/api/rules", json={"bmi_max": 32}))
    assert r["applies_today"] is True and r["rules"]["bmi_max"] == 32 and r["rules"]["version"] == "v3"
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 92, "height_cm": 170, "accept": True}))  # BMI 31.8
    # today has a result now: a change waits for the next test day
    men = [dict(b) for b in r0["men"]]; men[1]["pushup"] = [37, 42, 58]
    r = ok(client.put("/api/rules", json={"bmi_max": 30, "men": men}))
    assert r["applies_today"] is False and r["rules"]["bmi_max"] == 32 and r["next"]["version"] == "v4"
    got = ok(client.get("/api/rules"))
    assert got["bmi_max"] == 32 and got["men"][1]["pushup"] == [36, 42, 58] and got["next"]["men"][1]["pushup"] == [37, 42, 58]


def test_standard_by_age_gender_and_unit(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    # a special-nature unit needs 70 %
    ok(client.post("/api/entries", json={"external_id": "S-1", "name": "Special One", "gender": "M",
                                         "dob": "2004-10-21", "bib": "21", "unit": "خاصة"}), 201)
    s1 = entry_by_bib(client, 21)
    assert s1["candidate"]["unit"] == "special"
    std = ok(client.post(f"/api/entries/{s1['id']}/standard", json={}))
    # born 2004-10-21, tested 2026-10-20: still 21, the day before the birthday
    assert std["age"] == 21 and std["band"] == "17-22" and std["level"] == 70
    assert std["needs"]["pushup"] == {"60": 34, "70": 40, "100": 56} and std["needs"]["run"]["100"] == 830
    # missing date of birth / gender: asked for at the scale, then kept
    ok(client.post("/api/entries", json={"external_id": "W-1", "name": "No Dob", "bib": "22"}), 201)
    w = entry_by_bib(client, 22)
    assert w["candidate"]["unit"] == "combat"                               # the station default
    err(client.post(f"/api/entries/{w['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}), 409, "gender_required")
    err(client.post(f"/api/entries/{w['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True, "gender": "M"}), 409, "dob_required")
    # outside the table: refused
    err(client.post(f"/api/entries/{w['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True,
                                                              "gender": "M", "dob": "1970-01-01"}), 409, "no_standard_for_age")
    w = ok(client.post(f"/api/entries/{w['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True,
                                                                 "gender": "M", "dob": "1984-03-01", "unit": "admin"}))
    assert w["standard"] == {"age": 42, "band": "41-45", "level": 60}
    assert w["candidate"]["dob"] == "1984-03-01" and w["candidate"]["unit"] == "admin"
    err(client.post("/api/entries", json={"external_id": "W-2", "name": "Bad Unit", "unit": "navy"}), 400, "bad_unit")


def test_admin_pin_guards_admin_actions(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    err(client.put("/api/settings/admin-pin", json={"pin": "12a"}), 400, "bad_pin")
    assert ok(client.put("/api/settings/admin-pin", json={"pin": "2468"})) == {"admin_pin_set": True}
    assert ok(client.get("/api/settings"))["admin_pin_set"] is True
    assert "2468" not in client.get("/api/settings").text
    e = entry_by_bib(client, 15)
    err(client.put("/api/settings", json={"lanes": 2}), 403, "admin_pin_required")
    err(client.put("/api/settings", json={"lanes": 2}, headers={"X-Admin-Pin": "1111"}), 403, "admin_pin_required")
    err(client.put("/api/rules", json={"bmi_max": 31}), 403, "admin_pin_required")
    err(client.delete(f"/api/entries/{e['id']}"), 403, "admin_pin_required")
    err(client.post("/api/entries/import", content=CSV.encode()), 403, "admin_pin_required")
    err(client.post("/api/settings/admin-pin/check", headers={"X-Admin-Pin": "1111"}), 403, "admin_pin_required")
    pin = {"X-Admin-Pin": "2468"}
    ok(client.post("/api/settings/admin-pin/check", headers=pin))
    assert ok(client.put("/api/settings", json={"lanes": 2}, headers=pin))["lanes"] == 2
    assert client.delete(f"/api/entries/{e['id']}", headers=pin).status_code == 204
    # the desk's daily work needs no PIN
    entry_by_bib(client, 11)
    ok(client.post("/api/entries", json={"external_id": "W-1", "name": "Walk In", "auto_bib": True, "walk_in": True}), 201)
    # removing the PIN needs the PIN
    assert ok(client.put("/api/settings/admin-pin", json={"pin": ""}, headers=pin)) == {"admin_pin_set": False}
    ok(client.put("/api/settings", json={"lanes": 3}))


def test_walk_ins_site_name_and_face_required(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    ok(client.put("/api/settings", json={"walk_ins": False, "site_name": "  Camp   North  "}))
    assert ok(client.get("/api/settings"))["site_name"] == "Camp North"
    err(client.post("/api/entries", json={"external_id": "W-1", "name": "Walk In", "walk_in": True}), 403, "walk_ins_off")
    ok(client.post("/api/entries", json={"external_id": "W-1", "name": "From the list"}), 201)   # Advanced add still works
    err(client.put("/api/settings", json={"lanes": 5}), 422, "bad_input")
    err(client.put("/api/settings", json={"face_match": "loose"}), 422, "bad_input")
    assert ok(client.put("/api/settings", json={"camera_background": "blur"}))["camera_background"] == "blur"
    err(client.put("/api/settings", json={"camera_background": "beach"}), 422, "bad_input")
    # face required: only people with a face on file are refused a typed weigh-in check-in
    ok(client.put("/api/settings", json={"face_required": True}))
    entry_by_bib(client, 11)   # no face on file: typed check-in is fine


def test_rules_change_keeps_a_day_from_before_rule_versions(client):
    from testpoint.api import models
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 72, "height_cm": 176, "accept": True}))
    db = client.app.state.sessionmaker()
    db.query(models.Day).update({"rules_json": None}); db.commit(); db.close()
    r = ok(client.put("/api/rules", json={"exercise_time_s": 90}))
    assert r["applies_today"] is False and ok(client.get("/api/rules"))["exercise_time_s"] == 120


def test_edit_candidate_regrades_saved_results(client, monkeypatch):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)                                   # born 1999-04-12: 27, band 23-30

    def edit(body, **kw):
        return client.patch(f"/api/entries/{e['id']}/candidate", json=body, **kw)

    # before weigh-in: a plain correction
    got = ok(edit({"name_ar": "  أحمد  حسن ", "bib": "011"}))
    assert got["candidate"]["name_ar"] == "أحمد حسن" and got["candidate"]["bib"] == "11"
    err(edit({"bib": "12"}), 409, "bib_in_use")
    err(edit({"external_id": "c-2"}), 409, "external_id_in_use")
    err(edit({"name": "  "}), 400, "missing_fields")
    err(edit({"unit": "navy"}), 400, "bad_unit")
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 72, "height_cm": 176, "accept": True}))
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    ok(client.post("/api/run/start"))
    err(edit({"bib": "99"}), 409, "bib_locked")                   # on the run: the bib is what the cameras read
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=18))
    ok(client.post("/api/run/finish", json={"bib": "11"}), 201)    # 18:00: band 23-30 needs 18:30 for 60 %
    att = ok(client.post("/api/attempts", json={"entry_id": e["id"], "event": "pushup", "lane": "1"}), 201)
    ok(client.post(f"/api/attempts/{att['id']}/finish", json={"count": 35}))   # 23-30 needs 36: fail
    cur = ok(client.get(f"/api/entries/{e['id']}"))
    assert cur["status"] == "run_passed" and cur["run_grade"] == 60 and cur["events"]["pushup"]["grade"] == 0
    # wrong date of birth: really 44 -> band 41-45 (run 70 % 19:00, push-ups 70 % 33)
    got = ok(edit({"dob": "1982-09-23"}))
    assert got["standard"] == {"age": 44, "band": "41-45", "level": 60}
    assert got["run_grade"] == 70 and got["events"]["pushup"]["grade"] == 70 and got["events"]["pushup"]["passed"] is True
    # special unit: 70 % needed; the run (70) and push-ups (70) still pass
    got = ok(edit({"unit": "special"}))
    assert got["standard"]["level"] == 70 and got["status"] == "run_passed"
    # an age outside the table is refused and nothing changes
    err(edit({"dob": "1960-01-01"}), 409, "no_standard_for_age")
    assert ok(client.get(f"/api/entries/{e['id']}"))["candidate"]["dob"] == "1982-09-23"
    err(edit({"dob": None}), 409, "dob_required")
    # admin PIN guards edits
    ok(client.put("/api/settings/admin-pin", json={"pin": "2468"}))
    err(edit({"name_ar": "x"}), 403, "admin_pin_required")
    ok(edit({"name_ar": "x"}, headers={"X-Admin-Pin": "2468"}))


def test_camera_background_colour_and_picture(client):
    s = ok(client.put("/api/settings", json={"camera_background": "backdrop", "backdrop_color": "#1F3A5F"}))
    assert s["camera_background"] == "backdrop" and s["backdrop_color"] == "#1F3A5F"
    err(client.put("/api/settings", json={"backdrop_color": "navy"}), 422, "bad_input")
    assert client.get("/api/settings/background-image").status_code == 404
    jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\0" * 100
    err(client.put("/api/settings/background-image", content=b"GIF89a....", headers={"content-type": "image/gif"}), 400, "bad_image")
    assert client.put("/api/settings/background-image", content=jpeg, headers={"content-type": "image/jpeg"}).status_code == 204
    s = ok(client.put("/api/settings", json={"camera_background": "image"}))
    assert s["camera_background"] == "image" and s["background_image"]          # a version string
    r = client.get(f"/api/settings/background-image?v={s['background_image']}")
    assert r.status_code == 200 and r.content == jpeg and r.headers["content-type"] == "image/jpeg"
    assert client.delete("/api/settings/background-image").status_code == 204
    assert ok(client.get("/api/settings"))["background_image"] is None
    # admin PIN guards the picture too
    ok(client.put("/api/settings/admin-pin", json={"pin": "2468"}))
    err(client.put("/api/settings/background-image", content=jpeg), 403, "admin_pin_required")


def test_waves_time_from_their_own_start_and_close_themselves(client, monkeypatch):
    rows = "external_id,name,bib,gender,dob\n" + "".join(f"W-{b},Runner {b},{b},M,2000-01-01\n" for b in range(31, 36))
    ok(client.post("/api/entries/import", content=rows.encode()))
    ids = {b: entry_by_bib(client, b)["id"] for b in range(31, 36)}
    for b in ids:
        ok(client.post(f"/api/entries/{ids[b]}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
    ok(client.put("/api/settings", json={"wave_size": 2}))
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    ok(client.post(f"/api/entries/{ids[31]}/line-up"))
    ok(client.post(f"/api/entries/{ids[32]}/line-up", params={"by": "face"}))
    err(client.post(f"/api/entries/{ids[33]}/line-up"), 409, "wave_full")
    # a 10 s countdown: the time starts at the horn, every device shows the same countdown
    w1 = ok(client.post("/api/run/start", json={"countdown_s": 10}))
    assert w1["wave"] == 1 and w1["run_started_at"].startswith((t0 + timedelta(seconds=10)).isoformat()[:19])
    err(client.post("/api/run/finish", json={"bib": "31"}), 409, "before_start")
    # wave 2 lines up and starts 5 minutes later
    clock["t"] = t0 + timedelta(minutes=5)
    ok(client.post(f"/api/entries/{ids[33]}/line-up"))
    ok(client.post(f"/api/entries/{ids[34]}/line-up"))
    ok(client.post("/api/run/start"))
    # each runner is timed from their own wave
    clock["t"] = t0 + timedelta(minutes=20)
    assert ok(client.post("/api/run/finish", json={"bib": "31"}), 201)["time_s"] == 1190.0    # 20:00 - 0:10
    assert ok(client.post("/api/run/finish", json={"bib": "33"}), 201)["time_s"] == 900.0     # 20:00 - 5:00
    # an unknown crossing is timed from the newest wave; assigning it to a wave-1 runner re-times it
    u = ok(client.post("/api/run/finish", json={"bib": None}), 201)
    assert u["needs_review"] and u["time_s"] == 900.0
    r = ok(client.post(f"/api/run/crossings/{u['id']}/resolve", json={"entry_id": ids[32]}))
    assert r["time_s"] == 1190.0
    # wave 1 closes itself after the cut-off (25:00) and the grace period: 34 never finished
    clock["t"] = t0 + timedelta(seconds=10 + 1500 + 121)
    st = ok(client.get("/api/state"))
    assert [(w["number"], bool(w["closed_at"]), w["closed_by"]) for w in st["waves"]] == [(1, True, "auto"), (2, False, None)]
    clock["t"] = t0 + timedelta(minutes=5, seconds=1500 + 121)
    st = ok(client.get("/api/state"))
    assert st["waves"][1]["closed_by"] == "auto" and st["waves"][1]["dnf"] == 1
    assert ok(client.get(f"/api/entries/{ids[34]}"))["status"] == "dnf"
    # 35 never lined up: still waiting, can run in a later wave
    assert ok(client.get(f"/api/entries/{ids[35]}"))["status"] == "accepted"
    assert ok(client.post(f"/api/entries/{ids[35]}/line-up"))["wave"] == 3


def test_wave_with_a_pending_review_waits_for_the_examiner(client, monkeypatch):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 72, "height_cm": 176, "accept": True}))
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    ok(client.post("/api/run/start"))
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=10))
    ok(client.post("/api/run/finish", json={"bib": None}), 201)
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(hours=1))
    st = ok(client.get("/api/state"))
    assert st["waves"][0]["closed_at"] is None and st["waves"][0]["review"] == 1


def test_wave_closes_itself_once_everyone_is_in(client, monkeypatch):
    rows = "external_id,name,bib,gender,dob\n" + "".join(f"A-{b},Runner {b},{b},M,2000-01-01\n" for b in (41, 42, 43))
    ok(client.post("/api/entries/import", content=rows.encode()))
    ids = {b: entry_by_bib(client, b)["id"] for b in (41, 42, 43)}
    for b in ids:
        ok(client.post(f"/api/entries/{ids[b]}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
        ok(client.post(f"/api/entries/{ids[b]}/line-up"))
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    ok(client.post("/api/run/start"))
    clock["t"] = t0 + timedelta(minutes=15)
    ok(client.post("/api/run/finish", json={"bib": "41"}), 201)
    ok(client.post(f"/api/entries/{ids[42]}/dnf", json={"reason": "injured"}))
    assert ok(client.get("/api/state"))["waves"][0]["settles_at"] is None          # 43 still out
    clock["t"] = t0 + timedelta(minutes=16)
    ok(client.post("/api/run/finish", json={"bib": "43"}), 201)
    w = ok(client.get("/api/state"))["waves"][0]
    assert w["closed_at"] is None and w["settles_at"]                           # all in: a minute to fix a wrong bib
    clock["t"] = t0 + timedelta(minutes=16, seconds=61)
    w = ok(client.get("/api/state"))["waves"][0]
    assert w["closed_at"] and w["closed_by"] == "auto"
    assert ok(client.get(f"/api/entries/{ids[42]}"))["status"] == "dnf"


def test_camera_background_places(client):
    s = ok(client.put("/api/settings", json={"camera_bg_places": ["weigh", "finish"]}))
    assert s["camera_bg_places"] == ["weigh", "finish"]
    assert ok(client.put("/api/settings", json={"camera_bg_places": []}))["camera_bg_places"] == []
    err(client.put("/api/settings", json={"camera_bg_places": ["kitchen"]}), 422, "bad_input")



def test_defaults_for_reset(client):
    ok(client.put("/api/settings", json={"wave_size": 20, "theme": "falcon"}))
    d = ok(client.get("/api/settings/defaults"))
    assert d["settings"]["wave_size"] == 50 and d["settings"]["theme"] == "night"
    assert d["rules"]["men"] and d["rules"]["unit_level"]["combat"] == 60
