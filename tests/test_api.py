"""End-to-end test of the candidate journey through the HTTP API (in-memory SQLite)."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import app as app_mod
from testpoint.api import models
from testpoint.api.app import create_app
from runhelp import start_lined_up

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

    # line-up: 15 is late (starts later); rejected 14 is refused
    for b in (11, 12, 13):
        ok(client.post(f"/api/entries/{ids[b]}/line-up"))
    err(client.post(f"/api/entries/{ids[14]}/line-up"), 409, "rejected_at_weigh_in")

    # each starts on their own GO (no countdown here), with a controllable clock
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    assert ok(start_lined_up(client))["status"] == "running"
    assert start_lined_up(client) is None                      # nobody left at the line
    st = ok(client.get("/api/state"))
    assert st["counts"]["running"] == 3 and st["counts"]["accepted"] == 1 and st["counts"]["rejected"] == 1
    assert len(st["runners"]) == 3 and st["starting"] is None
    # the late one lines up while the others run
    assert ok(client.post(f"/api/entries/{ids[15]}/line-up"))["status"] == "lined_up"

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

    # exact percentages (age 27, band 23-30: push-ups 36/42/58, crunches 42/47/61, run 18:30/17:40/15:10)
    e11 = ok(client.get(f"/api/entries/{ids[11]}"))
    assert e11["events"]["pushup"]["percent"] == round(70 + 30 * (45 - 42) / (58 - 42), 1)   # 75.6
    assert e11["events"]["crunch"]["percent"] == round(60 * 10 / 42, 1)               # below 60 %: in proportion
    assert e11["run_percent"] == 100.0                                                # 14:12, under 15:10
    assert e11["overall_percent"] == round((100.0 + 75.6 + 14.3) / 3, 1)
    r = ok(client.get(f"/api/entries/{ids[11]}/results"))
    assert r["overall_percent"] == e11["overall_percent"] and [t["percent"] for t in r["tests"]] == [100.0, 75.6, 14.3]

    # an examiner corrects a saved result: a reason, regraded, the admin PIN when one is set
    err(client.post(f"/api/entries/{ids[11]}/correct", json={"test": "crunch", "value": 40, "reason": " "}), 409, "override_reason_required")
    e = ok(client.post(f"/api/entries/{ids[11]}/correct", json={"test": "crunch", "value": 80, "reason": "camera missed reps"}))
    assert e["events"]["crunch"]["count"] == 80 and e["events"]["crunch"]["passed"] is True and e["verdict"] == "pass"
    e = ok(client.post(f"/api/entries/{ids[11]}/correct", json={"test": "run", "value": 1200, "reason": "stopwatch"}))
    assert e["run_time_s"] == 1200 and e["status"] == "complete" and e["run_grade"] < 60 and e["verdict"] == "fail"
    err(client.post(f"/api/entries/{ids[15]}/correct", json={"test": "run", "value": 900, "reason": "x"}), 409, "no_result")
    client.put("/api/settings/admin-pin", json={"pin": "2468"})
    err(client.post(f"/api/entries/{ids[11]}/correct", json={"test": "run", "value": 852, "reason": "back"}), 403, "admin_pin_required")
    assert client.post(f"/api/entries/{ids[11]}/correct", json={"test": "run", "value": 852, "reason": "back"},
                       headers={"X-Admin-Pin": "2468"}).status_code == 200
    client.put("/api/settings/admin-pin", json={"pin": ""}, headers={"X-Admin-Pin": "2468"})

    # reset one result so that test is done again: the crunches, then the run
    e = ok(client.post(f"/api/entries/{ids[11]}/reset", json={"test": "crunch"}))
    assert "crunch" not in e["events"] and e["events"]["pushup"]["count"] == 45 and e["status"] == "run_passed"
    err(client.post(f"/api/entries/{ids[11]}/reset", json={"test": "crunch"}), 409, "no_result")
    mine = [c["id"] for c in ok(client.get("/api/run/crossings")) if c["entry_id"] == ids[11]]
    assert mine
    e = ok(client.post(f"/api/entries/{ids[11]}/reset", json={"test": "run"}))
    assert e["status"] == "accepted" and e["run_time_s"] is None and e["events"]["pushup"]["count"] == 45
    # its crossing is gone (not left as an unnamed finisher); the audit log keeps it
    assert not {c["id"] for c in ok(client.get("/api/run/crossings"))} & set(mine)
    err(client.post(f"/api/entries/{ids[11]}/reset", json={"test": "run"}), 409, "no_result")

    # a cancelled attempt can be retried
    att3 = ok(client.post("/api/attempts", json={"entry_id": ids[12], "event": "pushup"}), 201)
    assert client.post(f"/api/attempts/{att3['id']}/cancel").status_code == 204
    ok(client.post("/api/attempts", json={"entry_id": ids[12], "event": "pushup"}), 201)


def test_runner_not_running_cannot_be_dnf_before_close(client, monkeypatch):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    ok(start_lined_up(client))
    assert ok(client.post("/api/run/close"))["dnf"] == 1
    assert ok(client.get(f"/api/entries/{e['id']}"))["status"] == "dnf"
    assert ok(client.post("/api/run/close"))["dnf"] == 0


def test_delete_someone_with_results_only_when_asked(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    ok(start_lined_up(client))
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
        "auto_identify": True, "camera_background": "off", "camera_bg_places": ["weigh", "start", "photo"], "backdrop_color": "#8a949e", "face_match": "normal", "face_required": False, "face_fresh_min": 120, "start_rolling": True, "start_line_on": True, "start_setup_shown": True, "finish_line_on": True, "finish_setup_shown": True, "station_choose_first": True, "walk_ins": True,
        "default_gender": "M", "default_unit": "combat", "bib_start": 1, "lanes": 3, "station_stop_after_s": 20, "station_first_rep": True, "station_out_s": 3, "finish_shutter": "auto", "go_styles": {}, "go_effect": "racing", "go_dim": 60, "go_size": 100, "go_speed": 100, "go_duration": 1.7, "go_opacity": 100, "go_lines": 100, "go_lines_count": 220, "go_lines_opacity": 100, "start_countdown_s": 0, "cancel_window_s": 120, "finish_face_threshold": 0.0, "site_name": "", "default_language": "en", "trouble_run_cutoff_s": 0, "count_tuning": {}, "evidence_keep_days": 30, "evidence_max_gb": 0, "evidence_min_free_gb": 5, "evidence_held_days": [], "camera_sources": {"weigh": ["webcam", "video", "photo", "ipcam"], "start": ["webcam", "video", "photo", "ipcam"], "results": ["webcam", "video", "photo"], "finish": ["webcam", "video", "ipcam"], "station": ["webcam", "video", "ipcam"]}, "theme": "night",
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
    ok(start_lined_up(client))
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


def test_every_runner_is_timed_from_their_own_go(client, monkeypatch):
    ok(client.put("/api/settings", json={"start_countdown_s": 3}))   # with a 3-2-1 (the default is none)
    rows = "external_id,name,bib,gender,dob\n" + "".join(f"W-{b},Runner {b},{b},M,2000-01-01\n" for b in range(31, 36))
    ok(client.post("/api/entries/import", content=rows.encode()))
    ids = {b: entry_by_bib(client, b)["id"] for b in range(31, 36)}
    for b in ids:
        ok(client.post(f"/api/entries/{ids[b]}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    ok(client.post(f"/api/entries/{ids[31]}/line-up"))
    # 3-2-1: the clock starts at GO, every screen shows the same countdown
    e31 = ok(client.post(f"/api/entries/{ids[31]}/start", json={"countdown_s": 3}))
    assert e31["status"] == "running" and e31["run_started_at"].startswith((t0 + timedelta(seconds=3)).isoformat()[:19])
    st = ok(client.get("/api/state"))
    assert st["starting"]["entry_id"] == ids[31] and [r["entry_id"] for r in st["runners"]] == [ids[31]]
    # one runner at a time: the next waits for this GO
    ok(client.post(f"/api/entries/{ids[32]}/line-up", params={"by": "face"}))
    err(client.post(f"/api/entries/{ids[32]}/start", json={"countdown_s": 3}), 409, "start_in_progress")
    err(client.post("/api/run/finish", json={"bib": "31"}), 409, "before_start")
    # after the GO the next one starts; the default countdown is the setting (3 s)
    clock["t"] = t0 + timedelta(seconds=10)
    e32 = ok(client.post(f"/api/entries/{ids[32]}/start"))
    assert e32["run_started_at"].startswith((t0 + timedelta(seconds=13)).isoformat()[:19])
    # starting someone not weighed in / not at the line is refused by the gate
    err(client.post(f"/api/entries/{ids[31]}/start"), 409, "still_running")
    clock["t"] = t0 + timedelta(minutes=5)
    ok(client.post(f"/api/entries/{ids[33]}/start", json={"countdown_s": 0}))   # straight from accepted
    ok(client.post(f"/api/entries/{ids[34]}/start", json={"countdown_s": 0}))
    # each runner is timed from their own GO
    clock["t"] = t0 + timedelta(minutes=20)
    assert ok(client.post("/api/run/finish", json={"bib": "31"}), 201)["time_s"] == 1197.0    # 20:00 - 0:03
    assert ok(client.post("/api/run/finish", json={"bib": "33"}), 201)["time_s"] == 900.0     # 20:00 - 5:00
    # an unknown crossing waits for review; assigned to 32 it is timed from 32's GO
    u = ok(client.post("/api/run/finish", json={"bib": None}), 201)
    assert u["needs_review"] and u["entry_id"] is None
    r = ok(client.post(f"/api/run/crossings/{u['id']}/resolve", json={"entry_id": ids[32]}))
    assert r["time_s"] == 1187.0                                                             # 20:00 - 0:13
    # 34 never finishes: DNF by itself after their own cut-off (25:00) and the grace (2 min)
    clock["t"] = t0 + timedelta(minutes=5, seconds=1500 + 119)
    ok(client.get("/api/state"))
    assert ok(client.get(f"/api/entries/{ids[34]}"))["status"] == "running"
    clock["t"] = t0 + timedelta(minutes=5, seconds=1500 + 121)
    st = ok(client.get("/api/state"))
    assert ok(client.get(f"/api/entries/{ids[34]}"))["status"] == "dnf" and st["runners"] == []
    # 35 never lined up: still waiting, can still run
    assert ok(client.get(f"/api/entries/{ids[35]}"))["status"] == "accepted"
    assert ok(client.post(f"/api/entries/{ids[35]}/line-up"))["status"] == "lined_up"


def test_a_pending_review_holds_the_cutoff_dnf(client, monkeypatch):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 72, "height_cm": 176, "accept": True}))
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    ok(start_lined_up(client))
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=10))
    c = ok(client.post("/api/run/finish", json={"bib": None}), 201)
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(hours=1))
    st = ok(client.get("/api/state"))
    # the unread crossing may be theirs: nobody becomes DNF while it waits
    assert st["review_queue"] == 1 and ok(client.get(f"/api/entries/{e['id']}"))["status"] == "running"
    ok(client.post(f"/api/run/crossings/{c['id']}/resolve", json={"entry_id": e["id"]}))
    assert ok(client.get(f"/api/entries/{e['id']}"))["run_time_s"] == 600.0


def test_cancel_a_start_and_close_the_run(client, monkeypatch):
    rows = "external_id,name,bib,gender,dob\n" + "".join(f"A-{b},Runner {b},{b},M,2000-01-01\n" for b in (41, 42, 43))
    ok(client.post("/api/entries/import", content=rows.encode()))
    ids = {b: entry_by_bib(client, b)["id"] for b in (41, 42, 43)}
    for b in ids:
        ok(client.post(f"/api/entries/{ids[b]}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
        ok(client.post(f"/api/entries/{ids[b]}/line-up"))
    t0 = models.utcnow()
    clock = {"t": t0}
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: clock["t"])
    ok(start_lined_up(client))
    # the wrong person was started: back to the line, clock cleared
    back = ok(client.post(f"/api/entries/{ids[43]}/start/cancel"))
    assert back["status"] == "lined_up" and back["run_started_at"] is None
    clock["t"] = t0 + timedelta(minutes=15)
    ok(client.post("/api/run/finish", json={"bib": "41"}), 201)
    ok(client.post(f"/api/entries/{ids[42]}/dnf", json={"reason": "injured"}))
    # after the cancel window only an administrator can cancel a start
    err(client.post(f"/api/entries/{ids[41]}/start/cancel"), 409, "not_running")
    ok(client.post(f"/api/entries/{ids[43]}/start", json={"countdown_s": 0}))
    clock["t"] = t0 + timedelta(minutes=18)
    err(client.post(f"/api/entries/{ids[43]}/start/cancel"), 409, "too_late_to_cancel")
    # end of the run: whoever is still out becomes DNF
    assert ok(client.post("/api/run/close"))["dnf"] == 1
    assert ok(client.get(f"/api/entries/{ids[43]}"))["status"] == "dnf"
    assert ok(client.post("/api/run/close"))["dnf"] == 0


def test_old_days_get_each_runners_start_from_their_wave(tmp_path):
    """A database from the days of group starts: every runner's start is filled in from their wave."""
    import sqlite3
    db = tmp_path / "old.db"
    models.make_sessionmaker(f"sqlite:///{db}")          # today's schema, then make it look old
    con = sqlite3.connect(db)
    con.execute("INSERT INTO day (id, date, rules_version) VALUES (1, '2026-10-01', 'v1')")
    con.execute("INSERT INTO wave (id, day_id, number, started_at, created_at) VALUES (7, 1, 1, '2026-10-01 07:00:00', '2026-10-01 06:00:00')")
    con.execute("INSERT INTO candidate (id, external_id, name, bib, created_at) VALUES (1, 'X', 'Old Runner', '5', '2026-10-01 06:00:00')")
    con.execute("INSERT INTO entry (id, day_id, candidate_id, status, wave_id, run_time_s) VALUES (1, 1, 1, 'run_passed', 7, 900.0)")
    con.commit()
    con.close()
    Session = models.make_sessionmaker(f"sqlite:///{db}")
    with Session() as s:
        e = s.get(models.Entry, 1)
        assert e.run_started_at == datetime(2026, 10, 1, 7, 0) and e.run_time_s == 900.0


def test_camera_background_places(client):
    s = ok(client.put("/api/settings", json={"camera_bg_places": ["weigh", "finish"]}))
    assert s["camera_bg_places"] == ["weigh", "finish"]
    assert ok(client.put("/api/settings", json={"camera_bg_places": []}))["camera_bg_places"] == []
    err(client.put("/api/settings", json={"camera_bg_places": ["kitchen"]}), 422, "bad_input")



def test_defaults_for_reset(client):
    ok(client.put("/api/settings", json={"start_countdown_s": 5, "theme": "falcon"}))
    d = ok(client.get("/api/settings/defaults"))
    assert d["settings"]["start_countdown_s"] == 0 and d["settings"]["theme"] == "night"
    assert d["rules"]["men"] and d["rules"]["unit_level"]["combat"] == 60



def test_troubleshooting_run_cutoff_judges_but_leaves_the_rules(client):
    ok(client.put("/api/settings", json={"trouble_run_cutoff_s": 60}))
    assert ok(client.get("/api/rules"))["run_cutoff_s"] == 25 * 60          # the rules as saved
    ok(client.put("/api/settings", json={"trouble_run_cutoff_s": 0}))
    err(client.put("/api/settings", json={"trouble_run_cutoff_s": -5}), 422, "bad_input")


def test_reset_today_keeps_people_and_can_empty_the_list(client):
    e = ok(client.post("/api/entries", json={"external_id": "A-1", "name": "Alpha", "gender": "M", "dob": "1999-01-01"}), 201)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
    today = ok(client.get("/api/state"))["date"]
    err(client.post("/api/day/reset", json={"confirm": "yes"}), 400, "confirm_date")
    r = ok(client.post("/api/day/reset", json={"confirm": today}))
    assert r["people"] == 1 and r["with_results"] == 1
    rows = ok(client.get("/api/entries"))
    assert len(rows) == 1 and rows[0]["status"] == "expected" and rows[0]["bmi"] is None
    ok(client.post("/api/day/reset", json={"confirm": today, "keep_list": False}))
    assert ok(client.get("/api/entries")) == []
    assert ok(client.get("/api/candidates"))["total"] == 1                  # still in the register





def test_counting_rules_can_be_tuned_within_limits(client):
    s = ok(client.put("/api/settings", json={"count_tuning": {"pushup": {"down_elbow_deg": 90, "min_body_line_deg": 145}}}))
    assert s["count_tuning"] == {"pushup": {"down_elbow_deg": 90.0, "min_body_line_deg": 145.0}}
    err(client.put("/api/settings", json={"count_tuning": {"pushup": {"down_elbow_deg": 150}}}), 400, "bad_tuning")      # out of range
    err(client.put("/api/settings", json={"count_tuning": {"pushup": {"down_elbow_deg": 130, "up_elbow_deg": 140}}}), 400, "bad_tuning")
    err(client.put("/api/settings", json={"count_tuning": {"pushup": {"min_conf": 0.1}}}), 400, "bad_tuning")         # not tunable
    d = ok(client.get("/api/settings/defaults"))["tuning"]
    assert d["pushup"]["down_elbow_deg"]["default"] == 95 and d["crunch"]["up_torso_deg"]["min"] == 3
    assert ok(client.put("/api/settings", json={"count_tuning": {}}))["count_tuning"] == {}


def test_results_card_lists_every_test_against_the_level(client, monkeypatch):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 72, "height_cm": 176, "accept": True}))
    r = ok(client.get(f"/api/entries/{e['id']}/results"))
    assert r["verdict"] == "pending" and [t["test"] for t in r["tests"]] == ["run", "pushup", "crunch"]
    assert all(t["value"] is None for t in r["tests"]) and not r["complete"]
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    start_lined_up(client)
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=14))
    ok(client.post("/api/run/finish", json={"bib": "11"}), 201)
    for ev, n in (("pushup", 45), ("crunch", 10)):
        a = ok(client.post("/api/attempts", json={"entry_id": e["id"], "event": ev, "lane": "1"}), 201)
        ok(client.post(f"/api/attempts/{a['id']}/finish", json={"count": n}))
    r = ok(client.get(f"/api/entries/{e['id']}/results"))
    assert r["complete"] and r["verdict"] == "fail" and r["level"] == 60
    run, push, crunch = r["tests"]
    assert run["value"] == 840.0 and run["passed"] is True
    assert (push["value"], push["passed"]) == (45, True) and (crunch["value"], crunch["passed"]) == (10, False)
    assert r["entry"]["candidate"]["bib"] == "11"


def test_redo_takes_a_saved_exercise_back(client, monkeypatch):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    e = entry_by_bib(client, 11)
    ok(client.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 72, "height_cm": 176, "accept": True}))
    ok(client.post(f"/api/entries/{e['id']}/line-up"))
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    start_lined_up(client)
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=14))
    ok(client.post("/api/run/finish", json={"bib": "11"}), 201)
    ran = ok(client.get(f"/api/entries/{e['id']}/results"))["status"]
    ids = {}
    for ev, n in (("pushup", 0), ("crunch", 50)):
        a = ok(client.post("/api/attempts", json={"entry_id": e["id"], "event": ev, "lane": "1"}), 201)
        ok(client.post(f"/api/attempts/{a['id']}/finish", json={"count": n}))
        ids[ev] = a["id"]
    r = ok(client.get(f"/api/entries/{e['id']}/results"))
    assert r["complete"] and [t.get("attempt_id") for t in r["tests"][1:]] == [ids["pushup"], ids["crunch"]]

    ok(client.post(f"/api/attempts/{ids['pushup']}/redo"))
    r = ok(client.get(f"/api/entries/{e['id']}/results"))
    run, push, crunch = r["tests"]
    assert not r["complete"] and r["status"] == ran            # the run outcome is back, its time kept
    assert run["value"] == 840.0 and push["value"] is None and crunch["value"] == 50
    assert client.post(f"/api/attempts/{ids['pushup']}/redo").status_code == 404      # already gone
    # they can do it again
    a = ok(client.post("/api/attempts", json={"entry_id": e["id"], "event": "pushup", "lane": "1"}), 201)
    assert client.post(f"/api/attempts/{a['id']}/redo").status_code == 409             # still running
    ok(client.post(f"/api/attempts/{a['id']}/finish", json={"count": 45}))
    assert ok(client.get(f"/api/entries/{e['id']}/results"))["complete"]


def test_upload_a_video_for_the_camera_screens(client, monkeypatch, tmp_path):
    monkeypatch.setenv("TESTPOINT_VIDEO_DIRS", str(tmp_path))
    r = ok(client.post("/api/clips/crunch", params={"name": "..\\my clip?.MP4"}, content=b"abc"))
    assert r["label"] == "crunch/my clip_.MP4" and (tmp_path / "crunch" / "my clip_.MP4").read_bytes() == b"abc"
    # the same name again does not overwrite
    assert ok(client.post("/api/clips/crunch", params={"name": "my clip?.MP4"}, content=b"d"))["label"] == "crunch/my clip_-2.MP4"
    err(client.post("/api/clips/crunch", params={"name": "evil.exe"}, content=b"x"), 400, "bad_video")
    err(client.post("/api/clips/crunch", params={"name": "empty.mp4"}, content=b""), 400, "bad_video")
    err(client.post("/api/clips/secret", params={"name": "a.mp4"}, content=b"x"), 400, "unknown_event")
    assert not list((tmp_path / "crunch").glob("*.part"))


def test_station_manual_start_needs_a_camera(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))
    err(client.post("/api/stationflow/start", json={"query": "11"}), 409, "camera_off")
    err(client.post("/api/stationflow/go"), 409, "not_in_position")


def test_station_video_player_commands_ride_on_the_engine_reply(client):
    report = {"station": "pushup", "lanes": {"1": {}}, "player": {"t": 1.5, "duration": 12.0, "paused": False}}
    first = ok(client.post("/api/engine/report", json=report))["player"]
    assert first.get("action") is None
    sent = ok(client.post("/api/stationcam/player", json={"action": "seek", "t": 4.0}))
    cmd = ok(client.post("/api/engine/report", json=report))["player"]
    assert cmd["seq"] == sent["seq"] > first["seq"] and cmd["action"] == "seek" and cmd["t"] == 4.0
    err(client.post("/api/stationcam/player", json={"action": "seek"}), 400, "bad_player")
    assert ok(client.get("/api/lanes?station=pushup"))["engine"]["player"]["duration"] == 12.0


def test_evidence_retention_settings_storage_and_cleanup(monkeypatch, tmp_path):
    import os
    import time
    monkeypatch.setenv("TESTPOINT_RETENTION_EVERY_S", "0")
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", tz="Asia/Dubai", web_dir="does-not-exist",
                     today=lambda: "2026-10-20", evidence_dir=tmp_path)
    for rel, age in (("2026-09-01/7/full.mp4", 49), ("2026-09-02/8/full.mp4", 48), ("2026-10-19/9/full.mp4", 1)):
        f = tmp_path / rel
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x" * 1000)
        t = time.time() - age * 86400
        os.utime(f, (t, t))
    with TestClient(app) as c:
        err(c.put("/api/settings", json={"evidence_held_days": ["1 Sept"]}), 422, "bad_input")
        ok(c.put("/api/settings", json={"evidence_held_days": ["2026-09-02"], "evidence_min_free_gb": 0}))
        st = ok(c.get("/api/storage/evidence"))
        assert st["total_bytes"] == 3000 and st["next_cleanup"] == {"items": 1, "bytes": 1000, "by_age": 1, "by_space": 0}
        assert [d["day"] for d in st["days"]] == ["2026-10-19", "2026-09-02", "2026-09-01"] and st["days"][1]["held"]
        done = ok(c.post("/api/storage/evidence/cleanup"))
        assert done["deleted"] == 1 and not (tmp_path / "2026-09-01").exists() and (tmp_path / "2026-09-02").exists()
        with app.state.sessionmaker() as db:
            assert db.query(models.AuditLog).filter_by(action="evidence_cleanup").count() == 1


def test_recordings_of_a_day_person_by_person(monkeypatch, tmp_path):
    monkeypatch.setenv("TESTPOINT_RETENTION_EVERY_S", "0")
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", tz="Asia/Dubai", web_dir="does-not-exist",
                     today=lambda: "2026-10-20", evidence_dir=tmp_path)
    with TestClient(app) as c:
        ok(c.post("/api/entries/import", content=CSV.encode()))
        e = entry_by_bib(c, 11)
        ok(c.post(f"/api/entries/{e['id']}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True}))
        ok(c.post(f"/api/entries/{e['id']}/line-up"))
        ok(start_lined_up(c))
        ok(c.post("/api/run/finish", json={"bib": "11"}), 201)
        att = ok(c.post("/api/attempts", json={"entry_id": e["id"], "event": "pushup", "lane": "1"}), 201)
        folder = tmp_path / "2026-10-20" / str(att["id"])
        folder.mkdir(parents=True)
        (folder / "full.mp4").write_bytes(b"v")
        (folder / "rep-001.mp4").write_bytes(b"v")
        ok(c.post("/api/engine/report", json={"station": "pushup", "lanes": {"1": {"attempt_id": att["id"], "count": 2, "events": [
            {"index": 1, "valid": True, "t_start": 0, "t_end": 1, "clip": f"2026-10-20/{att['id']}/rep-001.mp4"},
            {"index": 2, "valid": False, "faults": ["knee"], "t_start": 1, "t_end": 2, "clip": f"2026-10-20/{att['id']}/rep-002.mp4"}]}}}))
        (tmp_path / "start").mkdir()
        (tmp_path / "start" / f"20261020-081500-entry{e['id']}.mp4").write_bytes(b"v")
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "000001-t1.mp4").write_bytes(b"v")      # a crossing nobody was given
        import os
        from datetime import datetime as dt
        t = dt(2026, 10, 20, 9).timestamp()
        os.utime(tmp_path / "run" / "000001-t1.mp4", (t, t))
        err(c.get("/api/recordings?day=20-10-2026"), 400, "bad_day")
        r = ok(c.get("/api/recordings?day=2026-10-20"))
        me = next(p for p in r["people"] if p["entry_id"] == e["id"])
        a = me["attempts"][0]
        assert a["full"] == f"2026-10-20/{att['id']}/full.mp4"
        assert [x["clip"] for x in a["reps"]] == [f"2026-10-20/{att['id']}/rep-001.mp4", None]   # rep 2's clip is gone
        assert me["start"][0]["at"] == "2026-10-20T08:15:00" and me["recordings"] == 3
        assert r["people"][0]["entry_id"] == e["id"]           # people with recordings first
        assert [o["folder"] for o in r["other"]] == ["run/000001-t1.mp4"]
        days = ok(c.get("/api/recordings/days"))
        assert days[0]["day"] == "2026-10-20" and days[0]["people"] == 5



def test_import_gives_a_bib_to_rows_without_one(client):
    ok(client.post("/api/entries/import", content=CSV.encode()))                     # bibs 11..
    rows = "external_id,name,bib\nW-1,No Bib,\nW-2,Own Bib,77\n"
    r = ok(client.post("/api/entries/import", content=rows.encode()))
    assert r["added"] == 2
    bibs = {e["candidate"]["external_id"]: e["candidate"]["bib"] for e in ok(client.get("/api/entries"))}
    assert bibs["W-2"] == "77" and bibs["W-1"] and bibs["W-1"].isdigit() and bibs["W-1"] not in ("11", "12", "13", "14", "77")


def test_each_screen_has_its_own_camera_choices(client):
    # the weigh-in keeps its webcam while the start and the finish use IP cameras
    got = ok(client.put("/api/settings", json={"camera_sources": {"weigh": ["webcam"], "start": ["ipcam"], "finish": ["ipcam", "photo"]}}))["camera_sources"]
    assert got["weigh"] == ["webcam"] and got["start"] == ["ipcam"]
    assert got["finish"] == ["ipcam"]                       # the finish cannot show a still photo
    assert got["station"] == ["webcam", "video", "ipcam"]   # a screen left out keeps what it had (all, here)
    again = ok(client.put("/api/settings", json={"camera_sources": {"results": ["webcam"]}}))["camera_sources"]
    assert again["weigh"] == ["webcam"] and again["start"] == ["ipcam"] and again["results"] == ["webcam"]   # only results changed
    assert got["results"] == ["webcam", "video", "photo"]
    # the old one-for-all list still works: every screen gets it (a screen that cannot use any keeps its first)
    got = ok(client.put("/api/settings", json={"camera_sources": ["ipcam"]}))["camera_sources"]
    assert got["weigh"] == ["ipcam"] and got["finish"] == ["ipcam"] and got["results"] == ["webcam"]


def test_a_persons_face_photos_can_be_listed_downloaded_and_put_in_a_folder(tmp_path):
    from test_faces import FakeEngine
    app = create_app(db_url=f"sqlite:///{tmp_path / 'a.db'}", web_dir="none", face_engine=lambda: FakeEngine(),
                     evidence_dir=tmp_path / "evidence", camhost_url="http://127.0.0.1:9")   # no camera service here
    with TestClient(app) as c:
        c.post("/api/entries/import", content=b"external_id,name,bib,gender,dob\nC-1,Alpha,11,M,2000-01-01\n")
        e = c.get("/api/entries").json()[0]
        c.post(f"/api/entries/{e['id']}/face", content=b"face:A")
        cid = e["candidate"]["id"]
        r = ok(c.get(f"/api/candidates/{cid}/faces"))
        assert len(r["faces"]) == 1 and r["folder"] == f"data/faces/{cid}-C-1"
        assert c.get(r["faces"][0]["url"]).headers["content-type"] == "image/jpeg"
        z = c.get(f"/api/candidates/{cid}/faces.zip")
        assert z.status_code == 200 and z.content[:2] == b"PK"
        f = ok(c.post(f"/api/candidates/{cid}/faces/folder"))
        assert f["photos"] == 1 and f["opened"] is False                     # written; nothing to open it with here
        folder = tmp_path / "faces" / f"{cid}-C-1"
        assert (folder / "face-1.jpg").exists()
        c.delete(f"/api/candidates/{cid}/face")                              # the faces go: the folder goes with them
        assert not folder.exists()
