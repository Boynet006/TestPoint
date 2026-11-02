"""Face check-in: matching rules (pure) and the enrol/identify API with a fake engine."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import models
from testpoint.api.app import create_app
from testpoint.api.faces import Face, FaceError, match
from runhelp import start_lined_up


def vec(seed, noise=0.0, base=None):
    rng = np.random.default_rng(seed)
    # noise is relative to the unit vector: 0.3 means a perturbation 30% of its length
    v = rng.normal(size=128) if base is None else base + noise * rng.normal(size=128) / np.sqrt(128)
    return (v / np.linalg.norm(v)).astype(np.float32)


A, B, C = vec(1), vec(2), vec(3)


def test_match_picks_the_right_person():
    gal = {1: [A], 2: [B], 3: [C]}
    m = match(vec(10, 0.3, A), gal)
    assert m.confident and m.candidate_id == 1 and m.score > 0.8


def test_match_refuses_strangers_and_close_calls():
    gal = {1: [A], 2: [B]}
    assert match(C, gal).candidate_id is None                     # stranger: below threshold
    twin = (A + B) / np.linalg.norm(A + B)                          # equally close to two people
    m = match(twin, gal)
    assert m.candidate_id is None and not m.confident
    assert match(A, {}).candidate_id is None                        # nobody enrolled


def test_best_of_several_templates_counts():
    m = match(vec(11, 0.3, B), {1: [A], 2: [C, B]})
    assert m.candidate_id == 2


class FakeEngine:
    """Image bytes b'face:A' -> person A, b'none' -> no face, b'two' -> two faces."""

    people = {b"A": A, b"B": B, b"C": C}

    def face_from_bytes(self, data, require_single=False):
        if data == b"none":
            raise FaceError("no_face")
        if data == b"two" and require_single:
            raise FaceError("multiple_faces")
        parts = data.split(b":")                        # b"face:A" or b"face:A:left" (a turned head)
        who = parts[1][:1] if len(parts) > 1 else parts[0][:1]
        # b"...:venue": the same person in other light and angle (a less alike embedding)
        emb = vec(77, 0.8, self.people.get(who, vec(42))) if b"venue" in data else vec(99, 0.2, self.people.get(who, vec(42)))
        turn = {b"left": 0.35, b"right": -0.35}.get(parts[2] if len(parts) > 2 else b"", 0.0)
        # b"...:blurry" / b"...:sharp": a measured picture quality (otherwise not measured)
        q, issues = (0.2, ("blurry",)) if b"blurry" in data else (0.95, ()) if b"sharp" in data else (None, ())
        return Face((10, 10, 100, 120), 0.95, emb, b"\xff\xd8jpeg" + data, yaw=turn, quality=q, issues=issues)


CSV = "external_id,name,bib\nC-1,Alpha,11\nC-2,Bravo,12\nC-3,Charlie,13\n"


@pytest.fixture()
def client(monkeypatch):
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: "2026-10-20",
                     face_engine=FakeEngine)
    with TestClient(app) as c:
        c.post("/api/entries/import", content=CSV.encode())
        yield c


def ids(client):
    return {e["candidate"]["bib"]: e["id"] for e in client.get("/api/entries").json()}


def test_enrol_identify_and_gate(client):
    i = ids(client)
    r = client.post("/api/face/identify?station=weigh", content=b"face:A").json()
    assert r["entry"] is None and r["reason"] == "nobody_enrolled"

    assert client.post(f"/api/entries/{i['11']}/face", content=b"face:A").json() == {"faces": 1}
    assert client.post(f"/api/entries/{i['12']}/face", content=b"face:B").json() == {"faces": 1}
    rows = {e["candidate"]["bib"]: e for e in client.get("/api/entries").json()}
    assert rows["11"]["faces"] == 1 and rows["13"]["faces"] == 0
    assert client.get(f"/api/entries/{i['11']}/face.jpg").content.startswith(b"\xff\xd8")

    r = client.post("/api/face/identify?station=weigh", content=b"face:A").json()
    assert r["confident"] and r["entry"]["candidate"]["bib"] == "11" and r["allowed"] is True

    r = client.post("/api/face/identify?station=pushup", content=b"face:B").json()
    assert r["entry"]["candidate"]["bib"] == "12" and r["allowed"] is False and r["reason"] == "not_weighed"

    r = client.post("/api/face/identify?station=weigh", content=b"face:C").json()
    assert r["entry"] is None and r["reason"] == "face_not_recognised"


def test_cannot_enrol_someone_elses_face(client):
    i = ids(client)
    client.post(f"/api/entries/{i['11']}/face", content=b"face:A")
    r = client.post(f"/api/entries/{i['12']}/face", content=b"face:A")   # A's photo under B's name
    assert r.status_code == 409 and r.json()["detail"]["code"] == "face_belongs_to_other"
    assert "Alpha" in r.json()["detail"]["message"]
    assert client.post(f"/api/entries/{i['11']}/face", content=b"face:A").json()["faces"] == 2   # same person: fine


def test_enrol_errors_and_limits(client):
    i = ids(client)
    r = client.post(f"/api/entries/{i['11']}/face", content=b"none")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "no_face"
    r = client.post(f"/api/entries/{i['11']}/face", content=b"two")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "multiple_faces"
    for _ in range(5):
        n = client.post(f"/api/entries/{i['11']}/face", content=b"face:A").json()["faces"]
    assert n == 4                                         # only the newest 4 registration templates are kept
    assert client.delete(f"/api/entries/{i['11']}/face").status_code == 204
    assert client.get(f"/api/entries/{i['11']}/face.jpg").status_code == 404


def test_without_models_face_endpoints_say_unavailable(monkeypatch):
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", face_engine=lambda: None)
    with TestClient(app) as c:
        assert c.get("/api/face/status").json() == {"available": False}
        r = c.post("/api/face/identify", content=b"x")
        assert r.status_code == 503 and r.json()["detail"]["code"] == "face_unavailable"


def test_removing_a_mistaken_entry_frees_the_face(client):
    # Registered by mistake, removed again (forget): the face must not block a fresh registration
    i = ids(client)
    client.post(f"/api/entries/{i['13']}/face", content=b"face:C")
    assert client.delete(f"/api/entries/{i['13']}?forget=true").status_code == 204
    r = client.post("/api/entries", json={"external_id": "C-9", "name": "Charlie Again", "bib": "19"})
    assert r.status_code == 201
    assert client.post(f"/api/entries/{r.json()['id']}/face", content=b"face:C").json() == {"faces": 1}
    # the removed person is gone too: re-adding the old ID is a new person, without the old face
    assert client.post("/api/entries", json={"external_id": "C-3", "name": "Charlie", "bib": "13"}).status_code == 201
    rows = {e["candidate"]["bib"]: e for e in client.get("/api/entries").json()}
    assert rows["13"]["faces"] == 0


def test_face_from_another_day_is_offered_as_returning_candidate(monkeypatch):
    real = models.create_engine
    monkeypatch.setattr(models, "create_engine", lambda url, **kw: real(url, poolclass=StaticPool, **kw))
    day = {"d": "2026-10-20"}
    app = create_app(db_url="sqlite:///:memory:", web_dir="none", today=lambda: day["d"], face_engine=FakeEngine)
    with TestClient(app) as c:
        c.post("/api/entries/import", content=CSV.encode())
        i = ids(c)
        c.post(f"/api/entries/{i['11']}/face", content=b"face:A")
        day["d"] = "2026-10-27"                                  # a week later: today's list is empty
        r = c.post("/api/face/identify?station=weigh", content=b"face:A").json()
        assert r["entry"] is None and r["reason"] == "not_on_todays_list"
        assert r["known"]["name"] == "Alpha" and r["known"]["bib"] == "11" and r["known"]["external_id"] == "C-1"
        # adding them back keeps the person (and the face): they are recognised at once
        e = c.post("/api/entries", json={k: r["known"][k] for k in ("external_id", "name", "bib")})
        assert e.status_code == 201
        r = c.post("/api/face/identify?station=weigh", content=b"face:A").json()
        assert r["entry"]["candidate"]["name"] == "Alpha" and r["allowed"] is True
        # a stranger is still a stranger
        assert "known" not in c.post("/api/face/identify?station=weigh", content=b"face:C").json()
        # removing a person who has another day keeps them (and their face) on file
        assert c.delete(f"/api/entries/{e.json()['id']}").status_code == 204
        assert c.post("/api/face/identify?station=weigh", content=b"face:A").json()["known"]["name"] == "Alpha"


def test_head_yaw_from_landmarks():
    import numpy as np

    from testpoint.api.faces import head_yaw
    # x, y, w, h, right eye (image left), left eye (image right), nose, mouth corners, score
    straight = np.array([0, 0, 100, 120, 30, 40, 70, 40, 50, 65, 35, 90, 65, 90, 0.9], np.float32)
    assert abs(head_yaw(straight)) < 1e-6
    turned = straight.copy(); turned[8] = 62           # nose 12 px right of the eye midpoint
    assert head_yaw(turned) == pytest.approx(0.3)      # 12 / 40 eye distance: their own left
    turned[8] = 38
    assert head_yaw(turned) == pytest.approx(-0.3)



def test_finish_camera_face_fallback(client, monkeypatch):
    """No bib read at the finish: the face picks the runner among those still out."""
    import base64
    from datetime import timedelta

    from testpoint.api import app as app_mod
    i = ids(client)
    for bib, who in (("11", b"A"), ("12", b"B")):
        client.post(f"/api/entries/{i[bib]}/face", content=b"face:" + who, headers={"content-type": "image/jpeg"})
        client.patch(f"/api/entries/{i[bib]}/candidate", json={"gender": "M", "dob": "2000-01-01"})
        client.post(f"/api/entries/{i[bib]}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True})
        client.post(f"/api/entries/{i[bib]}/line-up")
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    start_lined_up(client)
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=15))
    face = lambda b: base64.b64encode(b).decode()
    # A crosses, bib unreadable: recorded for 11 by face
    c = client.post("/api/engine/crossing", json={"bib": None, "face": face(b"face:A")}).json()
    assert not c["needs_review"] and c["entry_id"] == i["11"] and c["note_code"] == "identified_by_face"
    # a stranger (not enrolled): stays in review
    c = client.post("/api/engine/crossing", json={"bib": None, "face": face(b"face:C")}).json()
    assert c["needs_review"]
    # no face in the picture: review, no error
    c = client.post("/api/engine/crossing", json={"bib": None, "face": face(b"none")}).json()
    assert c["needs_review"]
    # 11 already finished: their face no longer matches anyone still out
    c = client.post("/api/engine/crossing", json={"bib": None, "face": face(b"face:A")}).json()
    assert c["needs_review"]
    # several snapshots: the one with a face counts (12 = B is still out)
    c = client.post("/api/engine/crossing", json={"bib": None, "faces": [face(b"none"), face(b"face:B")]}).json()
    assert not c["needs_review"] and c["entry_id"] == i["12"]
    # everyone is in now: the camera's crossings are not runners and are not queued for review
    c = client.post("/api/engine/crossing", json={"bib": "12", "face": face(b"face:A")}).json()
    assert c == {"ignored": True, "reason": "nobody_running"}


def test_burst_enrolment_keeps_the_best_frame_and_refuses_poor_ones(client):
    import base64
    i = ids(client)
    b64 = lambda b: base64.b64encode(b).decode()
    r = client.post(f"/api/entries/{i['11']}/face/burst", json={"frames": [b64(b"face:A:straight:blurry"), b64(b"none"),
                                                                         b64(b"face:A:straight:sharp")]})
    assert r.json() == {"faces": 1}
    assert client.get(f"/api/entries/{i['11']}/face.jpg").content.endswith(b"sharp")
    # nothing good enough: refused, with what is wrong, so the desk takes another
    r = client.post(f"/api/entries/{i['12']}/face/burst", json={"frames": [b64(b"face:B:straight:blurry")]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "low_quality" and r.json()["detail"]["issues"] == ["blurry"]
    r = client.post(f"/api/entries/{i['12']}/face/burst", json={"frames": [b64(b"none")]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "no_face"
    assert client.post(f"/api/entries/{i['12']}/face/burst", json={"frames": []}).status_code == 422


def test_very_sure_matches_at_stations_add_a_template_in_the_venue_light(client):
    i = ids(client)
    client.post(f"/api/entries/{i['11']}/face", content=b"face:A:straight:sharp")
    # the weigh-in desk never adapts (same light as registration)
    assert not client.post("/api/face/identify?station=weigh", content=b"face:A:x:venue:sharp").json().get("adapted")
    # at the start line: a sure match that looks different from every template is kept
    r = client.post("/api/face/identify?station=lineup", content=b"face:A:x:venue:sharp").json()
    assert r["entry"]["id"] == i["11"] and r["adapted"] is True and r["entry"]["faces"] == 2
    # the same look again adds nothing; a poor picture never does
    assert client.post("/api/face/identify?station=lineup", content=b"face:A:x:venue:sharp").json()["adapted"] is False
    assert client.post("/api/face/identify?station=lineup", content=b"face:A:x:venue:blurry").json()["adapted"] is False
    # the photo shown stays the registration one
    assert client.get(f"/api/entries/{i['11']}/face.jpg").content.endswith(b"sharp") and b"venue" not in client.get(f"/api/entries/{i['11']}/face.jpg").content


def test_finish_face_and_bib_must_agree(client, monkeypatch):
    """The bib read names a runner, but the face is clearly someone else still out: review."""
    import base64
    from datetime import timedelta

    from testpoint.api import app as app_mod
    i = ids(client)
    for bib, who in (("11", b"A"), ("12", b"B")):
        client.post(f"/api/entries/{i[bib]}/face", content=b"face:" + who)
        client.patch(f"/api/entries/{i[bib]}/candidate", json={"gender": "M", "dob": "2000-01-01"})
        client.post(f"/api/entries/{i[bib]}/weigh-in", json={"weight_kg": 70, "height_cm": 175, "accept": True})
        client.post(f"/api/entries/{i[bib]}/line-up")
    t0 = models.utcnow()
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0)
    start_lined_up(client)
    monkeypatch.setattr(app_mod.S, "utcnow", lambda: t0 + timedelta(minutes=15))
    face = lambda b: base64.b64encode(b).decode()
    # bib 12 read, but it is A (11) crossing: back to review, A suggested, 12 still out
    c = client.post("/api/engine/crossing", json={"bib": "12", "faces": [face(b"face:A")]}).json()
    assert c["needs_review"] and c["entry_id"] is None and c["note_code"] == "face_bib_disagree" and c["note_bib"] == "11"
    assert client.get(f"/api/entries/{i['12']}").json()["status"] == "running"
    assert client.get(f"/api/entries/{i['11']}").json()["status"] == "running"
    # bib and face agree: recorded
    c = client.post("/api/engine/crossing", json={"bib": "12", "faces": [face(b"face:B")]}).json()
    assert not c["needs_review"] and c["entry_id"] == i["12"]
    # no face in the snapshots: the bib alone decides
    c = client.post("/api/engine/crossing", json={"bib": "11", "faces": [face(b"none")]}).json()
    assert not c["needs_review"] and c["entry_id"] == i["11"]


def test_the_finish_matches_faces_at_the_normal_setting_when_few_are_out():
    from testpoint.api import service as S
    assert S.finish_face_match(1) == S.FACE_MATCH["normal"] and S.finish_face_match(10) == S.FACE_MATCH["normal"]
    assert S.finish_face_match(11) == S.FACE_MATCH["strict"]


def test_finish_face_match_from_settings():
    from testpoint.api import service as S
    assert S.finish_face_match(1, 0.4) == (0.4, S.FACE_MATCH["normal"][1])
    assert S.finish_face_match(30, 0.33)[0] == 0.33 and S.finish_face_match(30, 0.0) == S.FACE_MATCH["strict"]
