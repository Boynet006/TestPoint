"""Face check-in: matching rules (pure) and the enrol/identify API with a fake engine."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from testpoint.api import models
from testpoint.api.app import create_app
from testpoint.api.faces import Face, FaceError, match


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
        who = data.split(b":")[-1][:1]
        emb = vec(99, 0.2, self.people.get(who, vec(42)))
        return Face((10, 10, 100, 120), 0.95, emb, b"\xff\xd8jpeg")


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
    assert n == 3                                         # only the newest 3 templates are kept
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
    # Registered by mistake, removed again: the face must not block a fresh registration
    i = ids(client)
    client.post(f"/api/entries/{i['13']}/face", content=b"face:C")
    assert client.delete(f"/api/entries/{i['13']}").status_code == 204
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
