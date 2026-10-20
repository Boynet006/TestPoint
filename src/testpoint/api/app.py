"""FastAPI app: JSON API under /api, the station UI as static files under /.

    testpoint-app                     # http://0.0.0.0:8000, data in ./data/testpoint.db
    TESTPOINT_DB=sqlite:////data/testpoint.db TESTPOINT_TZ=Asia/Dubai testpoint-app

Live screens poll ``GET /api/state`` every second or two; a WebSocket can
replace that later without changing the rest of the API.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

# Windows' registry may not know these; a module script or wasm with the wrong type is refused.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import journey as J
from . import service as S
from .models import Attempt, Crossing, Day, Entry, FaceTemplate, Rep, make_sessionmaker
from .rules import EVENTS, GRADES, TESTS, DayRules


# ------------------------------------------------------------------ serialisation

def iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat(timespec="milliseconds") + "Z"


def entry_out(e: Entry, day: Day | None = None, faces: int | None = None) -> dict:
    c = e.candidate
    events = {}
    for a in e.attempts:
        if a.status in ("running", "finished"):
            events[a.event] = {"attempt_id": a.id, "status": a.status, "count": a.count,
                               "rejected": a.rejected, "passed": a.passed, "grade": a.grade, "lane": a.lane}
    out = {
        "id": e.id, "status": e.status,
        "candidate": {"id": c.id, "external_id": c.external_id, "name": c.name,
                      "name_ar": c.name_ar, "gender": c.gender, "dob": c.dob, "bib": c.bib, "unit": c.unit},
        "weight_kg": e.weight_kg, "height_cm": e.height_cm, "bmi": e.bmi,
        "run_time_s": e.run_time_s, "events": events,
        "standard": {"age": e.age, "band": e.band, "level": e.level},
        "wave": e.wave.number if e.wave else None,
        "run_grade": e.run_grade, "verdict": S.verdict(e),
    }
    if faces is not None:
        out["faces"] = faces
    if day is not None:
        out["gates"] = {st: S.gate_for(e, day, st).reason or "ok" for st in J.STATIONS}
    return out


def crossing_out(c: Crossing) -> dict:
    return {"id": c.id, "entry_id": c.entry_id, "at": iso(c.at), "time_s": c.time_s,
            "bib_read": c.bib_read, "source": c.source, "needs_review": c.needs_review,
            "note": c.note, "note_code": c.note_code, "note_bib": c.note_bib,
            "note_status": c.note_status, "confidence": c.confidence, "clip": c.clip_path}


def attempt_out(a: Attempt) -> dict:
    return {"id": a.id, "entry_id": a.entry_id, "event": a.event, "lane": a.lane,
            "status": a.status, "count": a.count, "rejected": a.rejected, "passed": a.passed, "grade": a.grade,
            "auto_count": a.auto_count, "auto_rejected": a.auto_rejected,
            "override_reason": a.override_reason,
            "started_at": iso(a.started_at), "ended_at": iso(a.ended_at)}


def review_out(a: Attempt) -> dict:
    """An attempt with its candidate and reps: one row of the review screen."""
    c = a.entry.candidate
    out = attempt_out(a)
    out["candidate"] = {"id": c.id, "external_id": c.external_id, "name": c.name,
                        "name_ar": c.name_ar, "bib": c.bib}
    out["reps"] = [rep_out(r) for r in a.reps]
    return out


def rep_out(r: Rep) -> dict:
    return {"index": r.index, "valid": r.valid, "faults": json.loads(r.faults),
            "t_start": r.t_start, "t_end": r.t_end, "clip": r.clip_path}


# ----------------------------------------------------------------------- requests

class CandidateIn(BaseModel):
    external_id: str
    name: str
    name_ar: str | None = None
    gender: str | None = None
    dob: str | None = None
    bib: str | None = None
    unit: str | None = None
    auto_bib: bool = False
    """Give the next free bib when ``bib`` is empty or already taken today."""
    walk_in: bool = False
    """Registered at the weigh-in desk (refused when the walk_ins setting is off)."""


class SettingsIn(BaseModel):
    """Station settings (service.SETTINGS); only the fields sent are changed."""
    auto_identify: bool | None = None
    camera_background: Literal["off", "blur", "backdrop", "image"] | None = None
    backdrop_color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    face_match: Literal["normal", "strict"] | None = None
    face_required: bool | None = None
    walk_ins: bool | None = None
    default_gender: Literal["M", "F", ""] | None = None
    default_unit: Literal["combat", "admin", "special"] | None = None
    bib_start: int | None = Field(default=None, ge=1, le=99999)
    lanes: int | None = Field(default=None, ge=1, le=4)
    wave_size: int | None = Field(default=None, ge=1, le=200)
    start_countdown_s: int | None = Field(default=None, ge=0, le=30)
    site_name: str | None = Field(default=None, max_length=80)
    default_language: Literal["en", "ar"] | None = None


class BandIn(BaseModel):
    age_min: int
    age_max: int
    run_s: list[int] = Field(min_length=3, max_length=3)
    pushup: list[int] = Field(min_length=3, max_length=3)
    crunch: list[int] = Field(min_length=3, max_length=3)


class RulesIn(BaseModel):
    """A new rule set; only the fields sent change (see rules.DayRules)."""
    bmi_min: float | None = None
    bmi_max: float | None = None
    run_cutoff_s: float | None = None
    exercise_time_s: int | None = None
    unit_level: dict[str, int] | None = None
    men: list[BandIn] | None = None
    women: list[BandIn] | None = None


class PinIn(BaseModel):
    pin: str | None = None
    """4 to 8 digits; empty removes the PIN."""


class StartIn(BaseModel):
    countdown_s: int | None = Field(default=None, ge=0, le=30)
    """Seconds of countdown before the time starts; default: the start_countdown_s setting."""


class CloseIn(BaseModel):
    wave_id: int | None = None


class CheckinIn(BaseModel):
    station: str
    query: str


class WeighInIn(BaseModel):
    weight_kg: float = Field(gt=0, lt=400)
    height_cm: float = Field(gt=50, lt=260)
    accept: bool
    override_reason: str | None = None
    gender: str | None = None
    dob: str | None = None
    unit: str | None = None
    """Missing gender / date of birth, and the unit type, confirmed at the scale."""


class CandidateEdit(BaseModel):
    """Corrections to a candidate; only the fields sent change (null clears an optional one)."""
    external_id: str | None = None
    name: str | None = None
    name_ar: str | None = None
    gender: str | None = None
    dob: str | None = None
    unit: str | None = None
    bib: str | None = None


class StandardIn(BaseModel):
    gender: str | None = None
    dob: str | None = None
    unit: str | None = None


class FinishIn(BaseModel):
    bib: str | None = None


class ResolveIn(BaseModel):
    entry_id: int


class AttemptIn(BaseModel):
    entry_id: int
    event: str
    lane: str | None = None


class UndoIn(BaseModel):
    reason: str | None = None


class AttemptResultIn(BaseModel):
    count: int = Field(ge=0, le=500)
    rejected: int = Field(default=0, ge=0, le=500)
    # Required when the camera counted and ``count`` differs from its number.
    override_reason: str | None = Field(default=None, max_length=300)


# The engine's report, see docs/ENGINE-LINK.md.
class EngineEventIn(BaseModel):
    index: int = Field(ge=1)
    valid: bool
    faults: list[str] = []
    t_start: float
    t_end: float
    clip: str | None = None


class EngineLaneIn(BaseModel):
    attempt_id: int | None = None
    count: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)
    phase: str = "waiting"
    person: bool = False
    pending_faults: list[str] = []
    work_seconds: float = 0.0
    events: list[EngineEventIn] = []


class EngineReportIn(BaseModel):
    station: str
    t: float = 0.0
    fps: float | None = None
    source: str | None = None
    lanes: dict[str, EngineLaneIn]


class CrossingIn(BaseModel):
    """A finish-line crossing seen by the camera (src/testpoint/finishcam.py)."""
    bib: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    latency_s: float = Field(default=0.0, ge=0, le=60)
    """How long ago the crossing happened when the report was sent; the server stamps the true instant."""
    readings: list[dict] = []
    clip: str | None = None
    track: int | None = None


ENGINE_STALE_S = 2.0
FRAME_STALE_S = 5.0
"""A camera picture older than this is not shown (the engine stopped)."""
FRAME_STATION = re.compile(r"[a-z][a-z0-9_-]{0,23}")
JPEG_SOI = bytes([0xFF, 0xD8])
"""JPEG start-of-image marker: the first two bytes of every JPEG file."""
"""No report for this long and the station UI falls back to manual counting."""


# ---------------------------------------------------------------------------- app

def default_face_engine():
    """The face engine, or None (with the reason logged) if its models are missing."""
    from .faces import FaceError, get_engine

    try:
        return get_engine()
    except FaceError as e:
        print(f"face check-in disabled: {e.message}")
        return None


def create_app(db_url: str | None = None, tz: str | None = None, rules: DayRules | None = None,
               web_dir: str | Path | None = None, today: callable | None = None,
               face_engine: callable | None = None, clock: callable = time.monotonic,
               evidence_dir: str | Path | None = None) -> FastAPI:
    db_url = db_url or os.environ.get("TESTPOINT_DB", "sqlite:///data/testpoint.db")
    # Where the engine writes its clips (``testpoint-engine --evidence``); rep clip paths
    # in the database are relative to it.
    evidence = Path(evidence_dir or os.environ.get("TESTPOINT_EVIDENCE", "data/evidence")).resolve()
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:///:memory:"):
        Path(db_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    zone = ZoneInfo(tz or os.environ.get("TESTPOINT_TZ", "Asia/Dubai"))
    rules = rules or DayRules()
    SessionLocal = make_sessionmaker(db_url)
    with SessionLocal() as _db:
        S.migrate_single_runs(_db)
    today = today or (lambda: datetime.now(zone).date().isoformat())

    app = FastAPI(title="TestPoint station", version="0.3.0")
    app.state.sessionmaker = SessionLocal
    face_engine = face_engine or default_face_engine
    _engine_cache: dict = {}
    # The live picture per station from the camera engine: the last report and when it
    # arrived. Not persisted; the reps it carried are in the database.
    _last_report: dict[str, tuple[float, EngineReportIn]] = {}
    # The engines' live pictures (src/testpoint/preview.py): newest JPEG per station.
    _last_frame: dict[str, tuple[float, bytes]] = {}

    def engine():
        if "e" not in _engine_cache:
            _engine_cache["e"] = face_engine()
        return _engine_cache["e"]

    def get_db() -> Iterator[Session]:
        db = SessionLocal()
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_day(db: Session = Depends(get_db)) -> Day:
        return S.get_or_create_day(db, today(), rules)

    def day_rules(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> DayRules:
        """The rules today is judged by (fixed once the day has a result)."""
        return S.rules_for(db, day, rules)

    def admin(request: Request, db: Session = Depends(get_db)) -> None:
        """Administrator actions need the PIN when one is set (header X-Admin-Pin)."""
        S.check_admin_pin(db, request.headers.get("x-admin-pin"))

    @app.middleware("http")
    async def _no_cache(request: Request, call_next):
        # Station devices must always get the current UI and live data, never a stale copy.
        response = await call_next(request)
        if not request.url.path.startswith("/ui/") or request.url.path.endswith((".html", "/")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    # One change at a time: every station checks a rule, then writes. Serialising the writes
    # (one server process, SQLite) removes the races between devices, e.g. a line-up that
    # lands during the mass start, a double START or a double-tapped "Start test".
    write_lock = asyncio.Lock()
    MAX_BODY = 15 * 1024 * 1024

    @app.middleware("http")
    async def _one_writer(request: Request, call_next):
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)
        if int(request.headers.get("content-length") or 0) > MAX_BODY:
            return JSONResponse({"detail": {"code": "too_large", "message": "upload larger than 15 MB"}}, 413)
        async with write_lock:
            return await call_next(request)

    def _auto_close() -> None:
        db = SessionLocal()
        try:
            day = S.get_or_create_day(db, today(), rules)
            S.auto_close(db, day, S.rules_for(db, day, rules))
        finally:
            db.close()

    @app.middleware("http")
    async def _close_overdue_waves(request: Request, call_next):
        # Every station polls /api/state every 2 s: a wave past its cut-off is closed here,
        # under the same lock as every other write, so nobody has to remember to close it.
        if request.method == "GET" and request.url.path == "/api/state":
            async with write_lock:
                await run_in_threadpool(_auto_close)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def _bad_input(_: Request, e: RequestValidationError) -> JSONResponse:
        first = e.errors()[0] if e.errors() else {}
        field = ".".join(str(x) for x in first.get("loc", [])[1:]) or "input"
        return JSONResponse({"detail": {"code": "bad_input", "message": f"{field}: {first.get('msg', 'invalid')}"}},
                            status_code=422)

    @app.exception_handler(S.ServiceError)
    async def _service_error(_: Request, e: S.ServiceError) -> JSONResponse:
        return JSONResponse({"detail": {"code": e.code, "message": e.message}}, status_code=e.status)

    # ------------------------------------------------------------------ read
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/rules")
    def get_rules(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """Today's rules (in force), plus ``next`` when a newer version waits for the next day."""
        today_r, next_r = S.rules_for(db, day, rules), S.current_rules(db, rules)
        out = today_r.to_dict()
        if next_r.version != today_r.version:
            out["next"] = next_r.to_dict()
        return out

    @app.put("/api/rules", dependencies=[Depends(admin)])
    def rules_update(body: RulesIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return S.update_rules(db, day, rules, body.model_dump(exclude_none=True))

    @app.get("/api/settings")
    def settings(db: Session = Depends(get_db)) -> dict:
        return S.get_settings(db)

    @app.put("/api/settings", dependencies=[Depends(admin)])
    def settings_update(body: SettingsIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return S.update_settings(db, day, body.model_dump(exclude_none=True))

    @app.get("/api/settings/background-image")
    def background_image(db: Session = Depends(get_db)) -> Response:
        got = S.get_background(db)
        if got is None:
            return Response(status_code=404)
        data, kind = got
        # the URL carries the version (?v=), so a desk can keep it until a new one is uploaded
        return Response(data, media_type=kind, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @app.put("/api/settings/background-image", status_code=204, dependencies=[Depends(admin)])
    async def background_image_set(request: Request, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> None:
        S.set_background(db, day, await request.body())

    @app.delete("/api/settings/background-image", status_code=204, dependencies=[Depends(admin)])
    def background_image_remove(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> None:
        S.set_background(db, day, None)

    @app.put("/api/settings/admin-pin", dependencies=[Depends(admin)])
    def admin_pin(body: PinIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        S.set_admin_pin(db, day, (body.pin or "").strip() or None)
        return {"admin_pin_set": S.pin_is_set(db)}

    @app.post("/api/settings/admin-pin/check")
    def admin_pin_check(request: Request, db: Session = Depends(get_db)) -> dict:
        """Unlock Advanced on a device: is this PIN right? (Every admin action checks it again.)"""
        S.check_admin_pin(db, request.headers.get("x-admin-pin"))
        return {"ok": True}

    @app.get("/api/state")
    def state(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        s = S.summary(db, day)
        s["run_started_at"] = iso(s["run_started_at"])
        s["run_closed_at"] = iso(s["run_closed_at"])
        for w in s["waves"]:
            w["started_at"], w["closed_at"] = iso(w["started_at"]), iso(w["closed_at"])
        s["server_time"] = iso(S.utcnow())
        return s

    @app.get("/api/entries")
    def entries(status: str | None = None, db: Session = Depends(get_db),
                day: Day = Depends(get_day)) -> list[dict]:
        rows = S.list_entries(db, day, status)
        counts = dict(db.execute(select(FaceTemplate.candidate_id, func.count())
                                 .group_by(FaceTemplate.candidate_id)).all())
        return [entry_out(e, day, counts.get(e.candidate_id, 0)) for e in rows]

    @app.get("/api/entries/{entry_id}")
    def entry(entry_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return entry_out(S.get_entry(db, day, entry_id), day)

    @app.get("/api/run/crossings")
    def crossings(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> list[dict]:
        return [crossing_out(c) for c in S.list_crossings(db, day)]

    # -------------------------------------------------------------- list admin
    @app.post("/api/entries", status_code=201)
    def add(body: CandidateIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        data = body.model_dump()
        if data.pop("walk_in") and not S.get_settings(db)["walk_ins"]:
            raise S.ServiceError("walk_ins_off", "registering new people at the desk is switched off", 403)
        return entry_out(S.add_candidate(db, day, **data), day)

    @app.post("/api/entries/import", dependencies=[Depends(admin)])
    async def import_csv(request: Request, db: Session = Depends(get_db),
                         day: Day = Depends(get_day)) -> dict:
        text = S.decode_csv(await request.body())
        return S.import_csv(db, day, text)

    # ------------------------------------------------------------------ check-in
    @app.post("/api/checkin")
    def checkin(body: CheckinIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """Identify a candidate (bib / ID now, face later) and say whether this station may take them."""
        if body.station not in J.STATIONS:
            raise S.ServiceError("unknown_station", f"unknown station {body.station!r}", 400)
        e = S.find_entry(db, day, body.query)
        if body.station in ("weigh", "lineup") and S.get_settings(db)["face_required"] and S.face_count(db, e.candidate_id):
            raise S.ServiceError("face_required", f"{e.candidate.name} has a face on file: check in by face", 409)
        g = S.gate_for(e, day, body.station)
        return {"entry": entry_out(e, day), "allowed": g.ok, "reason": g.reason or None}

    # --------------------------------------------------------------------- faces
    @app.get("/api/face/status")
    def face_status() -> dict:
        return {"available": engine() is not None}

    @app.post("/api/entries/{entry_id}/face")
    async def face_enroll(entry_id: int, request: Request, db: Session = Depends(get_db),
                          day: Day = Depends(get_day)) -> dict:
        data = await request.body()
        return {"faces": S.enroll_face(db, day, S.get_entry(db, day, entry_id), data, engine())}

    @app.delete("/api/entries/{entry_id}/face", status_code=204, dependencies=[Depends(admin)])
    def face_remove(entry_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> None:
        S.remove_faces(db, day, S.get_entry(db, day, entry_id))

    @app.get("/api/entries/{entry_id}/face.jpg")
    def face_jpg(entry_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> Response:
        jpg = S.face_photo(db, S.get_entry(db, day, entry_id))
        if not jpg:
            raise S.ServiceError("no_face_enrolled", "no face enrolled", 404)
        return Response(jpg, media_type="image/jpeg")

    @app.post("/api/face/pose")
    async def face_pose(request: Request) -> dict:
        """Live guide while enrolling: where the face is and how far the head is turned."""
        eng = engine()
        if eng is None:
            raise S.ServiceError("face_unavailable", "face recognition is not available on this server", 503)
        try:
            return eng.pose_from_bytes(await request.body())
        except Exception as e:     # FaceError from a bad upload
            raise S.ServiceError(getattr(e, "code", "bad_image"), str(e), 422) from None

    @app.post("/api/face/identify")
    async def face_identify(request: Request, station: str | None = None,
                            db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        if station is not None and station not in J.STATIONS:
            raise S.ServiceError("unknown_station", f"unknown station {station!r}", 400)
        r = S.identify(db, day, await request.body(), engine(), station)
        if r["entry"] is not None:
            r["entry"] = entry_out(r["entry"], day, S.face_count(db, r["entry"].candidate_id))
        return r

    # ------------------------------------------------------------------ weigh-in
    @app.post("/api/entries/{entry_id}/weigh-in")
    def weigh_in(entry_id: int, body: WeighInIn, db: Session = Depends(get_db),
                 day: Day = Depends(get_day)) -> dict:
        e = S.weigh_in(db, day, S.rules_for(db, day, rules), S.get_entry(db, day, entry_id), **body.model_dump())
        return entry_out(e, day)

    @app.post("/api/entries/{entry_id}/standard")
    def standard(entry_id: int, body: StandardIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """Before weighing: the age band, required level and what each grade needs (nothing is saved)."""
        r = S.rules_for(db, day, rules)
        std = S.standard_for(db, day, r, S.get_entry(db, day, entry_id), **body.model_dump())
        band = r.find_band(std["band"])
        std["needs"] = {t: dict(zip(map(str, GRADES), band.needs(t))) for t in TESTS} if band else None
        return std

    # ----------------------------------------------------------------------- run
    @app.post("/api/entries/{entry_id}/line-up")
    def line_up(entry_id: int, by: Literal["typed", "face"] = "typed", db: Session = Depends(get_db),
                day: Day = Depends(get_day)) -> dict:
        return entry_out(S.line_up(db, day, S.get_entry(db, day, entry_id), by=by), day)

    @app.post("/api/entries/{entry_id}/line-up/undo")
    def line_up_undo(entry_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return entry_out(S.undo_line_up(db, day, S.get_entry(db, day, entry_id)), day)

    @app.post("/api/entries/{entry_id}/weigh-in/undo")
    def weigh_in_undo(entry_id: int, body: UndoIn | None = None, db: Session = Depends(get_db),
                      day: Day = Depends(get_day)) -> dict:
        reason = body.reason if body else None
        return entry_out(S.undo_weigh_in(db, day, S.get_entry(db, day, entry_id), reason), day)

    @app.patch("/api/entries/{entry_id}/candidate", dependencies=[Depends(admin)])
    def candidate_edit(entry_id: int, body: CandidateEdit, db: Session = Depends(get_db),
                       day: Day = Depends(get_day)) -> dict:
        e = S.edit_candidate(db, day, S.rules_for(db, day, rules), S.get_entry(db, day, entry_id),
                             body.model_dump(exclude_unset=True))
        return entry_out(e, day, S.face_count(db, e.candidate_id))

    @app.delete("/api/entries/{entry_id}", status_code=204, dependencies=[Depends(admin)])
    def entry_remove(entry_id: int, delete_results: bool = False, db: Session = Depends(get_db),
                     day: Day = Depends(get_day)) -> None:
        """Take someone off today's list; ``delete_results=true`` also when they have results."""
        S.remove_entry(db, day, S.get_entry(db, day, entry_id), delete_results=delete_results)

    @app.post("/api/run/start/cancel")
    def run_start_cancel(request: Request, late: bool = False, db: Session = Depends(get_db),
                         day: Day = Depends(get_day)) -> dict:
        """``late=true``: after the first 2 minutes, an administrator decision (PIN when one is set)."""
        if late:
            S.check_admin_pin(db, request.headers.get("x-admin-pin"))
        w = S.cancel_start(db, day, late=late)
        return {"wave": w.number, "run_started_at": None}

    @app.post("/api/run/crossings/{crossing_id}/unassign")
    def run_unassign(crossing_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return crossing_out(S.unassign_crossing(db, day, crossing_id))

    @app.delete("/api/run/crossings/{crossing_id}", status_code=204)
    def run_discard(crossing_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> None:
        S.discard_crossing(db, day, crossing_id)

    @app.post("/api/run/start")
    def run_start(body: StartIn | None = None, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        countdown = body.countdown_s if body and body.countdown_s is not None else 0
        w = S.start_run(db, day, countdown_s=countdown)
        return {"wave": w.number, "wave_id": w.id, "run_started_at": iso(w.started_at)}

    @app.post("/api/run/finish", status_code=201)
    def run_finish(body: FinishIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return crossing_out(S.record_finish(db, day, S.rules_for(db, day, rules), body.bib))

    @app.post("/api/run/crossings/{crossing_id}/resolve")
    def run_resolve(crossing_id: int, body: ResolveIn, db: Session = Depends(get_db),
                    day: Day = Depends(get_day)) -> dict:
        c = S.resolve_crossing(db, day, S.rules_for(db, day, rules), crossing_id, S.get_entry(db, day, body.entry_id))
        return crossing_out(c)

    @app.post("/api/run/close")
    def run_close(body: CloseIn | None = None, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        w = S.close_run(db, day, wave_id=body.wave_id if body else None)
        return {"wave": w.number, "run_closed_at": iso(w.closed_at)}

    # ------------------------------------------------------------------ exercises
    @app.post("/api/attempts", status_code=201)
    def attempt_start(body: AttemptIn, db: Session = Depends(get_db),
                      day: Day = Depends(get_day)) -> dict:
        a = S.start_attempt(db, day, S.get_entry(db, day, body.entry_id), body.event, body.lane)
        return attempt_out(a)

    @app.post("/api/attempts/{attempt_id}/finish")
    def attempt_finish(attempt_id: int, body: AttemptResultIn, db: Session = Depends(get_db),
                       day: Day = Depends(get_day)) -> dict:
        return attempt_out(S.finish_attempt(db, day, S.rules_for(db, day, rules), attempt_id, **body.model_dump()))

    @app.post("/api/attempts/{attempt_id}/cancel", status_code=204)
    def attempt_cancel(attempt_id: int, db: Session = Depends(get_db),
                       day: Day = Depends(get_day)) -> None:
        S.cancel_attempt(db, day, attempt_id)

    @app.get("/api/attempts/{attempt_id}/reps")
    def attempt_reps(attempt_id: int, db: Session = Depends(get_db),
                     day: Day = Depends(get_day)) -> list[dict]:
        return [rep_out(r) for r in S.list_reps(db, day, attempt_id)]

    # -------------------------------------------------------------------- review
    @app.get("/api/attempts")
    def attempts_finished(event: str | None = None, db: Session = Depends(get_db),
                          day: Day = Depends(get_day)) -> list[dict]:
        """Today's finished attempts with saved and camera counts, override reason and reps."""
        if event is not None and event not in EVENTS:
            raise S.ServiceError("unknown_event", f"unknown event {event!r}", 400)
        return [review_out(a) for a in S.list_finished_attempts(db, day, event)]

    @app.get("/api/evidence/{path:path}")
    def evidence_file(path: str) -> FileResponse:
        """A clip or full video from the evidence folder; nothing outside it is served."""
        try:
            target = (evidence / path).resolve()
        except (OSError, ValueError):
            raise S.ServiceError("not_found", "no such evidence file", 404) from None
        if not target.is_relative_to(evidence) or not target.is_file():
            raise S.ServiceError("not_found", "no such evidence file", 404)
        media = "video/mp4" if target.suffix.lower() == ".mp4" else None
        return FileResponse(target, media_type=media)

    # --------------------------------------------------------------- engine link
    @app.post("/api/engine/report")
    def engine_report(body: EngineReportIn, db: Session = Depends(get_db),
                      day: Day = Depends(get_day)) -> dict:
        """The camera engine's report; the answer says which attempt each lane counts for."""
        reply = S.engine_report(db, day, body.station,
                                {name: lane.model_dump() for name, lane in body.lanes.items()})
        _last_report[body.station] = (clock(), body)
        return reply

    @app.post("/api/engine/frame/{station}", status_code=204)
    async def engine_frame(station: str, request: Request) -> None:
        if not FRAME_STATION.fullmatch(station):
            raise S.ServiceError("unknown_station", f"bad station name {station!r}", 400)
        data = await request.body()
        if not data.startswith(JPEG_SOI):
            raise S.ServiceError("bad_image", "expected a JPEG", 400)
        _last_frame[station] = (clock(), data)

    @app.get("/api/engine/frame/{station}.jpg")
    def engine_frame_jpg(station: str) -> Response:
        """The station camera's latest annotated picture; 404 when no engine sent one lately."""
        seen = _last_frame.get(station)
        if seen is None or clock() - seen[0] > FRAME_STALE_S:
            raise S.ServiceError("no_camera", "no live picture from this station's camera", 404)
        return Response(seen[1], media_type="image/jpeg")

    @app.post("/api/engine/crossing", status_code=201)
    def engine_crossing(body: CrossingIn, db: Session = Depends(get_db),
                        day: Day = Depends(get_day)) -> dict:
        """The finish camera saw a runner cross: a result when the bib is a known runner,
        otherwise a review-queue entry; either way with the evidence clip."""
        at = S.utcnow() - timedelta(seconds=body.latency_s)
        c = S.record_finish(db, day, S.rules_for(db, day, rules), body.bib, now=at, source="camera",
                            confidence=body.confidence, clip_path=body.clip)
        return crossing_out(c)

    @app.get("/api/lanes")
    def lanes(station: str, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """The live picture for the station tablet: engine health and per-lane counts."""
        if station not in J.STATIONS:
            raise S.ServiceError("unknown_station", f"unknown station {station!r}", 400)
        running = S.running_attempts(db, day, station)
        seen = _last_report.get(station)
        age = None if seen is None else clock() - seen[0]
        online = age is not None and age <= ENGINE_STALE_S
        out = {}
        if seen is not None:
            for name, lane in seen[1].lanes.items():
                out[name] = {"count": lane.count, "rejected": lane.rejected, "phase": lane.phase,
                             "person": lane.person, "pending_faults": lane.pending_faults,
                             "work_seconds": lane.work_seconds, "attempt_id": lane.attempt_id,
                             # The engine counts for this attempt only once it has heard of it.
                             "current": name in running and lane.attempt_id == running[name].id}
        return {"station": station,
                "engine": {"online": online, "age_s": None if age is None else round(age, 2),
                           "fps": seen[1].fps if seen else None,
                           "source": seen[1].source if seen else None},
                "lanes": out}

    # ---------------------------------------------------------------- web UI
    web = Path(web_dir or os.environ.get("TESTPOINT_WEB", "web/station"))
    if web.is_dir():
        app.mount("/ui", StaticFiles(directory=web, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse("/ui/")

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="TestPoint station app")
    p.add_argument("--host", default=os.environ.get("TESTPOINT_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TESTPOINT_PORT", "8000")))
    p.add_argument("--db", default=None, help="SQLAlchemy URL (default $TESTPOINT_DB or sqlite:///data/testpoint.db)")
    p.add_argument("--web", default=None, help="station UI folder (default $TESTPOINT_WEB or web/station)")
    args = p.parse_args(argv)
    uvicorn.run(create_app(db_url=args.db, web_dir=args.web), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
