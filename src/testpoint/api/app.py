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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

# Windows' registry may not know these; a module script or wasm with the wrong type is refused.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

import numpy as np
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
from .startline import Look, StartLine
from .station import StationFlow, StationLook, next_event
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
        "run_started_at": iso(e.run_started_at),
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
    camera_bg_places: list[Literal["weigh", "start", "photo", "finish", "station"]] | None = Field(default=None, max_length=5)
    backdrop_color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    face_match: Literal["normal", "strict"] | None = None
    face_required: bool | None = None
    walk_ins: bool | None = None
    default_gender: Literal["M", "F", ""] | None = None
    default_unit: Literal["combat", "admin", "special"] | None = None
    bib_start: int | None = Field(default=None, ge=1, le=99999)
    lanes: int | None = Field(default=None, ge=1, le=4)
    start_countdown_s: int | None = Field(default=None, ge=0, le=30)
    cancel_window_s: int | None = Field(default=None, ge=0, le=3600)
    site_name: str | None = Field(default=None, max_length=80)
    default_language: Literal["en", "ar"] | None = None
    trouble_run_cutoff_s: int | None = Field(default=None, ge=0, le=3600)
    count_tuning: dict[str, dict[str, float]] | None = None
    theme: Literal["night", "desert", "desert_night", "desert_sand", "daylight", "track", "falcon", "neural", "mirage", "desert_stars"] | None = None


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


class FinishCamIn(BaseModel):
    kind: Literal["webcam", "video", "url"]
    source: str = Field(min_length=1, max_length=500)
    line_y: float = Field(default=0.85, ge=0.05, le=0.95)
    direction: Literal["down", "up", "any"] = "down"
    app_url: str = Field(max_length=100)
    """The app as the station PC sees it (the browser's own address), for the camera's reports."""
    loop: bool = False
    """A video file: play it on repeat."""


class StartCamIn(BaseModel):
    kind: Literal["webcam", "video", "url"]
    source: str = Field(min_length=1, max_length=500)
    app_url: str = Field(max_length=100)
    loop: bool = False


class StationCamIn(BaseModel):
    kind: Literal["webcam", "video", "url"]
    source: str = Field(min_length=1, max_length=500)
    exercise: Literal["pushup", "crunch", "auto"]
    """"auto": the combined station (one person, face check-in, push-ups then crunches)."""
    lanes: int = Field(default=3, ge=1, le=4)
    dividers: list[float] = Field(default=[], max_length=3)
    app_url: str = Field(max_length=100)
    loop: bool = False


class TagIn(BaseModel):
    bib: str


class DnfIn(BaseModel):
    reason: str


class BurstIn(BaseModel):
    frames: list[str] = Field(min_length=1, max_length=12)
    """Base64 JPEG frames of one pose; the best one is kept (each at most 2 MB)."""


def burst_frames(body: BurstIn) -> list[bytes]:
    import base64
    import binascii
    out = []
    for f in body.frames:
        if len(f) > 2_800_000:
            raise S.ServiceError("too_large", "a frame is larger than 2 MB", 413)
        try:
            out.append(base64.b64decode(f.split(",", 1)[-1], validate=True))
        except (binascii.Error, ValueError):
            raise S.ServiceError("bad_image", "a frame is not base64", 400) from None
    return out


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


class ResetDayIn(BaseModel):
    confirm: str
    """Today's date, typed: a reset is never one tap away."""
    keep_list: bool = True
    stop_running: bool = False
    """Also reset exercise attempts still open (the examiner saw who and agreed)."""


class TodayIn(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)


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
    depth: float | None = Field(default=None, ge=0, le=2)
    events: list[EngineEventIn] = []
    # the combined exercise station (engine --station exercise): what it watches, the start
    # posture, and the face of the person on the mat while nobody is being tested
    exercise: str | None = None
    ready: bool = False
    face: str | None = Field(default=None, max_length=2000)


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
    face: str | None = Field(default=None, max_length=700_000)
    """One snapshot (base64 JPEG): older finish cameras."""
    faces: list[str] = Field(default=[], max_length=5)
    """Snapshots of the runner just before the line (base64 JPEG), sent when the bib was not read
    with confidence."""


ENGINE_STALE_S = 2.0
FRAME_STALE_S = 5.0
"""A camera picture older than this is not shown (the engine stopped)."""
class StartObsIn(BaseModel):
    """What the start-line camera sees at the mark (testpoint.startcam), a few times a second."""
    present: bool
    hand: Literal["left", "right"] | None = None
    emb: str | None = Field(default=None, max_length=2000)
    """The face embedding (128 float32, base64) when a face was looked at in this frame."""
    quality: float | None = None


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
               evidence_dir: str | Path | None = None, camhost_url: str | None = None) -> FastAPI:
    db_url = db_url or os.environ.get("TESTPOINT_DB", "sqlite:///data/testpoint.db")
    # Where the engine writes its clips (``testpoint-engine --evidence``); rep clip paths
    # in the database are relative to it.
    evidence = Path(evidence_dir or os.environ.get("TESTPOINT_EVIDENCE", "data/evidence")).resolve()
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:///:memory:"):
        Path(db_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    zone = ZoneInfo(tz or os.environ.get("TESTPOINT_TZ", "Asia/Dubai"))
    rules = rules or DayRules()
    SessionLocal = make_sessionmaker(db_url)
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
    _frame_waiters: dict[str, asyncio.Event] = {}   # wakes the open streams when a new picture comes

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
    async def _dnf_overdue_runners(request: Request, call_next):
        # Every station polls /api/state every 2 s: a runner past their cut-off becomes DNF here,
        # under the same lock as every other write, so nobody has to remember to close the run.
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
        return JSONResponse({"detail": {"code": e.code, "message": e.message, **e.extra}}, status_code=e.status)

    # ------------------------------------------------------------------ read
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/rules")
    def get_rules(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """Today's rules (in force), plus ``next`` when a newer version waits for the next day."""
        today_r, next_r = S.rules_for(db, day, rules, judging=False), S.current_rules(db, rules)
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

    @app.post("/api/day/reset", dependencies=[Depends(admin)])
    def day_reset(body: ResetDayIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        if body.confirm.strip() != day.date:
            raise S.ServiceError("confirm_date", f"type today's date ({day.date}) to reset it", 400)
        return S.reset_day(db, day, rules, keep_list=body.keep_list, stop_running=body.stop_running)

    @app.get("/api/settings/defaults")
    def settings_defaults() -> dict:
        """What Reset to defaults fills in: the station settings as shipped and the official rules."""
        from ..vision.exercises import tuning_defaults
        return {"settings": dict(S.SETTINGS), "rules": rules.to_dict(), "tuning": tuning_defaults()}

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
        for r in s["runners"]:
            r["started_at"], r["cutoff_at"] = iso(r["started_at"]), iso(r["cutoff_at"])
        if s["starting"]:
            s["starting"]["go_at"] = iso(s["starting"]["go_at"])
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

    # ------------------------------------------------------------------ register
    # Everyone saved, on a test day or not (service: "the register"). Static paths come before
    # /{candidate_id} so "export.csv" and "lookup" are never read as an id.
    @app.get("/api/candidates")
    def register(q: str = "", show: Literal["all", "no_face", "today", "not_today"] = "all",
                 offset: int = 0, limit: int = 100, db: Session = Depends(get_db),
                 day: Day = Depends(get_day)) -> dict:
        return S.register_rows(db, day, q, show, offset, limit)

    @app.get("/api/candidates/export.csv", dependencies=[Depends(admin)])
    def register_export(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> Response:
        return Response(S.export_register(db, day), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="testpoint-register-{day.date}.csv"'})

    @app.get("/api/candidates/lookup")
    def register_lookup(q: str, db: Session = Depends(get_db)) -> dict:
        """A person in the register by ID: the desk offers to add them to today's list."""
        return S.candidate_dict(S.lookup_candidate(db, q))

    @app.post("/api/candidates", status_code=201)
    def register_add(body: CandidateIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """Save a person without putting them on a test day."""
        data = body.model_dump(exclude={"auto_bib", "walk_in"})
        result: dict = {}
        c = S.register_candidate(db, day, **data, result=result)
        return S.candidate_dict(c) | {"outcome": result["outcome"]}

    @app.post("/api/candidates/import", dependencies=[Depends(admin)])
    async def register_import(request: Request, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return S.import_csv(db, day, S.decode_csv(await request.body()), to_day=False)

    @app.post("/api/candidates/today")
    def register_to_today(body: TodayIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return S.add_to_today(db, day, body.ids)

    @app.get("/api/candidates/{candidate_id}")
    def register_person(candidate_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return S.candidate_history(db, day, S.get_candidate(db, candidate_id))

    @app.patch("/api/candidates/{candidate_id}", dependencies=[Depends(admin)])
    def register_edit(candidate_id: int, body: CandidateEdit, db: Session = Depends(get_db),
                      day: Day = Depends(get_day)) -> dict:
        c = S.get_candidate(db, candidate_id)
        e = db.scalar(select(Entry).where(Entry.day_id == day.id, Entry.candidate_id == c.id))
        S.edit_candidate(db, day, S.rules_for(db, day, rules), e, body.model_dump(exclude_unset=True), candidate=c)
        return S.candidate_history(db, day, c)

    @app.delete("/api/candidates/{candidate_id}", status_code=204, dependencies=[Depends(admin)])
    def register_delete(candidate_id: int, delete_results: bool = False, db: Session = Depends(get_db),
                        day: Day = Depends(get_day)) -> None:
        S.delete_candidate(db, day, S.get_candidate(db, candidate_id), delete_results=delete_results)

    @app.post("/api/candidates/{candidate_id}/face")
    async def register_face(candidate_id: int, request: Request, db: Session = Depends(get_db),
                            day: Day = Depends(get_day)) -> dict:
        c = S.get_candidate(db, candidate_id)
        return {"faces": S.enroll_face(db, day, None, await request.body(), engine(), candidate=c)}

    @app.post("/api/candidates/{candidate_id}/face/burst")
    def register_face_burst(candidate_id: int, body: BurstIn, db: Session = Depends(get_db),
                            day: Day = Depends(get_day)) -> dict:
        """Several frames of one pose: the sharpest, best-lit one is kept (low_quality if none is good)."""
        c = S.get_candidate(db, candidate_id)
        return {"faces": S.enroll_face(db, day, None, burst_frames(body), engine(), candidate=c)}

    @app.delete("/api/candidates/{candidate_id}/face", status_code=204, dependencies=[Depends(admin)])
    def register_face_remove(candidate_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> None:
        S.remove_faces(db, day, None, candidate=S.get_candidate(db, candidate_id))

    @app.get("/api/candidates/{candidate_id}/face.jpg")
    def register_face_jpg(candidate_id: int, db: Session = Depends(get_db)) -> Response:
        jpg = S.face_photo(db, None, candidate_id)
        if not jpg:
            raise S.ServiceError("no_face_enrolled", "no face enrolled", 404)
        return Response(jpg, media_type="image/jpeg")

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

    @app.post("/api/entries/{entry_id}/face/burst")
    def face_enroll_burst(entry_id: int, body: BurstIn, db: Session = Depends(get_db),
                          day: Day = Depends(get_day)) -> dict:
        """Several frames of one pose: the sharpest, best-lit one is kept (low_quality if none is good)."""
        return {"faces": S.enroll_face(db, day, S.get_entry(db, day, entry_id), burst_frames(body), engine())}

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
        if station is not None and station not in J.STATIONS and station != "results":
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
    def entry_remove(entry_id: int, delete_results: bool = False, forget: bool = False,
                     db: Session = Depends(get_db), day: Day = Depends(get_day)) -> None:
        """Take someone off today's list (they stay in the register); ``delete_results=true`` also
        when they have results; ``forget=true`` for a registration made by mistake."""
        S.remove_entry(db, day, S.get_entry(db, day, entry_id), delete_results=delete_results, forget=forget)

    @app.post("/api/run/line/clear")
    def run_line_clear(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return {"cleared": S.clear_line(db, day)}

    @app.post("/api/entries/{entry_id}/start")
    def runner_start(entry_id: int, body: StartIn | None = None, db: Session = Depends(get_db),
                     day: Day = Depends(get_day)) -> dict:
        """Start one runner: 3-2-1 (the start_countdown_s setting unless given), then their clock runs."""
        countdown = body.countdown_s if body and body.countdown_s is not None else int(S.get_settings(db)["start_countdown_s"])
        e = S.start_runner(db, day, S.rules_for(db, day, rules), S.get_entry(db, day, entry_id), countdown_s=countdown)
        return entry_out(e, day)

    @app.post("/api/entries/{entry_id}/start/cancel")
    def runner_start_cancel(entry_id: int, request: Request, late: bool = False, db: Session = Depends(get_db),
                            day: Day = Depends(get_day)) -> dict:
        """A start by mistake: back to the line. ``late=true``: after the cancel window, an
        administrator decision (PIN when one is set)."""
        if late:
            S.check_admin_pin(db, request.headers.get("x-admin-pin"))
        return entry_out(S.cancel_start(db, day, S.get_entry(db, day, entry_id), late=late), day)

    @app.post("/api/run/crossings/{crossing_id}/tag")
    def run_tag(crossing_id: int, body: TagIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """Attach a bib to a time tapped at the instant of crossing."""
        return crossing_out(S.tag_crossing(db, day, S.rules_for(db, day, rules), crossing_id, body.bib))

    @app.post("/api/entries/{entry_id}/dnf")
    def entry_dnf(entry_id: int, body: DnfIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return entry_out(S.mark_dnf(db, day, S.get_entry(db, day, entry_id), body.reason), day)

    @app.post("/api/entries/{entry_id}/dnf/undo")
    def entry_dnf_undo(entry_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return entry_out(S.undo_dnf(db, day, S.get_entry(db, day, entry_id)), day)

    @app.post("/api/run/crossings/{crossing_id}/unassign")
    def run_unassign(crossing_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return crossing_out(S.unassign_crossing(db, day, crossing_id))

    @app.delete("/api/run/crossings/{crossing_id}", status_code=204)
    def run_discard(crossing_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> None:
        S.discard_crossing(db, day, crossing_id)

    @app.post("/api/run/finish", status_code=201)
    def run_finish(body: FinishIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return crossing_out(S.record_finish(db, day, S.rules_for(db, day, rules), body.bib))

    @app.post("/api/run/crossings/{crossing_id}/resolve")
    def run_resolve(crossing_id: int, body: ResolveIn, db: Session = Depends(get_db),
                    day: Day = Depends(get_day)) -> dict:
        c = S.resolve_crossing(db, day, S.rules_for(db, day, rules), crossing_id, S.get_entry(db, day, body.entry_id))
        return crossing_out(c)

    @app.post("/api/run/close")
    def run_close(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """End of the run: everyone still out becomes DNF."""
        return {"dnf": S.close_run(db, day)}

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
        if body.station == S.EXERCISE_STATION:
            flow_look(db, day, body)
        reply = S.engine_report(db, day, body.station,
                                {name: lane.model_dump() for name, lane in body.lanes.items()})
        _last_report[body.station] = (clock(), body)
        if body.station == S.EXERCISE_STATION:
            flow = station_flow(db, day)
            watch = flow.watching or flow.event or "pushup"
            for lane in reply["lanes"].values():
                lane["exercise"] = watch
            return reply | {"tuning": S.get_settings(db)["count_tuning"].get(watch, {})}
        # the counting rules to use now: a change in Settings or the tune panel reaches the engine
        # with its next report (a fraction of a second), without restarting the camera
        return reply | {"tuning": S.get_settings(db)["count_tuning"].get(body.station, {})}

    @app.post("/api/engine/frame/{station}", status_code=204)
    async def engine_frame(station: str, request: Request) -> None:
        if not FRAME_STATION.fullmatch(station):
            raise S.ServiceError("unknown_station", f"bad station name {station!r}", 400)
        data = await request.body()
        if not data.startswith(JPEG_SOI):
            raise S.ServiceError("bad_image", "expected a JPEG", 400)
        _last_frame[station] = (clock(), data)
        waiters = _frame_waiters.pop(station, None)
        if waiters is not None:
            waiters.set()

    @app.get("/api/engine/frame/{station}.jpg")
    def engine_frame_jpg(station: str) -> Response:
        """The station camera's latest annotated picture; 404 when no engine sent one lately."""
        seen = _last_frame.get(station)
        if seen is None or clock() - seen[0] > FRAME_STALE_S:
            raise S.ServiceError("no_camera", "no live picture from this station's camera", 404)
        return Response(seen[1], media_type="image/jpeg")

    @app.get("/api/engine/stream/{station}")
    async def engine_stream(station: str, request: Request):
        """The station camera's picture as a live stream (MJPEG): each frame goes to the page as
        soon as the camera sends it, nothing is polled. The stream ends when the camera stops."""
        from fastapi.responses import StreamingResponse
        if not FRAME_STATION.fullmatch(station):
            raise S.ServiceError("unknown_station", f"bad station name {station!r}", 400)

        async def frames():
            last = None
            while True:
                new = _frame_waiters.setdefault(station, asyncio.Event())
                seen = _last_frame.get(station)
                if seen is None or clock() - seen[0] > FRAME_STALE_S:
                    return                                   # no camera: the page goes back to checking
                if seen is not last:
                    last = seen
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(seen[1])).encode()
                           + b"\r\n\r\n" + seen[1] + b"\r\n")
                try:                                         # sleep until the next picture arrives
                    await asyncio.wait_for(new.wait(), 0.5)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        return

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.post("/api/engine/crossing", status_code=201)
    def engine_crossing(body: CrossingIn, db: Session = Depends(get_db),
                        day: Day = Depends(get_day)) -> dict:
        """The finish camera saw a runner cross. The face (always sent) and the bib read must agree:
        a bib's runner whose face is clearly someone else goes to review with the face's runner
        suggested; no usable bib: the face among the runners still out names them; neither: review.
        Either way with the evidence clip."""
        at = S.utcnow() - timedelta(seconds=body.latency_s)
        r = S.rules_for(db, day, rules)
        c = S.record_finish(db, day, r, body.bib, now=at, source="camera",
                            confidence=body.confidence, clip_path=body.clip)
        shots_b64 = [*body.faces, *([body.face] if body.face else [])]
        if shots_b64:
            import base64
            import binascii
            data = []
            for b in shots_b64:
                try:
                    if len(b) <= 700_000:
                        data.append(base64.b64decode(b, validate=True))
                except (binascii.Error, ValueError):
                    pass
            eng = None
            try:
                eng = engine()
            except Exception:       # no face models on this install: the bib alone decides
                eng = None
            if data and eng is not None:
                if c.needs_review:
                    S.face_for_crossing(db, day, r, c, data, eng)
                else:
                    S.check_finish_face(db, day, c, data, eng)
                db.refresh(c)
        return crossing_out(c)

    # ---------------------------------------------------------- finish camera
    # The camera runs on the station PC (testpoint.camhost): this app has no camera access or
    # vision libraries in Docker. The finish page picks the source here; the service starts it.
    camhost = (camhost_url or os.environ.get("TESTPOINT_CAMHOST", "http://127.0.0.1:8099")).rstrip("/")

    def _camhost(method: str, path: str, body: dict | None = None, timeout: float = 4.0) -> dict:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(camhost + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"content-type": "application/json"} if body is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = json.loads(e.read() or b"{}").get("detail", "refused")
            raise S.ServiceError("camera_refused", detail if isinstance(detail, str) else str(detail), 400) from None
        except (urllib.error.URLError, OSError, TimeoutError):
            raise S.ServiceError("camera_service_offline", "the camera service is not running on the station PC", 503) from None

    @app.get("/api/finishcam")
    def finishcam_status() -> dict:
        try:
            return {"available": True, **_camhost("GET", "/finish", timeout=2.0)}
        except S.ServiceError as e:
            return {"available": False, "running": False, "reason": e.code}

    @app.get("/api/finishcam/sources")
    def finishcam_sources(refresh: bool = False) -> dict:
        return _camhost("GET", f"/sources{'?refresh=true' if refresh else ''}", timeout=15.0)

    def _loopback(request: Request) -> str:
        """This app as the camera processes on the station PC reach it: the scheme and port the
        request came in on, on the loopback address, whichever device's page asked (a page
        opened on another computer has that computer's view of the address)."""
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        return f"{request.url.scheme}://127.0.0.1:{port}"

    @app.post("/api/finishcam/start")
    def finishcam_start(body: FinishCamIn, request: Request) -> dict:
        return _camhost("POST", "/finish/start", body.model_dump() | {"app_url": _loopback(request)}, timeout=10.0)

    @app.post("/api/finishcam/stop")
    def finishcam_stop() -> dict:
        return _camhost("POST", "/finish/stop", {}, timeout=10.0)

    # the exercise-station engine: same service, its own process
    @app.get("/api/stationcam")
    def stationcam_status() -> dict:
        try:
            return {"available": True, **_camhost("GET", "/station", timeout=2.0)}
        except S.ServiceError as e:
            return {"available": False, "running": False, "reason": e.code}

    @app.post("/api/stationcam/start")
    def stationcam_start(body: StationCamIn, request: Request, db: Session = Depends(get_db)) -> dict:
        tuning = S.get_settings(db)["count_tuning"].get(body.exercise, {})
        return _camhost("POST", "/station/start", body.model_dump() | {"tuning": tuning, "app_url": _loopback(request)},
                        timeout=10.0)

    @app.post("/api/stationcam/stop")
    def stationcam_stop() -> dict:
        return _camhost("POST", "/station/stop", {}, timeout=10.0)

    # the start-line camera: an IP camera's picture relayed for face check-in (testpoint.startcam)
    @app.get("/api/startcam")
    def startcam_status() -> dict:
        try:
            return {"available": True, **_camhost("GET", "/startcam", timeout=2.0)}
        except S.ServiceError as e:
            return {"available": False, "running": False, "reason": e.code}

    @app.post("/api/startcam/start")
    def startcam_start(body: StartCamIn, request: Request) -> dict:
        return _camhost("POST", "/startcam/start", body.model_dump() | {"app_url": _loopback(request)}, timeout=10.0)

    @app.get("/testpoint-ca.crt")
    def https_ca() -> Response:
        """TestPoint's local certificate authority (public part), for the devices that open this
        app over HTTPS: installed once as trusted, the browser shows no warning and allows the
        camera. Made by scripts/https-cert.ps1."""
        ca = Path(os.environ.get("TESTPOINT_CERT_DIR", "data/certs")) / "testpoint-ca.crt"
        if not ca.is_file():
            raise S.ServiceError("no_https", "HTTPS is not set up on this station", 404)
        return Response(ca.read_bytes(), media_type="application/x-x509-ca-cert",
                        headers={"Content-Disposition": 'attachment; filename="testpoint-ca.crt"'})

    @app.post("/api/startcam/stop")
    def startcam_stop() -> dict:
        return _camhost("POST", "/startcam/stop", {}, timeout=10.0)

    # --------------------------------------------------------------- start line
    # The start camera recognises the runner at the mark, asks for a raised hand, and starts
    # them (startline.py decides; here its actions are carried out on the day's list).
    lines: dict[str, StartLine] = {}

    def wall() -> float:
        return S.utcnow().replace(tzinfo=timezone.utc).timestamp()

    def at_wall(t: float | None) -> str | None:
        return None if t is None else iso(datetime.fromtimestamp(t, timezone.utc).replace(tzinfo=None))

    def start_line(db: Session, day: Day) -> StartLine:
        sl = lines.get(day.date)
        if sl is None:
            lines.clear()
            sl = lines[day.date] = StartLine(countdown_s=float(S.get_settings(db)["start_countdown_s"]))
        # the examiner may have acted meanwhile (removed from the line, started or cancelled by hand)
        if sl.entry_id is not None and sl.state in ("raise_hand", "countdown"):
            e = db.get(Entry, sl.entry_id)
            want = J.LINED_UP if sl.state == "raise_hand" else J.RUNNING
            if e is None or e.status != want:
                sl.reset(wall())
        return sl

    def start_match(db: Session, day: Day, emb: np.ndarray) -> tuple[int, bool, str | None] | None:
        gallery: dict[int, list[np.ndarray]] = {}
        for entry_id, t in db.execute(select(Entry.id, FaceTemplate.embedding).join(
                FaceTemplate, FaceTemplate.candidate_id == Entry.candidate_id).where(Entry.day_id == day.id)).all():
            gallery.setdefault(entry_id, []).append(np.frombuffer(t, np.float32))
        threshold, margin = S.FACE_MATCH.get(S.get_settings(db)["face_match"], S.FACE_MATCH["normal"])
        m = S.face_match(emb, gallery, threshold=threshold, margin=margin)
        if m.candidate_id is None:
            return None
        e = db.get(Entry, m.candidate_id)
        if e.status == J.LINED_UP:
            return e.id, True, None
        g = S.gate_for(e, day, "lineup")
        return e.id, g.ok, g.reason or None

    def startline_out(db: Session, day: Day, sl: StartLine) -> dict:
        v = sl.view(wall())
        e = db.get(Entry, v["entry_id"]) if v["entry_id"] else None
        v["entry"] = ({"id": e.id, "bib": e.candidate.bib, "name": e.candidate.name, "name_ar": e.candidate.name_ar,
                       "faces": S.face_count(db, e.candidate_id), "verdict": S.verdict(e)} if e else None)
        if e is not None and v["state"] in ("countdown", "started") and e.run_started_at is not None:
            v["go_at"] = iso(e.run_started_at)
        else:
            v["go_at"] = at_wall(v["go_at"])
        v["since"] = at_wall(v["since"])
        return v

    @app.post("/api/engine/start/obs")
    def start_obs(body: StartObsIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        import base64
        sl = start_line(db, day)
        now = wall()
        match = None
        if body.emb and sl.state == "idle":
            try:
                emb = np.frombuffer(base64.b64decode(body.emb, validate=True), np.float32)
            except ValueError:
                raise S.ServiceError("bad_input", "emb: not base64", 422) from None
            if emb.shape == (128,):
                match = start_match(db, day, emb)
        for act, entry_id in sl.observe(now, Look(body.present, match, body.hand)):
            e = db.get(Entry, entry_id)
            try:
                if act == "line_up" and e.status != J.LINED_UP:
                    S.line_up(db, day, e, by="face")
                elif act == "start":
                    S.start_runner(db, day, S.rules_for(db, day, rules), e, countdown_s=round(sl.countdown_s), by="camera")
                elif act == "cancel" and e.status == J.RUNNING:
                    S.cancel_start(db, day, e)
            except S.ServiceError as err:          # the examiner did something else meanwhile
                db.rollback()
                sl.reset(now)
                sl.reason = err.code
        return startline_out(db, day, sl)

    @app.get("/api/startline")
    def startline_view(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return startline_out(db, day, start_line(db, day))

    @app.post("/api/startline/reset")
    def startline_reset(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """The examiner clears the start line screen (the runner stays at the line)."""
        sl = start_line(db, day)
        sl.reset(wall())
        return startline_out(db, day, sl)

    # ------------------------------------------------------- exercise station
    # One station, one camera, one person at a time: recognised by face, their next test chosen
    # (push-ups, then crunches), 3-2-1 once in position, counted for the time, saved (station.py).
    flows: dict[str, StationFlow] = {}

    def station_flow(db: Session, day: Day) -> StationFlow:
        f = flows.get(day.date)
        if f is None:
            flows.clear()
            f = flows[day.date] = StationFlow(countdown_s=float(S.get_settings(db)["start_countdown_s"]),
                                              time_s=float(S.rules_for(db, day, rules).exercise_time_s))
        if f.attempt_id is not None and f.state == "testing":
            a = db.get(Attempt, f.attempt_id)
            if a is None or a.status != "running":       # saved or cancelled by the examiner
                f.reset(wall())
        return f

    def flow_match(db: Session, day: Day, emb: np.ndarray) -> tuple[int, str | None, str | None] | None:
        gallery: dict[int, list[np.ndarray]] = {}
        for entry_id, t in db.execute(select(Entry.id, FaceTemplate.embedding).join(
                FaceTemplate, FaceTemplate.candidate_id == Entry.candidate_id).where(Entry.day_id == day.id)).all():
            gallery.setdefault(entry_id, []).append(np.frombuffer(t, np.float32))
        threshold, margin = S.FACE_MATCH.get(S.get_settings(db)["face_match"], S.FACE_MATCH["normal"])
        m = S.face_match(emb, gallery, threshold=threshold, margin=margin)
        if m.candidate_id is None:
            return None
        e = db.get(Entry, m.candidate_id)
        ev = next_event(S.done_events(e), EVENTS)
        if ev is None:
            return e.id, None, "all_done"
        g = S.gate_for(e, day, ev)
        return (e.id, ev, None) if g.ok else (e.id, None, g.reason)

    def flow_act(db: Session, day: Day, flow: StationFlow, acts: list[tuple]) -> None:
        for act in acts:
            try:
                if act[0] == "start":
                    a = S.start_attempt(db, day, S.get_entry(db, day, act[1]), act[2], S.EXERCISE_LANE)
                    flow.started(a.id)
                elif act[0] == "finish":
                    a = db.get(Attempt, act[1])
                    if a is not None and a.status == "running":
                        S.finish_attempt(db, day, S.rules_for(db, day, rules), a.id,
                                         count=a.auto_count or 0, rejected=a.auto_rejected or 0)
            except S.ServiceError as err:           # the examiner did something else meanwhile
                db.rollback()
                flow.reset(wall())
                flow.reason = err.code

    def flow_look(db: Session, day: Day, body: EngineReportIn) -> None:
        import base64
        flow = station_flow(db, day)
        lane = body.lanes.get(S.EXERCISE_LANE) or next(iter(body.lanes.values()), None)
        if lane is None:
            return
        flow_act(db, day, flow, flow.tick(wall()))       # a score shown long enough clears first
        match = None
        if lane.face and flow.state == "idle":
            try:
                emb = np.frombuffer(base64.b64decode(lane.face, validate=True), np.float32)
            except ValueError:
                emb = None
            if emb is not None and emb.shape == (128,):
                match = flow_match(db, day, emb)
        ready = lane.ready and lane.exercise is not None and lane.exercise == flow.watching
        flow_act(db, day, flow, flow.observe(wall(), StationLook(lane.person, match, ready)))

    @app.get("/api/stationflow")
    def stationflow_view(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """What the exercise station screen shows (and the time limit, also without the camera)."""
        flow = station_flow(db, day)
        flow_act(db, day, flow, flow.tick(wall()))
        v = flow.view(wall())
        e = db.get(Entry, v["entry_id"]) if v["entry_id"] else None
        v["entry"] = ({"id": e.id, "bib": e.candidate.bib, "name": e.candidate.name, "name_ar": e.candidate.name_ar,
                       "faces": S.face_count(db, e.candidate_id), "done": sorted(S.done_events(e))} if e else None)
        a = db.get(Attempt, v["attempt_id"]) if v["attempt_id"] else None
        v["result"] = ({"count": a.count, "passed": a.passed, "grade": a.grade, "status": a.status} if a and a.status == "finished" else None)
        seen = _last_report.get(S.EXERCISE_STATION)
        lane = seen[1].lanes.get(S.EXERCISE_LANE) if seen and clock() - seen[0] <= ENGINE_STALE_S else None
        v["live"] = ({"count": lane.count, "rejected": lane.rejected, "phase": lane.phase, "person": lane.person,
                      "faults": lane.pending_faults, "depth": lane.depth} if lane else None)
        v["engine"] = lane is not None
        for k in ("go_at", "ends_at", "since"):
            v[k] = at_wall(v[k])
        return v

    @app.post("/api/stationflow/reset")
    def stationflow_reset(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        station_flow(db, day).reset(wall())
        return {"state": "idle"}

    @app.get("/api/lanes")
    def lanes(station: str, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """The live picture for the station tablet: engine health and per-lane counts."""
        if station not in J.STATIONS and station != S.EXERCISE_STATION:
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
                             "work_seconds": lane.work_seconds, "depth": lane.depth, "attempt_id": lane.attempt_id,
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
    p.add_argument("--reload", action="store_true",
                   help="development: restart by itself whenever the Python code changes")
    p.add_argument("--https", action="store_true", default=os.environ.get("TESTPOINT_HTTPS") == "1",
                   help="serve HTTPS with the certificate in data/certs (scripts/https-cert.ps1): "
                        "other devices can then use their camera in the browser")
    args = p.parse_args(argv)
    tls = {}
    if args.https:
        certs = Path(os.environ.get("TESTPOINT_CERT_DIR", "data/certs"))
        cert, key = certs / "server.crt", certs / "server.key"
        if not (cert.is_file() and key.is_file()):
            raise SystemExit(f"no certificate in {certs}: run scripts/https-cert.ps1 first")
        tls = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    if args.reload:
        # The reloader imports the app afresh in a child process: hand the options over by environment.
        if args.db:
            os.environ["TESTPOINT_DB"] = args.db
        if args.web:
            os.environ["TESTPOINT_WEB"] = args.web
        uvicorn.run("testpoint.api.app:create_app", factory=True, reload=True,
                    reload_dirs=[str(Path(__file__).resolve().parents[1])], host=args.host, port=args.port, **tls)
        return
    uvicorn.run(create_app(db_url=args.db, web_dir=args.web), host=args.host, port=args.port, **tls)


if __name__ == "__main__":
    main()
