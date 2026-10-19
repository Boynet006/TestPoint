"""FastAPI app: JSON API under /api, the station UI as static files under /.

    testpoint-app                     # http://0.0.0.0:8000, data in ./data/testpoint.db
    TESTPOINT_DB=sqlite:////data/testpoint.db TESTPOINT_TZ=Asia/Dubai testpoint-app

Live screens poll ``GET /api/state`` every second or two; a WebSocket can
replace that later without changing the rest of the API.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import journey as J
from . import service as S
from .models import Attempt, Crossing, Day, Entry, FaceTemplate, make_sessionmaker
from .rules import DayRules


# ------------------------------------------------------------------ serialisation

def iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat(timespec="milliseconds") + "Z"


def entry_out(e: Entry, day: Day | None = None, faces: int | None = None) -> dict:
    c = e.candidate
    events = {}
    for a in e.attempts:
        if a.status in ("running", "finished"):
            events[a.event] = {"attempt_id": a.id, "status": a.status, "count": a.count,
                               "rejected": a.rejected, "passed": a.passed, "lane": a.lane}
    out = {
        "id": e.id, "status": e.status,
        "candidate": {"id": c.id, "external_id": c.external_id, "name": c.name,
                      "name_ar": c.name_ar, "gender": c.gender, "dob": c.dob, "bib": c.bib},
        "weight_kg": e.weight_kg, "height_cm": e.height_cm, "bmi": e.bmi,
        "run_time_s": e.run_time_s, "events": events,
    }
    if faces is not None:
        out["faces"] = faces
    if day is not None:
        out["gates"] = {st: S.gate_for(e, day, st).reason or "ok" for st in J.STATIONS}
    return out


def crossing_out(c: Crossing) -> dict:
    return {"id": c.id, "entry_id": c.entry_id, "at": iso(c.at), "time_s": c.time_s,
            "bib_read": c.bib_read, "source": c.source, "needs_review": c.needs_review,
            "note": c.note}


def attempt_out(a: Attempt) -> dict:
    return {"id": a.id, "entry_id": a.entry_id, "event": a.event, "lane": a.lane,
            "status": a.status, "count": a.count, "rejected": a.rejected, "passed": a.passed,
            "started_at": iso(a.started_at), "ended_at": iso(a.ended_at)}


# ----------------------------------------------------------------------- requests

class CandidateIn(BaseModel):
    external_id: str
    name: str
    name_ar: str | None = None
    gender: str | None = Field(default=None, pattern="^[MF]$")
    dob: str | None = None
    bib: str | None = None


class CheckinIn(BaseModel):
    station: str
    query: str


class WeighInIn(BaseModel):
    weight_kg: float = Field(gt=0, lt=400)
    height_cm: float = Field(gt=50, lt=260)
    accept: bool
    override_reason: str | None = None


class FinishIn(BaseModel):
    bib: str | None = None


class ResolveIn(BaseModel):
    entry_id: int


class AttemptIn(BaseModel):
    entry_id: int
    event: str
    lane: str | None = None


class AttemptResultIn(BaseModel):
    count: int = Field(ge=0, le=500)
    rejected: int = Field(default=0, ge=0, le=500)


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
               web_dir: str | Path | None = None, today: callable = None,
               face_engine: callable = None) -> FastAPI:
    db_url = db_url or os.environ.get("TESTPOINT_DB", "sqlite:///data/testpoint.db")
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:///:memory:"):
        Path(db_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    zone = ZoneInfo(tz or os.environ.get("TESTPOINT_TZ", "Asia/Dubai"))
    rules = rules or DayRules()
    SessionLocal = make_sessionmaker(db_url)
    today = today or (lambda: datetime.now(zone).date().isoformat())

    app = FastAPI(title="TestPoint station", version="0.3.0")
    face_engine = face_engine or default_face_engine
    _engine_cache: dict = {}

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

    @app.middleware("http")
    async def _no_cache(request: Request, call_next):
        # Station devices must always get the current UI and live data, never a stale copy.
        response = await call_next(request)
        if not request.url.path.startswith("/ui/") or request.url.path.endswith((".html", "/")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    @app.exception_handler(S.ServiceError)
    async def _service_error(_: Request, e: S.ServiceError) -> JSONResponse:
        return JSONResponse({"detail": {"code": e.code, "message": e.message}}, status_code=e.status)

    # ------------------------------------------------------------------ read
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/rules")
    def get_rules() -> dict:
        return rules.to_dict()

    @app.get("/api/state")
    def state(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        s = S.summary(db, day)
        s["run_started_at"] = iso(s["run_started_at"])
        s["run_closed_at"] = iso(s["run_closed_at"])
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
        return entry_out(S.add_candidate(db, day, **body.model_dump()), day)

    @app.post("/api/entries/import")
    async def import_csv(request: Request, db: Session = Depends(get_db),
                         day: Day = Depends(get_day)) -> dict:
        text = (await request.body()).decode("utf-8-sig", errors="replace")
        return S.import_csv(db, day, text)

    # ------------------------------------------------------------------ check-in
    @app.post("/api/checkin")
    def checkin(body: CheckinIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        """Identify a candidate (bib / ID now, face later) and say whether this station may take them."""
        if body.station not in J.STATIONS:
            raise S.ServiceError("unknown_station", f"unknown station {body.station!r}", 400)
        e = S.find_entry(db, day, body.query)
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

    @app.delete("/api/entries/{entry_id}/face", status_code=204)
    def face_remove(entry_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> None:
        S.remove_faces(db, day, S.get_entry(db, day, entry_id))

    @app.get("/api/entries/{entry_id}/face.jpg")
    def face_jpg(entry_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> Response:
        jpg = S.face_photo(db, S.get_entry(db, day, entry_id))
        if not jpg:
            raise S.ServiceError("no_face_enrolled", "no face enrolled", 404)
        return Response(jpg, media_type="image/jpeg")

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
        e = S.weigh_in(db, day, rules, S.get_entry(db, day, entry_id), **body.model_dump())
        return entry_out(e, day)

    # ----------------------------------------------------------------------- run
    @app.post("/api/entries/{entry_id}/line-up")
    def line_up(entry_id: int, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return entry_out(S.line_up(db, day, S.get_entry(db, day, entry_id)), day)

    @app.post("/api/run/start")
    def run_start(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        S.start_run(db, day)
        return {"run_started_at": iso(day.run_started_at)}

    @app.post("/api/run/finish", status_code=201)
    def run_finish(body: FinishIn, db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        return crossing_out(S.record_finish(db, day, rules, body.bib))

    @app.post("/api/run/crossings/{crossing_id}/resolve")
    def run_resolve(crossing_id: int, body: ResolveIn, db: Session = Depends(get_db),
                    day: Day = Depends(get_day)) -> dict:
        c = S.resolve_crossing(db, day, rules, crossing_id, S.get_entry(db, day, body.entry_id))
        return crossing_out(c)

    @app.post("/api/run/close")
    def run_close(db: Session = Depends(get_db), day: Day = Depends(get_day)) -> dict:
        S.close_run(db, day)
        return {"run_closed_at": iso(day.run_closed_at)}

    # ------------------------------------------------------------------ exercises
    @app.post("/api/attempts", status_code=201)
    def attempt_start(body: AttemptIn, db: Session = Depends(get_db),
                      day: Day = Depends(get_day)) -> dict:
        a = S.start_attempt(db, day, S.get_entry(db, day, body.entry_id), body.event, body.lane)
        return attempt_out(a)

    @app.post("/api/attempts/{attempt_id}/finish")
    def attempt_finish(attempt_id: int, body: AttemptResultIn, db: Session = Depends(get_db),
                       day: Day = Depends(get_day)) -> dict:
        return attempt_out(S.finish_attempt(db, day, rules, attempt_id, **body.model_dump()))

    @app.post("/api/attempts/{attempt_id}/cancel", status_code=204)
    def attempt_cancel(attempt_id: int, db: Session = Depends(get_db),
                       day: Day = Depends(get_day)) -> None:
        S.cancel_attempt(db, day, attempt_id)

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
