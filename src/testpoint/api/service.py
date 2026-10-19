"""Station operations on one test day. The HTTP layer is a thin wrapper over these.

Every operation checks its gate with ``journey.check_gate`` first and raises a
``ServiceError`` with a machine-readable code when it refuses; every change is
written to the audit log.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import numpy as np

from . import journey as J
from .faces import FaceError, match as face_match
from .models import Attempt, AuditLog, Candidate, Crossing, Day, Entry, FaceTemplate, utcnow
from .rules import EVENTS, DayRules, bmi as compute_bmi


class ServiceError(Exception):
    def __init__(self, code: str, message: str = "", status: int = 409) -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.status = status


def _audit(db: Session, day: Day, station: str, action: str, entry: Entry | None = None,
           **detail) -> None:
    db.add(AuditLog(day_id=day.id, entry_id=entry.id if entry else None, station=station,
                    action=action, detail=json.dumps(detail, default=str) if detail else None))


# --------------------------------------------------------------------------- day

def get_or_create_day(db: Session, date: str, rules: DayRules) -> Day:
    # A run that is still open keeps its day, even past midnight.
    open_run = db.scalar(select(Day).where(Day.run_started_at.is_not(None), Day.run_closed_at.is_(None))
                         .order_by(Day.date.desc()))
    if open_run is not None and open_run.date < date:
        return open_run
    day = db.scalar(select(Day).where(Day.date == date))
    if day is None:
        day = Day(date=date, rules_version=rules.version)
        db.add(day)
        try:
            db.commit()
        except IntegrityError:          # another device created it first
            db.rollback()
            day = db.scalar(select(Day).where(Day.date == date))
    return day


def summary(db: Session, day: Day) -> dict:
    counts = dict(db.execute(
        select(Entry.status, func.count()).where(Entry.day_id == day.id).group_by(Entry.status)
    ).all())
    review = db.scalar(select(func.count()).select_from(Crossing).where(
        Crossing.day_id == day.id, Crossing.needs_review.is_(True)))
    return {
        "date": day.date,
        "rules_version": day.rules_version,
        "counts": {s: counts.get(s, 0) for s in J.STATUSES},
        "total": sum(counts.values()),
        "run_started_at": day.run_started_at,
        "run_closed_at": day.run_closed_at,
        "review_queue": review or 0,
    }


# --------------------------------------------------------------------- candidates

def list_entries(db: Session, day: Day, status: str | None = None) -> list[Entry]:
    q = select(Entry).where(Entry.day_id == day.id)
    if status:
        q = q.where(Entry.status.in_(status.split(",")))
    return list(db.scalars(q.join(Candidate).order_by(Candidate.bib, Candidate.name)).unique())


GENDERS = {"m": "M", "male": "M", "ذكر": "M", "f": "F", "female": "F", "أنثى": "F", "انثى": "F"}


def norm_bib(bib: str | None) -> str | None:
    """'  011 ' -> '11': numeric bibs lose leading zeros so typing 11 or 011 finds the same runner."""
    b = (bib or "").strip()
    if not b:
        return None
    return str(int(b)) if b.isdigit() else b.upper()


def norm_gender(g: str | None) -> str | None:
    g = (g or "").strip()
    if not g:
        return None
    out = GENDERS.get(g.lower()) or GENDERS.get(g)
    if out is None:
        raise ServiceError("bad_gender", f"gender {g!r}: use M or F", 400)
    return out


def norm_dob(d: str | None) -> str | None:
    d = (d or "").strip()
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise ServiceError("bad_dob", f"date of birth {d!r}: use YYYY-MM-DD", 400) from None


def add_candidate(db: Session, day: Day, *, external_id: str, name: str, name_ar: str | None = None,
                  gender: str | None = None, dob: str | None = None, bib: str | None = None,
                  commit: bool = True, result: dict | None = None) -> Entry:
    """Add a candidate to today's list (or refresh the details of one already known).

    External IDs are compared without case. A known ID with a different name is refused
    rather than silently renaming that person. The bib a candidate ends up with must be
    unique among today's entries. ``result['outcome']`` says added / updated / unchanged.
    """
    external_id = (external_id or "").strip()
    name = " ".join((name or "").split())
    if not external_id or not name:
        raise ServiceError("missing_fields", "external_id and name are required", 400)
    bib = norm_bib(bib)
    gender = norm_gender(gender)
    dob = norm_dob(dob)
    name_ar = " ".join((name_ar or "").split()) or None
    cand = db.scalar(select(Candidate).where(func.lower(Candidate.external_id) == external_id.lower()))
    if cand is not None and cand.name.casefold() != name.casefold():
        raise ServiceError("external_id_in_use",
                           f"ID {cand.external_id} already belongs to {cand.name}", 409)
    effective_bib = bib or (cand.bib if cand else None)
    if effective_bib:
        clash = db.scalar(select(Entry).join(Candidate).where(
            Entry.day_id == day.id, Candidate.bib == effective_bib,
            *( [Candidate.id != cand.id] if cand is not None else [] )))
        if clash:
            raise ServiceError("bib_in_use", f"bib {effective_bib}, {clash.candidate.name}")
    outcome = "unchanged"
    if cand is None:
        cand = Candidate(external_id=external_id, name=name, name_ar=name_ar, gender=gender, dob=dob, bib=bib)
        db.add(cand)
        db.flush()
        outcome = "added"
    else:
        new = {"name_ar": name_ar or cand.name_ar, "gender": gender or cand.gender,
               "dob": dob or cand.dob, "bib": effective_bib}
        if any(getattr(cand, k) != v for k, v in new.items()):
            for k, v in new.items():
                setattr(cand, k, v)
            outcome = "updated"
    entry = db.scalar(select(Entry).where(Entry.day_id == day.id, Entry.candidate_id == cand.id))
    if entry is None:
        entry = Entry(day_id=day.id, candidate_id=cand.id, status=J.EXPECTED)
        db.add(entry)
        db.flush()
        outcome = "added"
        _audit(db, day, "admin", "add_candidate", entry, external_id=cand.external_id, bib=effective_bib)
    if result is not None:
        result["outcome"] = outcome
    if commit:
        db.commit()
    return entry


CSV_COLUMNS = ("external_id", "name", "name_ar", "gender", "dob", "bib")


def decode_csv(data: bytes) -> str:
    """UTF-8 (with or without BOM) first; Arabic Excel on Windows saves CP-1256."""
    for enc in ("utf-8-sig", "cp1256"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ServiceError("bad_csv", "the file is not UTF-8 or Arabic Windows text", 400)


def import_csv(db: Session, day: Day, text: str) -> dict:
    """Header row required; columns external_id,name[,name_ar,gender,dob,bib] in any order."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")), restkey="__extra__")
    header = [(h or "").strip().lower() for h in (reader.fieldnames or [])]
    if "external_id" not in header or "name" not in header:
        raise ServiceError("bad_csv", "the header must contain external_id and name", 400)
    counts = {"added": 0, "updated": 0, "unchanged": 0}
    errors, seen = [], {}
    for line_no, raw in enumerate(reader, start=2):
        extra = [v for v in (raw.pop("__extra__", None) or []) if (v or "").strip()]
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items() if isinstance(v, str) or v is None}
        if not any(row.values()) and not extra:
            continue                                   # blank line
        if extra:
            errors.append({"line": line_no, "code": "extra_columns", "message": f"extra values {extra}"})
            continue
        key = row.get("external_id", "").lower()
        if key and key in seen:
            errors.append({"line": line_no, "code": "duplicate_row", "message": f"same ID as line {seen[key]}"})
            continue
        seen[key] = line_no
        result: dict = {}
        try:
            with db.begin_nested():
                add_candidate(db, day, commit=False, result=result,
                              **{c: row.get(c) or None for c in CSV_COLUMNS if c in row}
                              | {"external_id": row.get("external_id", ""), "name": row.get("name", "")})
            counts[result["outcome"]] += 1
        except ServiceError as e:
            errors.append({"line": line_no, "code": e.code, "message": e.message})
    db.commit()
    return counts | {"errors": errors}


def find_entry(db: Session, day: Day, query: str) -> Entry:
    """By bib, external ID (any case) or entry id (``#12``). Refuses an ambiguous query."""
    q = (query or "").strip()
    if not q:
        raise ServiceError("empty_query", "type a bib or ID", 400)
    base = select(Entry).join(Candidate).where(Entry.day_id == day.id)
    if q.startswith("#") and q[1:].isdigit():
        entry = db.scalar(base.where(Entry.id == int(q[1:])))
    else:
        by_bib = db.scalar(base.where(Candidate.bib == norm_bib(q)))
        by_id = db.scalar(base.where(func.lower(Candidate.external_id) == q.lower()))
        if by_bib is not None and by_id is not None and by_bib.id != by_id.id:
            raise ServiceError("ambiguous_query",
                               f"{q} is bib of {by_bib.candidate.name} and ID of {by_id.candidate.name}", 409)
        entry = by_bib or by_id
    if entry is None:
        raise ServiceError("not_on_todays_list", f"{q} is not on today's list", 404)
    return entry


def get_entry(db: Session, day: Day, entry_id: int) -> Entry:
    entry = db.get(Entry, entry_id)
    if entry is None or entry.day_id != day.id:
        raise ServiceError("not_on_todays_list", f"entry {entry_id} is not on today's list", 404)
    return entry


def done_events(entry: Entry) -> set[str]:
    return {a.event for a in entry.attempts if a.status == "finished"}


def gate_for(entry: Entry, day: Day, station: str) -> J.Gate:
    return J.check_gate(entry.status, station, done_events(entry), day.run_started_at is not None)


def require_gate(entry: Entry, day: Day, station: str) -> None:
    g = gate_for(entry, day, station)
    if not g.ok:
        raise ServiceError(g.reason, f"{entry.candidate.name}: {g.reason.replace('_', ' ')}")


# --------------------------------------------------------------------------- faces

MAX_TEMPLATES = 3


def _face(engine, data: bytes, require_single: bool):
    if engine is None:
        raise ServiceError("face_unavailable", "face recognition is not available on this server", 503)
    try:
        return engine.face_from_bytes(data, require_single=require_single)
    except FaceError as e:
        raise ServiceError(e.code, e.message, 422) from None


def face_count(db: Session, candidate_id: int) -> int:
    return db.scalar(select(func.count()).select_from(FaceTemplate)
                     .where(FaceTemplate.candidate_id == candidate_id)) or 0


def enroll_face(db: Session, day: Day, entry: Entry, data: bytes, engine) -> int:
    """Store the face in ``data`` for this candidate; keeps the newest MAX_TEMPLATES."""
    face = _face(engine, data, require_single=True)
    # Guard against a mixed-up photo: refuse a face that already belongs to someone else.
    others: dict[int, list[np.ndarray]] = {}
    for cid, emb in db.execute(select(FaceTemplate.candidate_id, FaceTemplate.embedding)
                               .where(FaceTemplate.candidate_id != entry.candidate_id)).all():
        others.setdefault(cid, []).append(np.frombuffer(emb, np.float32))
    clash = face_match(face.embedding, others, margin=0.0)
    if clash.candidate_id is not None:
        owner = db.get(Candidate, clash.candidate_id)
        raise ServiceError("face_belongs_to_other",
                           f"{owner.name}, bib {owner.bib}, match {clash.score}", 409)
    db.add(FaceTemplate(candidate_id=entry.candidate_id,
                        embedding=np.asarray(face.embedding, np.float32).tobytes(),
                        photo_jpeg=face.crop_jpeg))
    db.flush()
    old = list(db.scalars(select(FaceTemplate).where(FaceTemplate.candidate_id == entry.candidate_id)
                          .order_by(FaceTemplate.created_at.desc(), FaceTemplate.id.desc()))
               )[MAX_TEMPLATES:]
    for t in old:
        db.delete(t)
    _audit(db, day, "admin", "enroll_face", entry, detection_score=face.score)
    db.commit()
    return face_count(db, entry.candidate_id)


def remove_faces(db: Session, day: Day, entry: Entry) -> None:
    for t in db.scalars(select(FaceTemplate).where(FaceTemplate.candidate_id == entry.candidate_id)):
        db.delete(t)
    _audit(db, day, "admin", "remove_faces", entry)
    db.commit()


def face_photo(db: Session, entry: Entry) -> bytes | None:
    t = db.scalar(select(FaceTemplate).where(FaceTemplate.candidate_id == entry.candidate_id)
                  .order_by(FaceTemplate.created_at.desc(), FaceTemplate.id.desc()))
    return t.photo_jpeg if t else None


def identify(db: Session, day: Day, data: bytes, engine, station: str | None = None) -> dict:
    """Who is in this frame, among today's candidates with an enrolled face?"""
    face = _face(engine, data, require_single=False)
    rows = db.execute(select(Entry.id, FaceTemplate.embedding).join(
        FaceTemplate, FaceTemplate.candidate_id == Entry.candidate_id).where(Entry.day_id == day.id)).all()
    gallery: dict[int, list[np.ndarray]] = {}
    for entry_id, emb in rows:
        gallery.setdefault(entry_id, []).append(np.frombuffer(emb, np.float32))
    m = face_match(face.embedding, gallery)
    out = {"box": list(face.box), "score": m.score, "runner_up": m.runner_up,
           "confident": m.confident, "enrolled_today": len(gallery), "entry": None,
           "allowed": None, "reason": None}
    if m.candidate_id is not None:
        entry = db.get(Entry, m.candidate_id)
        out["entry"] = entry
        if station:
            g = gate_for(entry, day, station)
            out["allowed"], out["reason"] = g.ok, (g.reason or None)
    elif not gallery:
        out["reason"] = "nobody_enrolled"
    else:
        out["reason"] = "face_not_recognised"
    _audit(db, day, station or "checkin", "identify", out["entry"], score=m.score,
           runner_up=m.runner_up, confident=m.confident)
    db.commit()
    return out


# ----------------------------------------------------------------------- weigh-in

def weigh_in(db: Session, day: Day, rules: DayRules, entry: Entry, *, weight_kg: float,
             height_cm: float, accept: bool, override_reason: str | None = None) -> Entry:
    require_gate(entry, day, "weigh")
    try:
        value = compute_bmi(weight_kg, height_cm)
    except ValueError as e:
        raise ServiceError("bad_measurement", str(e), 400) from None
    in_range = rules.bmi_min <= value <= rules.bmi_max
    if not accept and in_range and not (override_reason or "").strip():
        raise ServiceError("reason_required",
                           f"BMI {value} is within the accepted range; rejecting needs a reason")
    if accept and not in_range and not (override_reason or "").strip():
        raise ServiceError("bmi_out_of_range",
                           f"BMI {value} is outside {rules.bmi_min}-{rules.bmi_max}; "
                           "accepting needs an override reason")
    entry.weight_kg, entry.height_cm, entry.bmi = weight_kg, height_cm, value
    entry.weighed_at = utcnow()
    entry.status = J.after_weigh_in(accept)
    _audit(db, day, "weigh", "weigh_in", entry, weight_kg=weight_kg, height_cm=height_cm,
           bmi=value, accepted=accept, override_reason=override_reason or None)
    db.commit()
    return entry


# ---------------------------------------------------------------------------- run

def line_up(db: Session, day: Day, entry: Entry) -> Entry:
    require_gate(entry, day, "lineup")
    entry.status = J.LINED_UP
    entry.lined_up_at = utcnow()
    _audit(db, day, "start", "line_up", entry)
    db.commit()
    return entry


def undo_line_up(db: Session, day: Day, entry: Entry) -> Entry:
    """Marked at the line by mistake: back to accepted (only before the start)."""
    if day.run_started_at is not None:
        raise ServiceError("run_already_started", "the run has already started")
    if entry.status != J.LINED_UP:
        raise ServiceError("not_lined_up", f"{entry.candidate.name} is not at the line")
    entry.status = J.ACCEPTED
    entry.lined_up_at = None
    _audit(db, day, "start", "undo_line_up", entry)
    db.commit()
    return entry


def undo_weigh_in(db: Session, day: Day, entry: Entry, reason: str | None = None) -> Entry:
    """Wrong weight or height typed: back to expected so the weigh-in can be redone."""
    if entry.status == J.LINED_UP:
        raise ServiceError("remove_from_line_first", f"{entry.candidate.name} is at the line; remove them from the line first")
    if entry.status not in (J.ACCEPTED, J.REJECTED):
        raise ServiceError("cannot_undo_weigh_in", f"{entry.candidate.name} is {entry.status}")
    if day.run_started_at is not None:
        raise ServiceError("run_already_started", "the run has already started")
    old = {"weight_kg": entry.weight_kg, "height_cm": entry.height_cm, "bmi": entry.bmi, "status": entry.status}
    entry.status = J.EXPECTED
    entry.weight_kg = entry.height_cm = entry.bmi = None
    entry.weighed_at = None
    _audit(db, day, "admin", "undo_weigh_in", entry, reason=reason or None, **old)
    db.commit()
    return entry


def remove_entry(db: Session, day: Day, entry: Entry) -> None:
    """Added to today's list by mistake. Only before anything happened to them today."""
    if entry.status != J.EXPECTED or entry.attempts:
        raise ServiceError("cannot_remove", f"{entry.candidate.name} is already {entry.status}; it cannot be removed")
    cand = entry.candidate
    for a in db.scalars(select(AuditLog).where(AuditLog.entry_id == entry.id)):
        a.entry_id = None   # keep the history, drop the link to the deleted row
    _audit(db, day, "admin", "remove_from_day", None, external_id=cand.external_id, name=cand.name, bib=cand.bib)
    db.delete(entry)
    db.commit()


CANCEL_START_WINDOW_S = 120


def cancel_start(db: Session, day: Day, now: datetime | None = None) -> Day:
    """START pressed by mistake: undo it within 2 minutes, before anyone crossed the finish."""
    if day.run_started_at is None:
        raise ServiceError("run_not_started", "the run has not started")
    if day.run_closed_at is not None:
        raise ServiceError("run_already_closed", "the run is closed")
    if (now or utcnow()) - day.run_started_at > timedelta(seconds=CANCEL_START_WINDOW_S):
        raise ServiceError("too_late_to_cancel", "the start can only be cancelled in the first 2 minutes")
    if db.scalar(select(func.count()).select_from(Crossing).where(Crossing.day_id == day.id)):
        raise ServiceError("finishers_exist", "runners have already crossed the finish line")
    for e in list_entries(db, day):
        if e.status == J.RUNNING:
            e.status = J.LINED_UP
        elif e.status == J.ABSENT:
            e.status = J.ACCEPTED
    _audit(db, day, "start", "cancel_start", started_at=day.run_started_at)
    day.run_started_at = None
    db.commit()
    return day


def start_run(db: Session, day: Day, now: datetime | None = None) -> Day:
    if day.run_started_at is not None:
        raise ServiceError("run_already_started", "the run has already started")
    entries = list_entries(db, day)
    if not any(e.status == J.LINED_UP for e in entries):
        raise ServiceError("nobody_lined_up", "nobody is at the start line")
    day.run_started_at = now or utcnow()
    moved = {}
    for e in entries:
        new = J.after_run_start(e.status)
        if new != e.status:
            moved[new] = moved.get(new, 0) + 1
            e.status = new
    _audit(db, day, "start", "mass_start", at=day.run_started_at, **moved)
    db.commit()
    return day


def _elapsed(day: Day, at: datetime) -> float:
    return round((at - day.run_started_at).total_seconds(), 1)


def record_finish(db: Session, day: Day, rules: DayRules, bib: str | None,
                  now: datetime | None = None, source: str = "manual") -> Crossing:
    if day.run_started_at is None:
        raise ServiceError("run_not_started", "the run has not started")
    if day.run_closed_at is not None:
        raise ServiceError("run_already_closed", "the run is closed")
    at = now or utcnow()
    crossing = Crossing(day_id=day.id, at=at, time_s=_elapsed(day, at), source=source,
                        bib_read=norm_bib(bib))
    entry = None
    if crossing.bib_read:
        entry = db.scalar(select(Entry).join(Candidate).where(
            Entry.day_id == day.id, Candidate.bib == crossing.bib_read))
    if entry is not None and entry.status == J.RUNNING:
        _assign(day, rules, crossing, entry)
    else:
        crossing.needs_review = True
        if crossing.bib_read is None:
            crossing.note = "no bib read"
        elif entry is None:
            crossing.note = f"bib {crossing.bib_read} is not on today's list"
        else:
            crossing.note = f"bib {crossing.bib_read} is {entry.status}, not running"
    db.add(crossing)
    db.flush()
    _audit(db, day, "finish", "crossing", entry, bib=crossing.bib_read, time_s=crossing.time_s,
           needs_review=crossing.needs_review)
    db.commit()
    return crossing


def _assign(day: Day, rules: DayRules, crossing: Crossing, entry: Entry) -> None:
    crossing.entry_id = entry.id
    crossing.needs_review = False
    entry.run_time_s = crossing.time_s
    entry.finished_at = crossing.at
    entry.status = J.after_finish(crossing.time_s, rules.run_pass_s, rules.run_cutoff_s)


def resolve_crossing(db: Session, day: Day, rules: DayRules, crossing_id: int,
                     entry: Entry) -> Crossing:
    crossing = db.get(Crossing, crossing_id)
    if crossing is None or crossing.day_id != day.id:
        raise ServiceError("not_found", f"crossing {crossing_id} not found", 404)
    if not crossing.needs_review:
        raise ServiceError("already_resolved", "this crossing is already assigned")
    if entry.status != J.RUNNING:
        raise ServiceError("not_running", f"{entry.candidate.name} is {entry.status}, not running")
    _assign(day, rules, crossing, entry)
    crossing.note = (crossing.note or "") + " | assigned by examiner"
    _audit(db, day, "finish", "resolve_crossing", entry, crossing_id=crossing_id)
    db.commit()
    return crossing


def unassign_crossing(db: Session, day: Day, crossing_id: int) -> Crossing:
    """Wrong bib typed at the finish: the crossing goes back to the review queue and
    the runner back to running (their real crossing can then be assigned)."""
    crossing = db.get(Crossing, crossing_id)
    if crossing is None or crossing.day_id != day.id:
        raise ServiceError("not_found", f"crossing {crossing_id} not found", 404)
    if day.run_closed_at is not None:
        raise ServiceError("run_already_closed", "the run is closed")
    if crossing.needs_review or crossing.entry_id is None:
        raise ServiceError("not_assigned", "this crossing is not assigned to anyone")
    entry = db.get(Entry, crossing.entry_id)
    if any(a.status in ("running", "finished") for a in entry.attempts):
        raise ServiceError("results_exist", f"{entry.candidate.name} already has exercise results")
    entry.status = J.RUNNING
    entry.run_time_s = None
    entry.finished_at = None
    crossing.entry_id = None
    crossing.needs_review = True
    crossing.note = f"unassigned from bib {entry.candidate.bib} by examiner"
    _audit(db, day, "finish", "unassign_crossing", entry, crossing_id=crossing_id, time_s=crossing.time_s)
    db.commit()
    return crossing


def discard_crossing(db: Session, day: Day, crossing_id: int) -> None:
    """A review-queue crossing that was not a runner (double press, official, spectator)."""
    crossing = db.get(Crossing, crossing_id)
    if crossing is None or crossing.day_id != day.id:
        raise ServiceError("not_found", f"crossing {crossing_id} not found", 404)
    if not crossing.needs_review:
        raise ServiceError("already_resolved", "this crossing is assigned; unassign it first")
    _audit(db, day, "finish", "discard_crossing", crossing_id=crossing_id, time_s=crossing.time_s,
           bib=crossing.bib_read, note=crossing.note)
    db.delete(crossing)
    db.commit()


def list_crossings(db: Session, day: Day) -> list[Crossing]:
    return list(db.scalars(select(Crossing).where(Crossing.day_id == day.id).order_by(Crossing.at)))


def close_run(db: Session, day: Day, now: datetime | None = None) -> Day:
    if day.run_started_at is None:
        raise ServiceError("run_not_started", "the run has not started")
    if day.run_closed_at is not None:
        raise ServiceError("run_already_closed", "the run is already closed")
    pending = db.scalar(select(func.count()).select_from(Crossing).where(
        Crossing.day_id == day.id, Crossing.needs_review.is_(True)))
    if pending:
        raise ServiceError("review_pending",
                           f"{pending} crossing(s) still wait for review; assign or discard them first")
    day.run_closed_at = now or utcnow()
    dnf = 0
    for e in list_entries(db, day, J.RUNNING):
        e.status = J.after_close(e.status)
        dnf += 1
    _audit(db, day, "finish", "close_run", dnf=dnf)
    db.commit()
    return day


# ---------------------------------------------------------------------- exercises

def start_attempt(db: Session, day: Day, entry: Entry, event: str, lane: str | None) -> Attempt:
    if event not in EVENTS:
        raise ServiceError("unknown_event", f"unknown event {event!r}", 400)
    require_gate(entry, day, event)
    if any(a.status == "running" for a in entry.attempts):
        raise ServiceError("attempt_in_progress", f"{entry.candidate.name} is already being tested")
    attempt = Attempt(entry_id=entry.id, event=event, lane=lane, status="running")
    db.add(attempt)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise ServiceError("attempt_in_progress", f"{entry.candidate.name} is already being tested") from None
    _audit(db, day, event, "start_attempt", entry, attempt_id=attempt.id, lane=lane)
    db.commit()
    db.refresh(entry)
    return attempt


def finish_attempt(db: Session, day: Day, rules: DayRules, attempt_id: int, *, count: int,
                   rejected: int = 0) -> Attempt:
    attempt = db.get(Attempt, attempt_id)
    if attempt is None or attempt.entry.day_id != day.id:
        raise ServiceError("not_found", f"attempt {attempt_id} not found", 404)
    if attempt.status != "running":
        raise ServiceError("attempt_not_running", "this attempt is already finished")
    if count < 0 or rejected < 0:
        raise ServiceError("bad_count", "counts cannot be negative", 400)
    attempt.count, attempt.rejected = count, rejected
    attempt.passed = count >= rules.exercise_pass(attempt.event)
    attempt.ended_at = utcnow()
    attempt.status = "finished"
    entry = attempt.entry
    entry.status = J.after_event(done_events(entry))
    _audit(db, day, attempt.event, "finish_attempt", entry, attempt_id=attempt.id,
           count=count, rejected=rejected, passed=attempt.passed)
    db.commit()
    return attempt


def cancel_attempt(db: Session, day: Day, attempt_id: int) -> None:
    attempt = db.get(Attempt, attempt_id)
    if attempt is None or attempt.entry.day_id != day.id or attempt.status != "running":
        raise ServiceError("not_found", f"no running attempt {attempt_id}", 404)
    attempt.status = "cancelled"
    attempt.ended_at = utcnow()
    _audit(db, day, attempt.event, "cancel_attempt", attempt.entry, attempt_id=attempt.id)
    db.commit()
