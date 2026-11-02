"""Station operations on one test day. The HTTP layer is a thin wrapper over these.

Every operation checks its gate with ``journey.check_gate`` first and raises a
``ServiceError`` with a machine-readable code when it refuses; every change is
written to the audit log.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import numpy as np

from . import journey as J
from .faces import FaceError, match as face_match
from .models import Attempt, AuditLog, Candidate, Crossing, Day, Entry, FaceTemplate, Rep, Setting, Wave, utcnow
from .rules import EVENTS, GRADES, UNITS, DayRules, age_on, bmi as compute_bmi


class ServiceError(Exception):
    def __init__(self, code: str, message: str = "", status: int = 409, **extra) -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.status = status
        self.extra = extra
        """More for the page (e.g. who already has this face), sent next to code and message."""


def _audit(db: Session, day: Day, station: str, action: str, entry: Entry | None = None,
           **detail) -> None:
    db.add(AuditLog(day_id=day.id, entry_id=entry.id if entry else None, station=station,
                    action=action, detail=json.dumps(detail, default=str) if detail else None))


# --------------------------------------------------------------------------- day

def current_rules(db: Session, seed: DayRules) -> DayRules:
    """The rule set for days that have not started yet: the saved one, else ``seed``."""
    row = db.get(Setting, "rules")
    return DayRules.from_dict(json.loads(row.value)) if row is not None else seed


def rules_for(db: Session, day: Day, seed: DayRules, judging: bool = True) -> DayRules:
    """The rules ``day`` is judged by. Fixed once the day starts, so two candidates on the
    same day are never scored against different numbers. ``judging``: with the troubleshooting
    run cut-off when one is set (a setting, shown on every screen); False for the rules as saved."""
    r = DayRules.from_dict(json.loads(day.rules_json)) if day.rules_json else current_rules(db, seed)
    cut = (get_settings(db).get("trouble_run_cutoff_s") or 0) if judging else 0
    return dataclasses.replace(r, run_cutoff_s=float(cut)) if cut else r


def day_has_results(db: Session, day: Day) -> bool:
    if db.scalar(select(func.count()).select_from(Entry).where(Entry.day_id == day.id, Entry.status != J.EXPECTED)):
        return True
    return bool(db.scalar(select(func.count()).select_from(Attempt).join(Entry).where(Entry.day_id == day.id)))


def update_rules(db: Session, day: Day, seed: DayRules, changes: dict) -> dict:
    """Save a new rule set (a new version). It applies to today only while today has no
    result yet; otherwise from the next test day."""
    old = current_rules(db, seed)
    data = {**old.to_dict(), **{k: v for k, v in changes.items() if v is not None}}
    n = int(old.version[1:].split("-")[0]) + 1 if old.version.startswith("v") and old.version[1:].split("-")[0].isdigit() else 2
    data["version"] = f"v{n}"
    try:
        new = DayRules.from_dict(data)
        new.check()
    except (ValueError, KeyError, TypeError) as e:
        raise ServiceError("bad_rules", str(e), 400) from None
    old_d, new_d = old.to_dict(), new.to_dict()
    if {k: v for k, v in new_d.items() if k != "version"} == {k: v for k, v in old_d.items() if k != "version"}:
        return {"rules": rules_for(db, day, seed).to_dict(), "next": old.to_dict(), "applies_today": True}
    # Days from before rule versions were stored keep the rules they were judged by.
    for d in db.scalars(select(Day).where(Day.rules_json.is_(None))):
        d.rules_json = json.dumps(old.to_dict())
    row = db.get(Setting, "rules")
    if row is None:
        db.add(Setting(key="rules", value=json.dumps(new.to_dict())))
    else:
        row.value, row.updated_at = json.dumps(new.to_dict()), utcnow()
    today = not day_has_results(db, day)
    if today:
        day.rules_json, day.rules_version = json.dumps(new.to_dict()), new.version
    _audit(db, day, "admin", "rules", None, version=new.version, applies_today=today,
           changed={k: {"from": old_d.get(k), "to": new_d[k]} for k in new_d
                    if k != "version" and old_d.get(k) != new_d[k]})
    db.commit()
    return {"rules": rules_for(db, day, seed).to_dict(), "next": new.to_dict(), "applies_today": today}


def get_or_create_day(db: Session, date: str, rules: DayRules) -> Day:
    # Runners still out on the course keep their day, even past midnight (they become DNF by
    # themselves after the cut-off, see auto_close, so this never lasts long).
    open_run = db.scalar(select(Day).join(Entry, Entry.day_id == Day.id)
                         .where(Entry.status == J.RUNNING)
                         .order_by(Day.date.desc()))
    if open_run is not None and open_run.date < date:
        return open_run
    day = db.scalar(select(Day).where(Day.date == date))
    if day is None:
        r = current_rules(db, rules)
        day = Day(date=date, rules_version=r.version, rules_json=json.dumps(r.to_dict()))
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
    now = utcnow()
    # the runners out on the course (and the one in their 3-2-1), for every screen's clocks
    runners = [{"entry_id": e.id, "bib": e.candidate.bib, "name": e.candidate.name, "name_ar": e.candidate.name_ar,
                "started_at": e.run_started_at, "cutoff_at": e.run_cutoff_at}
               for e in list_entries(db, day, J.RUNNING) if e.run_started_at is not None]
    runners.sort(key=lambda r: r["started_at"])
    last = db.scalar(select(func.max(Entry.run_started_at)).where(Entry.day_id == day.id))
    return {
        "date": day.date,
        "rules_version": day.rules_version,
        "counts": {s: counts.get(s, 0) for s in J.STATUSES},
        "total": sum(counts.values()),
        "runners": runners,
        "starting": next(({"entry_id": r["entry_id"], "go_at": r["started_at"]} for r in runners if r["started_at"] > now), None),
        "run_started_at": last,
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


# Unit types as they appear on lists: English codes or the Arabic wording of the pass table.
UNIT_NAMES = {"combat": "combat", "c": "combat", "قتالية": "combat", "قتالي": "combat", "مقاتلة": "combat",
              "admin": "admin", "administrative": "admin", "a": "admin", "إدارية": "admin", "ادارية": "admin", "إداري": "admin", "اداري": "admin",
              "special": "special", "s": "special", "خاصة": "special", "ذات الطبيعة الخاصة": "special"}


def norm_unit(u: str | None) -> str | None:
    u = " ".join((u or "").split())
    if not u:
        return None
    out = UNIT_NAMES.get(u.lower())
    if out is None:
        raise ServiceError("bad_unit", f"unit type {u!r}: use {', '.join(UNITS)}", 400)
    return out


def norm_dob(d: str | None) -> str | None:
    d = (d or "").strip()
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise ServiceError("bad_dob", f"date of birth {d!r}: use YYYY-MM-DD", 400) from None


# ---------------------------------------------------------------------- settings

# Station settings: shared by every device, changed in Advanced -> Settings, each change audited.
# Per-device choices (role, simple/advanced, camera source, a language picked on that device)
# stay in the browser.
SETTINGS: dict[str, object] = {
    # check-in and weigh-in
    "auto_identify": True,       # identify faces continuously when the weigh-in camera is on
    "camera_background": "off",  # camera picture on screen: "off", "blur", "backdrop" (a colour) or "image" (recognition uses the raw frame)
    "camera_bg_places": ["weigh", "start", "photo"],   # where it is shown: weigh, start, photo (face windows), finish, station (display only)
    "backdrop_color": "#8a949e", # the colour for "backdrop" (#rrggbb)
    "face_match": "normal",      # "normal" or "strict" (see FACE_MATCH)
    "face_required": False,      # at weigh-in, people with a face on file must check in by face
    # registration
    "walk_ins": True,            # the weigh-in desk may register people who are not on the list
    "default_gender": "M",       # new candidates: "M", "F" or "" (none)
    "default_unit": "combat",    # unit type when a list or walk-in gives none: combat, admin or special
    "bib_start": 1,              # first number for automatically assigned bibs
    # exercise station
    "lanes": 3,                  # lanes per exercise station (must match the engine's --lanes)
    # run start
    "start_countdown_s": 3,      # the 3-2-1 before a runner's GO (their clock starts at GO)
    "cancel_window_s": 120,      # a start can be cancelled without the admin PIN this long; 0 = until the first finisher
    # display
    "site_name": "",             # shown in the header; empty = "Fitness examination station"
    "default_language": "en",    # a device that never chose a language: "en" or "ar"
    "trouble_run_cutoff_s": 0,
    "count_tuning": {},          # counting rules changed from the calibrated ones: {exercise: {rule: value}}   # troubleshooting: a short run cut-off in seconds instead of the rule's (0 = off)
    "theme": "night",            # colours of every screen: night, desert, desert_night, desert_sand, daylight, track, falcon, neural, mirage or desert_stars (a device may pick its own)
}
# Face match thresholds: similarity needed, and the lead over the second-best person.
# Normal is SFace's published operating point; strict asks more of both, so fewer faces are
# matched automatically and more are typed in, but a wrong match is less likely still.
FACE_MATCH = {"normal": (0.363, 0.05), "strict": (0.45, 0.08)}
ADMIN_PIN_KEY = "admin_pin_hash"   # stored next to the settings, never sent to a browser
BACKGROUND_KEY = "background_image"  # the uploaded camera background (base64 JPEG/PNG), shared by every desk
BACKGROUND_MAX = 3 * 1024 * 1024


def get_background(db: Session) -> tuple[bytes, str] | None:
    row = db.get(Setting, BACKGROUND_KEY)
    if row is None:
        return None
    v = json.loads(row.value)
    import base64
    return base64.b64decode(v["data"]), v["type"]


def set_background(db: Session, day: Day, data: bytes | None) -> None:
    """Save (or with None remove) the camera background picture. JPEG or PNG, up to 3 MB
    (the page scales it to 1920 px before sending)."""
    import base64
    row = db.get(Setting, BACKGROUND_KEY)
    if data is None:
        if row is not None:
            db.delete(row)
            _audit(db, day, "admin", "background_image", None, removed=True)
    else:
        if len(data) > BACKGROUND_MAX:
            raise ServiceError("too_large", "the picture is larger than 3 MB", 413)
        kind = "image/jpeg" if data[:3] == bytes([0xFF, 0xD8, 0xFF]) else "image/png" if data[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) else None
        if kind is None:
            raise ServiceError("bad_image", "use a JPEG or PNG picture", 400)
        value = json.dumps({"type": kind, "data": base64.b64encode(data).decode("ascii")})
        if row is None:
            db.add(Setting(key=BACKGROUND_KEY, value=value))
        else:
            row.value, row.updated_at = value, utcnow()
        _audit(db, day, "admin", "background_image", None, bytes=len(data), type=kind)
    db.commit()


def _pin_hash(pin: str, salt: str) -> str:
    import hashlib
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt.encode(), 100_000).hex()


def pin_is_set(db: Session) -> bool:
    return db.get(Setting, ADMIN_PIN_KEY) is not None


def check_admin_pin(db: Session, pin: str | None) -> None:
    """Refuse an administrator action without the right PIN (when one is set)."""
    row = db.get(Setting, ADMIN_PIN_KEY)
    if row is None:
        return
    salt, digest = json.loads(row.value)
    import hmac
    if not pin or not hmac.compare_digest(_pin_hash(pin, salt), digest):
        raise ServiceError("admin_pin_required", "administrator PIN required", 403)


def set_admin_pin(db: Session, day: Day, pin: str | None) -> None:
    import secrets
    row = db.get(Setting, ADMIN_PIN_KEY)
    if not pin:
        if row is not None:
            db.delete(row)
            _audit(db, day, "admin", "admin_pin", None, set=False)
    else:
        if not (pin.isdigit() and 4 <= len(pin) <= 8):
            raise ServiceError("bad_pin", "the PIN must be 4 to 8 digits", 400)
        salt = secrets.token_hex(8)
        value = json.dumps([salt, _pin_hash(pin, salt)])
        if row is None:
            db.add(Setting(key=ADMIN_PIN_KEY, value=value))
        else:
            row.value, row.updated_at = value, utcnow()
        _audit(db, day, "admin", "admin_pin", None, set=True)
    db.commit()


def get_settings(db: Session) -> dict:
    stored = {s.key: json.loads(s.value) for s in db.scalars(select(Setting).where(Setting.key.in_(list(SETTINGS))))}
    out = {k: stored.get(k, default) for k, default in SETTINGS.items()}
    out["admin_pin_set"] = pin_is_set(db)
    img = db.get(Setting, BACKGROUND_KEY)
    out["background_image"] = img.updated_at.strftime("%Y%m%d%H%M%S%f") if img is not None else None   # a version, for caching
    return out


def update_settings(db: Session, day: Day, changes: dict) -> dict:
    unknown = set(changes) - set(SETTINGS)
    if unknown:
        raise ServiceError("unknown_setting", f"unknown setting {sorted(unknown)[0]!r}", 400)
    before = get_settings(db)
    if changes.get("site_name") is not None:
        changes["site_name"] = " ".join(str(changes["site_name"]).split())[:60]
    if changes.get("count_tuning") is not None:
        from ..vision.exercises import check_tuning
        try:
            changes["count_tuning"] = check_tuning(changes["count_tuning"])
        except (ValueError, TypeError) as e:
            raise ServiceError("bad_tuning", str(e), 400) from None
    changed = {k: v for k, v in changes.items() if v is not None and v != before[k]}
    for k, v in changed.items():
        row = db.get(Setting, k)
        if row is None:
            db.add(Setting(key=k, value=json.dumps(v)))
        else:
            row.value, row.updated_at = json.dumps(v), utcnow()
    if changed:
        _audit(db, day, "admin", "settings", None, **{k: {"from": before[k], "to": v} for k, v in changed.items()})
    db.commit()
    return get_settings(db)


def next_free_bib(db: Session, day: Day) -> str:
    """The next bib number for today: one above the highest numeric bib on today's list, and
    not below the ``bib_start`` setting. Called inside the request that saves the person, and
    writes are serialised, so two desks registering at the same moment never get the same number."""
    used = db.scalars(select(Candidate.bib).join(Entry).where(Entry.day_id == day.id))
    start = int(get_settings(db)["bib_start"])
    return str(max(max((int(b) for b in used if b and b.isdigit()), default=0) + 1, start))


def add_candidate(db: Session, day: Day, *, external_id: str, name: str, name_ar: str | None = None,
                  gender: str | None = None, dob: str | None = None, bib: str | None = None,
                  unit: str | None = None, auto_bib: bool = False, commit: bool = True, result: dict | None = None) -> Entry:
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
    unit = norm_unit(unit)
    name_ar = " ".join((name_ar or "").split()) or None
    cand = db.scalar(select(Candidate).where(func.lower(Candidate.external_id) == external_id.lower()))
    if cand is not None and cand.name.casefold() != name.casefold():
        raise ServiceError("external_id_in_use",
                           f"ID {cand.external_id} already belongs to {cand.name}", 409)
    effective_bib = bib or (cand.bib if cand else None)
    clash = None
    if effective_bib:
        clash = db.scalar(select(Entry).join(Candidate).where(
            Entry.day_id == day.id, Candidate.bib == effective_bib,
            *( [Candidate.id != cand.id] if cand is not None else [] )))
    if auto_bib and (not effective_bib or clash):
        # No bib, or theirs is taken today: the system gives the next free one. Nobody at the
        # desk should have to invent or remember a bib number.
        effective_bib = bib = next_free_bib(db, day)
        clash = None
    if clash:
        raise ServiceError("bib_in_use", f"bib {effective_bib}, {clash.candidate.name}")
    outcome = "unchanged"
    if cand is None:
        cand = Candidate(external_id=external_id, name=name, name_ar=name_ar, gender=gender, dob=dob, bib=bib,
                         unit=unit or get_settings(db)["default_unit"])
        db.add(cand)
        db.flush()
        outcome = "added"
    else:
        new = {"name_ar": name_ar or cand.name_ar, "gender": gender or cand.gender,
               "dob": dob or cand.dob, "bib": effective_bib, "unit": unit or cand.unit}
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


CSV_COLUMNS = ("external_id", "name", "name_ar", "gender", "dob", "bib", "unit")


def decode_csv(data: bytes) -> str:
    """UTF-8 (with or without BOM) first; Arabic Excel on Windows saves CP-1256."""
    for enc in ("utf-8-sig", "cp1256"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ServiceError("bad_csv", "the file is not UTF-8 or Arabic Windows text", 400)


def import_csv(db: Session, day: Day, text: str, to_day: bool = True) -> dict:
    """Header row required; columns external_id,name[,name_ar,gender,dob,bib,unit] in any order.
    ``to_day``: put everyone on today's list; otherwise save them in the register only."""
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
                (add_candidate if to_day else register_candidate)(
                    db, day, commit=False, result=result,
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


def regrade(rules: DayRules, entry: Entry) -> None:
    """Re-score saved results after the person's standard changed (age band or unit level)."""
    band = rules.find_band(entry.band)
    if entry.run_time_s is not None:
        entry.run_grade = rules.grade("run", entry.run_time_s, band) if band else None
        if entry.status in (J.RUN_PASSED, J.RUN_FAILED, J.RUN_DONE):
            pass_s = band.run_s[GRADES.index(entry.level)] if band and entry.level in GRADES else None
            entry.status = J.after_finish(entry.run_time_s, pass_s, rules.run_cutoff_s)
    for a in entry.attempts:
        if a.status == "finished" and a.count is not None:
            a.grade = rules.grade(a.event, a.count, band) if band else None
            a.passed = a.grade >= entry.level if band and entry.level else None


EDITABLE = ("external_id", "name", "name_ar", "gender", "dob", "unit", "bib")


def edit_candidate(db: Session, day: Day, rules: DayRules, entry: Entry | None, changes: dict,
                   candidate: Candidate | None = None) -> Entry | None:
    """Correct a candidate's details. Only the fields given change; each change is audited.
    When the person is already weighed in, the standard (age band, level) is worked out again
    and every saved result is re-scored, so a wrong date of birth or unit never stands.
    ``entry`` is today's entry; from the register a person not on today's list comes as
    ``candidate`` alone."""
    unknown = set(changes) - set(EDITABLE)
    if unknown:
        raise ServiceError("bad_input", f"cannot edit {sorted(unknown)[0]!r}", 400)
    c, new = (entry.candidate if entry is not None else candidate), {}
    if "external_id" in changes:
        v = (changes["external_id"] or "").strip()
        if not v:
            raise ServiceError("missing_fields", "the ID cannot be empty", 400)
        other = db.scalar(select(Candidate).where(func.lower(Candidate.external_id) == v.lower(), Candidate.id != c.id))
        if other is not None:
            raise ServiceError("external_id_in_use", f"ID {other.external_id} already belongs to {other.name}", 409)
        new["external_id"] = v
    if "name" in changes:
        v = " ".join((changes["name"] or "").split())
        if not v:
            raise ServiceError("missing_fields", "the name cannot be empty", 400)
        new["name"] = v
    if "name_ar" in changes:
        new["name_ar"] = " ".join((changes["name_ar"] or "").split()) or None
    weighed = entry is not None and entry.age is not None
    if "gender" in changes:
        new["gender"] = norm_gender(changes["gender"])
        if new["gender"] is None and weighed:
            raise ServiceError("gender_required", f"{c.name} is weighed in: gender is needed", 409)
    if "dob" in changes:
        new["dob"] = norm_dob(changes["dob"])
        if new["dob"] is None and weighed:
            raise ServiceError("dob_required", f"{c.name} is weighed in: date of birth is needed", 409)
    if "unit" in changes:
        new["unit"] = norm_unit(changes["unit"])
        if new["unit"] is None:
            raise ServiceError("bad_unit", "choose a unit type", 400)
    if "bib" in changes:
        v = norm_bib(changes["bib"])
        if v != c.bib:
            if entry is not None and entry.status in (J.LINED_UP, J.RUNNING):
                raise ServiceError("bib_locked", f"{c.name} is on the run: the bib cannot change now", 409)
            clash = db.scalar(select(Entry).join(Candidate).where(
                Entry.day_id == day.id, Candidate.bib == v, Candidate.id != c.id)) if v else None
            if clash is not None:
                raise ServiceError("bib_in_use", f"bib {v}, {clash.candidate.name}")
        new["bib"] = v
    changed = {k: {"from": getattr(c, k), "to": v} for k, v in new.items() if getattr(c, k) != v}
    if not changed:
        return entry
    if entry is None:
        for k, v in changed.items():
            setattr(c, k, v["to"])
        _audit(db, day, "admin", "edit_candidate", None, external_id=c.external_id, changed=changed)
        db.commit()
        return None
    before = {"standard": [entry.age, entry.band, entry.level], "verdict": verdict(entry)}
    for k, v in changed.items():
        setattr(c, k, v["to"])
    try:
        if weighed and {"gender", "dob", "unit"} & set(changed):
            std = standard_for(db, day, rules, entry)
            entry.age, entry.band, entry.level = std["age"], std["band"], std["level"]
            regrade(rules, entry)
    except ServiceError:
        db.rollback()
        raise
    after = {"standard": [entry.age, entry.band, entry.level], "verdict": verdict(entry)}
    _audit(db, day, "admin", "edit_candidate", entry, changed=changed,
           **({"before": before, "after": after} if before != after else {}))
    db.commit()
    return entry


def verdict(entry: Entry) -> str | None:
    """The day's result: pass (every test at the required level), fail, pending (tests still to
    do), no_standard (measured, but no pass table applies), or None before weigh-in."""
    if entry.status == J.EXPECTED:
        return None
    if entry.status in (J.REJECTED, J.ABSENT, J.DNF):
        return "fail"
    finished = {a.event: a for a in entry.attempts if a.status == "finished"}
    if entry.band is None:
        return "no_standard" if entry.run_time_s is not None or finished else "pending"
    grades = [entry.run_grade] + [finished[e].grade if e in finished else None for e in EVENTS]
    if any(g is not None and g < (entry.level or 0) for g in grades):
        return "fail"
    return "pass" if all(g is not None for g in grades) else "pending"


def done_events(entry: Entry) -> set[str]:
    return {a.event for a in entry.attempts if a.status == "finished"}


def gate_for(entry: Entry, day: Day, station: str) -> J.Gate:
    return J.check_gate(entry.status, station, done_events(entry))


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


def enroll_face(db: Session, day: Day, entry: Entry | None, data: bytes, engine,
                candidate: Candidate | None = None) -> int:
    """Store the face in ``data`` for this candidate (today's ``entry``, or a ``candidate``
    from the register); keeps the newest MAX_TEMPLATES."""
    cid = entry.candidate_id if entry is not None else candidate.id
    face = _face(engine, data, require_single=True)
    # Guard against a mixed-up photo: refuse a face that already belongs to someone else.
    others: dict[int, list[np.ndarray]] = {}
    for other, emb in db.execute(select(FaceTemplate.candidate_id, FaceTemplate.embedding)
                                 .where(FaceTemplate.candidate_id != cid)).all():
        others.setdefault(other, []).append(np.frombuffer(emb, np.float32))
    clash = face_match(face.embedding, others, margin=0.0)
    if clash.candidate_id is not None:
        owner = db.get(Candidate, clash.candidate_id)
        raise ServiceError("face_belongs_to_other",
                           f"{owner.name}, bib {owner.bib}, match {clash.score}", 409,
                           owner={"id": owner.id, "name": owner.name, "name_ar": owner.name_ar, "external_id": owner.external_id})
    db.add(FaceTemplate(candidate_id=cid,
                        embedding=np.asarray(face.embedding, np.float32).tobytes(),
                        photo_jpeg=face.crop_jpeg, yaw=getattr(face, "yaw", None)))
    db.flush()
    # Keep the newest MAX_TEMPLATES, but never drop the photo that faces the camera best:
    # it is the one shown on the card and at the finish line.
    kept = list(db.scalars(select(FaceTemplate).where(FaceTemplate.candidate_id == cid)
                           .order_by(FaceTemplate.created_at.desc(), FaceTemplate.id.desc())))
    front = _front(kept)
    keep = kept[:MAX_TEMPLATES]
    if front is not None and front not in keep:
        keep = kept[:MAX_TEMPLATES - 1] + [front]
    for t in kept:
        if t not in keep:
            db.delete(t)
    _audit(db, day, "admin", "enroll_face", entry, detection_score=face.score,
           **({} if entry is not None else {"external_id": candidate.external_id}))
    db.commit()
    return face_count(db, cid)


def remove_faces(db: Session, day: Day, entry: Entry | None, candidate: Candidate | None = None) -> None:
    cid = entry.candidate_id if entry is not None else candidate.id
    for t in db.scalars(select(FaceTemplate).where(FaceTemplate.candidate_id == cid)):
        db.delete(t)
    _audit(db, day, "admin", "remove_faces", entry, **({} if entry is not None else {"external_id": candidate.external_id}))
    db.commit()


def _front(templates: list[FaceTemplate]) -> FaceTemplate | None:
    """The photo facing the camera best: smallest head turn. Photos saved before the turn was
    recorded count as the oldest first (the guided capture starts looking straight)."""
    if not templates:
        return None
    return min(templates, key=lambda t: (t.yaw is None, abs(t.yaw) if t.yaw is not None else 0.0, t.created_at, t.id))


def face_photo(db: Session, entry: Entry | None, candidate_id: int | None = None) -> bytes | None:
    cid = entry.candidate_id if entry is not None else candidate_id
    t = _front(list(db.scalars(select(FaceTemplate).where(FaceTemplate.candidate_id == cid))))
    return t.photo_jpeg if t else None


def identify(db: Session, day: Day, data: bytes, engine, station: str | None = None) -> dict:
    """Who is in this frame, among today's candidates with an enrolled face?"""
    face = _face(engine, data, require_single=False)
    rows = db.execute(select(Entry.id, FaceTemplate.embedding).join(
        FaceTemplate, FaceTemplate.candidate_id == Entry.candidate_id).where(Entry.day_id == day.id)).all()
    gallery: dict[int, list[np.ndarray]] = {}
    for entry_id, emb in rows:
        gallery.setdefault(entry_id, []).append(np.frombuffer(emb, np.float32))
    threshold, margin = FACE_MATCH.get(get_settings(db)["face_match"], FACE_MATCH["normal"])
    m = face_match(face.embedding, gallery, threshold=threshold, margin=margin)
    out = {"box": list(face.box), "score": m.score, "runner_up": m.runner_up,
           "confident": m.confident, "enrolled_today": len(gallery), "entry": None,
           "allowed": None, "reason": None}
    if m.candidate_id is not None:
        entry = db.get(Entry, m.candidate_id)
        out["entry"] = entry
        if station:
            g = gate_for(entry, day, station)
            out["allowed"], out["reason"] = g.ok, (g.reason or None)
    else:
        out["reason"] = "nobody_enrolled" if not gallery else "face_not_recognised"
        # Not on today's list: a candidate from another day with this face is a returning
        # candidate, to be added to today's list, not registered a second time.
        others: dict[int, list[np.ndarray]] = {}
        for cid, emb in db.execute(select(FaceTemplate.candidate_id, FaceTemplate.embedding).where(
                FaceTemplate.candidate_id.not_in(select(Entry.candidate_id).where(Entry.day_id == day.id)))).all():
            others.setdefault(cid, []).append(np.frombuffer(emb, np.float32))
        known = face_match(face.embedding, others, threshold=threshold, margin=margin)
        if known.candidate_id is not None:
            c = db.get(Candidate, known.candidate_id)
            out["reason"] = "not_on_todays_list"
            out["known"] = {"external_id": c.external_id, "name": c.name, "name_ar": c.name_ar,
                            "gender": c.gender, "dob": c.dob, "bib": c.bib, "score": known.score}
    _audit(db, day, station or "checkin", "identify", out["entry"], score=m.score,
           runner_up=m.runner_up, confident=m.confident)
    db.commit()
    return out


# ----------------------------------------------------------------------- weigh-in

def standard_for(db: Session, day: Day, rules: DayRules, entry: Entry, *, gender: str | None = None,
                 dob: str | None = None, unit: str | None = None) -> dict:
    """Who this person is judged as today: age on the test day, age band and required level.
    Missing gender / date of birth may be given here (they fill the candidate's record)."""
    c = entry.candidate
    gender, dob, unit = norm_gender(gender), norm_dob(dob), norm_unit(unit)
    gender, dob = c.gender or gender, c.dob or dob
    if not gender:
        raise ServiceError("gender_required", f"{c.name}: gender is needed to pick the pass table", 409)
    if not dob:
        raise ServiceError("dob_required", f"{c.name}: date of birth is needed to pick the age band", 409)
    age = age_on(dob, day.date)
    unit = unit or c.unit or get_settings(db)["default_unit"]
    band = rules.band(gender, age)
    if band is None and rules.has_table(gender):
        ages = [b for b in (rules.men if gender == "M" else rules.women)]
        raise ServiceError("no_standard_for_age",
                           f"{c.name} is {age}; the pass table covers {ages[0].age_min} to {ages[-1].age_max}", 409)
    return {"gender": gender, "dob": dob, "age": age, "unit": unit,
            "band": band.label if band else None, "level": rules.level(unit)}


def weigh_in(db: Session, day: Day, rules: DayRules, entry: Entry, *, weight_kg: float,
             height_cm: float, accept: bool, override_reason: str | None = None,
             gender: str | None = None, dob: str | None = None, unit: str | None = None) -> Entry:
    require_gate(entry, day, "weigh")
    std = standard_for(db, day, rules, entry, gender=gender, dob=dob, unit=unit)
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
    c = entry.candidate
    filled = {k: std[k] for k in ("gender", "dob", "unit") if getattr(c, k) != std[k]}
    for k, v in filled.items():
        setattr(c, k, v)
    entry.weight_kg, entry.height_cm, entry.bmi = weight_kg, height_cm, value
    entry.age, entry.band, entry.level = std["age"], std["band"], std["level"]
    entry.weighed_at = utcnow()
    entry.status = J.after_weigh_in(accept)
    _audit(db, day, "weigh", "weigh_in", entry, weight_kg=weight_kg, height_cm=height_cm,
           bmi=value, accepted=accept, override_reason=override_reason or None,
           age=std["age"], band=std["band"], level=std["level"], unit=std["unit"],
           candidate_changed=filled or None)
    db.commit()
    return entry


# ---------------------------------------------------------------------------- run
#
# Every runner starts on their own: recognised at the start line, 3-2-1-GO, and their clock
# runs from their own GO (``Entry.run_started_at``) until their finish crossing. Nobody waits
# for a group; whoever is still out after the cut-off becomes DNF by themselves.

RUN_GRACE_S = 120
"""After the run cut-off, a runner still out is given this long (a late finish may still come in)
before they become DNF by themselves."""


def running_entries(db: Session, day: Day, at: datetime | None = None) -> list[Entry]:
    """Out on the course: running and started (not still in their countdown) at ``at``."""
    at = at or utcnow()
    return [e for e in list_entries(db, day, J.RUNNING) if e.run_started_at is not None and e.run_started_at <= at]


def counting_down(db: Session, day: Day, now: datetime | None = None) -> Entry | None:
    """The runner whose 3-2-1 is on now (their GO still ahead), if any."""
    now = now or utcnow()
    return next((e for e in list_entries(db, day, J.RUNNING)
                 if e.run_started_at is not None and e.run_started_at > now), None)


def line_up(db: Session, day: Day, entry: Entry, by: str = "typed") -> Entry:
    """At the start line, checked: next to start."""
    require_gate(entry, day, "lineup")
    entry.status = J.LINED_UP
    entry.lined_up_at = utcnow()
    _audit(db, day, "start", "line_up", entry, by=by)
    db.commit()
    return entry


def clear_line(db: Session, day: Day) -> int:
    """Everyone at the line (not started) goes back to waiting."""
    n = 0
    for e in list_entries(db, day, J.LINED_UP):
        e.status, e.lined_up_at = J.ACCEPTED, None
        n += 1
    if n:
        _audit(db, day, "start", "clear_line", people=n)
    db.commit()
    return n


def undo_line_up(db: Session, day: Day, entry: Entry) -> Entry:
    """Marked at the line by mistake: back to accepted (only before they start)."""
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
    old = {"weight_kg": entry.weight_kg, "height_cm": entry.height_cm, "bmi": entry.bmi, "status": entry.status}
    entry.status = J.EXPECTED
    entry.weight_kg = entry.height_cm = entry.bmi = None
    entry.age = entry.band = entry.level = None
    entry.weighed_at = None
    _audit(db, day, "admin", "undo_weigh_in", entry, reason=reason or None, **old)
    db.commit()
    return entry


def remove_entry(db: Session, day: Day, entry: Entry, delete_results: bool = False, forget: bool = False,
                 commit: bool = True) -> None:
    """Take someone off today's list. The person stays in the register (with their face) and
    can be added again or recognised as a returning candidate.

    Without ``delete_results`` only before anything happened to them today (added by
    mistake). With it, also when they have results: their weigh-in, run crossing and
    exercise attempts (with the reps) are deleted, and the audit log keeps a copy of what
    was there. ``forget``: a registration made by mistake (e.g. the face turned out to be
    someone else's): when the person is on no other day's list either, the person and their
    face photos go too."""
    has_results = entry.status != J.EXPECTED or bool(entry.attempts)
    if has_results and not delete_results:
        raise ServiceError("cannot_remove", f"{entry.candidate.name} is already {entry.status}; it cannot be removed")
    cand = entry.candidate
    results = None
    if has_results:
        results = {"status": entry.status, "weight_kg": entry.weight_kg, "height_cm": entry.height_cm,
                   "bmi": entry.bmi, "run_time_s": entry.run_time_s,
                   "attempts": [{"event": a.event, "status": a.status, "count": a.count, "rejected": a.rejected,
                                 "auto_count": a.auto_count, "reps": len(a.reps)} for a in entry.attempts]}
        for a in list(entry.attempts):
            for r in list(a.reps):
                db.delete(r)
            db.delete(a)
        for c in db.scalars(select(Crossing).where(Crossing.entry_id == entry.id)):
            db.delete(c)       # their finish crossing: the finishers list no longer shows them
        db.flush()
    for a in db.scalars(select(AuditLog).where(AuditLog.entry_id == entry.id)):
        a.entry_id = None   # keep the history, drop the link to the deleted row
    other_days = db.scalar(select(func.count()).select_from(Entry).where(
        Entry.candidate_id == cand.id, Entry.id != entry.id)) or 0
    drop = forget and not other_days
    faces = 0
    if drop:
        for t in db.scalars(select(FaceTemplate).where(FaceTemplate.candidate_id == cand.id)):
            db.delete(t)
            faces += 1
    _audit(db, day, "admin", "remove_from_day", None, external_id=cand.external_id, name=cand.name, bib=cand.bib,
           person_removed=drop, faces_removed=faces, deleted_results=results)
    db.delete(entry)
    if drop:
        db.flush()
        db.delete(cand)
    if commit:
        db.commit()


START_COUNTDOWN_MAX_S = 30


def start_runner(db: Session, day: Day, rules: DayRules, entry: Entry, countdown_s: int = 0,
                 now: datetime | None = None, by: str = "examiner") -> Entry:
    """Start one runner: their GO is ``countdown_s`` from now (every screen shows the same 3-2-1)
    and their clock runs from it. One runner at a time: a start waits for the countdown before it."""
    if not 0 <= countdown_s <= START_COUNTDOWN_MAX_S:
        raise ServiceError("bad_countdown", f"the countdown is 0 to {START_COUNTDOWN_MAX_S} s", 400)
    now = now or utcnow()
    if entry.status != J.LINED_UP:
        require_gate(entry, day, "lineup")
    busy = counting_down(db, day, now)
    if busy is not None and busy.id != entry.id:
        raise ServiceError("start_in_progress", f"{busy.candidate.name} is starting now; wait for their GO")
    go = now + timedelta(seconds=countdown_s)
    entry.status = J.RUNNING
    entry.lined_up_at = entry.lined_up_at or now
    entry.run_started_at = go
    entry.run_cutoff_at = go + timedelta(seconds=rules.run_cutoff_s + RUN_GRACE_S)
    _audit(db, day, "start", "runner_start", entry, at=go, countdown_s=countdown_s, by=by)
    db.commit()
    return entry


def cancel_start(db: Session, day: Day, entry: Entry, now: datetime | None = None, late: bool = False) -> Entry:
    """A start by mistake (wrong person, false start): back to the line, clock cleared.

    During the countdown and the first minutes anyone at the start line can; later only with
    ``late`` (an administrator, PIN-checked by the API), and never once they crossed the finish."""
    now = now or utcnow()
    if entry.status != J.RUNNING or entry.run_started_at is None:
        raise ServiceError("not_running", f"{entry.candidate.name} has not started")
    window = int(get_settings(db)["cancel_window_s"])   # 0: no limit (until they finish)
    if window and now - entry.run_started_at > timedelta(seconds=window) and not late:
        raise ServiceError("too_late_to_cancel", f"after {window // 60} min a start can only be cancelled by an administrator")
    _audit(db, day, "start", "cancel_start", entry, started_at=entry.run_started_at, late=late)
    entry.status = J.LINED_UP
    entry.run_started_at = entry.run_cutoff_at = None
    db.commit()
    return entry


def _elapsed(entry: Entry, at: datetime) -> float:
    return round((at - entry.run_started_at).total_seconds(), 1)


def record_finish(db: Session, day: Day, rules: DayRules, bib: str | None,
                  now: datetime | None = None, source: str = "manual",
                  confidence: float | None = None, clip_path: str | None = None) -> Crossing:
    """A finish crossing. A runner known by bib gets their time from their own GO; anything else
    goes to the review queue (the time is set when it is assigned to a runner)."""
    at = now or utcnow()
    bib = norm_bib(bib)
    entry = db.scalar(select(Entry).join(Candidate).where(Entry.day_id == day.id, Candidate.bib == bib)) if bib else None
    if entry is not None and entry.status == J.RUNNING and entry.run_started_at is not None and at < entry.run_started_at:
        raise ServiceError("before_start", f"the crossing is before {entry.candidate.name}'s GO")
    if not db.scalar(select(func.count()).select_from(Entry).where(
            Entry.day_id == day.id, Entry.run_started_at.is_not(None), Entry.run_started_at <= at)):
        raise ServiceError("run_not_started", "nobody has started the run yet")
    out = running_entries(db, day, at)
    crossing = Crossing(day_id=day.id, at=at, time_s=0.0, source=source, bib_read=bib,
                        confidence=confidence, clip_path=clip_path)
    if entry is not None and entry in out:
        _assign(day, rules, crossing, entry)
    else:
        crossing.needs_review = True
        if crossing.bib_read is None:
            crossing.note = "no bib read"
            crossing.note_code = "no_bib_read"
        elif entry is None:
            crossing.note = f"bib {crossing.bib_read} is not on today's list"
            crossing.note_code, crossing.note_bib = "bib_not_on_list", crossing.bib_read
        else:
            crossing.note = f"bib {crossing.bib_read} is {entry.status}, not running"
            crossing.note_code, crossing.note_bib, crossing.note_status = "bib_not_running", crossing.bib_read, entry.status
    db.add(crossing)
    db.flush()
    _audit(db, day, "finish", "crossing", entry if not crossing.needs_review else None, bib=crossing.bib_read,
           time_s=crossing.time_s if not crossing.needs_review else None,
           needs_review=crossing.needs_review, source=source, confidence=confidence)
    db.commit()
    return crossing


def _assign(day: Day, rules: DayRules, crossing: Crossing, entry: Entry) -> None:
    # the time counts from the runner's own GO
    crossing.time_s = _elapsed(entry, crossing.at)
    crossing.entry_id = entry.id
    crossing.needs_review = False
    entry.run_time_s = crossing.time_s
    entry.finished_at = crossing.at
    band = rules.find_band(entry.band)
    entry.run_grade = rules.grade("run", crossing.time_s, band) if band else None
    pass_s = band.run_s[GRADES.index(entry.level)] if band and entry.level in GRADES else None
    entry.status = J.after_finish(crossing.time_s, pass_s, rules.run_cutoff_s)


def resolve_crossing(db: Session, day: Day, rules: DayRules, crossing_id: int,
                     entry: Entry) -> Crossing:
    crossing = db.get(Crossing, crossing_id)
    if crossing is None or crossing.day_id != day.id:
        raise ServiceError("not_found", f"crossing {crossing_id} not found", 404)
    if not crossing.needs_review:
        raise ServiceError("already_resolved", "this crossing is already assigned")
    if entry.status != J.RUNNING:
        raise ServiceError("not_running", f"{entry.candidate.name} is {entry.status}, not running")
    if entry.run_started_at is None or crossing.at < entry.run_started_at:
        raise ServiceError("before_start", f"this crossing is before {entry.candidate.name} started")
    _assign(day, rules, crossing, entry)
    crossing.note = (crossing.note or "") + " | assigned by examiner"
    crossing.note_code, crossing.note_bib, crossing.note_status = "assigned_by_examiner", entry.candidate.bib, None
    _audit(db, day, "finish", "resolve_crossing", entry, crossing_id=crossing_id)
    db.commit()
    return crossing


def face_for_crossing(db: Session, day: Day, rules: DayRules, crossing: Crossing, data: bytes | list[bytes],
                      engine) -> dict | None:
    """A finish crossing the camera could not tie to a runner by bib: compare the runner's face
    with the runners still out on the course (nobody else can be crossing). The camera sends a
    few snapshots from the last moments before the line; the best one counts. A confident match
    (the strict setting, whatever the desk uses) is recorded; a likely one is only suggested.
    Every outcome is in the audit log (to see why a face was not used). The pictures are not kept."""
    if not crossing.needs_review:
        return None
    shots = [data] if isinstance(data, (bytes, bytearray)) else list(data)
    live = [e.id for e in running_entries(db, day, crossing.at)]
    rows = db.execute(select(Entry.id, FaceTemplate.embedding).join(
        FaceTemplate, FaceTemplate.candidate_id == Entry.candidate_id).where(
        Entry.day_id == day.id, Entry.id.in_(live))).all()
    gallery: dict[int, list[np.ndarray]] = {}
    for entry_id, emb in rows:
        gallery.setdefault(entry_id, []).append(np.frombuffer(emb, np.float32))
    if not gallery:
        _audit(db, day, "finish", "face_at_finish", None, crossing_id=crossing.id, outcome="nobody_out_with_a_face")
        db.commit()
        return None
    threshold, margin = FACE_MATCH["strict"]
    m = best = None
    found = 0
    for shot in shots:
        try:
            face = engine.face_from_bytes(shot, require_single=False)
        except Exception:        # no face in this snapshot (looking down, an arm across it)
            continue
        found += 1
        sm = face_match(face.embedding, gallery, threshold=threshold, margin=margin)
        sb = face_match(face.embedding, gallery, threshold=0.0, margin=0.0)
        if sm.candidate_id is not None and (m is None or sm.score > m.score):
            m = sm
        if best is None or sb.score > best.score:
            best = sb
    if m is None and best is not None and best.score < FACE_MATCH["normal"][0]:
        _audit(db, day, "finish", "face_at_finish", None, crossing_id=crossing.id, outcome="no_match",
               score=round(float(best.score), 3), snapshots=len(shots), faces_found=found)
        db.commit()
        return None
    if best is None:
        _audit(db, day, "finish", "face_at_finish", None, crossing_id=crossing.id, outcome="no_face_found",
               snapshots=len(shots))
        db.commit()
        return None
    if m is not None:
        entry = db.get(Entry, m.candidate_id)
        resolve_crossing(db, day, rules, crossing.id, entry)
        crossing.note = f"identified by face ({m.score:.2f})"
        crossing.note_code, crossing.note_bib, crossing.note_status = "identified_by_face", entry.candidate.bib, None
        crossing.confidence = round(float(m.score), 3)
        _audit(db, day, "finish", "face_at_finish", entry, crossing_id=crossing.id, outcome="identified",
               score=m.score, runner_up=m.runner_up, snapshots=len(shots), faces_found=found)
        db.commit()
        return {"entry_id": entry.id, "score": m.score, "assigned": True}
    if best.candidate_id is not None:
        entry = db.get(Entry, best.candidate_id)
        # second-best for the record: a strict match also needs a lead over the next runner
        crossing.note = f"{crossing.note or ''} | face looks like bib {entry.candidate.bib} ({best.score:.2f})".strip(" |")
        crossing.note_code, crossing.note_bib, crossing.note_status = "face_suggests", entry.candidate.bib, None
        _audit(db, day, "finish", "face_at_finish", entry, crossing_id=crossing.id, outcome="suggested",
               score=best.score, snapshots=len(shots), faces_found=found)
        db.commit()
        return {"entry_id": entry.id, "score": best.score, "assigned": False}
    return None


def tag_crossing(db: Session, day: Day, rules: DayRules, crossing_id: int, bib: str | None) -> Crossing:
    """Tap first, identify after: attach a bib to a time recorded at the instant of crossing."""
    bib = norm_bib(bib)
    if not bib:
        raise ServiceError("empty_query", "type the bib", 400)
    entry = db.scalar(select(Entry).join(Candidate).where(Entry.day_id == day.id, Candidate.bib == bib))
    if entry is None:
        raise ServiceError("bib_not_on_list", f"bib {bib} is not on today's list", 404)
    return resolve_crossing(db, day, rules, crossing_id, entry)


def mark_dnf(db: Session, day: Day, entry: Entry, reason: str | None, by: str = "examiner") -> Entry:
    """One runner stops (injured, gave up, left the course): DNF now, with the reason."""
    reason = " ".join((reason or "").split())
    if entry.status != J.RUNNING:
        raise ServiceError("not_running", f"{entry.candidate.name} is {entry.status}, not running")
    if not reason:
        raise ServiceError("reason_required", "say why the runner did not finish", 400)
    entry.status = J.DNF
    _audit(db, day, "finish", "mark_dnf", entry, reason=reason, by=by)
    db.commit()
    return entry


def undo_dnf(db: Session, day: Day, entry: Entry, now: datetime | None = None) -> Entry:
    """DNF marked by mistake while they are still out: running again (until their cut-off)."""
    if entry.status != J.DNF or entry.run_time_s is not None:
        raise ServiceError("not_dnf", f"{entry.candidate.name} is not marked DNF", 409)
    if entry.run_cutoff_at is None or (now or utcnow()) > entry.run_cutoff_at:
        raise ServiceError("run_already_closed", f"{entry.candidate.name}'s run time is over")
    entry.status = J.RUNNING
    _audit(db, day, "finish", "undo_dnf", entry)
    db.commit()
    return entry


def unassign_crossing(db: Session, day: Day, crossing_id: int) -> Crossing:
    """Wrong runner given a finish: the crossing goes back to the review queue and the runner
    back to running (their real crossing can then be assigned)."""
    crossing = db.get(Crossing, crossing_id)
    if crossing is None or crossing.day_id != day.id:
        raise ServiceError("not_found", f"crossing {crossing_id} not found", 404)
    if crossing.needs_review or crossing.entry_id is None:
        raise ServiceError("not_assigned", "this crossing is not assigned to anyone")
    entry = db.get(Entry, crossing.entry_id)
    if any(a.status in ("running", "finished") for a in entry.attempts):
        raise ServiceError("results_exist", f"{entry.candidate.name} already has exercise results")
    entry.status = J.RUNNING
    entry.run_time_s = entry.run_grade = None
    entry.finished_at = None
    crossing.entry_id = None
    crossing.needs_review = True
    crossing.note = f"unassigned from bib {entry.candidate.bib} by examiner"
    crossing.note_code, crossing.note_bib, crossing.note_status = "unassigned_by_examiner", entry.candidate.bib, None
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


def close_run(db: Session, day: Day, now: datetime | None = None, by: str = "examiner") -> int:
    """End of the run for today: everyone still out becomes DNF. Refused while a crossing waits
    for review (it may be one of them)."""
    now = now or utcnow()
    out = running_entries(db, day, now)
    pending = db.scalar(select(func.count()).select_from(Crossing).where(
        Crossing.day_id == day.id, Crossing.needs_review.is_(True)))
    if pending:
        raise ServiceError("review_pending", f"{pending} crossing(s) still wait for review; assign or discard them first")
    if not out:
        return 0
    for e in out:
        e.status = J.after_close(e.status)
    _audit(db, day, "finish", "close_run", dnf=len(out), by=by)
    db.commit()
    return len(out)


def auto_close(db: Session, day: Day, rules: DayRules, now: datetime | None = None) -> list[int]:
    """Runners still out after their cut-off (plus the grace for late finishers) become DNF by
    themselves; nobody has to close the run. Not while a crossing waits for review: it may be
    theirs (the screens show the queue)."""
    now = now or utcnow()
    late = [e for e in list_entries(db, day, J.RUNNING) if e.run_cutoff_at is not None and now > e.run_cutoff_at]
    if not late:
        return []
    if db.scalar(select(func.count()).select_from(Crossing).where(
            Crossing.day_id == day.id, Crossing.needs_review.is_(True))):
        return []
    for e in late:
        e.status = J.after_close(e.status)
        _audit(db, day, "finish", "dnf_cutoff", e, started_at=e.run_started_at, by="auto")
    db.commit()
    return [e.id for e in late]


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
                   rejected: int = 0, override_reason: str | None = None) -> Attempt:
    attempt = db.get(Attempt, attempt_id)
    if attempt is None or attempt.entry.day_id != day.id:
        raise ServiceError("not_found", f"attempt {attempt_id} not found", 404)
    if attempt.status != "running":
        raise ServiceError("attempt_not_running", "this attempt is already finished")
    if count < 0 or rejected < 0:
        raise ServiceError("bad_count", "counts cannot be negative", 400)
    override_reason = (override_reason or "").strip() or None
    # The engine counted and the examiner saves a different number: that is an
    # override and needs a reason. Without a camera count any number is simply the count.
    overridden = attempt.auto_count is not None and count != attempt.auto_count
    if overridden and override_reason is None:
        raise ServiceError("override_reason_required",
                           f"the camera counted {attempt.auto_count}; saving {count} needs a reason")
    attempt.count, attempt.rejected = count, rejected
    attempt.override_reason = override_reason
    entry = attempt.entry
    band = rules.find_band(entry.band)
    attempt.grade = rules.grade(attempt.event, count, band) if band else None
    attempt.passed = attempt.grade >= entry.level if band and entry.level else None
    attempt.ended_at = utcnow()
    attempt.status = "finished"
    entry.status = J.after_event(done_events(entry), entry.status)
    # The engine's numbers and the override reason go into the audit row next to
    # the saved ones, so the review screen and the audit trail agree.
    _audit(db, day, attempt.event, "finish_attempt", entry, attempt_id=attempt.id,
           count=count, rejected=rejected, passed=attempt.passed, grade=attempt.grade,
           auto_count=attempt.auto_count, auto_rejected=attempt.auto_rejected,
           override_reason=override_reason)
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


# ------------------------------------------------------------------ engine link

def running_attempts(db: Session, day: Day, station: str) -> dict[str, Attempt]:
    """The running attempt per lane name at this station (see docs/ENGINE-LINK.md)."""
    rows = db.scalars(select(Attempt).join(Entry).where(
        Entry.day_id == day.id, Attempt.event == station, Attempt.status == "running",
        Attempt.lane.is_not(None)))
    return {a.lane: a for a in rows}


def engine_report(db: Session, day: Day, station: str, lanes: dict[str, dict]) -> dict:
    """Take one engine report and answer with what the engine should count.

    ``lanes`` maps lane name -> {attempt_id, count, rejected, events: [...]}.
    Events are attached to the lane's running attempt only when the engine
    counted them for that attempt (same ``attempt_id``); events for a cancelled
    or finished attempt, or sent before the engine learnt about the new one,
    are dropped. Every received event is acknowledged either way so the engine
    stops resending it. A rep index already stored is left as it is.
    """
    if station not in EVENTS:
        raise ServiceError("unknown_station", f"unknown station {station!r}", 400)
    running = running_attempts(db, day, station)
    acked: dict[str, int] = {}
    for name, lane in lanes.items():
        events = lane.get("events") or []
        if events:
            acked[name] = max(int(e["index"]) for e in events)
        attempt = running.get(name)
        if attempt is None or lane.get("attempt_id") != attempt.id:
            continue
        attempt.auto_count = int(lane.get("count") or 0)
        attempt.auto_rejected = int(lane.get("rejected") or 0)
        if events:
            have = set(db.scalars(select(Rep.index).where(Rep.attempt_id == attempt.id)))
            for e in events:
                if int(e["index"]) in have:
                    continue
                db.add(Rep(attempt_id=attempt.id, index=int(e["index"]), valid=bool(e["valid"]),
                           faults=json.dumps(list(e.get("faults") or [])),
                           t_start=float(e["t_start"]), t_end=float(e["t_end"]),
                           clip_path=e.get("clip")))
    db.commit()
    return {
        "lanes": {name: {"attempt_id": running[name].id if name in running else None}
                  for name in lanes},
        "acked": acked,
    }


def list_reps(db: Session, day: Day, attempt_id: int) -> list[Rep]:
    attempt = db.get(Attempt, attempt_id)
    if attempt is None or attempt.entry.day_id != day.id:
        raise ServiceError("not_found", f"attempt {attempt_id} not found", 404)
    return list(attempt.reps)


def list_finished_attempts(db: Session, day: Day, event: str | None = None) -> list[Attempt]:
    """Today's finished attempts, newest first, for the review screen."""
    q = (select(Attempt).join(Entry, Entry.id == Attempt.entry_id)
         .where(Entry.day_id == day.id, Attempt.status == "finished"))
    if event is not None:
        q = q.where(Attempt.event == event)
    return list(db.scalars(q.order_by(Attempt.ended_at.desc(), Attempt.id.desc())))


# --------------------------------------------------------------------------- the register
# Everyone saved, whether or not they are on a test day: the people database. A test day's
# list (Entry rows) is who is tested that day; the register is who the station knows.

def register_candidate(db: Session, day: Day, *, external_id: str, name: str, name_ar: str | None = None,
                       gender: str | None = None, dob: str | None = None, bib: str | None = None,
                       unit: str | None = None, commit: bool = True, result: dict | None = None) -> Candidate:
    """Save a person in the register without putting them on a test day, or refresh the
    details of one already known. Same ID rules as ``add_candidate``: IDs compare without case
    and a known ID with a different name is refused. A bib here is the person's usual number;
    it is checked against a day's list when they are added to one."""
    external_id = (external_id or "").strip()
    name = " ".join((name or "").split())
    if not external_id or not name:
        raise ServiceError("missing_fields", "external_id and name are required", 400)
    fields = {"name_ar": " ".join((name_ar or "").split()) or None, "gender": norm_gender(gender),
              "dob": norm_dob(dob), "bib": norm_bib(bib), "unit": norm_unit(unit)}
    cand = db.scalar(select(Candidate).where(func.lower(Candidate.external_id) == external_id.lower()))
    if cand is not None and cand.name.casefold() != name.casefold():
        raise ServiceError("external_id_in_use", f"ID {cand.external_id} already belongs to {cand.name}", 409)
    if cand is None:
        cand = Candidate(external_id=external_id, name=name, **fields | {"unit": fields["unit"] or get_settings(db)["default_unit"]})
        db.add(cand)
        db.flush()
        outcome = "added"
        _audit(db, day, "admin", "register_candidate", None, external_id=external_id)
    else:
        new = {k: v for k, v in fields.items() if v is not None and getattr(cand, k) != v}
        for k, v in new.items():
            setattr(cand, k, v)
        outcome = "updated" if new else "unchanged"
    if result is not None:
        result["outcome"] = outcome
    if commit:
        db.commit()
    return cand


def get_candidate(db: Session, candidate_id: int) -> Candidate:
    c = db.get(Candidate, candidate_id)
    if c is None:
        raise ServiceError("not_registered", f"person {candidate_id} is not in the register", 404)
    return c


def lookup_candidate(db: Session, query: str) -> Candidate:
    """A person in the register by their ID (any case). Bibs are not used: they change from day to day."""
    q = (query or "").strip()
    c = db.scalar(select(Candidate).where(func.lower(Candidate.external_id) == q.lower())) if q else None
    if c is None:
        raise ServiceError("not_registered", f"{q} is not in the register", 404)
    return c


def register_rows(db: Session, day: Day, query: str = "", show: str = "all", offset: int = 0,
                  limit: int = 100) -> dict:
    """One page of the register, by name, with a search over names, ID and bib. ``show``:
    all, no_face, today (on today's list) or not_today. Each row says how many faces are on
    file, how many days they were tested and the last day's verdict."""
    faces = select(FaceTemplate.candidate_id, func.count().label("n")).group_by(FaceTemplate.candidate_id).subquery()
    today = select(Entry.candidate_id, Entry.id.label("entry_id")).where(Entry.day_id == day.id).subquery()
    base = (select(Candidate, func.coalesce(faces.c.n, 0), today.c.entry_id)
            .outerjoin(faces, faces.c.candidate_id == Candidate.id)
            .outerjoin(today, today.c.candidate_id == Candidate.id))
    for term in (query or "").split():
        like = f"%{term.lower()}%"
        base = base.where(func.lower(Candidate.name).like(like) | Candidate.name_ar.like(f"%{term}%")
                          | func.lower(Candidate.external_id).like(like) | (Candidate.bib == norm_bib(term)))
    if show == "no_face":
        base = base.where(faces.c.n.is_(None))
    elif show == "today":
        base = base.where(today.c.entry_id.is_not(None))
    elif show == "not_today":
        base = base.where(today.c.entry_id.is_(None))
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    page = db.execute(base.order_by(func.lower(Candidate.name), Candidate.id).offset(max(offset, 0))
                      .limit(max(1, min(limit, 500)))).all()
    hist: dict[int, list[tuple[Entry, str]]] = {}
    if page:
        for e, date in db.execute(select(Entry, Day.date).join(Day, Day.id == Entry.day_id)
                                  .where(Entry.candidate_id.in_([c.id for c, _, _ in page]))
                                  .order_by(Day.date.desc())).all():
            hist.setdefault(e.candidate_id, []).append((e, date))
    rows = []
    for c, n, entry_id in page:
        days = hist.get(c.id, [])
        tested = [(e, d) for e, d in days if e.status != J.EXPECTED]
        last = tested[0] if tested else None
        rows.append(candidate_dict(c) | {"faces": n, "today_entry": entry_id, "days_tested": len(tested),
                                         "last_date": last[1] if last else None,
                                         "last_verdict": verdict(last[0]) if last else None})
    return {"total": total, "rows": rows}


def candidate_dict(c: Candidate) -> dict:
    return {"id": c.id, "external_id": c.external_id, "name": c.name, "name_ar": c.name_ar, "gender": c.gender,
            "dob": c.dob, "bib": c.bib, "unit": c.unit, "registered_at": c.created_at.isoformat() if c.created_at else None}


def candidate_history(db: Session, day: Day, c: Candidate) -> dict:
    """The person with every test day they were on, newest first: measurements, standard,
    each test's result and grade, and the day's verdict."""
    days = []
    for e, date in db.execute(select(Entry, Day.date).join(Day, Day.id == Entry.day_id)
                              .where(Entry.candidate_id == c.id).order_by(Day.date.desc())).all():
        tests = {a.event: {"count": a.count, "grade": a.grade, "passed": a.passed}
                 for a in e.attempts if a.status == "finished"}
        days.append({"date": date, "today": e.day_id == day.id, "entry_id": e.id, "status": e.status,
                     "weight_kg": e.weight_kg, "height_cm": e.height_cm, "bmi": e.bmi,
                     "age": e.age, "level": e.level, "run_time_s": e.run_time_s, "run_grade": e.run_grade,
                     "tests": tests, "verdict": verdict(e)})
    return candidate_dict(c) | {"faces": face_count(db, c.id), "days": days}


def add_to_today(db: Session, day: Day, candidate_ids: list[int]) -> dict:
    """Put people from the register on today's list. Each keeps their usual bib when it is
    free today, else gets the next free one. Already on the list: left as they are."""
    counts, errors = {"added": 0, "already": 0}, []
    for cid in dict.fromkeys(candidate_ids):
        c = db.get(Candidate, cid)
        if c is None:
            errors.append({"id": cid, "code": "not_registered", "message": f"person {cid} is not in the register"})
            continue
        if db.scalar(select(Entry.id).where(Entry.day_id == day.id, Entry.candidate_id == cid)) is not None:
            counts["already"] += 1
            continue
        try:
            with db.begin_nested():
                add_candidate(db, day, external_id=c.external_id, name=c.name, auto_bib=True, commit=False)
            counts["added"] += 1
        except ServiceError as e:
            errors.append({"id": cid, "code": e.code, "message": e.message})
    db.commit()
    return counts | {"errors": errors}


def delete_candidate(db: Session, day: Day, c: Candidate, delete_results: bool = False) -> None:
    """Erase a person from the register: their faces, and every day's entry. Results on any
    day are deleted only with ``delete_results`` (the audit log keeps a copy). Refused while
    they are on the run or at the start line today."""
    entries = list(db.scalars(select(Entry).where(Entry.candidate_id == c.id)))
    for e in entries:
        if e.day_id == day.id and (e.status in (J.LINED_UP, J.RUNNING) or any(a.status == "running" for a in e.attempts)):
            raise ServiceError("in_progress", f"{c.name} is being tested right now", 409)
    tested = [e for e in entries if e.status != J.EXPECTED or e.attempts]
    if tested and not delete_results:
        raise ServiceError("has_results", f"{c.name} has results on {len(tested)} day(s)")
    info = {"external_id": c.external_id, "name": c.name, "days": len(entries), "days_with_results": len(tested)}
    for e in entries:
        remove_entry(db, db.get(Day, e.day_id), e, delete_results=True, commit=False)
    faces = 0
    for t in db.scalars(select(FaceTemplate).where(FaceTemplate.candidate_id == c.id)):
        db.delete(t)
        faces += 1
    db.flush()
    _audit(db, day, "admin", "delete_person", None, **info, faces_removed=faces)
    db.delete(c)
    db.commit()


EXPORT_COLUMNS = CSV_COLUMNS + ("faces", "days_tested", "last_test_date", "last_verdict")


def _cell(v) -> str:
    """A spreadsheet must never run a cell as a formula (CSV injection): text starting with
    = + - @ or a tab is kept as text."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def export_register(db: Session, day: Day) -> str:
    """The whole register as CSV (UTF-8 with BOM for Excel), importable again as it is."""
    out = io.StringIO()
    out.write("\ufeff")
    w = csv.writer(out)
    w.writerow(EXPORT_COLUMNS)
    offset = 0
    while True:
        page = register_rows(db, day, offset=offset, limit=500)["rows"]
        for r in page:
            w.writerow([_cell(r[k]) for k in CSV_COLUMNS] + [r["faces"], r["days_tested"], r["last_date"] or "",
                                                             r["last_verdict"] or ""])
        if len(page) < 500:
            break
        offset += 500
    _audit(db, day, "admin", "export_register", None)
    db.commit()
    return out.getvalue()


# --------------------------------------------------------------------------- reset a day

def reset_day(db: Session, day: Day, seed: DayRules, keep_list: bool = True, stop_running: bool = False) -> dict:
    """Start today again: every weigh-in, run start, finish and exercise attempt (with its reps) of
    the day is deleted. ``keep_list``: everyone stays on today's list as expected; otherwise
    the list is emptied too. The register (people and faces), the settings and the rules are
    kept; the audit log keeps the history and a summary of what was reset. An exercise attempt
    still open is reset only with ``stop_running`` (it may be someone being counted now)."""
    entries = list(db.scalars(select(Entry).where(Entry.day_id == day.id)))
    open_ = [e.candidate.name for e in entries if any(a.status == "running" for a in e.attempts)]
    if open_ and not stop_running:
        raise ServiceError("in_progress", f"an exercise attempt is open for {', '.join(open_)}", 409, open=open_)
    counts = {"people": len(entries), "with_results": sum(1 for e in entries if e.status != J.EXPECTED or e.attempts),
              "attempts": 0, "crossings": 0}
    for e in entries:
        for a in list(e.attempts):
            for r in list(a.reps):
                db.delete(r)
            db.delete(a)
            counts["attempts"] += 1
    for c in db.scalars(select(Crossing).where(Crossing.day_id == day.id)):
        db.delete(c)
        counts["crossings"] += 1
    db.flush()
    for e in entries:
        if keep_list:
            e.status = J.EXPECTED
            for k in ("weight_kg", "height_cm", "bmi", "weighed_at", "lined_up_at", "wave_id", "run_started_at", "run_cutoff_at", "run_time_s",
                      "finished_at", "age", "band", "level", "run_grade"):
                setattr(e, k, None)
        else:
            for a in db.scalars(select(AuditLog).where(AuditLog.entry_id == e.id)):
                a.entry_id = None   # keep the history, drop the link to the deleted row
            db.delete(e)
    db.flush()
    for w in db.scalars(select(Wave).where(Wave.day_id == day.id)):   # legacy group starts
        db.delete(w)
    day.run_started_at = day.run_closed_at = None
    r = current_rules(db, seed)   # nothing is scored any more: today follows the current rules again
    day.rules_version, day.rules_json = r.version, json.dumps(r.to_dict())
    _audit(db, day, "admin", "reset_day", None, keep_list=keep_list, **counts)
    db.commit()
    return counts
