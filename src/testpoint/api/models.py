"""Database tables (SQLAlchemy 2, SQLite now, PostgreSQL later).

All timestamps are stored as naive UTC; the API adds the "Z" on the way out
and the browser shows local time.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text,
                        UniqueConstraint, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    __tablename__ = "candidate"
    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    name_ar: Mapped[str | None] = mapped_column(String(200))
    gender: Mapped[str | None] = mapped_column(String(1))
    dob: Mapped[str | None] = mapped_column(String(10))
    bib: Mapped[str | None] = mapped_column(String(16))
    unit: Mapped[str | None] = mapped_column(String(16))
    """Unit type (rules.UNITS): sets the level needed in every test."""
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class FaceTemplate(Base):
    """An enrolled face: 128-d SFace embedding plus the aligned 112x112 crop.

    Face data is personal data: the retention job (Phase 2 backlog) deletes
    these after the configured number of days.
    """

    __tablename__ = "face_template"
    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidate.id"), index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    photo_jpeg: Mapped[bytes] = mapped_column(LargeBinary)
    yaw: Mapped[float | None] = mapped_column(Float)
    """Head turn when taken (faces.head_yaw): the photo shown is the one nearest 0, facing the camera."""
    quality: Mapped[float | None] = mapped_column(Float)
    """How good a template the picture made (faces.quality), 0..1; None for older ones."""
    source: Mapped[str | None] = mapped_column(String(8))
    """"enrol" (registration), "adapt" (a very sure match at a station, in the venue's light) or
    "start" (the start camera, once the runner was started: the course cameras' light)."""
    day_id: Mapped[int | None] = mapped_column(Integer, index=True)
    """The test day it was taken on: faces change (a beard, a haircut) and so does the light, so
    on a day only that day's are used once there are any. None: taken before this was recorded."""
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Day(Base):
    __tablename__ = "day"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String(10), unique=True)
    rules_version: Mapped[str] = mapped_column(String(32))
    rules_json: Mapped[str | None] = mapped_column(Text)
    """The rules this day is judged by (a copy taken when the day starts; see service.rules_for)."""
    run_started_at: Mapped[datetime | None] = mapped_column(DateTime)
    run_closed_at: Mapped[datetime | None] = mapped_column(DateTime)


class Wave(Base):
    """Legacy: group starts of the 3.2 km run, from before every runner started on their own.
    Kept only so databases of those days still open; nothing writes it any more (each
    runner's start is ``Entry.run_started_at``, filled in from the wave for old days)."""

    __tablename__ = "wave"
    __table_args__ = (UniqueConstraint("day_id", "number"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    day_id: Mapped[int] = mapped_column(ForeignKey("day.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)
    closed_by: Mapped[str | None] = mapped_column(String(16))
    """"examiner", or "auto" when it closed itself after the cut-off."""
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Entry(Base):
    """One candidate on one test day: the journey state and day results."""

    __tablename__ = "entry"
    __table_args__ = (UniqueConstraint("day_id", "candidate_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    day_id: Mapped[int] = mapped_column(ForeignKey("day.id"), index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidate.id"))
    status: Mapped[str] = mapped_column(String(16), default="expected", index=True)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    height_cm: Mapped[float | None] = mapped_column(Float)
    bmi: Mapped[float | None] = mapped_column(Float)
    weighed_at: Mapped[datetime | None] = mapped_column(DateTime)
    lined_up_at: Mapped[datetime | None] = mapped_column(DateTime)
    wave_id: Mapped[int | None] = mapped_column(ForeignKey("wave.id"))
    """Legacy (group starts): see ``Wave``."""
    run_started_at: Mapped[datetime | None] = mapped_column(DateTime)
    """This runner's GO (server clock): the run time counts from here. In the future while the
    3-2-1 countdown runs."""
    run_cutoff_at: Mapped[datetime | None] = mapped_column(DateTime)
    """Still out after this: DNF by itself (the run cut-off plus a short grace for late finishers)."""
    run_time_s: Mapped[float | None] = mapped_column(Float)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    # The standard this person is judged by, fixed at weigh-in (age on the test day).
    age: Mapped[int | None] = mapped_column(Integer)
    band: Mapped[str | None] = mapped_column(String(8))
    """Age band of the pass table ("17-22"); None when no table applies (e.g. no women's table yet)."""
    level: Mapped[int | None] = mapped_column(Integer)
    """Grade needed in every test (60 or 70), from the unit type."""
    run_grade: Mapped[int | None] = mapped_column(Integer)

    candidate: Mapped[Candidate] = relationship(lazy="joined")
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="entry", lazy="selectin",
                                                     order_by="Attempt.id")


class Crossing(Base):
    """A finish-line crossing: from the camera later, typed by the examiner for now."""

    __tablename__ = "crossing"
    id: Mapped[int] = mapped_column(primary_key=True)
    day_id: Mapped[int] = mapped_column(ForeignKey("day.id"), index=True)
    entry_id: Mapped[int | None] = mapped_column(ForeignKey("entry.id"))
    at: Mapped[datetime] = mapped_column(DateTime)
    time_s: Mapped[float] = mapped_column(Float)
    bib_read: Mapped[str | None] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(16), default="manual")
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str | None] = mapped_column(String(200))
    # Machine-readable reason for the screen to translate; ``note`` stays the English audit text.
    note_code: Mapped[str | None] = mapped_column(String(32))
    note_bib: Mapped[str | None] = mapped_column(String(16))
    note_status: Mapped[str | None] = mapped_column(String(16))
    # From the finish camera: OCR vote share for the bib and the evidence clip (docs/ENGINE-LINK.md).
    confidence: Mapped[float | None] = mapped_column(Float)
    clip_path: Mapped[str | None] = mapped_column(String(300))
    wave_id: Mapped[int | None] = mapped_column(ForeignKey("wave.id"))
    """Legacy (group starts): see ``Wave``."""


class Attempt(Base):
    __tablename__ = "attempt"
    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("entry.id"), index=True)
    event: Mapped[str] = mapped_column(String(16))
    lane: Mapped[str | None] = mapped_column(String(8))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    count: Mapped[int | None] = mapped_column(Integer)
    rejected: Mapped[int | None] = mapped_column(Integer)
    # What the camera engine counted (see docs/ENGINE-LINK.md); ``count`` / ``rejected``
    # are what the examiner saved, so a difference is an examiner override.
    auto_count: Mapped[int | None] = mapped_column(Integer)
    auto_rejected: Mapped[int | None] = mapped_column(Integer)
    # Why the examiner saved a count different from the engine's (required then).
    override_reason: Mapped[str | None] = mapped_column(String(300))
    passed: Mapped[bool | None] = mapped_column(Boolean)
    grade: Mapped[int | None] = mapped_column(Integer)
    """Column of the pass table reached: 100, 70, 60, or 0 below 60."""
    status: Mapped[str] = mapped_column(String(16), default="running")

    entry: Mapped[Entry] = relationship(back_populates="attempts")
    reps: Mapped[list["Rep"]] = relationship(back_populates="attempt", order_by="Rep.index")


class Rep(Base):
    """One repetition seen by the engine, valid or rejected, with its evidence clip."""

    __tablename__ = "rep"
    __table_args__ = (UniqueConstraint("attempt_id", "index"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempt.id"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    valid: Mapped[bool] = mapped_column(Boolean)
    faults: Mapped[str] = mapped_column(Text, default="[]")   # JSON list of fault names
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    clip_path: Mapped[str | None] = mapped_column(String(300))

    attempt: Mapped[Attempt] = relationship(back_populates="reps")


class Setting(Base):
    """Station settings shared by every device (see service.SETTINGS). Value is JSON."""

    __tablename__ = "setting"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    day_id: Mapped[int | None] = mapped_column(ForeignKey("day.id"))
    entry_id: Mapped[int | None] = mapped_column(ForeignKey("entry.id"))
    station: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str | None] = mapped_column(Text)


def make_sessionmaker(db_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")   # crash-safe, readers don't block the writer
            cur.close()
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Columns added after the first release: add them to existing databases.
        have = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(crossing)")} if db_url.startswith("sqlite") else None
        for col, typ in (("note_code", "VARCHAR"), ("note_bib", "VARCHAR"), ("note_status", "VARCHAR"),
                         ("confidence", "FLOAT"), ("clip_path", "VARCHAR")):
            if have is not None and col not in have:
                conn.exec_driver_sql(f"ALTER TABLE crossing ADD COLUMN {col} {typ}")
        for table, cols in (("day", (("rules_json", "TEXT"),)),
                            ("candidate", (("unit", "VARCHAR"),)),
                            ("face_template", (("yaw", "FLOAT"), ("quality", "FLOAT"), ("source", "VARCHAR"), ("day_id", "INTEGER"))),
                            ("crossing", (("wave_id", "INTEGER"),)),
                            ("entry", (("age", "INTEGER"), ("band", "VARCHAR"), ("level", "INTEGER"), ("run_grade", "INTEGER"),
                                       ("wave_id", "INTEGER"), ("run_started_at", "DATETIME"), ("run_cutoff_at", "DATETIME"))),
                            ("attempt", (("auto_count", "INTEGER"), ("auto_rejected", "INTEGER"),
                                         ("override_reason", "VARCHAR"), ("grade", "INTEGER")))):
            have = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")} if db_url.startswith("sqlite") else None
            for col, typ in cols:
                if have is not None and col not in have:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        # Runs from the days of group starts: each runner's start is their wave's (or the day's
        # one mass start before waves existed).
        conn.exec_driver_sql(
            "UPDATE entry SET run_started_at = COALESCE("
            "(SELECT started_at FROM wave WHERE wave.id = entry.wave_id),"
            "(SELECT run_started_at FROM day WHERE day.id = entry.day_id)) "
            "WHERE run_started_at IS NULL AND (wave_id IS NOT NULL OR run_time_s IS NOT NULL OR status = 'running')")
        # At most one running attempt per candidate, enforced by the database itself
        # (a double tap or two devices cannot create two). IF NOT EXISTS also upgrades
        # databases created before this index existed.
        conn.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS uq_attempt_one_running "
                             "ON attempt (entry_id) WHERE status = 'running'")
    return sessionmaker(engine, expire_on_commit=False)
