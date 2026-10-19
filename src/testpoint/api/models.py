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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Day(Base):
    __tablename__ = "day"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String(10), unique=True)
    rules_version: Mapped[str] = mapped_column(String(32))
    run_started_at: Mapped[datetime | None] = mapped_column(DateTime)
    run_closed_at: Mapped[datetime | None] = mapped_column(DateTime)


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
    run_time_s: Mapped[float | None] = mapped_column(Float)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

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
    passed: Mapped[bool | None] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(16), default="running")

    entry: Mapped[Entry] = relationship(back_populates="attempts")


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
        for col in ("note_code", "note_bib", "note_status"):
            if have is not None and col not in have:
                conn.exec_driver_sql(f"ALTER TABLE crossing ADD COLUMN {col} VARCHAR")
        # At most one running attempt per candidate, enforced by the database itself
        # (a double tap or two devices cannot create two). IF NOT EXISTS also upgrades
        # databases created before this index existed.
        conn.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS uq_attempt_one_running "
                             "ON attempt (entry_id) WHERE status = 'running'")
    return sessionmaker(engine, expire_on_commit=False)
