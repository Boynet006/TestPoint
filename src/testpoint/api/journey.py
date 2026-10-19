"""The candidate journey for one test day, as pure functions.

    expected ─ weigh-in ─► accepted ─ line-up ─► lined_up ─ START ─► running
        │                                 │                            │
        └──► rejected (BMI)               └──► absent (missed start)   ├─► run_passed ─► exercises ─► complete
                                                                       ├─► run_failed
                                                                       └─► dnf

Each station asks ``check_gate`` before it lets a candidate in. The database
layer applies the transitions; this module only decides, so it can be tested
without a database (see tests/test_journey.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .rules import EVENTS

EXPECTED = "expected"
ACCEPTED = "accepted"
REJECTED = "rejected"
LINED_UP = "lined_up"
ABSENT = "absent"
RUNNING = "running"
RUN_PASSED = "run_passed"
RUN_FAILED = "run_failed"
DNF = "dnf"
COMPLETE = "complete"

STATUSES = (EXPECTED, ACCEPTED, REJECTED, LINED_UP, ABSENT, RUNNING,
            RUN_PASSED, RUN_FAILED, DNF, COMPLETE)
FINISHED_FOR_DAY = (REJECTED, ABSENT, RUN_FAILED, DNF, COMPLETE)

STATIONS = ("weigh", "lineup", "pushup", "crunch")


@dataclass(frozen=True)
class Gate:
    ok: bool
    reason: str = ""
    """Machine-readable reason code when refused; the UI translates it."""


ALLOW = Gate(True)


def check_gate(status: str, station: str, done_events: set[str] | None = None,
               run_started: bool = False) -> Gate:
    """May a candidate with ``status`` enter ``station`` now?

    ``done_events`` are the exercises this candidate already finished today.
    ``run_started`` is whether the mass start has happened (the line-up closes).
    """
    done_events = done_events or set()
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if station not in STATIONS:
        raise ValueError(f"unknown station {station!r}")

    if station == "weigh":
        if status == EXPECTED:
            return ALLOW
        return Gate(False, "already_weighed")

    if station == "lineup":
        if run_started:
            return Gate(False, "run_already_started")
        if status == ACCEPTED:
            return ALLOW
        if status == LINED_UP:
            return Gate(False, "already_lined_up")
        if status == EXPECTED:
            return Gate(False, "not_weighed")
        if status == REJECTED:
            return Gate(False, "rejected_at_weigh_in")
        return Gate(False, "not_eligible")

    # exercise stations
    event = station
    if status == RUN_PASSED:
        if event in done_events:
            return Gate(False, "event_already_done")
        return ALLOW
    reasons = {
        EXPECTED: "not_weighed",
        ACCEPTED: "has_not_run",
        LINED_UP: "has_not_run",
        RUNNING: "still_running",
        REJECTED: "rejected_at_weigh_in",
        ABSENT: "missed_the_run",
        RUN_FAILED: "run_not_passed",
        DNF: "run_not_finished",
        COMPLETE: "event_already_done",
    }
    return Gate(False, reasons[status])


def after_weigh_in(accepted: bool) -> str:
    return ACCEPTED if accepted else REJECTED


def after_run_start(status: str) -> str:
    """Status of every candidate when the mass start happens."""
    if status == LINED_UP:
        return RUNNING
    if status == ACCEPTED:
        return ABSENT
    return status


def after_finish(time_s: float, pass_s: float, cutoff_s: float | None = None) -> str:
    """Passed within the pass time, DNF after the cut-off, failed in between."""
    if cutoff_s is not None and time_s > cutoff_s:
        return DNF
    return RUN_PASSED if time_s <= pass_s else RUN_FAILED


def after_close(status: str) -> str:
    """Closing the run: anyone still running did not finish."""
    return DNF if status == RUNNING else status


def after_event(done_events: set[str]) -> str:
    """Status after an exercise result is saved (``done_events`` includes it)."""
    return COMPLETE if set(EVENTS) <= done_events else RUN_PASSED
