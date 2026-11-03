"""The exercise station run by its camera: one person at a time, both exercises.

    idle ── face recognised (2 of the last 3 looks) ──► position ("PUSH-UPS: get into position")
      ▲                                                    │ start posture held 1 s
      │                                                    ▼
      ├── done (score shown) ◄── testing (counting) ◄── countdown (3-2-1-GO: the attempt starts)
      └── refused (not run yet, both done, ...) for a few seconds

A test ends when the time is up, or earlier when the person cannot go on (as in an official
fitness test, the count so far stands): the examiner presses End test, the person leaves the mat
for LEFT_MAT_S, or no rep has been counted for ``idle_s`` (a setting; 0 = only the time).

The test is chosen for them: push-ups first, crunches when they come back (after a break,
others testing meanwhile). The station's camera engine (testpoint.engine --station exercise)
sends, a few times a second: whether someone is on the mat, their face embedding while nobody
is being tested, and whether they hold the start posture of the exercise it was told to watch.
This module decides; ``observe`` returns actions for the app: ("start", entry id, event) at GO,
("finish", attempt id) when the time is up. Pure logic, tested with plain values.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

RECOGNISE_VOTES, VOTE_WINDOW = 2, 3
READY_HOLD_S = 1.0          # the start posture held this long: 3-2-1
POSITION_TIMEOUT_S = 60.0   # never got into position: back to waiting
AWAY_S = 5.0                # nobody on the mat this long before GO: back to waiting
REFUSED_S = 5.0
DONE_S = 8.0                # the score stays on screen
COOLDOWN_S = 20.0           # the person just tested is not picked up again meanwhile
STALE_S = 3.0               # no word from the camera this long before GO: idle
LEFT_MAT_S = 5.0            # off the mat this long during the test: it ends with the count so far


@dataclass
class StationLook:
    present: bool
    match: tuple[int, str | None, str | None] | None = None
    """(entry id, the test they do next or None, reason when None) for a face recognised now."""
    ready: bool = False
    """The person holds the start posture of the exercise being watched."""
    count: int | None = None
    """The engine's count for the attempt being tested (None: not counting it yet)."""


@dataclass
class StationFlow:
    countdown_s: float = 3.0
    time_s: float = 120.0
    idle_s: float = 0.0
    """No rep for this long during the test: it ends (the person stopped). 0 = off."""
    state: str = "idle"
    entry_id: int | None = None
    event: str | None = None
    reason: str | None = None
    attempt_id: int | None = None
    since: float = 0.0
    ready_since: float | None = None
    go_at: float | None = None
    until: float = 0.0
    last_seen: float = 0.0
    last_look: float = 0.0
    votes: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    recent: dict = field(default_factory=dict)
    ended: str | None = None
    """Why the last test ended before its time: "examiner", "left_mat" or "no_reps"."""
    last_count: int = -1
    last_rep_at: float = 0.0
    off_since: float | None = None

    def _to(self, state: str, now: float, **kw) -> None:
        self.state, self.since, self.ready_since, self.reason = state, now, None, None
        if state == "idle":
            self.entry_id = self.event = self.go_at = self.attempt_id = self.ended = None
            self.votes.clear()
        if state == "testing":
            self.ended, self.last_count, self.off_since = None, -1, None
            self.last_rep_at = kw.get("go_at", now)
        for k, v in kw.items():
            setattr(self, k, v)

    def reset(self, now: float) -> None:
        self._to("idle", now)

    @property
    def watching(self) -> str | None:
        """The exercise the engine should watch now (start posture, then counting)."""
        return self.event if self.state in ("position", "countdown", "testing") else None

    def tick(self, now: float) -> list[tuple]:
        acts: list[tuple] = []
        if self.state in ("position", "countdown") and now - self.last_look > STALE_S:
            self._to("idle", now)
        if self.state in ("refused", "done") and now >= self.until:
            self._to("idle", now)
        if self.state == "position" and now - self.since > POSITION_TIMEOUT_S:
            self._to("idle", now, reason="timeout")
        if self.state == "countdown" and self.go_at is not None and now >= self.go_at:
            acts.append(("start", self.entry_id, self.event))
            self._to("testing", now, entry_id=self.entry_id, event=self.event, go_at=self.go_at)
        if self.state == "testing" and self.attempt_id is not None and now >= self.go_at + self.time_s:
            acts += self._finish(now, None)
        return acts

    def _finish(self, now: float, why: str | None) -> list[tuple]:
        acts = [("finish", self.attempt_id, why)]
        self.recent[self.entry_id] = now + COOLDOWN_S
        self._to("done", now, entry_id=self.entry_id, event=self.event, attempt_id=self.attempt_id,
                 until=now + DONE_S, go_at=self.go_at, ended=why)
        return acts

    def hold(self, now: float) -> None:
        """The picture is a paused test video: a pause is not the person stopping."""
        if self.state == "testing":
            self.last_rep_at, self.off_since = now, None

    def end_now(self, now: float, why: str = "examiner") -> list[tuple]:
        """End the test being done with the count so far (the person cannot go on)."""
        if self.state != "testing" or self.attempt_id is None:
            return []
        return self._finish(now, why)

    def observe(self, now: float, look: StationLook) -> list[tuple]:
        self.last_look = now
        if look.present:
            self.last_seen = now
        acts = self.tick(now)
        self.recent = {k: v for k, v in self.recent.items() if v > now}

        if self.state == "idle":
            m = look.match
            fresh = m is not None and m[0] not in self.recent
            self.votes.append(m[0] if fresh else None)
            if fresh and list(self.votes).count(m[0]) >= RECOGNISE_VOTES:
                entry_id, event, reason = m
                if event is None:
                    self._to("refused", now, entry_id=entry_id, reason=reason, until=now + REFUSED_S)
                else:
                    self._to("position", now, entry_id=entry_id, event=event)
            return acts

        if self.state == "position":
            if now - self.last_seen > AWAY_S:
                self._to("idle", now, reason="stepped_away")
                return acts
            if look.ready:
                self.ready_since = self.ready_since if self.ready_since is not None else now
                if now - self.ready_since >= READY_HOLD_S:
                    self._to("countdown", now, entry_id=self.entry_id, event=self.event, go_at=now + self.countdown_s)
            else:
                self.ready_since = None
            return acts

        if self.state == "testing" and self.attempt_id is not None:
            # the person stopped: off the mat, or no rep for a while. The count so far stands.
            self.off_since = None if look.present else (self.off_since if self.off_since is not None else now)
            if look.count is not None and look.count != self.last_count:
                if look.count > max(self.last_count, 0):
                    self.last_rep_at = now
                self.last_count = look.count
            if self.off_since is not None and now - self.off_since >= LEFT_MAT_S:
                return acts + self._finish(now, "left_mat")
            if self.idle_s > 0 and now - self.last_rep_at >= self.idle_s:
                return acts + self._finish(now, "no_reps")
            return acts

        if self.state == "countdown" and now - self.last_seen > 2.0:
            self._to("position", now, entry_id=self.entry_id, event=self.event, reason="left_position")
        return acts

    def manual(self, now: float, entry_id: int, event: str) -> None:
        """The examiner names the person the camera did not recognise: same posture, same 3-2-1."""
        self.recent.pop(entry_id, None)
        self.last_look = self.last_seen = now
        self._to("position", now, entry_id=entry_id, event=event)

    def go_now(self, now: float) -> None:
        """The examiner starts the countdown without waiting for the start posture to be seen."""
        self.last_look = self.last_seen = now
        self._to("countdown", now, entry_id=self.entry_id, event=self.event, go_at=now + self.countdown_s)

    def started(self, attempt_id: int) -> None:
        """The app started the attempt for the "start" action."""
        self.attempt_id = attempt_id

    def view(self, now: float) -> dict:
        self.tick(now)
        return {"state": self.state, "entry_id": self.entry_id, "event": self.event, "reason": self.reason, "ended": self.ended,
                "idle_s": self.idle_s, "last_rep_at": self.last_rep_at if self.state == "testing" else None,
                "attempt_id": self.attempt_id, "go_at": self.go_at, "since": self.since,
                "ends_at": (self.go_at + self.time_s) if self.go_at is not None and self.state in ("testing", "done") else None,
                "ready": self.ready_since is not None, "camera": now - self.last_look <= STALE_S,
                "present": now - self.last_seen <= 1.5}


def next_event(done: set[str], events: tuple[str, ...] = ("pushup", "crunch")) -> str | None:
    """The test this person does next at the station: push-ups first, then crunches."""
    return next((e for e in events if e not in done), None)
