"""The exercise station run by its camera: one person at a time, both exercises.

    idle ── face recognised (2 of the last 3 looks) ──► [choose] ──► position ("PUSH-UPS: get into position")
      ▲                                                    │ start posture held 1 s
      │                                                    ▼
      ├── done (score shown) ◄── testing (counting) ◄── countdown (3-2-1-GO: the attempt starts)
      └── refused (not run yet, both done, ...) for a few seconds

A test ends when the time is up, or earlier when the person cannot go on (as in an official
fitness test, the count so far stands): the examiner presses End test, the person leaves the mat
for LEFT_MAT_S, they leave the exercise's posture (stand up, sit up and stay up, kneel: the only
rest allowed is the start position, arms straight or lying back) for ``out_s`` (a setting; the
screen warns meanwhile), or, after their first rep, no rep has been counted for ``idle_s`` while
they are out of the start position (a setting; 0 = only the time).

Push-ups first, crunches when they come back (after a break, others testing meanwhile); or, with
both still to do and the setting on, the person chooses which first: a raised hand moves a cursor
on the screen and held on one test for CHOOSE_DWELL_S picks it (push-ups after CHOOSE_TIMEOUT_S). The station's camera engine (testpoint.engine --station exercise)
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
CHOOSE_DWELL_S = 1.5        # the cursor held on one test this long picks it
CHOOSE_TIMEOUT_S = 20.0     # no choice: push-ups, as without choosing
CHOICES = ("pushup", "crunch")   # on the screen left to right (push-ups first, as without choosing)


def choice_at(hand: tuple[float, float] | None) -> str | None:
    """The test under the cursor: the left or the right part of the screen (a gap between)."""
    if hand is None:
        return None
    return CHOICES[0] if hand[0] < 0.42 else CHOICES[1] if hand[0] > 0.58 else None


@dataclass
class StationLook:
    present: bool
    match: tuple[int, str | None, str | None] | None = None
    """(entry id, the test they do next or None, reason when None) for a face recognised now."""
    ready: bool = False
    """The person holds the start posture of the exercise being watched."""
    count: int | None = None
    """The engine's count for the attempt being tested (None: not counting it yet)."""
    hand: tuple[float, float] | None = None
    """Where a raised hand points on the screen (0..1 each), for choosing the first test."""
    ready_ex: str | None = None
    """The exercise whose start position the person holds now (the one the camera watches)."""
    out: bool = False
    """Seen, but not in the exercise's posture: standing, sitting up, kneeling."""


@dataclass
class StationFlow:
    countdown_s: float = 3.0
    time_s: float = 120.0
    idle_s: float = 0.0
    """No rep for this long during the test: it ends (the person stopped). 0 = off."""
    out_s: float = 0.0
    """Out of the exercise's posture this long during the test: it ends. 0 = off."""
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
    """Why the last test ended before its time: "examiner", "left_mat", "left_position" or "no_reps"."""
    last_count: int = -1
    out_since: float | None = None
    last_rep_at: float = 0.0
    off_since: float | None = None
    hand: tuple[float, float] | None = None
    hover: str | None = None
    hover_since: float | None = None

    def _to(self, state: str, now: float, **kw) -> None:
        self.state, self.since, self.ready_since, self.reason = state, now, None, None
        self.hover = self.hover_since = self.hand = None
        if state == "idle":
            self.entry_id = self.event = self.go_at = self.attempt_id = self.ended = None
            self.votes.clear()
        if state == "testing":
            self.ended, self.last_count, self.off_since, self.out_since = None, -1, None, None
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
        if self.state in ("choose", "position", "countdown") and now - self.last_look > STALE_S:
            self._to("idle", now)
        if self.state in ("refused", "done") and now >= self.until:
            self._to("idle", now)
        if self.state == "choose" and now - self.since > CHOOSE_TIMEOUT_S:
            self._to("position", now, entry_id=self.entry_id, event=CHOICES[0])
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
            self.last_rep_at, self.off_since, self.out_since = now, None, None

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
                elif event == "choose":
                    self._to("choose", now, entry_id=entry_id)
                else:
                    self._to("position", now, entry_id=entry_id, event=event)
            return acts

        if self.state == "choose":
            if now - self.last_seen > AWAY_S:
                self._to("idle", now, reason="stepped_away")
                return acts
            h = look.hand
            if h is not None and self.hand is not None:   # a little smoothing: the pose shakes a few pixels
                h = (round(0.5 * self.hand[0] + 0.5 * h[0], 3), round(0.5 * self.hand[1] + 0.5 * h[1], 3))
            self.hand = h
            # getting straight into a test's start position chooses it, as clearly as a raised hand
            if look.ready_ex in CHOICES:
                self.ready_since = self.ready_since if self.ready_since is not None else now
                if now - self.ready_since >= READY_HOLD_S:
                    self._to("position", now, entry_id=self.entry_id, event=look.ready_ex)
                    self.ready_since = now - READY_HOLD_S   # already in position: the 3-2-1 follows at once
                    return acts
            else:
                self.ready_since = None
            on = choice_at(h)
            if on is None or on != self.hover:
                self.hover, self.hover_since = on, (now if on else None)
            elif now - self.hover_since >= CHOOSE_DWELL_S:
                self._to("position", now, entry_id=self.entry_id, event=on)
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
            if look.ready:          # resting in the start position (arms straight, lying back) is allowed
                self.last_rep_at = now
            if self.off_since is not None and now - self.off_since >= LEFT_MAT_S:
                return acts + self._finish(now, "left_mat")
            # left the exercise (stood up, sat up and stayed up, kneeling): a short grace for the
            # pose's jitter and a slip; then the test is over with the count so far
            self.out_since = (self.out_since if self.out_since is not None else now) if look.present and look.out else None
            if self.out_s > 0 and self.out_since is not None and now - self.out_since >= self.out_s:
                return acts + self._finish(now, "left_position")
            # "stopped": no rep for idle_s after doing some; before the first rep (getting down on
            # the mat, a slow start) only the time, the examiner or leaving the mat ends the test
            if self.idle_s > 0 and self.last_count > 0 and now - self.last_rep_at >= self.idle_s:
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

    def choose(self, now: float, event: str) -> bool:
        """The examiner taps the test the person chose (or picks it for them)."""
        if self.state != "choose" or event not in CHOICES:
            return False
        self._to("position", now, entry_id=self.entry_id, event=event)
        return True

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
                # out of the exercise's posture: seconds before the test ends (the screen warns)
                "out_left": (round(max(0.0, self.out_s - (now - self.out_since)), 1)
                             if self.state == "testing" and self.out_s > 0 and self.out_since is not None else None),
                "attempt_id": self.attempt_id, "go_at": self.go_at, "since": self.since,
                "ends_at": (self.go_at + self.time_s) if self.go_at is not None and self.state in ("testing", "done") else None,
                "ready": self.ready_since is not None, "camera": now - self.last_look <= STALE_S,
                **({"hand": self.hand, "hover": self.hover, "choices": list(CHOICES),
                    "dwell": round(min(1.0, (now - self.hover_since) / CHOOSE_DWELL_S), 2) if self.hover_since is not None else 0.0,
                    "choose_ends_at": self.since + CHOOSE_TIMEOUT_S} if self.state == "choose" else {}),
                "present": now - self.last_seen <= 1.5}


def next_event(done: set[str], events: tuple[str, ...] = ("pushup", "crunch")) -> str | None:
    """The test this person does next at the station: push-ups first, then crunches."""
    return next((e for e in events if e not in done), None)
