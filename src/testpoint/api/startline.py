"""The start line, run by its camera: one runner at a time.

    idle ── face recognised (2 of the last 3 looks) ──► raise_hand ("raise a hand")
      ▲                                                    │ a hand (either, or both) held up 0.5 s
      │                                                    ▼
      ├── cooldown ◄── started (GO shown) ◄── countdown (3-2-1, their clock runs from GO)
      └── refused (already ran, not weighed in, ...) for a few seconds

The camera process (testpoint.startcam) sends what it sees a few times a second: whether
someone stands at the mark, their face embedding, and which hand is raised. This module
decides; ``observe`` returns the actions for the app to carry out on the database (line the
runner up, start them, cancel a start when they step away during the 3-2-1). Pure logic, no
database and no clock of its own: tests drive it with plain values.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

RECOGNISE_VOTES = 2          # the same person in this many of the last VOTE_WINDOW looks
VOTE_WINDOW = 3
HOLD_S = 0.5                 # a hand held up this long
RAISE_TIMEOUT_S = 15.0       # no hand: back to waiting (they stay at the line)
AWAY_S = 2.0                 # nobody at the mark this long: back to waiting
AWAY_COUNTDOWN_S = 1.5       # stepped away during the 3-2-1: the start is cancelled
REFUSED_S = 4.0
STARTED_S = 2.5              # "GO" stays on screen
COOLDOWN_S = 8.0             # the runner just started is not recognised again meanwhile
STALE_S = 3.0                # no word from the camera this long: idle


@dataclass
class Look:
    """One observation of the person at the mark."""
    present: bool
    match: tuple[int, bool, str | None] | None = None
    """(entry id, may start, reason when not) for a face recognised in this look, else None."""
    hand: str | None = None   # "left" / "right" / "both" raised, or None


@dataclass
class StartLine:
    countdown_s: float = 3.0
    state: str = "idle"
    entry_id: int | None = None
    reason: str | None = None
    since: float = 0.0
    held_since: float | None = None
    go_at: float | None = None
    until: float = 0.0
    last_seen: float = 0.0
    last_look: float = 0.0
    votes: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    recent: dict = field(default_factory=dict)    # entry id -> ignored until

    def _to(self, state: str, now: float, **kw) -> None:
        self.state, self.since = state, now
        self.held_since, self.reason = None, None
        if state == "idle":
            self.entry_id = self.go_at = None
            self.votes.clear()
        for k, v in kw.items():
            setattr(self, k, v)

    def reset(self, now: float) -> None:
        """The examiner took over (cancel, manual start) or the camera stopped."""
        self._to("idle", now)

    def tick(self, now: float) -> list[tuple[str, int]]:
        """Time passing without a new look (also called before each look)."""
        if self.state != "idle" and now - self.last_look > STALE_S:
            acts = [("cancel", self.entry_id)] if self.state == "countdown" and self.go_at and now < self.go_at else []
            self._to("idle", now)
            return acts
        if self.state in ("refused", "started") and now >= self.until:
            self._to("idle", now)
        if self.state == "countdown" and self.go_at is not None and now >= self.go_at:
            self.recent[self.entry_id] = now + COOLDOWN_S
            self._to("started", now, until=self.go_at + STARTED_S)
        if self.state == "raise_hand" and now - self.since > RAISE_TIMEOUT_S:
            self._to("idle", now, reason="timeout")
        return []

    def observe(self, now: float, look: Look) -> list[tuple[str, int]]:
        """A look from the camera; returns the actions to carry out: ("line_up", entry id),
        ("start", entry id) (its GO is now + countdown_s), ("cancel", entry id)."""
        acts = self.tick(now)
        self.last_look = now
        if look.present:
            self.last_seen = now
        self.recent = {k: v for k, v in self.recent.items() if v > now}

        if self.state == "idle":
            m = look.match
            self.votes.append(m[0] if m and m[0] not in self.recent else None)
            if m and m[0] not in self.recent and list(self.votes).count(m[0]) >= RECOGNISE_VOTES:
                entry_id, ok, reason = m
                if not ok:
                    self._to("refused", now, entry_id=entry_id, reason=reason, until=now + REFUSED_S)
                else:
                    self._to("raise_hand", now, entry_id=entry_id)
                    acts.append(("line_up", entry_id))
            return acts

        if self.state == "raise_hand":
            if now - self.last_seen > AWAY_S:
                self._to("idle", now, reason="stepped_away")
                return acts
            if look.hand:   # either hand, or both: the runner is ready
                self.held_since = self.held_since if self.held_since is not None else now
                if now - self.held_since >= HOLD_S:
                    entry_id = self.entry_id
                    self._to("countdown", now, entry_id=entry_id, go_at=now + self.countdown_s)
                    acts.append(("start", entry_id))
            else:
                self.held_since = None
            return acts

        if self.state == "countdown" and now - self.last_seen > AWAY_COUNTDOWN_S:
            entry_id = self.entry_id
            self._to("idle", now, reason="stepped_away")
            acts.append(("cancel", entry_id))
        return acts

    def view(self, now: float) -> dict:
        """What the start screen shows."""
        self.tick(now)
        return {"state": self.state, "entry_id": self.entry_id, 
                "reason": self.reason, "since": self.since, "go_at": self.go_at,
                "camera": now - self.last_look <= STALE_S, "present": now - self.last_seen <= 1.0}
