"""Evidence retention: how long the cameras' recordings are kept, and what goes when space runs out.

The cameras record into the evidence folder (src/testpoint/evidence.py):

- ``<day>/<attempt>/``: the exercise station, one full video per attempt and a clip per rep;
- ``run/``, ``start/``: the finish line's and the start line's clips, one file each.

The rules work like a CCTV recorder's, oldest first:

1. **Age**: recordings of days older than ``keep_days`` go (0 = no age limit).
2. **Space**: while the folder is above ``max_bytes``, or the disk has less than
   ``min_free_bytes`` free, the oldest recordings go, a whole attempt or clip at a time.

Never deleted, whatever the rules say: today's recordings, the days on hold (an appeal, an
investigation), recordings still in use (an attempt not saved yet, a crossing waiting for
review) and anything written in the last hour (a camera may still be writing it). When
these alone are over the limit, nothing more is deleted and the report says so.

``plan`` only decides (the settings page shows what the next clean-up would remove);
``apply`` deletes.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STAMP = re.compile(r"^(\d{4})(\d{2})(\d{2})-\d{6}")
SETTLE_S = 3600.0
"""Recordings younger than this are left alone: a camera may still be writing them."""


@dataclass
class Item:
    """One deletable unit: an attempt's folder or a single clip."""
    key: str            # path relative to the evidence folder, with "/"
    path: Path
    day: str            # ISO date it belongs to
    size: int
    mtime: float
    kind: str           # "station", or the camera's folder name ("run", "start")


@dataclass
class Plan:
    delete: list[tuple[Item, str]] = field(default_factory=list)   # (item, "age" | "space")
    total_bytes: int = 0
    free_bytes: int | None = None
    held_bytes: int = 0
    """Bytes the rules may not touch (today, holds, in use, being written)."""
    over_limit: bool = False
    """The limits cannot be met without deleting recordings that are kept on purpose."""

    @property
    def delete_bytes(self) -> int:
        return sum(i.size for i, _ in self.delete)


def _files(path: str):
    """(path, size, mtime) of every file under ``path``; one directory listing per folder
    (stat calls are slow on a mounted Windows folder)."""
    with os.scandir(path) as it:
        for e in it:
            if e.is_dir(follow_symlinks=False):
                yield from _files(e.path)
            elif e.is_file(follow_symlinks=False):
                st = e.stat()
                yield e.path, st.st_size, st.st_mtime


def scan(root: Path) -> list[Item]:
    """Every recording in the evidence folder, as deletable units."""
    items: list[Item] = []
    if not root.is_dir():
        return items
    with os.scandir(root) as it:
        tops = sorted((e for e in it if e.is_dir(follow_symlinks=False)), key=lambda e: e.name)
    for top in tops:
        if DAY.match(top.name):
            with os.scandir(top.path) as it:
                units = sorted(it, key=lambda e: e.name)
            for u in units:
                if u.is_dir(follow_symlinks=False):
                    files = list(_files(u.path))
                    size = sum(f[1] for f in files)
                    mtime = max((f[2] for f in files), default=u.stat().st_mtime)   # the files' times, not the folder's
                else:
                    st = u.stat()
                    size, mtime = st.st_size, st.st_mtime
                items.append(Item(f"{top.name}/{u.name}", Path(u.path), top.name, size, mtime, "station"))
        else:
            for path, size, mtime in sorted(_files(top.path)):
                key = Path(path).relative_to(root).as_posix()
                m = STAMP.match(Path(path).name)     # "20261020-081500-..." (start line): the name says the day
                day = f"{m[1]}-{m[2]}-{m[3]}" if m else date.fromtimestamp(mtime).isoformat()
                items.append(Item(key, Path(path), day, size, mtime, top.name))
    return items


def in_use(item: Item, paths: set[str]) -> bool:
    """Whether a recording still in use (a clip path from the database) is this item or inside it."""
    return any(p == item.key or p.startswith(item.key + "/") for p in paths)


def plan(items: list[Item], *, today: str, keep_days: int, max_bytes: int, free_bytes: int | None,
         min_free_bytes: int, held_days: set[str], in_use_paths: set[str], open_attempts: set[str],
         now: float | None = None, settle_s: float = SETTLE_S) -> Plan:
    """What the rules would delete now. ``open_attempts``: attempt ids whose folder is in use."""
    now = time.time() if now is None else now
    out = Plan(total_bytes=sum(i.size for i in items), free_bytes=free_bytes)

    def kept(i: Item) -> bool:
        return (i.day >= today or i.day in held_days or now - i.mtime < settle_s
                or in_use(i, in_use_paths) or (i.kind == "station" and i.key.split("/")[1] in open_attempts))

    free = [i for i in items if not kept(i)]
    out.held_bytes = out.total_bytes - sum(i.size for i in free)
    free.sort(key=lambda i: (i.day, i.mtime))
    if keep_days > 0:
        oldest = (date.fromisoformat(today) - timedelta(days=keep_days)).isoformat()
        out.delete += [(i, "age") for i in free if i.day < oldest]
    gone = {i.key for i, _ in out.delete}
    rest = [i for i in free if i.key not in gone]
    left = out.total_bytes - out.delete_bytes
    room = None if free_bytes is None else free_bytes + out.delete_bytes

    def short() -> bool:
        return (max_bytes > 0 and left > max_bytes) or (min_free_bytes > 0 and room is not None and room < min_free_bytes)

    while short() and rest:
        i = rest.pop(0)
        out.delete.append((i, "space"))
        left -= i.size
        if room is not None:
            room += i.size
    out.over_limit = short()
    return out


def apply(p: Plan, root: Path) -> tuple[int, int]:
    """Delete what the plan says; empty day folders go too. Returns (items, bytes) deleted."""
    n = freed = 0
    for item, _ in p.delete:
        try:
            if item.path.is_dir():
                shutil.rmtree(item.path)
            else:
                item.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            continue                    # in use by a player: next time
        n += 1
        freed += item.size
    for top in root.iterdir() if root.is_dir() else []:
        if top.is_dir() and DAY.match(top.name) and not any(top.iterdir()):
            top.rmdir()
    return n, freed


def disk_free(root: Path) -> int | None:
    try:
        return shutil.disk_usage(root if root.exists() else root.parent).free
    except OSError:
        return None


def by_day(items: list[Item]) -> dict[str, dict]:
    days: dict[str, dict] = {}
    for i in items:
        d = days.setdefault(i.day, {"bytes": 0, "items": 0, "kinds": {}})
        d["bytes"] += i.size
        d["items"] += 1
        d["kinds"][i.kind] = d["kinds"].get(i.kind, 0) + 1
    return days

