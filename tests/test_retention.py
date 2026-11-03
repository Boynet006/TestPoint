import os
import time

from testpoint import retention as R

DAY = 24 * 3600


def put(root, rel, size, age_days):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    t = time.time() - age_days * DAY
    os.utime(p, (t, t))
    return p


def make(tmp_path):
    put(tmp_path, "2026-10-01/1/full.mp4", 100, 32)
    put(tmp_path, "2026-10-01/1/rep-001.mp4", 10, 32)
    put(tmp_path, "2026-10-20/2/full.mp4", 100, 13)
    put(tmp_path, "2026-11-02/3/full.mp4", 100, 1)
    put(tmp_path, "2026-11-03/4/full.mp4", 100, 0.1)          # today
    put(tmp_path, "run/old-t1.mp4", 50, 40)
    return R.scan(tmp_path)


def rules(items, **kw):
    base = {"today": "2026-11-03", "keep_days": 30, "max_bytes": 0, "free_bytes": None, "min_free_bytes": 0,
                "held_days": set(), "in_use_paths": set(), "open_attempts": set()}
    return R.plan(items, **(base | kw))


def test_scan_groups_an_attempt_and_dates_loose_clips(tmp_path):
    items = {i.key: i for i in make(tmp_path)}
    assert items["2026-10-01/1"].size == 110 and items["2026-10-01/1"].kind == "station"
    assert items["run/old-t1.mp4"].kind == "run" and items["run/old-t1.mp4"].day < "2026-10-01"


def test_age_deletes_old_days_but_not_held_or_in_use(tmp_path):
    items = make(tmp_path)
    p = rules(items)
    assert {i.key for i, _ in p.delete} == {"2026-10-01/1", "run/old-t1.mp4"}
    held = rules(items, held_days={"2026-10-01"}, in_use_paths={"run/old-t1.mp4"})
    assert held.delete == []


def test_space_deletes_oldest_first_and_never_today(tmp_path):
    items = make(tmp_path)
    p = rules(items, keep_days=0, max_bytes=250)
    assert [i.key for i, _ in p.delete] == ["run/old-t1.mp4", "2026-10-01/1", "2026-10-20/2"]
    assert not p.over_limit
    tight = rules(items, keep_days=0, max_bytes=50)
    assert "2026-11-03/4" not in {i.key for i, _ in tight.delete} and tight.over_limit
    floor = rules(items, keep_days=0, free_bytes=100, min_free_bytes=200)   # 100 short: the oldest two go
    assert [i.key for i, _ in floor.delete] == ["run/old-t1.mp4", "2026-10-01/1"]


def test_an_open_attempt_and_fresh_files_are_kept(tmp_path):
    items = make(tmp_path)
    assert rules(items, keep_days=0, max_bytes=1, open_attempts={"3"}, settle_s=0).delete[-1][0].key == "2026-10-20/2"
    fresh = put(tmp_path, "run/new-t9.mp4", 10, 40)
    os.utime(fresh, None)               # written just now, though its name looks old
    assert "run/new-t9.mp4" not in {i.key for i, _ in rules(R.scan(tmp_path), keep_days=0, max_bytes=1).delete}


def test_apply_deletes_and_tidies_empty_days(tmp_path):
    items = make(tmp_path)
    n, freed = R.apply(rules(items), tmp_path)
    assert (n, freed) == (2, 160)
    assert not (tmp_path / "2026-10-01").exists() and (tmp_path / "2026-10-20").exists()
