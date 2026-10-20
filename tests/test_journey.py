import pytest

from testpoint.api import journey as J
from testpoint.api.rules import DayRules, bmi


def test_bmi():
    assert bmi(72, 176) == 23.2
    with pytest.raises(ValueError):
        bmi(0, 170)


def test_weigh_in_only_once():
    assert J.check_gate(J.EXPECTED, "weigh").ok
    for s in J.STATUSES:
        if s != J.EXPECTED:
            g = J.check_gate(s, "weigh")
            assert not g.ok and g.reason == "already_weighed"


def test_lineup_needs_acceptance_and_runs_once():
    assert J.check_gate(J.ACCEPTED, "lineup").ok
    assert J.check_gate(J.EXPECTED, "lineup").reason == "not_weighed"
    assert J.check_gate(J.REJECTED, "lineup").reason == "rejected_at_weigh_in"
    assert J.check_gate(J.LINED_UP, "lineup").reason == "already_lined_up"
    assert J.check_gate(J.ABSENT, "lineup").ok                     # an old day's absentee may run in a wave
    assert J.check_gate(J.RUNNING, "lineup").reason == "still_running"
    assert J.check_gate(J.RUN_PASSED, "lineup").reason == "already_ran"


@pytest.mark.parametrize("status,reason", [
    (J.EXPECTED, "not_weighed"),
    (J.ACCEPTED, "has_not_run"),
    (J.LINED_UP, "has_not_run"),
    (J.RUNNING, "still_running"),
    (J.REJECTED, "rejected_at_weigh_in"),
    (J.ABSENT, "missed_the_run"),
    (J.COMPLETE, "event_already_done"),
])
def test_exercise_stations_refuse_without_a_run_pass(status, reason):
    for event in ("pushup", "crunch"):
        g = J.check_gate(status, event)
        assert not g.ok and g.reason == reason


def test_exercise_stations_accept_run_pass_once_per_event():
    assert J.check_gate(J.RUN_PASSED, "pushup").ok
    assert J.check_gate(J.RUN_PASSED, "crunch", done_events={"pushup"}).ok
    assert J.check_gate(J.RUN_PASSED, "pushup", done_events={"pushup"}).reason == "event_already_done"


def test_transitions():
    assert J.after_weigh_in(True) == J.ACCEPTED
    assert J.after_weigh_in(False) == J.REJECTED
    assert J.after_finish(960, 960) == J.RUN_PASSED
    assert J.after_finish(960.1, 960) == J.RUN_FAILED
    assert J.after_finish(960, None) == J.RUN_DONE           # no pass table: recorded only
    assert J.after_finish(1600, 960, 1500) == J.DNF
    assert J.after_close(J.RUNNING) == J.DNF
    assert J.after_close(J.RUN_PASSED) == J.RUN_PASSED
    assert J.after_event({"pushup"}, J.RUN_FAILED) == J.RUN_FAILED
    assert J.after_event({"pushup", "crunch"}, J.RUN_PASSED) == J.COMPLETE


def test_unknown_inputs_raise():
    with pytest.raises(ValueError):
        J.check_gate("flying", "weigh")
    with pytest.raises(ValueError):
        J.check_gate(J.EXPECTED, "pullup")
    with pytest.raises(KeyError):
        DayRules().men[0].needs("pullup")


def test_official_table_grades():
    r = DayRules()
    b = r.band("M", 27)
    assert b.label == "23-30" and r.band("M", 51) is None and r.band("F", 27) is None
    assert r.band("M", 17).label == "17-22" and r.band("M", 50).label == "46-50"
    # run: at or under the time reaches the column
    assert [r.grade("run", t, b) for t in (910, 911, 1060, 1110, 1111)] == [100, 70, 70, 60, 0]
    # reps: at or over the count
    assert [r.grade("pushup", n, b) for n in (35, 36, 41, 42, 57, 58)] == [0, 60, 60, 70, 70, 100]
    assert r.level("combat") == r.level("admin") == 60 and r.level("special") == 70
    r.check()
    assert J.check_gate(J.RUN_FAILED, "pushup").ok and J.check_gate(J.DNF, "crunch").ok
    assert J.check_gate(J.ABSENT, "pushup").reason == "missed_the_run"
