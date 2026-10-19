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


def test_lineup_needs_acceptance_and_closes_at_start():
    assert J.check_gate(J.ACCEPTED, "lineup").ok
    assert J.check_gate(J.EXPECTED, "lineup").reason == "not_weighed"
    assert J.check_gate(J.REJECTED, "lineup").reason == "rejected_at_weigh_in"
    assert J.check_gate(J.LINED_UP, "lineup").reason == "already_lined_up"
    assert J.check_gate(J.ACCEPTED, "lineup", run_started=True).reason == "run_already_started"


@pytest.mark.parametrize("status,reason", [
    (J.EXPECTED, "not_weighed"),
    (J.ACCEPTED, "has_not_run"),
    (J.LINED_UP, "has_not_run"),
    (J.RUNNING, "still_running"),
    (J.REJECTED, "rejected_at_weigh_in"),
    (J.ABSENT, "missed_the_run"),
    (J.RUN_FAILED, "run_not_passed"),
    (J.DNF, "run_not_finished"),
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
    assert J.after_run_start(J.LINED_UP) == J.RUNNING
    assert J.after_run_start(J.ACCEPTED) == J.ABSENT
    assert J.after_run_start(J.REJECTED) == J.REJECTED
    r = DayRules()
    assert J.after_finish(r.run_pass_s, r.run_pass_s) == J.RUN_PASSED
    assert J.after_finish(r.run_pass_s + 0.1, r.run_pass_s) == J.RUN_FAILED
    assert J.after_close(J.RUNNING) == J.DNF
    assert J.after_close(J.RUN_PASSED) == J.RUN_PASSED
    assert J.after_event({"pushup"}) == J.RUN_PASSED
    assert J.after_event({"pushup", "crunch"}) == J.COMPLETE


def test_unknown_inputs_raise():
    with pytest.raises(ValueError):
        J.check_gate("flying", "weigh")
    with pytest.raises(ValueError):
        J.check_gate(J.EXPECTED, "pullup")
    with pytest.raises(ValueError):
        DayRules().exercise_pass("pullup")
