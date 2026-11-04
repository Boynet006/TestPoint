"""The exact percentage of a result (rules.DayRules.percent): straight lines between the table's
columns, in proportion below 60 %, at most 100 %."""
from testpoint.api.rules import DayRules

R = DayRules()
B41 = R.band("M", 41)          # push-ups 27/33/49, crunches 34/36/50, run 19:50/19:00/16:30


def test_counts_at_the_columns_and_between():
    assert [R.percent("pushup", n, B41) for n in (27, 33, 49, 60)] == [60.0, 70.0, 100.0, 100.0]
    assert R.percent("pushup", 40, B41) == round(70 + 30 * 7 / 16, 1)          # 83.1
    assert R.percent("pushup", 30, B41) == 65.0                                # half way 60-70
    assert R.percent("pushup", 10, B41) == round(60 * 10 / 27, 1)              # below 60 %: 22.2
    assert R.percent("pushup", 0, B41) == 0.0


def test_the_run_faster_is_better():
    m = lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])
    assert [R.percent("run", m(t), B41) for t in ("19:50", "19:00", "16:30", "14:00")] == [60.0, 70.0, 100.0, 100.0]
    assert R.percent("run", m("17:45"), B41) == 85.0                          # half way 70-100
    assert R.percent("run", 2 * m("19:50"), B41) == 30.0                       # twice the 60 % time
