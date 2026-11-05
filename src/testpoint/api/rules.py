"""Day rules for the station app: the official pass table, by age, and the level each unit needs.

Source: "جدول درجات النجاح في اختبارات اللياقة البدنية - رجال" (the men's pass table, see
docs/RULES.md). Three tests (3,200 m run; push-ups and crunches, two minutes each); for every age
band the table gives the result needed for 60 %, 70 % and 100 %. Combat and administrative units
need 60 % in each test, special-nature units 70 %.

The whole set is versioned and fixed per test day (service.rules_for); the Settings screen edits it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

EVENTS = ("pushup", "crunch")
TESTS = ("run", "pushup", "crunch")
GRADES = (60, 70, 100)
"""The table's columns, lowest first. A result between two columns gets the lower one."""
UNITS = ("combat", "admin", "special")


def _mmss(s: str) -> int:
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


@dataclass(frozen=True)
class AgeBand:
    """One row of the table. Each tuple is the result for (60 %, 70 %, 100 %)."""
    age_min: int
    age_max: int
    run_s: tuple[int, int, int]
    """Run time (seconds) at or under which the grade is reached."""
    pushup: tuple[int, int, int]
    crunch: tuple[int, int, int]

    @property
    def label(self) -> str:
        return f"{self.age_min}-{self.age_max}"

    def needs(self, test: str) -> tuple:
        return {"run": self.run_s, "pushup": self.pushup, "crunch": self.crunch}[test]


def _band(lo, hi, run, push, crunch) -> AgeBand:
    return AgeBand(lo, hi, tuple(_mmss(t) for t in run), tuple(push), tuple(crunch))


# The official men's table, exactly as printed (the run column reads 17.10 = 17 min 10 s).
MEN = (
    _band(17, 22, ("17:10", "16:20", "13:50"), (34, 40, 56), (40, 45, 59)),
    _band(23, 30, ("18:30", "17:40", "15:10"), (36, 42, 58), (42, 47, 61)),
    _band(31, 35, ("19:10", "18:20", "15:50"), (33, 39, 55), (38, 43, 57)),
    _band(36, 40, ("19:30", "18:40", "16:10"), (31, 37, 53), (36, 41, 55)),
    _band(41, 45, ("19:50", "19:00", "16:30"), (27, 33, 49), (34, 36, 50)),
    _band(46, 50, ("20:30", "19:40", "17:10"), (24, 30, 46), (28, 33, 47)),
)


@dataclass(frozen=True)
class DayRules:
    bmi_min: float = 18.5
    bmi_max: float = 30.0
    run_cutoff_s: float = 25 * 60
    """Runners without a finish after this are DNF when the run is closed."""
    exercise_time_s: int = 120
    unit_level: dict = field(default_factory=lambda: {"combat": 60, "admin": 60, "special": 70})
    """The grade each unit type must reach in every test."""
    men: tuple[AgeBand, ...] = MEN
    women: tuple[AgeBand, ...] = ()
    """No women's table yet: women are measured and recorded without a pass/fail verdict."""
    version: str = "v2"
    """v1 was the placeholder set used before the official table arrived."""

    # ------------------------------------------------------------------ scoring
    def band(self, gender: str | None, age: int) -> AgeBand | None:
        table = self.men if gender == "M" else self.women if gender == "F" else ()
        return next((b for b in table if b.age_min <= age <= b.age_max), None)

    def has_table(self, gender: str | None) -> bool:
        return bool(self.men if gender == "M" else self.women if gender == "F" else ())

    def find_band(self, label: str | None) -> AgeBand | None:
        return next((b for b in (*self.men, *self.women) if b.label == label), None) if label else None

    def grade(self, test: str, value: float, band: AgeBand) -> int:
        """The highest column reached (60, 70 or 100), 0 below 60."""
        needs = band.needs(test)
        reached = [g for g, need in zip(GRADES, needs)
                   if (value <= need if test == "run" else value >= need)]
        return max(reached, default=0)

    def percent(self, test: str, value: float, band: AgeBand) -> float:
        """The exact percentage of a result (decided 2026-11-04): on the straight line between the
        table's columns (60, 70, 100), at most 100; below the 60 % column in proportion to it (half
        the 60 % count scores 30 %; for the run, a time twice the 60 % time scores 30 %)."""
        n60, n70, n100 = band.needs(test)
        if test == "run":                      # faster is better
            if value <= n100:
                p = 100.0
            elif value <= n70:
                p = 70 + 30 * (n70 - value) / max(n70 - n100, 1e-9)
            elif value <= n60:
                p = 60 + 10 * (n60 - value) / max(n60 - n70, 1e-9)
            else:
                p = 60 * n60 / value if value > 0 else 0.0
        else:
            if value >= n100:
                p = 100.0
            elif value >= n70:
                p = 70 + 30 * (value - n70) / max(n100 - n70, 1e-9)
            elif value >= n60:
                p = 60 + 10 * (value - n60) / max(n70 - n60, 1e-9)
            else:
                p = 60 * max(value, 0) / n60 if n60 > 0 else 0.0
        return round(min(100.0, max(0.0, p)), 1)

    def level(self, unit: str | None) -> int:
        return int(self.unit_level.get(unit or "", min(self.unit_level.values())))

    # ------------------------------------------------------------------ storage
    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("men", "women"):
            d[k] = [{**b, "run_s": list(b["run_s"]), "pushup": list(b["pushup"]), "crunch": list(b["crunch"])} for b in d[k]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DayRules":
        """Rebuild from JSON. Keys this version does not know (an older rule set) are ignored."""
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in d.items() if k in known}
        for k in ("men", "women"):
            if k in data:
                data[k] = tuple(AgeBand(int(b["age_min"]), int(b["age_max"]), tuple(int(x) for x in b["run_s"]),
                                        tuple(int(x) for x in b["pushup"]), tuple(int(x) for x in b["crunch"]))
                                for b in data[k])
        return cls(**data)

    def check(self) -> None:
        """Refuse a rule set that cannot be applied consistently."""
        if not 10 <= self.bmi_min < self.bmi_max <= 60:
            raise ValueError("BMI range must be between 10 and 60, minimum below maximum")
        if not 10 <= self.exercise_time_s <= 600:
            raise ValueError("exercise time limit must be 10 s to 10 min")
        if set(self.unit_level) != set(UNITS) or any(v not in GRADES for v in self.unit_level.values()):
            raise ValueError(f"each unit type ({', '.join(UNITS)}) needs a level of 60, 70 or 100")
        if not self.men:
            raise ValueError("the men's table is empty")
        for name, table in (("men", self.men), ("women", self.women)):
            last = None
            for b in sorted(table, key=lambda b: b.age_min):
                if not 15 <= b.age_min <= b.age_max <= 70:
                    raise ValueError(f"{name} {b.label}: ages must be 15 to 70, low to high")
                if last is not None and b.age_min <= last.age_max:
                    raise ValueError(f"{name} {last.label} and {b.label} overlap")
                if not (b.run_s[0] >= b.run_s[1] >= b.run_s[2] > 0):
                    raise ValueError(f"{name} {b.label}: run times must get faster from 60 % to 100 %")
                if b.run_s[0] > self.run_cutoff_s:
                    raise ValueError(f"{name} {b.label}: the 60 % run time is after the cut-off")
                for test in ("pushup", "crunch"):
                    n = b.needs(test)
                    if not (0 < n[0] <= n[1] <= n[2] <= 500):
                        raise ValueError(f"{name} {b.label}: {test} counts must rise from 60 % to 100 %")
                last = b
        if not self.run_cutoff_s <= 4 * 3600:
            raise ValueError("the run cut-off must be under 4 hours")


def age_on(dob: str, date: str) -> int:
    """Completed years on ``date`` (both YYYY-MM-DD)."""
    by, bm, bd = (int(x) for x in dob.split("-"))
    y, m, d = (int(x) for x in date.split("-"))
    return y - by - ((m, d) < (bm, bd))


def bmi(weight_kg: float, height_cm: float) -> float:
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("weight and height must be positive")
    m = height_cm / 100.0
    return round(weight_kg / (m * m), 1)
