# Rule set v2: the official men's pass table

The pass table ("جدول درجات النجاح في اختبارات اللياقة البدنية - رجال") is the
source of every pass mark (see [Scoring tables](#scoring-tables)). The form rules below
(how a valid rep or a valid run is measured) are still proposals for sign-off.

Everything is a setting in the app (⚙ Settings → Test rules) and is versioned. Each test day
keeps the rule set it started with: a change applies to today only while today
has no result yet, otherwise from the next test day, so everyone tested on one
day is judged by the same numbers.

## Push-ups

| Rule | Proposed value | Measured how |
|------|----------------|--------------|
| Start position | Arms extended, elbow angle ≥ 155°; body straight (shoulder–hip–ankle ≥ 160°); hands on the ground | Counting is armed only after this position is held for 1 s |
| Valid rep | Elbow angle goes to ≤ 90° (upper arm parallel to the floor) and back to ≥ 155° | Hysteresis state machine, 2 frames at each end |
| Body line | Shoulder–hip–ankle angle must stay ≥ 150° during the rep (no sagging hips, no piking) | Checked every frame from leaving the top until back at the top; a fault rejects that rep only when it lasts 0.2 s (one-frame pose jitter is ignored). Was 160°: rejected clean reps on real video, see [CALIBRATION.md](CALIBRATION.md) |
| Pause | Allowed, only in the up position; no limit on pause length within the time limit | Resting in the down position (over 3 s) or on the knees is a fault for that rep. Knee on the ground: bent knee as low as the ankle, or below hip and ankle with the feet raised, for 0.2 s ([CALIBRATION.md](CALIBRATION.md)) |
| Time limit | 2:00 (to confirm) | Countdown from the examiner's Start |
| Score | Count of valid reps | Rejected reps shown separately with reasons |

## Crunches

| Rule | Proposed value | Measured how |
|------|----------------|--------------|
| Start position | Lying on the back, knees bent 20–135°, feet on the floor, hands crossed on the chest (hand position not enforced in v1) | Armed after 1 s of lying (torso under 40°); the lying level is kept when the examiner presses Start |
| Valid rep | Torso elevation (hip→shoulder line above horizontal) rises ≥ 7° above the candidate's own lying level, then back to ≤ 4° above it; at least 0.4 s | Measured from the lying level so the camera angle does not matter. A crunch lifts the torso 10–15° (shoulder blades off the floor); for a full sit-up set about 60°. Calibrated on public clips, [CALIBRATION.md](CALIBRATION.md) |
| Knees | Knee angle stays within 20–135° during the rep | A fault lasting 0.2 s rejects that rep (straight legs) |
| Pause | Allowed in the down position | — |
| Time limit | 2:00 (to confirm) | Countdown from Start |
| Score | Count of valid reps | — |

## 3.2 km run

| Rule | Proposed value |
|------|----------------|
| Course | Point-to-point, measured 3.2 km (certified by the organiser) |
| Start | Waves of up to 50 runners; wave start time recorded when the examiner presses Start with the horn |
| Finish | Time = finish-line crossing − wave start, interpolated between frames, resolution 0.1 s |
| Splits | Crossings at 1 km and 2 km recorded per runner; a runner missing a split, or crossing points out of order, is held for review before a result is published |
| Identification | Bib number (front and back) read at all four timing points; face as secondary check; unresolved crossings reviewed by the examiner from video before results are published |
| Result | mm:ss.s; pass/fail and score from the score table for the candidate's age and gender |
| Did not finish | A runner with no crossing within the cut-off time is marked DNF |

## Check-in

| Rule | Proposed value |
|------|----------------|
| Who may test | Only candidates on today's list |
| Match threshold | Face similarity ≥ 0.70 (tuned on site); below that, the examiner searches by name/ID and the override is logged |
| Evidence | Check-in photo stored with the attempt |

## Data retention (proposed)

| Data | Kept for |
|------|----------|
| Results, scores, audit log | Permanently (exported to CSV/PDF per session) |
| Evidence video and rep clips | 30 days, then auto-deleted |
| Check-in photos and face embeddings | 7 days after the test day, then auto-deleted |

## Scoring tables

Three tests, each graded 60 %, 70 % or 100 % by the candidate's **age on the test day**:
the 3,200 m run, push-ups (2 minutes) and crunches (2 minutes). A result between two
columns gets the lower one; below the 60 % column the test is not passed.

| Age | Run 60 % | 70 % | 100 % | Push-ups 60 % | 70 % | 100 % | Crunches 60 % | 70 % | 100 % |
|-----|---------|------|-------|---------------|------|-------|---------------|------|-------|
| 17–22 | 17:10 | 16:20 | 13:50 | 34 | 40 | 56 | 40 | 45 | 59 |
| 23–30 | 18:30 | 17:40 | 15:10 | 36 | 42 | 58 | 42 | 47 | 61 |
| 31–35 | 19:10 | 18:20 | 15:50 | 33 | 39 | 55 | 38 | 43 | 57 |
| 36–40 | 19:30 | 18:40 | 16:10 | 31 | 37 | 53 | 36 | 41 | 55 |
| 41–45 | 19:50 | 19:00 | 16:30 | 27 | 33 | 49 | 34 | 36 | 50 |
| 46–50 | 20:30 | 19:40 | 17:10 | 24 | 30 | 46 | 28 | 33 | 47 |

Run: at or under the time. Push-ups and crunches: at or over the count.

**Level needed** (in every test), from the candidate's unit type: combat and administrative
units 60 %, special-nature units 70 %. The unit type comes from the list (CSV column `unit`:
`combat`, `admin`, `special`, or the Arabic wording) or the registration form, with a station
default for lists without it; the examiner confirms it at weigh-in.

**How the app applies it**

- At weigh-in the age band and level are fixed for the day (age from the date of birth on
  the test day). A missing gender or date of birth is asked for at the scale; an age outside
  17–50 is refused.
- Everyone who runs takes all three tests; the day's result is **pass** only when every test
  reaches the required level. Each test's grade (100 / 70 / 60 / below 60) is recorded.
- **Women:** the sheet is the men's table. Until a women's table is entered, women are
  measured and recorded without a pass/fail verdict ("No standard").

To check with the issuing authority: at ages 23–30 the push-up and crunch counts are
**higher** than at 17–22 (58/42/36 vs 56/40/34). The app uses the sheet exactly as printed.
