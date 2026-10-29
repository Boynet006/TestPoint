# Test-day flow (candidate journey)

Described by the owner on 2026-10-15. The app enforces this order: a station
refuses a candidate who has not passed the previous gate.

```
 Today's list
     │
 1. WEIGH-IN  (face check-in on a tablet camera, weight + height → BMI)
     │  BMI within limits → ACCEPTED        BMI outside → REJECTED (day over)
     ▼
 2. RUN LINE-UP  (start-line camera / examiner laptop: roster of accepted
     │            candidates, who is present, who is missing)
     ▼
 3. MASS START  (examiner presses START with the horn; one start time for all)
     │
 4. FINISH LINE (camera: crossing time + bib; time ≤ limit → RUN PASSED,
     │           otherwise FAILED; no crossing by cut-off → DNF)
     ▼  run passed
 5. PUSH-UP STATION  (lane check-in refuses anyone who has not passed the run)
     ▼
 6. CRUNCH STATION   (same gate; order between 5 and 6 is free)
     ▼
 7. DONE → results, scores, export
```

## Candidate status

`expected → weighed(accepted | rejected) → lined_up → running → run(passed |
failed | dnf) → pushups_done → crunches_done → complete`

Every transition is stamped with time, station and examiner. Advanced mode
shows the whole list with its status, so the organiser can see at any moment
how many are at each stage.

## Screens in Simple mode (one per station type)

The same web app runs on every device; the administrator sets each
device's station type once.

| Station type | Device | What the examiner sees |
|--------------|--------|------------------------|
| Weigh-in | Tablet with front camera + scale | Face check-in (tablet camera), candidate card, weight and height fields (typed, or read from a connected scale later), BMI computed live, ACCEPT / REJECT per the BMI rule, next. |
| Run start | Laptop at the start (wired to the start field switch) | Roster: accepted N, present M, missing list; big START (mass start) with confirmation; after the start: elapsed clock and finisher count. |
| Run finish | Laptop/tablet at the finish | Elapsed clock, finishers streaming in (bib, name, time, pass/fail), review-queue counter for unidentified crossings, DNF at cut-off. |
| Exercise station | Laptop/tablet by the mats | Lane board (3–4 lanes); check-in shows NOT QUALIFIED if the run is not passed. |

## Start: one runner at a time (superseded)

> Since 2026-10-29 every runner starts on their own: recognised at the line, 3-2-1-GO, their own clock. The section below is the earlier reasoning, kept for the record.

### Earlier: mass start, not waves

The owner wants a marathon-style single start once everyone is lined up.
Consequences:

- One start timestamp for all runners; the start camera records evidence
  only (no bib reading needed at the start, since weigh-in already
  established who is in the run).
- All the timing load is at the finish. Over 3.2 km, 500 runners spread
  between roughly 11 and 25 minutes, so the finish sees about 0.6 runners per
  second on average, with peaks of 2–3 per second around the median time.
  A two- or three-lane chute and the front-bib camera at 25 fps can handle
  that if bibs are large and visible; this is the number to prove in the
  Phase 4 exit test. The 1 km and 2 km cameras give two more bib reads per
  runner, which helps resolve finish-line misses.
- Fallback stays: a UHF RFID reader at the finish if the exit test fails.

## Weigh-in

- Weight and height typed by the examiner in v1. Many scales (e.g. with a
  serial/USB or Bluetooth output) can feed the value directly; that is an
  add-on once the scale model is known.
- BMI = weight / height². Acceptance limits come from the regulation
  (unknown yet); until then a configurable range (e.g. 18.5–30) in Rules.
- The weigh-in is the first and only place a candidate is added to the day's
  active set, so a candidate who skips it cannot run.

## Open questions raised by this flow

1. BMI limits (and whether they depend on age/gender).
2. How weight and height are measured today (scale model, output port).
3. Run pass time limits by age/gender, and the DNF cut-off.
4. Whether a candidate who fails the run is finished for the day (assumed yes).
5. Whether push-ups must come before crunches or the order is free (assumed free).
