# Proposed rule set v1 (for sign-off)

No official regulation has been provided yet, so these are proposed rules
based on common military/police fitness tests. Every number is a setting in
the app (Advanced → Rules) and is versioned; results always record which rule
set version scored them. Please mark anything that differs from your
regulation.

## Push-ups

| Rule | Proposed value | Measured how |
|------|----------------|--------------|
| Start position | Arms extended, elbow angle ≥ 155°; body straight (shoulder–hip–ankle ≥ 160°); hands on the ground | Counting is armed only after this position is held for 1 s |
| Valid rep | Elbow angle goes to ≤ 90° (upper arm parallel to the floor) and back to ≥ 155° | Hysteresis state machine, 2 frames at each end |
| Body line | Shoulder–hip–ankle angle must stay ≥ 160° during the rep (no sagging hips, no piking) | Checked every frame from leaving the top until back at the top; a fault rejects that rep |
| Pause | Allowed, only in the up position; no limit on pause length within the time limit | Resting in the down position or on the knees is a fault for that rep (knee-on-ground check added in Phase 1) |
| Time limit | 2:00 (to confirm) | Countdown from the examiner's Start |
| Score | Count of valid reps | Rejected reps shown separately with reasons |

## Crunches

| Rule | Proposed value | Measured how |
|------|----------------|--------------|
| Start position | Lying on the back, knees bent 45–135°, feet on the floor, hands crossed on the chest (hand position not enforced in v1) | Armed after the lying position is held for 1 s |
| Valid rep | Torso elevation (hip→shoulder line above horizontal) rises from ≤ 15° to ≥ 35°, then back to ≤ 15° | Hysteresis state machine. Set 70° instead of 35° if the test is a full sit-up |
| Knees | Knee angle stays within 45–135° during the rep | A fault rejects that rep |
| Pause | Allowed in the down position | — |
| Time limit | 2:00 (to confirm) | Countdown from Start |
| Score | Count of valid reps | — |

## 3.2 km run

| Rule | Proposed value |
|------|----------------|
| Course | Point-to-point, measured 3.2 km (certified by the organiser) |
| Start | Waves of up to 50 runners; wave start time recorded when the examiner presses Start with the horn |
| Finish | Time = finish-line crossing − wave start, interpolated between frames, resolution 0.1 s |
| Identification | Bib number (front and back) read by the finish cameras; face as secondary check; unresolved crossings reviewed by the examiner from video before results are published |
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

Unknown. The app ships with an editable table per event, gender and age band
(Advanced → Rules → Score tables). Placeholder values are used until the
official tables are provided.
