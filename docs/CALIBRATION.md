# Counter calibration: push-ups and crunches (2026-10-19)

The rep counter and its fault rules were checked against public videos standing
in for the station camera, before real footage exists. Clips are in
`data/clips/pushup/` and `data/clips/crunch/` (not in git); the tool is `scripts/clip_eval.py`:

```bash
python scripts/clip_eval.py extract data/clips/pushup/*.mp4 --device cuda   # pose once, cached
python scripts/clip_eval.py score data/clips/pushup/*.mp4 --truth data/clips/pushup/truth.json --plot plots/
python scripts/clip_eval.py score clip.mp4 --set down_elbow_deg=100           # try a threshold
```

## Clips and ground truth

| Clip | What it is | Truth (by eye, from frames) |
|------|-----------|------|
| `0Qb9S8laqco` | Clean push-ups, side view, 8 s | 3 valid |
| `WDIpL0pjun0` | NASM demo, 3 camera cuts and zooms | 4 valid |
| `HMWeJ6dyq9w` | Royal Marines press-up test, oblique view, 2 min | 60 visible full reps |
| `hIkeJVV-Djk` | 1-minute record attempt with on-screen counter (76) | 21 with arms straight at the top (see below) |
| `EGf3cTu8aoQ` | 1-minute test, camera in front and above (oblique) | 5 |
| `knee_GnjOtan1wZ0` | Knee push-ups, shins on the mat | 4, all on the knees |
| `knee_3gUi-wP5NAw` | Toes, then knees with the feet raised | 2, both finished on the knees |

## What was wrong, what changed

| Rule | Before | Problem on real video | Now |
|------|--------|-----------------------|-----|
| Knee on the ground | knee distance from the wrist–ankle line < 10 % of the body | From the side the knee lies *on* that line in every clean rep (ratio ≈ 0): every rep rejected | Knee lift = knee height above the ankle ÷ hip height above the ankle < −0.3 **and** knee bent < 165°; with the feet raised to hip height, knee more than 15 % of the thigh below both hip and ankle |
| Body not straight | shoulder–hip–ankle < 160° on any frame | Single-frame pose jitter dips to 150–160 on clean reps, oblique camera too: 37 of 60 Royal Marines reps rejected | < 150° |
| Any form fault | counts on one frame | as above | must last **0.2 s** without a break (`fault_min_s`, `RepCounterConfig.min_fault_seconds`) |

Depth (elbow ≤ 95°) and lockout (≥ 155°) are unchanged.

## Result

| Clip | Truth | Before | After |
|------|-------|--------|-------|
| Clean side view | 3 | 0 valid, 3 rejected | **3** |
| NASM, camera cuts | 4 | 2 valid | **4** |
| Royal Marines | 60 | 0 valid, 60 rejected | **60** |
| Knee push-ups | 0 valid | 0 valid (for the wrong reason) | **0 valid, 4 rejected: knee on the ground** |
| Knees, feet raised | 0 valid | 2 valid | **0 valid, 2 rejected** |
| Record attempt | 21 locked-out | 0 valid, 22 rejected | **21 valid**, 1 rejected (resting down) |
| Oblique front camera | 5 | 2 | 2 — **undercounts** |

The record clip's own counter says 76: for the first 30 s he never straightens
his arms (133° at the top, `screenshots/15-calibration-lockout-fast-vs-slow.jpg`).
Under a full-lockout rule those are not reps; the counter is right to skip them.

## What this means for the site

- **Camera placement matters more than thresholds.** Side-on or up to about
  45° gives correct counts; a camera in front and above foreshortens the elbow
  and reps at 90° measure 95–100°, so they are missed (2 of 5). Keep the
  [SITE.md](SITE.md) placement: side view, 6–7 m from the mats, 0.8–1 m high.
- **Lowering the depth threshold to 100°** would rescue that oblique clip but
  over-counted it (6 for 5). It stays at 95° until our own footage says
  otherwise; the signed-off rule says 90° ([RULES.md](RULES.md)).
- These are public clips, not our candidates, mats or cameras. The Phase 1 exit
  test (±1 rep on our own reference clips) still has to be run on test-day
  footage; `clip_eval.py` is ready for it.

# Crunches

## Clips and ground truth

| Clip | What it is | Truth (by eye, 0.25 s frames) |
|------|-----------|------|
| `zHS9sClmjI8` | 2-minute crunch test, static side camera | 7 in 26–36 s, 8 in 36–46 s; 15–24 s is not crunching (arm up, fiddling) |
| `lN78c_3CRBg` | 1-minute crunch test, oblique, examiner crouched at the feet | 7 in 108–118 s |
| `Xyd_fa5zoEU` | Crunches from a 3/4 front view, trainer explaining | 14 after lying down |
| `straight_whQ7SNKdjeo` | Straight-leg sit-ups, nonstop | every rep must be rejected |
| `UFOsN57jtnE`, `0t4t3IpiEao` | Edited multi-angle sit-ups / studio demo | used for shape only (camera cuts) |

## What was wrong, what changed

| Rule | Before | Problem on real video | Now |
|------|--------|-----------------------|-----|
| Height | torso elevation 15° → 35° → 15°, absolute | A crunch (shoulder blades off the floor) lifts the torso only **10–15°**; and the lying level moves with the camera (0° side-on, 18° 3/4, 26–33° oblique). 0 counted on both test clips | Rise **above the candidate's own lying level** (low percentile over the last 6 s): up at **+7°**, back down at **+4°** |
| Lying | — | Sitting and wobbling counted as crunches | The rest position must be under **40°** absolute; sitting never arms or completes a rep and is left out of the lying level |
| Arming | 2 frames | The lie-down motion counted as a rep | Counting starts when the lying level rests on **1 s** of lying data. Not "1 s held still": someone crunching nonstop is never still for a second and scored 0 |
| Start | fresh counter | The first crunch right after Start was lost | Start keeps the lane's lying level, so the first crunch counts |
| Noise | — | Keypoint jitter made 0.2–0.4 s "reps" | Movements under **0.4 s** are ignored (real crunches take 0.8–1.4 s) |
| Knees | 45–135° | A candidate with her feet tucked in measures 31°: faulted for nothing | **20–135°**, fault held 0.2 s; straight legs (170–180°) still rejected |

## Result

| Window | Truth | Before | After |
|--------|-------|--------|-------|
| 2-min test 26–36 s | 7 | 0 | **7** |
| 2-min test 36–46 s | 8 | 0 | **8** |
| 1-min test 108–118 s (examiner in frame) | 7 | 0 | **7** |
| 3/4 view, after lying down | 14 | 0 | **14** |
| Straight-leg sit-ups | all rejected | not armed | **all rejected: knees not bent** |
| Pre-start fiddling (2-min test 15–24 s) | 0 | 0 | 2 — before Start on a real station |

The 3-lane crunch station (`screenshots/17-crunch-station-3-lanes-live.jpg`) runs
the three behaviours side by side: side-view counting, oblique with the examiner
in frame, and straight legs rejected on every rep.

## Open decision for the client

**Is the test a crunch or a full sit-up?** ([QUESTIONS.md](QUESTIONS.md) #6.)
The default now counts a crunch: shoulder blades off the floor, +7° of torso
rise. A full sit-up is a different standard: set `up_torso_deg` to about 60 (the
army sit-up clip rises 60–77°). Whichever it is, the time limit and the pass mark
come from the signed-off rules, not from these clips.

## Station camera, 2026-11-02

Real recordings from the station's IP camera, low and to the side, with the true counts known:
10 push-ups and 10 crunches, each with getting into position, walking away and sitting at a desk
around them. Their poses are in `tests/data` (keypoints only, no pictures) and
`tests/test_counting_recordings.py` replays them through the real session and counter.

| Finding | Before | Now |
|---|---|---|
| Elbow never read below 123 degrees on real push-ups | 0 of 10 | Shoulders coming down 12 % of the body's length also marks the bottom: 10 of 10 |
| Straight legs on the toes read 145-153 degrees with the hips level with the ankles | knee on the ground, rep rejected | The knee test applies below 140 degrees (knees measure 95-135) |
| Arm moving while seated at a desk; standing with straight arms | counted; started the test | Torso above 60 degrees or knee below 90 degrees is not measured |
| Not lying fully back between fast crunches | 10 merged into 5 | Rise 15 degrees, back down 12 degrees: 10 of 10 |

Not yet checked on real footage: that clearly shallow push-ups and half crunches are rejected.
A crunch rising less than 15 degrees no longer counts; for a test that accepts shoulder blades
just off the floor (10-15 degrees) lower `up_torso_deg` and `down_torso_deg` in the tune panel.
