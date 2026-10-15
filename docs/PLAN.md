# TestPoint — delivery plan

Last updated: 2026-10-15

Each phase ends with something that can be demonstrated and a clear exit test.
We do not start the next phase until the current one passes its exit test on
real footage from the real site.

## Guiding principles (best practice for a scored test)

1. **Evidence first.** Every counted rep and every run time is backed by a
   stored video clip and a timestamp. An examiner can replay, override and
   annotate. Without this the system cannot be used for an official result.
2. **Deterministic rules, tunable thresholds.** The exercise rules (depth,
   body line, pause) are written as explicit, configurable thresholds, not
   learned black boxes, so they can be matched to the official test regulation
   and explained to a candidate who disputes a result.
3. **Human in the loop.** AI proposes, the examiner confirms. The UI always
   allows a manual correction, and corrections are logged.
4. **Offline on-premise.** No internet needed at the test site. All video and
   data stay on the PC.
5. **Test-footage driven development.** We record real candidates early
   (Phase 1) and build a labelled clip library. Every change is measured
   against that library (precision/recall of rep counts, timing error).
6. **Two UI modes.** A **Simple** mode for examiners on test day (big buttons,
   live count, pass/fail, nothing else) and an **Advanced** mode for the
   administrator (camera setup, calibration, rule thresholds, per-rep review,
   exports, diagnostics). One toggle switches between them; Simple is the
   default.
7. **Camera check-in gates every test.** The administrator loads the day's
   candidate list in advance. A candidate steps in front of the station
   camera, is recognised by face against that list, and only then can the
   examiner start the test. Unrecognised people cannot be tested; a
   candidate who is not on today's list is refused with a clear message.
   The check-in photo is stored with the attempt as evidence.

## Test-day flow

```
Admin (Advanced): import today's list (ID, name, DOB, gender, bib, photo)
        │
Candidate steps in front of station camera
        │
Face match against today's list ──no match──> "Not registered for today" (examiner can search manually)
        │ match
"Verified: <name>, bib 23"  ──examiner taps Accept──> test is armed
        │
Examiner taps Start → countdown → counting with live overlay → time limit ends
        │
Result shown (count / time, faults, pass/fail) → examiner confirms → next candidate
```

## Phase 0 — Setup and requirements (1 week)

- Install the toolchain on the PC: Python 3.12, Git, Node.js LTS, CUDA-ready
  ONNX Runtime.
- Confirm hardware details (see [QUESTIONS.md](QUESTIONS.md)): camera model
  and protocol, switch PoE, run course layout.
- Write down the official test rules for each event (what is a valid push-up,
  what is a valid crunch, time limits, scoring tables by age/gender).
- Record the first batch of reference videos: at least 10 people doing
  push-ups and crunches from the planned camera angle, plus a few runners
  crossing a line wearing bibs.

**Exit test:** repository builds, `pytest` passes, 10+ reference clips stored
under `data/clips/`, rules document agreed.

## Phase 1 — Rep-counting engine (2–3 weeks)

Scope: push-ups and crunches, one person per camera, offline video and live
webcam.

- Pose estimation backend (RTMPose via `rtmlib`, ONNX Runtime, GPU).
- Joint-angle geometry and a hysteresis-based rep state machine.
- Push-up rules: elbow angle down/up thresholds, straight body line, hands on
  ground. Crunch rules: torso elevation thresholds, feet/knees position.
- Form faults are flagged per rep (e.g. "hips sagging", "not deep enough")
  so an invalid rep is *not counted* and the reason is recorded.
- Demo CLI with overlay, and a measurement script that runs the engine over
  the clip library and prints accuracy against hand-counted truth.

**Exit test:** on the clip library, rep count within ±1 of the human count on
95% of clips; no false counts when a person gets into position or rests.

## Phase 2 — Station app: web UI, database, evidence (3–4 weeks)

- FastAPI backend, PostgreSQL (or SQLite to begin), React web UI.
- Candidate registration (ID, name, bib number, photo), daily candidate
  list import (CSV/Excel), test sessions, station assignment.
- **Camera check-in:** face embedding (InsightFace, on-premise) of the
  person in front of the station camera matched against today's list;
  verified identity shown to the examiner for Accept/Reject; check-in
  photo stored with the attempt. Manual search fallback for a failed match.
- Live station view: camera feed with overlay, countdown timer, count,
  fault messages. Start/stop by the examiner, time limit enforced.
- Evidence: full station video per attempt plus a short clip per rep,
  stored with timestamps; review screen with override and audit log.
- Simple / Advanced toggle: Simple shows only what the examiner needs on
  test day; Advanced exposes calibration, rule thresholds, camera
  assignment, per-rep review and diagnostics.
- Scoring tables and pass/fail; CSV/PDF export.

**Exit test:** a full push-up and crunch test for 5 candidates run end to end
from the web UI with results and evidence stored.

## Phase 3 — Multi-camera capture service (2 weeks)

- Capture service that reads all 4 cameras over RTSP (or USB), keeps a
  common clock, and feeds frames to the vision workers.
- Camera health monitoring, reconnect, frame-drop reporting.
- GPU scheduling so that 2 pose stations plus 2 line cameras run in real
  time on one PC.

**Exit test:** 4 cameras running for 2 hours with no stall and timestamps
agreeing within 50 ms.

## Phase 4 — 3.2 km run timing (4–6 weeks, highest risk)

- Line camera(s) at start and finish (or one camera at a combined
  start/finish line for a loop course).
- Runner identification: bib-number detection + OCR, backed by person
  tracking and candidate photo match. Bib design is chosen for the camera
  (large, high-contrast digits, front and back).
- Start event: examiner presses Start (or a start signal is detected); all
  runners share the same start timestamp. Finish: line-crossing time per
  runner from the tracked bounding box, interpolated between frames.
- Unknown/uncertain crossings are queued for the examiner to resolve from
  the finish-line video, in Advanced mode.
- Lap counting if the course is a loop.

**Exit test:** 20 runners, finish time within ±0.5 s of a stopwatch for every
identified runner, and every unidentified crossing presented for review.

Risk note: RFID chip timing is the proven standard for official race timing.
Camera-only timing is achievable for groups of this size with good bibs, but
we design the finish module so that an RFID reader can be added later as a
second source if the accuracy target is not met.

## Phase 5 — Hardening and deployment (2–3 weeks)

- Installer / service setup on the PC (auto-start, logs, backups).
- Load test, failure drills (camera unplugged, PC reboot mid-session).
- Role-based access (examiner vs administrator), audit export.
- User manual for Simple mode, admin manual for Advanced mode.
- Site acceptance test with the customer.

## Phase 6 — Optional extensions

- Additional events (pull-ups, plank, sit-and-reach).
- Multi-PC / multi-site central reporting.
- Mobile examiner app.

## Timeline summary

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 0     | 1 wk     | 1 wk       |
| 1     | 2–3 wk   | 4 wk       |
| 2     | 3–4 wk   | 8 wk       |
| 3     | 2 wk     | 10 wk      |
| 4     | 4–6 wk   | 16 wk      |
| 5     | 2–3 wk   | 19 wk      |

Phases 3 and 4 can start in parallel with the end of Phase 2 if a second
developer is available.
