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

- [x] Install the toolchain on the PC: Python 3.12, Git, GPU ONNX Runtime
  (done 2026-10-15; Node.js LTS still to install for Phase 2).
- [x] Confirm hardware (see [SITE.md](SITE.md)): Hikvision 4 MP PoE cameras,
  TP-Link PoE+ switch, no internet at the site.
- [x] Draft the rule set for sign-off ([RULES.md](RULES.md)).
- [x] Run course layout decided: point-to-point, cameras at start, 1 km,
  2 km and finish, connected by single-mode fiber through field switches
  ([SITE.md](SITE.md)).
- [x] 2 extra cameras for the push-up and crunch stations approved (same
  model).
- [ ] Settle fiber installation, field switches and power at the remote
  points with the installer.
- [ ] Decide wave size (recommended ~50).
- [ ] Connect one camera over RTSP and record the first batch of reference
  videos: at least 10 people doing push-ups and crunches from the planned
  camera angle (side view, 3–4 lanes), plus a few runners crossing a line
  wearing bibs.
- [ ] Offline bundling: cache all model files and wheels so the installer
  never needs the internet.

**Exit test:** repository builds, `pytest` passes, 10+ reference clips stored
under `data/clips/`, rules document agreed, one Hikvision camera streaming
into the demo CLI.

## Phase 1 — Rep-counting engine (2–3 weeks)

Scope: push-ups and crunches, **3–4 lanes per camera**, offline video and
live RTSP.

- [x] Pose estimation backend (RTMPose via `rtmlib`, ONNX Runtime, GPU).
- [x] Joint-angle geometry and a hysteresis-based rep state machine.
- [x] Push-up rules: elbow angle down/up thresholds, straight body line.
  Crunch rules: torso elevation thresholds, knee angle.
- [x] Knee-on-ground and resting-in-down-position faults for push-ups.
- [x] Lane zones: assign each detected person to a lane rectangle; one
  counter per lane; ignore people outside all lanes.
- [x] RTSP input (Hikvision main stream) in the demo CLI, with reconnect
  (not yet tried against a real camera).
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
- **English and Arabic UI** with right-to-left layout; language switch in the
  header; bilingual exports.
- **Daily list import:** typed entry form, CSV/Excel upload, and scanned
  paper (photo or scanner) through OCR with a side-by-side review screen;
  every row is confirmed by the administrator before it is on the list.
- Multi-lane station screen: one tile per lane with its own candidate,
  count, timer and faults.
- Retention jobs: auto-delete evidence video and face data per the
  retention settings.

**Exit test:** a full push-up and crunch test for 12 candidates (3 lanes × 4
rounds) run end to end from the web UI in Arabic and English, with results
and evidence stored, and one list imported from a scanned sheet.

## Phase 3 — Multi-camera capture service (2 weeks)

- Capture service that reads all 6 cameras over RTSP, keeps a common clock
  (PC time, NTP served to the cameras), and feeds frames to the vision
  workers.
- Camera health monitoring, reconnect, frame-drop reporting; fiber-chain
  outage detection (which points are reachable) shown in Diagnostics.
- GPU scheduling so that 2 pose stations plus 4 line cameras run in real
  time on one PC (line cameras only need the detector at full rate; OCR runs
  on crops).
- Bring-up of the fiber network with the installer: field switches, SFPs,
  power, camera IP plan.

**Exit test:** 6 cameras running for 2 hours with no stall and timestamps
agreeing within 50 ms, including the start camera over 3 fiber hops.

## Phase 4 — 3.2 km run timing (4–6 weeks, highest risk)

Point-to-point course, up to 500 runners, in waves. Site design in
[SITE.md](SITE.md).

- Wave management: create waves from the daily list; check runners into a
  wave at the start gate with camera 1 (face or bib); the examiner's wired
  laptop at the start presses "Start wave N", which records the wave start
  time.
- Timing lines at start, 1 km, 2 km and finish: person tracking (ByteTrack)
  + bib detection + OCR; crossing time interpolated between frames; the four
  points' reads are chained per runner for splits, a course-integrity check
  (every point, in order) and extra identification chances.
- Runner identification: bib primary, face secondary; the result is only
  published when the crossing chain is consistent and the identity is
  confirmed or reviewed.
- Review queue: unknown, low-confidence or out-of-order crossings shown with
  the clips for the examiner to resolve, in Advanced mode. DNF handling at
  cut-off.
- Live board: runners in progress with last split, for the examiner at the
  finish.

**Exit test:** a wave of 50 runners through a two-lane chute: every runner
either identified automatically or queued for review; finish time within
±0.5 s of a reference stopwatch for all identified runners; every runner has
all four crossings or a review entry; zero runners assigned to the wrong bib
after review.

Risk note: RFID chip timing is the proven standard for mass events, and 500
runners is a mass event. Waves and the chute make camera-only timing
workable; if the exit test fails on real footage, a UHF RFID reader is added
at the finish as a second crossing source (the data model already supports
multiple sources per crossing).

## Phase 5 — Hardening and deployment (2–3 weeks)

- Installer / service setup on the PC (auto-start, logs, backups), fully
  offline: bundled models, wheels and fonts; UPS and evidence disk.
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
