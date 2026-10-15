# TestPoint — architecture

## Overview

```
 4 x cameras ──(RTSP/PoE)──> switch ──> PC (Windows 11, RTX 5080)
                                          │
                                          ├─ capture service   (one reader per camera, shared clock)
                                          ├─ vision workers    (pose + rep counting; line crossing + bib OCR)
                                          ├─ API server        (FastAPI, WebSocket for live updates)
                                          ├─ database          (SQLite now, PostgreSQL later)
                                          ├─ evidence store    (video files on disk, indexed in DB)
                                          └─ web UI            (React, served by the API server)
                                                 ▲
                      examiner tablet / laptop ──┘  (same LAN, browser only)
```

Everything runs on the single PC. The examiner never installs anything; they
open a browser on the LAN.

## Camera plan (4 cameras)

| Camera | Role                        | Placement                                   |
|--------|-----------------------------|---------------------------------------------|
| 1      | Push-up station             | Side view, 3–4 m from the mat, ~0.8 m high  |
| 2      | Crunch station              | Side view, 3–4 m from the mat, ~0.8 m high  |
| 3      | Run start line              | Facing runners, bibs visible, ~1.5 m high   |
| 4      | Run finish line             | Facing runners, bibs visible, ~1.5 m high   |

If the course is a loop, cameras 3 and 4 both watch the same line from two
angles (front and back), which doubles bib-read chances.

Side view is the right angle for push-ups and crunches: elbow, hip and knee
angles are all visible, and depth is unambiguous.

## Components

### Capture service (Phase 3)
- One process per camera. Reads RTSP via OpenCV/GStreamer (hardware
  decoding on the GPU where available).
- Stamps every frame with a monotonic PC clock time. All timing is done in
  PC time, never camera time, so the 4 streams are comparable.
- Writes a rolling recording to disk (evidence) and publishes frames to the
  vision worker over shared memory / local socket.
- Reports health (fps, drops, reconnects) to the API.

### Vision workers (Phase 1 and 4)
- **Pose worker** (stations 1 and 2): RTMPose (via `rtmlib`, ONNX Runtime
  on CUDA) → 17 COCO keypoints per person → picks the candidate in the
  station zone → joint angles → exercise rule set → rep state machine →
  events (`rep_counted`, `rep_rejected(reason)`, `phase_changed`).
- **Line worker** (cameras 3 and 4): person detector + tracker (ByteTrack)
  → bib detector + OCR on each track → virtual line in the image; the
  crossing time is interpolated between the two frames around the line →
  event `line_crossed(bib or unknown, t)`.
- Workers are plain Python modules that can also run on a video file, which
  is how we test them against the clip library.

### Check-in worker (Phase 2)
- Runs on the same station camera before a test starts. Detects the
  nearest face, computes an embedding (InsightFace / ArcFace, ONNX,
  on-premise), and matches it against the embeddings of **today's list only**.
- A match above the threshold shows "Verified: name, bib" to the examiner,
  who taps Accept; below the threshold the UI says "Not registered for
  today" and offers a manual search (logged as an override).
- Enrollment photos come from the candidate import; a photo taken at the
  first check-in can be stored as a second reference.
- The check-in frame is saved with the attempt as evidence.

### API server (Phase 2)
- FastAPI. REST for CRUD (candidates, sessions, attempts, results) and
  WebSocket for live station updates (count, phase, faults, timer).
- Owns the test workflow state machine: `idle → checking_in → verified →
  running → finished → reviewed`. A test cannot enter `running` without a
  verified (or manually overridden) candidate.
- Enforces time limits, computes scores from configurable tables.
- Audit log of every manual override.

### Web UI (Phase 2)
- React + TypeScript, single page app served by the API.
- **Simple mode** (default, examiner): one station per screen, large
  text, four actions: select candidate, start, stop, confirm result.
- **Advanced mode** (administrator): camera assignment and preview, zone
  and line calibration, rule thresholds with live preview, per-rep review
  with clip playback and override, run crossing review queue, exports,
  diagnostics, user management.
- The toggle is a single control in the header; Advanced requires the
  administrator role.

### Storage
- SQLite in Phase 2 (zero setup), migrate to PostgreSQL in Phase 5 if the
  deployment needs concurrent writers or central reporting.
- Evidence video under `data/evidence/<session>/<attempt>/`, per-rep clips
  as small MP4 files with the rep index and verdict in the file name.

## Data model (first cut)

```
Candidate(id, external_id, name, dob, gender, bib, photo_path, face_embedding)
Session(id, date, site, examiner_id, status)
DayList(id, session_id, candidate_id, status[expected|checked_in|done|absent])
CheckIn(id, session_id, candidate_id, station_id, t, match_score, photo_path,
        accepted_by, manual_override)
Attempt(id, session_id, candidate_id, event[pushup|crunch|run], station_id,
        started_at, ended_at, raw_count, final_count, time_s, score,
        pass, status[running|finished|reviewed], video_path)
Rep(id, attempt_id, index, t_down, t_up, valid, fault_reason, clip_path)
Crossing(id, attempt_id?, camera_id, t, bib_read, confidence, resolved_by)
Override(id, attempt_id, user_id, field, old, new, reason, created_at)
RuleSet(id, event, json thresholds, version, active)
ScoreTable(id, event, gender, age_min, age_max, json)
```

## Rep counting design

The state machine works on one scalar metric per frame (e.g. elbow angle
for push-ups, torso elevation for crunches) and uses **hysteresis**: the
"down" threshold must be crossed to enter the down phase and a separate,
higher "up" threshold to complete the rep. Each phase must last a minimum
number of frames. This removes double counts from jitter.

Form checks run during the rep (e.g. body-line angle for push-ups). A rep
whose checks fail is *rejected with a reason*, not counted, and the reason is
shown to the examiner and stored.

All thresholds live in a `RuleSet` so they can be tuned in Advanced mode and
versioned.

## Key decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Pose model | RTMPose via `rtmlib` (Apache-2.0) | Multi-person, accurate, runs on ONNX Runtime GPU, permissive licence. YOLO-pose is AGPL, which complicates a commercial product. MediaPipe is single-person only. |
| Runtime | ONNX Runtime (CUDA), TensorRT later | Simple install; TensorRT only if we need more throughput. |
| Language | Python for vision and API, TypeScript for the UI | Vision ecosystem is Python; the UI is a normal web app. |
| Run identification | Bib OCR + tracking, face match as backup | Cameras-only is a requirement. Bibs designed for cameras make this reliable for small groups. RFID can be added as a second source. |
| Where it runs | Single PC, LAN only | Offline, private, cheap to deploy. |
| Evidence | Full video + per-rep clips | Needed for appeals and for measuring accuracy. |

## Non-functional targets

- Pose stations: ≥ 25 fps per camera at 1080p on the RTX 5080.
- Live UI latency: count appears < 300 ms after the rep completes.
- Run timing: ± 0.5 s against a reference stopwatch.
- Zero data loss on PC crash: attempts are written to the DB at start and
  updated on every rep.
