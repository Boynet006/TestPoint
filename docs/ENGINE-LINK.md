# Proposal: live rep counting on the station lanes

Status: **implemented** (2026-10-19, steps 1–5 of the work plan below). Covers the Phase 2 item *"Connect the
rep-counting engine to the station lanes (live count and faults instead of
tapping), with evidence video and per-rep clips"* in [PLAN.md](PLAN.md).

## Goal

An examiner at the push-up or crunch station checks a candidate into a lane and
presses *Start test* exactly as today. From then on the lane tile shows the
count, the phase and the live faults coming from the camera; the examiner only
watches, and can correct the result before saving. Every valid and rejected rep
is stored with its times, faults and a short video clip.

## Shape

Two processes, as [ARCHITECTURE.md](ARCHITECTURE.md) and `compose.yaml` already
assume: the **station app** (FastAPI, no vision stack, no GPU) and the
**engine** (one per camera: pose model, lanes, counters, GPU). They talk over
HTTP only, engine → app, so the app never opens a camera and the engine never
touches the database.

```
camera ──► engine (MultiLaneSession) ──POST /api/engine/report──► app ──GET /api/lanes──► tablet
                    ▲                        (lane states + rep events)   │
                    └────── reply: which attempt is running per lane ─────┘
```

The engine **reports** and the app **answers with what it wants**; there is no
second channel. One request every 250 ms carries both directions, so a dead
engine is noticed within a second and the engine learns about a new attempt
within one report.

## Engine (`src/testpoint/engine.py`, new; `testpoint-engine` script)

The loop in `cli.py` moved into a reusable function, minus the preview window:

- Arguments: `--source`, `--exercise`, `--lanes`/`--lanes-file`, `--device`,
  `--mode` (as the CLI), plus `--app http://127.0.0.1:8000`, `--station pushup`,
  `--evidence data/evidence`.
- Every frame: `session.process(frame, t)`. Every 250 ms (and immediately when a
  rep event occurs) POST a **report**:

  ```json
  {"station": "pushup", "t": 12.4, "fps": 31.2, "source": "0",
   "lanes": {"1": {"attempt_id": 17, "count": 4, "rejected": 1, "phase": "work",
                   "person": true, "pending_faults": ["knee_down"], "work_seconds": 1.3,
                   "events": [{"index": 5, "valid": false, "faults": ["knee_down"],
                               "t_start": 11.0, "t_end": 12.3, "clip": "2026-10-19/17/rep-005.mp4"}]}}}
  ```

  `events` holds only reps completed since the last acknowledged report
  (resent until the app confirms the `index`, so a lost request loses nothing).
- The **reply** tells the engine what to count:

  ```json
  {"lanes": {"1": {"attempt_id": 17}, "2": {"attempt_id": null}, "3": {"attempt_id": 18}},
   "acked": {"1": 5}}
  ```

  When a lane's `attempt_id` changes the engine replaces that lane's
  `RepCounter` (fresh WAITING state, so getting into position is never counted)
  and starts the evidence writer for the attempt; `null` stops the writer and
  keeps counting only for the overlay. The counter is therefore reset by the
  examiner's *Start test*, not by the engine guessing.
- **Evidence** (`data/evidence/<day>/<attempt_id>/`): `full.mp4` for the whole
  attempt, and `rep-NNN.mp4` cut from a ring buffer (last ~4 s of frames) from
  `t_start − 0.5 s` to `t_end + 0.5 s`, written on a worker thread so the pose
  loop never waits on disk. Paths are relative to `--evidence` and go into the
  report; the app only stores them.
  **Requires ffmpeg.** The review screen plays clips in a browser `<video>`,
  which needs H.264/MP4 (yuv420p, `moov` first). OpenCV's writer cannot produce
  that without an OpenH264 DLL it normally lacks, so the writer pipes frames to
  an `ffmpeg` subprocess (`libx264 -preset veryfast -pix_fmt yuv420p -movflags
  +faststart`) found on PATH or named by `TESTPOINT_FFMPEG`. It logs which path
  it took at start-up; without ffmpeg it falls back to OpenCV's `mp4v`, which
  still records but does not play in the browser. Odd lane sizes are padded by
  one pixel to the even dimensions yuv420p needs.
- If the app is unreachable the engine keeps counting and retries; it never
  blocks the frame loop (reports are sent from a background thread with the
  latest state, one in flight at a time).

## App

- **`Rep` table** (new, from the ARCHITECTURE data model): `attempt_id, index,
  valid, faults (json), t_start, t_end, clip_path`. Unique on
  `(attempt_id, index)` so a resent event is idempotent.
- **`Attempt`** gets `auto_count`, `auto_rejected` (what the engine saw) next
  to the existing `count`, `rejected` (what the examiner saved). An attempt
  whose saved count differs from the automatic one is an override; the audit
  row for `finish_attempt` records both. The override *reason* and review
  screen are the next plan item and stay out of scope here.
- **In-memory `EngineState`** on the app (one per station): the last report,
  when it arrived, and the per-lane live values. Not persisted: it is the live
  picture, the database holds the reps.
- **`POST /api/engine/report`**: validates, attaches each event to the running
  attempt of that lane (ignores events for a lane with no running attempt or a
  different `attempt_id` — the examiner cancelled, or the report is late),
  updates `auto_count`/`auto_rejected` on the attempt, and answers with the
  running attempt per lane for that station. Goes through the existing
  write lock like every other write.
- **`GET /api/lanes?station=pushup`**: the live picture for the tablet:
  per-lane `count, rejected, phase, person, pending_faults, work_seconds,
  attempt_id`, plus `engine: {"online": true, "age_s": 0.3, "fps": 31}`.
  `online` is false when no report arrived for 2 s; the UI then falls back to
  manual tapping and says so.
- `start_attempt` is unchanged: the engine learns about it from the next reply.
  `finish_attempt` keeps its signature (`count`, `rejected`) so the manual flow
  and the existing tests keep working.

## Station UI (`web/station/index.html`)

- While a lane is `running` the page polls `/api/lanes` every 300 ms (one
  request for all lanes) and writes `count`, `rejected`, the phase word and the
  faults into the tile; the tap buttons stay, now labelled as corrections, and
  adjust the number that will be saved. A badge on the station bar shows
  *Camera: live 31 fps* or *Camera: offline — manual count*.
- *Stop* / *Finish & save* post `count` and `rejected` as today; the server
  records the engine's numbers alongside.
- The lane number on the tile is the engine lane name (`"1".."3"`), which is
  already what `startLane` sends.

## Finish camera (added 2026-10-19)

The same shape for the run: `testpoint-finishcam` (src/testpoint/finishcam.py)
is one process per finish camera. Per frame: YOLOX person detector
(`vision/detect.py`, the detector the pose presets already download) → IoU
tracker (`vision/tracker.py`) → virtual finish line (`vision/line.py`): a
crossing is the runner's feet changing side between two frames, and the instant
is interpolated between the two frame times. While a runner approaches, the
largest torso crops are kept; at the crossing they go through OCR (RapidOCR,
ONNX Runtime) and the digit strings are voted (`vision/bib.py`): a partial
reading supports the longer number it is part of, and a bib is only trusted when
one number has at least half of the weighted votes.

Each crossing is POSTed to `POST /api/engine/crossing`:

```json
{"bib": "21", "confidence": 0.81, "latency_s": 0.4, "t": 18.007,
 "readings": [{"text": "21", "conf": 0.93}], "clip": "run/000018007-t2.mp4", "track": 2}
```

The app stamps the crossing at *now − latency_s* and hands it to the existing
`record_finish`: a known running bib becomes the result, anything else
(no bib, unknown bib, not running, before the start) is a review-queue entry,
both with `source: "camera"`, the OCR share as `confidence` and the clip path.
The finish screen shows a 📷 badge and the clip link on review rows; the clip is
served by `GET /api/evidence/{path}`. The engine keeps counting and queues
crossings while the app is unreachable.

Tried on 2026-10-19 with two public clips (`data/clips/run/`, not in git):
a 640×360 sample (two runners, bibs 21 and 30 read correctly, the second one
after the partial-reading rule) and a 720p Mount Marathon finish-cam feed
(`--crop 0.21,0,1,1 --line 0,0.75,1,0.75`): four finishers in three minutes,
crossing intervals within about a second of the official results, bibs
unreadable at that distance (about 25 px wide; a 4 MP camera 8–10 m before the
line gives 80–100 px). Detector alone runs at ~30 fps at 720p on the RTX 5090.

Not done: waves and the start camera, 1 km / 2 km splits, course-integrity
chaining, lane-aware two-chute finish, OCR tuned on our own bibs.

## Out of scope (later plan items)

Override reason and review screen; retention job for evidence; per-station
camera configuration in the database (`Station` table) — this round takes the
camera and lanes from the engine's command line / `.env` as the compose
`engine` service does now.

## Running it on the dev machine

```bash
testpoint-app --port 8000 --db sqlite:///data/dev.db
testpoint-engine --source 0 --exercise pushup --lanes 1 --device cuda --app http://127.0.0.1:8000
```

`Start TestPoint.cmd` with `START_ENGINE=1` does the same through compose.

## Tests (no camera, no model)

- `tests/test_engine_link.py`: the app with `TestClient`: a report with events
  for a running attempt creates `Rep` rows and updates `auto_count`; a resent
  event is not duplicated; events for a cancelled/other attempt are dropped;
  `/api/lanes` reports `online` false after the age limit; the reply names the
  running attempt per lane.
- `tests/test_engine.py`: the engine's report builder and the reset-on-new-
  attempt rule driven by a fake session (scripted `MultiLaneResult`s) and a
  fake transport; the clip cutter on synthetic frames.

## Work plan

1. `Rep` model + `auto_*` columns, service functions, `/api/engine/report`,
   `/api/lanes`, tests.
2. `engine.py` report loop + reset rule, `testpoint-engine` entry point, CLI
   refactored to share the frame loop, tests.
3. Evidence writer (full video + per-rep clips).
4. Station UI live tile + camera badge; try on this machine with the webcam.
5. Docs: PLAN.md tick, README section, compose `engine` command updated.
