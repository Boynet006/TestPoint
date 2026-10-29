# TestPoint — camera-based fitness test station

TestPoint scores a standard three-event physical fitness test using cameras and
on-premise AI, with no wearables and no manual counting:

| Event            | What is measured                       | How                                           |
|------------------|----------------------------------------|-----------------------------------------------|
| 3.2 km run       | Time from start to finish per runner   | Start/finish-line cameras, bib OCR + tracking |
| Push-ups         | Valid reps in the time limit           | Pose estimation + rep state machine           |
| Crunches         | Valid reps in the time limit           | Pose estimation + rep state machine           |

Everything runs on one PC with an NVIDIA GPU. Cameras connect through a network
switch. Examiners use a web UI in a browser (tablet, laptop or phone on the same
network).

## Documents

- [docs/PLAN.md](docs/PLAN.md) — phased delivery plan (what we build, in what order, with exit criteria)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system design, components, data model, key decisions
- [docs/QUESTIONS.md](docs/QUESTIONS.md) — answers received so far and open points
- [docs/SITE.md](docs/SITE.md) — hardware, camera placement, capacity for 500 candidates, run-course options
- [docs/RULES.md](docs/RULES.md) — proposed rule set v1 for sign-off (thresholds, time limits, retention)
- [docs/DEMO.md](docs/DEMO.md) — running the demo on one PC with photos and videos instead of cameras
- [docs/TEST-DAY.md](docs/TEST-DAY.md) — the test-day flow: weigh-in gate, mass-start run, exercises only after a run pass
- [docs/CALIBRATION.md](docs/CALIBRATION.md) — push-up and crunch counters checked and recalibrated on public videos (2026-10-19)
- [docs/RESEARCH.md](docs/RESEARCH.md) — existing products, libraries and the reasoning behind the choices

## Repository layout

```
TestPoint/
  docs/                 plan, architecture, research, open questions
  src/testpoint/        Python package
    vision/             pose backends, geometry, rep counter, exercise rules
    cli.py              demo: count reps per lane on a webcam, video file or RTSP camera
    engine.py           station engine: counts reps and reports them live to the station app
    evidence.py         evidence video per attempt and per-rep clips
  scripts/bench_pose.py pose pipeline benchmark (cpu vs cuda, all rtmlib modes)
  tests/                unit tests (no camera or model needed)
  docker/               build helpers (model download for the offline images)
  web/prototype/        clickable UI mock-up
  src/testpoint/api/    station app: journey rules, database, HTTP API
  web/station/          station UI (weigh-in, run start, finish, exercise lanes)
  Start TestPoint.cmd   double-click to start (Docker Desktop + services + UI)
  Stop TestPoint.cmd    double-click to stop
  Dockerfile            images: gpu, cpu, test, prototype
  compose.yaml          services: prototype, engine (gpu), engine-cpu, tests
  .env.example          camera address and settings for compose
  pyproject.toml        package + dependencies
```

## Quick start (Phase 1 prototype)

Requires Python 3.11 or 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Count push-ups from the default webcam, with an on-screen overlay:

```bash
testpoint-demo --exercise pushup --source 0 --show
```

Count crunches from a recorded video and save an annotated copy:

```bash
testpoint-demo --exercise crunch --source sample.mp4 --out annotated.mp4
```

Count push-ups on a live Hikvision camera (main stream, channel 101) with three
candidates side by side, one lane per mat:

```bash
testpoint-demo --exercise pushup --source rtsp://user:pass@<ip>:554/Streaming/Channels/101 --lanes 3 --device cuda --show
```

RTSP streams are read with FFmpeg over TCP (set before OpenCV loads through
`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`), use the PC wall clock for
timing, and reconnect with exponential backoff when the stream drops.
`--connect-timeout S` (default 10) and `--connect-attempts N` (default 3,
`0` = retry for ever) control how long a dead camera is retried before the demo
exits with an error. The password is never printed.

Lanes:

- `--lanes N` splits the frame into N equal vertical strips, named `1`..`N`
  from left to right.
- `--lanes-file lanes.json` uses rectangles drawn by the administrator, in
  normalized `0..1` image coordinates (x right, y down), so the same file works
  at any stream resolution:

  ```json
  {"lanes": [
    {"name": "1", "rect": [0.00, 0.30, 0.33, 1.00]},
    {"name": "2", "rect": [0.33, 0.30, 0.66, 1.00]},
    {"name": "3", "rect": [0.66, 0.30, 1.00, 1.00]}
  ]}
  ```

Each detected person is assigned to the lane containing their hip midpoint;
people outside every lane (examiners, the next candidate waiting) are ignored,
and each lane keeps its own counter, phase and fault list. The overlay draws
every lane with its count; the final JSON summary is per lane.

Add `--device cuda` to run the pose model on the GPU (see *GPU setup* below).
The first run downloads the RTMPose ONNX models (about 140 MB; the `performance`
mode adds another 530 MB).

## GPU setup (Phase 0)

The station PC has an RTX 5080 (Blackwell). Blackwell needs an ONNX Runtime build
against CUDA 12.8 or newer; the pinned `onnxruntime-gpu` 1.31.0 wheel is built
against CUDA 13 and the `nvidia-*` pip wheels supply the CUDA 13.4 / cuDNN 9.27
runtime, so no system-wide CUDA toolkit is needed. Only a recent NVIDIA driver is
required (verified with driver 617.42, which reports CUDA 13.4).

`onnxruntime` (CPU) and `onnxruntime-gpu` install into the same package
directory, and rtmlib depends on the CPU wheel, so the CPU wheel has to be swapped
out after the install:

```bash
pip install -e ".[gpu,dev]"
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu==1.31.0
```

Check that the GPU is actually used (the `provider` column must say
`CUDAExecutionProvider`; `RTMLibBackend(device="cuda")` raises if ONNX Runtime
fell back to CPU):

```bash
python scripts/bench_pose.py --device cuda
```

Any later `pip install` that depends on `onnxruntime` (for example
`rapidocr-onnxruntime` for the finish camera's bib OCR, `pip install -e ".[run]"`)
pulls the CPU wheel back in and silently disables the GPU again: repeat the
uninstall / force-reinstall pair above after such an install.

The backend calls `onnxruntime.preload_dlls()` before creating a CUDA session.
Without that call ONNX Runtime cannot find the DLLs inside the `nvidia` pip
packages and silently runs on the CPU.

### Pose pipeline speed

`scripts/bench_pose.py`, 2026-10-15, RTX 5080, onnxruntime-gpu 1.31.0, median
over 30 frames (5 on CPU) after warm-up. One frame = YOLOX detector + one
RTMPose inference, i.e. the cost of one person in view; each extra person adds
one more pose inference (about 2 ms on the GPU).

| device | mode        | frame     | median ms | fps  |
|--------|-------------|-----------|----------:|-----:|
| cpu    | lightweight | 640x480   |     146.1 |  6.8 |
| cpu    | lightweight | 1920x1080 |      94.2 | 10.6 |
| cpu    | balanced    | 640x480   |     638.5 |  1.6 |
| cpu    | balanced    | 1920x1080 |     655.7 |  1.5 |
| cpu    | performance | 640x480   |    1283.2 |  0.8 |
| cpu    | performance | 1920x1080 |    1167.9 |  0.9 |
| cuda   | lightweight | 640x480   |      14.0 | 71.4 |
| cuda   | lightweight | 1920x1080 |      14.1 | 71.1 |
| cuda   | balanced    | 640x480   |      23.2 | 43.1 |
| cuda   | balanced    | 1920x1080 |      23.9 | 41.9 |
| cuda   | performance | 640x480   |      33.2 | 30.1 |
| cuda   | performance | 1920x1080 |      33.4 | 29.9 |

Frame size barely matters because both models resize their input; the GPU
numbers are dominated by the fixed per-model cost. `balanced` at 1080p leaves
headroom for two cameras at 20 fps on one GPU; `performance` is at the limit for
a single 30 fps stream.

## Station app (Phase 2)

The station app is a FastAPI server with a SQLite database and one web page
for every device on test day. Each device picks its role in the header:

| Role | What the examiner does |
|------|------------------------|
| Weigh-in | Check in by face (webcam, or a photo/video for demos) or by bib, enter weight and height, accept or reject on BMI |
| Run · start line | Mark runners present, then START (press twice) for the mass start |
| Run · finish line | Type each bib as the runner crosses; unread bibs go to a review queue |
| Exercise station | Push-ups or crunches, 3 lanes, check-in per lane, live count from the camera (or tap-count without it), save |

**Advanced** shows today's list with everyone's status, CSV import, face
enrolment and the rules. To demonstrate on one PC with photos and videos
instead of cameras, follow [docs/DEMO.md](docs/DEMO.md). The server enforces the order of the day: a station refuses anyone who
has not passed the step before it, with the reason on screen.

It starts with the one-click **Start TestPoint.cmd** and runs at
http://localhost:8080. Other devices on the LAN open
`http://<station PC address>:8080`. Data lives in `data/testpoint.db`.

To run it without Docker while developing:

```bash
.venv\Scripts\python.exe -m testpoint.api.app --port 8000 --db sqlite:///data/dev.db
```

### HTTPS: cameras on other devices

Browsers allow a camera only on `localhost` or HTTPS pages, so a registration desk or a tablet
that opens `http://<station PC>:8000` cannot use its webcam. Serve HTTPS instead:

1. On the station PC: `.\scripts\https-cert.ps1` makes TestPoint's own certificate authority
   and a certificate for this PC's addresses in `data\certs` (again after an IP change; the
   authority stays, so devices keep trusting it).
2. Start the app with `--https` (or `TESTPOINT_HTTPS=1`).
3. Allow TCP 8000 in Windows Firewall for the LAN (as administrator):
   `New-NetFirewallRule -DisplayName "TestPoint app 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress 192.168.250.0/24 -Action Allow`
4. On each device, once: open `https://<station PC>:8000/testpoint-ca.crt`, open the file,
   *Install Certificate* → *Local Machine* → *Trusted Root Certification Authorities*, then
   restart the browser. `https://<station PC>:8000` then opens with no warning and the camera works.

The camera processes on the station PC are always given the app's loopback address
(`https://127.0.0.1:8000`), whichever device starts them, and skip the certificate check there only.

A sample list to import is in `samples/candidates.csv` (columns
`external_id,name,name_ar,gender,dob,bib`). The API is documented at
http://localhost:8080/docs.

### Live rep counting at the station

The camera engine is a second process, one per camera, that counts reps and
reports to the app every 250 ms; the app answers with the attempt each lane is
counting for, so pressing *Start* on the tablet resets that lane's counter.
The lane tile then shows the count, the phase (*get into position / up / down*)
and the live faults, the examiner's tap buttons become corrections on top of
the camera's number, and the saved result keeps both (`count` and
`auto_count`). Every rep is stored with its times and faults, a recording of the
attempt and a clip per rep go under `data/evidence/<day>/<attempt>/`. Design
and protocol: [docs/ENGINE-LINK.md](docs/ENGINE-LINK.md).

On the development PC (app from the command above, webcam 0, GPU):

```bash
.venv\Scripts\python.exe -m testpoint.engine --source 0 --exercise pushup --device cuda --app http://127.0.0.1:8000
```

### Live camera on the station screens

Both engines also send their annotated picture (skeletons, lane boxes, counts;
runner boxes, finish line, bib reads) to the app about three times a second
(`POST /api/engine/frame/<station>`), and the exercise and finish screens show it
above the lane tiles / finishers list, so the examiner sees what the camera
sees. It disappears on its own when the engine stops; `--no-preview` turns it off.
Screenshots of every screen are in [docs/screenshots/](docs/screenshots/).

### Exercise-station camera

On the exercise station page, *Camera* opens the station camera card: pick a webcam on this PC,
a video (`data/clips/pushup|crunch`, or the Desktop *TestPoint test videos*) or an IP camera, then
*Start camera*. The camera service runs the counting engine (`testpoint.engine`) for the
exercise and the number of lanes on the page, and follows them when they change (while no lane
is testing). Drag the dashed lane dividers on the live picture so each lane holds one person,
then *Restart with these settings*. The station plays a beep at the start, two at 10 s left and
a long tone at time; *Start all* starts every checked-in lane together after 3-2-1. A lane shows
the person's enrolment photo for the examiner to check before the start.

### Finish-line camera

**From the finish page (normal use).** *Start TestPoint.cmd* also starts the camera
service on the station PC (`python -m testpoint.camhost`, listening on 127.0.0.1:8099
only). On the finish page, *Finish camera* picks the source (a webcam on this PC, a
video file from `data/clips/run` or the Desktop *TestPoint test videos* folder, or an IP
camera's `rtsp://` address), the finish-line position on the live picture and the
running direction, then *Start camera* / *Stop camera*. The service runs the finish
camera on the GPU and writes its output to `data/logs/finishcam.log`; the page shows
its state and last lines. A webcam can be used by one program at a time.

Runners are identified by **bib**. When no bib is read (or only with low confidence), the
camera also sends a snapshot of the runner's head; the app compares it only with the runners
still out on the course and records the finish when the match is confident (the strict face
setting), otherwise it goes to the review queue with the likely runner suggested. The snapshot
is not stored.

**By hand (development):**

`testpoint-finishcam` times runners crossing a virtual line and reads their
bib, then posts each crossing to the app (`POST /api/engine/crossing`): a known
running bib is a result, anything else lands in the finish screen's review
queue with a 📷 badge and a clip. Bib OCR needs `pip install -e ".[run]"`
(RapidOCR; then redo the GPU swap in *GPU setup*). On a clip:

```bash
.venv\Scripts\python.exe -m testpoint.finishcam --source data/clips/run/clip.mp4 --line 0,0.85,1,0.85 --device cuda --app http://127.0.0.1:8000 --out annotated.mp4
```

`--line x0,y0,x1,y1` is the finish line in 0..1 image coordinates, `--direction`
which way counts (default `down`: runners come towards the camera), `--crop`
the camera part of a feed with side panels, `--no-app` for clip analysis only.

With Docker, `Start TestPoint.cmd -Engine` (or `START_ENGINE=1` in `.env`)
starts the engine on `CAMERA_URL` reporting to the app; the station bar shows
*Camera live · N fps*, or *Camera offline · manual count* when no engine
reports for 2 s. A USB webcam is not reachable from a container on Windows, so
for webcam tests run the engine natively as above.

## Docker

Everything also ships as Docker images, which is how the station PC will be
installed: one tested image per role, no Python setup on site, and an offline
transfer with `docker save` / `docker load`.

| Image | Dockerfile target | What it is |
|-------|-------------------|------------|
| `testpoint/app` | `app` | Station app: API, database, station UI and face check-in models (653 MB) |
| `testpoint/engine-gpu` | `gpu` (default) | Rep-counting engine on the NVIDIA GPU, models baked in |
| `testpoint/engine-cpu` | `cpu` | Same engine on the CPU (laptops, CI) |
| `testpoint/tests` | `test` | Runs the unit tests |
| `testpoint/prototype` | `prototype` | The original UI mock-up on nginx (port 8081, `--profile prototype`) |

Measured sizes: engine-gpu 5.5 GB, engine-cpu 1.5 GB, tests 1.2 GB,
prototype 74 MB. The GPU image carries the CUDA 13 and cuDNN runtime; only the
NVIDIA driver comes from the host. Models for `lightweight` and `balanced`
are baked in; `performance` is downloaded on first use unless you build with
`MODEL_MODES=lightweight balanced performance` in `.env`, which the offline
site needs if it will use that mode.

### Install Docker on the Windows PC (once)

Docker on Windows runs Linux containers inside WSL2. Both steps need an
administrator PowerShell and a reboot after the first one.

```bash
wsl --install
```

```bash
winget install -e --id Docker.DockerDesktop
```

GPU access works out of the box with Docker Desktop and the installed NVIDIA
driver. Docker Desktop is free only for personal use, education and small
businesses; larger organisations need a paid subscription. The free
alternative is Docker Engine installed inside the WSL2 Ubuntu distribution
plus the NVIDIA Container Toolkit.

### One-click start and stop

Double-click **Start TestPoint.cmd** in the project folder. It starts Docker
Desktop if needed, waits for it, starts the station services from the images
already on the PC (no internet needed) and opens the station UI. On a PC that
has never built or loaded the images (copying the folder does not copy them),
the first start builds them, which needs internet once: about 3 minutes for the
app, much longer for the GPU engine (`-Engine`). Offline PCs get the images with
`docker load` instead (see *Move to the offline site*). **Stop
TestPoint.cmd** stops the services. Both call `scripts/testpoint.ps1`, which
also works from a PowerShell prompt:

```bash
powershell -ExecutionPolicy Bypass -File scripts\testpoint.ps1 status
```

| Action | What it does |
|--------|--------------|
| `start` | Docker Desktop up, services up, browser opens. Add `-Engine` (or `START_ENGINE=1` in `.env`) to also start the rep-counting engine on `CAMERA_URL`; `-NoBrowser` skips the browser. |
| `stop` | Stops and removes the TestPoint containers. Docker Desktop keeps running. |
| `status` | Docker engine, running services and the UI address. |
| `logs` | Follows the services' logs (Ctrl+C to leave). |

For a station PC that should come up on its own after a power cut, enable
Docker Desktop → Settings → General → *Start Docker Desktop when you sign in*.
The app service has `restart: unless-stopped`, so it comes back by
itself once Docker is up; no script needs to run.

### Build and run

Copy the settings file and put the camera address in it:

```bash
copy .env.example .env
```

Build all images. Profiled services are only built when their profile is named:

```bash
docker compose --profile gpu --profile cpu --profile test build
```

Run the unit tests:

```bash
docker compose --profile test run --rm tests
```

Check that the container really uses the GPU (the provider column must say
`CUDAExecutionProvider`):

```bash
docker compose --profile gpu run --rm --entrypoint python engine scripts/bench_pose.py --device cuda --mode balanced
```

Count reps from the camera in `CAMERA_URL`; the annotated video is written to
`data/` on the host:

```bash
docker compose --profile gpu up engine
```

Count reps in a recorded clip placed in `data/`:

```bash
docker compose --profile gpu run --rm engine --exercise pushup --source /data/clip.mp4 --lanes 3 --device cuda --out /data/clip-annotated.mp4
```

Start the station app on http://localhost:8080:

```bash
docker compose up -d app
```

### Move to the offline site

On a machine with internet, after building:

```bash
docker save testpoint/app:0.1.0 testpoint/engine-gpu:0.1.0 -o testpoint-images.tar
```

On the station PC (Docker installed from an offline installer), copy the tar
file, `compose.yaml` and `.env`, then:

```bash
docker load -i testpoint-images.tar
```

```bash
docker compose up -d --no-build app
```

### Limits inside a container

- USB webcams (`--source 0`) and the preview window (`--show`) do not work in
  Docker on Windows. Use the RTSP cameras or video files, and `--out` to watch
  the result.
- The UI prototype loads its fonts from Google Fonts. Offline it falls back to
  system fonts; the Phase 2 UI bundles its fonts.

## Status (2026-10-19)

- Live rep counting (2026-10-19): `testpoint-engine` reports lane counts, phase,
  faults and rep events to the app (`POST /api/engine/report`, `GET /api/lanes`),
  the station tile shows them live with examiner corrections, `Rep` rows and
  `auto_count` are stored, evidence video and per-rep clips are written. Verified
  on this PC with a webcam and the RTX 5090 (balanced, ~31 fps end to end).
  103 tests passing.

- Unit tests: 61 passing (`pytest`), including a full simulated test day through the API.
- Station app (2026-10-19): weigh-in, run start, finish and exercise-lane screens
  wired to the API and database, in Docker on port 8080. Check-in is by bib or ID
  and exercise counts are tapped by the examiner until face check-in and the
  camera engine are connected.
- Face check-in (2026-10-19): enrol photos, identify from a webcam, video file or
  photo at the weigh-in screen (OpenCV YuNet + SFace, offline, models baked into the
  app image). Demo guide in docs/DEMO.md. 68 tests passing.
- Engine: 3-4 lanes per camera (`vision/lanes.py`, `MultiLaneSession`), push-up
  faults `body_not_straight`, `knee_on_ground`, `rest_in_down` (thresholds in
  `PushupRules`, to be calibrated on real footage), RTSP input with reconnect.
- GPU inference works: `onnxruntime-gpu` 1.31.0 (CUDA 13) on the RTX 5080,
  `balanced` mode at ~42 fps for 1080p frames (CPU was ~1.5 fps). See *GPU setup*.
- `web/prototype/index.html` is a clickable UI mock-up (no backend) of the
  Simple/Advanced modes and the check-in → test → result flow.
- Docker (2026-10-19): all four images build (first build 8 min, mostly the
  CUDA download; later builds reuse the cache). In the containers: 42 tests
  pass, the GPU engine runs on `CUDAExecutionProvider` at ~46 fps (`balanced`,
  1080p, same as native Windows), the prototype serves on port 8080, and an
  unreachable camera fails cleanly with the password masked. Not yet tried
  against a real camera.

