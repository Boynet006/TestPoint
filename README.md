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
- [docs/TEST-DAY.md](docs/TEST-DAY.md) — the test-day flow: weigh-in gate, mass-start run, exercises only after a run pass
- [docs/RESEARCH.md](docs/RESEARCH.md) — existing products, libraries and the reasoning behind the choices

## Repository layout

```
TestPoint/
  docs/                 plan, architecture, research, open questions
  src/testpoint/        Python package
    vision/             pose backends, geometry, rep counter, exercise rules
    cli.py              demo: count reps per lane on a webcam, video file or RTSP camera
  scripts/bench_pose.py pose pipeline benchmark (cpu vs cuda, all rtmlib modes)
  tests/                unit tests (no camera or model needed)
  docker/               build helpers (model download for the offline images)
  web/prototype/        clickable UI mock-up
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

## Docker

Everything also ships as Docker images, which is how the station PC will be
installed: one tested image per role, no Python setup on site, and an offline
transfer with `docker save` / `docker load`.

| Image | Dockerfile target | What it is |
|-------|-------------------|------------|
| `testpoint/engine-gpu` | `gpu` (default) | Rep-counting engine on the NVIDIA GPU, models baked in |
| `testpoint/engine-cpu` | `cpu` | Same engine on the CPU (laptops, CI) |
| `testpoint/tests` | `test` | Runs the unit tests |
| `testpoint/prototype` | `prototype` | The UI mock-up on nginx |

The GPU image is several GB because it carries the CUDA 13 and cuDNN runtime.
Only the NVIDIA driver comes from the host.

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

Serve the UI prototype on http://localhost:8080:

```bash
docker compose up -d prototype
```

### Move to the offline site

On a machine with internet, after building:

```bash
docker save testpoint/engine-gpu:0.1.0 testpoint/prototype:0.1.0 -o testpoint-images.tar
```

On the station PC (Docker installed from an offline installer), copy the tar
file, `compose.yaml` and `.env`, then:

```bash
docker load -i testpoint-images.tar
```

```bash
docker compose up -d --no-build prototype
```

### Limits inside a container

- USB webcams (`--source 0`) and the preview window (`--show`) do not work in
  Docker on Windows. Use the RTSP cameras or video files, and `--out` to watch
  the result.
- The UI prototype loads its fonts from Google Fonts. Offline it falls back to
  system fonts; the Phase 2 UI bundles its fonts.

## Status (2026-10-15)

- Unit tests: 42 passing (`pytest`).
- Engine: 3-4 lanes per camera (`vision/lanes.py`, `MultiLaneSession`), push-up
  faults `body_not_straight`, `knee_on_ground`, `rest_in_down` (thresholds in
  `PushupRules`, to be calibrated on real footage), RTSP input with reconnect.
- GPU inference works: `onnxruntime-gpu` 1.31.0 (CUDA 13) on the RTX 5080,
  `balanced` mode at ~42 fps for 1080p frames (CPU was ~1.5 fps). See *GPU setup*.
- `web/prototype/index.html` is a clickable UI mock-up (no backend) of the
  Simple/Advanced modes and the check-in → test → result flow.
- Docker (2026-10-19): Dockerfile, `compose.yaml` and `.env.example` written;
  the Linux dependency set resolves and the model-fetch script runs, but the
  images have not been built yet because Docker is not installed on this PC.

