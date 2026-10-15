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
- [docs/QUESTIONS.md](docs/QUESTIONS.md) — what we need from the project owner before each phase
- [docs/RESEARCH.md](docs/RESEARCH.md) — existing products, libraries and the reasoning behind the choices

## Repository layout

```
TestPoint/
  docs/                 plan, architecture, research, open questions
  src/testpoint/        Python package
    vision/             pose backends, geometry, rep counter, exercise rules
    cli.py              demo: count reps on a webcam or video file
  tests/                unit tests (no camera or model needed)
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

Add `--device cuda` to run the pose model on the GPU once `onnxruntime-gpu` is installed.
The first run downloads the RTMPose ONNX models (about 140 MB).

## Status (2026-10-15)

- Unit tests: 20 passing (`pytest`).
- Pipeline smoke test on CPU: models load in ~20 s, ~550 ms per frame at 640x480.
  That is fine for offline clips but not for live use; the RTX 5080 needs
  `onnxruntime-gpu` with a CUDA 12.8+ build (Blackwell GPUs are not supported by
  older CUDA 12 wheels). Setting this up is a Phase 0 task.
- `web/prototype/index.html` is a clickable UI mock-up (no backend) of the
  Simple/Advanced modes and the check-in → test → result flow.

