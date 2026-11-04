# Continue on another machine

Everything needed is on GitHub, except what is private to the station PC: the database
(`data/dev.db`: people, faces, results, the cameras' passwords), the HTTPS certificates and the
recordings in `data/`. Those are not in git on purpose.

## Where things are (2026-11-04)

| Branch / tag | What |
|---|---|
| `main`, tag `v1.0` | Shipped: daily face photos, start line camera (hand to start, start line timing, false start), finish by face, one-mat station, results, desktop icon |
| `feat/percentages` (PR #81) | **Latest work, not merged:** correct a saved result (PR #80), exact percentages per test and overall, the overall as a ring on the results card |
| `phase2/out-of-position` | Phase two, kept apart: leaving the exercise position ends the station test |

Work continues on `feat/percentages` (it contains everything in `main` too).

## 1. Get the code

```bash
git clone https://github.com/Boynet006/TestPoint.git
cd TestPoint
git checkout feat/percentages
```

## 2. Set it up (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[app,dev]"
.venv\Scripts\python.exe -m testpoint.api.faces --download data\models\face
winget install --id Gyan.FFmpeg -e        # browser-playable evidence clips (H.264)
```

GPU: install `onnxruntime-gpu` as the README says; without it everything runs on the CPU, slower.
The fonts are in the app (`web/station/fonts`): no internet needed once set up.

## 3. Run it

Once: `scripts\https-cert.ps1` (README, "HTTPS"), then put the icon on the desktop:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make-shortcut.ps1
```

The **TestPoint** icon starts the app and the camera service (logs in `data\logs`) and opens
Chrome; each screen starts its own camera. While developing, run them by hand instead (the app
then reloads on code changes):

```powershell
$env:DISABLE_SQLALCHEMY_CEXT_RUNTIME = "1"
.venv\Scripts\python.exe -m testpoint.api.app --host 0.0.0.0 --port 8000 --db sqlite:///data/dev.db --reload --https
.venv\Scripts\python.exe -m testpoint.camhost        # second window: the cameras
```

`scripts\start-station.ps1` opens the address `192.168.250.149` when this PC has it, else
`localhost`: change it there for another PC's address.

## 4. What to redo on the new machine

- **Cameras:** Settings > Cameras: type each camera's address with its password again.
- **People:** the database starts empty; register candidates again (fresh face photos each day).
- **Lines:** the finish and start line positions are kept per browser: set them again on the
  finish and start line screens (Line set-up).

## 5. Tests

```powershell
$env:DISABLE_SQLALCHEMY_CEXT_RUNTIME = "1"; .venv\Scripts\python.exe -m pytest -q
```
