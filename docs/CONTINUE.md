# Continue on another machine

Everything needed is on GitHub in the `feat/camera-settings` branch, except what is private to
the station PC: the database (`data/dev.db`: people, faces, results, the cameras' passwords), the
HTTPS certificates and the recordings in `data/`. Those are not in git on purpose.

## 1. Get the code

```bash
git clone https://github.com/Boynet006/TestPoint.git
cd TestPoint
git checkout feat/camera-settings
```

## 2. Set it up (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[app,dev]"
.venv\Scripts\python.exe -m testpoint.api.faces --download data\models\face
winget install --id Gyan.FFmpeg -e        # browser-playable evidence clips (H.264)
```

GPU: install `onnxruntime-gpu` as the README says; without it everything runs on the CPU, slower.

## 3. Run it

```powershell
$env:DISABLE_SQLALCHEMY_CEXT_RUNTIME = "1"
.venv\Scripts\python.exe -m testpoint.api.app --host 127.0.0.1 --port 8000 --db sqlite:///data/dev.db --reload
.venv\Scripts\python.exe -m testpoint.camhost        # second window: the cameras
```

Open http://127.0.0.1:8000. For tablets and other PCs use `--host 0.0.0.0 --https` after
`scripts\https-cert.ps1` (README, "HTTPS").

## 4. What to redo on the new machine

- **Cameras:** Settings > Cameras: type each camera's address with its password again.
- **People:** the database starts empty; register candidates again.
- **Try the counting without a camera:** Exercise station > Camera > Video file. The videos in
  `test-videos/` (true counts in its README) are listed there.
- **Check the counting:** `.venv\Scripts\python.exe -m pytest -q`. `tests/test_counting_recordings.py`
  replays the two real station recordings and must count 10 push-ups and 10 crunches.

## Where things stand (2026-11-02)

- Station: one screen, manual start, Start now, Redo a test, video upload, big timer.
- Counting: 10/10 on one person's recordings. Push-ups count by the shoulders' drop as well as the
  elbow; crunches by the torso's rise. Auto-calibration learns "back at the start" from each
  person's first 3 reps (they count too); the depth that counts is the same for everyone and is
  set with the sliders (Tune) until the official rules are known.
- Speed: the person detector runs every 5th frame: about 26 fps with three cameras.
- Not checked yet: that shallow push-ups and half crunches are rejected; 5-8 more people.
- Management deck: `docs/presentation/TestPoint-management-2026-11-02.pptx`.
