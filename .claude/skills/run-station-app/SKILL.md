---
name: run-station-app
description: Launch the TestPoint station app (FastAPI + web UI) locally on Windows and open it in a browser, including the venv setup and the Application Control workaround for SQLAlchemy.
---

# Run the TestPoint station app (local dev, Windows)

Starts `testpoint.api.app` on http://127.0.0.1:8000 with auto-reload and a dev
SQLite database at `data/dev.db`. The station UI is served at `/ui/`.

## 1. One-time setup

The README quick start installs only `.[dev]`, which is enough for the tests but
not for the web server. The server also needs the `app` extra (FastAPI etc.).

```powershell
python -m venv .venv          # Python 3.11 or 3.12
.venv\Scripts\python.exe -m pip install -e ".[app,dev]"
New-Item -ItemType Directory -Force data | Out-Null
# face check-in models (YuNet + SFace, ~39 MB; the Docker image bakes these in)
.venv\Scripts\python.exe -m testpoint.api.faces --download data\models\face
```

Without the face models the app still starts, but logs "face check-in disabled" and
every face lookup answers `face_unavailable` ("Face recognition is not available on
this server"). The app checks for them once, so restart it after downloading.

## 2. Start the server

On machines with Windows Application Control / Smart App Control enabled,
importing SQLAlchemy fails with:

```
ImportError: DLL load failed while importing _immutabledict_cy: An Application Control policy has blocked this file.
```

Do not change system security settings. Set `DISABLE_SQLALCHEMY_CEXT_RUNTIME=1`
so SQLAlchemy uses its pure-Python code paths (same behaviour, slightly slower).
It is harmless on machines without the policy, so always set it.

Run from the repo root, detached, with logs in `data/`:

```powershell
$env:DISABLE_SQLALCHEMY_CEXT_RUNTIME = "1"
$p = Start-Process -FilePath .venv\Scripts\python.exe `
  -ArgumentList "-m","testpoint.api.app","--host","127.0.0.1","--port","8000","--db","sqlite:///data/dev.db","--reload" `
  -RedirectStandardOutput data\srv.out -RedirectStandardError data\srv.err -PassThru -WindowStyle Hidden
```

Wait until it answers (uvicorn logs go to `data\srv.err`):

```powershell
for ($i=0; $i -lt 30; $i++) {
  try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/ -TimeoutSec 2).StatusCode; break }
  catch { if ($p.HasExited) { Get-Content data\srv.err -Tail 20; break }; Start-Sleep 1 }
}
```

Expect `200` and `Application startup complete.` in `data\srv.err`.

### Other devices (registration desk, tablets)

Use `--host 0.0.0.0 --https` instead (see README, "HTTPS: cameras on other devices"):
run `scripts\https-cert.ps1` once first. The URL becomes `https://<PC address>:8000`;
`http://` on a LAN address leaves the browser's camera blocked. Opening port 8000 in
Windows Firewall and trusting `testpoint-ca.crt` on each device are the user's steps
(security settings), not yours. Also start the camera service: `python -m testpoint.camhost`.

## 3. Drive it

Open http://127.0.0.1:8000 in the browser pane; it redirects to `/ui/`
("TestPoint Station"). A healthy start shows the header with the stage selector
(WEIGH-IN), a green "ONLINE" badge, today's date and clock, and the camera
source picker (Webcam / Video file / Photo). API docs: http://127.0.0.1:8000/docs.

## Stop

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object CommandLine -like "*testpoint.api.app*--port*8000*" |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Notes

- `.claude/launch.json` has the same command as `app-dev`, plus `app-qa` (8002)
  and `deck-app` (8001) variants on separate databases. Those entries don't set
  the SQLAlchemy env var, so on a machine with the policy start them as above.
- Production-style run is Docker: `Start TestPoint.cmd` (app on :8080, Linux
  image, so the Windows policy doesn't apply). See README.
