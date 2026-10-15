# Site, hardware and capacity design

Based on the answers of 2026-10-15.

## Hardware

| Item | Model | Notes |
|------|-------|-------|
| Cameras ×4 | Hikvision iDS-2CD7A46G0/P-LZHS | 4 MP (2688×1520), 1/1.8" sensor, DarkFighter low light, 140 dB WDR, motorised vari-focal, PoE, RTSP. Use main stream `rtsp://user:pass@<ip>:554/Streaming/Channels/101` at 1080p/25 fps for AI, record the same stream as evidence. Built-in ANPR analytics are not used. |
| Switch | TP-Link TL-SG1218MP | 16× PoE+ (30 W/port, 250 W total), 2× SFP. Cameras ≈ 15–25 W each → ~100 W used. Cat6 ≤ 100 m per run. |
| PC | Windows 11, RTX 5080 16 GB, Core Ultra 9 285K | Runs capture, AI, database, web UI. Add a UPS and a second disk for evidence video. |
| Examiner devices | Any tablet/laptop with a browser | Connect to the switch (Wi-Fi access point on the LAN). No internet needed. |

Offline site: the installer bundles all models (RTMPose, YOLOX detector,
InsightFace, OCR), Python wheels and fonts. Nothing is downloaded at runtime.

## Capacity: 500 candidates per event

### Push-up and crunch stations

A 2-minute test plus about 1 minute of check-in and change-over is 3 minutes
per candidate per lane. One 4 MP camera can watch **3–4 mats side by side**
(camera 6–7 m back, 0.8–1 m high, mats 1.5 m apart, side view). RTMPose is
multi-person, so each lane is a zone in the image with its own counter.

| Lanes per camera | Candidates/hour/camera | Time for 500 |
|------------------|------------------------|--------------|
| 1                | 20                     | 25 h         |
| 3                | 60                     | 8.3 h        |
| 4                | 80                     | 6.3 h        |

With camera 1 on push-ups and camera 2 on crunches, both at 4 lanes, 500
candidates finish both events in about 6.5 hours if they rotate between
stations. Two extra cameras would halve that. Lane zones are drawn once in
Advanced mode and saved.

### 3.2 km run, point-to-point, up to 500 runners

Two issues, both solved by site layout rather than by software:

**1. Finish-line density.** 500 runners starting together finish in bunches,
and bunches hide bibs. Camera-only timing is reliable when runners cross one
or two at a time. Design:

- **Waves of ~50** starting 2–3 minutes apart (10 waves ≈ 30 minutes of starts).
- **Finish chute:** barriers narrow the last 10 m to two lanes, so runners
  cross the line one behind the other, not side by side.
- **Camera 3** faces the finish line head-on, 1.5 m high, 8–10 m before the
  line, reading front bibs. **Camera 4** is behind the line facing back,
  reading back bibs. Two reads per runner.
- Bibs: front and back, digits ≥ 12 cm tall, black on white or yellow.
- Every crossing is recorded; unidentified crossings go to the examiner's
  review queue with the clip.
- Fallback: if the exit test (±0.5 s, every runner identified or queued) is
  not met with real footage, add a UHF RFID reader at the finish. The data
  model already stores crossings per source.

**2. Remote start line.** A PoE camera 3.2 km away cannot be cabled to the
switch (100 m limit). Options, in order of preference:

| Option | How | Cost / risk |
|--------|-----|-------------|
| A. Out-and-back course | Start and finish lines 20–50 m apart, same site, same switch. Cameras 3 and 4 serve both lines. | Cheapest, simplest. Needs the course to allow a turnaround. |
| B. Wireless bridge | Point-to-point Wi-Fi bridge (e.g. 5 GHz, line of sight, 3 km is routine) with a small PoE injector and battery/mains at the start. Start camera joins the LAN. | ~2 bridge units + power at the start. Needs line of sight. |
| C. Offline start | No camera at the start. An examiner tablet (clock synced to the PC in the morning) records each wave's start time and the runners checked into the wave; data merges over the LAN when the tablet returns. | No network needed. Less evidence at the start. |

In all cases the **start time is the wave start**, recorded when the examiner
presses "Start wave N" (with a horn). Runners are checked into a wave by face
or bib at the start gate, so the finish only needs the bib plus the wave.

## Camera placement summary

| Camera | Role | Position |
|--------|------|----------|
| 1 | Push-up station, 3–4 lanes | Side view, 6–7 m from mats, 0.8–1 m high |
| 2 | Crunch station, 3–4 lanes | Side view, 6–7 m from mats, 0.8–1 m high |
| 3 | Finish line, front bibs (+ start gate check-in in option A) | Head-on, 1.5 m high, 8–10 m before the line |
| 4 | Finish line, back bibs | Behind the line, facing back, 1.5 m high |

Lighting: outdoor daylight is fine for all; avoid cameras facing low sun. For
indoor stations, even light from the side, no strong backlight behind the mats.
