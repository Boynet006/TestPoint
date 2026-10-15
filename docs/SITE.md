# Site, hardware and capacity design

Based on the answers of 2026-10-15 (updated the same day for the fiber run
course).

## Hardware

| Item | Model | Notes |
|------|-------|-------|
| Cameras ×4 | Hikvision iDS-2CD7A46G0/P-LZHS | 4 MP (2688×1520), 1/1.8" sensor, DarkFighter low light, 140 dB WDR, motorised vari-focal, PoE, RTSP. Use main stream `rtsp://user:pass@<ip>:554/Streaming/Channels/101` at 1080p/25 fps for AI, record the same stream as evidence. Built-in ANPR analytics are not used. |
| Switch | TP-Link TL-SG1218MP | 16× PoE+ (30 W/port, 250 W total), 2× combo SFP slots for fiber uplinks. Cameras ≈ 15–25 W each. Cat6 ≤ 100 m per run. |
| PC | Windows 11, RTX 5080 16 GB, Core Ultra 9 285K | Runs capture, AI, database, web UI. Add a UPS and a second disk for evidence video. |
| Examiner devices | Any tablet/laptop with a browser | Wired to the LAN (no Wi-Fi on site). No internet needed. |

Offline site: the installer bundles all models (RTMPose, YOLOX detector,
InsightFace, OCR), Python wheels and fonts. Nothing is downloaded at runtime.

## Camera allocation: the run takes all four

The owner's decision is one camera at each of **start, 1 km, 2 km and
finish**. That uses all four cameras on the run, so the push-up and crunch
stations have none.

| Option | Consequence |
|--------|-------------|
| **A. Buy 2 more cameras (recommended)** | Same model, one per exercise station, cabled to the main switch. Run and exercises can take place the same day. |
| B. Run on a different day | The four run cameras are mounted along the course and are not practical to move; this only works if the exercise stations use 2 of them on days without a run, which means re-mounting and re-calibrating each time. |
| C. One camera per exercise station from the run set | Drop the 1 km and 2 km split cameras. Loses split times and course checks. |

The rest of this document assumes option A (6 cameras total). The main
switch has capacity for it.

## 3.2 km run: fiber network and timing lines

### Network topology

```
 START (3.2 km)      2 km point         1 km point            FINISH / PC
 ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌──────────────────┐
 │ cam 1      │     │ cam 2      │     │ cam 3      │     │ cam 4 (finish)   │
 │ field sw.  │─SMF─│ field sw.  │─SMF─│ field sw.  │─SMF─│ TL-SG1218MP (SFP)│
 │ examiner   │     │ (unmanned) │     │ (unmanned) │     │ PC, cams 5+6     │
 │ laptop     │     │            │     │            │     │ exercise stations│
 └────────────┘     └────────────┘     └────────────┘     └──────────────────┘
```

- **Fiber:** single-mode (OS2) is the safe choice for 1 km hops; multimode
  OM3/OM4 is limited to ~300–550 m at 1 Gbit. Use 1000BASE-LX SFP modules on
  both ends of each hop. Two main-switch SFP slots are enough for a daisy
  chain (one slot used) or a ring (both slots, gives redundancy).
- **Field switch at each remote point:** a small industrial/outdoor PoE
  switch with **two SFP ports** (in and out of the chain) and at least two
  PoE+ RJ45 ports (camera + examiner laptop at the start). It needs mains
  power or a battery box (a camera + switch is ~30–40 W; a 12 V 100 Ah
  battery with an inverter or a DC-input switch runs it all day). Place it in
  an IP66 enclosure.
- **Latency** over fiber is negligible; what matters is that all timestamps
  come from the PC clock when frames arrive, and that the cameras are synced
  to the PC by NTP (the PC runs a local NTP server on the LAN). Timing target
  is ±0.5 s; this design gives tens of ms.
- **Daisy-chain risk:** a cut between 2 km and 1 km takes out the start and
  2 km cameras. Each field switch should keep recording to the camera's SD
  card so footage can be recovered; a ring (second fiber back from the start
  to the main switch's other SFP slot) removes the risk entirely if the cable
  route allows it.

### Timing lines and what each camera does

| Point | Camera role | Software |
|-------|-------------|----------|
| Start | Start-gate check-in (face or bib) assigns each runner to a wave; the examiner's wired laptop presses "Start wave N"; the camera records the start as evidence and reads bibs crossing the start line | wave check-in, start crossing |
| 1 km | Split line: crossing time + bib read | split 1, course-integrity check |
| 2 km | Split line: crossing time + bib read | split 2, course-integrity check |
| Finish | Finish line: crossing time + bib read; final time = crossing − wave start | finish, review queue |

The split cameras add three things: split times per runner, a check that
every runner passed every point in order (no shortcut), and two extra chances
to read a bib that is hidden at the finish. A runner's finish is published
only when the chain start → 1 km → 2 km → finish is consistent, or the
examiner has reviewed it.

### Finish-line density

500 runners in one mass start finish in bunches, and bunches hide bibs.
Design for the camera to see runners one or two at a time:

- **Waves of ~50** starting 2–3 minutes apart (10 waves ≈ 30 minutes of
  starts). With a start camera, wave check-in is automatic.
- **Finish chute:** barriers narrow the last 10 m to two lanes so runners
  cross in file, not side by side.
- **Camera placement at each line:** head-on to the approaching runners,
  1.5 m high, 8–10 m before the line, lens zoomed so a bib fills ≥ 60 px
  in height. The line itself is a virtual line in the image, drawn once in
  Advanced mode and marked on the ground with tape for the examiner.
- **Bibs:** front and back, digits ≥ 12 cm tall, black on white or yellow.
- **Fallback:** if the Phase 4 exit test (±0.5 s, every runner identified or
  queued) fails on real footage, add a UHF RFID reader at the finish as a
  second crossing source. The data model stores crossings per source.

## Capacity: 500 candidates per exercise

A 2-minute test plus about 1 minute of check-in and change-over is 3 minutes
per candidate per lane. One 4 MP camera can watch **3–4 mats side by side**
(camera 6–7 m back, 0.8–1 m high, mats 1.5 m apart, side view). RTMPose is
multi-person, so each lane is a zone in the image with its own counter.

| Lanes per camera | Candidates/hour/camera | Time for 500 |
|------------------|------------------------|--------------|
| 1                | 20                     | 25 h         |
| 3                | 60                     | 8.3 h        |
| 4                | 80                     | 6.3 h        |

With one camera on push-ups and one on crunches, both at 4 lanes, 500
candidates finish both events in about 6.5 hours if they rotate between
stations. Lane zones are drawn once in Advanced mode and saved.

## Camera placement summary (6 cameras)

| Camera | Role | Position |
|--------|------|----------|
| 1 | Run start line + wave check-in | Head-on to the start gate, 1.5 m high |
| 2 | Run 2 km split | Head-on, 8–10 m before the line, 1.5 m high |
| 3 | Run 1 km split | Head-on, 8–10 m before the line, 1.5 m high |
| 4 | Run finish line | Head-on, 8–10 m before the line, 1.5 m high, two-lane chute |
| 5 (new) | Push-up station, 3–4 lanes | Side view, 6–7 m from mats, 0.8–1 m high |
| 6 (new) | Crunch station, 3–4 lanes | Side view, 6–7 m from mats, 0.8–1 m high |

Lighting: outdoor daylight is fine; avoid cameras facing low sun (orient the
course lines so the camera looks away from sunrise/sunset). For indoor
stations, even light from the side, no strong backlight behind the mats.

## Suggested network parts (to verify with the installer)

| Part | Candidate | Notes |
|------|-----------|-------|
| Field switch (×3: start, 2 km, 1 km) | D-Link DIS-100G-06P (4× PoE+ 120 W, 2× SFP, 48–55 V DC, −40…75 °C) or StarTech IES1G52UP12V (4× PoE, 2× SFP, 12/24/48/56 V DC input, useful for battery sites) | Both are IP30/IP40 housings, so each goes inside an IP66 outdoor cabinet with the power supply or battery. DIN-rail mount. |
| SFP modules (2 per hop, 6 total + 1 for the main switch) | TP-Link TL-SM311LS or an AddOn/Axiom "TL-SM311LS-compatible" 1000BASE-LX, 1310 nm, single-mode, LC, 10 km | Use the same type at both ends of a hop. The TL-SG1218MP's two SFP slots are combo ports (each shares with an RJ45 port) and the switch is listed as end-of-life on some TP-Link sites, so confirm module compatibility with the seller before buying. |
| Fiber | OS2 single-mode, outdoor-rated (direct burial or aerial), LC pigtails in a small splice box at each point | 2 strands per hop minimum; 4 if a ring back to the main switch is wanted. |
| Power at remote points | Mains if available; otherwise a 12 V 100 Ah battery box with a DC-input switch (StarTech) — camera + switch ≈ 30–40 W → 20+ hours | Camera heater raises draw in cold weather. |

## Still to confirm for the site

- Fiber: who installs it, total route length, single-mode confirmed, number
  of strands (2 per hop minimum, 4 for a ring).
- Power at the 1 km, 2 km and start points: mains or battery.
- Field switch model (needs 2× SFP + PoE+ ports, outdoor rated).
- Approval to buy 2 more cameras for the exercise stations.
