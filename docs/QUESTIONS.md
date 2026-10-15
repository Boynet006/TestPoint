# Questions and answers

Answers given by the project owner on 2026-10-15 are recorded here with the
decision taken. Open points are at the bottom.

## Answered

| # | Question | Answer | Decision / consequence |
|---|----------|--------|------------------------|
| 1 | Cameras | Hikvision iDS-2CD7A46G0/P-LZHS (4 MP, 2688×1520, DarkFighter, 140 dB WDR, PoE, RTSP, motorised vari-focal) | Use RTSP main stream at 1080p or 1440p, 25 fps. The ANPR features of this model are not used. See [SITE.md](SITE.md). |
| 2 | Switch | TP-Link TL-SG1218MP, 16× PoE+, 250 W budget | Ample for 4 cameras (≈ 15–25 W each incl. heater). Cat6 runs must stay under 100 m. |
| 3 | Internet at site | No | Everything bundled offline: models, pip wheels, fonts. No runtime downloads. Added to Phase 0 and 5. |
| 4 | Regulation | Unknown | We propose a rule set in [RULES.md](RULES.md) for sign-off; all numbers are configurable. |
| 5 | Push-up rule | Depth; pause allowed; time limit | Depth = elbow ≤ 90°; pause allowed only in the up position; time limit default 2:00 (to confirm). |
| 6 | Crunch rule | How high; time limit | Height = torso elevation ≥ 35° (crunch) — or ≥ 70° if a full sit-up is wanted; time limit default 2:00 (to confirm). |
| 7 | Run | Point-to-point; up to 500 runners | Big design impact: wave starts, finish-chute design, remote start line network. See [SITE.md](SITE.md) and Phase 4. |
| 8 | Scoring tables | Unknown | App ships with editable score tables; placeholder values until the official ones arrive. |
| 9 | Throughput | Up to 500 per test | Multi-lane stations (one camera watches 3–4 mats). See throughput maths in [SITE.md](SITE.md). |
| 10 | Who operates | Unknown | Simple mode designed for a non-technical examiner with 10 minutes of training. |
| 11 | Simple/Advanced contents | "Whatever you see fit" | Proposed split kept (see [ARCHITECTURE.md](ARCHITECTURE.md)); will refine with feedback on the prototype. |
| 12 | Examiner override | Yes | Overrides allowed, always with a reason, always logged. |
| 13 | Languages | English and Arabic | Full right-to-left UI, bilingual exports, from Phase 2. |
| 14 | Retention | Not sure | Best-practice default: results kept permanently; evidence video 30 days; face data 7 days after test day. Configurable. |
| 15 | Bibs | Unknown | We design camera-friendly bibs (large digits, front and back). Numbers assigned from the daily list. |
| 16 | Face use | For recognising the person at stations and during the run; delete after a few days | Face check-in at every station and at the run start gate; bib is the primary identifier at the finish, face is secondary. Deletion after 7 days. |
| 17 | Daily list source | Scanned from paper or typed | Typed entry form + CSV/Excel import + scanned-paper OCR with a mandatory review screen. |

| 18 | Run course layout | Start and finish far apart; Wi-Fi not an option; fiber possible; cameras at start, 1 km, 2 km, finish | Single-mode fiber daisy chain through field switches; four timing lines with split times and course-integrity checks. All four cameras go to the run, so the exercise stations need 2 more. See [SITE.md](SITE.md). |

| 20 | Exercise choice per station or per candidate | Per station | One camera = one exercise; candidates rotate between stations. The station bar sets the exercise; lanes inherit it. |
| 19 | Two more cameras for the exercise stations | Approved in principle, may be added later | Target: 6 cameras, 4 on the run, 1 per exercise station with 3–4 lanes each. Until then, the split cameras are reassigned to the stations on non-run days (Advanced → Cameras). |

| 21 | Test-day order | Weigh-in/BMI gate → run line-up → mass start → finish → push-ups and crunches only if the run is passed | Candidate journey and gates defined in [TEST-DAY.md](TEST-DAY.md); mass start replaces waves; weigh-in station added. |

## Still open

0. **Weigh-in and run rules:** BMI limits, how weight/height are measured
   (scale model, data output), run pass time limits by age/gender, DNF
   cut-off. See [TEST-DAY.md](TEST-DAY.md).
1. **Fiber details:** who installs it, route length per hop, single-mode
   confirmed, strands per hop (2 minimum, 4 for a ring), and whether a ring
   back from the start is possible.
2. **Power at the start, 1 km and 2 km points:** mains or battery.
3. **Field switch** model at each remote point — candidates are listed in
   [SITE.md](SITE.md); confirm with the installer.
4. ~~Wave size~~ — decided: single mass start. The finish-line load is the
   Phase 4 risk to prove (see TEST-DAY.md).
5. **Official time limits and depth rules** for push-ups and crunches, and
   whether the crunch is a crunch (shoulders off the floor) or a full sit-up.
6. **Scoring tables** by age and gender, when available.
7. **A sample of the paper list** used today (to design the OCR import).
8. **Reference videos:** 10+ people doing push-ups and crunches from the
   planned camera angle, with the human count for each (needed for Phase 1
   exit test).
9. **Photos of the site** (station area, each timing point) for camera
   placement and sun direction.
