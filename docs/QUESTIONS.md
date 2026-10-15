# What we need from you

Answer what you can now; the rest is needed before the phase shown.

## Needed now (Phase 0)

### Hardware
1. **Cameras:** make and model. Are they IP cameras (RTSP over the switch) or
   USB webcams? Resolution and frame rate? Do they support PoE?
2. **Switch:** is it PoE? Managed or unmanaged? Gigabit?
3. **PC:** we found an RTX 5080 (16 GB), Core Ultra 9 285K, Windows 11.
   Is this the PC that will run at the test site, and can we install
   software on it (Python, Git, Node.js)?
4. Is there internet at the site, or must everything work fully offline?

### Test rules
5. Which regulation defines the test (e.g. military, police, school)? Please
   share the official rule text or scoring tables.
6. **Push-up rule:** required depth (elbow at 90°? chest to fist?), body
   straight, hands position, is pausing allowed, time limit (1 min? 2 min?).
7. **Crunch rule:** feet held or free, knees bent at what angle, where the
   hands go, how high the torso must rise, time limit.
8. **Run:** 3.2 km exactly? Is the course a loop (laps past one camera) or
   point-to-point (start camera + finish camera)? How many runners start
   together? Maximum group size?
9. Scoring tables by age and gender, and pass/fail thresholds.

### Operations
10. How many candidates per day, and how many stations run at the same time?
11. Who operates the system on test day (examiner skill level)? This shapes
    the **Simple** mode.
12. Who configures it (administrator)? This shapes the **Advanced** mode.
    Please confirm what each mode should contain. Proposed split:
    - *Simple:* pick candidate, start/stop, live count, fault messages,
      result, next candidate.
    - *Advanced:* camera assignment and preview, calibration, rule
      thresholds, per-rep review with video, overrides and audit log,
      exports, diagnostics, user management.
13. Must a human examiner be able to override an AI count? (We strongly
    recommend yes.)
14. Which language(s) must the UI support?

### Data and privacy
15. How long must videos and results be kept? Who may view them?
16. Do candidates have bibs or ID numbers already? Can we design the bibs
    (large digits, front and back) for the camera?
17. Face check-in is now part of the flow. Where do the enrollment photos
    come from (ID card photo, HR system, a photo taken at the first
    check-in)? Is storing face data acceptable under your privacy rules?
17b. Who prepares the daily list and in what format (Excel/CSV columns)?
    Can a candidate be added on the day, and by whom?

## Needed before Phase 1

18. Reference videos: at least 10 different people doing push-ups and
    crunches, recorded from the camera position you intend to use (side
    view, about 3–4 m away, camera at roughly 0.8 m height). Include bad
    reps on purpose (half depth, sagging hips, resting).
19. The human count for each clip (used as ground truth).

## Needed before Phase 4

20. Photos or a drawing of the run course and where cameras can be mounted
    at the start/finish line (height, distance, direction of sun).
21. A sample bib.
