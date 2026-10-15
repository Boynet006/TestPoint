# Research notes (2026-10-15)

## Existing products that do something close

| Product | What it does | Relevance |
|---------|--------------|-----------|
| [Singapore Polytechnic — Intelligent Calisthenics Assistant II](https://www.rp.edu.sg/industry/solutions/technologies-for-licensing/intelligent-calisthenics-assistant-ii/) | Pose estimation + video analytics to count push-ups and sit-ups against standardized criteria; available for licensing | Closest match for the push-up/crunch stations. Worth contacting for a demo or licence if we want to shorten Phase 1. |
| [RaceClocker Finish Cam](https://www.racedirectorshq.com/news/raceclockers-finish-cam-app-introduces-live-electronic-timing-without-the-chips-1350/) | Camera reads bibs at the finish, tracks runners and interpolates the crossing instant, no chips | Same approach we plan for the run. Shows camera-only timing is viable for small events. |
| [Kamero AI bib recognition](https://kamero.ai/post/the-speed-of-victory-solving-the-marathon-bib-gap-with-ai-recognition) | Bib OCR + face identification for race photos; vendor claims 99.2% | Hybrid bib + face is the robust pattern for identification. |
| [FinishLynx chip-timing cameras](https://finishlynx.com/packages/chip-timing-cameras) | Photo-finish cameras paired with RFID | Proven hybrid; our fallback if camera-only timing misses the target. |
| [Pushupstar](https://apps.apple.com/us/app/id1493744133), [RepStandard](https://www.producthunt.com/products/repstandard), [Motion Coach sit-up counter](https://motion-coach.com/situp-counter), [Setshot](https://www.setshot.app/) | Consumer phone apps counting push-ups/sit-ups with the camera; some include military test standards | Useful for UX ideas (how counts and faults are shown). Not built for proctored testing: no evidence, no multi-camera, no audit. |
| [Push-up Android system reliability study (UPNM, 2024)](https://ir.upnm.edu.my/handle/20.500.14847/614) | Validation of an app-based push-up counter on military cadets, r = 0.91 between trials | Shows the kind of validation we should run in Phase 1. |

No product found covers all three events with cameras, evidence storage and
an examiner workflow, which is what TestPoint is.

## Libraries and models

- Pose: [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
  through [`rtmlib`](https://github.com/Tau-J/rtmlib) (ONNX, Apache-2.0).
  A multi-model comparison found RTMPose the most accurate and most robust
  of the common real-time models, with MediaPipe fastest
  ([Oxford study](https://ora.ox.ac.uk/objects/uuid:ff7ad4b5-3e9f-4d71-819d-15d19154a947)).
  YOLO-pose models are AGPL-3.0 ([Roboflow overview](https://playground.roboflow.com/models/task/pose-estimation)).
  MediaPipe is single-person ([ITU comparison](https://research.itu.edu.tr/en/publications/comparative-evaluation-of-mediapipe-and-yolov8-for-real-time-pose/)).
- Rep counting: joint-angle thresholds + hysteresis + minimum phase
  duration. Developers of consumer apps report the hard part is avoiding
  false counts while getting into position
  ([RepStandard](https://chatgate.ai/post/repstandard)); hence our explicit
  "ready pose" gate before counting starts.
- Tracking: ByteTrack (via `supervision` or `boxmot`).
- Bib OCR: PaddleOCR or a small custom digit detector trained on our bibs.
- Timing: camera timing claims vs RFID are summarised in the patents and
  vendor pages linked in [PLAN.md](PLAN.md) Phase 4; RFID remains the
  reference standard
  ([patent background](https://data.epo.org/publication-server/rest/v1.2/patents/EP3792815NWA1/document.html)).
