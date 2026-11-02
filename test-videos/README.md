# Test videos

Recordings from the real stations (2026-11-02), for trying the counting through the app on any
machine: the camera screens list them under **Video file** (the camera service looks in this folder
as well as `data/clips`). H.264, 1920 wide, at real speed.

| File | What is in it | True count |
|---|---|---|
| `pushup/omar-10-pushups.mp4` | Only the push-ups, 14 s | 10 good push-ups |
| `pushup/omar-full-test-pushups.mp4` | The whole 2-minute test: getting into position, the push-ups, walking off, sitting at a desk | 10 push-ups, nothing else |
| `crunch/omar-10-crunches.mp4` | Only the crunches, 21 s (ends sitting up) | 10 good crunches |
| `crunch/omar-full-test-crunches.mp4` | The whole 2-minute test, the lights go off near the end | 10 crunches, nothing else |
| `run/omar-finish-crossing.mp4` | One finish-line crossing | 1 crossing |

Station camera: about 3 m away, low and to the side. A short clip played on repeat counts one
rep fewer per round (it starts while he is still getting into position); the full tests count 10.
The poses of the two full tests are in `tests/recordings` and `tests/test_counting_recordings.py`
replays them, so the counting is checked by the tests without these videos.
