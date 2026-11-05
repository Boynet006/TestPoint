import cv2
import numpy as np

from testpoint.api.light import DARK, LADDER, PROBE_EVERY_S, SAMPLE_EVERY_S, SETTLE_S, WINDOW, ShutterControl, measure


def feed(ctl, t, sample, n=WINDOW):
    """n pictures, 10 s apart, from t; returns (time after, the last change or None)."""
    change = None
    for _ in range(n):
        c = ctl.step(t, sample)
        change = c if c is not None else change
        t += SAMPLE_EVERY_S
    return t, change


def test_dark_picture_steps_to_a_slower_shutter_and_stops_at_the_slowest():
    ctl = ShutterControl(level=0)                       # 1/1000 at dusk
    t, ch = feed(ctl, 0.0, (DARK - 20, 5.0))
    assert ch == 1 and ctl.shutter == "1/500"
    t, ch = feed(ctl, t + SETTLE_S, (DARK - 20, 5.0))
    assert ch == 2 and ctl.shutter == "1/250"
    t, ch = feed(ctl, t + SETTLE_S, (DARK - 20, 5.0))
    assert ch is None and ctl.shutter == LADDER[-1]     # nothing slower


def test_one_dark_picture_does_not_move_it():
    ctl = ShutterControl(level=0)
    t, _ = feed(ctl, 0.0, (110, 5.0), WINDOW - 1)
    assert ctl.step(t, (20, 5.0)) is None and ctl.shutter == "1/1000"


def test_bright_daylight_tries_faster_and_keeps_it_when_the_picture_holds():
    ctl = ShutterControl()                              # starts at 1/250
    t, ch = feed(ctl, 0.0, (110, 4.0))
    assert ch == 1 and ctl.probe is not None            # trying 1/500
    t, ch = feed(ctl, t + SETTLE_S, (108, 4.5))         # as bright, hardly grainier: kept
    assert ch is None and ctl.shutter == "1/500" and ctl.probe is None
    t, ch = feed(ctl, t + PROBE_EVERY_S, (108, 4.5))    # later: 1/1000
    assert ch == 0 and ctl.shutter == "1/1000"


def test_a_faster_shutter_that_darkens_or_grains_the_picture_is_undone():
    for after in [(80, 4.0), (110, 8.0)]:               # darker / much grainier
        ctl = ShutterControl()
        t, _ = feed(ctl, 0.0, (110, 4.0))
        t, ch = feed(ctl, t + SETTLE_S, after)
        assert ch == 2 and ctl.shutter == "1/250"
        t, ch = feed(ctl, t + SETTLE_S, (110, 4.0))     # not tried again before PROBE_EVERY_S
        assert ch is None


def test_measure_sees_grain_and_brightness():
    rng = np.random.default_rng(1)
    flat = np.full((360, 640), 100, np.uint8)
    grainy = np.clip(flat + rng.normal(0, 12, flat.shape), 0, 255).astype(np.uint8)
    b1, n1 = measure(cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    b2, n2 = measure(cv2.imencode(".jpg", grainy, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    assert abs(b1 - 100) < 2 and abs(b2 - 100) < 3 and n2 > 4 * max(n1, 0.5)
    assert measure(b"not a picture") is None


def test_no_trial_while_someone_runs_and_a_trial_under_way_is_undone():
    ctl = ShutterControl()
    t, ch = feed(ctl, 0.0, (110, 4.0))
    assert ch == 1 and ctl.probe is not None
    assert ctl.step(t, (110, 4.0), busy=True) == 2 and ctl.shutter == "1/250"   # a runner set off: back at once
    ctl2 = ShutterControl()
    change = None
    for i in range(WINDOW * 3):
        change = ctl2.step(i * SAMPLE_EVERY_S, (110, 4.0), busy=True) or change
    assert change is None and ctl2.shutter == "1/250"
