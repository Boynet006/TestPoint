"""Start-line camera: an IP camera's picture relayed to the station app, for face check-in.

The start page cannot open an ``rtsp://`` stream itself (a browser only reads its own webcams),
so this process reads the camera on the station PC and sends its newest frame to
``/api/engine/frame/start`` a few times a second. The page shows that picture and sends the
newest frame to face identification, exactly as it does with a webcam frame. No analysis runs
here: the picture goes out as the camera sent it, large enough for faces a few metres away.

    python -m testpoint.startcam --source rtsp://user:pass@<ip>:554/Streaming/Channels/101 --app http://127.0.0.1:8000

(The camera service starts it from the start page; see testpoint.camhost.)
"""

from __future__ import annotations

import argparse
import time

from .preview import FramePublisher
from .vision.sources import SourceError, open_source, redact_url

STATION = "start"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="TestPoint start-line camera (relays the picture for face check-in)")
    p.add_argument("--source", required=True, help="webcam index, video file or rtsp:// address")
    p.add_argument("--app", required=True, help="station app address, e.g. http://127.0.0.1:8000")
    p.add_argument("--station", default=STATION, help="screen this camera shows on (default start)")
    p.add_argument("--fps", type=float, default=10.0, help="pictures sent a second (default 10)")
    p.add_argument("--max-width", type=int, default=1920, help="widest picture sent (default 1920)")
    p.add_argument("--realtime", action="store_true", help="play a video file at its own speed")
    p.add_argument("--loop", action="store_true", help="play a video file on repeat")
    args = p.parse_args(argv)

    src = open_source(args.source, realtime=args.realtime, loop=args.loop)
    pub = FramePublisher(args.app, args.station, draw=lambda img, state: None,
                         interval=1 / args.fps, max_width=args.max_width, quality=88)
    pub.start()
    print(f"start cam: source {redact_url(args.source)}  app {args.app}  station {args.station}", flush=True)
    frames, t0 = 0, time.monotonic()
    try:
        while True:
            frame = src.read()
            if frame is None:
                print("start cam: the video ended", flush=True)
                break
            pub.offer_frame(frame)
            frames += 1
            if frames == 1:
                h, w = frame.shape[:2]
                print(f"start cam: picture {w}x{h}", flush=True)
            if time.monotonic() - t0 >= 60:
                print(f"start cam: {frames / (time.monotonic() - t0):.0f} fps from the camera, {pub.sent} sent, {pub.failed} failed", flush=True)
                frames, t0 = 0, time.monotonic()
    except SourceError as e:
        print(f"start cam: {e}", flush=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        pass
    finally:
        pub.stop()
        src.release()


if __name__ == "__main__":
    main()
