"""Download the rtmlib pose models into $TORCH_HOME while the image is built.

    python docker/fetch_models.py lightweight balanced

The test site has no internet, so every mode the station may use has to be in
the image. rtmlib skips the download when the .onnx file is already in
$TORCH_HOME/hub/checkpoints, so running this twice is harmless.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODES = ("lightweight", "balanced", "performance")


def main(argv: list[str]) -> int:
    modes = argv or ["balanced"]
    bad = [m for m in modes if m not in MODES]
    if bad:
        print(f"unknown mode(s) {bad}; choose from {list(MODES)}", file=sys.stderr)
        return 2

    from rtmlib import Body

    for mode in modes:
        print(f"fetching rtmlib models for mode={mode}", flush=True)
        Body(to_openpose=False, mode=mode, backend="onnxruntime", device="cpu")

    home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "rtmlib"))
    ckpt = home / "hub" / "checkpoints"
    files = sorted(ckpt.glob("*.onnx"))
    if not files:
        print(f"no models found in {ckpt}", file=sys.stderr)
        return 1
    for f in files:
        print(f"  {f.stat().st_size / 1e6:7.1f} MB  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
