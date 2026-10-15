"""Benchmark the rtmlib pose pipeline (YOLOX detector + RTMPose) on CPU and CUDA.

    python scripts/bench_pose.py                 # all devices, modes and sizes
    python scripts/bench_pose.py --device cuda   # GPU only
    python scripts/bench_pose.py --iters 50 --warmup 10

Frames are synthetic (random noise), so the detector finds nobody and rtmlib
then runs the pose model once on the whole frame. One frame therefore costs
"detector + one pose inference", which is what a single person in view costs.
Each extra person in a real frame adds one more pose inference.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

MODES = ("lightweight", "balanced", "performance")
SIZES = {"640x480": (480, 640), "1920x1080": (1080, 1920)}


def bench(backend, frame: np.ndarray, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        backend.detect(frame)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        backend.detect(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), min(times)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    p.add_argument("--mode", choices=[*MODES, "all"], default="all")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--cpu-iters", type=int, default=5, help="iterations for the (slow) CPU runs")
    args = p.parse_args(argv)

    import onnxruntime as ort

    from testpoint.vision.pose import RTMLibBackend

    available = ort.get_available_providers()
    print(f"onnxruntime {ort.__version__}  providers: {available}")
    devices = ["cpu", "cuda"] if args.device == "all" else [args.device]
    if "cuda" in devices and "CUDAExecutionProvider" not in available:
        print("CUDAExecutionProvider is not available; install onnxruntime-gpu[cuda,cudnn]")
        return 1
    modes = list(MODES) if args.mode == "all" else [args.mode]

    rng = np.random.default_rng(0)
    frames = {name: rng.integers(0, 255, (*hw, 3), dtype=np.uint8) for name, hw in SIZES.items()}

    rows = []
    for device in devices:
        iters = args.cpu_iters if device == "cpu" else args.iters
        for mode in modes:
            t0 = time.perf_counter()
            backend = RTMLibBackend(device=device, mode=mode)
            load_s = time.perf_counter() - t0
            prov = backend.providers[0]
            for size, frame in frames.items():
                med, best = bench(backend, frame, args.warmup, iters)
                rows.append((device, mode, size, prov, load_s, med, best, 1000.0 / med))
                print(
                    f"  {device:4s} {mode:11s} {size:9s} {med:8.1f} ms  {1000 / med:6.1f} fps",
                    file=sys.stderr,
                )
            del backend

    print()
    print(
        f"{'device':6s} {'mode':11s} {'frame':9s} {'provider':22s} {'load s':>6s} "
        f"{'median ms':>9s} {'min ms':>7s} {'fps':>6s}"
    )
    print("-" * 82)
    for device, mode, size, prov, load_s, med, best, fps in rows:
        print(
            f"{device:6s} {mode:11s} {size:9s} {prov:22s} {load_s:6.1f} "
            f"{med:9.1f} {best:7.1f} {fps:6.1f}"
        )
    print()
    print("| device | mode | frame | median ms | fps |")
    print("|---|---|---|---|---|")
    for device, mode, size, _prov, _load, med, _best, fps in rows:
        print(f"| {device} | {mode} | {size} | {med:.1f} | {fps:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
