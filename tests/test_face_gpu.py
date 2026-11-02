"""The face models on the GPU give the same faces and embeddings as on the CPU.

Needs the face models (data/models/face) and ONNX Runtime with CUDA; the comparison on a real
face also needs a picture with one face: TESTPOINT_FACE_SAMPLE=<path to a photo>.
"""

import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from testpoint.vision import face_gpu

MODELS = Path(os.environ.get("TESTPOINT_FACE_MODELS", "data/models/face"))
needs_gpu = pytest.mark.skipif(not (face_gpu.cuda_available() and (MODELS / "face_detection_yunet_2023mar.onnx").is_file()),
                               reason="needs ONNX Runtime CUDA and the face models")


def test_tiles_cover_the_picture_with_overlap():
    assert face_gpu._starts(640) == [0] and face_gpu._starts(300) == [0]
    s = face_gpu._starts(1280)
    assert s[0] == 0 and s[-1] == 1280 - 640 and all(b - a <= 640 * 0.75 for a, b in pairwise(s))


@needs_gpu
def test_no_face_in_an_empty_picture():
    from testpoint.api.faces import FaceEngine
    eng = FaceEngine(MODELS, device="cuda")
    assert eng.device == "cuda"
    assert len(eng.detect(np.full((1080, 1920, 3), 90, np.uint8))) == 0


@needs_gpu
@pytest.mark.skipif(not os.environ.get("TESTPOINT_FACE_SAMPLE"), reason="set TESTPOINT_FACE_SAMPLE to a photo with one face")
def test_gpu_and_cpu_agree_on_a_real_face():
    import cv2

    from testpoint.api.faces import FaceEngine
    img = cv2.imread(os.environ["TESTPOINT_FACE_SAMPLE"])
    cpu, gpu = FaceEngine(MODELS, device="cpu"), FaceEngine(MODELS, device="cuda")
    a, b = cpu.largest_face(img), gpu.largest_face(img)
    assert max(abs(p - q) for p, q in zip(a.box, b.box)) <= 8
    assert float(np.dot(a.embedding, b.embedding)) > 0.9          # same person by far (match needs 0.36)
    assert abs(a.quality - b.quality) < 0.1
