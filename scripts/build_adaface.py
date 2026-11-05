"""Build the AdaFace face recognition model (data/models/face/face_recognition_adaface_ir101.onnx).

The weights: AdaFace IR-101 trained on WebFace12M, by the AdaFace author
(huggingface.co/minchul/cvlface_adaface_ir101_webface12m, model.safetensors, 261 MB, checked by
its SHA-256 below). Only the weights are downloaded (data, no code is run from there); the network
is defined here, as in CVLFace's models/iresnet/model.py (MIT). Licence: the code is MIT; the
authors ask to follow the training data's licence (WebFace260M/12M, research terms).

Needs PyTorch, once (pip install torch safetensors onnx); the station then runs the ONNX file on the
GPU with ONNX Runtime and needs neither PyTorch nor the internet.

    python scripts/build_adaface.py              # download the weights, build the model
    python scripts/build_adaface.py WEIGHTS      # from weights already downloaded
"""
import hashlib
import sys
import urllib.request
from collections import namedtuple
from pathlib import Path

URL = "https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/resolve/main/model.safetensors"
SHA256 = "2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56"
OUT = Path("data/models/face/face_recognition_adaface_ir101.onnx")

import torch
from safetensors.torch import load_file
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, PReLU, Sequential)


class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class BasicBlockIR(Module):
    def __init__(self, cin, depth, stride):
        super().__init__()
        self.shortcut_layer = MaxPool2d(1, stride) if cin == depth else Sequential(Conv2d(cin, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(BatchNorm2d(cin), Conv2d(cin, depth, (3, 3), (1, 1), 1, bias=False), BatchNorm2d(depth), PReLU(depth),
                                    Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


B = namedtuple("B", "cin depth stride")


def block(cin, depth, n):
    return [B(cin, depth, 2)] + [B(depth, depth, 1) for _ in range(n - 1)]


class IR101(Module):
    def __init__(self):
        super().__init__()
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        blocks = block(64, 64, 3) + block(64, 128, 13) + block(128, 256, 30) + block(256, 512, 3)
        self.body = Sequential(*[BasicBlockIR(b.cin, b.depth, b.stride) for b in blocks])
        self.output_layer = Sequential(BatchNorm2d(512), Dropout(0.4), Flatten(), Linear(512 * 7 * 7, 512), BatchNorm1d(512, affine=False))

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))


def weights() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    p = Path("data/models/adaface_ir101_webface12m.safetensors")
    if not p.is_file():
        print(f"downloading {URL}")
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, p)
    return p


src = weights()
got = hashlib.sha256(src.read_bytes()).hexdigest()
if got != SHA256:
    raise SystemExit(f"checksum mismatch: {got} (expected {SHA256})")
dst = OUT
dst.parent.mkdir(parents=True, exist_ok=True)
sd = load_file(str(src))
print("keys", len(sd), "e.g.", list(sd)[:3])
prefix = next(p for p in ("model.net.", "net.", "model.", "") if any(k.startswith(p + "input_layer") for k in sd))
sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
net = IR101().eval()
missing, unexpected = net.load_state_dict(sd, strict=False)
print("missing", missing, "unexpected", unexpected)
assert not missing, "weights do not fit the network"
x = torch.randn(2, 3, 112, 112)
torch.onnx.export(net, x, str(dst), input_names=["input"], output_names=["embedding"],
                  dynamic_axes={"input": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)
print("exported", dst)
