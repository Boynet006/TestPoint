"""The TestPoint icon: the header's mark (a dark T on an amber square with a cut corner) as
web/station/icon.ico (the desktop shortcut, the browser tab) and web/station/icon.png.
Run from the repository root: python scripts/make_icon.py"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

AMBER, INK = (255, 176, 0, 255), (26, 18, 0, 255)


def mark(size: int) -> Image.Image:
    s = 1024                                   # drawn large, then made smaller (smooth edges)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cut = int(s * 0.16)                        # the cut top-right corner, as in the header
    d.polygon([(0, 0), (s - cut, 0), (s, cut), (s, s), (0, s)], fill=AMBER)
    try:
        font = ImageFont.truetype("arialbd.ttf", int(s * 0.78))
    except OSError:
        font = ImageFont.load_default(size=int(s * 0.78))
    box = d.textbbox((0, 0), "T", font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    d.text(((s - w) / 2 - box[0], (s - h) / 2 - box[1] + s * 0.02), "T", font=font, fill=INK)
    return img.resize((size, size), Image.LANCZOS)


out = Path("web/station")
big = mark(256)
big.save(out / "icon.png")
big.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("icon.ico and icon.png written to", out)
