"""Save the page's Google Fonts (woff2, Latin + Arabic subsets) under web/station/fonts and write
fonts.css with local @font-face rules, so the page looks the same without the internet."""
import re
import urllib.request
from pathlib import Path

CSS_URL = ("https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800"
           "&family=Barlow:wght@400;500;600&family=Cairo:wght@400;600;700;800&display=swap")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
out = Path("web/station/fonts")
out.mkdir(parents=True, exist_ok=True)

css = urllib.request.urlopen(urllib.request.Request(CSS_URL, headers={"User-Agent": UA}), timeout=30).read().decode()
blocks = re.findall(r"/\* ([\w-]+) \*/\s*(@font-face \{.*?\})", css, re.S)
keep, local = {"latin", "latin-ext", "arabic"}, []
for subset, block in blocks:
    if subset not in keep:
        continue
    family = re.search(r"font-family: '([^']+)'", block).group(1)
    weight = re.search(r"font-weight: (\d+)", block).group(1)
    url = re.search(r"url\((https://[^)]+)\)", block).group(1)
    name = f"{family.replace(' ', '')}-{weight}-{subset}.woff2"
    if not (out / name).exists():
        (out / name).write_bytes(urllib.request.urlopen(url, timeout=30).read())
    local.append(f"/* {subset} */\n" + block.replace(url, f"fonts/{name}"))
header = ("/* Barlow, Barlow Condensed and Cairo (SIL Open Font License 1.1), saved from Google Fonts so\n"
          "   the screens look the same without the internet. Rebuilt by scripts/save_fonts.py. */\n")
(out.parent / "fonts.css").write_text(header + "\n".join(local) + "\n", encoding="utf-8")
print(len(local), "faces,", sum(f.stat().st_size for f in out.glob("*.woff2")) // 1024, "KB")
