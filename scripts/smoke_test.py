"""Standalone OCR smoke test: runs recognize() on the sample image for both
models (triggers first-run model download), prints text + boxes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.ocr_service import recognize  # noqa: E402

img = Image.open(ROOT / "examples" / "sample.png")

for model in ("mobile", "server"):
    print(f"\n===== {model} =====", flush=True)
    res = recognize(img, model)
    print(f"size={res['width']}x{res['height']}  items={len(res['items'])}", flush=True)
    for it in res["items"]:
        score = it["score"] if it["score"] is not None else float("nan")
        box = [round(v) for v in it["box"]]
        print(f"  {score:.3f}  box={box}  {it['text']!r}", flush=True)

print("\nOK", flush=True)
