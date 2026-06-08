"""Benchmark PP-OCRv5 (mobile) CPU latency under different speed parameters,
on a representative high-DPI scan. Prints median/min inference time + #lines so
we can see the speed/accuracy trade-off before committing parameters."""
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

img = Image.open(ROOT / "examples" / "scan.png").convert("RGB")
bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])

BASE = dict(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)

CONFIGS = {
    "mkldnn=off (reference)": dict(enable_mkldnn=False),
    "mkldnn=on": dict(enable_mkldnn=True),
    "mkldnn=on + threads=16": dict(enable_mkldnn=True, cpu_threads=16),
    "mkldnn=on + rec_batch=8": dict(enable_mkldnn=True, text_recognition_batch_size=8),
    "mkldnn=on + det736": dict(
        enable_mkldnn=True,
        text_det_limit_side_len=736,
        text_det_limit_type="max",
    ),
}


def bench(extra, runs=3):
    ocr = PaddleOCR(**BASE, **extra)
    warm = list(ocr.predict(bgr))  # load + graph compile
    n = len(warm[0]["rec_texts"]) if warm else 0
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        list(ocr.predict(bgr))
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), min(ts), n


print(f"image {img.width}x{img.height}", flush=True)
for name, extra in CONFIGS.items():
    med, mn, n = bench(extra)
    print(f"{name:34s} median={med:7.0f} ms  min={mn:7.0f} ms  lines={n}", flush=True)
