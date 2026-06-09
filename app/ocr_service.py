"""PaddleOCR (PP-OCRv5) wrapper with cached server / mobile model instances.

Two PaddleOCR pipelines are kept alive so the UI can switch between the
server-grade and the mobile-grade PP-OCRv5 models for side-by-side comparison.
Each pipeline is built lazily on first use (the first call downloads weights).
"""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

# (detection model, recognition model) per variant.
_MODEL_NAMES = {
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
}

# Downscale large images before OCR to keep CPU latency reasonable; the detected
# coordinates are mapped back to the original pixel space afterwards so the
# frontend can overlay them on the full-resolution image.
MAX_SIDE = 2048

# Detection tuning for clean printed text. The input is regular print, so bias
# detection toward precision: raise the score thresholds and tighten the box
# expansion so logos, icons and other non-text marks are not picked up as text.
# Brackets show the PP-OCRv5 / DBNet defaults these override.
DET_THRESH = 0.4        # prob-map pixel -> text mask      [default 0.3]
DET_BOX_THRESH = 0.7    # mean region score to keep a box  [default 0.6]
DET_UNCLIP_RATIO = 1.2  # box dilation; lower = tighter    [default 1.5]

# Drop recognitions whose recognition confidence is below this, so uncertain
# reads (faint marks, partial glyphs, noise) are not shown, counted or copied.
MIN_REC_SCORE = 0.85

_instances: Dict[str, Any] = {}
_lock = threading.Lock()


def _log(msg: str) -> None:
    print(f"[ocr] {msg}", file=sys.stderr, flush=True)


def _warmup(inst) -> None:
    """Compile Paddle's graphs now (one-time cold start) using the bundled
    sample image, so the user's first real request — and the reported
    inference time — reflect warm, steady-state performance."""
    sample = Path(__file__).resolve().parent.parent / "examples" / "sample.png"
    try:
        if sample.exists():
            arr = np.asarray(Image.open(sample).convert("RGB"))[:, :, ::-1]
            list(inst.predict(np.ascontiguousarray(arr)))
    except Exception as exc:  # best-effort: never block startup on warmup
        _log(f"warmup skipped: {exc}")


def get_ocr(model_type: str):
    """Return a cached PaddleOCR instance for ``server`` or ``mobile``."""
    if model_type not in _MODEL_NAMES:
        raise ValueError(f"unknown model_type: {model_type!r}")

    inst = _instances.get(model_type)
    if inst is not None:
        return inst

    with _lock:
        inst = _instances.get(model_type)
        if inst is not None:
            return inst

        from paddleocr import PaddleOCR

        det, rec = _MODEL_NAMES[model_type]
        _log(f"loading '{model_type}' pipeline ({det} + {rec}); first run may download weights ...")
        inst = PaddleOCR(
            text_detection_model_name=det,
            text_recognition_model_name=rec,
            # Keep the demo to det + rec only: skip the extra orientation /
            # unwarping models so startup stays light and fast.
            # Target: high-resolution scanned documents only (no handwriting /
            # photos / rotated text), so every robustness stage for those hard
            # cases is off — keeping the pipeline to detection + recognition.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # Bias detection toward precision on clean print so logos / icons
            # are not picked up as text (see DET_* constants above).
            text_det_thresh=DET_THRESH,
            text_det_box_thresh=DET_BOX_THRESH,
            text_det_unclip_ratio=DET_UNCLIP_RATIO,
            device="cpu",
            # oneDNN (MKLDNN) makes CPU inference ~10x faster on PP-OCRv5
            # (~25s -> ~2.4s for a 51-line A4 scan, mobile). Requires
            # paddlepaddle==3.2.2 — 3.3.x has an oneDNN+PIR regression that
            # crashes here (ConvertPirAttribute2RuntimeAttribute); see
            # requirements.txt.
            enable_mkldnn=True,
            # Batch line recognition — small extra win on multi-line scans.
            text_recognition_batch_size=8,
        )
        _warmup(inst)
        _instances[model_type] = inst
        _log(f"'{model_type}' pipeline ready")
        return inst


def _get(res: Any, key: str, default=None):
    """Read ``key`` from a PaddleOCR result (dict-like or attribute access)."""
    try:
        if key in res:
            return res[key]
    except TypeError:
        pass
    return getattr(res, key, default)


def _to_poly(poly: Any) -> List[List[float]]:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in pts]


def _poly_box(poly: List[List[float]]) -> List[float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


# Detection quads come back slightly rotated when a line has asymmetric
# ascenders/descenders (e.g. an English line containing "y"/"g"/"p"): DB's
# min-area-rect tilts to hug the uneven ink even though the baseline is level.
# Tilt is measured as the top/bottom edge slope (rise/run == tan of the angle,
# no trig, and independent of line length); when it is small we snap the quad
# to axis-aligned about its own centre while preserving width and height.
_LEVEL_MAX_SLOPE = 0.14  # tan(8 deg); above this the text is treated as rotated


def _level_poly(poly: List[List[float]]) -> List[List[float]]:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    top_slope = abs(y1 - y0) / max(1.0, abs(x1 - x0))
    bot_slope = abs(y2 - y3) / max(1.0, abs(x2 - x3))
    if max(top_slope, bot_slope) > _LEVEL_MAX_SLOPE:
        return poly  # genuinely rotated text — keep as detected
    cx = (x0 + x1 + x2 + x3) / 4.0
    cy = (y0 + y1 + y2 + y3) / 4.0
    w = (math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x3, y2 - y3)) / 2.0
    h = (math.hypot(x3 - x0, y3 - y0) + math.hypot(x2 - x1, y2 - y1)) / 2.0
    hw, hh = w / 2.0, h / 2.0
    return [
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ]


def recognize(image: Image.Image, model_type: str) -> Dict[str, Any]:
    """Run OCR and return ``{width, height, items[]}`` in original pixel space.

    Each item: ``{text, score, poly:[[x,y]*4], box:[x0,y0,x1,y1]}``.
    """
    t_load = time.perf_counter()
    ocr = get_ocr(model_type)
    load_ms = (time.perf_counter() - t_load) * 1000  # ~0 once cached

    t0 = time.perf_counter()
    image = image.convert("RGB")
    w, h = image.size

    scale = 1.0
    proc = image
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / float(max(w, h))
        proc = image.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
        )

    # PaddleOCR's preprocessing (built on cv2) expects BGR channel order.
    bgr = np.ascontiguousarray(np.asarray(proc)[:, :, ::-1])
    t1 = time.perf_counter()

    # predict() may return a list or a generator depending on the version.
    results = list(ocr.predict(bgr))
    t2 = time.perf_counter()

    items: List[Dict[str, Any]] = []
    if results:
        res = results[0]
        texts = _get(res, "rec_texts", []) or []
        scores = _get(res, "rec_scores", []) or []
        polys = _get(res, "rec_polys", None)
        boxes = _get(res, "rec_boxes", None)

        inv = 1.0 / scale
        for i, text in enumerate(texts):
            score = None
            if i < len(scores) and scores[i] is not None:
                score = float(scores[i])
            if score is not None and score < MIN_REC_SCORE:
                continue  # ignore low-confidence reads (faint marks / noise)

            if polys is not None and i < len(polys):
                poly = _to_poly(polys[i])
            elif boxes is not None and i < len(boxes):
                x0, y0, x1, y1 = [float(v) for v in np.asarray(boxes[i]).reshape(-1)[:4]]
                poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            else:
                continue

            if inv != 1.0:
                poly = [[x * inv, y * inv] for x, y in poly]
            poly = _level_poly(poly)

            items.append(
                {"text": text, "score": score, "poly": poly, "box": _poly_box(poly)}
            )

    t3 = time.perf_counter()
    timings = {
        "load_ms": round(load_ms, 1),
        "preprocess_ms": round((t1 - t0) * 1000, 1),
        "inference_ms": round((t2 - t1) * 1000, 1),
        "postprocess_ms": round((t3 - t2) * 1000, 1),
    }
    return {"width": w, "height": h, "items": items, "timings": timings}


def recognize_text(image: Image.Image, model_type: str) -> Dict[str, Any]:
    """Recognize text in a (usually pre-cropped) region; return ``{text, score}``.

    Used by the "fine-tune region" feature: the caller already owns the box
    geometry and only wants the words inside it. Unlike :func:`recognize` this
    returns no polygons and does NOT apply ``MIN_REC_SCORE`` — a region the user
    drew on purpose should report whatever it found, with its confidence.
    """
    ocr = get_ocr(model_type)
    t0 = time.perf_counter()
    image = image.convert("RGB")
    bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    results = list(ocr.predict(bgr))

    lines: List[Any] = []  # (top_y, left_x, text, score) for reading-order sort
    if results:
        res = results[0]
        texts = _get(res, "rec_texts", []) or []
        scores = _get(res, "rec_scores", []) or []
        polys = _get(res, "rec_polys", None)
        boxes = _get(res, "rec_boxes", None)
        for i, text in enumerate(texts):
            if not text:
                continue
            top = left = 0.0
            if polys is not None and i < len(polys):
                pts = np.asarray(polys[i], dtype=float).reshape(-1, 2)
                top, left = float(pts[:, 1].min()), float(pts[:, 0].min())
            elif boxes is not None and i < len(boxes):
                b = np.asarray(boxes[i]).reshape(-1)[:4]
                top, left = float(b[1]), float(b[0])
            sc = float(scores[i]) if i < len(scores) and scores[i] is not None else None
            lines.append((round(top, 1), round(left, 1), text, sc))

    lines.sort(key=lambda t: (t[0], t[1]))
    text = " ".join(t[2] for t in lines)
    valid = [t[3] for t in lines if t[3] is not None]
    score = sum(valid) / len(valid) if valid else None
    timings = {"region_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return {"text": text, "score": score, "timings": timings}
