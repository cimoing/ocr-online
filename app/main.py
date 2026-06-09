"""FastAPI app: serves the static OCR UI and the /api/ocr endpoint."""
from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from .ocr_service import recognize, recognize_text

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="PP-OCRv5 OCR Service")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ocr")
async def api_ocr(image: UploadFile = File(...), model: str = Form("mobile")) -> dict:
    if model not in ("server", "mobile"):
        raise HTTPException(status_code=400, detail="model must be 'server' or 'mobile'")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    t_decode = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")
    decode_ms = round((time.perf_counter() - t_decode) * 1000, 1)

    # OCR is blocking / CPU-bound: run it off the event loop.
    t0 = time.perf_counter()
    result = await run_in_threadpool(recognize, img, model)
    recognize_ms = (time.perf_counter() - t0) * 1000

    timings = dict(result["timings"])
    timings["decode_ms"] = decode_ms
    timings["server_total_ms"] = round(decode_ms + recognize_ms, 1)

    return {
        "model": model,
        "width": result["width"],
        "height": result["height"],
        "count": len(result["items"]),
        "items": result["items"],
        "timings": timings,
    }


@app.post("/api/ocr_region")
async def api_ocr_region(
    image: UploadFile = File(...), model: str = Form("mobile")
) -> dict:
    """Recognize a single cropped region (the "fine-tune region" feature).

    The frontend crops the adjusted/drawn box out of the original-resolution
    image and posts just that patch; we run OCR on it and return the text only.
    """
    if model not in ("server", "mobile"):
        raise HTTPException(status_code=400, detail="model must be 'server' or 'mobile'")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    result = await run_in_threadpool(recognize_text, img, model)
    return {
        "model": model,
        "text": result["text"],
        "score": result["score"],
        "timings": result["timings"],
    }
