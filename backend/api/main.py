"""
FastAPI backend: the inference pipeline behind the demo UI.

Endpoints:
    POST /predict
        body: image file (multipart form field "file")
        returns: {
            "label": "defect" | "good",
            "confidence": float,
            "defect_probability": float,
            "heatmap_png_b64": str,        # Grad-CAM overlay
            "description": str | null      # LLaVA description (if enabled)
        }
    GET  /health -> {"status": "ok", "model_ready": bool, ...}

Loads the best classifier checkpoint once at startup (see inference.py). Calls:
    src.classifier.model (load weights)
    src.explain.gradcam   (heatmap)
    src.vlm.describe      (optional description)

Config is via environment variables (see inference.engine_from_env). Before a
checkpoint exists, start a dev server with an untrained model:
    DEFECT_API_ALLOW_RANDOM=1 uvicorn backend.api.main:app --reload --port 8000

Run (after training):
    DEFECT_API_CHECKPOINT=experiments/checkpoints/bottle_ours.pt \
        uvicorn backend.api.main:app --reload --port 8000

Owner: Member 4 (you)
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.api.inference import engine_from_env


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the inference engine once at startup. Don't crash the server if the
    model isn't available yet — /health stays up and /predict reports 503."""
    try:
        app.state.engine = engine_from_env()
        app.state.engine_error = None
    except Exception as exc:  # FileNotFoundError (no ckpt) or load failure
        app.state.engine = None
        app.state.engine_error = str(exc)
    yield


app = FastAPI(title="Defect Detection API", version="0.1.0", lifespan=lifespan)

# Allow the Next.js dev server (and common local hosts) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_engine():
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(app.state, "engine_error", None)
            or "inference engine not initialised",
        )
    return engine


@app.get("/health")
def health():
    engine = getattr(app.state, "engine", None)
    return {
        "status": "ok",
        "model_ready": bool(engine and engine.ready),
        "model": engine.info() if engine else None,
        "error": getattr(app.state, "engine_error", None),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    engine = _get_engine()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file upload")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="could not decode image file")

    return engine.predict(image)
