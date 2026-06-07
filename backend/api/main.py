"""
FastAPI backend: the inference pipeline behind the demo UI.

Endpoints:
    POST /predict
        body: image file
        returns: {
            "label": "defect" | "good",
            "confidence": float,
            "heatmap_png_b64": str,        # Grad-CAM overlay
            "description": str | null      # LLaVA description (if enabled)
        }
    GET  /health -> {"status": "ok"}

Loads the best classifier checkpoint once at startup. Calls:
    src.classifier.model (load weights)
    src.explain.gradcam   (heatmap)
    src.vlm.describe      (optional description)

Run:
    uvicorn backend.api.main:app --reload --port 8000

Owner: Member 4 (you)
"""

from fastapi import FastAPI

app = FastAPI(title="Defect Detection API")


@app.get("/health")
def health():
    return {"status": "ok"}


# @app.post("/predict")
# async def predict(file: UploadFile):
#     Member 4: decode image -> classify -> gradcam -> (optional) describe -> JSON
