"""
FastAPI wrapper for the Deepfake Detection model.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Efficiency features:
  • Model loaded once at startup via lifespan event
  • Inference offloaded to a thread-pool so the async loop never blocks
  • Temp files cleaned up immediately after use
  • CORS enabled for frontend integration
"""

import os
import asyncio
import tempfile
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import (
    DEVICE, MODEL_DIR, FAKE_THRESHOLD, SUSPICIOUS_THRESHOLD,
    FRAMES_PER_VIDEO, IMAGE_SIZE,
)
from detect import DeepfakeInference
from api_models import (
    HealthResponse,
    ImagePrediction,
    VideoPrediction,
    FramePrediction,
    Probabilities,
    ErrorResponse,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_inference: DeepfakeInference | None = None
_executor = ThreadPoolExecutor(max_workers=2)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", "video/x-matroska"}
MAX_IMAGE_SIZE = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB


# ---------------------------------------------------------------------------
# Lifespan — load model once, reuse forever
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    print("🚀 Loading deepfake detection model …")
    _inference = DeepfakeInference(model_path=model_path, use_face_detection=True)
    print("✅ Model ready.")
    yield
    # Cleanup
    _inference = None
    torch.cuda.empty_cache()
    print("🛑 Model unloaded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deepfake Detection API",
    description="Detect deepfakes in images and videos using an EfficientNet model with face detection.",
    version="1.0.0",
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_inference() -> DeepfakeInference:
    if _inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return _inference


async def _run_in_pool(fn, *args, **kwargs):
    """Run a blocking function in the thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, partial(fn, *args, **kwargs))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check API health and model status."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return HealthResponse(
        status="healthy",
        device=str(DEVICE),
        model_loaded=_inference is not None,
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        fake_threshold=FAKE_THRESHOLD,
        suspicious_threshold=SUSPICIOUS_THRESHOLD,
    )


@app.post(
    "/predict/image",
    response_model=ImagePrediction,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"],
)
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, BMP)"),
    use_face_detection: bool = Query(False, description="Run face detection before classification"),
):
    """
    Upload a single image and get a deepfake prediction.

    Returns the predicted class (REAL / FAKE), confidence, and per-class
    probabilities.
    """
    # Validate content type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    # Read file bytes (with size guard)
    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail=f"Image exceeds {MAX_IMAGE_SIZE // (1024*1024)} MB limit.")

    # Decode image
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    inf = _get_inference()

    # Toggle face detection per-request
    original_fd = inf.use_face_detection
    inf.use_face_detection = use_face_detection

    try:
        result = await _run_in_pool(inf.predict_image, image)
    finally:
        inf.use_face_detection = original_fd

    return ImagePrediction(
        classification=result["class"],
        confidence=result["confidence"],
        probabilities=Probabilities(**result["probabilities"]),
        face_detected=use_face_detection,
    )


@app.post(
    "/predict/video",
    response_model=VideoPrediction,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"],
)
async def predict_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV)"),
    use_face_detection: bool = Query(True, description="Run face detection on each frame"),
):
    """
    Upload a video and get a per-frame + aggregate deepfake prediction.

    The video is sampled at `FRAMES_PER_VIDEO` uniformly-spaced frames.
    The final verdict is one of REAL, SUSPICIOUS, or FAKE.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{file.content_type}'. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}",
        )

    # Save to a temp file (OpenCV needs a file path)
    suffix = os.path.splitext(file.filename or "video.mp4")[1]
    tmp_path = os.path.join(tempfile.gettempdir(), f"dfapi_{uuid.uuid4().hex}{suffix}")

    try:
        # Stream upload to disk
        data = await file.read()
        if len(data) > MAX_VIDEO_SIZE:
            raise HTTPException(status_code=400, detail=f"Video exceeds {MAX_VIDEO_SIZE // (1024*1024)} MB limit.")

        with open(tmp_path, "wb") as f:
            f.write(data)

        inf = _get_inference()
        original_fd = inf.use_face_detection
        inf.use_face_detection = use_face_detection

        try:
            result = await _run_in_pool(inf.predict_video, tmp_path, False)
        finally:
            inf.use_face_detection = original_fd

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    frame_preds = [
        FramePrediction(
            frame=fp["frame"],
            classification=fp["class"],
            confidence=fp["confidence"],
            probabilities=Probabilities(**fp["probabilities"]),
        )
        for fp in result["frame_predictions"]
    ]

    return VideoPrediction(
        verdict=result["verdict"],
        confidence=result["confidence"],
        average_fake_probability=result["average_fake_probability"],
        total_frames=result["total_frames"],
        sampled_frames=result["sampled_frames"],
        frame_predictions=frame_preds,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
