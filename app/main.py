"""
Real-Time Harassment Detection System – FastAPI Entry Point.

Launches the async API server with CORS middleware, lifespan hooks
for model pre-loading, and defines the core analysis endpoint.
"""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.core.config import settings
from app.modules.analysis import analyser
from app.modules.detection import detector
from app.modules.tracking import tracker
from app.schemas.schemas import AnalysisResponse, VideoFrame

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – pre-load models on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the YOLOv9Detector, SORTTracker, and LSTMAnalyser 
    so models are loaded into memory exactly once during server startup.
    """
    logger.info("🚀 Starting %s v%s …", settings.APP_NAME, settings.APP_VERSION)
    
    # 1. Load YOLOv9 weights
    await detector.load_model()
    
    # 2. Initialize SORT tracking states
    await tracker.initialise()
    
    # 3. Load LSTM temporal analyser weights
    await analyser.load_model()
    
    logger.info("✅ All models loaded – server is ready.")
    yield
    logger.info("🛑 Shutting down %s.", settings.APP_NAME)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Real-time harassment detection backend powered by "
        "YOLOv9 (detection) → SORT (tracking) → LSTM (temporal analysis)."
    ),
    lifespan=lifespan,
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core Analysis Endpoint
# ---------------------------------------------------------------------------
def _decode_frame(frame_req: VideoFrame) -> np.ndarray:
    """Decode a VideoFrame request into a NumPy BGR array."""
    if frame_req.base64_image:
        try:
            raw = base64.b64decode(frame_req.base64_image)
            img = Image.open(BytesIO(raw)).convert("RGB")
            return np.array(img)[:, :, ::-1]  # RGB → BGR
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to decode base64 image: {exc}",
            ) from exc

    if frame_req.file_path:
        # Fallback for file_path payloads (requires storage volume binding)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="No image source provided in the payload.",
    )


@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    summary="Analyse a single video frame",
    description=(
        "Executes the YOLOv9 → SORT → LSTM pipeline for physical harassment detection."
    ),
)
async def analyze_frame(frame_req: VideoFrame) -> AnalysisResponse:
    """
    Accepts a VideoFrame payload, runs the detection and tracking pipeline,
    and returns an AlertPayload if the harassment probability > 0.85.
    """
    try:
        # 1. Decode Frame
        frame = _decode_frame(frame_req)

        # 2. YOLOv9 Detection (Person Class Only)
        detections = await detector.detect(frame)

        # 3. SORT Tracking
        detections = await tracker.update(detections)

        # 4. LSTM Analysis
        # Internally loops through tracked pairs (if more than one person is tracked), 
        # passes them to analyser.analyze(pair), and checks the result probability.
        harassment_prob = await analyser.compute_harassment_score(detections)

        # 5. Alert Generation
        # Returns an AlertPayload if the harassment probability > 0.85
        alert = await analyser.maybe_generate_alert(
            frame_id=frame_req.frame_id,
            camera_id=frame_req.camera_id,
            detections=detections,
            harassment_score=harassment_prob,
        )

        return AnalysisResponse(
            frame_id=frame_req.frame_id,
            detections=detections,
            alert=alert,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during frame analysis pipeline.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Ops"])
async def health_check():
    """Lightweight liveness probe."""
    return {"status": "ok", "version": settings.APP_VERSION}


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

