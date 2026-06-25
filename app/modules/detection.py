"""
Object detection module – YOLOv9 integration.

Uses the Ultralytics YOLO API to load a YOLOv9 checkpoint and run
inference on individual video frames.  Only **person** detections
(COCO class ID 0) are retained so downstream modules (SORT, LSTM)
operate exclusively on human subjects.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.core.config import settings
from app.schemas.schemas import BoundingBox, DetectionClass, DetectionResult

logger = logging.getLogger(__name__)

# COCO class ID for "person"
_PERSON_CLASS_ID: int = 0


class YOLOv9Detector:
    """
    Wrapper around a YOLOv9 model for frame-level person detection.

    Attributes:
        weights_path: Filesystem path to the ``.pt`` weights file.
        confidence_threshold: Default minimum confidence for keeping
            a detection (can be overridden per call).
        device: ``"cuda"`` when a GPU is available, otherwise ``"cpu"``.
    """

    def __init__(
        self,
        weights_path: str = settings.YOLO_WEIGHTS_PATH,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: Optional[YOLO] = None

        logger.info(
            "YOLOv9Detector initialised (weights=%s, device=%s, conf=%.2f)",
            self.weights_path,
            self.device,
            self.confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    async def load_model(self) -> None:
        """
        Load the YOLOv9 weights into memory and move to the
        appropriate device (GPU / CPU).
        """
        logger.info("Loading YOLOv9 weights from %s …", self.weights_path)
        try:
            self._model = YOLO(self.weights_path)
            self._model.to(self.device)
            logger.info(
                "YOLOv9 model loaded successfully on %s.", self.device,
            )
        except Exception:
            logger.exception("Failed to load YOLOv9 model.")
            raise

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> list[list[float]]:
        """
        Run YOLOv9 inference on a single frame and return **person-only**
        bounding boxes.

        Args:
            frame: BGR image as a NumPy array (H × W × 3).
            confidence_threshold: Override the instance-level default.

        Returns:
            A list of detections, each formatted as
            ``[x1, y1, x2, y2, confidence]`` (pixel coordinates).
            Returns an empty list when the input is invalid or no
            persons are detected.
        """
        # ---- Input validation ----
        if frame is None or not isinstance(frame, np.ndarray):
            logger.warning("process_frame received invalid input (not a numpy array).")
            return []

        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning(
                "process_frame expected (H, W, 3) array, got shape %s.",
                frame.shape,
            )
            return []

        if frame.size == 0:
            logger.warning("process_frame received an empty frame.")
            return []

        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # ---- Ensure model is loaded (sync guard) ----
        if self._model is None:
            logger.error("Model not loaded. Call load_model() before process_frame().")
            return []

        # ---- Inference ----
        try:
            results = self._model.predict(
                source=frame,
                conf=conf,
                classes=[_PERSON_CLASS_ID],   # filter to person class only
                verbose=False,                # suppress per-frame ultralytics logs
            )
        except Exception:
            logger.exception("YOLOv9 inference failed.")
            return []

        # ---- Parse results ----
        person_detections: list[list[float]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id != _PERSON_CLASS_ID:
                    # Extra safety – should already be filtered by `classes=`
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf.item())
                person_detections.append([
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                    round(score, 4),
                ])

        logger.debug(
            "YOLOv9 produced %d person detection(s).", len(person_detections),
        )
        return person_detections

    # ------------------------------------------------------------------
    # Schema-level wrapper (used by the API layer)
    # ------------------------------------------------------------------

    async def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.25,
    ) -> list[DetectionResult]:
        """
        High-level async wrapper that returns ``DetectionResult`` schema
        objects – consumed directly by the ``/analyze`` endpoint.

        Internally delegates to :meth:`process_frame` for the raw
        inference and converts the ``[x1, y1, x2, y2, conf]`` lists
        into Pydantic models.
        """
        if self._model is None:
            await self.load_model()

        raw_detections = self.process_frame(frame, confidence_threshold)

        detections: list[DetectionResult] = []
        for det in raw_detections:
            x1, y1, x2, y2, conf = det
            detections.append(
                DetectionResult(
                    detection_class=DetectionClass.NORMAL,  # classification refined by LSTM
                    confidence=conf,
                    bounding_box=BoundingBox(
                        x_min=x1,
                        y_min=y1,
                        x_max=x2,
                        y_max=y2,
                    ),
                )
            )

        logger.debug("detect() returning %d DetectionResult(s).", len(detections))
        return detections


# ---------------------------------------------------------------------------
# Module-level singleton (uses settings for the weights path)
# ---------------------------------------------------------------------------
detector = YOLOv9Detector()
