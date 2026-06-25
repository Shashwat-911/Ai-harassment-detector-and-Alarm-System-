"""
Pydantic models for the Real-Time Harassment Detection System.

Defines request/response schemas for video frame analysis,
detection results, and alert payloads.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DetectionClass(str, Enum):
    """Possible detection categories produced by the model."""
    NORMAL = "normal"
    HARASSMENT = "harassment"
    AGGRESSION = "aggression"
    STALKING = "stalking"
    VERBAL_ABUSE = "verbal_abuse"


class AlertSeverity(str, Enum):
    """Severity tiers for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class VideoFrame(BaseModel):
    """
    Represents a single video frame sent for analysis.

    Accepts either a base64-encoded image string **or** a file path /
    object-store URI pointing to the frame.
    """
    frame_id: str = Field(
        ...,
        description="Unique identifier for this frame.",
        examples=["frame_00042"],
    )
    base64_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (JPEG / PNG).",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Server-side file path or object-store URI for the frame.",
    )
    camera_id: Optional[str] = Field(
        default=None,
        description="Identifier for the source camera / stream.",
        examples=["cam_lobby_01"],
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Capture timestamp (ISO-8601). Defaults to server time.",
    )

    @field_validator("base64_image", "file_path")
    @classmethod
    def at_least_one_source(cls, v, info):
        """Ensure the frame carries at least one image source."""
        # Validation runs per-field; full cross-field check in model_validator
        return v

    def model_post_init(self, __context) -> None:
        if self.base64_image is None and self.file_path is None:
            raise ValueError(
                "At least one of 'base64_image' or 'file_path' must be provided."
            )


# ---------------------------------------------------------------------------
# Detection / tracking results
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""
    x_min: float = Field(..., description="Left edge (px).")
    y_min: float = Field(..., description="Top edge (px).")
    x_max: float = Field(..., description="Right edge (px).")
    y_max: float = Field(..., description="Bottom edge (px).")


class DetectionResult(BaseModel):
    """
    Single object detection produced by the YOLOv9 → SORT → LSTM pipeline.

    Contains the predicted class, confidence score, bounding box,
    and an optional track ID assigned by the SORT tracker.
    """
    detection_class: DetectionClass = Field(
        ...,
        description="Predicted class label.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0 – 1).",
    )
    bounding_box: BoundingBox = Field(
        ...,
        description="Bounding box around the detected subject.",
    )
    track_id: Optional[int] = Field(
        default=None,
        description="SORT tracker ID (assigned after tracking stage).",
    )


# ---------------------------------------------------------------------------
# Analysis response
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """Full response for a single frame analysis request."""
    frame_id: str = Field(
        ...,
        description="Echo of the incoming frame identifier.",
    )
    detections: list[DetectionResult] = Field(
        default_factory=list,
        description="List of detections found in the frame.",
    )
    alert: Optional["AlertPayload"] = Field(
        default=None,
        description="Alert payload (present when harassment confidence > 85 %).",
    )
    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Server timestamp when analysis completed.",
    )


# ---------------------------------------------------------------------------
# Alert payload
# ---------------------------------------------------------------------------

class AlertPayload(BaseModel):
    """
    Structured alert emitted when the harassment confidence exceeds
    the 85 % threshold.
    """
    alert_id: str = Field(
        ...,
        description="Unique alert identifier.",
    )
    frame_id: str = Field(
        ...,
        description="Frame that triggered the alert.",
    )
    camera_id: Optional[str] = Field(
        default=None,
        description="Source camera identifier.",
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity tier.",
    )
    harassment_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate harassment confidence that triggered the alert.",
    )
    detections: list[DetectionResult] = Field(
        default_factory=list,
        description="Detections associated with the alert.",
    )
    message: str = Field(
        ...,
        description="Human-readable alert summary.",
    )
    triggered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the alert was raised.",
    )


# Rebuild forward references
AnalysisResponse.model_rebuild()
