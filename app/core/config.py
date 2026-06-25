"""
Application configuration loaded from environment variables.

Uses pydantic-settings so values can be overridden via a `.env` file
or exported shell variables.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    # ---- General ----
    APP_NAME: str = "Harassment Detection API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ---- Server ----
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ---- Detection thresholds ----
    HARASSMENT_ALERT_THRESHOLD: float = Field(
        default=0.85,
        description="Confidence above which an alert is triggered.",
    )

    # ---- Model paths (placeholders) ----
    YOLO_WEIGHTS_PATH: str = "weights/yolov9.pt"
    LSTM_WEIGHTS_PATH: str = "weights/lstm_classifier.pt"
    ONNX_WEIGHTS_PATH: str = "weights/harassment_model.onnx"

    # ---- CORS ----
    ALLOWED_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton settings instance
settings = Settings()
