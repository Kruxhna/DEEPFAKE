"""
Pydantic response models for the Deepfake Detection API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str = Field(description="Server status")
    device: str = Field(description="Inference device (cuda / cpu)")
    model_loaded: bool = Field(description="Whether the model checkpoint is loaded")
    cuda_available: bool
    gpu_name: Optional[str] = None
    fake_threshold: float
    suspicious_threshold: float


class Probabilities(BaseModel):
    REAL: float = Field(ge=0, le=1)
    FAKE: float = Field(ge=0, le=1)


class ImagePrediction(BaseModel):
    classification: str = Field(description="REAL or FAKE")
    confidence: float = Field(ge=0, le=1)
    probabilities: Probabilities
    face_detected: bool = Field(description="Whether a face was detected in the image")


class FramePrediction(BaseModel):
    frame: int
    classification: str
    confidence: float
    probabilities: Probabilities


class VideoPrediction(BaseModel):
    verdict: str = Field(description="REAL, SUSPICIOUS, or FAKE")
    confidence: float = Field(ge=0, le=1)
    average_fake_probability: float = Field(ge=0, le=1)
    total_frames: int
    sampled_frames: int
    frame_predictions: list[FramePrediction]


class ErrorResponse(BaseModel):
    detail: str
