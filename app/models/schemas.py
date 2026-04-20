"""
Pydantic models that define the shape of all API responses.
Think of these as 'contracts' — the API guarantees this exact structure.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class BoundingBox(BaseModel):
    """Pixel coordinates of a detected object."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = Field(ge=0.0, le=1.0)


class KickEvent(BaseModel):
    """Metadata about a single detected kick event."""

    kick_index: int = Field(description="1-based kick number")
    frame_number: int = Field(description="Frame where kick was detected")
    timestamp_seconds: float = Field(description="Time in video (seconds)")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="How confident the system is this was a real kick"
    )


class ClipDetail(BaseModel):
    """All metadata for one generated clip."""

    kick_index: int
    clip_filename: str
    clip_path: str
    start_timestamp: float = Field(description="Clip start time (seconds)")
    end_timestamp: float = Field(description="Clip end time (seconds)")
    duration_seconds: float
    kick_timestamp: float = Field(
        description="When kick happened in original video (seconds)"
    )
    frame_count: int
    ball_detections: int = Field(description="Number of frames with ball detected")
    pose_detections: int = Field(description="Number of frames with pose detected")


class VideoMetadata(BaseModel):
    """Basic technical info about the uploaded video."""

    filename: str
    duration_seconds: float
    fps: float
    total_frames: int
    width: int
    height: int
    file_size_mb: float


class AnalysisResponse(BaseModel):
    """
    The main API response.
    Contains all results from analyzing a free-kick video.
    """

    success: bool
    message: str
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    video_metadata: VideoMetadata
    total_kicks_detected: int
    kick_events: List[KickEvent]
    clips: List[ClipDetail]
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    """Returned when something goes wrong."""

    success: bool = False
    error_code: str
    message: str
    detail: Optional[str] = None
