from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    APP_PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)

    OUTPUT_DIR: str = Field(default="outputs")
    TEMP_DIR: str = Field(default="temp")
    MAX_VIDEO_SIZE_MB: int = Field(default=500)

    # Frames per second to sample when detecting kicks
    PROCESSING_FPS: int = Field(default=30)
    # Seconds before kick to include in clip
    CLIP_PRE_KICK_SECONDS: float = Field(default=1.5)
    # Seconds after kick to include in clip
    CLIP_POST_KICK_SECONDS: float = Field(default=2.5)

    YOLO_MODEL_SIZE: str = Field(default="yolo11m.pt")
    # Minimum YOLO confidence to accept a ball detection
    BALL_CONFIDENCE_THRESHOLD: float = Field(default=0.3)
    # Minimum ball bounding box area (pixels²) to accept detection
    MIN_BALL_AREA: int = Field(default=20)
    # COCO class ID for sports ball
    BALL_CLASS_ID: int = Field(default=32)

    # Minimum frames between two separate kick events
    MIN_FRAMES_BETWEEN_KICKS: int = Field(default=120)  # 4 seconds at 30fps
    # How many consecutive frames ball must be missing to trigger "kick" detection
    BALL_DISAPPEAR_FRAMES: int = Field(default=3)
    # Minimum ball displacement (pixels) to count as a kick
    MIN_KICK_DISPLACEMENT: float = Field(
        default=80.0
    )  # Raised from 40 - running stride too close
    ENABLE_KICK_REFINEMENT: bool = Field(default=False)
    # Only flag disappearance as kick if last detection had confidence > this threshold
    # This prevents false kicks from detection dropout
    MIN_DETECTION_CONFIDENCE_FOR_KICK: float = Field(default=0.5)
    # Enable temporal interpolation: treat brief detection gaps as continuous if trajectory is smooth
    ENABLE_TEMPORAL_INTERPOLATION: bool = Field(default=False)

    # Minimum ball velocity (pixels/frame) to trigger kick signal
    MIN_KICK_VELOCITY: float = Field(
        default=200.0
    )  # Foot-first algorithm requires strong ball response
    # Minimum ball acceleration (pixels/frame²) to trigger kick signal
    MIN_KICK_ACCELERATION: float = Field(
        default=500.0
    )  # Foot-first algorithm with physics validation
    # Minimum player foot velocity (pixels/frame) to validate kicking motion
    MIN_FOOT_VELOCITY_FOR_KICK: float = Field(
        default=100.0
    )  # Foot-first algorithm detects acceleration spikes
    # Maximum distance from foot to ball (pixels) for kick validation
    MAX_FOOT_TO_BALL_DISTANCE: float = Field(default=50.0)  # Strict foot proximity

    MEDIAPIPE_POSE_MODEL_PATH: str = Field(default="pose_landmarker_full.task")
    POSE_MIN_DETECTION_CONFIDENCE: float = Field(default=0.5)
    POSE_MIN_TRACKING_CONFIDENCE: float = Field(default=0.5)
    ENABLE_BALL_TRACKING: bool = Field(default=True)
    # Number of frames to batch per YOLO track() call
    BALL_TRACK_BATCH_SIZE: int = Field(default=8)

    # BGR colours
    KEYPOINT_COLOR: tuple = Field(default=(0, 255, 0))  # Green
    SKELETON_COLOR: tuple = Field(default=(255, 255, 0))  # Cyan-yellow
    BALL_BOX_COLOR: tuple = Field(default=(0, 165, 255))  # Orange
    TRAJECTORY_COLOR: tuple = Field(default=(0, 0, 255))  # Red
    TEXT_COLOR: tuple = Field(default=(255, 255, 255))  # White
    # Max trajectory trail length in frames
    TRAJECTORY_MAX_POINTS: int = Field(default=30)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = Settings()
