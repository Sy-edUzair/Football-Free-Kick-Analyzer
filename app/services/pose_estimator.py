"""
Pose Estimation Service — uses MediaPipe PoseLandmarker to detect player body keypoints.

What are keypoints?
  MediaPipe detects 33 landmarks on the human body (nose, shoulders, elbows,
  wrists, hips, knees, ankles, etc.).  Each landmark has (x, y, z, visibility, presence).
  x and y are normalised 0–1 relative to frame size, so we multiply by width/height
  to get pixel coordinates.
  z represents depth (with hip midpoint as origin).

Why MediaPipe PoseLandmarker?
  - Runs fast on CPU (designed for mobile/edge)
  - No GPU required
  - Pre-trained, no fine-tuning needed
  - Modern tasks API (2024+)
  - Provides both normalized and world coordinates
  - Free and open source

Model Setup:
  Download the pose_landmarker.task file from:
  https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models

  Set MEDIAPIPE_POSE_MODEL_PATH in config or place in working directory.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.core.config import settings
from app.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

# MediaPipe Pose connections — pairs of landmark indices that form skeleton lines
# Each tuple is (landmark_a, landmark_b)
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # Arms
    (11, 23),
    (12, 24),
    (23, 24),  # Torso
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),  # Left leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),  # Right leg
]


@dataclass
class Keypoint:
    """A single body landmark in pixel coordinates."""

    index: int
    x: int
    y: int
    z: float  # Depth with hip midpoint as origin
    visibility: float  # 0.0 = not visible, 1.0 = clearly visible
    presence: float  # Confidence that landmark is present (new in PoseLandmarker)


@dataclass
class PoseDetection:
    """All keypoints detected for one person in one frame."""

    keypoints: List[Keypoint] = field(default_factory=list)
    _keypoint_by_index: Dict[int, Keypoint] = field(init=False, repr=False)

    def __post_init__(self):
        # Build O(1) lookup map once for repeated keypoint access.
        self._keypoint_by_index = {kp.index: kp for kp in self.keypoints}

    def get_keypoint(self, index: int) -> Optional[Keypoint]:
        """Retrieve a keypoint by its MediaPipe landmark index."""
        return self._keypoint_by_index.get(index)

    @property
    def visible_count(self) -> int:
        return sum(1 for kp in self.keypoints if kp.visibility > 0.5)


class PoseEstimator:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API) for player skeleton detection.

    This uses the modern MediaPipe Tasks API (2024+) which provides better
    performance and more features than the legacy mp.solutions.pose.

    Usage:
        estimator = PoseEstimator()
        pose = estimator.detect(frame)      # Returns PoseDetection or None
        pose = estimator.detect_video(frame, timestamp_ms)  # For video mode
    """

    def __init__(self):
        self._landmarker = None
        self._last_timestamp_ms = -1
        self._load_model()

    def _load_model(self):
        """Initialize MediaPipe PoseLandmarker with Tasks API."""
        try:
            logger.info("Loading MediaPipe PoseLandmarker model...")

            # Get model path from config or use default
            model_path = getattr(settings, "MEDIAPIPE_POSE_MODEL_PATH", None)

            if not model_path:
                logger.info(
                    "MEDIAPIPE_POSE_MODEL_PATH not set. "
                    "Please download pose_landmarker.task from "
                    "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models"
                )
                raise ModelLoadError(
                    "PoseLandmarker model path not configured.",
                    detail="Set MEDIAPIPE_POSE_MODEL_PATH in config",
                )

            # Create base options with model path
            base_options = python.BaseOptions(model_asset_path=model_path)

            # Create pose landmarker options for VIDEO mode
            # VIDEO mode is best for frame-by-frame processing with timestamps
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,  # We expect single person (player)
                min_pose_detection_confidence=settings.POSE_MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=settings.POSE_MIN_TRACKING_CONFIDENCE,
                min_tracking_confidence=settings.POSE_MIN_TRACKING_CONFIDENCE,
                output_segmentation_masks=False,  # Not needed for kick detection
            )

            # Create the landmarker
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info("MediaPipe PoseLandmarker loaded successfully.")

        except Exception as exc:
            if isinstance(exc, ModelLoadError):
                raise
            raise ModelLoadError(
                "Failed to load MediaPipe PoseLandmarker model.", detail=str(exc)
            ) from exc

    def detect(
        self, frame: np.ndarray, timestamp_ms: int = 0
    ) -> Optional[PoseDetection]:
        """
        Detect pose keypoints in a BGR frame (VIDEO mode with timestamp).

        Args:
            frame: BGR frame from OpenCV
            timestamp_ms: Frame timestamp in milliseconds. Important for tracking.
                          For consistent tracking, provide monotonically increasing timestamps.

        Returns:
            PoseDetection if a person is found, None otherwise.
        """
        if not self._landmarker:
            return None

        try:
            # MediaPipe VIDEO mode requires strictly increasing timestamps.
            # Clamp duplicates/regressions to preserve tracker stability.
            if timestamp_ms <= self._last_timestamp_ms:
                logger.debug(
                    "Non-monotonic pose timestamp received (new=%d, last=%d); clamping.",
                    timestamp_ms,
                    self._last_timestamp_ms,
                )
                timestamp_ms = self._last_timestamp_ms + 1

            # Convert BGR to RGB for MediaPipe
            rgb = self._bgr_to_rgb(frame)

            # Create MediaPipe Image object from numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect poses with timestamp for video mode
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            self._last_timestamp_ms = timestamp_ms

            # Process results (handle multiple poses, but we use first one)
            return self._process_result(result, frame.shape[:2])

        except Exception as exc:
            logger.warning(f"Pose detection failed: {exc}")
            return None

    def detect_image(self, frame: np.ndarray) -> Optional[PoseDetection]:
        """
        Detect pose keypoints in a BGR frame (IMAGE mode, for single frames).

        Slower than detect() but doesn't require timestamps.
        Use this for batch processing or one-off detections.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            PoseDetection if a person is found, None otherwise.
        """
        if not self._landmarker:
            return None

        try:
            # Convert BGR to RGB
            rgb = self._bgr_to_rgb(frame)

            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect poses (IMAGE mode, no timestamp needed)
            result = self._landmarker.detect(mp_image)

            return self._process_result(result, frame.shape[:2])

        except Exception as exc:
            logger.warning(f"Pose detection failed: {exc}")
            return None

    def _bgr_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB (required by MediaPipe)."""
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _process_result(
        self, result, frame_shape: Tuple[int, int]
    ) -> Optional[PoseDetection]:
        """
        Process PoseLandmarkerResult and convert to PoseDetection.

        Args:
            result: PoseLandmarkerResult from landmarker.detect()
            frame_shape: (height, width) of the frame

        Returns:
            PoseDetection or None if no poses detected
        """
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # Use first detected pose (we expect single person)
        landmarks = result.pose_landmarks[0]
        h, w = frame_shape
        keypoints = []

        for idx, lm in enumerate(landmarks):
            keypoints.append(
                Keypoint(
                    index=idx,
                    x=int(lm.x * w),
                    y=int(lm.y * h),
                    z=float(lm.z),
                    visibility=float(lm.visibility),
                    presence=float(lm.presence),
                )
            )

        return PoseDetection(keypoints=keypoints)

    def close(self):
        """Release MediaPipe resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
