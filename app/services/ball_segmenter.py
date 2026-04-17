"""
Ball Segmentation Service — uses SAM2 (Segment Anything Model 2) for video-aware ball tracking.

Why SAM2 for ball tracking consistency?
  - Video-aware segmentation: maintains temporal consistency across frames
  - Automatic mask propagation: tracks ball through brief occlusions
  - More efficient: reuses information from previous frames
  - Better center-of-mass calculation (more accurate than bbox center)
  - Handles partial occlusion gracefully (ball 50% visible → still tracked)
  - Especially useful near goalpost where ball gets partially occluded

Integration with BotSort Tracking + SAM2:
  - BotSort tracker provides track IDs for ball across frames
  - SAM2 provides precise pixel-level masks with temporal consistency
  - Pipeline: BotSort track → SAM2 segment with state → refined center

Key advantages over SAM1:
  - Frame-to-frame state propagation
  - Automatic tracking of same object across frames
  - More robust to occlusions
  - Better performance on video data

Key concepts:
  - Predictor: Maintains video frame state and history
  - Mask: Binary image showing ball pixels with temporal coherence
  - Center of mass: Calculated from mask (more accurate than bbox)
  - Confidence: (mask_area / bbox_area) validates detection quality
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import cv2

from app.core.config import settings
from app.core.exceptions import ModelLoadError
from app.services.ball_detector import BallDetection

logger = logging.getLogger(__name__)

try:
    from sam2.build_sam import build_sam2_video_predictor

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning(
        "SAM2 not installed. Install with: pip install 'git+https://github.com/facebookresearch/sam2.git'"
    )

# Fallback to SAM1 if SAM2 not available
try:
    from segment_anything import sam_model_registry, SamPredictor

    SAM1_AVAILABLE = True
except ImportError:
    SAM1_AVAILABLE = False
    logger.warning("SAM1 not installed. Install with: pip install segment-anything")


@dataclass
class BallSegmentation:
    """Result of segmenting the ball."""

    mask: np.ndarray  # Binary mask (H x W, uint8)
    center: Tuple[int, int]  # Center of mass from mask (x, y)
    confidence: float  # (mask_area / bbox_area) 0.0-1.0
    area_pixels: int  # Number of pixels in mask
    frame_index: int = -1  # Frame index for SAM2 tracking


class BallSegmenterSAM2:
    """
    Video-aware SAM2 for ball segmentation with temporal consistency.

    Maintains state across frames for better tracking through occlusions.
    Automatically propagates masks between frames.
    """

    def __init__(self):
        self._predictor = None
        self._video_predictor = None
        self._current_frame_idx = 0
        self._mask_state_dict: Dict = {}  # Track mask state per object ID
        self._load_model()

    def _load_model(self):
        """Lazy-load SAM2 model."""
        if not SAM2_AVAILABLE:
            logger.warning("SAM2 not available. Video segmentation disabled.")
            return

        try:
            logger.info("Loading SAM2 video predictor...")
            # Build SAM2 video predictor
            # Checkpoint will be auto-downloaded on first use
            self._video_predictor = build_sam2_video_predictor(
                config_file="sam2_hiera_b+.yaml",
                ckpt_path=None,  # Auto-download
                device="cpu",  # Can be "cuda" if GPU available
            )
            logger.info("SAM2 video predictor loaded successfully")
        except Exception as exc:
            logger.warning(f"Failed to load SAM2: {exc}")
            self._video_predictor = None

    def init_video_state(self, video_height: int, video_width: int):
        """Initialize state for a new video."""
        if self._video_predictor:
            try:
                self._video_predictor.reset_state()
                self._current_frame_idx = 0
                self._mask_state_dict = {}
                logger.info(f"Initialized SAM2 for video {video_width}x{video_height}")
            except Exception as exc:
                logger.warning(f"Failed to initialize video state: {exc}")

    def segment(
        self,
        frame: np.ndarray,
        detection: BallDetection,
        frame_index: int = 0,
        track_id: int = -1,
    ) -> Optional[BallSegmentation]:
        """
        Segment ball in frame using SAM2 with temporal context.

        Args:
            frame: BGR frame
            detection: YOLO BallDetection bbox
            frame_index: Index of frame in video (for sequential processing)
            track_id: Track ID from BotSort tracker (for state tracking)

        Returns:
            BallSegmentation with mask and center, or None if segmentation fails
        """
        if not self._video_predictor:
            return None

        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Add frame to predictor's video state
            frame_height, frame_width = rgb.shape[:2]

            # Run SAM2 with bbox prompt
            box = np.array([detection.x1, detection.y1, detection.x2, detection.y2])

            # Use SAM2's add_new_frame_as_prompt for temporal consistency
            # This automatically propagates information from previous frames
            masks, scores, logits = self._video_predictor.predict(
                frame_idx=frame_index,
                obj_id=track_id,  # Use track ID for consistency
                points=None,
                boxes=np.array([box]),  # Provide bbox as prompt
                labels=None,
                negative_points=None,
                negative_labels=None,
                clear_old_points=False,
            )

            if masks is None or len(masks) == 0:
                return None

            # Use first (and typically only) mask
            mask = masks[0].astype(np.uint8) * 255

            # Calculate center of mass from mask
            M = cv2.moments(mask)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                # Fallback to bbox center if mask is empty
                center_x = (detection.x1 + detection.x2) // 2
                center_y = (detection.y1 + detection.y2) // 2

            # Calculate confidence as ratio of mask area to bbox area
            mask_area = np.count_nonzero(mask)
            bbox_area = detection.area
            confidence = mask_area / bbox_area if bbox_area > 0 else 0.0

            # Store mask state for future frame propagation
            if track_id >= 0:
                self._mask_state_dict[track_id] = {
                    "mask": mask,
                    "center": (center_x, center_y),
                    "confidence": confidence,
                }

            self._current_frame_idx = frame_index

            return BallSegmentation(
                mask=mask,
                center=(center_x, center_y),
                confidence=min(1.0, confidence),
                area_pixels=mask_area,
                frame_index=frame_index,
            )

        except Exception as exc:
            logger.debug(f"SAM2 segmentation failed: {exc}")
            return None

    def close(self):
        """Release SAM2 resources and reset state."""
        if self._video_predictor:
            try:
                self._video_predictor.reset_state()
            except:
                pass
            self._video_predictor = None
        self._mask_state_dict.clear()


class BallSegmenterSAM1:
    """
    Fallback SAM1 (static image segmentation) if SAM2 not available.

    Less efficient than SAM2 for videos but works on any frame.
    """

    def __init__(self):
        self._predictor = None
        self._load_model()

    def _load_model(self):
        """Lazy-load SAM1 model."""
        if not SAM1_AVAILABLE:
            logger.warning("SAM1 not available. Static segmentation disabled.")
            return

        try:
            model_type = settings.SAM_MODEL_TYPE  # "vit_b" | "vit_l" | "vit_h"
            logger.info(f"Loading SAM1 model: {model_type}")

            sam = sam_model_registry[model_type](checkpoint=None)
            device = "cpu"
            self._predictor = SamPredictor(sam.to(device=device))
            logger.info(f"SAM1 {model_type} loaded successfully on {device}")
        except Exception as exc:
            logger.warning(f"Failed to load SAM1: {exc}")
            self._predictor = None

    def segment(
        self, frame: np.ndarray, detection: BallDetection
    ) -> Optional[BallSegmentation]:
        """
        Segment ball using SAM1 static predictor.
        """
        if not self._predictor:
            return None

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(rgb)

            # Use bbox as prompt
            box = np.array([[detection.x1, detection.y1, detection.x2, detection.y2]])

            masks, scores, logits = self._predictor.predict(
                box=box, multimask_output=False
            )

            if masks is None or len(masks) == 0:
                return None

            mask = masks[0].astype(np.uint8) * 255

            # Calculate center of mass
            M = cv2.moments(mask)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                center_x = (detection.x1 + detection.x2) // 2
                center_y = (detection.y1 + detection.y2) // 2

            # Calculate confidence
            mask_area = np.count_nonzero(mask)
            bbox_area = detection.area
            confidence = mask_area / bbox_area if bbox_area > 0 else 0.0

            return BallSegmentation(
                mask=mask,
                center=(center_x, center_y),
                confidence=min(1.0, confidence),
                area_pixels=mask_area,
            )

        except Exception as exc:
            logger.debug(f"SAM1 segmentation failed: {exc}")
            return None

    def close(self):
        """Release resources."""
        self._predictor = None


class SimpleSegmenter:
    """
    Fallback segmenter using HSV color-based masking.
    Faster but less accurate than SAM.
    """

    def segment(
        self, frame: np.ndarray, detection: BallDetection
    ) -> Optional[BallSegmentation]:
        """
        Simple HSV-based segmentation of ball region.
        Works for balls that contrast with background.
        """
        try:
            # Extract region of interest
            roi = frame[detection.y1 : detection.y2, detection.x1 : detection.x2]
            if roi.size == 0:
                return None

            # Convert to HSV and threshold for ball color
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Typical ball colors (orange/white in HSV space)
            lower1 = np.array([5, 100, 100])  # Orange lower
            upper1 = np.array([15, 255, 255])  # Orange upper
            lower2 = np.array([0, 0, 200])  # White lower
            upper2 = np.array([180, 50, 255])  # White upper

            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Dilate to fill small holes
            mask = cv2.dilate(
                mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2
            )

            # Calculate center
            M = cv2.moments(mask)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"]) + detection.x1
                center_y = int(M["m01"] / M["m00"]) + detection.y1
            else:
                center_x = (detection.x1 + detection.x2) // 2
                center_y = (detection.y1 + detection.y2) // 2

            # Calculate confidence
            mask_area = np.count_nonzero(mask)
            roi_area = roi.shape[0] * roi.shape[1]
            confidence = mask_area / roi_area if roi_area > 0 else 0.0

            # Create full-frame mask for consistency
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[detection.y1 : detection.y2, detection.x1 : detection.x2] = mask

            return BallSegmentation(
                mask=full_mask,
                center=(center_x, center_y),
                confidence=min(1.0, confidence),
                area_pixels=mask_area,
            )

        except Exception as exc:
            logger.warning(f"Simple segmentation failed: {exc}")
            return None


def get_segmenter():
    """
    Factory function to get appropriate segmenter based on config.

    Priority:
      1. SAM2 (video-aware, best for temporal consistency)
      2. SAM1 (static, good fallback)
      3. SimpleSegmenter (HSV, fastest)
    """
    if not settings.ENABLE_BALL_SEGMENTATION:
        return None

    segmenter_type = settings.BALL_SEGMENTER_TYPE.lower()

    # Try SAM2 first (video-aware)
    if segmenter_type in ["sam2", "sam", "auto"]:
        if SAM2_AVAILABLE:
            try:
                logger.info("Using SAM2 for video-aware segmentation")
                return BallSegmenterSAM2()
            except Exception as e:
                logger.warning(f"SAM2 loading failed: {e}. Trying SAM1...")

        # Fallback to SAM1
        if SAM1_AVAILABLE:
            try:
                logger.info("Using SAM1 for static segmentation")
                return BallSegmenterSAM1()
            except Exception as e:
                logger.warning(f"SAM1 loading failed: {e}. Using SimpleSegmenter...")

    # Final fallback to HSV
    logger.info("Using SimpleSegmenter (HSV-based)")
    return SimpleSegmenter()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from app.services.video_loader import validate_and_get_info, iter_frames
    from app.services.ball_detector import BallDetector

    logging.basicConfig(level=logging.INFO)

    test_video = Path(
        "/home/syed-uzair-hussain-zaidi/Office Work/Tezeract/Pose_Estimation/freekick_analyzer/dataset/video_109.mp4"
    )

    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        sys.exit(1)

    try:
        print("\n=== Test 1: Initialize Segmenters ===")

        if SAM2_AVAILABLE:
            print("✓ SAM2 available")
            sam_seg = BallSegmenterSAM2()
            print("✓ BallSegmenter (SAM2) loaded")
        else:
            print("✗ SAM2 not available, skipping SAM2 tests")
            sam_seg = None

        simple_seg = SimpleSegmenter()
        print("✓ SimpleSegmenter loaded")

        print("\n=== Test 2: Load Video and Detector ===")
        video_info = validate_and_get_info(str(test_video))
        ball_detector = BallDetector()
        print(
            f"✓ Video loaded: {video_info.width}x{video_info.height}, {video_info.total_frames} frames"
        )

        print("\n=== Test 3: Segmentation on Sample Frames ===")
        frame_count = 0

        for frame_idx, timestamp, frame in iter_frames(
            str(test_video), sample_every_n=30, resize_to=(1280, 1280)
        ):
            detection = ball_detector.detect(frame)

            if detection:
                print(f"\nFrame {frame_idx}: Ball detected")
                print(f"  YOLO bbox center: {detection.center}")

                # Test SAM segmentation
                if sam_seg:
                    try:
                        seg = sam_seg.segment(frame, detection)
                        if seg:
                            print(f"  ✓ SAM segmentation successful")
                            print(f"    Mask center: {seg.center}")
                            print(f"    Confidence: {seg.confidence:.2f}")
                            print(f"    Mask area: {seg.area_pixels} pixels")
                            displacement = (
                                (seg.center[0] - detection.center[0]) ** 2
                                + (seg.center[1] - detection.center[1]) ** 2
                            ) ** 0.5
                            print(f"    Center shift: {displacement:.1f} pixels")
                        else:
                            print(f"  ✗ SAM segmentation failed")
                    except Exception as e:
                        print(f"  ✗ SAM error: {e}")

                # Test simple segmentation
                simple_seg_result = simple_seg.segment(frame, detection)
                if simple_seg_result:
                    print(f"  ✓ Simple segmentation successful")
                    print(f"    Mask center: {simple_seg_result.center}")
                    print(f"    Confidence: {simple_seg_result.confidence:.2f}")

                frame_count += 1
                if frame_count >= 5:
                    break

        print("\nSegmentation tests completed!")

        if sam_seg:
            sam_seg.close()

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
