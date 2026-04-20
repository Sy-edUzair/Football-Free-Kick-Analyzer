import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
import numpy as np
from ultralytics import YOLO
from app.core.config import settings
from app.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


@dataclass
class BallDetection:
    """Result of detecting the ball in a single frame."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        """Return center of bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class TrackedBall:
    """Result of tracking the ball with BotSort tracking."""

    track_id: int  # Unique ID from YOLOv11 tracking
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    frame_id: int  # Frame number

    @property
    def center(self) -> Tuple[int, int]:
        """Return center of bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class BallDetector:
    """
    Wraps YOLOv11 for sports-ball detection with BotSort tracking.

    Pipeline:
      1. YOLO detection/tracking → get bbox + track_id
      2. Use bbox center for ball position

    Usage:
        detector = BallDetector()
        detection = detector.detect(frame)   # Returns BallDetection or None
        tracked = detector.track(frame)      # Returns TrackedBall or None (with track ID)
    """

    def __init__(self):
        self._model = None
        self._frame_index = 0
        self._load_model()

    def init_video(self, video_path: str = None):
        """
        Initialize video processing for YOLO tracking.

        Args:
            video_path: Path to video file (for future use)
        """
        self._frame_index = 0
        logger.debug("Video state initialized for YOLO tracker")

    def _load_model(self):
        """Lazy-load YOLO model. Downloads on first run (~6 MB for nano)."""
        try:
            logger.info(f"Loading YOLO model: {settings.YOLO_MODEL_SIZE}")
            self._model = YOLO(settings.YOLO_MODEL_SIZE)
            logger.info("YOLO model loaded successfully.")
        except Exception as exc:
            raise ModelLoadError("Failed to load YOLO model.", detail=str(exc)) from exc

    def detect(self, frame: np.ndarray) -> Optional[BallDetection]:
        """
        Run inference on a single BGR frame and return the highest-confidence
        sports ball detection (or None if nothing found).
        """
        results = self._model(
            frame,
            verbose=False,
            conf=settings.BALL_CONFIDENCE_THRESHOLD,
            classes=[settings.BALL_CLASS_ID],  # Only look for sports ball
        )

        best: Optional[BallDetection] = None

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != settings.BALL_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det = BallDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)

                if best is None or conf > best.confidence:
                    best = det

        # Increment frame index for next frame
        self._frame_index += 1
        return best

    def track(self, frame: np.ndarray, frame_id: int = 0) -> Optional[TrackedBall]:
        """
        Run BotSort tracking on a single frame using YOLO integration.

        Returns the highest-confidence sports ball with track ID,
        or None if no ball found.

        Args:
            frame: BGR frame
            frame_id: Frame number (optional, for logging)

        Returns:
            TrackedBall with track ID, or None
        """
        tracked = self.track_batch([frame], frame_ids=[frame_id])
        return tracked[0] if tracked else None

    def track_batch(
        self, frames: List[np.ndarray], frame_ids: Optional[List[int]] = None
    ) -> List[Optional[TrackedBall]]:
        """
        Run BotSort tracking on a batch of frames.

        Batching improves GPU utilization and throughput while preserving
        tracking continuity via persist=True.
        """
        if not frames:
            return []

        if frame_ids is None:
            frame_ids = list(range(self._frame_index, self._frame_index + len(frames)))
        elif len(frame_ids) != len(frames):
            raise ValueError("frame_ids length must match frames length")

        try:
            # Use YOLO's track() method with BotSort tracker on a frame batch.
            results = self._model.track(
                frames,
                verbose=False,
                persist=True,  # Enable prediction mode: fills gaps when distant ball loses detection
                conf=settings.BALL_CONFIDENCE_THRESHOLD,
                classes=[settings.BALL_CLASS_ID],
                tracker="botsort.yaml",  # Use BotSort for better tracking accuracy
            )

            tracked_results: List[Optional[TrackedBall]] = []

            for result, frame_id in zip(results, frame_ids):
                best: Optional[TrackedBall] = None
                boxes = result.boxes
                if boxes is None:
                    tracked_results.append(None)
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != settings.BALL_CLASS_ID:
                        continue

                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    tracked = TrackedBall(
                        track_id=track_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        frame_id=frame_id,
                    )

                    if best is None or conf > best.confidence:
                        best = tracked

                tracked_results.append(best)

            self._frame_index += len(frames)
            return tracked_results

        except Exception as exc:
            logger.warning(
                f"BotSort batch tracking failed, falling back to detect(): {exc}"
            )
            detections = self.detect_batch(frames)
            tracked_results: List[Optional[TrackedBall]] = []

            for det, frame_id in zip(detections, frame_ids):
                if not det:
                    tracked_results.append(None)
                    continue

                tracked_results.append(
                    TrackedBall(
                        track_id=-1,
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                        confidence=det.confidence,
                        frame_id=frame_id,
                    )
                )

            return tracked_results

    def detect_batch(self, frames: List[np.ndarray]) -> List[Optional[BallDetection]]:
        """
        Detect balls in a list of frames.  More efficient than calling
        detect() in a loop because YOLO batches the inference.
        """
        if not frames:
            return []

        results = self._model(
            frames,
            verbose=False,
            conf=settings.BALL_CONFIDENCE_THRESHOLD,
            classes=[settings.BALL_CLASS_ID],
        )

        detections: List[Optional[BallDetection]] = []

        for result in results:
            best: Optional[BallDetection] = None
            boxes = result.boxes

            if boxes is None:
                detections.append(None)
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != settings.BALL_CLASS_ID:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det = BallDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)

                if best is None or conf > best.confidence:
                    best = det

            detections.append(best)

        self._frame_index += len(frames)
        return detections


# if __name__ == "__main__":
#     import sys
#     import cv2
#     import time
#     import math
#     from pathlib import Path
#     from app.services.video_loader import validate_and_get_info, iter_frames

#     # Configure logging
#     logging.basicConfig(level=logging.INFO)

#     # Test video path
#     test_video = Path("/content/video_109.mp4")

#     if not test_video.exists():
#         print(f"Test video not found: {test_video}")
#         print("Usage: python -m app.services.ball_detector <path_to_video>")
#         sys.exit(1)

#     try:
#         # Test 1: Initialize detector
#         print("\n=== Test 1: Initializing Ball Detector ===")
#         detector = BallDetector()
#         print("Ball detector initialized successfully")

#         # Test 2: Get video info
#         print("\n=== Test 2: Loading Video ===")
#         video_info = validate_and_get_info(str(test_video))
#         print(f"Video: {video_info.filename}")
#         print(f"  Resolution: {video_info.width}x{video_info.height}")
#         print(f"  Total frames: {video_info.total_frames}")
#         print(f"  FPS: {video_info.fps:.2f}")

#         # Test 3: Detect on sample frames (every 3rd frame)
#         print("\n=== Test 3: Ball Detection on Sample Frames ===")
#         detections_found = 0
#         detections_missed = 0
#         total_frames = 0
#         avg_inference_time = 0
#         confidences = []

#         for frame_idx, timestamp, frame in iter_frames(
#             str(test_video), sample_every_n=3
#         ):
#             start_time = time.perf_counter()
#             detection = detector.detect(frame)
#             inference_time = time.perf_counter() - start_time
#             avg_inference_time += inference_time

#             total_frames += 1

#             if detection:
#                 detections_found += 1
#                 confidences.append(detection.confidence)
#                 print(
#                     f"  Frame {frame_idx:4d}-{timestamp:6.2f}s | "
#                     f"Ball detected| Center: {detection.center} | "
#                     f"Conf: {detection.confidence:.2%} | Area: {detection.area} px²"
#                 )
#             else:
#                 detections_missed += 1
#                 if total_frames % 5 == 0:
#                     print(
#                         f"  Frame {frame_idx:4d}-{timestamp:6.2f}s | "
#                         f"Ball not detected ✗"
#                     )

#             # Limit to first 100 sampled frames for demo
#             if total_frames >= 100:
#                 break

#         avg_inference_time /= max(1, total_frames)
#         print(f"\nDetection Summary:")
#         print(f"  Frames processed: {total_frames}")
#         print(
#             f"  Detections found: {detections_found} ({100*detections_found/total_frames:.1f}%)"
#         )
#         print(f"  Detections missed: {detections_missed}")
#         if confidences:
#             print(
#                 f"  Confidence - Mean: {np.mean(confidences):.2%}, Min: {min(confidences):.2%}, Max: {max(confidences):.2%}"
#             )
#         print(f"  Avg inference time: {avg_inference_time*1000:.2f}ms per frame")

#         # Test 3b: BotSort tracking on sample frames
#         print("\n=== Test 3b: BotSort Tracking on Sample Frames ===")
#         tracked_count = 0
#         track_ids_seen = set()
#         frame_number = 0

#         for frame_idx, timestamp, frame in iter_frames(
#             str(test_video), sample_every_n=3
#         ):
#             frame_number += 1
#             tracked = detector.track(frame, frame_id=frame_idx)

#             if tracked:
#                 tracked_count += 1
#                 track_ids_seen.add(tracked.track_id)
#                 print(
#                     f"  Frame {frame_idx:4d}-{timestamp:6.2f}s | "
#                     f"Track ID: {tracked.track_id} | "
#                     f"Center: {tracked.center} | "
#                     f"Conf: {tracked.confidence:.2%}"
#                 )

#             if frame_number >= 100:
#                 break

#         print(f"\nBotSort Tracking Summary:")
#         print(f"  Frames tracked: {tracked_count}")
#         print(f"  Unique track IDs: {len(track_ids_seen)}")
#         print(f"  Track IDs seen: {sorted(track_ids_seen)}")

#         # Test 4: Save annotated video showing ball detection with trajectory
#         print("\n=== Test 4: Saving Annotated Detection Video with Trajectory ===")
#         output_dir = Path("outputs")
#         output_dir.mkdir(parents=True, exist_ok=True)

#         output_video = output_dir / f"ball_detection_{video_info.filename}"

#         # Create video writer with actual video dimensions
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         video_dims = (video_info.width, video_info.height)
#         writer = cv2.VideoWriter(str(output_video), fourcc, video_info.fps, video_dims)

#         if not writer.isOpened():
#             print(f"Cannot create video writer for {output_video}")
#         else:
#             frame_count = 0
#             for frame_idx, timestamp, frame in iter_frames(
#                 str(test_video), sample_every_n=1
#             ):
#                 tracked = detector.track(frame, frame_id=frame_idx)

#                 # Annotate frame
#                 annotated = frame.copy()
#                 if tracked:
#                     # Draw bounding box (orange)
#                     cv2.rectangle(
#                         annotated,
#                         (tracked.x1, tracked.y1),
#                         (tracked.x2, tracked.y2),
#                         (0, 165, 255),
#                         2,
#                     )
#                     # Draw center point (green)
#                     cv2.circle(annotated, tracked.center, 5, (0, 255, 0), -1)
#                     # Draw confidence and track ID label
#                     label = f"ID:{tracked.track_id} Ball {tracked.confidence:.0%}"
#                     cv2.putText(
#                         annotated,
#                         label,
#                         (tracked.x1, tracked.y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7,
#                         (0, 165, 255),
#                         2,
#                     )
#                 else:
#                     cv2.putText(
#                         annotated,
#                         "No ball detected",
#                         (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         (0, 0, 255),
#                         2,
#                     )

#                 # Add timestamp and frame info
#                 cv2.putText(
#                     annotated,
#                     f"Frame {frame_idx} | {timestamp:.2f}s",
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (255, 255, 255),
#                     1,
#                 )

#                 # Write frame to video
#                 writer.write(annotated)
#                 frame_count += 1

#                 if frame_count % 100 == 0:
#                     print(f"  Processed {frame_count} frames...")

#                 # Limit for demo (process full video)
#                 if frame_count >= video_info.total_frames:
#                     break

#             writer.release()
#             output_size_mb = output_video.stat().st_size / (1024 * 1024)
#             print(f"  Ball detection video saved: {output_video}")
#             print(f"  Total frames: {frame_count}")
#             print(f"  Duration: {frame_count/video_info.fps:.2f}s")
#             print(f"  File size: {output_size_mb:.2f}MB")

#         print("\nAll ball detector tests passed!")

#     except Exception as e:
#         print(f"\nTest failed: {e}")
#         import traceback

#         traceback.print_exc()
#         sys.exit(1)
