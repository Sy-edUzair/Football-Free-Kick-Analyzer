"""
Clip Extractor — splits the original video into individual kick clips
and annotates each frame with pose + ball overlay.

For each detected kick:
  1. Calculate clip window: [kick_time - PRE, kick_time + POST]
  2. Seek to start frame in original video
  3. For each frame in window: run detection + annotate + write to new video
  4. Save to disk and return ClipDetail metadata
"""

import cv2
import logging
import os
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from app.core.config import settings
from app.models.schemas import ClipDetail
from app.services.video_loader import VideoInfo
from app.services.ball_detector import BallDetector
from app.services.pose_estimator import PoseEstimator
from app.services.annotator import FrameAnnotator
from app.services.kick_detector import DetectedKick
from app.core.exceptions import ClipExtractionError

logger = logging.getLogger(__name__)


class ClipExtractor:
    """
    Generates an annotated video clip for each detected kick.

    One instance is reused for all kicks in a video — the detectors
    are expensive to load, so we create them once and pass them in.
    """

    def __init__(
        self,
        ball_detector: BallDetector,
        pose_estimator: PoseEstimator,
        output_dir: str = None,
    ):
        self._ball_detector = ball_detector
        self._pose_estimator = pose_estimator
        self._output_dir = output_dir or settings.CLIPS_DIR

    def extract_all(
        self,
        video_info: VideoInfo,
        kicks: List[DetectedKick],
    ) -> List[ClipDetail]:
        """
        Extract and annotate one clip per kick.

        Returns list of ClipDetail, one per kick, in order.
        """
        os.makedirs(self._output_dir, exist_ok=True)
        clip_details: List[ClipDetail] = []

        for kick in kicks:
            logger.info(
                f"Extracting clip for kick #{kick.kick_index} @ {kick.timestamp_seconds:.2f}s"
            )
            try:
                detail = self._extract_single_clip(video_info, kick)
                clip_details.append(detail)
            except Exception as exc:
                logger.error(
                    f"Failed to extract clip for kick #{kick.kick_index}: {exc}"
                )
                raise ClipExtractionError(
                    f"Clip extraction failed for kick #{kick.kick_index}",
                    detail=str(exc),
                ) from exc

        return clip_details

    def _extract_single_clip(
        self,
        video_info: VideoInfo,
        kick: DetectedKick,
    ) -> ClipDetail:
        """Extract, annotate, and save one clip. Returns its metadata."""
        fps = video_info.fps
        total_frames = video_info.total_frames

        # Calculate frame window
        pre_frames = int(settings.CLIP_PRE_KICK_SECONDS * fps)
        post_frames = int(settings.CLIP_POST_KICK_SECONDS * fps)

        start_frame = max(0, kick.frame_number - pre_frames)
        end_frame = min(total_frames - 1, kick.frame_number + post_frames)

        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time

        # Setup video writer
        clip_filename = f"kick_{kick.kick_index:02d}_{uuid.uuid4().hex[:8]}.mp4"
        clip_path = os.path.join(self._output_dir, clip_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            clip_path, fourcc, fps, (video_info.width, video_info.height)
        )

        if not writer.isOpened():
            raise ClipExtractionError(f"Cannot open VideoWriter for {clip_path}")

        # Annotator (fresh per clip — resets trajectory trail)
        annotator = FrameAnnotator(kick_index=kick.kick_index)

        # Read + annotate + write frames
        cap = cv2.VideoCapture(video_info.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        ball_detections = 0
        pose_detections = 0

        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Unexpected end of video at frame {frame_idx}")
                break

            timestamp = frame_idx / fps

            # Run detectors
            ball = self._ball_detector.detect(frame)
            pose = self._pose_estimator.detect(frame)

            if ball:
                ball_detections += 1
            if pose:
                pose_detections += 1

            # Annotate and write
            annotated = annotator.annotate(frame, ball, pose, timestamp)
            writer.write(annotated)
            frame_count += 1

        cap.release()
        writer.release()

        logger.info(
            f"  Clip saved: {clip_filename} | "
            f"{frame_count} frames | "
            f"ball={ball_detections} | pose={pose_detections}"
        )

        return ClipDetail(
            kick_index=kick.kick_index,
            clip_filename=clip_filename,
            clip_path=clip_path,
            start_timestamp=round(start_time, 3),
            end_timestamp=round(end_time, 3),
            duration_seconds=round(duration, 3),
            kick_timestamp=round(kick.timestamp_seconds, 3),
            frame_count=frame_count,
            ball_detections=ball_detections,
            pose_detections=pose_detections,
        )
