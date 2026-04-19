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
import math
from typing import List, Optional

from app.core.config import settings
from app.models.schemas import ClipDetail
from app.services.video_loader import VideoInfo
from app.services.ball_detector import BallDetector
from app.services.pose_estimator import PoseEstimator
from app.services.annotator import FrameAnnotator
from app.services.kick_detector import DetectedKick, FrameBallState
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
        prev_state: Optional[FrameBallState] = None

        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Unexpected end of video at frame {frame_idx}")
                break

            timestamp = frame_idx / fps

            # Run detectors
            ball = self._ball_detector.detect(frame)
            pose = self._pose_estimator.detect(frame, timestamp_ms=int(timestamp * 1000))

            if ball:
                ball_detections += 1
            if pose:
                pose_detections += 1

            # Create current state
            state = FrameBallState(
                frame_number=frame_idx,
                timestamp=timestamp,
                detection=ball,
                pose=pose,
            )

            # Compute metrics using FrameBallState pattern from kick_detector
            metrics = {}
            
            # Ball motion metrics
            if prev_state and ball and prev_state.detection:
                dt = state.timestamp - prev_state.timestamp
                if dt > 0:
                    dx = ball.center[0] - prev_state.detection.center[0]
                    dy = ball.center[1] - prev_state.detection.center[1]
                    displacement = math.hypot(dx, dy)
                    state.velocity = displacement / dt
                    metrics["ball_velocity"] = state.velocity

            # Foot metrics
            if pose and ball:
                ball_cx, ball_cy = ball.center
                # Collect visible feet (MediaPipe: 31=left foot, 32=right foot)
                feet = []
                for kp_idx in (31, 32):
                    kp = pose.get_keypoint(kp_idx)
                    if kp and kp.visibility > 0.5:
                        feet.append((kp.x, kp.y))

                if feet:
                    dists = [math.hypot(fx - ball_cx, fy - ball_cy) for fx, fy in feet]
                    min_dist = min(dists)
                    closest_foot = feet[dists.index(min_dist)]
                    state.foot_distance_to_ball = min_dist
                    state.foot_screen_pos = closest_foot
                    metrics["displacement_from_foot"] = min_dist

                    # Foot velocity
                    if prev_state and prev_state.foot_screen_pos:
                        dt = state.timestamp - prev_state.timestamp
                        if dt > 0:
                            foot_dx = closest_foot[0] - prev_state.foot_screen_pos[0]
                            foot_dy = closest_foot[1] - prev_state.foot_screen_pos[1]
                            state.foot_velocity = math.hypot(foot_dx, foot_dy) / dt
                            metrics["foot_velocity"] = state.foot_velocity

                # Leg angle from hip-knee-ankle
                angles = []
                hip_l = pose.get_keypoint(23)
                knee_l = pose.get_keypoint(25)
                ankle_l = pose.get_keypoint(27)
                if hip_l and knee_l and ankle_l and all(kp.visibility > 0.5 for kp in [hip_l, knee_l, ankle_l]):
                    v1 = (hip_l.x - knee_l.x, hip_l.y - knee_l.y)
                    v2 = (ankle_l.x - knee_l.x, ankle_l.y - knee_l.y)
                    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (
                        math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1]) + 1e-6
                    )
                    cos_angle = max(-1, min(1, cos_angle))
                    angles.append(math.degrees(math.acos(cos_angle)))

                hip_r = pose.get_keypoint(24)
                knee_r = pose.get_keypoint(26)
                ankle_r = pose.get_keypoint(28)
                if hip_r and knee_r and ankle_r and all(kp.visibility > 0.5 for kp in [hip_r, knee_r, ankle_r]):
                    v1 = (hip_r.x - knee_r.x, hip_r.y - knee_r.y)
                    v2 = (ankle_r.x - knee_r.x, ankle_r.y - knee_r.y)
                    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (
                        math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1]) + 1e-6
                    )
                    cos_angle = max(-1, min(1, cos_angle))
                    angles.append(math.degrees(math.acos(cos_angle)))

                if angles:
                    metrics["leg_angle"] = sum(angles) / len(angles)

            # Annotate and write
            annotated = annotator.annotate(frame, ball, pose, timestamp, metrics)
            writer.write(annotated)
            frame_count += 1
            prev_state = state

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
