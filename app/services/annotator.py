import cv2
import logging
import numpy as np
from collections import deque
from typing import Optional, Deque, Tuple

from app.core.config import settings
from app.services.ball_detector import BallDetection
from app.services.pose_estimator import PoseDetection, POSE_CONNECTIONS

logger = logging.getLogger(__name__)


class FrameAnnotator:
    """
    Draws detection results onto a frame.

    The trajectory deque is stateful — it accumulates ball positions across
    frames, so you should use ONE annotator instance per clip.

    Usage:
        annotator = FrameAnnotator(kick_index=1)
        annotated = annotator.annotate(frame, ball_det, pose_det, timestamp)
    """

    def __init__(self, kick_index: int):
        self.kick_index = kick_index
        # Store recent ball centers so we can draw a trajectory trail
        self._trajectory: Deque[Tuple[int, int]] = deque(
            maxlen=settings.TRAJECTORY_MAX_POINTS
        )
        self.last_track_id = None  # For displaying track ID

    def annotate(
        self,
        frame: np.ndarray,
        ball: Optional[BallDetection],
        pose: Optional[PoseDetection],
        timestamp: float,
    ) -> np.ndarray:
        """
        Apply all overlays to a copy of the frame and return the result.

        Args:
            frame:     Original BGR frame (not modified in place).
            ball:      Ball detection result, or None.
            pose:      Pose detection result, or None.
            timestamp: Current time in the original video.

        Returns:
            Annotated BGR frame (same size).
        """
        out = frame.copy()

        if ball is not None:
            self._trajectory.append(ball.center)
            self._draw_ball_box(out, ball)

        self._draw_trajectory(out)

        if pose is not None:
            self._draw_skeleton(out, pose)
            self._draw_keypoints(out, pose)

        self._draw_hud(out, timestamp, ball, pose)

        return out

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _draw_ball_box(self, frame: np.ndarray, ball: BallDetection):
        """Draw orange bounding box and confidence label around the ball."""
        cv2.rectangle(
            frame,
            (ball.x1, ball.y1),
            (ball.x2, ball.y2),
            settings.BALL_BOX_COLOR,
            thickness=2,
        )
        label = f"Ball {ball.confidence:.0%}"
        self._put_label(frame, label, ball.x1, ball.y1 - 6, settings.BALL_BOX_COLOR)

    def _draw_trajectory(self, frame: np.ndarray):
        """
        Draw a fading trail of dots showing where the ball has been.
        Older points are smaller and more transparent.
        """
        points = list(self._trajectory)
        n = len(points)
        for i, center in enumerate(points):
            # Alpha: 0.2 for oldest → 1.0 for newest
            alpha = 0.2 + 0.8 * (i / max(n - 1, 1))
            radius = max(2, int(6 * alpha))
            color = tuple(int(c * alpha) for c in settings.TRAJECTORY_COLOR)
            cv2.circle(frame, center, radius, color, thickness=-1)

        # Connect dots with a line for better readability
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], settings.TRAJECTORY_COLOR, 1)

    def _draw_skeleton(self, frame: np.ndarray, pose: PoseDetection):
        """Draw lines connecting body landmark pairs."""
        for a_idx, b_idx in POSE_CONNECTIONS:
            a = pose.get_keypoint(a_idx)
            b = pose.get_keypoint(b_idx)
            if a and b and a.visibility > 0.4 and b.visibility > 0.4:
                cv2.line(frame, (a.x, a.y), (b.x, b.y), settings.SKELETON_COLOR, 2)

    def _draw_keypoints(self, frame: np.ndarray, pose: PoseDetection):
        """Draw circles at each visible body landmark."""
        for kp in pose.keypoints:
            if kp.visibility > 0.4:
                cv2.circle(frame, (kp.x, kp.y), 4, settings.KEYPOINT_COLOR, -1)
                cv2.circle(frame, (kp.x, kp.y), 4, (0, 0, 0), 1)  # Thin black outline

    def _draw_hud(
        self,
        frame: np.ndarray,
        timestamp: float,
        ball: Optional[BallDetection],
        pose: Optional[PoseDetection],
    ):
        """Draw heads-up display with kick index, time, and detection status."""
        h, w = frame.shape[:2]

        # Semi-transparent dark banner at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        mins = int(timestamp) // 60
        secs = timestamp % 60
        time_str = f"{mins:02d}:{secs:05.2f}"

        ball_str = f"Ball: {ball.confidence:.0%}" if ball else "Ball: --"
        pose_str = f"Pose: {pose.visible_count}kp" if pose else "Pose: --"

        text = f"Kick #{self.kick_index}   {time_str}   {ball_str}   {pose_str}"
        cv2.putText(
            frame,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            settings.TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _put_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple):
        """Helper: draw label with a dark shadow for readability."""
        cv2.putText(
            frame,
            text,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
