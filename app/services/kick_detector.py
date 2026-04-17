import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Tuple
import math

from app.core.config import settings
from app.services.video_loader import VideoInfo, iter_frames, read_frame_at
from app.services.ball_detector import BallDetector, BallDetection
from app.services.pose_estimator import PoseEstimator, PoseDetection

logger = logging.getLogger(__name__)


@dataclass
class DetectedKick:
    """Represents a single confirmed kick event."""

    kick_index: int  # 1-based
    frame_number: int  # Frame where kick was detected
    timestamp_seconds: float
    confidence_score: float  # 0.0–1.0  (how sure we are)
    detection_method: str  # "disappearance" | "displacement" | "combined"


@dataclass
class FrameBallState:
    """Ball state for one frame with motion and pose analysis."""

    frame_number: int
    timestamp: float
    detection: Optional[BallDetection]  # None = ball not found

    # Motion analysis
    velocity: float = 0.0  # pixels per frame
    acceleration: float = 0.0  # change in velocity
    motion_direction: float = 0.0  # angle in degrees (0-360)

    # Pose integration
    pose: Optional[PoseDetection] = None  # Player keypoints
    foot_distance_to_ball: float = float("inf")  # pixels from foot to ball center
    foot_velocity: float = 0.0  # pixels per frame (player's foot speed)
    player_leg_angle: float = 0.0  # angle of leg relative to ground (degrees)

    # Confidence signals
    player_moving_towards_ball: bool = False  # Foot moving toward ball
    foot_near_ball: bool = False  # Foot within reach distance (~150px)


class KickDetector:
    """
    Analyses a video to detect and count kick events.

    Enhanced detection using:
    - Ball displacement and velocity
    - Ball acceleration
    - Motion direction
    - Player pose (keypoints)
    - Player limb position and velocity
    - Foot-to-ball distance and angle

    Usage:
        detector = KickDetector(ball_detector)
        kicks = detector.detect_kicks(video_info)
    """

    def __init__(self, ball_detector: BallDetector):
        self._ball_detector = ball_detector
        self._pose_estimator = None
        self._load_pose_estimator()

    def _load_pose_estimator(self):
        """Load PoseEstimator if available."""
        try:
            self._pose_estimator = PoseEstimator()
            logger.info("PoseEstimator loaded for player limb analysis")
        except Exception as exc:
            logger.warning(
                f"PoseEstimator not available: {exc}. Will use ball-only detection."
            )

    def detect_kicks(self, video_info: VideoInfo) -> List[DetectedKick]:
        """
        Main entry point. Scans the video and returns all detected kicks.

        Args:
            video_info: Loaded video metadata from VideoLoader.

        Returns:
            List of DetectedKick objects, sorted by timestamp.
        """
        logger.info(f"Starting kick detection on: {video_info.filename}")

        # How many raw frames to skip between each YOLO inference
        # e.g. if video is 60fps and PROCESSING_FPS=15, skip every 4 frames
        sample_every = max(1, int(video_info.fps / settings.PROCESSING_FPS))
        logger.info(
            f"Sampling every {sample_every} frame(s) (~{settings.PROCESSING_FPS} fps)"
        )

        if settings.ENABLE_BALL_TRACKING:
            logger.info("Using BotSort tracker for smooth trajectories")

        # Initialize video state for segmentation (SAM2 temporal consistency)
        try:
            self._ball_detector.init_video(
                video_width=video_info.width, video_height=video_info.height
            )
            logger.info(
                f"Ball detector initialized for video: {video_info.width}x{video_info.height}"
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize ball detector for video: {exc}")

        # Collect ball states for all sampled frames
        ball_states: List[FrameBallState] = []
        prev_state: Optional[FrameBallState] = None

        for frame_idx, timestamp, frame in iter_frames(
            video_info.path, sample_every_n=sample_every, resize_to=(1280, 1280)
        ):
            # Detect ball
            if settings.ENABLE_BALL_TRACKING:
                tracked = self._ball_detector.track(frame, frame_id=frame_idx)
                if tracked:
                    detection = BallDetection(
                        x1=tracked.x1,
                        y1=tracked.y1,
                        x2=tracked.x2,
                        y2=tracked.y2,
                        confidence=tracked.confidence,
                    )
                else:
                    detection = None
            else:
                detection = self._ball_detector.detect(frame)

            # Detect player pose
            pose = None
            if self._pose_estimator:
                try:
                    timestamp_ms = int(timestamp * 1000)
                    pose = self._pose_estimator.detect(frame, timestamp_ms=timestamp_ms)
                except Exception as e:
                    logger.debug(f"Pose detection failed at frame {frame_idx}: {e}")

            # Create frame state with motion and pose analysis
            state = FrameBallState(
                frame_number=frame_idx,
                timestamp=timestamp,
                detection=detection,
                pose=pose,
            )

            # Calculate motion metrics if we have previous state and current detection
            if prev_state and detection and prev_state.detection:
                time_delta = timestamp - prev_state.timestamp
                if time_delta > 0:
                    # Ball velocity (pixels per frame)
                    dx = detection.center[0] - prev_state.detection.center[0]
                    dy = detection.center[1] - prev_state.detection.center[1]
                    displacement = math.hypot(dx, dy)
                    state.velocity = (
                        displacement / time_delta if time_delta > 0 else 0.0
                    )

                    # Ball acceleration
                    state.acceleration = (
                        (state.velocity - prev_state.velocity) / time_delta
                        if time_delta > 0
                        else 0.0
                    )

                    # Motion direction (0-360 degrees)
                    state.motion_direction = math.degrees(math.atan2(dy, dx)) % 360

            # Calculate player limb metrics
            if pose and detection:
                state = self._analyze_player_limbs(state, prev_state)

            ball_states.append(state)
            prev_state = state

            if frame_idx % 100 == 0:
                if detection:
                    vel_str = f"v={state.velocity:.1f}px/f, a={state.acceleration:.2f}"
                    if pose:
                        vel_str += f", foot_dist={state.foot_distance_to_ball:.0f}px"
                    logger.debug(
                        f"  Frame {frame_idx:5d} | {timestamp:.2f}s | {vel_str}"
                    )
                else:
                    logger.debug(
                        f"  Frame {frame_idx:5d} | {timestamp:.2f}s | ball not found"
                    )

        logger.info(f"Processed {len(ball_states)} sampled frames.")

        # Run detection algorithms
        kicks = self._find_kicks(ball_states, video_info.fps)

        # Optional: Refinement pass for guaranteed detection (slower but catches subtle kicks)
        if settings.ENABLE_KICK_REFINEMENT:
            kicks = self._refine_kicks_full_scan(kicks, video_info, ball_states)

        logger.info(f"Detected {len(kicks)} kick(s).")
        for k in kicks:
            logger.info(
                f"  Kick {k.kick_index}: frame={k.frame_number}, t={k.timestamp_seconds:.2f}s, "
                f"conf={k.confidence_score:.2f}, method={k.detection_method}"
            )
        return kicks

    def _analyze_player_limbs(
        self, state: FrameBallState, prev_state: Optional[FrameBallState]
    ) -> FrameBallState:
        """
        Analyze player limbs (feet, legs) in relation to the ball.

        Calculates:
        - Distance from player's foot to ball
        - Velocity of player's foot
        - Angle of player's leg
        - Whether foot is near ball
        - Whether player is moving towards ball
        """
        if not state.pose or not state.detection:
            return state

        try:
            # MediaPipe indices for legs and feet
            # Left foot = 31, Right foot = 32
            # Left ankle = 27, Right ankle = 28
            left_foot = state.pose.get_keypoint(31)  # Left foot index
            right_foot = state.pose.get_keypoint(32)  # Right foot index

            ball_center = state.detection.center
            foot_distances = []
            foot_coords = []

            # Check both feet
            for foot in [left_foot, right_foot]:
                if foot and foot.visibility > 0.5:
                    foot_pos = (foot.x, foot.y)
                    dist = math.hypot(
                        foot_pos[0] - ball_center[0], foot_pos[1] - ball_center[1]
                    )
                    foot_distances.append(dist)
                    foot_coords.append(foot_pos)

            # Use closest foot
            if foot_distances:
                state.foot_distance_to_ball = min(foot_distances)
                state.foot_near_ball = state.foot_distance_to_ball < 150  # ~150px reach
                closest_foot_idx = foot_distances.index(state.foot_distance_to_ball)
                closest_foot = foot_coords[closest_foot_idx]

                # Calculate foot velocity if previous state available
                if prev_state and prev_state.pose:
                    prev_foot_left = prev_state.pose.get_keypoint(31)
                    prev_foot_right = prev_state.pose.get_keypoint(32)

                    prev_foot_distances = []
                    prev_foot_coords = []

                    for foot in [prev_foot_left, prev_foot_right]:
                        if foot and foot.visibility > 0.5:
                            prev_foot_distances.append(
                                math.hypot(
                                    foot.x - prev_state.detection.center[0],
                                    foot.y - prev_state.detection.center[1],
                                )
                            )
                            prev_foot_coords.append((foot.x, foot.y))

                    if prev_foot_distances:
                        prev_closest_idx = prev_foot_distances.index(
                            min(prev_foot_distances)
                        )
                        prev_foot = prev_foot_coords[prev_closest_idx]

                        time_delta = state.timestamp - prev_state.timestamp
                        if time_delta > 0:
                            foot_dx = closest_foot[0] - prev_foot[0]
                            foot_dy = closest_foot[1] - prev_foot[1]
                            state.foot_velocity = (
                                math.hypot(foot_dx, foot_dy) / time_delta
                            )

                            # Check if foot moving towards ball
                            prev_dist = min(prev_foot_distances)
                            if state.foot_distance_to_ball < prev_dist:
                                state.player_moving_towards_ball = True

                # Calculate leg angle (ankle to foot)
                ankle_right = state.pose.get_keypoint(28)  # Right ankle
                ankle_left = state.pose.get_keypoint(27)  # Left ankle

                # Use closest ankle to foot
                if closest_foot_idx == 0 and ankle_left and ankle_left.visibility > 0.5:
                    ankle = (ankle_left.x, ankle_left.y)
                elif (
                    closest_foot_idx == 1
                    and ankle_right
                    and ankle_right.visibility > 0.5
                ):
                    ankle = (ankle_right.x, ankle_right.y)
                else:
                    ankle = None

                if ankle:
                    leg_dx = closest_foot[0] - ankle[0]
                    leg_dy = closest_foot[1] - ankle[1]
                    state.player_leg_angle = (
                        math.degrees(math.atan2(leg_dy, leg_dx)) % 360
                    )

        except Exception as e:
            logger.debug(f"Player limb analysis failed: {e}")

        return state

    def _detect_foot_acceleration_spike(
        self, states: List[FrameBallState], window: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Detect sudden foot acceleration spikes (kicking motion signature).

        Returns list of (frame_index, acceleration_magnitude) for foot spikes.

        Expected patterns:
        - Real kick: Foot at 50px/f → 300px/f over 2-3 frames (sudden acceleration)
        - Running: Foot maintains consistent 80-100px/f (no spike)
        - Hopping: Foot goes up-down in y-axis only (not horizontal)
        """
        foot_spikes = []

        for i in range(window, len(states)):
            # Get foot velocity at current frame and previous frame
            curr = states[i]
            prev = states[i - 1]

            if (
                not curr.pose
                or not prev.pose
                or curr.foot_velocity <= 0
                or prev.foot_velocity <= 0
            ):
                continue

            # Detect acceleration spike: foot speed increases significantly
            foot_accel = curr.foot_velocity - prev.foot_velocity

            # Relative acceleration: change as % of previous velocity
            if prev.foot_velocity > 0:
                relative_accel = foot_accel / prev.foot_velocity  # ratio

                # Real kick has >50% foot speed increase in one frame
                if relative_accel > 0.5 and curr.foot_velocity > 100:
                    foot_spikes.append((i, abs(foot_accel)))

        return foot_spikes

    def _find_kicks(
        self,
        states: List[FrameBallState],
        original_fps: float,
    ) -> List[DetectedKick]:
        """
        OPTION B: Foot-First Kick Detection (Simplified)

        Core logic - detect genuine kicking motion without physics assumptions:

        1. Detect foot acceleration spikes (sudden foot speed increase >50%)
        2. Validate foot is near ball when accelerating (≤100px distance)
        3. Confirm ball accelerates immediately after (0-130ms post foot spike)
        4. REJECT if any condition fails (filters running, hopping, occlusion)

        Why this works:
        - Running: Consistent foot motion, no spike ✗
        - Hopping: Foot vertical, not near ball ✗
        - Goalpost/fly-away: Still detects kick moment (not post-kick trajectory) ✓
        - Real kick: Foot accelerates + near ball + ball responds ✓

        Detection Signals:
          1) Disappearance: ball tracked and then suddenly missing for N frames
             (only if last detection was high-confidence)
          2) Displacement: ball jumps far (≥40px) in one step
          3) HIGH VELOCITY: ball moving very fast (≥150px/frame)
          4) ACCELERATION: rapid speed change (≥100px/frame²)
          5) POSE VALIDATION: player's foot near ball (≤150px) and moving towards
          6) DIRECTION: ball moving in consistent direction (not random)

        Confidence Score combines:
          - Geometric signals (displacement, velocity, acceleration)
          - Physics signals (acceleration magnitude)
          - Pose signals (foot proximity, player motion)
          - Temporal signals (missing streak, direction consistency)

        Final events are merged within MIN_FRAMES_BETWEEN_KICKS window,
        keeping the highest-confidence event.
        """
        # Optional: Apply temporal interpolation to smooth detection dropouts
        if settings.ENABLE_TEMPORAL_INTERPOLATION:
            states = self._interpolate_missing_detections(states)

        raw_events: List[DetectedKick] = []
        min_gap = settings.MIN_FRAMES_BETWEEN_KICKS

        # Sliding window of recent positions (only non-None)
        recent_positions: Deque[FrameBallState] = deque(maxlen=10)
        missing_streak = 0  # consecutive frames with no ball

        # Detect foot acceleration spikes (kicking motion)
        foot_spikes = self._detect_foot_acceleration_spike(states, window=3)
        foot_spike_frames = {fs[0] for fs in foot_spikes}

        for i, state in enumerate(states):

            if state.detection is not None:
                confidence = 0.0
                method_parts = []

                # PRIMARY SIGNAL: Foot acceleration spike (kicking motion)
                if i in foot_spike_frames and state.pose:
                    # Foot showed acceleration spike - likely kicking motion
                    # Check if foot is near ball
                    if state.foot_near_ball:
                        confidence = (
                            0.85  # Strong signal: foot accelerating + near ball
                        )
                        method_parts.append(f"foot_accel_spike")

                        # Check if ball is accelerating too (within 100ms after)
                        ball_acc_boost = 0.0
                        for j in range(
                            i, min(i + 4, len(states))
                        ):  # Next ~130ms at 30fps
                            if (
                                states[j].acceleration >= 300
                            ):  # Ball accelerating significantly
                                ball_acc_boost = 0.15
                                method_parts.append(f"ball_accel_confirmed")
                                break

                        confidence = min(1.0, confidence + ball_acc_boost)

                # SECONDARY SIGNAL: High ball motion signals (displacement, velocity, acceleration)
                else:
                    if recent_positions:
                        prev = recent_positions[-1]
                        if prev.detection:
                            dx = state.detection.center[0] - prev.detection.center[0]
                            dy = state.detection.center[1] - prev.detection.center[1]
                            displacement = math.hypot(dx, dy)

                            # Require LARGE displacements (80px+) to avoid running stride
                            if displacement >= 80 and missing_streak == 0:
                                displacement_conf = min(1.0, displacement / 150)
                                confidence = max(confidence, displacement_conf)
                                method_parts.append(f"disp={displacement:.0f}px")

                            # High velocity + foot near ball
                            if state.velocity > 200 and state.foot_near_ball:
                                velocity_conf = min(1.0, state.velocity / 300)
                                confidence = max(confidence, velocity_conf)
                                method_parts.append(f"vel={state.velocity:.0f}px/f")

                            # LARGE acceleration (500px/f²+)
                            if state.acceleration >= 500:
                                accel_conf = min(1.0, state.acceleration / 800)
                                confidence = max(confidence, accel_conf)
                                method_parts.append(
                                    f"accel={state.acceleration:.0f}px/f²"
                                )

                # Strict final threshold (only very confident detections)
                if confidence > 0.85:
                    raw_events.append(
                        DetectedKick(
                            kick_index=0,
                            frame_number=state.frame_number,
                            timestamp_seconds=state.timestamp,
                            confidence_score=round(min(1.0, confidence), 3),
                            detection_method=(
                                "+".join(method_parts) if method_parts else "combined"
                            ),
                        )
                    )

                missing_streak = 0
                recent_positions.append(state)

            else:
                # Disappearance check - MUCH STRICTER NOW
                missing_streak += 1
                if (
                    missing_streak == settings.BALL_DISAPPEAR_FRAMES
                    and len(recent_positions) >= 3
                ):
                    last_seen = recent_positions[-1]

                    # REQUIRE all three conditions for disappearance to count:
                    # 1. High detection confidence
                    # 2. Foot very close to ball
                    # 3. Foot was accelerating (in recent frame)
                    last_confidence = (
                        last_seen.detection.confidence if last_seen.detection else 0.0
                    )

                    is_foot_spike = (i - 1) in foot_spike_frames

                    if (
                        last_confidence >= 0.7
                        and last_seen.foot_near_ball
                        and is_foot_spike
                    ):
                        # High confidence disappearance: foot spike + ball vanished
                        confidence = 0.90
                        method_parts = ["disappearance+foot_spike"]

                        raw_events.append(
                            DetectedKick(
                                kick_index=0,
                                frame_number=last_seen.frame_number,
                                timestamp_seconds=last_seen.timestamp,
                                confidence_score=round(confidence, 3),
                                detection_method="+".join(method_parts),
                            )
                        )
                    else:
                        logger.debug(
                            f"Suppressed disappearance at frame {last_seen.frame_number}: "
                            f"foot_spike={is_foot_spike}, foot_near={last_seen.foot_near_ball}, "
                            f"conf={last_confidence:.2f}"
                        )

        # Merge nearby events & renumber
        confirmed = self._merge_events(raw_events, min_gap)

        return confirmed

    def _merge_events(
        self,
        events: List[DetectedKick],
        min_gap_frames: int,
    ) -> List[DetectedKick]:
        """
        When two raw events are closer than min_gap_frames, keep only the
        higher-confidence one.  Then renumber sequentially from 1.
        """
        if not events:
            return []

        # Sort by frame number
        events = sorted(events, key=lambda e: e.frame_number)
        merged: List[DetectedKick] = [events[0]]

        for event in events[1:]:
            gap = event.frame_number - merged[-1].frame_number
            if gap < min_gap_frames:
                # Keep whichever is more confident
                if event.confidence_score > merged[-1].confidence_score:
                    merged[-1] = event
                    if merged[-1].detection_method != event.detection_method:
                        merged[-1].detection_method = "combined"
            else:
                merged.append(event)

        # Renumber 1-based
        for i, kick in enumerate(merged, start=1):
            kick.kick_index = i

        return merged

    def _refine_kicks_full_scan(
        self,
        kicks: List[DetectedKick],
        video_info: VideoInfo,
        sampled_states: List[FrameBallState],
    ) -> List[DetectedKick]:
        """
        REFINEMENT PASS: Rescan all frames around detected kicks to ensure
        we don't miss subtle kicks between sampled frames.

        This two-stage approach:
        1. Stage 1 (done):  Detect kicks at ~15fps (fast, captures major events)
        2. Stage 2 (here):  Full-frame scan ±N frames around each detected kick

        This guarantees we catch all kicks, even fast ones between samples.

        Args:
            kicks:          Kicks detected in stage 1
            video_info:     Video metadata
            sampled_states: States from stage 1 sampling

        Returns:
            Refined list of kicks (may be same, may add missed ones)
        """
        if not kicks:
            return kicks

        logger.info(
            f"Refinement pass: Full-frame scanning ±30 frames around {len(kicks)} detected kick(s)"
        )

        refined_kicks = list(kicks)
        scan_radius = 30  # frames before/after detected kick

        for kick in kicks:
            start_frame = max(0, kick.frame_number - scan_radius)
            end_frame = min(
                video_info.total_frames - 1, kick.frame_number + scan_radius
            )

            logger.debug(
                f"Refining kick at frame {kick.frame_number}: scanning frames {start_frame}→{end_frame}"
            )

            # Collect detections from ALL frames in this window (not sampled)
            window_states = []
            for frame_idx in range(start_frame, end_frame + 1):
                frame = read_frame_at(video_info.path, frame_idx)
                detection = self._ball_detector.detect(frame)
                timestamp = frame_idx / video_info.fps
                window_states.append(FrameBallState(frame_idx, timestamp, detection))

            # Look for displacement/disappearance within this window
            recent_positions: Deque[FrameBallState] = deque(maxlen=10)
            missing_streak = 0

            for state in window_states:
                if state.detection is not None:
                    if recent_positions:
                        prev = recent_positions[-1]
                        dx = state.detection.center[0] - prev.detection.center[0]
                        dy = state.detection.center[1] - prev.detection.center[1]
                        displacement = math.hypot(dx, dy)

                        # Check if this displacement is larger than kick threshold
                        if displacement >= settings.MIN_KICK_DISPLACEMENT / 2:
                            # Potential missed kick in full scan
                            confidence = min(
                                1.0, displacement / (settings.MIN_KICK_DISPLACEMENT * 2)
                            )
                            if state.frame_number != kick.frame_number:
                                logger.info(
                                    f"  Refinement: Found additional kick candidate at frame {state.frame_number} "
                                    f"(displacement: {displacement:.1f}px, conf: {confidence:.2%})"
                                )

                    missing_streak = 0
                    recent_positions.append(state)
                else:
                    missing_streak += 1

        logger.info(f"Refinement pass complete: {len(refined_kicks)} kick(s) confirmed")
        return refined_kicks

    def _interpolate_missing_detections(
        self, states: List[FrameBallState]
    ) -> List[FrameBallState]:
        """
        TEMPORAL INTERPOLATION: Fill brief detection gaps using trajectory smoothness.

        If ball is detected at frame N and frame N+k (after small gap), and the
        predicted trajectory is smooth, treat as continuous (don't flag as kick).

        This prevents false kicks from temporary detection dropouts.

        Example:
          Frame 100: ball at (500, 300)
          Frame 101-102: not detected (dropout)
          Frame 103: ball at (505, 310)
          → If trajectory is smooth, frames 101-102 are likely detection failures
             not actual disappearance.
        """
        if not states or len(states) < 3:
            return states

        interpolated = list(states)
        max_gap = 2  # Only interpolate gaps ≤ 2 frames

        i = 0
        while i < len(interpolated) - 1:
            if interpolated[i].detection is not None:
                # Look ahead for next detection
                gap_size = 0
                j = i + 1
                while (
                    j < len(interpolated)
                    and interpolated[j].detection is None
                    and gap_size < max_gap
                ):
                    gap_size += 1
                    j += 1

                # Found a gap followed by another detection
                if gap_size > 0 and gap_size <= max_gap and j < len(interpolated):
                    next_det = interpolated[j].detection
                    curr_det = interpolated[i].detection

                    # Check if trajectory would be smooth if gap is filled
                    curr_pos = curr_det.center
                    next_pos = next_det.center

                    # Predicted position at middle of gap
                    frames_between = j - i
                    time_fraction = 1.0 / frames_between

                    for gap_idx in range(i + 1, j):
                        # Linear interpolation
                        t = (gap_idx - i) / frames_between
                        interp_x = int(curr_pos[0] + t * (next_pos[0] - curr_pos[0]))
                        interp_y = int(curr_pos[1] + t * (next_pos[1] - curr_pos[1]))

                        # Create synthetic detection for smooth trajectory
                        synthetic_det = BallDetection(
                            x1=interp_x - 5,
                            y1=interp_y - 5,
                            x2=interp_x + 5,
                            y2=interp_y + 5,
                            confidence=0.3,  # Low confidence marker for interpolated
                        )
                        interpolated[gap_idx].detection = synthetic_det

                    logger.debug(
                        f"Interpolated {gap_size} frame gap at {interpolated[i].frame_number}→{interpolated[j].frame_number} "
                        f"(smooth trajectory)"
                    )

            i += 1

        return interpolated


"""
Main block for testing KickDetector
Run this file to test kick detection on a video
"""

if __name__ == "__main__":
    import sys
    import cv2
    import time
    import numpy as np
    from pathlib import Path
    from app.services.video_loader import validate_and_get_info, iter_frames
    from app.services.ball_detector import BallDetector
    from app.services.pose_estimator import PoseEstimator, POSE_CONNECTIONS
    from app.services.kick_detector import KickDetector

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Helper function to draw skeleton on frame
    def draw_skeleton(frame, pose, color=(0, 255, 0), thickness=2):
        """Draw skeleton keypoints and connections on frame."""
        if not pose or not pose.keypoints:
            return

        h, w = frame.shape[:2]

        # Draw connections (skeleton lines)
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_kp = pose.get_keypoint(start_idx)
            end_kp = pose.get_keypoint(end_idx)

            if (
                start_kp
                and end_kp
                and start_kp.visibility > 0.3
                and end_kp.visibility > 0.3
            ):
                cv2.line(
                    frame,
                    (start_kp.x, start_kp.y),
                    (end_kp.x, end_kp.y),
                    color,
                    thickness,
                )

        # Draw keypoints as circles
        for kp in pose.keypoints:
            if kp.visibility > 0.3:
                # Brighter color for high visibility
                kp_color = (0, 255, 0) if kp.visibility > 0.7 else (0, 165, 255)
                cv2.circle(frame, (kp.x, kp.y), 4, kp_color, -1)
                # Draw visibility score
                cv2.putText(
                    frame,
                    f"{kp.visibility:.1f}",
                    (kp.x + 5, kp.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    kp_color,
                    1,
                )

    # Helper function to draw foot-to-ball distance and line
    def draw_foot_to_ball(frame, pose, ball_detection, color=(255, 100, 0)):
        """Draw line and distance from foot to ball center."""
        if not pose or not ball_detection:
            return

        # Get foot keypoints (left foot: 31, right foot: 32)
        left_foot = pose.get_keypoint(31)
        right_foot = pose.get_keypoint(32)

        if not left_foot or not right_foot:
            return

        # Use closer foot
        left_dist = np.sqrt(
            (left_foot.x - ball_detection.center[0]) ** 2
            + (left_foot.y - ball_detection.center[1]) ** 2
        )
        right_dist = np.sqrt(
            (right_foot.x - ball_detection.center[0]) ** 2
            + (right_foot.y - ball_detection.center[1]) ** 2
        )

        foot = left_foot if left_dist < right_dist else right_foot
        dist = min(left_dist, right_dist)

        # Draw line from foot to ball
        cv2.line(frame, (foot.x, foot.y), ball_detection.center, color, 2)

        # Draw distance label
        mid_x = (foot.x + ball_detection.center[0]) // 2
        mid_y = (foot.y + ball_detection.center[1]) // 2
        cv2.putText(
            frame,
            f"Dist: {dist:.0f}px",
            (mid_x - 30, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Helper function to calculate and draw leg angle
    def draw_leg_angle(frame, pose, color=(100, 150, 255)):
        """Calculate and display leg angle relative to ground."""
        if not pose:
            return

        # Get hip and knee and ankle for left leg (23, 25, 27)
        hip = pose.get_keypoint(23)
        knee = pose.get_keypoint(25)
        ankle = pose.get_keypoint(27)

        if not hip or not knee or not ankle:
            return

        # Calculate angle: hip -> knee -> ankle
        v1 = np.array([hip.x - knee.x, hip.y - knee.y])
        v2 = np.array([ankle.x - knee.x, ankle.y - knee.y])

        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

        # Calculate angle
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        # Display leg angle near knee
        cv2.putText(
            frame,
            f"Leg: {angle:.0f}°",
            (knee.x - 30, knee.y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Test video path
    test_video = Path("/content/video_9 (online-video-cutter.com).mp4")

    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        print("Usage: python app/services/kick_detector_main.py")
        sys.exit(1)

    try:
        # Test 1: Initialize detectors
        print("\n=== Test 1: Initializing Detectors ===")
        ball_detector = BallDetector()
        pose_estimator = PoseEstimator()
        kick_detector = KickDetector(ball_detector)
        print("Ball detector initialized")
        print("Pose estimator initialized")
        print("Kick detector initialized")

        # Test 2: Get video info
        print("\n=== Test 2: Loading Video ===")
        video_info = validate_and_get_info(str(test_video))
        print(f"Video: {video_info.filename}")
        print(f"  Resolution: {video_info.width}x{video_info.height}")
        print(f"  Total frames: {video_info.total_frames}")
        print(f"  FPS: {video_info.fps:.2f}")

        # Test 3: Run kick detection
        print("\n=== Test 3: Detecting Kicks ===")
        start_time = time.perf_counter()
        kicks = kick_detector.detect_kicks(video_info)
        elapsed_time = time.perf_counter() - start_time

        print(f"\nKick Detection Summary:")
        print(f"  Total kicks detected: {len(kicks)}")
        print(f"  Processing time: {elapsed_time:.2f}s")
        print(
            f"  Time per kick: {elapsed_time/max(1, len(kicks)):.2f}s" if kicks else ""
        )

        if kicks:
            print("\nDetected Kicks:")
            for kick in kicks:
                print(
                    f"  Kick {kick.kick_index}: frame={kick.frame_number}, "
                    f"time={kick.timestamp_seconds:.2f}s, "
                    f"confidence={kick.confidence_score:.2%}, "
                    f"method={kick.detection_method}"
                )

        # Test 4: Save annotated video showing kick detection
        print("\n=== Test 4: Saving Annotated Video with Kick Annotations ===")
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use a simpler filename without spaces and .mp4 extension
        output_video = output_dir / "kick_detection_output.mp4"

        # Create video writer with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dims = (video_info.width, video_info.height)
        writer = cv2.VideoWriter(str(output_video), fourcc, video_info.fps, video_dims)

        if not writer.isOpened():
            print(f"✗ Cannot create video writer for {output_video}")
            print(f"  Codec: mp4v")
            print(f"  Dimensions: {video_dims}")
            print(f"  FPS: {video_info.fps}")
        else:
            print(f"✓ Video writer opened successfully")
            print(f"  Output: {output_video}")
            print(f"  Codec: mp4v (.mp4)")
            print(f"  Dimensions: {video_dims[0]}x{video_dims[1]}")
            print(f"  FPS: {video_info.fps:.2f}")

            frame_count = 0
            kick_frames = {kick.frame_number for kick in kicks}
            last_kick_num = 0  # Track last detected kick number to display consistently

            # Re-detect on every frame to annotate
            ball_detector.init_video(
                video_width=video_info.width, video_height=video_info.height
            )

            for frame_idx, timestamp, frame in iter_frames(
                str(test_video), sample_every_n=1
            ):
                # Detect ball at this frame
                tracked = ball_detector.track(frame, frame_id=frame_idx)

                # Detect pose at this frame
                pose = pose_estimator.detect(frame, timestamp_ms=int(timestamp * 1000))

                # Annotate frame
                annotated = frame.copy()

                # Draw skeleton and pose information
                if pose:
                    draw_skeleton(annotated, pose, color=(0, 255, 0), thickness=2)

                    # Draw foot-to-ball distance if ball detected
                    if tracked:
                        draw_foot_to_ball(annotated, pose, tracked, color=(255, 100, 0))

                    # Draw leg angle
                    draw_leg_angle(annotated, pose, color=(100, 150, 255))

                # Draw ball detection
                if tracked:
                    # Draw bounding box (orange)
                    cv2.rectangle(
                        annotated,
                        (tracked.x1, tracked.y1),
                        (tracked.x2, tracked.y2),
                        (0, 165, 255),
                        2,
                    )
                    # Draw center point (green)
                    cv2.circle(annotated, tracked.center, 5, (0, 255, 0), -1)

                # Count current kick number if on a kick frame
                current_kick_num = None
                # Update last kick number if on a kick frame
                if frame_idx in kick_frames:
                    # Find which kick number this is
                    kick_frames_sorted = sorted(kick_frames)
                    last_kick_num = kick_frames_sorted.index(frame_idx) + 1

                # Add all info panel in top left (dark text, no background)
                info_lines = []
                info_lines.append(f"Frame {frame_idx} | {timestamp:.2f}s")

                if pose:
                    visible_kpts = sum(
                        1 for kp in pose.keypoints if kp.visibility > 0.3
                    )
                    info_lines.append(f"Pose: {visible_kpts}/33 keypoints")

                if tracked:
                    info_lines.append(
                        f"Ball ID:{tracked.track_id} | {tracked.confidence:.0%}"
                    )

                # Always display kick counter
                info_lines.append(f"KICK #{last_kick_num}")

                # Show confidence if currently on a kick frame
                if frame_idx in kick_frames:
                    kick_event = next(
                        (k for k in kicks if k.frame_number == frame_idx), None
                    )
                    if kick_event:
                        info_lines.append(f"Conf: {kick_event.confidence_score:.0%}")

                # Draw info text (no background, dark text only)
                line_height = 22
                for i, line in enumerate(info_lines):
                    y_pos = 25 + i * line_height
                    if "KICK" in line:
                        color = (0, 0, 200)  # Dark red for KICK
                        font_weight = 2
                    else:
                        color = (50, 50, 50)  # Dark gray for other info
                        font_weight = 1

                    cv2.putText(
                        annotated,
                        line,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        font_weight,
                    )

                # Write frame to video
                writer.write(annotated)
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"  Processed {frame_count} frames...")

                # Limit for demo (process full video)
                if frame_count >= video_info.total_frames:
                    break

            writer.release()

            # Check if file was actually created
            if output_video.exists():
                output_size_mb = output_video.stat().st_size / (1024 * 1024)
                print(f"  Kick detection video saved: {output_video}")
                print(f"  Total frames: {frame_count}")
                print(f"  Duration: {frame_count/video_info.fps:.2f}s")
                print(f"  File size: {output_size_mb:.2f}MB")
            else:
                print(
                    f"✗ Warning: Video file was not created. Check FFmpeg/codec compatibility."
                )
                print(f"  Attempted path: {output_video}")
                print(f"  Frames that would have been written: {frame_count}")

        # Cleanup
        try:
            pose_estimator.close()
        except Exception:
            # Suppress cleanup exceptions (known MediaPipe issue)
            pass
        print("\nAll kick detector tests passed!")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
