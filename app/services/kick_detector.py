import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Tuple, Union

from app.core.config import settings
from app.services.video_loader import VideoInfo, iter_frames
from app.services.ball_detector import BallDetector, BallDetection
from app.services.pose_estimator import PoseEstimator, PoseDetection

logger = logging.getLogger(__name__)


@dataclass
class DetectedKick:
    """Represents a single confirmed kick event."""

    kick_index: int  # 1-based
    frame_number: int  # Frame where kick was detected
    timestamp_seconds: float
    confidence_score: float  # 0.0–1.0
    detection_method: str


@dataclass
class FrameBallState:
    """Ball state for one frame with motion and pose analysis."""

    frame_number: int
    timestamp: float
    detection: Optional[BallDetection]  # None = ball not found

    # Ball motion
    velocity: float = 0.0  # pixels/second
    acceleration: float = 0.0  # change in velocity (pixels/second²)
    motion_direction: float = 0.0  # degrees (0–360)

    # Pose
    pose: Optional[PoseDetection] = None
    foot_distance_to_ball: float = float("inf")
    foot_velocity: float = 0.0  # actual foot speed in pixels/second
    foot_screen_pos: Optional[Tuple[float, float]] = None  # (x, y) of closest foot

    # Derived flags
    foot_near_ball: bool = False


class KickDetector:
    """
    Detects kick events in a football video.

    Detection philosophy (foot-first, physics-validated):

    A genuine kick requires ALL of the following to be true:
      1. Foot acceleration spike  — foot speed jumps >80% in one step
      2. Foot proximity           — foot is within MAX_FOOT_TO_BALL_DISTANCE of ball
      3. Ball was stationary      — ball velocity near zero in the 5 frames before contact
      4. Ball moves fast after    — ball velocity exceeds MIN_KICK_VELOCITY post-contact
      5. (optional boost) direction change — ball direction shifts significantly

    Every condition is independently required; a failure on any single one
    suppresses the event.  This eliminates the main false-positive sources:
      - Running past the ball  (no ball-stationary + no ball-fast-after)
      - Tracking dropout        (suppressed by suspicious-frame window)
      - Dribbling               (ball not stationary, no single spike)
    """

    def __init__(self, ball_detector: BallDetector):
        self._ball_detector = ball_detector
        self._pose_estimator: Optional[PoseEstimator] = None
        self._load_pose_estimator()

    def _load_pose_estimator(self) -> None:
        try:
            self._pose_estimator = PoseEstimator()
            logger.info("PoseEstimator loaded.")
        except Exception as exc:
            logger.warning(f"PoseEstimator unavailable: {exc}. Ball-only mode.")

    def detect_kicks(
        self, video_info: VideoInfo, return_states: bool = False
    ) -> Union[List[DetectedKick], Tuple[List[DetectedKick], List[FrameBallState]]]:
        """
        Scan the video and return all confirmed kick events.

        When return_states=True, also return per-frame FrameBallState values
        so downstream consumers can reuse detections/metrics without rerunning
        model inference.
        """
        logger.info(f"Starting kick detection: {video_info.filename}")

        # sample_every = max(1, int(video_info.fps / settings.PROCESSING_FPS))
        logger.info(f"Sampling every 1 frame(s) (~{settings.PROCESSING_FPS} fps)")

        try:
            logger.debug("Initializing ball detector with video_path=%s", video_info.path)
            self._ball_detector.init_video(video_path=video_info.path)
            logger.debug("Ball detector initialized successfully")
        except Exception as exc:
            logger.warning(f"Ball detector init failed: {exc}")

        ball_states: List[FrameBallState] = []
        prev_state: Optional[FrameBallState] = None

        def _process_state(
            frame_idx: int,
            timestamp: float,
            frame,
            detection: Optional[BallDetection],
        ) -> None:
            nonlocal prev_state
            logger.debug("[Frame %d] Ball detection: %s", frame_idx, detection)

            # --- Pose detection ---
            logger.debug("[Frame %d] Detecting pose...", frame_idx)
            pose = None
            if self._pose_estimator:
                try:
                    pose = self._pose_estimator.detect(
                        frame, timestamp_ms=int(timestamp * 1000)
                    )
                except Exception as exc:
                    logger.debug("Pose failed at frame %d: %s", frame_idx, exc)
            logger.debug(
                "[Frame %d] Pose detection: %s",
                frame_idx,
                "OK" if pose else "None",
            )

            state = FrameBallState(
                frame_number=frame_idx,
                timestamp=timestamp,
                detection=detection,
                pose=pose,
            )

            if prev_state and detection and prev_state.detection:
                self._compute_ball_motion(state, prev_state)

            if pose and detection:
                self._compute_foot_metrics(state, prev_state)

            logger.debug("[Frame %d] Complete", frame_idx)

            ball_states.append(state)
            prev_state = state

            if frame_idx % 100 == 0:
                self._log_frame(frame_idx, timestamp, state)

        logger.debug("Starting frame iteration for %s...", video_info.path)

        if settings.ENABLE_BALL_TRACKING:
            batch_size = max(1, settings.BALL_TRACK_BATCH_SIZE)
            frame_batch = []
            frame_idx_batch: List[int] = []
            timestamp_batch: List[float] = []

            def _flush_tracking_batch() -> None:
                if not frame_batch:
                    return

                tracked_batch = self._ball_detector.track_batch(
                    frame_batch,
                    frame_ids=frame_idx_batch,
                )

                for frame_idx, timestamp, frame, tracked in zip(
                    frame_idx_batch,
                    timestamp_batch,
                    frame_batch,
                    tracked_batch,
                ):
                    detection = (
                        BallDetection(
                            x1=tracked.x1,
                            y1=tracked.y1,
                            x2=tracked.x2,
                            y2=tracked.y2,
                            confidence=tracked.confidence,
                        )
                        if tracked
                        else None
                    )
                    _process_state(frame_idx, timestamp, frame, detection)

                frame_batch.clear()
                frame_idx_batch.clear()
                timestamp_batch.clear()

            for frame_idx, timestamp, frame in iter_frames(
                video_info.path, sample_every_n=1
            ):
                logger.debug(
                    "[Frame %d] Buffering frame for batched tracking...",
                    frame_idx,
                )
                frame_batch.append(frame)
                frame_idx_batch.append(frame_idx)
                timestamp_batch.append(timestamp)

                if len(frame_batch) >= batch_size:
                    _flush_tracking_batch()

            _flush_tracking_batch()
        else:
            for frame_idx, timestamp, frame in iter_frames(
                video_info.path, sample_every_n=1
            ):
                logger.debug("[Frame %d] Processing...", frame_idx)
                detection = self._ball_detector.detect(frame)
                _process_state(frame_idx, timestamp, frame, detection)

        logger.info(f"Processed {len(ball_states)} sampled frames.")
        kicks = self._find_kicks(ball_states)
        logger.info(f"Detected {len(kicks)} kick(s).")
        for k in kicks:
            logger.info(
                f"  Kick {k.kick_index}: frame={k.frame_number}, "
                f"t={k.timestamp_seconds:.2f}s, conf={k.confidence_score:.2f}, "
                f"method={k.detection_method}"
            )
        if return_states:
            return kicks, ball_states
        return kicks

    def _compute_ball_motion(self, state: FrameBallState, prev: FrameBallState) -> None:
        """Fill velocity, acceleration and direction on *state* in place."""
        dt = state.timestamp - prev.timestamp
        if dt <= 0:
            return

        dx = state.detection.center[0] - prev.detection.center[0]
        dy = state.detection.center[1] - prev.detection.center[1]
        displacement = math.hypot(dx, dy)

        state.velocity = displacement / dt
        state.acceleration = (state.velocity - prev.velocity) / dt
        state.motion_direction = math.degrees(math.atan2(dy, dx)) % 360

    def _compute_foot_metrics(
        self, state: FrameBallState, prev: Optional[FrameBallState]
    ) -> None:
        """
        Compute foot-to-ball distance and actual foot screen velocity.

        Foot velocity is the real pixel displacement of the foot per second,
        NOT the rate of change of foot-to-ball distance (which was the
        original bug causing false spikes on approach).
        """
        if not state.pose or not state.detection:
            return

        ball_cx, ball_cy = state.detection.center

        # Collect visible feet (MediaPipe: 31=left foot index, 32=right foot index)
        feet = []
        for kp_idx in (31, 32):
            kp = state.pose.get_keypoint(kp_idx)
            if kp and kp.visibility > 0.5:
                feet.append((kp.x, kp.y))

        if not feet:
            return

        # Closest foot to ball
        dists = [math.hypot(fx - ball_cx, fy - ball_cy) for fx, fy in feet]
        min_dist = min(dists)
        closest_foot = feet[dists.index(min_dist)]

        state.foot_distance_to_ball = min_dist
        state.foot_screen_pos = closest_foot
        state.foot_near_ball = min_dist < settings.MAX_FOOT_TO_BALL_DISTANCE

        # Real foot screen velocity (pixels/second)
        if prev and prev.foot_screen_pos:
            dt = state.timestamp - prev.timestamp
            if dt > 0:
                foot_dx = closest_foot[0] - prev.foot_screen_pos[0]
                foot_dy = closest_foot[1] - prev.foot_screen_pos[1]
                state.foot_velocity = math.hypot(foot_dx, foot_dy) / dt

    def _find_kicks(self, states: List[FrameBallState]) -> List[DetectedKick]:
        """
        Gate-based kick detection.  All gates must pass; any failure rejects.

        Gates (in order):
          G1  Foot acceleration spike (foot speed increases >80%)
          G2  Foot proximity         (within MAX_FOOT_TO_BALL_DISTANCE)
          G3  Ball was stationary    (pre-kick ball velocity < 40 px/s)
          G4  Ball moves after kick  (post-kick ball velocity > MIN_KICK_VELOCITY)

        Confidence boosters (additive, capped at 1.0):
          +0.10  Ball direction changes significantly (>30°) after contact
          +0.10  Ball acceleration spike in the 4 frames following the kick

        Base confidence when all 4 gates pass: 0.70
        Required final confidence: > 0.70  (i.e. at least one booster must fire,
        or the base signals are very strong — see inline notes)
        """
        # Build suspicious-frame set (ball just reappeared after a gap)
        suspicious_frames = self._build_suspicious_frames(states)

        raw_events: List[DetectedKick] = []

        for i, state in enumerate(states):
            if state.detection is None:
                continue
            if i in suspicious_frames:
                logger.debug("Frame %d: skipped (detection recovery window)", i)
                continue

            # ---- G1: foot acceleration spike ----
            if not self._gate_foot_spike(states, i):
                continue

            # ---- G2: foot proximity ----
            if not state.foot_near_ball:
                logger.debug(
                    f"Frame {i}: G2 fail — foot dist={state.foot_distance_to_ball:.0f}px"
                )
                continue

            # ---- G3: ball was stationary BEFORE this frame ----
            ball_was_stationary, pre_vel = self._gate_ball_stationary_before(states, i)
            if not ball_was_stationary:
                logger.debug(
                    f"Frame {i}: G3 fail — ball pre-velocity={pre_vel:.0f}px/s (not stationary)"
                )
                continue

            # ---- G4: ball moves fast AFTER this frame ----
            ball_moves_after, post_vel = self._gate_ball_moves_after(states, i)
            if not ball_moves_after:
                logger.debug(
                    f"Frame {i}: G4 fail — ball post-velocity={post_vel:.0f}px/s (too slow)"
                )
                continue

            # All gates passed — compute confidence
            confidence = 0.70  # base for passing all 4 gates

            method_parts = ["foot_spike", "proximity", "ball_stationary", "ball_fast"]

            # Booster: significant direction change
            if self._boost_direction_change(states, i):
                confidence += 0.10
                method_parts.append("dir_change")

            # Booster: ball acceleration spike right after contact
            if self._boost_ball_acceleration(states, i):
                confidence += 0.10
                method_parts.append("ball_accel")

            # Booster: strong foot velocity (high confidence kick motion)
            foot_vel_ratio = state.foot_velocity / max(
                1, settings.MIN_FOOT_VELOCITY_FOR_KICK
            )
            if foot_vel_ratio > 2.0:
                confidence += 0.10
                method_parts.append("strong_foot")

            confidence = round(min(1.0, confidence), 3)

            # Final threshold — require at least one booster to fire
            if confidence <= 0.70:
                logger.debug(
                    f"Frame {i}: below final threshold (conf={confidence:.2f}, no boosters fired)"
                )
                continue

            raw_events.append(
                DetectedKick(
                    kick_index=0,
                    frame_number=state.frame_number,
                    timestamp_seconds=state.timestamp,
                    confidence_score=confidence,
                    detection_method="+".join(method_parts),
                )
            )

        return self._merge_events(raw_events, settings.MIN_FRAMES_BETWEEN_KICKS)

    def _gate_foot_spike(self, states: List[FrameBallState], i: int) -> bool:
        """
        G1: Foot speed must jump by >80% in one step AND exceed
        MIN_FOOT_VELOCITY_FOR_KICK absolute threshold.

        Using real screen velocity (pixels/second), not foot-to-ball distance,
        so normal walking approach does NOT trigger this gate.
        """
        curr = states[i]
        if i == 0:
            return False

        prev = states[i - 1]
        curr_fv = curr.foot_velocity
        prev_fv = prev.foot_velocity

        if curr_fv < settings.MIN_FOOT_VELOCITY_FOR_KICK:
            return False  # Absolute minimum not met

        if prev_fv <= 0:
            return False  # Cannot compute ratio without previous velocity

        relative_increase = (curr_fv - prev_fv) / prev_fv
        if relative_increase < 0.80:  # Require 80% jump (was 50%, too permissive)
            return False

        logger.debug(
            f"Frame {i}: G1 pass — foot_vel {prev_fv:.0f}→{curr_fv:.0f} px/s "
            f"(+{relative_increase:.0%})"
        )
        return True

    def _gate_ball_stationary_before(
        self, states: List[FrameBallState], i: int, window: int = 5
    ) -> Tuple[bool, float]:
        """
        G3: Ball must be near-stationary in the N frames before the kick frame.

        Free kicks always start with a stationary ball.
        Dribbling / running past / tracking glitches all fail this gate.

        Returns (passed, max_pre_velocity).
        """
        start = max(0, i - window)
        pre_vels = [
            states[j].velocity
            for j in range(start, i)
            if states[j].detection is not None
        ]

        if len(pre_vels) < 2:
            # Not enough data — be conservative and reject
            return False, float("inf")

        max_pre_vel = max(pre_vels)
        # Ball must be very slow — tighter than original 30px/s to survive real noise
        threshold = 50.0  # px/s  (≈3px/frame at 15fps sample rate)
        return max_pre_vel < threshold, max_pre_vel

    def _gate_ball_moves_after(
        self, states: List[FrameBallState], i: int, window: int = 2
    ) -> Tuple[bool, float]:
        """
        G4: Ball must reach MIN_KICK_VELOCITY within N frames after contact.

        Ensures the kick actually propelled the ball. Eliminates cases where
        the foot was near the ball but didn't make significant contact
        (e.g., player reaching over the ball, feint, stepping over).

        Returns (passed, max_post_velocity).
        """
        end = min(len(states), i + window + 1)
        post_vels = [
            states[j].velocity
            for j in range(i + 1, end)
            if states[j].detection is not None
        ]

        if not post_vels:
            return False, 0.0

        max_post_vel = max(post_vels)
        return max_post_vel >= settings.MIN_KICK_VELOCITY, max_post_vel

    def _boost_direction_change(
        self, states: List[FrameBallState], i: int, window: int = 3
    ) -> bool:
        """
        Booster: Ball direction shifts by more than 30° at the kick frame.

        Calculates average direction before vs. after kick.
        """
        pre_dirs = [
            states[j].motion_direction
            for j in range(max(0, i - window), i)
            if states[j].detection is not None and states[j].velocity > 10
        ]
        post_dirs = [
            states[j].motion_direction
            for j in range(i + 1, min(len(states), i + window + 1))
            if states[j].detection is not None and states[j].velocity > 10
        ]

        if not pre_dirs or not post_dirs:
            return False

        avg_pre = sum(pre_dirs) / len(pre_dirs)
        avg_post = sum(post_dirs) / len(post_dirs)

        diff = abs(avg_post - avg_pre)
        diff = min(diff, 360 - diff)  # Handle wrap-around
        return diff > 30.0

    def _boost_ball_acceleration(
        self, states: List[FrameBallState], i: int, window: int = 4
    ) -> bool:
        """
        Booster: Ball shows a significant acceleration spike in the frames
        immediately following the kick.
        """
        for j in range(i + 1, min(len(states), i + window + 1)):
            if states[j].acceleration >= settings.MIN_KICK_ACCELERATION:
                return True
        return False

    def _build_suspicious_frames(
        self, states: List[FrameBallState], suppress_window: int = 5
    ) -> set:
        """
        Mark frames where ball just reappeared after a gap as suspicious.

        These frames often look like velocity spikes (ball "jumped") but are
        really tracking failures.  We suppress kick detection here.
        """
        suspicious: set = set()
        consecutive_missing = 0

        for i, state in enumerate(states):
            if state.detection is None:
                consecutive_missing += 1
            else:
                if consecutive_missing >= 2:
                    for j in range(i, min(i + suppress_window, len(states))):
                        suspicious.add(j)
                consecutive_missing = 0

        return suspicious

    def _merge_events(
        self, events: List[DetectedKick], min_gap_frames: int
    ) -> List[DetectedKick]:
        """
        Merge events closer than min_gap_frames, keeping the highest-confidence
        one, then renumber sequentially from 1.
        """
        if not events:
            return []

        events = sorted(events, key=lambda e: e.frame_number)
        merged: List[DetectedKick] = [events[0]]

        for event in events[1:]:
            if event.frame_number - merged[-1].frame_number < min_gap_frames:
                if event.confidence_score > merged[-1].confidence_score:
                    merged[-1] = event
            else:
                merged.append(event)

        for idx, kick in enumerate(merged, start=1):
            kick.kick_index = idx

        return merged

    def _log_frame(
        self, frame_idx: int, timestamp: float, state: FrameBallState
    ) -> None:
        if state.detection:
            parts = [f"v={state.velocity:.0f}px/s", f"a={state.acceleration:.0f}"]
            if state.pose:
                parts.append(f"foot_dist={state.foot_distance_to_ball:.0f}px")
                parts.append(f"foot_vel={state.foot_velocity:.0f}px/s")
            logger.debug(
                "Frame %5d | %.2fs | %s",
                frame_idx,
                timestamp,
                ", ".join(parts),
            )
        else:
            logger.debug("Frame %5d | %.2fs | ball not found", frame_idx, timestamp)


if __name__ == "__main__":
    import sys
    import cv2
    import time
    import numpy as np
    from pathlib import Path
    from app.services.video_loader import validate_and_get_info, iter_frames
    from app.services.ball_detector import BallDetector
    from app.services.pose_estimator import PoseEstimator, POSE_CONNECTIONS

    logging.basicConfig(level=logging.INFO)

    # ---- drawing helpers ------------------------------------------------

    def draw_skeleton(frame, pose, color=(0, 255, 0), thickness=2):
        if not pose or not pose.keypoints:
            return
        for start_idx, end_idx in POSE_CONNECTIONS:
            s = pose.get_keypoint(start_idx)
            e = pose.get_keypoint(end_idx)
            if s and e and s.visibility > 0.3 and e.visibility > 0.3:
                cv2.line(frame, (s.x, s.y), (e.x, e.y), color, thickness)
        for kp in pose.keypoints:
            if kp.visibility > 0.3:
                c = (0, 255, 0) if kp.visibility > 0.7 else (0, 165, 255)
                cv2.circle(frame, (kp.x, kp.y), 4, c, -1)

    def draw_foot_to_ball(frame, pose, ball_detection, color=(255, 100, 0)):
        if not pose or not ball_detection:
            return
        lf = pose.get_keypoint(31)
        rf = pose.get_keypoint(32)
        if not lf or not rf:
            return
        ld = math.hypot(
            lf.x - ball_detection.center[0], lf.y - ball_detection.center[1]
        )
        rd = math.hypot(
            rf.x - ball_detection.center[0], rf.y - ball_detection.center[1]
        )
        foot = lf if ld < rd else rf
        dist = min(ld, rd)
        cv2.line(frame, (foot.x, foot.y), ball_detection.center, color, 2)
        mid = (
            (foot.x + ball_detection.center[0]) // 2,
            (foot.y + ball_detection.center[1]) // 2,
        )
        cv2.putText(
            frame,
            f"Dist:{dist:.0f}px",
            (mid[0] - 30, mid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    test_video = Path("/content/video_9 (online-video-cutter.com).mp4")
    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        sys.exit(1)

    try:
        print("\n=== Initialising detectors ===")
        ball_detector = BallDetector()
        pose_estimator = PoseEstimator()
        kick_detector = KickDetector(ball_detector)

        print("\n=== Loading video ===")
        video_info = validate_and_get_info(str(test_video))
        print(
            f"  {video_info.filename}  {video_info.width}x{video_info.height} "
            f"@ {video_info.fps:.2f}fps  ({video_info.total_frames} frames)"
        )

        print("\n=== Detecting kicks ===")
        t0 = time.perf_counter()
        kicks = kick_detector.detect_kicks(video_info)
        elapsed = time.perf_counter() - t0
        print(f"  {len(kicks)} kick(s) found in {elapsed:.2f}s")
        for k in kicks:
            print(
                f"  Kick {k.kick_index}: frame={k.frame_number}, "
                f"t={k.timestamp_seconds:.2f}s, conf={k.confidence_score:.0%}, "
                f"method={k.detection_method}"
            )

        print("\n=== Saving annotated video ===")
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / "kick_detection_output.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            video_info.fps,
            (video_info.width, video_info.height),
        )

        kick_frames = {k.frame_number: k for k in kicks}
        last_kick_label = ""
        ball_detector.init_video(video_path=video_info.path)

        for frame_idx, timestamp, frame in iter_frames(
            str(test_video), sample_every_n=1
        ):
            tracked = ball_detector.track(frame, frame_id=frame_idx)
            pose = pose_estimator.detect(frame, timestamp_ms=int(timestamp * 1000))

            annotated = frame.copy()
            if pose:
                draw_skeleton(annotated, pose)
                if tracked:
                    draw_foot_to_ball(annotated, pose, tracked)

            if tracked:
                cv2.rectangle(
                    annotated,
                    (tracked.x1, tracked.y1),
                    (tracked.x2, tracked.y2),
                    (0, 165, 255),
                    2,
                )
                cv2.circle(annotated, tracked.center, 5, (0, 255, 0), -1)

            if frame_idx in kick_frames:
                k = kick_frames[frame_idx]
                last_kick_label = f"KICK #{k.kick_index}  {k.confidence_score:.0%}"

            lines = [f"Frame {frame_idx} | {timestamp:.2f}s"]
            if tracked:
                lines.append(f"Ball conf={tracked.confidence:.0%}")
            if last_kick_label:
                lines.append(last_kick_label)

            for row, line in enumerate(lines):
                is_kick = "KICK" in line
                cv2.putText(
                    annotated,
                    line,
                    (10, 25 + row * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 200) if is_kick else (50, 50, 50),
                    2 if is_kick else 1,
                )

            writer.write(annotated)
            if frame_idx >= video_info.total_frames:
                break

        writer.release()
        if output_video.exists():
            size_mb = output_video.stat().st_size / 1024 / 1024
            print(f"  Saved: {output_video} ({size_mb:.1f} MB)")
        else:
            print("  Warning: output file not created.")

    except Exception as exc:
        import traceback

        print(f"\nFailed: {exc}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            pose_estimator.close()
        except Exception:
            pass
