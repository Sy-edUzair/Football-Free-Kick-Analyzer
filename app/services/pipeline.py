"""
Analysis Pipeline — the orchestrator.

This is the single entry point for a full analysis run.  It wires together
all the individual services in the correct order:

  VideoLoader → KickDetector → ClipExtractor → AnalysisResponse

Why have a separate pipeline class?
  The FastAPI route handler should be thin (handle HTTP concerns only).
  The pipeline holds all the domain logic and is independently testable.
"""

import logging
import time
import math
import os
import cv2
from datetime import datetime
from typing import Optional

from app.core.config import settings
from app.core.exceptions import NoKicksDetectedError
from app.models.schemas import (
    AnalysisResponse,
    VideoMetadata,
    KickEvent,
)
from app.services.video_loader import validate_and_get_info
from app.services.ball_detector import BallDetector
from app.services.pose_estimator import PoseEstimator
from app.services.kick_detector import KickDetector, FrameBallState
from app.services.clip_extractor import ClipExtractor
from app.services.annotator import FrameAnnotator

logger = logging.getLogger(__name__)

# Module-level singletons — loaded once on first request, reused afterwards.
# This avoids reloading heavy ML models on every API call.
_ball_detector: BallDetector = None
_pose_estimator: PoseEstimator = None


def _get_detectors():
    """Lazy initialise and return the shared detector instances."""
    global _ball_detector, _pose_estimator
    if _ball_detector is None:
        logger.info("Initialising BallDetector (first request)...")
        _ball_detector = BallDetector()
    if _pose_estimator is None:
        logger.info("Initialising PoseEstimator (first request)...")
        _pose_estimator = PoseEstimator()
    return _ball_detector, _pose_estimator


class VideoAnnotator:
    """
    Annotates an entire video with detections and metrics.

    Uses KickDetector to compute metrics, then renders the full video.
    Unlike ClipExtractor which creates clips around kicks, this annotates
    the whole video frame-by-frame for visual inspection of results.

    Full pipeline:
      1. Load & validate video
      2. Detect kicks
      3. Annotate & render full video
      4. Return metadata and response
    """

    def __init__(
        self,
        ball_detector: BallDetector,
        pose_estimator: PoseEstimator,
        output_dir: str = None,
    ):
        self._ball_detector = ball_detector
        self._pose_estimator = pose_estimator
        self._output_dir = output_dir or settings.OUTPUT_DIR

    def annotate_full_video(
        self, video_path: str, output_filename: str = "annotated_full_video.mp4"
    ) -> dict:
        """
        Complete pipeline: load video → detect kicks → annotate → return response.

        Steps:
          1. Load & validate video
          2. Detect kicks
          3. Annotate full video
          4. Build response

        Returns: Dict with video_info, detected_kicks, output_path, and metadata.
        """
        logger.info(f"Starting full video annotation pipeline: {video_path}")
        start_time = time.perf_counter()

        # Step 1: Load & validate video
        logger.info("Step 1: Loading and validating video...")
        video_info = validate_and_get_info(video_path)
        logger.info(
            f"  Video loaded: {video_info.filename} | "
            f"{video_info.width}x{video_info.height} @ {video_info.fps}fps | "
            f"{video_info.total_frames} frames"
        )

        # Step 2: Detect kicks
        logger.info("Step 2: Detecting kicks...")
        kick_detector = KickDetector(self._ball_detector)
        detected_kicks = kick_detector.detect_kicks(video_info)
        logger.info(f"  Detected {len(detected_kicks)} kick(s)")
        for k in detected_kicks:
            logger.info(
                f"    Kick {k.kick_index}: frame={k.frame_number}, "
                f"time={k.timestamp_seconds:.2f}s, conf={k.confidence_score:.2f}"
            )

        # Step 3: Annotate full video
        logger.info("Step 3: Annotating full video with metrics...")
        output_path = self._render_annotated_video(
            video_info, kick_detector, detected_kicks, output_filename
        )
        logger.info(f"  Video saved: {output_path}")

        # Step 4: Build response
        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info(f"Step 4: Building response...")
        logger.info(f"  Pipeline complete in {elapsed}s")

        response = {
            "success": True,
            "message": f"Full video annotation complete. Detected {len(detected_kicks)} kick(s).",
            "video_metadata": {
                "filename": video_info.filename,
                "duration_seconds": round(video_info.duration_seconds, 3),
                "fps": round(video_info.fps, 2),
                "total_frames": video_info.total_frames,
                "width": video_info.width,
                "height": video_info.height,
                "file_size_mb": round(video_info.file_size_mb, 2),
            },
            "kick_events": [
                {
                    "kick_index": k.kick_index,
                    "frame_number": k.frame_number,
                    "timestamp_seconds": round(k.timestamp_seconds, 3),
                    "confidence_score": k.confidence_score,
                }
                for k in detected_kicks
            ],
            "annotated_video_path": output_path,
            "annotated_video_filename": output_filename,
            "processing_time_seconds": elapsed,
        }

        return response

    def _render_annotated_video(
        self, video_info, kick_detector, detected_kicks, output_filename: str
    ) -> str:
        """
        Render the full video with annotations and metrics.

        Returns: Path to saved annotated video.
        """
        os.makedirs(self._output_dir, exist_ok=True)
        output_path = os.path.join(self._output_dir, output_filename)

        # Collect FrameBallState for all frames (metrics already computed)
        from app.services.video_loader import iter_frames

        frame_states = []
        frames_data = []  # Store frames alongside states
        logger.debug("Computing metrics for all frames using KickDetector logic...")
        for frame_idx, timestamp, frame in iter_frames(
            video_info.path, sample_every_n=1
        ):
            ball = self._ball_detector.detect(frame)
            pose = self._pose_estimator.detect(
                frame, timestamp_ms=int(timestamp * 1000)
            )

            state = FrameBallState(
                frame_number=frame_idx,
                timestamp=timestamp,
                detection=ball,
                pose=pose,
            )

            if frame_states and ball and frame_states[-1].detection:
                kick_detector._compute_ball_motion(state, frame_states[-1])

            if pose and ball:
                kick_detector._compute_foot_metrics(
                    state, frame_states[-1] if frame_states else None
                )

            frame_states.append(state)
            frames_data.append(frame)  # Keep frame in sync with state

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info.fps,
            (video_info.width, video_info.height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

        # Build a frame-to-kick mapping for tracking current kick
        # Map frame numbers to kick indices
        frame_to_kick_index = {}
        for kick in detected_kicks:
            frame_to_kick_index[kick.frame_number] = kick.kick_index

        # Create annotator (will be updated per frame)
        annotator = FrameAnnotator(kick_index=0)

        # Annotate frames from computed states
        frame_count = 0
        current_kick_index = 0

        for state, frame in zip(frame_states, frames_data):
            # Update kick_index if this frame is a detected kick
            if state.frame_number in frame_to_kick_index:
                current_kick_index = frame_to_kick_index[state.frame_number]
                logger.debug(
                    f"Frame {state.frame_number}: Kick detected! Updated to Kick #{current_kick_index}"
                )

            # Update the annotator with current kick index
            annotator.kick_index = current_kick_index

            # Build metrics dict from computed state
            metrics = {}
            if state.velocity is not None:
                metrics["ball_velocity"] = state.velocity
            if state.foot_velocity is not None:
                metrics["foot_velocity"] = state.foot_velocity
            if state.foot_distance_to_ball != float("inf"):
                metrics["displacement_from_foot"] = state.foot_distance_to_ball

            # Annotate and write
            annotated = annotator.annotate(
                frame, state.detection, state.pose, state.timestamp, metrics
            )
            writer.write(annotated)
            frame_count += 1

        writer.release()

        logger.info(f"Annotated video saved: {output_filename} | {frame_count} frames")

        return output_path


class AnalysisPipeline:
    """
    Orchestrates a full free-kick analysis.

    Usage:
        pipeline = AnalysisPipeline()
        result   = await pipeline.run(video_path)
    """

    async def run(self, video_path: str) -> AnalysisResponse:
        """
        Execute the full pipeline and return a structured response.

        Steps:
          1. Load & validate video
          2. Detect kicks
          3. Extract & annotate clips
          4. Build response
        """
        start_time = time.perf_counter()
        logger.info(f"Pipeline started for: {video_path}")

        #  Load video
        video_info = validate_and_get_info(video_path)

        video_metadata = VideoMetadata(
            filename=video_info.filename,
            duration_seconds=round(video_info.duration_seconds, 3),
            fps=round(video_info.fps, 2),
            total_frames=video_info.total_frames,
            width=video_info.width,
            height=video_info.height,
            file_size_mb=round(video_info.file_size_mb, 2),
        )

        # Detect kicks
        ball_detector, pose_estimator = _get_detectors()
        kick_detector = KickDetector(ball_detector)
        detected_kicks = kick_detector.detect_kicks(video_info)

        if not detected_kicks:
            raise NoKicksDetectedError(
                "No kick events were detected in this video.",
                detail=(
                    "Try a video with clearer ball visibility or adjust "
                    "BALL_CONFIDENCE_THRESHOLD and MIN_KICK_DISPLACEMENT in config."
                ),
            )

        kick_events = [
            KickEvent(
                kick_index=k.kick_index,
                frame_number=k.frame_number,
                timestamp_seconds=round(k.timestamp_seconds, 3),
                confidence_score=k.confidence_score,
            )
            for k in detected_kicks
        ]

        #  Extract & annotate clips
        extractor = ClipExtractor(
            ball_detector=ball_detector,
            pose_estimator=pose_estimator,
            output_dir=settings.CLIPS_DIR,
        )
        clip_details = extractor.extract_all(video_info, detected_kicks)

        #  Build response
        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info(
            f"Pipeline complete in {elapsed}s. Kicks found: {len(detected_kicks)}"
        )

        return AnalysisResponse(
            success=True,
            message=f"Analysis complete. Detected {len(detected_kicks)} kick(s).",
            analyzed_at=datetime.utcnow(),
            video_metadata=video_metadata,
            total_kicks_detected=len(detected_kicks),
            kick_events=kick_events,
            clips=clip_details,
            processing_time_seconds=elapsed,
        )


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # async def test_pipeline():
    #     """Test the full analysis pipeline."""
    #     # Try to find a test video
    #     test_video = None
    #     candidates = [
    #         Path("dataset/test_video.mp4"),
    #         Path("test_video.mp4"),
    #         Path("/content/video_9.mp4"),
    #     ]

    #     for candidate in candidates:
    #         if candidate.exists():
    #             test_video = str(candidate)
    #             break

    #     if not test_video:
    #         print("Test video not found. Tried:")
    #         for c in candidates:
    #             print(f"  - {c}")
    #         sys.exit(1)

    #     print(f"\n{'='*60}")
    #     print(f"Testing Pipeline with: {test_video}")
    #     print(f"{'='*60}\n")

    #     try:
    #         pipeline = AnalysisPipeline()
    #         result = await pipeline.run(test_video)

    #         print(f"\n{'='*60}")
    #         print("✓ PIPELINE TEST PASSED")
    #         print(f"{'='*60}")
    #         print(f"Message: {result.message}")
    #         print(f"Kicks Detected: {result.total_kicks_detected}")
    #         print(f"Processing Time: {result.processing_time_seconds}s")
    #         print(f"\nVideo Metadata:")
    #         print(f"  Filename: {result.video_metadata.filename}")
    #         print(f"  Duration: {result.video_metadata.duration_seconds}s")
    #         print(f"  FPS: {result.video_metadata.fps}")
    #         print(
    #             f"  Resolution: {result.video_metadata.width}x{result.video_metadata.height}"
    #         )
    #         print(f"  Total Frames: {result.video_metadata.total_frames}")
    #         print(f"  File Size: {result.video_metadata.file_size_mb} MB")

    #         if result.kick_events:
    #             print(f"\nKick Events:")
    #             for kick in result.kick_events:
    #                 print(
    #                     f"  Kick #{kick.kick_index}: "
    #                     f"frame={kick.frame_number}, "
    #                     f"time={kick.timestamp_seconds}s, "
    #                     f"confidence={kick.confidence_score:.2f}"
    #                 )

    #         if result.clips:
    #             print(f"\nGenerated Clips:")
    #             for clip in result.clips:
    #                 print(
    #                     f"  Clip {clip.kick_index}: "
    #                     f"{clip.clip_filename} "
    #                     f"({clip.frame_count} frames, "
    #                     f"ball={clip.ball_detections}, "
    #                     f"pose={clip.pose_detections})"
    #                 )

    #         print(f"\n{'='*60}\n")

    #     except Exception as exc:
    #         import traceback

    #         print(f"\n{'='*60}")
    #         print("✗ PIPELINE TEST FAILED")
    #         print(f"{'='*60}")
    #         print(f"Error: {exc}\n")
    #         traceback.print_exc()
    #         sys.exit(1)

    async def test_full_video_annotation():
        """Test annotating the full video without clipping."""
        test_video = None
        candidates = [
            Path("dataset/test_video.mp4"),
            Path("test_video.mp4"),
            Path("/content/video_9.mp4"),
        ]

        for candidate in candidates:
            if candidate.exists():
                test_video = str(candidate)
                break

        if not test_video:
            print("Test video not found for annotation test.")
            return

        print(f"\n{'='*60}")
        print(f"Testing Full Video Annotation Pipeline")
        print(f"{'='*60}\n")

        try:
            ball_detector, pose_estimator = _get_detectors()

            annotator = VideoAnnotator(
                ball_detector=ball_detector,
                pose_estimator=pose_estimator,
                output_dir=settings.OUTPUT_DIR,
            )

            response = annotator.annotate_full_video(test_video)

            print(f"\n{'='*60}")
            print("FULL VIDEO ANNOTATION PIPELINE PASSED")
            print(f"{'='*60}")
            print(f"Message: {response['message']}")
            print(f"Processing Time: {response['processing_time_seconds']}s")
            print(f"\nVideo Metadata:")
            vm = response["video_metadata"]
            print(f"  Filename: {vm['filename']}")
            print(f"  Duration: {vm['duration_seconds']}s")
            print(f"  FPS: {vm['fps']}")
            print(f"  Resolution: {vm['width']}x{vm['height']}")
            print(f"  Total Frames: {vm['total_frames']}")
            print(f"  File Size: {vm['file_size_mb']} MB")

            if response["kick_events"]:
                print(f"\nKick Events Detected:")
                for kick in response["kick_events"]:
                    print(
                        f"  Kick #{kick['kick_index']}: "
                        f"frame={kick['frame_number']}, "
                        f"time={kick['timestamp_seconds']}s, "
                        f"confidence={kick['confidence_score']:.2f}"
                    )

            print(f"\nAnnotated Video:")
            print(f"  Path: {response['annotated_video_path']}")
            print(f"  Filename: {response['annotated_video_filename']}")
            print(f"\nYou can now view this video to inspect:")
            print(f"  - Ball detection & trajectory")
            print(f"  - Pose skeleton & keypoints")
            print(f"  - Metrics panel (ball velocity, foot velocity, etc.)")
            print(f"\n{'='*60}\n")

        except Exception as exc:
            import traceback

            print(f"\n{'='*60}")
            print("FULL VIDEO ANNOTATION PIPELINE FAILED")
            print(f"{'='*60}")
            print(f"Error: {exc}\n")
            traceback.print_exc()
            sys.exit(1)

    # Run both tests
    print("\n" + "=" * 60)
    print("RUNNING PIPELINE TESTS")
    print("=" * 60)

    # asyncio.run(test_pipeline())
    asyncio.run(test_full_video_annotation())
