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
from datetime import datetime

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
from app.services.kick_detector import KickDetector
from app.services.clip_extractor import ClipExtractor

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

    async def test_pipeline():
        """Test the full analysis pipeline."""
        # Try to find a test video
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
            print("Test video not found. Tried:")
            for c in candidates:
                print(f"  - {c}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Testing Pipeline with: {test_video}")
        print(f"{'='*60}\n")

        try:
            pipeline = AnalysisPipeline()
            result = await pipeline.run(test_video)

            print(f"\n{'='*60}")
            print("✓ PIPELINE TEST PASSED")
            print(f"{'='*60}")
            print(f"Message: {result.message}")
            print(f"Kicks Detected: {result.total_kicks_detected}")
            print(f"Processing Time: {result.processing_time_seconds}s")
            print(f"\nVideo Metadata:")
            print(f"  Filename: {result.video_metadata.filename}")
            print(f"  Duration: {result.video_metadata.duration_seconds}s")
            print(f"  FPS: {result.video_metadata.fps}")
            print(
                f"  Resolution: {result.video_metadata.width}x{result.video_metadata.height}"
            )
            print(f"  Total Frames: {result.video_metadata.total_frames}")
            print(f"  File Size: {result.video_metadata.file_size_mb} MB")

            if result.kick_events:
                print(f"\nKick Events:")
                for kick in result.kick_events:
                    print(
                        f"  Kick #{kick.kick_index}: "
                        f"frame={kick.frame_number}, "
                        f"time={kick.timestamp_seconds}s, "
                        f"confidence={kick.confidence_score:.2f}"
                    )

            if result.clips:
                print(f"\nGenerated Clips:")
                for clip in result.clips:
                    print(
                        f"  Clip {clip.kick_index}: "
                        f"{clip.clip_filename} "
                        f"({clip.frame_count} frames, "
                        f"ball={clip.ball_detections}, "
                        f"pose={clip.pose_detections})"
                    )

            print(f"\n{'='*60}\n")

        except Exception as exc:
            import traceback

            print(f"\n{'='*60}")
            print("✗ PIPELINE TEST FAILED")
            print(f"{'='*60}")
            print(f"Error: {exc}\n")
            traceback.print_exc()
            sys.exit(1)

    # Run the async test
    asyncio.run(test_pipeline())
