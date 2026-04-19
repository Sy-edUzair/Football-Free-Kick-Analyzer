"""
Video Merger — combines multiple annotated clip videos into a single output video.

This service takes individual kick clip videos (already annotated with pose + ball)
and merges them sequentially into one consolidated video file.

Design rationale:
  - Reduces download overhead (one file instead of N)
  - Keeps separation of concerns (clip generation separate from merging)
  - Reuses existing annotated clips produced by ClipExtractor
  - Supports cleanup of temporary clips after merge

Usage:
    merger = VideoMerger(output_dir=settings.OUTPUT_DIR)
    merged_path = merger.merge_clips(
        clip_paths=["/path/clip_1.mp4", "/path/clip_2.mp4"],
        output_filename="merged_kicks.mp4"
    )
"""

import cv2
import logging
import os
from typing import List, Optional

from app.core.config import settings
from app.core.exceptions import ClipExtractionError

logger = logging.getLogger(__name__)


class VideoMerger:
    """Merges multiple video clips into a single output video."""

    def __init__(self, output_dir: str = None):
        """
        Initialize the merger.

        Args:
            output_dir: Directory to save merged video. Defaults to settings.OUTPUT_DIR
        """
        self._output_dir = output_dir or settings.OUTPUT_DIR

    def merge_clips(
        self,
        clip_paths: List[str],
        output_filename: str = "merged_kicks.mp4",
    ) -> str:
        """
        Merge multiple video clips into a single video.

        The clips are concatenated in the order provided. All clips should have
        the same resolution and frame rate for best results.

        Args:
            clip_paths: List of absolute paths to video clips to merge
            output_filename: Name of the output merged video file

        Returns:
            str: Absolute path to the merged video file

        Raises:
            ClipExtractionError: If merge fails (missing files, codec issues, etc.)
        """
        if not clip_paths:
            raise ClipExtractionError(
                "Cannot merge: no clip paths provided",
                detail="clip_paths list is empty",
            )

        # Validate that all clips exist
        for clip_path in clip_paths:
            if not os.path.exists(clip_path):
                raise ClipExtractionError(
                    f"Clip not found: {clip_path}",
                    detail=f"Cannot merge non-existent file",
                )

        os.makedirs(self._output_dir, exist_ok=True)
        output_path = os.path.join(self._output_dir, output_filename)

        logger.info(
            f"Merging {len(clip_paths)} clip(s) into: {output_filename}"
        )

        # Read first clip to get video properties
        first_clip = cv2.VideoCapture(clip_paths[0])
        if not first_clip.isOpened():
            raise ClipExtractionError(
                f"Cannot open first clip: {clip_paths[0]}",
                detail="Failed to read video properties",
            )

        fps = first_clip.get(cv2.CAP_PROP_FPS)
        width = int(first_clip.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_clip.release()

        logger.debug(f"Video properties: {width}x{height} @ {fps}fps")

        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise ClipExtractionError(
                f"Cannot open VideoWriter for {output_path}",
                detail="Failed to initialize video writer",
            )

        total_frames_written = 0

        # Process each clip
        for clip_idx, clip_path in enumerate(clip_paths, 1):
            logger.debug(f"Processing clip {clip_idx}/{len(clip_paths)}: {clip_path}")

            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                writer.release()
                raise ClipExtractionError(
                    f"Cannot open clip: {clip_path}",
                    detail=f"Failed to read clip {clip_idx}",
                )

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Ensure frame matches expected dimensions
                if frame.shape[:2] != (height, width):
                    logger.warning(
                        f"Clip {clip_idx} frame {frame_count}: dimension mismatch "
                        f"({frame.shape[1]}x{frame.shape[0]} vs {width}x{height}), skipping"
                    )
                    continue

                writer.write(frame)
                frame_count += 1
                total_frames_written += 1

            cap.release()
            logger.debug(
                f"  Clip {clip_idx}: {frame_count} frames written"
            )

        writer.release()

        logger.info(
            f"Merge complete: {output_filename} | "
            f"{len(clip_paths)} clips | "
            f"{total_frames_written} total frames"
        )

        return output_path

    def cleanup_clips(self, clip_paths: List[str]) -> None:
        """
        Delete temporary clip files after merge is complete.

        Useful for removing intermediate clips from TEMP_DIR after creating
        the final merged video.

        Args:
            clip_paths: List of clip file paths to delete
        """
        for clip_path in clip_paths:
            try:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                    logger.debug(f"Deleted temporary clip: {clip_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {clip_path}: {e}")
