import cv2
import logging
import os
from dataclasses import dataclass
from typing import Generator, Tuple
from pathlib import Path
from app.core.config import settings
from app.core.exceptions import (
    VideoLoadError,
    VideoTooLargeError,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class VideoInfo:
    """Holds all technical metadata for a loaded video."""

    path: str
    filename: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float
    file_size_mb: float


def validate_and_get_info(video_path: str) -> VideoInfo:
    filename = os.path.basename(video_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported format: '{ext}'",
            detail=f"Accepted formats: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
        raise VideoTooLargeError(
            f"Video is {file_size_mb:.1f} MB — limit is {settings.MAX_VIDEO_SIZE_MB} MB",
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoLoadError(
            f"Could not open video: {filename}",
            detail="File may be corrupted or encoded in an unsupported codec.",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0 or width <= 0 or height <= 0:
        raise VideoLoadError(f"Video has invalid dimensions or no frames: {filename}")

    duration = total_frames / fps

    logger.info(
        f"Loaded video: {filename} | "
        f"{width}x{height} @ {fps:.1f}fps | "
        f"{duration:.1f}s | {file_size_mb:.1f}MB"
    )

    return VideoInfo(
        path=video_path,
        filename=filename,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_seconds=duration,
        file_size_mb=file_size_mb,
    )


def iter_frames(
    video_path: str,
    sample_every_n: int = 1,
    resize_to: Tuple[int, int] = None,
    convert_to_rgb: bool = False,
) -> Generator[Tuple[int, float, object], None, None]:
    """
    Yield (frame_number, timestamp_seconds, frame) for every nth frame.

    Args:
        video_path:     Path to the video file.
        sample_every_n: Only yield every nth frame (use for fast scanning).
        resize_to:      Optional (width, height) tuple to resize frames. e.g., (1280, 1280)
        convert_to_rgb: If True, convert BGR to RGB. Default False (OpenCV native BGR).

    Yields:
        (frame_idx, timestamp_sec, processed_frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoLoadError(f"Cannot open video for frame extraction: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                timestamp = frame_idx / fps

                # Apply preprocessing if requested
                processed_frame = frame
                if resize_to is not None:
                    processed_frame = cv2.resize(processed_frame, resize_to)
                if convert_to_rgb:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                yield frame_idx, timestamp, processed_frame

            frame_idx += 1
    finally:
        cap.release()


def read_frame_at(video_path: str, frame_number: int):
    """Read a single specific frame by its index number."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise VideoLoadError(f"Could not read frame {frame_number} from {video_path}")
    return frame


# if __name__ == "__main__":
#     import sys
#     from pathlib import Path

#     # Configure logging
#     logging.basicConfig(level=logging.INFO)

#     # Test video path (modify as needed)
#     test_video = Path(
#         "/home/syed-uzair-hussain-zaidi/Office Work/Tezeract/Pose_Estimation/freekick_analyzer/dataset/video_109.mp4"
#     )

#     if not Path(test_video).exists():
#         print(f"Test video not found: {test_video}")
#         print("Usage: python video_loader.py <path_to_video>")
#         sys.exit(1)

#     try:
#         # Test 1: Validate and get video info
#         print("\n=== Test 1: Video Information ===")
#         info = validate_and_get_info(test_video)
#         print(f"File: {info.filename}")
#         print(f"Resolution: {info.width}x{info.height}")
#         print(f"FPS: {info.fps:.2f}")
#         print(f"Total Frames: {info.total_frames}")
#         print(f"Duration: {info.duration_seconds:.2f}s")
#         print(f"File Size: {info.file_size_mb:.2f}MB")

#         # Test 2: Iterate all frames
#         print("\n=== Test 2: Reading All Frames (1280x1280 resize) ===")
#         frame_count = 0
#         for frame_idx, timestamp, frame in iter_frames(test_video, sample_every_n=1):
#             frame_count += 1
#             if frame_count % 30 == 0:
#                 print(f"  Frame {frame_idx} @ {timestamp:.2f}s | Shape: {frame.shape}")
#         print(f"Total frames read: {frame_count}")

#         # Test 3: Frame hopping (every 3rd frame) with resize
#         print("\n=== Test 3: Frame Hopping (every 3rd frame, 1280x1280 resize) ===")
#         hopped_frames = []
#         for frame_idx, timestamp, frame in iter_frames(test_video, sample_every_n=3):
#             hopped_frames.append(frame_idx)
#         print(f"  Sampled frame indices: {hopped_frames[:10]}...")
#         print(
#             f"Total hopped frames: {len(hopped_frames)} "
#             f"(expected ~{info.total_frames // 3})"
#         )

#         # Test 4: Read specific frame
#         print("\n=== Test 4: Reading Specific Frame ===")
#         frame_num = info.total_frames // 2
#         frame = read_frame_at(test_video, frame_num)
#         print(f"Successfully read frame {frame_num}")
#         print(f"  Shape: {frame.shape}")

#         # Test 5: Save preprocessed video for comparison
#         print("\n=== Test 5: Saving Preprocessed Video (1280x1280) ===")
#         output_dir = Path("outputs")
#         output_dir.mkdir(exist_ok=True)

#         preprocessed_output = output_dir / f"preprocessed_{info.filename}"

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter(
#             str(preprocessed_output),
#             fourcc,
#             info.fps,
#             (1280, 1280)
#         )

#         if not writer.isOpened():
#             print(f"✗ Cannot create video writer for {preprocessed_output}")
#         else:
#             frames_saved = 0
#             for frame_idx, timestamp, frame in iter_frames(test_video, sample_every_n=1):
#                 writer.write(frame)
#                 frames_saved += 1
#                 if frames_saved % 50 == 0:
#                     print(f"  Saved {frames_saved} frames...")

#             writer.release()
#             output_size_mb = preprocessed_output.stat().st_size / (1024 * 1024)
#             print(f"✓ Preprocessed video saved: {preprocessed_output}")
#             print(f"  Original: {info.width}x{info.height} ({info.file_size_mb:.2f}MB)")
#             print(f"  Preprocessed: 1280x1280 ({output_size_mb:.2f}MB)")
#             print(f"  Total frames saved: {frames_saved}")

#         print("\nAll tests passed!")

#     except Exception as e:
#         print(f"\nTest failed: {e}")
#         import traceback

#         traceback.print_exc()
#         sys.exit(1)
