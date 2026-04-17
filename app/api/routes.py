"""
API Routes — thin HTTP layer.

This file handles ONLY:
  - Receiving the HTTP request
  - Saving the uploaded file temporarily
  - Calling the pipeline
  - Mapping domain exceptions → HTTP error responses
  - Returning the JSON response

All business logic lives in the services layer.
"""

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import (
    VideoLoadError,
    VideoTooLargeError,
    UnsupportedFormatError,
    NoKicksDetectedError,
    ClipExtractionError,
    ModelLoadError,
    FreekickAnalyzerError,
)
from app.models.schemas import AnalysisResponse, ErrorResponse
from app.services.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Reuse one pipeline instance per process (it holds the model references)
_pipeline = AnalysisPipeline()

SUPPORTED_MIME_TYPES = {
    "video/mp4", "video/avi", "video/quicktime",
    "video/x-msvideo", "video/x-matroska", "video/webm",
}


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request (invalid video, unsupported format, etc.)"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "No kicks detected"},
        500: {"model": ErrorResponse, "description": "Internal processing error"},
    },
    summary="Analyze a free-kick video",
    description=(
        "Upload a football free-kick video. The server will detect kick events, "
        "split the video into individual clips, and annotate each clip with "
        "player pose, ball bounding box, and ball trajectory. "
        "Returns a JSON response with kick count, timestamps, and clip file paths."
    ),
    tags=["Analysis"],
)
async def analyze_video(
    video: UploadFile = File(..., description="Football free-kick video file (.mp4, .avi, .mov, .mkv)"),
):
    """
    POST /api/v1/analyze

    Body: multipart/form-data with field `video`
    """
    # ── 1. Basic content-type check ───────────────────────────────────────────
    if video.content_type and video.content_type not in SUPPORTED_MIME_TYPES:
        # Be lenient — some clients send generic types. Log but don't block.
        logger.warning(f"Unexpected content-type: {video.content_type}")

    # ── 2. Size check before saving ───────────────────────────────────────────
    # We read in chunks to avoid loading the whole file into memory
    temp_dir = settings.TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)

    ext = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{ext}")

    try:
        total_bytes = 0
        max_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024

        with open(temp_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):  # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise VideoTooLargeError(
                        f"Upload exceeds {settings.MAX_VIDEO_SIZE_MB} MB limit"
                    )
                f.write(chunk)

        logger.info(f"Saved upload: {video.filename} → {temp_path} ({total_bytes/1024/1024:.1f} MB)")

        # ── 3. Run the analysis pipeline ───────────────────────────────────────
        result = await _pipeline.run(temp_path)
        return result

    except VideoTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error_code": "VIDEO_TOO_LARGE", "message": exc.message},
        )
    except UnsupportedFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "UNSUPPORTED_FORMAT", "message": exc.message, "detail": exc.detail},
        )
    except VideoLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "VIDEO_LOAD_ERROR", "message": exc.message, "detail": exc.detail},
        )
    except NoKicksDetectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error_code": "NO_KICKS_DETECTED", "message": exc.message, "detail": exc.detail},
        )
    except ModelLoadError as exc:
        logger.critical(f"Model load failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "MODEL_LOAD_ERROR", "message": exc.message},
        )
    except FreekickAnalyzerError as exc:
        logger.error(f"Pipeline error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "PROCESSING_ERROR", "message": exc.message, "detail": exc.detail},
        )
    except Exception as exc:
        logger.exception(f"Unexpected error during analysis: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
        )
    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"Cleaned up temp file: {temp_path}")


@router.get(
    "/clips",
    summary="List all generated clips",
    tags=["Clips"],
)
async def list_clips():
    """Return a list of all clip files currently stored on disk."""
    clips_dir = settings.CLIPS_DIR
    if not os.path.exists(clips_dir):
        return {"clips": []}

    files = [
        {
            "filename": f,
            "url": f"/clips/{f}",
            "size_mb": round(os.path.getsize(os.path.join(clips_dir, f)) / (1024 * 1024), 2),
        }
        for f in sorted(os.listdir(clips_dir))
        if f.endswith(".mp4")
    ]
    return {"total": len(files), "clips": files}
