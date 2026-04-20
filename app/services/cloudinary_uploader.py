"""Cloudinary upload helper for generated clip videos."""

import os
import logging

import cloudinary
import cloudinary.uploader

from app.core.config import settings
from app.core.exceptions import CloudUploadError

logger = logging.getLogger(__name__)


class CloudinaryUploader:
    """Uploads clip videos to Cloudinary and returns public URLs."""

    def __init__(self):
        if not settings.CLOUDINARY_CLOUD_NAME:
            raise CloudUploadError(
                "Cloudinary is not configured.",
                detail="Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET.",
            )
        if not settings.CLOUDINARY_API_KEY:
            raise CloudUploadError(
                "Cloudinary is not configured.",
                detail="Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET.",
            )
        if not settings.CLOUDINARY_API_SECRET:
            raise CloudUploadError(
                "Cloudinary is not configured.",
                detail="Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET.",
            )

        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True,
        )

    def upload_clip(self, clip_path: str) -> str:
        """Upload one clip file to Cloudinary and return secure URL."""
        if not os.path.exists(clip_path):
            raise CloudUploadError(
                "Clip file missing before upload.",
                detail=f"Path not found: {clip_path}",
            )

        base_name = os.path.splitext(os.path.basename(clip_path))[0]

        try:
            result = cloudinary.uploader.upload_large(
                clip_path,
                resource_type="video",
                folder=settings.CLOUDINARY_FOLDER,
                public_id=base_name,
                overwrite=True,
            )
        except Exception as exc:
            raise CloudUploadError(
                "Failed to upload clip to Cloudinary.",
                detail=str(exc),
            ) from exc

        secure_url = result.get("secure_url")
        if not secure_url:
            raise CloudUploadError(
                "Cloudinary upload did not return a secure URL.",
                detail=f"Upload response keys: {sorted(result.keys())}",
            )

        logger.debug("Uploaded clip to Cloudinary: %s", secure_url)
        return secure_url

    def upload_full_video(self, video_path: str) -> str:
        """Upload one full annotated video to Cloudinary and return secure URL."""
        if not os.path.exists(video_path):
            raise CloudUploadError(
                "Full annotated video file missing before upload.",
                detail=f"Path not found: {video_path}",
            )

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        try:
            result = cloudinary.uploader.upload_large(
                video_path,
                resource_type="video",
                folder=settings.CLOUDINARY_FULL_VIDEO_FOLDER,
                public_id=base_name,
                overwrite=True,
            )
        except Exception as exc:
            raise CloudUploadError(
                "Failed to upload full annotated video to Cloudinary.",
                detail=str(exc),
            ) from exc

        secure_url = result.get("secure_url")
        if not secure_url:
            raise CloudUploadError(
                "Cloudinary full-video upload did not return a secure URL.",
                detail=f"Upload response keys: {sorted(result.keys())}",
            )

        logger.debug("Uploaded full annotated video to Cloudinary: %s", secure_url)
        return secure_url
