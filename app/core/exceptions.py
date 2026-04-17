"""
Custom exceptions.

Why custom exceptions?  Because catching `Exception` everywhere hides bugs.
These give us precise, readable error handling at every layer.
"""


class FreekickAnalyzerError(Exception):
    """Base exception for all application errors."""
    def __init__(self, message: str, detail: str = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


class VideoLoadError(FreekickAnalyzerError):
    """Raised when a video file cannot be opened or is corrupted."""
    pass


class VideoTooLargeError(FreekickAnalyzerError):
    """Raised when uploaded video exceeds the size limit."""
    pass


class UnsupportedFormatError(FreekickAnalyzerError):
    """Raised when the video format is not supported."""
    pass


class NoKicksDetectedError(FreekickAnalyzerError):
    """Raised when no kick events are found in the video."""
    pass


class ClipExtractionError(FreekickAnalyzerError):
    """Raised when a clip cannot be saved to disk."""
    pass


class ModelLoadError(FreekickAnalyzerError):
    """Raised when YOLO or MediaPipe models fail to initialise."""
    pass
