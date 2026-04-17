"""
Logging configuration — structured, readable logs with timestamps.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean, readable format."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=date_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.ERROR)
