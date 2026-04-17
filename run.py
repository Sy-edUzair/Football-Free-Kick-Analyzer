"""
Server entry point.

Run with:
    python run.py

Or directly with uvicorn for more control:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn
from app.core.config import settings
from app.core.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging("DEBUG" if settings.DEBUG else "INFO")
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
