"""
Football Free Kick Analyzer - FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import setup_logging

# Setup logging before anything else
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting Football Free Kick Analyzer...")
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.CLIPS_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    logger.info(f"Output directory: {settings.OUTPUT_DIR}")
    logger.info(f"Clips directory:  {settings.CLIPS_DIR}")
    yield
    # Shutdown
    logger.info("Shutting down Football Free Kick Analyzer...")


app = FastAPI(
    title="Football Free Kick Analyzer",
    description=(
        "A CV-powered API that analyzes football free-kick videos. "
        "Detects kicks, splits into clips, and annotates each with "
        "player pose, ball bounding box, and ball trajectory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (serve generated clips) ──────────────────────────────────────
app.mount("/clips", StaticFiles(directory=settings.CLIPS_DIR), name="clips")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {"message": "Football Free Kick Analyzer is running", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
