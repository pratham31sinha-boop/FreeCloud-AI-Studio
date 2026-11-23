"""Uploads and Utility API routes."""

from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from backend.config import settings
from backend.generation.model_registry import model_registry
from backend.storage.manager import storage_manager
from backend.workers.allocation import calculate_worker_allocation

router = APIRouter(prefix="", tags=["Uploads & Utility"])


@router.post("/upload/video")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an original source video to be converted to anime style."""
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    filename = file.filename or "uploaded_video.mp4"
    ext = filename[filename.rfind(".") :].lower() if "." in filename else ""

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Check size constraint
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.upload_max_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.upload_max_mb} MB (received {size_mb:.1f} MB)",
        )

    saved_meta = storage_manager.save_upload(
        filename=filename, stream_or_bytes=content, prefix="orig_vid"
    )
    return {
        "status": "success",
        "file_id": saved_meta["file_id"],
        "filename": saved_meta["original_filename"],
        "size_bytes": saved_meta["size_bytes"],
        "sha256": saved_meta["sha256"],
        "message": "Original video successfully uploaded for anime style recreation",
    }


@router.get("/allocation-estimate")
def get_allocation_estimate(duration_minutes: float = 1.0) -> Dict[str, Any]:
    """Return planned GitHub Actions worker allocation for the given video duration.

    Target allocation:
    - 1 min: 4 workers
    - 2 min: 5 workers
    - 3 min: 6 workers
    - 5 min: 7 workers
    - 10 min: 8 workers
    - 15 min: 10 workers
    """
    workers = calculate_worker_allocation(duration_minutes)
    return {
        "duration_minutes": duration_minutes,
        "planned_workers": workers,
        "note": "Target allocation, not a verified performance guarantee.",
        "table": [
            {"duration": "1 minute", "workers": 4},
            {"duration": "2 minutes", "workers": 5},
            {"duration": "3 minutes", "workers": 6},
            {"duration": "5 minutes", "workers": 7},
            {"duration": "10 minutes", "workers": 8},
            {"duration": "15 minutes", "workers": 10},
        ],
    }


@router.get("/models")
def list_supported_models() -> Dict[str, Any]:
    """List AI models in the modular registry."""
    return {"models": [m.model_dump() for m in model_registry.list_models()]}
