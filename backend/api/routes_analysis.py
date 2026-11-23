"""API routes for Video Analysis and Shot Boundary Extraction."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from backend.config import settings
from backend.storage.manager import storage_manager
from backend.video_analysis.analyzer import VideoAnalysisReport, video_analyzer

router = APIRouter(prefix="/analysis", tags=["Video Analysis"])

DEFAULT_JAGDHATRI_PATH = Path(
    "/home/prathamsinha/Downloads/New Folder/jagadhatri-ep-224-reel-jun-23-2026-zee-tv-1920-ytshorts.savetube.me.mp4"
)


class VideoAnalysisRequest(BaseModel):
    file_id_or_path: str
    max_duration_sec: float = 40.0
    sample_fps: float = 6.0


@router.post("/analyze", response_model=VideoAnalysisReport)
def analyze_video(req: VideoAnalysisRequest) -> VideoAnalysisReport:
    """Analyze an uploaded video or server video file for shot cuts and motion kinematics."""
    # Resolve path: first check uploads dir, then direct path
    candidate = storage_manager.uploads_dir / req.file_id_or_path
    if not candidate.is_file():
        candidate = Path(req.file_id_or_path)

    if not candidate.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file '{req.file_id_or_path}' not found.",
        )

    output_dir = storage_manager.data_dir / "analysis" / candidate.stem
    try:
        report = video_analyzer.analyze_video(
            video_path=candidate,
            output_dir=output_dir,
            max_analyze_duration_sec=req.max_duration_sec,
            sample_fps=req.sample_fps,
        )
        return report
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis failed: {exc}",
        )


@router.get("/jagdhatri-sample", response_model=VideoAnalysisReport)
def analyze_jagdhatri_sample(max_duration_sec: float = 15.0) -> VideoAnalysisReport:
    """One-click analysis of the real Jagdhatri serial episode 224 reference footage."""
    if not DEFAULT_JAGDHATRI_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Jagdhatri sample footage not found at {DEFAULT_JAGDHATRI_PATH}",
        )

    # Copy to uploads so it is tracked in the system
    target_upload = storage_manager.uploads_dir / "jagdhatri_ep224_reel.mp4"
    if not target_upload.is_file():
        shutil.copy2(str(DEFAULT_JAGDHATRI_PATH), str(target_upload))

    output_dir = storage_manager.data_dir / "analysis" / "jagdhatri_ep224"
    report = video_analyzer.analyze_video(
        video_path=target_upload,
        output_dir=output_dir,
        max_analyze_duration_sec=max_duration_sec,
        sample_fps=6.0,
    )
    return report
