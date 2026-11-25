"""Job management API routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import FileResponse
from backend.generation.pipeline import generation_pipeline
from backend.storage.manager import storage_manager
from backend.workers.models import JobRecord, JobRequest
from backend.workers.orchestrator import worker_orchestrator

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("", response_model=JobRecord, status_code=status.HTTP_201_CREATED)
async def create_job(request: JobRequest) -> JobRecord:
    """Submit a new anime video generation job."""
    return await generation_pipeline.initiate_job(request)


@router.get("", response_model=List[JobRecord])
def list_jobs() -> List[JobRecord]:
    """Retrieve all submitted jobs ordered by creation time."""
    return worker_orchestrator.list_jobs()


@router.get("/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Retrieve live status, chunk allocations, and continuity quality report."""
    status_data = generation_pipeline.get_pipeline_status(job_id)
    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return status_data


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, str]:
    """Cancel an in-flight video generation job."""
    success = worker_orchestrator.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} could not be cancelled or has already completed.",
        )
    return {"message": f"Job {job_id} cancelled"}


@router.post("/{job_id}/boost")
def boost_job_with_turbo_worker(job_id: str) -> Dict[str, Any]:
    """Attach an independent turbo worker to an in-flight job for faster completion."""
    res = worker_orchestrator.boost_job(job_id)
    if not res.get("boosted"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.get("reason", f"Could not boost job {job_id}"),
        )
    return res


@router.get("/{job_id}/video")
def download_final_video(job_id: str):
    """Download or stream the generated coherent video output."""
    output_path = storage_manager.get_output_video(job_id)
    if not output_path or not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Final video output for job {job_id} is not yet available.",
        )
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"{job_id}_anime_final.mp4",
    )


def _serve_anchor_image(anchor_path: Path, filename: str, fmt: Optional[str] = None) -> Response:
    """Serve anchor frame bytes atomically, supporting compact JPEG mode for remote runners."""
    if fmt and fmt.lower() in ("jpg", "jpeg"):
        jpg_path = anchor_path.with_suffix(".jpg")
        try:
            if not jpg_path.is_file() or jpg_path.stat().st_mtime < anchor_path.stat().st_mtime:
                import cv2

                img = cv2.imread(str(anchor_path))
                if img is not None and img.size > 0:
                    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    if ok:
                        tmp_jpg = jpg_path.with_suffix(f".{os.getpid()}.tmp.jpg")
                        tmp_jpg.write_bytes(encoded.tobytes())
                        os.replace(str(tmp_jpg), str(jpg_path))
            if jpg_path.is_file() and jpg_path.stat().st_size > 0:
                return Response(
                    content=jpg_path.read_bytes(),
                    media_type="image/jpeg",
                    headers={"Content-Disposition": f'inline; filename="{Path(filename).stem}.jpg"'},
                )
        except Exception:
            pass

    return Response(
        content=anchor_path.read_bytes(),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.get("/{job_id}/master-anchor")
def get_master_anchor_image(job_id: str, format: Optional[str] = None):
    """Serve the master anchor reference image to guarantee 100% subject visual consistency across workers."""
    job = worker_orchestrator.get_job(job_id)
    prompt = job.request.story_prompt if job else "cinematic high quality video subject"
    clean_story = (prompt or "").strip()
    m_seed = abs(hash(f"{job_id}_{clean_story[:40]}")) % 1000000

    req_w = getattr(job.request, "width", 1920) if job else 1920
    req_h = getattr(job.request, "height", 1080) if job else 1080

    anchor_path = storage_manager.get_or_create_master_anchor(
        job_id=job_id, prompt=clean_story, seed=m_seed, width=req_w, height=req_h
    )
    return _serve_anchor_image(anchor_path, f"{job_id}_master_anchor.png", fmt=format)


@router.get("/{job_id}/anchor/{seq_idx}")
def get_act_anchor_image(job_id: str, seq_idx: int, format: Optional[str] = None):
    """Serve the story-progressive anchor image for the specific act chunk."""
    job = worker_orchestrator.get_job(job_id)
    if job and 0 <= seq_idx < len(job.chunks):
        chunk_prompt = job.chunks[seq_idx].prompt_segment
    elif job:
        chunk_prompt = job.request.story_prompt
    else:
        chunk_prompt = "cinematic high quality video subject"

    clean_story = (chunk_prompt or "").strip()
    m_seed = abs(hash(f"{job_id}_{clean_story[:40]}")) % 1000000

    req_w = getattr(job.request, "width", 1920) if job else 1920
    req_h = getattr(job.request, "height", 1080) if job else 1080
    ch_idx = job.chunks[seq_idx].chapter_index if (job and 0 <= seq_idx < len(job.chunks)) else 0
    is_ch_start = job.chunks[seq_idx].is_chapter_start if (job and 0 <= seq_idx < len(job.chunks)) else (seq_idx == 0)

    anchor_path = storage_manager.get_or_create_act_anchor(
        job_id=job_id,
        seq_idx=seq_idx,
        prompt=clean_story,
        seed=m_seed,
        width=req_w,
        height=req_h,
        chapter_idx=ch_idx,
        is_chapter_start=is_ch_start,
    )
    return _serve_anchor_image(anchor_path, f"{job_id}_anchor_act_{seq_idx}.png", fmt=format)


@router.get("/{job_id}/video-code/{seq_idx}")
def get_chunk_video_code(job_id: str, seq_idx: int) -> Dict[str, Any]:
    """Serve the coded Video-as-Code continuity manifest for a completed chunk."""
    code_data = storage_manager.get_video_code(job_id=job_id, seq_idx=seq_idx)
    if code_data:
        return code_data
    job = worker_orchestrator.get_job(job_id)
    if job and 0 <= seq_idx < len(job.chunks):
        chunk = job.chunks[seq_idx]
        if chunk.video_code:
            return chunk.video_code
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Video-as-Code manifest for job {job_id} chunk {seq_idx} not found.",
    )

