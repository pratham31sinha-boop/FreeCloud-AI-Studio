"""End-to-End Generation Pipeline for FreeCloud AI Studio.

Orchestrates input ingestion, continuity conditioning, worker dispatching,
and final video compilation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from backend.continuity.character_refs import character_ref_manager
from backend.continuity.quality_check import quality_checker
from backend.continuity.story_continuity import story_continuity_engine
from backend.storage.manager import storage_manager
from backend.storage.video_processor import video_processor
from backend.workers.models import JobRecord, JobRequest
from backend.workers.orchestrator import worker_orchestrator

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Coordinates end-to-end processing of anime video generation tasks."""

    async def initiate_job(self, request: JobRequest) -> JobRecord:
        """Process incoming request through continuity preparation and worker orchestration."""
        logger.info(
            "Initiating generation pipeline for duration %.1fm (style: %s)",
            request.duration_minutes,
            request.anime_style,
        )

        # 1. Analyze script / prompt continuity beats
        beats = story_continuity_engine.analyze_script(request.story_prompt)
        logger.info("Extracted %d story beats from script", len(beats))

        # 2. Verify requested character profiles exist
        for cid in request.character_ids:
            char = character_ref_manager.get_character(cid)
            if not char:
                logger.warning(
                    "Character ID %s not found in registry; using default anime persona", cid
                )

        # 3. Create job and dispatch workers via WorkerOrchestrator
        job_record = await worker_orchestrator.submit_job(request)

        # 4. If source video uploaded, extract sample frames for chunk conditioning
        if request.original_video_id:
            src_video = storage_manager.uploads_dir / request.original_video_id
            if src_video.is_file():
                frames_dir = storage_manager.artifacts_dir / f"extracted_{job_record.job_id}"
                try:
                    video_processor.extract_frames(src_video, frames_dir, target_fps=2.0, max_frames=24)
                    logger.info("Extracted sample frames for job %s into %s", job_record.job_id, frames_dir)
                except Exception as exc:
                    logger.error("Failed extracting frames for job %s: %s", job_record.job_id, exc)

        return job_record

    def get_pipeline_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve aggregated job metrics, worker statuses, and continuity evaluations."""
        job = worker_orchestrator.get_job(job_id)
        if not job:
            return None

        # Build chunk metrics snapshot
        chunk_metrics = [
            chunk.metrics for chunk in job.chunks if chunk.metrics
        ]
        quality_eval = quality_checker.evaluate_job(job_id, chunk_metrics)

        return {
            "job_id": job.job_id,
            "status": job.status,
            "execution_mode": getattr(job.request, "execution_mode", "cloud_github"),
            "model_relay": getattr(job.request, "model_relay", False),
            "progress_percent": job.progress_percent,
            "stage_message": job.stage_message,
            "target_workers": job.target_worker_count,
            "active_workers_count": getattr(job, "active_workers_count", 0),
            "elapsed_sec": getattr(job, "elapsed_sec", 0.0),
            "estimated_remaining_sec": getattr(job, "estimated_remaining_sec", None),
            "estimated_total_sec": getattr(job, "estimated_total_sec", None),
            "total_chunks": job.total_chunks,
            "completed_chunks": job.completed_chunks,
            "failed_chunks": job.failed_chunks,
            "output_video_path": job.output_video_path,
            "quality_report": quality_eval.model_dump(),
            "chunks": [c.model_dump() for c in job.chunks],
        }


generation_pipeline = GenerationPipeline()
