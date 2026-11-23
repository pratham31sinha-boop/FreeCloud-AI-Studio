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
            "progress_percent": job.progress_percent,
            "stage_message": job.stage_message,
            "target_workers": job.target_worker_count,
            "total_chunks": job.total_chunks,
            "completed_chunks": job.completed_chunks,
            "failed_chunks": job.failed_chunks,
            "output_video_path": job.output_video_path,
            "quality_report": quality_eval.model_dump(),
            "chunks": [c.model_dump() for c in job.chunks],
        }


generation_pipeline = GenerationPipeline()
