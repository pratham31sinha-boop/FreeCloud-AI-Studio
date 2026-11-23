"""Worker allocation and task chunking policies.

Maps video generation duration to planned worker counts according to the
FreeCloud AI Studio architecture specification:

| Duration   | Workers |
| 1 minute   | 4       |
| 2 minutes  | 5       |
| 3 minutes  | 6       |
| 5 minutes  | 7       |
| 10 minutes | 8       |
| 15 minutes | 10      |
"""

from __future__ import annotations

import math
import uuid
from typing import List, Tuple
from backend.workers.models import JobRequest, TaskChunk, TaskChunkStatus


# Exact planned table checkpoints
ALLOCATION_TABLE: List[Tuple[float, int]] = [
    (1.0, 4),
    (2.0, 5),
    (3.0, 6),
    (5.0, 7),
    (10.0, 8),
    (15.0, 10),
]


def calculate_worker_allocation(duration_minutes: float) -> int:
    """Calculate target number of worker nodes for a given video duration in minutes.

    Target allocation rules:
    - <= 1 min: 4 workers
    - <= 2 min: 5 workers
    - <= 3 min: 6 workers
    - <= 5 min: 7 workers
    - <= 10 min: 8 workers
    - <= 15 min: 10 workers
    - > 15 min: capped at 10 workers for the current architecture milestone.
    """
    if duration_minutes <= 0:
        return 4

    for threshold_mins, worker_count in ALLOCATION_TABLE:
        if duration_minutes <= threshold_mins:
            return worker_count

    return 10


def plan_task_chunks(
    job_id: str,
    request: JobRequest,
    target_chunk_sec: float = 15.0,
) -> List[TaskChunk]:
    """Break a video generation job into sequential task chunks for distributed workers.

    Each chunk corresponds to a continuous segment of time with its narrative prompt,
    character IDs, and continuity references.
    """
    total_seconds = max(10.0, float(request.duration_minutes * 60.0))
    worker_count = calculate_worker_allocation(request.duration_minutes)

    # Determine chunk count: at least equal to worker_count so all workers can be utilized,
    # or divided by target_chunk_sec if the video is longer.
    calculated_chunks = max(worker_count, math.ceil(total_seconds / target_chunk_sec))
    chunk_duration = total_seconds / calculated_chunks

    chunks: List[TaskChunk] = []
    current_time = 0.0

    prompt_sentences = [
        s.strip()
        for s in request.story_prompt.replace("\n", " ").split(".")
        if s.strip()
    ]
    if not prompt_sentences:
        prompt_sentences = [request.story_prompt]

    for idx in range(calculated_chunks):
        end_time = min(total_seconds, current_time + chunk_duration)
        # Assign relevant sentence slice or full prompt with scene context
        sentence_idx = idx % len(prompt_sentences)
        scene_prompt = (
            f"[Scene {idx + 1}/{calculated_chunks} | {current_time:.1f}s-{end_time:.1f}s] "
            f"{prompt_sentences[sentence_idx]}"
        )

        chunk = TaskChunk(
            chunk_id=f"{job_id}_chk_{idx + 1:03d}_{uuid.uuid4().hex[:6]}",
            job_id=job_id,
            sequence_index=idx,
            start_sec=round(current_time, 2),
            end_sec=round(end_time, 2),
            duration_sec=round(end_time - current_time, 2),
            prompt_segment=scene_prompt,
            anime_style=str(request.anime_style),
            character_ids=list(request.character_ids),
            motion_reference_id=f"motion_{job_id}_seq_{idx:03d}" if request.continuity_motion_tracking else None,
            status=TaskChunkStatus.PENDING,
        )
        chunks.append(chunk)
        current_time = end_time

    return chunks
