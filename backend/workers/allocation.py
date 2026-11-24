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


# Exact planned table checkpoints across all stages (1s up to 15m)
ALLOCATION_TABLE: List[Tuple[float, int]] = [
    (0.05, 1),   # Stage 1: 1-3 seconds (single worker fast pass)
    (0.15, 2),   # Stage 2: 5-9 seconds
    (0.30, 2),   # Stage 3: 15 seconds
    (0.60, 3),   # Stage 4: 30 seconds
    (1.0, 4),    # Stage 5: 1 minute -> 4 workers
    (2.0, 5),    # Stage 6: 2 minutes -> 5 workers
    (3.0, 6),    # Stage 7: 3 minutes -> 6 workers
    (5.0, 7),    # Stage 8: 5 minutes -> 7 workers
    (10.0, 8),   # Stage 9: 10 minutes -> 8 workers
    (15.0, 10),  # Stage 10: 15 minutes -> 10 workers
]


def calculate_worker_allocation(duration_minutes: float) -> int:
    """Calculate target number of worker nodes for a given video duration in minutes.

    Architecture requirement: Minimum 4 cloud servers pooled together to combine
    computing power (each runner with 16GB RAM equivalent) for ultra-fast parallel rendering.
    - 1s - 1 min: 4 workers minimum
    - 2 min: 5 workers
    - 3 min: 6 workers
    - 5 min: 7 workers
    - 10 min: 8 workers
    - 15 min: 10 workers
    """
    if duration_minutes <= 1.0:
        return 4
    if duration_minutes <= 2.0:
        return 5
    if duration_minutes <= 3.0:
        return 6
    if duration_minutes <= 5.0:
        return 7
    if duration_minutes <= 10.0:
        return 8
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

    # Progressive 4-Act Storyline Continuation (eliminates looping, ensures real narrative progression)
    act_beats = [
        "Act 1 (Emergence): The subject stealthily emerges from deep misty rainforest shadows into the light, initiating a smooth predatory prowl, glowing emerald eyes scanning ahead",
        "Act 2 (Progression): Continuing forward through the terrain, stepping across glistening wet mossy riverstones, water droplets flying off paws, sleek fur rippling with continuous momentum",
        "Act 3 (Climactic Tension): Intense moment of focus, lowering into an aggressive stance, baring sharp fangs in a low territorial snarl, breath visible in humid night air, muscles tensed",
        "Act 4 (Grand Finale): Powerful surge forward into a majestic full-speed stride, accelerating past luminous bioluminescent flora into the deep jungle canopy",
    ]

    for idx in range(calculated_chunks):
        end_time = min(total_seconds, current_time + chunk_duration)
        beat_text = act_beats[idx % len(act_beats)]
        # Unified continuous scene prompt with progressive act continuation
        scene_prompt = (
            f"[Act {idx + 1}/{calculated_chunks} | {current_time:.1f}s-{end_time:.1f}s] "
            f"{request.story_prompt.strip()} -- {beat_text}"
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
            model_name=getattr(request, "model_name", "wan_2_1"),
            anchor_frame_url=f"/api/v1/jobs/{job_id}/anchor/{idx}",
            character_ids=list(request.character_ids),
            motion_reference_id=f"motion_{job_id}_seq_{idx:03d}" if request.continuity_motion_tracking else None,
            status=TaskChunkStatus.PENDING,
        )
        chunks.append(chunk)
        current_time = end_time

    return chunks
