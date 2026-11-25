"""Tests for Worker Allocation, Orchestration, and Fault Handling."""

import asyncio
from backend.workers.allocation import calculate_worker_allocation, plan_task_chunks
from backend.workers.models import (
    AnimeStyle,
    JobRequest,
    JobStatus,
    TaskChunkStatus,
    WorkerState,
)
from backend.workers.orchestrator import WorkerOrchestrator
from backend.workers.protocol import (
    ChunkCompletionRequest,
    ChunkFailureRequest,
    WorkerHeartbeatRequest,
    WorkerRegisterRequest,
)


def test_worker_allocation_table():
    """Verify target worker allocation table matches the design specification:

    1 min -> 4
    2 min -> 5
    3 min -> 6
    5 min -> 7
    10 min -> 8
    15 min -> 10
    """
    assert calculate_worker_allocation(0.5) == 4
    assert calculate_worker_allocation(1.0) == 4
    assert calculate_worker_allocation(1.5) == 5
    assert calculate_worker_allocation(2.0) == 5
    assert calculate_worker_allocation(2.5) == 6
    assert calculate_worker_allocation(3.0) == 6
    assert calculate_worker_allocation(4.0) == 7
    assert calculate_worker_allocation(5.0) == 7
    assert calculate_worker_allocation(8.0) == 8
    assert calculate_worker_allocation(10.0) == 8
    assert calculate_worker_allocation(12.0) == 10
    assert calculate_worker_allocation(15.0) == 10
    assert calculate_worker_allocation(20.0) == 10


def test_task_chunking():
    """Verify job request is broken into discrete sequential chunks."""
    req = JobRequest(
        story_prompt="Jagdhatri confronts the syndicate. Jash covers the exit.",
        duration_minutes=1.0,
        anime_style=AnimeStyle.UFOTABLE_DIGITAL,
        character_ids=["jagdhatri_main", "jash_lead"],
    )
    chunks = plan_task_chunks("test_job_001", req, target_chunk_sec=15.0)

    # 1 minute (60s) with 15s target chunks -> 4 chunks
    assert len(chunks) == 4
    assert chunks[0].start_sec == 0.0
    assert chunks[-1].end_sec == 60.0

    # Ensure sequence indices are strictly ordered
    for idx, chunk in enumerate(chunks):
        assert chunk.sequence_index == idx
        assert chunk.status == TaskChunkStatus.PENDING
        assert "jagdhatri_main" in chunk.character_ids


def test_worker_orchestrator_lifecycle():
    """Test full worker orchestrator lifecycle: submit, register, lease, complete, assemble."""
    async def _run():
        orchestrator = WorkerOrchestrator()

        # 1. Submit Job
        req = JobRequest(
            story_prompt="Jagdhatri leaps across rooftops in Kolkata pursuit scene.",
            duration_minutes=1.0,
            anime_style=AnimeStyle.MAPPA_ACTION,
        )
        job = await orchestrator.submit_job(req)
        assert job.status == JobStatus.RENDERING
        assert job.total_chunks >= 4
        assert job.target_worker_count == 4

        # 2. Register 4 Workers
        for w_idx in range(4):
            reg = orchestrator.register_worker(
                WorkerRegisterRequest(worker_id=f"worker_test_{w_idx}")
            )
            assert reg.accepted is True

        # 3. First worker claims a chunk
        lease = orchestrator.claim_task("worker_test_0")
        assert lease.has_task is True
        assert lease.chunk_id is not None
        assert lease.job_id == job.job_id

        # 4. Heartbeat
        hb_resp = orchestrator.heartbeat(
            WorkerHeartbeatRequest(
                worker_id="worker_test_0",
                current_chunk_id=lease.chunk_id,
                progress_percent=50.0,
            )
        )
        assert hb_resp.acknowledged is True
        assert hb_resp.command == "continue"

        # 5. Complete all chunks to trigger final assembly
        for chunk in job.chunks:
            comp_req = ChunkCompletionRequest(
                worker_id="worker_test_0",
                chunk_id=chunk.chunk_id,
                job_id=job.job_id,
                sha256_checksum="abc123sha256",
                render_duration_sec=2.1,
                output_frames_count=24,
                continuity_metrics={"identity_score": 0.90},
            )
            orchestrator.submit_chunk_result(comp_req)

        # Verify Job is COMPLETED and video stitched
        completed_job = orchestrator.get_job(job.job_id)
        assert completed_job.status == JobStatus.COMPLETED
        assert completed_job.progress_percent == 100.0
        assert completed_job.output_video_path is not None

    asyncio.run(_run())


def test_worker_failure_and_retry():
    """Verify that chunk failure triggers retries and eventually fails after max retries."""
    async def _run():
        orchestrator = WorkerOrchestrator()

        req = JobRequest(
            story_prompt="Action test failure recovery sequence.",
            duration_minutes=1.0,
        )
        job = await orchestrator.submit_job(req)
        orchestrator.register_worker(WorkerRegisterRequest(worker_id="worker_fail_node"))

        lease = orchestrator.claim_task("worker_fail_node")
        assert lease.has_task is True
        chunk_id = lease.chunk_id

        # Report failure 1 (attempt 1 -> should retry)
        fail_1 = ChunkFailureRequest(
            worker_id="worker_fail_node",
            chunk_id=chunk_id,
            job_id=job.job_id,
            error_code="CUDA_OUT_OF_MEMORY",
            error_details="Simulated VRAM allocation failure",
        )
        orchestrator.report_chunk_failure(fail_1)

        chunk = next(c for c in job.chunks if c.chunk_id == chunk_id)
        assert chunk.status == TaskChunkStatus.RETRYING

        # Claim again
        lease_2 = orchestrator.claim_task("worker_fail_node")
        assert lease_2.has_task is True
        assert lease_2.chunk_id == chunk_id

        # Report failure 2 (attempt 2 -> max retries reached)
        fail_2 = ChunkFailureRequest(
            worker_id="worker_fail_node",
            chunk_id=chunk_id,
            job_id=job.job_id,
            error_code="CUDA_OUT_OF_MEMORY",
            error_details="Second simulated failure",
        )
        orchestrator.report_chunk_failure(fail_2)

        assert chunk.status == TaskChunkStatus.FAILED
        assert job.status == JobStatus.FAILED

    asyncio.run(_run())


def test_word_boundary_prompt_beats_and_multi_model_relay():
    """Verify word-boundary theme beats (no 'cat'->'cathedral' false positive) and multi-model relay."""
    req = JobRequest(
        story_prompt="Enchanted elven forest cathedral bathed in celestial bioluminescence.",
        duration_minutes=0.5,
        model_name="auto_relay",
        model_relay=True,
    )
    chunks = plan_task_chunks("test_relay_001", req, target_chunk_sec=15.0)
    assert len(chunks) == 4
    assert "predator" not in chunks[0].prompt_segment.lower()
    assert chunks[0].model_name == "wan_2_1"
    assert chunks[1].model_name == "minimax_h3"
    assert chunks[2].model_name == "hunyuan_video"
    assert chunks[0].is_extended_clip is False
    assert chunks[1].is_extended_clip is True
    assert "EXTENDED CLIP" in chunks[1].prompt_segment


