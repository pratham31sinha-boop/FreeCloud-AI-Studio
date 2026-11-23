"""Worker communication and orchestration API routes.

Provides explicit endpoints for distributed runners (e.g. GitHub Actions)
to register, send heartbeats, claim task chunks, upload rendered artifacts,
and report completions or failures.
"""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from backend.continuity.character_refs import character_ref_manager
from backend.storage.manager import storage_manager
from backend.workers.models import WorkerNode
from backend.workers.orchestrator import worker_orchestrator
from backend.workers.protocol import (
    ChunkCompletionRequest,
    ChunkFailureRequest,
    TaskLeaseResponse,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerRegisterRequest,
    WorkerRegisterResponse,
)

router = APIRouter(prefix="/workers", tags=["Workers"])


@router.post("/register", response_model=WorkerRegisterResponse)
def register_worker(req: WorkerRegisterRequest) -> WorkerRegisterResponse:
    """Register a new distributed worker runner."""
    return worker_orchestrator.register_worker(req)


@router.post("/heartbeat", response_model=WorkerHeartbeatResponse)
def worker_heartbeat(req: WorkerHeartbeatRequest) -> WorkerHeartbeatResponse:
    """Process runner heartbeat to keep task leases active."""
    return worker_orchestrator.heartbeat(req)


@router.post("/claim-task", response_model=TaskLeaseResponse)
def claim_task(worker_id: str) -> TaskLeaseResponse:
    """Claim the next pending task chunk with continuity references."""
    char_dict = {
        c.character_id: c.model_dump()
        for c in character_ref_manager.list_characters()
    }
    return worker_orchestrator.claim_task(worker_id, character_registry=char_dict)


@router.post("/{worker_id}/chunks/{chunk_id}/artifact")
async def upload_chunk_artifact(
    worker_id: str,
    chunk_id: str,
    job_id: str,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Receive rendered video/frames artifact from a worker node."""
    content = await file.read()
    artifact_path = storage_manager.save_chunk_artifact(
        job_id=job_id,
        chunk_id=chunk_id,
        content=content,
        ext=file.filename[file.filename.rfind(".") :] if file.filename and "." in file.filename else ".mp4",
    )
    return {
        "status": "received",
        "worker_id": worker_id,
        "chunk_id": chunk_id,
        "size_bytes": len(content),
        "saved_path": str(artifact_path),
    }


@router.post("/complete-chunk")
def report_chunk_completion(req: ChunkCompletionRequest) -> Dict[str, Any]:
    """Report that a worker has finished rendering and verified its chunk."""
    success = worker_orchestrator.submit_chunk_result(req)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to record chunk completion",
        )
    return {"status": "recorded", "chunk_id": req.chunk_id}


@router.post("/fail-chunk")
def report_chunk_failure(req: ChunkFailureRequest) -> Dict[str, Any]:
    """Report an error or exception encountered during chunk rendering."""
    success = worker_orchestrator.report_chunk_failure(req)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to record chunk failure",
        )
    return {"status": "failure_handled", "chunk_id": req.chunk_id}


@router.get("", response_model=List[WorkerNode])
def list_workers() -> List[WorkerNode]:
    """List all registered worker runners and their operational states."""
    worker_orchestrator.clean_stale_leases()
    return list(worker_orchestrator.workers.values())
