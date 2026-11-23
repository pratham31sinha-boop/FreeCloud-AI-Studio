"""Worker management module for FreeCloud AI Studio."""

from backend.workers.allocation import calculate_worker_allocation, plan_task_chunks
from backend.workers.dispatcher import GitHubActionsDispatcher, LocalWorkerDispatcher
from backend.workers.models import (
    JobRecord,
    JobRequest,
    JobStatus,
    TaskChunk,
    TaskChunkStatus,
    WorkerNode,
    WorkerState,
)
from backend.workers.orchestrator import WorkerOrchestrator, worker_orchestrator
from backend.workers.protocol import (
    ChunkCompletionRequest,
    ChunkFailureRequest,
    TaskLeaseResponse,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerRegisterRequest,
    WorkerRegisterResponse,
)

__all__ = [
    "calculate_worker_allocation",
    "plan_task_chunks",
    "GitHubActionsDispatcher",
    "LocalWorkerDispatcher",
    "JobRecord",
    "JobRequest",
    "JobStatus",
    "TaskChunk",
    "TaskChunkStatus",
    "WorkerNode",
    "WorkerState",
    "WorkerOrchestrator",
    "worker_orchestrator",
    "ChunkCompletionRequest",
    "ChunkFailureRequest",
    "TaskLeaseResponse",
    "WorkerHeartbeatRequest",
    "WorkerHeartbeatResponse",
    "WorkerRegisterRequest",
    "WorkerRegisterResponse",
]
