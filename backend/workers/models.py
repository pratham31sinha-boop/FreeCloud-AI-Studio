"""Data models for worker orchestration and task distribution."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkerState(str, Enum):
    IDLE = "idle"
    PROVISIONING = "provisioning"
    BUSY = "busy"
    OFFLINE = "offline"
    FAILED = "failed"


class TaskChunkStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    ORCHESTRATING = "orchestrating"
    RENDERING = "rendering"
    VERIFYING = "verifying"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnimeStyle(str, Enum):
    SHINKAI_CINEMATIC = "shinkai_cinematic"
    MAPPA_ACTION = "mappa_action"
    UFOTABLE_DIGITAL = "ufotable_digital"
    GHIBLI_CLASSIC = "ghibli_classic"
    KYOANI_SOFT = "kyoani_soft"
    MINIMAX_H3_CINEMATIC = "minimax_h3_cinematic"
    MINIMAX_H3_ANIME = "minimax_h3_anime"


class TaskChunk(BaseModel):
    """A discrete temporal segment of the video generation job assigned to a worker."""

    chunk_id: str
    job_id: str
    sequence_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    prompt_segment: str
    anime_style: str
    character_ids: List[str] = Field(default_factory=list)
    motion_reference_id: Optional[str] = None
    input_frame_range: Optional[tuple[int, int]] = None
    status: TaskChunkStatus = TaskChunkStatus.PENDING
    assigned_worker_id: Optional[str] = None
    lease_expires_at: Optional[float] = None
    attempts: int = 0
    max_retries: int = 2
    artifact_id: Optional[str] = None
    artifact_path: Optional[str] = None
    artifact_checksum: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class WorkerNode(BaseModel):
    """Metadata and live status of a worker instance (e.g. GitHub Actions runner)."""

    worker_id: str
    runner_type: str = "github-actions"  # github-actions | local-worker | cloud-gpu
    runner_os: str = "linux"
    state: WorkerState = WorkerState.IDLE
    assigned_job_id: Optional[str] = None
    current_chunk_id: Optional[str] = None
    registration_time: float = Field(default_factory=time.time)
    last_heartbeat: float = Field(default_factory=time.time)
    total_completed: int = 0
    failure_count: int = 0
    client_version: str = "1.0.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobRequest(BaseModel):
    """Job creation request from user or API."""

    original_video_id: Optional[str] = None
    story_prompt: str = Field(
        ..., min_length=5, description="Scene narrative or Jagdhatri scene script"
    )
    duration_minutes: float = Field(
        default=1.0, ge=0.01, le=15.0, description="Target output duration in minutes"
    )
    anime_style: AnimeStyle = AnimeStyle.SHINKAI_CINEMATIC
    character_ids: List[str] = Field(
        default_factory=lambda: ["jagdhatri_main", "jash_lead"]
    )
    fps: int = 24
    continuity_identity_strict: bool = True
    continuity_motion_tracking: bool = True
    continuity_audio_sync: bool = True


class JobRecord(BaseModel):
    """Complete state record of a video generation job."""

    job_id: str
    request: JobRequest
    status: JobStatus = JobStatus.QUEUED
    target_worker_count: int = 4
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    progress_percent: float = 0.0
    stage_message: str = "Job initialized in queue"
    chunks: List[TaskChunk] = Field(default_factory=list)
    output_video_path: Optional[str] = None
    continuity_report: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    elapsed_sec: float = 0.0
    estimated_remaining_sec: Optional[float] = None
    estimated_total_sec: Optional[float] = None
    active_workers_count: int = 0
    error: Optional[str] = None
