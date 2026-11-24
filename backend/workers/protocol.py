"""Explicit communication and artifact-transfer protocol for distributed workers.

Defines the message formats, leases, and endpoints used between the FreeCloud
AI Studio backend and distributed workers (such as temporary GitHub Actions runners).
Does NOT rely on shared memory, shared GPU, or shared local disk.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkerRegisterRequest(BaseModel):
    worker_id: str
    runner_type: str = "github-actions"
    runner_os: str = "linux"
    client_version: str = "1.0.0"
    capabilities: Dict[str, Any] = Field(
        default_factory=lambda: {"gpu": False, "comfyui_ready": False}
    )


class WorkerRegisterResponse(BaseModel):
    accepted: bool
    worker_id: str
    session_token: str
    server_time: float = Field(default_factory=time.time)
    heartbeat_interval_sec: int = 30
    message: str = "Worker registered successfully"


class WorkerHeartbeatRequest(BaseModel):
    worker_id: str
    current_chunk_id: Optional[str] = None
    progress_percent: float = 0.0
    status_message: str = "Active"


class WorkerHeartbeatResponse(BaseModel):
    acknowledged: bool
    command: str = "continue"  # continue | cancel_chunk | shutdown
    server_time: float = Field(default_factory=time.time)


class CharacterReferenceBundle(BaseModel):
    """Explicit character data sent to the runner for consistent generation."""

    character_id: str
    name: str
    hair_style: str
    clothing_palette: str
    face_image_urls: List[str] = Field(default_factory=list)
    lora_trigger_word: Optional[str] = None
    embedding_reference_id: Optional[str] = None


class TaskLeaseResponse(BaseModel):
    """Payload delivered to a worker claiming a task chunk."""

    has_task: bool
    job_id: Optional[str] = None
    chunk_id: Optional[str] = None
    sequence_index: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    duration_sec: Optional[float] = None
    prompt_segment: Optional[str] = None
    anime_style: Optional[str] = None
    model_name: Optional[str] = "wan_2_1"
    anchor_frame_url: Optional[str] = None
    master_seed: Optional[int] = None
    characters: List[CharacterReferenceBundle] = Field(default_factory=list)
    motion_guidance: Optional[Dict[str, Any]] = None
    total_chunks: Optional[int] = 4
    total_duration_sec: Optional[float] = None
    lease_expires_at: Optional[float] = None
    artifact_upload_url: Optional[str] = None
    message: str = "Task assigned"


class ChunkCompletionRequest(BaseModel):
    """Worker notification that chunk rendering and artifact upload is complete."""

    worker_id: str
    chunk_id: str
    job_id: str
    sha256_checksum: str
    render_duration_sec: float
    output_frames_count: int = 0
    continuity_metrics: Dict[str, float] = Field(default_factory=dict)
    notes: Optional[str] = None


class ChunkFailureRequest(BaseModel):
    """Worker notification of an error during chunk processing."""

    worker_id: str
    chunk_id: str
    job_id: str
    error_code: str
    error_details: str
    will_retry: bool = False
