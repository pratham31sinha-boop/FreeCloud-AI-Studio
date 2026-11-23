"""Configuration module for FreeCloud AI Studio."""

import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings and runtime environment parameters."""

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Backend bind host")
    port: int = Field(default=8000, description="Backend bind port")
    env: str = Field(default="development", description="Environment mode")
    log_level: str = Field(default="INFO", description="Logging verbosity")

    # Storage paths
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "FREECLOUD_DATA_DIR",
                str(Path(__file__).resolve().parent.parent / "data"),
            )
        )
    )
    upload_max_mb: int = Field(
        default=250, description="Max uploaded video file size in MB"
    )

    # Worker Orchestration (GitHub Actions & Local)
    github_actions_enabled: bool = Field(
        default=os.getenv("GITHUB_ACTIONS_ENABLED", "false").lower() == "true"
    )
    github_repository: str = Field(
        default=os.getenv("GITHUB_REPOSITORY", "owner/FreeCloud-AI-Studio")
    )
    github_workflow_id: str = Field(
        default=os.getenv("GITHUB_WORKFLOW_ID", "worker_runner.yml")
    )
    github_pat_token: str = Field(
        default=os.getenv("GITHUB_PAT_TOKEN", "")
    )
    worker_heartbeat_timeout_sec: int = Field(default=120)
    worker_task_lease_sec: int = Field(default=300)
    worker_max_retries: int = Field(default=2)

    # ComfyUI Integration
    comfyui_host: str = Field(
        default=os.getenv("COMFYUI_HOST", "127.0.0.1")
    )
    comfyui_port: int = Field(
        default=int(os.getenv("COMFYUI_PORT", "8188"))
    )
    comfyui_enabled: bool = Field(
        default=os.getenv("COMFYUI_ENABLED", "false").lower() == "true"
    )
    comfyui_timeout_sec: int = Field(default=180)

    # Continuity Engine
    continuity_identity_strict: bool = Field(default=True)
    continuity_motion_tracking: bool = Field(default=True)
    continuity_audio_sync: bool = Field(default=True)
    continuity_min_quality_score: float = Field(default=0.75)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def characters_dir(self) -> Path:
        return self.data_dir / "characters"

    def ensure_directories(self) -> None:
        """Create runtime storage directories if missing."""
        for path in (
            self.data_dir,
            self.uploads_dir,
            self.artifacts_dir,
            self.outputs_dir,
            self.characters_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
