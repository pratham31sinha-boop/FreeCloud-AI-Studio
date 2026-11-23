"""Storage abstraction layer for FreeCloud AI Studio.

Manages video uploads, worker chunk artifacts, character reference files,
and stitched output videos.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional
from backend.config import settings


class StorageManager:
    """Handles persistence of media files, job artifacts, and continuity metadata."""

    def __init__(self, base_data_dir: Optional[Path] = None):
        self.data_dir = base_data_dir or settings.data_dir
        self.uploads_dir = self.data_dir / "uploads"
        self.artifacts_dir = self.data_dir / "artifacts"
        self.outputs_dir = self.data_dir / "outputs"
        self.characters_dir = self.data_dir / "characters"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.uploads_dir,
            self.artifacts_dir,
            self.outputs_dir,
            self.characters_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        filename: str,
        stream_or_bytes: bytes | BinaryIO,
        prefix: str = "src",
    ) -> Dict[str, str]:
        """Save an uploaded source video or asset."""
        ext = Path(filename).suffix.lower() or ".mp4"
        unique_id = f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"
        destination = self.uploads_dir / unique_id

        hasher = hashlib.sha256()
        total_bytes = 0

        if isinstance(stream_or_bytes, bytes):
            total_bytes = len(stream_or_bytes)
            hasher.update(stream_or_bytes)
            destination.write_bytes(stream_or_bytes)
        else:
            with open(destination, "wb") as f:
                while chunk := stream_or_bytes.read(1024 * 1024):
                    hasher.update(chunk)
                    total_bytes += len(chunk)
                    f.write(chunk)

        return {
            "file_id": unique_id,
            "original_filename": filename,
            "path": str(destination),
            "size_bytes": str(total_bytes),
            "sha256": hasher.hexdigest(),
        }

    def save_chunk_artifact(
        self,
        job_id: str,
        chunk_id: str,
        content: bytes,
        ext: str = ".mp4",
    ) -> Path:
        """Save a generated chunk video/frames artifact from a worker."""
        job_artifact_dir = self.artifacts_dir / job_id
        job_artifact_dir.mkdir(parents=True, exist_ok=True)
        file_path = job_artifact_dir / f"{chunk_id}{ext}"
        file_path.write_bytes(content)
        return file_path

    def get_chunk_artifact(
        self, job_id: str, chunk_id: str, ext: str = ".mp4"
    ) -> Optional[Path]:
        """Retrieve the path of a worker chunk artifact if it exists."""
        candidate = self.artifacts_dir / job_id / f"{chunk_id}{ext}"
        return candidate if candidate.is_file() else None

    def list_job_artifacts(self, job_id: str) -> List[Path]:
        """List all artifact files stored for a particular job."""
        job_artifact_dir = self.artifacts_dir / job_id
        if not job_artifact_dir.is_dir():
            return []
        return sorted(list(job_artifact_dir.glob("*.*")))

    def save_output_video(
        self,
        job_id: str,
        video_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Path:
        """Save the final stitched coherent video."""
        target_name = filename or f"{job_id}_final.mp4"
        output_path = self.outputs_dir / target_name
        output_path.write_bytes(video_bytes)
        return output_path

    def get_output_video(self, job_id: str) -> Optional[Path]:
        """Return path to final output video if generated."""
        candidate = self.outputs_dir / f"{job_id}_final.mp4"
        return candidate if candidate.is_file() else None

    def save_character_profile(
        self, character_id: str, profile_data: dict, face_images: Optional[List[tuple[str, bytes]]] = None
    ) -> Path:
        """Persist a character reference profile and its turnaround/reference images."""
        char_dir = self.characters_dir / character_id
        char_dir.mkdir(parents=True, exist_ok=True)

        meta_file = char_dir / "profile.json"
        meta_file.write_text(json.dumps(profile_data, indent=2))

        if face_images:
            img_dir = char_dir / "references"
            img_dir.mkdir(parents=True, exist_ok=True)
            for img_name, img_bytes in face_images:
                (img_dir / img_name).write_bytes(img_bytes)

        return char_dir

    def load_character_profile(self, character_id: str) -> Optional[dict]:
        """Load character profile metadata from disk."""
        meta_file = self.characters_dir / character_id / "profile.json"
        if not meta_file.is_file():
            return None
        return json.loads(meta_file.read_text())


# Global default instance
storage_manager = StorageManager()
