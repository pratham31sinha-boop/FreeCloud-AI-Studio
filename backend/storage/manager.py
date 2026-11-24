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
        video_bytes_or_source_path: bytes | Path | str,
        filename: Optional[str] = None,
    ) -> Path:
        """Save the final stitched coherent video."""
        target_name = filename or f"{job_id}_final.mp4"
        output_path = self.outputs_dir / target_name

        if isinstance(video_bytes_or_source_path, (Path, str)):
            src = Path(video_bytes_or_source_path)
            if src.resolve() != output_path.resolve():
                shutil.copy2(str(src), str(output_path))
        else:
            output_path.write_bytes(video_bytes_or_source_path)

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

    def get_or_create_master_anchor(
        self, job_id: str, prompt: str, seed: int = 42, width: int = 640, height: int = 360
    ) -> Path:
        """Fetch or synthesize the master anchor reference image for a job to guarantee visual consistency."""
        anchor_path = self.artifacts_dir / f"{job_id}_master_anchor.png"
        if anchor_path.is_file() and anchor_path.stat().st_size > 1000:
            return anchor_path

        import urllib.request
        import urllib.parse
        import cv2
        import numpy as np

        import logging
        import time
        logger = logging.getLogger(__name__)

        clean_prompt = prompt.replace("\n", " ").strip()
        encoded = urllib.parse.quote(clean_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&seed={seed}&nologo=true"

        for attempt in range(1, 5):
            try:
                logger.info("Fetching master anchor image from cloud inference (attempt %d/4)...", attempt)
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw_data = resp.read()
                if len(raw_data) > 1000:
                    arr = np.frombuffer(raw_data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                        if (img.shape[1], img.shape[0]) != (width, height):
                            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
                        cv2.imwrite(str(anchor_path), img)
                        logger.info("Master anchor successfully saved to %s (%d bytes)", anchor_path, len(raw_data))
                        return anchor_path
            except Exception as exc:
                logger.warning("Master anchor fetch attempt %d failed: %s", attempt, exc)
                time.sleep(2.5)

        # Failsafe procedural anchor
        np.random.seed(seed % (2**31 - 1))
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        r = np.clip((np.sin(xx * 3.14 + seed * 0.1) * 0.5 + 0.5) * 220 + 30, 0, 255).astype(np.uint8)
        g = np.clip((np.cos(yy * 3.14 + seed * 0.2) * 0.5 + 0.5) * 180 + 40, 0, 255).astype(np.uint8)
        b = np.clip((np.sin((xx + yy) * 3.14 + seed * 0.3) * 0.5 + 0.5) * 240 + 15, 0, 255).astype(np.uint8)
        frame = cv2.GaussianBlur(cv2.merge([b, g, r]), (51, 51), 0)
        cv2.imwrite(str(anchor_path), frame)
        return anchor_path


# Global default instance
storage_manager = StorageManager()
