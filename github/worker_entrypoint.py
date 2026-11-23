#!/usr/bin/env python3
"""FreeCloud AI Studio - Distributed Worker Runner Entrypoint.

Executed inside GitHub Actions runners or standalone compute nodes.
Communicates strictly via HTTP APIs; does NOT assume shared memory, GPU, or local disk.
Gracefully terminates when tasks are complete, respecting GitHub Actions limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from typing import Any, Dict, Optional
import urllib.error
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Worker %(name)s]: %(message)s",
)
logger = logging.getLogger("Runner")


class WorkerRunner:
    """Manages the lifecycle of a single worker node on a remote runner."""

    def __init__(self, backend_url: str, worker_id: str, job_id: Optional[str] = None):
        self.backend_url = backend_url.rstrip("/")
        self.worker_id = worker_id
        self.target_job_id = job_id
        self.session_token: Optional[str] = None

    def _post_json(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.backend_url}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": f"FreeCloudWorker/{self.worker_id}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            logger.error("HTTP error calling %s: %d - %s", url, exc.code, body)
            raise
        except Exception as exc:
            logger.error("Network error calling %s: %s", url, exc)
            raise

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = f"{self.backend_url}{path}"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": f"FreeCloudWorker/{self.worker_id}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _upload_file(self, path: str, file_path: str) -> Dict[str, Any]:
        """Upload chunk artifact using standard multipart form."""
        url = f"{self.backend_url}{path}"
        boundary = "----FreeCloudBoundary" + hashlib.md5(str(time.time()).encode()).hexdigest()
        filename = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: video/mp4\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": f"FreeCloudWorker/{self.worker_id}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def register(self) -> bool:
        """Register runner identity with the FreeCloud backend."""
        logger.info("Registering worker node %s with backend %s", self.worker_id, self.backend_url)
        payload = {
            "worker_id": self.worker_id,
            "runner_type": "github-actions" if "GITHUB_ACTIONS" in os.environ else "local-runner",
            "runner_os": platform.system().lower(),
            "client_version": "1.0.0",
            "capabilities": {
                "gpu": False,
                "comfyui_ready": False,
                "hostname": platform.node(),
            },
        }
        resp = self._post_json("/api/v1/workers/register", payload)
        if resp.get("accepted"):
            self.session_token = resp.get("session_token")
            logger.info("Registration accepted. Session token acquired.")
            return True
        logger.error("Registration rejected: %s", resp.get("message"))
        return False

    def heartbeat(self, current_chunk_id: Optional[str] = None, progress: float = 0.0) -> bool:
        """Send operational heartbeat."""
        payload = {
            "worker_id": self.worker_id,
            "current_chunk_id": current_chunk_id,
            "progress_percent": progress,
            "status_message": "Rendering in progress" if current_chunk_id else "Idle waiting for task",
        }
        try:
            resp = self._post_json("/api/v1/workers/heartbeat", payload)
            return resp.get("acknowledged", False)
        except Exception as exc:
            logger.warning("Heartbeat failed: %s", exc)
            return False

    def execute(self, max_idle_seconds: int = 60) -> None:
        """Main worker execution loop."""
        if not self.register():
            sys.exit(1)

        idle_start = time.time()
        logger.info("Starting task claim loop...")

        while True:
            self.heartbeat()

            # Attempt to claim a task chunk
            try:
                lease = self._post_json(
                    f"/api/v1/workers/claim-task?worker_id={self.worker_id}", {}
                )
            except Exception as exc:
                logger.error("Error during task claim: %s. Retrying in 5s...", exc)
                time.sleep(5)
                continue

            if lease.get("has_task"):
                idle_start = time.time()
                chunk_id = lease["chunk_id"]
                job_id = lease["job_id"]
                prompt = lease["prompt_segment"]
                logger.info(
                    "Leased task chunk %s for job %s (Sequence %d, %.1fs-%.1fs)",
                    chunk_id,
                    job_id,
                    lease.get("sequence_index", 0),
                    lease.get("start_sec", 0.0),
                    lease.get("end_sec", 0.0),
                )
                logger.info("Scene prompt: '%s'", prompt)

                # Process chunk
                self._process_chunk(lease)
            else:
                # No tasks currently available
                elapsed_idle = time.time() - idle_start
                if elapsed_idle > max_idle_seconds:
                    logger.info(
                        "No tasks available for %d seconds. Exiting runner gracefully to avoid wasting runner minutes.",
                        max_idle_seconds,
                    )
                    break
                logger.info("No chunk available. Sleeping 4 seconds... (Idle %ds)", int(elapsed_idle))
                time.sleep(4)

        logger.info("Worker runner session ended.")

    def _process_chunk(self, lease: Dict[str, Any]) -> None:
        """Simulate/Render the chunk frames and upload artifact."""
        chunk_id = lease["chunk_id"]
        job_id = lease["job_id"]
        start_time = time.time()

        try:
            logger.info("Generating anime frames for chunk %s...", chunk_id)
            time.sleep(2.0)  # Simulated model inference pass

            # Create sample chunk output video/artifact
            dummy_artifact_path = f"/tmp/{chunk_id}_rendered.mp4"
            artifact_content = (
                f"FreeCloud-AI-Studio-Anime-Segment-{chunk_id}\n"
                f"Prompt: {lease.get('prompt_segment')}\n"
                f"Characters: {[c['character_id'] for c in lease.get('characters', [])]}\n"
            ).encode("utf-8") + b"\x00" * 256

            with open(dummy_artifact_path, "wb") as f:
                f.write(artifact_content)

            sha256 = hashlib.sha256(artifact_content).hexdigest()
            logger.info("Uploading artifact to FreeCloud backend (size: %d bytes)...", len(artifact_content))

            # Explicit HTTP artifact upload
            upload_url = f"/api/v1/workers/{self.worker_id}/chunks/{chunk_id}/artifact?job_id={job_id}"
            self._upload_file(upload_url, dummy_artifact_path)

            # Report chunk completion
            render_elapsed = time.time() - start_time
            completion_payload = {
                "worker_id": self.worker_id,
                "chunk_id": chunk_id,
                "job_id": job_id,
                "sha256_checksum": sha256,
                "render_duration_sec": round(render_elapsed, 2),
                "output_frames_count": 24,
                "continuity_metrics": {
                    "identity_score": 0.89,
                    "motion_score": 0.86,
                    "flicker_score": 0.11,
                    "audio_sync_score": 0.92,
                },
                "notes": "Rendered successfully via FreeCloud worker protocol",
            }
            self._post_json("/api/v1/workers/complete-chunk", completion_payload)
            logger.info("Chunk %s successfully completed and reported!", chunk_id)

            if os.path.exists(dummy_artifact_path):
                os.remove(dummy_artifact_path)

        except Exception as exc:
            logger.error("Failed processing chunk %s: %s", chunk_id, exc)
            fail_payload = {
                "worker_id": self.worker_id,
                "chunk_id": chunk_id,
                "job_id": job_id,
                "error_code": "RENDER_EXCEPTION",
                "error_details": str(exc),
                "will_retry": True,
            }
            try:
                self._post_json("/api/v1/workers/fail-chunk", fail_payload)
            except Exception as report_err:
                logger.error("Failed to report failure to backend: %s", report_err)


def main():
    parser = argparse.ArgumentParser(description="FreeCloud AI Studio Worker Runner")
    parser.add_argument(
        "--backend-url",
        default=os.getenv("FREECLOUD_BACKEND_URL", "http://localhost:8000"),
        help="FreeCloud AI Studio backend URL",
    )
    parser.add_argument(
        "--worker-id",
        default=f"gha_worker_{os.getenv('GITHUB_RUN_ID', 'local')}_{os.getpid()}",
        help="Unique worker identifier",
    )
    parser.add_argument(
        "--job-id",
        default=os.getenv("FREECLOUD_JOB_ID"),
        help="Target job ID (optional)",
    )
    parser.add_argument(
        "--max-idle-seconds",
        type=int,
        default=30,
        help="Max idle time before runner terminates",
    )
    args = parser.parse_args()

    runner = WorkerRunner(
        backend_url=args.backend_url,
        worker_id=args.worker_id,
        job_id=args.job_id,
    )
    runner.execute(max_idle_seconds=args.max_idle_seconds)


if __name__ == "__main__":
    main()
