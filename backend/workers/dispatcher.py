"""Worker dispatchers for orchestrating remote runners (GitHub Actions or local)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import httpx
from backend.config import settings

logger = logging.getLogger(__name__)


class BaseWorkerDispatcher(ABC):
    """Abstract interface for launching and provisioning worker runners."""

    @abstractmethod
    async def dispatch_workers(
        self, job_id: str, worker_count: int, extra_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Request the provisioning of worker_count runners for a job."""
        pass


class GitHubActionsDispatcher(BaseWorkerDispatcher):
    """Triggers GitHub Actions workflow runs via repository dispatch / workflow_dispatch."""

    def __init__(
        self,
        repository: Optional[str] = None,
        workflow_id: Optional[str] = None,
        pat_token: Optional[str] = None,
    ):
        self.repository = repository or settings.github_repository
        self.workflow_id = workflow_id or settings.github_workflow_id
        self.pat_token = pat_token or settings.github_pat_token

    async def dispatch_workers(
        self, job_id: str, worker_count: int, extra_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Dispatch a GitHub Actions workflow with job metadata."""
        if not self.pat_token or "placeholder" in self.pat_token:
            logger.warning(
                "GitHub PAT token is not configured or is placeholder. "
                "Skipping live GitHub dispatch. Fallback to registered workers."
            )
            return {
                "dispatched": False,
                "reason": "GITHUB_PAT_TOKEN not configured",
                "simulated": True,
                "target_workers": worker_count,
            }

        url = f"https://api.github.com/repos/{self.repository}/actions/workflows/{self.workflow_id}/dispatches"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {self.pat_token}",
            "User-Agent": "FreeCloud-AI-Studio-Orchestrator",
        }
        payload = {
            "ref": "main",
            "inputs": {
                "job_id": job_id,
                "worker_count": str(worker_count),
                "backend_url": f"http://{settings.host}:{settings.port}",
                **(extra_inputs or {}),
            },
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code in (200, 204):
                    logger.info(
                        "Dispatched GitHub Actions workflow for job %s with %d workers",
                        job_id,
                        worker_count,
                    )
                    return {
                        "dispatched": True,
                        "status_code": response.status_code,
                        "target_workers": worker_count,
                    }
                else:
                    logger.error(
                        "GitHub Actions dispatch failed: %d - %s",
                        response.status_code,
                        response.text,
                    )
                    return {
                        "dispatched": False,
                        "status_code": response.status_code,
                        "error": response.text,
                    }
        except Exception as exc:
            logger.error("Exception during GitHub Actions dispatch: %s", exc)
            return {"dispatched": False, "error": str(exc)}


class LocalWorkerDispatcher(BaseWorkerDispatcher):
    """Local worker launcher / simulator for offline testing without GitHub credentials."""

    def __init__(self):
        self.dispatched_jobs: Dict[str, int] = {}

    async def dispatch_workers(
        self, job_id: str, worker_count: int, extra_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.dispatched_jobs[job_id] = worker_count
        logger.info(
            "LocalDispatcher simulated launch of %d worker(s) for job %s",
            worker_count,
            job_id,
        )
        return {
            "dispatched": True,
            "mode": "local_mock",
            "target_workers": worker_count,
            "job_id": job_id,
        }
