"""ComfyUI API Client.

Interfaces with a running ComfyUI instance via REST and WebSocket APIs.
Includes fallback simulation for environments where ComfyUI is not yet running.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional
import httpx
from backend.config import settings

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for dispatching and tracking node execution in ComfyUI."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout_sec: Optional[int] = None,
    ):
        self.host = host or settings.comfyui_host
        self.port = port or settings.comfyui_port
        self.timeout_sec = timeout_sec or settings.comfyui_timeout_sec
        self.base_url = f"http://{self.host}:{self.port}"
        self.client_id = f"freecloud_{uuid.uuid4().hex[:8]}"

    async def check_health(self) -> Dict[str, Any]:
        """Check if ComfyUI service is reachable."""
        url = f"{self.base_url}/system_stats"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    return {
                        "online": True,
                        "base_url": self.base_url,
                        "stats": res.json(),
                    }
        except Exception as exc:
            logger.debug("ComfyUI not reachable at %s: %s", self.base_url, exc)

        return {
            "online": False,
            "base_url": self.base_url,
            "message": "ComfyUI instance not currently running (simulation mode active)",
        }

    async def queue_prompt(
        self, prompt_graph: Dict[str, Any], extra_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Submit a workflow prompt graph to ComfyUI for execution."""
        health = await self.check_health()
        if not health.get("online"):
            # Provide structured mock response when ComfyUI is offline
            mock_prompt_id = f"mock_prompt_{uuid.uuid4().hex[:12]}"
            logger.info("ComfyUI offline; returning simulated prompt execution ID %s", mock_prompt_id)
            return {
                "prompt_id": mock_prompt_id,
                "number": 1,
                "node_errors": {},
                "simulated": True,
            }

        url = f"{self.base_url}/prompt"
        payload = {
            "prompt": prompt_graph,
            "client_id": self.client_id,
            "extra_data": extra_data or {},
        }
        async with httpx.AsyncClient(timeout=float(self.timeout_sec)) as client:
            res = await client.post(url, json=payload)
            res.raise_for_status()
            return res.json()

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Retrieve execution history and generated image/video outputs for a prompt ID."""
        if prompt_id.startswith("mock_"):
            return {
                prompt_id: {
                    "status": {"completed": True, "messages": ["Execution completed (simulated)"]},
                    "outputs": {
                        "video_output_node": {
                            "videos": [{"filename": "simulated_scene.mp4", "type": "output"}]
                        }
                    },
                }
            }

        url = f"{self.base_url}/history/{prompt_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(url)
            res.raise_for_status()
            return res.json()


comfyui_client = ComfyUIClient()
