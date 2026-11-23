"""Real ComfyUI API & WebSocket Client.

Provides full REST and real-time WebSocket communication with a running ComfyUI instance.
Streams node execution progress, tracks queues, downloads real image/video artifacts,
and surfaces genuine server errors.
Simulation mode is strictly disabled by default.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional
import httpx
import websockets
from backend.config import settings

logger = logging.getLogger(__name__)


class ComfyUIClientError(RuntimeError):
    """Raised when ComfyUI encounters an execution error or is unreachable."""
    pass


class ComfyUIClient:
    """Client for dispatching workflows and tracking live execution in ComfyUI."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout_sec: Optional[int] = None,
        simulation_mode: bool = False,
    ):
        self.host = host or settings.comfyui_host
        self.port = port or settings.comfyui_port
        self.timeout_sec = timeout_sec or settings.comfyui_timeout_sec
        self.simulation_mode = simulation_mode
        self.base_url = f"http://{self.host}:{self.port}"
        self.ws_url = f"ws://{self.host}:{self.port}/ws"
        self.client_id = f"freecloud_{uuid.uuid4().hex[:8]}"

    async def check_health(self) -> Dict[str, Any]:
        """Check if ComfyUI service is reachable and query its system stats."""
        url = f"{self.base_url}/system_stats"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    return {
                        "online": True,
                        "base_url": self.base_url,
                        "stats": res.json(),
                        "simulation_mode": self.simulation_mode,
                    }
        except Exception as exc:
            logger.debug("ComfyUI check_health failed: %s", exc)

        return {
            "online": False,
            "base_url": self.base_url,
            "message": "ComfyUI server is offline or unreachable.",
            "simulation_mode": self.simulation_mode,
        }

    async def get_object_info(self, node_class: Optional[str] = None) -> Dict[str, Any]:
        """Query ComfyUI for available node types, inputs, and output types."""
        url = f"{self.base_url}/object_info"
        if node_class:
            url = f"{url}/{node_class}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(url)
            if res.status_code != 200:
                raise ComfyUIClientError(
                    f"Failed to fetch object_info from ComfyUI ({res.status_code}): {res.text}"
                )
            return res.json()

    async def queue_prompt(
        self, prompt_graph: Dict[str, Any], extra_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Submit a workflow prompt graph to ComfyUI for execution."""
        if self.simulation_mode:
            mock_prompt_id = f"mock_prompt_{uuid.uuid4().hex[:12]}"
            logger.info("Simulation mode active; returning mock prompt ID %s", mock_prompt_id)
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

        try:
            async with httpx.AsyncClient(timeout=float(self.timeout_sec)) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    error_text = res.text
                    logger.error("ComfyUI rejected prompt (%d): %s", res.status_code, error_text)
                    raise ComfyUIClientError(
                        f"ComfyUI prompt submission failed ({res.status_code}): {error_text}"
                    )
                return res.json()
        except httpx.RequestError as exc:
            raise ComfyUIClientError(f"Could not connect to ComfyUI at {url}: {exc}")

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Retrieve execution history and output filenames for a prompt ID."""
        if self.simulation_mode and prompt_id.startswith("mock_"):
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
            if res.status_code != 200:
                raise ComfyUIClientError(
                    f"History query failed for {prompt_id} ({res.status_code}): {res.text}"
                )
            return res.json()

    async def get_output_file_bytes(
        self, filename: str, subfolder: str = "", folder_type: str = "output"
    ) -> bytes:
        """Download a generated image or video file directly from ComfyUI /view."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url = f"{self.base_url}/view"
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.get(url, params=params)
            if res.status_code != 200:
                raise ComfyUIClientError(
                    f"Failed to download output file {filename} ({res.status_code}): {res.text}"
                )
            return res.content

    async def execute_and_wait(
        self,
        prompt_graph: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout_sec: float = 300.0,
    ) -> Dict[str, Any]:
        """Queue prompt and monitor execution to completion via WebSocket.

        Streams live progress steps, catches execution errors, and returns the final outputs.
        """
        if self.simulation_mode:
            logger.info("Executing workflow in simulated mode.")
            await asyncio.sleep(1.0)
            return {
                "prompt_id": f"sim_{uuid.uuid4().hex[:8]}",
                "status": "completed",
                "outputs": {"simulated": True},
            }

        # 1. Connect WebSocket first to avoid missing early messages
        ws_endpoint = f"{self.ws_url}?clientId={self.client_id}"
        try:
            async with websockets.connect(ws_endpoint) as ws:
                # 2. Queue the prompt
                queue_res = await self.queue_prompt(prompt_graph)
                prompt_id = queue_res.get("prompt_id")
                if not prompt_id:
                    raise ComfyUIClientError(f"No prompt_id returned from ComfyUI: {queue_res}")

                logger.info("Tracking ComfyUI execution for prompt %s over WebSocket...", prompt_id)
                start_time = asyncio.get_event_loop().time()

                # 3. Listen to WebSocket events until completion or error
                while True:
                    if asyncio.get_event_loop().time() - start_time > timeout_sec:
                        raise TimeoutError(f"Execution timed out after {timeout_sec}s for prompt {prompt_id}")

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Check history in case completion was missed
                        history = await self.get_history(prompt_id)
                        if prompt_id in history:
                            return {
                                "prompt_id": prompt_id,
                                "status": "completed",
                                "history": history[prompt_id],
                            }
                        continue

                    # Binary messages are previews; JSON messages are execution events
                    if isinstance(msg, str):
                        event = json.loads(msg)
                        event_type = event.get("type")
                        data = event.get("data", {})

                        if data.get("prompt_id") and data.get("prompt_id") != prompt_id:
                            continue

                        logger.debug("ComfyUI WS Event: %s -> %s", event_type, data)

                        if progress_callback:
                            try:
                                progress_callback({"type": event_type, "data": data})
                            except Exception as cb_err:
                                logger.warning("Progress callback error: %s", cb_err)

                        if event_type == "progress":
                            val = data.get("value", 0)
                            max_val = data.get("max", 1)
                            logger.info("ComfyUI Progress: %d/%d (%.1f%%)", val, max_val, (val / max_val) * 100.0)

                        elif event_type == "execution_error":
                            error_details = data.get("exception_message") or data.get("traceback") or str(data)
                            raise ComfyUIClientError(f"ComfyUI execution error in node {data.get('node_id')}: {error_details}")

                        elif event_type == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                            # Finished executing the entire prompt graph
                            logger.info("ComfyUI execution finished for prompt %s", prompt_id)
                            history = await self.get_history(prompt_id)
                            return {
                                "prompt_id": prompt_id,
                                "status": "completed",
                                "history": history.get(prompt_id, {}),
                            }

        except (websockets.exceptions.WebSocketException, OSError) as ws_err:
            raise ComfyUIClientError(f"WebSocket connection to ComfyUI failed: {ws_err}")


# Default global client (simulation_mode is strictly False)
comfyui_client = ComfyUIClient(simulation_mode=False)
