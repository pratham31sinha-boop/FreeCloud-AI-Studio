"""ComfyUI server management and node discovery routes."""

from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from comfyui.client import comfyui_client
from comfyui.server_manager import comfyui_server_manager

router = APIRouter(prefix="/comfyui", tags=["ComfyUI Engine"])


@router.get("/health")
async def get_comfyui_health() -> Dict[str, Any]:
    """Check live status of the ComfyUI generation engine."""
    return await comfyui_server_manager.check_health()


@router.post("/start")
async def start_comfyui() -> Dict[str, Any]:
    """Launch the local ComfyUI server instance."""
    try:
        res = await comfyui_server_manager.start(wait_timeout_sec=30)
        return {"status": "success", "details": res}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start ComfyUI: {exc}",
        )


@router.post("/stop")
def stop_comfyui() -> Dict[str, Any]:
    """Terminate the local ComfyUI server process."""
    success = comfyui_server_manager.stop()
    return {"status": "stopped" if success else "failed"}


@router.get("/nodes")
async def list_available_nodes() -> Dict[str, Any]:
    """List all 700+ registered node classes in the live ComfyUI instance."""
    health = await comfyui_server_manager.check_health()
    if not health.get("online"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ComfyUI server is currently offline. Start the server first.",
        )
    nodes = await comfyui_client.get_object_info()
    return {"count": len(nodes), "sample_nodes": list(nodes.keys())[:25]}
