"""Tests for ComfyUI Integration and Workflow Graph Builder."""

import asyncio
from comfyui.client import ComfyUIClient
from comfyui.workflow_builder import ComfyUIWorkflowBuilder


def test_workflow_builder_structure():
    """Verify generated ComfyUI node graph contains essential execution nodes."""
    builder = ComfyUIWorkflowBuilder()
    graph = builder.build_anime_video_workflow(
        prompt="Jagdhatri action kick scene",
        checkpoint_name="animagine_xl_v3.safetensors",
        lora_name="jagdhatri_anime_v1.safetensors",
        frame_count=24,
    )

    # Validate essential nodes exist
    class_types = [node.get("class_type") for node in graph.values()]
    assert "CheckpointLoaderSimple" in class_types
    assert "CLIPTextEncode" in class_types
    assert "EmptyLatentImage" in class_types
    assert "AnimateDiffLoaderWithContext" in class_types
    assert "AnimateDiffApply" in class_types
    assert "KSampler" in class_types
    assert "VAEDecode" in class_types
    assert "VHS_VideoCombine" in class_types
    assert "LoraLoader" in class_types


def test_comfyui_client_offline_simulation():
    """Verify that ComfyUI client gracefully falls back to simulation mode when offline."""
    async def _run():
        client = ComfyUIClient(host="127.0.0.1", port=65530)  # Unbound port
        health = await client.check_health()
        assert health["online"] is False
        assert "simulation mode" in health["message"]

        # Queue prompt should return simulated prompt id rather than crashing
        res = await client.queue_prompt({"1": {"class_type": "EmptyLatentImage"}})
        assert res.get("simulated") is True
        assert "mock_prompt_" in res.get("prompt_id")

    asyncio.run(_run())

