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
    """Verify that ComfyUI client fails by default when offline, and only simulates when explicitly enabled."""
    async def _run():
        # 1. Default client (simulation_mode=False) must NOT report fake success
        client_strict = ComfyUIClient(host="127.0.0.1", port=65530, simulation_mode=False)
        health_strict = await client_strict.check_health()
        assert health_strict["online"] is False
        assert health_strict["simulation_mode"] is False

        # Queue prompt must raise error when offline by default
        try:
            await client_strict.queue_prompt({"1": {"class_type": "EmptyLatentImage"}})
            assert False, "Should have raised ComfyUIClientError"
        except Exception as exc:
            assert "Could not connect to ComfyUI" in str(exc)

        # 2. Client with explicit simulation_mode=True
        client_sim = ComfyUIClient(host="127.0.0.1", port=65530, simulation_mode=True)
        health_sim = await client_sim.check_health()
        assert health_sim["simulation_mode"] is True

        res = await client_sim.queue_prompt({"1": {"class_type": "EmptyLatentImage"}})
        assert res.get("simulated") is True
        assert "mock_prompt_" in res.get("prompt_id")

    asyncio.run(_run())

