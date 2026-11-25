"""Unit tests for the Cloud Video Diffusion client, multi-engine quota failover, and DIS optical-flow 1080p streamer."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np

from backend.generation.cloud_diffusion import (
    CloudVideoDiffusionClient,
    center_crop_aspect,
    clean_diffusion_prompt,
    stream_diffusion_frames,
)
from github.worker_entrypoint import WorkerRunner


def _make_moving_frames(count: int = 12, width: int = 384, height: int = 256) -> list:
    frames = []
    for i in range(count):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (24, 30, 46)
        x = int((i / max(1, count - 1)) * (width - 80)) + 20
        img[80:160, x : x + 48] = (210, 225, 240)
        frames.append(img)
    return frames


def test_clean_diffusion_prompt_and_aspect_crop():
    """Prompt cleaner should prioritize Next Scene Continuation and strip brackets."""
    p_ext, neg = clean_diffusion_prompt(
        "[EXTENDED CLIP 2/4 | (5.0s-10.0s)] Next Scene Continuation: Act 2 (Drift): Drifting past cliffs | Original Story Prompt: Silver hypercar race",
        anime_style="wan_2_1_realistic",
        model_name="wan_2_1",
    )
    assert "Drifting past cliffs" in p_ext
    assert "Silver hypercar race" in p_ext
    assert "photorealistic" in p_ext
    assert "EXTENDED CLIP" not in p_ext
    assert "static picture" in neg

    p_anime, _ = clean_diffusion_prompt("Samurai duel under crimson moon", anime_style="ufotable_digital")
    assert "Ufotable" in p_anime

    wide = np.zeros((400, 1000, 3), dtype=np.uint8)
    cropped = center_crop_aspect(wide, 1920, 1080)
    assert cropped.shape[0] == 400
    assert abs((cropped.shape[1] / cropped.shape[0]) - (1920 / 1080)) < 0.03


def test_cloud_diffusion_quota_failover_and_subhop_chaining():
    """Client should skip exhausted ZeroGPU spaces cleanly and chain a 2nd I2V sub-hop on >=8s chunks."""
    client = CloudVideoDiffusionClient(hf_token=None)
    anchor = np.zeros((256, 384, 3), dtype=np.uint8)
    anchor[50:150, 100:220] = 200

    with (
        patch.object(client, "_call_wan_2_2_i2v", side_effect=lambda *a, **kw: _mark_and_none(client, "wan_2_2_i2v")) as m_wan,
        patch.object(client, "_call_ltx", return_value=_make_moving_frames(16)) as m_ltx,
    ):
        frames, engine = client.synthesize_chunk_frames(
            prompt="Silver hypercar accelerating along coastal highway",
            anime_style="wan_2_1_realistic",
            model_name="wan_2_1",
            seed=101,
            total_frames=240,
            duration_sec=10.0,
            anchor_bgr=anchor,
            is_extended=True,
        )

    assert frames is not None
    # Two chained hops of 16 frames minus 1 duplicate seam frame = 31 frames
    assert len(frames) == 31
    assert engine == "ltx_i2v+ltx_i2v"
    assert m_wan.call_count == 1
    assert "wan_2_2_i2v" in client.exhausted_engines
    assert m_ltx.call_count == 2


def _mark_and_none(client: CloudVideoDiffusionClient, engine: str):
    client._mark_error(engine, RuntimeError("You have exceeded your ZeroGPU quota"))
    return None


def test_stream_diffusion_frames_dis_optical_flow_interpolation():
    """DIS optical-flow streamer should interpolate smoothly without duplicate freeze frames."""
    src = _make_moving_frames(count=8, width=384, height=216)
    out = []
    state = stream_diffusion_frames(
        diffusion_frames=src,
        total_frames=16,
        width=640,
        height=360,
        is_chapter_start=True,
        chapter_title="CHAPTER I",
        on_frame=lambda _idx, f: out.append(f.copy()),
    )
    assert len(out) == 16
    assert out[0].shape == (360, 640, 3)
    deltas = [float(np.mean(np.abs(out[i].astype(np.float32) - out[i - 1].astype(np.float32)))) for i in range(1, len(out))]
    assert max(deltas) > 0.5
    assert state["end_zoom"] == 1.02


def test_worker_runner_streams_cloud_diffusion_mp4(monkeypatch):
    """WorkerRunner should stream CloudVideoDiffusionClient frames straight to H.264 MP4 when available."""
    monkeypatch.setenv("FREECLOUD_CLOUD_DIFFUSION", "auto")
    runner = WorkerRunner(backend_url="http://127.0.0.1:8000", worker_id="test_cloud_worker", model="wan_2_1")
    fake_frames = _make_moving_frames(count=12, width=384, height=216)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_mp4 = os.path.join(tmpdir, "cloud_chunk.mp4")
        with patch("backend.generation.cloud_diffusion.CloudVideoDiffusionClient.synthesize_chunk_frames", return_value=(fake_frames, "wan_2_2_i2v")):
            state = runner._render_generative_text_to_video(
                output_path=out_mp4,
                chunk_id="test_cloud_chk_001",
                prompt="Photorealistic hypercar coastal drift",
                total_frames=12,
                fps=24.0,
                width=640,
                height=360,
                lease={"sequence_index": 0, "total_chunks": 4, "model_name": "wan_2_1"},
            )
        assert os.path.exists(out_mp4)
        assert os.path.getsize(out_mp4) > 5000
        assert state["kinetic_theme"] == "cloud_diffusion_wan_2_2_i2v"
