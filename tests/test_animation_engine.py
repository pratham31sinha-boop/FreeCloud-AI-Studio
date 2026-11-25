"""Unit tests for the PyCairo cel-shaded vector animation engine and worker integration."""

from __future__ import annotations

import os
import tempfile
import numpy as np

from backend.animation.anime_engine import HAS_CAIRO, AnimeSceneRenderer, classify_animation_scene
from github.worker_entrypoint import WorkerRunner


def test_classify_animation_scene():
    """Verify prompt classification selects the expected visual archetype."""
    assert classify_animation_scene("Cyberpunk anime samurai lightning katana duel") == "anime_duel"
    assert classify_animation_scene("Emerald celestial dragon soaring above crystal floating sanctuary") == "celestial_dragon"
    assert classify_animation_scene("Journey across the cosmos and spiral galaxy nebula") == "cosmos_odyssey"
    assert classify_animation_scene("Neon hypercar drifting at 320 kmh on Tokyo highway") == "neon_hypercar"


def test_anime_scene_renderer_produces_genuine_motion():
    """All four animation archetypes must render moving multi-frame sequences."""
    assert HAS_CAIRO, "PyCairo must be available for high-end cel-shaded animation tests"

    prompts = {
        "anime_duel": "Cyberpunk anime samurai warrior unleashes lightning katana duel",
        "celestial_dragon": "Emerald celestial dragon soaring through glowing sanctuary",
        "cosmos_odyssey": "Voyager traversing a rotating spiral galaxy nebula in deep cosmos",
        "neon_hypercar": "Cyber hypercar drifting down rain-slicked Tokyo neon expressway",
    }

    renderer = AnimeSceneRenderer(width=640, height=360, fps=24.0)
    for expected_arch, prompt in prompts.items():
        frames = []
        state = renderer.render_sequence(
            prompt=prompt,
            total_frames=8,
            seq_idx=1,
            total_chunks=4,
            chunk_id=f"test_{expected_arch}",
            duration_sec=8 / 24.0,
            model_name="wan_2_1",
            is_chapter_start=(expected_arch == "cosmos_odyssey"),
            chapter_title="CHAPTER I" if expected_arch == "cosmos_odyssey" else None,
            on_frame=lambda _idx, frm: frames.append(frm.copy()),
        )

        assert len(frames) == 8
        assert frames[0].shape == (360, 640, 3)
        assert state["kinetic_theme"] == expected_arch
        assert "stride_phase_rad" in state
        assert "breath_phase_rad" in state
        assert "flare_phase_t" in state

        # Verify strong frame-to-frame animation delta across the sequence
        deltas = [
            float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
            for i in range(1, len(frames))
        ]
        assert max(deltas) > 1.0, f"Expected real frame motion for {expected_arch}, got deltas={deltas}"


def test_worker_runner_renders_animated_mp4_with_handoff(monkeypatch):
    """Verify WorkerRunner pipes fallback AnimeSceneRenderer output to H.264 MP4 cleanly when cloud diffusion is off."""
    monkeypatch.setenv("FREECLOUD_CLOUD_DIFFUSION", "off")
    runner = WorkerRunner(backend_url="http://127.0.0.1:8000", worker_id="test_anim_worker", model="minimax_h3")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_mp4 = os.path.join(tmpdir, "anim_chunk.mp4")
        prev_code = {
            "motion_and_physics": {"stride_phase_rad": 1.2, "breath_phase_rad": 0.4, "flare_phase_t": 2.5},
            "camera_kinematics": {"end_center_x": 320.0},
            "audio_continuity": {"end_phase_rad": 0.5, "end_rpm_hz": 110.0},
        }
        state = runner._render_generative_text_to_video(
            output_path=out_mp4,
            chunk_id="test_anim_chk_001",
            prompt="[EXTENDED CLIP 2 | Next Scene] Cyberpunk samurai katana lightning battle in Tokyo",
            total_frames=12,
            fps=24.0,
            width=640,
            height=360,
            lease={"sequence_index": 1, "total_chunks": 4, "model_name": "minimax_h3"},
            prev_video_code=prev_code,
        )
        assert os.path.exists(out_mp4)
        assert os.path.getsize(out_mp4) > 5000
        assert state["kinetic_theme"] == "anime_duel"
        assert "audio_theme" in state
        assert "audio_end_phase_rad" in state
