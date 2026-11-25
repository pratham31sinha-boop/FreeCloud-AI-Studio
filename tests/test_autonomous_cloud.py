"""Tests for 100% serverless autonomous GitHub cloud pipeline and static frontend mounts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from github.autonomous_cloud_pipeline import (
    build_autonomous_plan,
    parse_release_catalog,
    render_chapter_autonomous,
)

client = TestClient(app)


def test_autonomous_plan_parallel_chapter_sharding():
    """Verify duration splits accurately across parallel cloud VM chapters and sequential acts."""
    plan_24 = build_autonomous_plan(
        job_id="fc_test_24",
        prompt="Sunset hypercar race",
        duration_seconds=24.0,
        model_name="auto_relay",
        anime_style="wan_2_1_realistic",
        width=1920,
        height=1080,
    )
    assert len(plan_24["chapters"]) == 1
    assert len(plan_24["chapters"][0]["acts"]) == 4

    plan_60 = build_autonomous_plan(
        job_id="fc_test_60",
        prompt="Space odyssey",
        duration_seconds=60.0,
        model_name="auto_relay",
        anime_style="wan_2_1_realistic",
        width=1920,
        height=1080,
    )
    assert len(plan_60["chapters"]) == 3


def test_autonomous_render_chapter_with_diffusion_handoff(monkeypatch):
    """Autonomous chapter renderer must synthesize multi-act MP4 and extract last-frame Extend-Clip handoffs."""
    monkeypatch.setenv("FREECLOUD_CLOUD_DIFFUSION", "auto")
    plan = build_autonomous_plan(
        job_id="fc_test_auto",
        prompt="Silver hypercar drifting sunset highway",
        duration_seconds=12.0,
        model_name="auto_relay",
        anime_style="wan_2_1_realistic",
        width=640,
        height=360,
    )
    assert len(plan["chapters"]) == 1
    assert len(plan["chapters"][0]["acts"]) == 2

    fake_frames = [np.full((216, 384, 3), 40 + i * 8, dtype=np.uint8) for i in range(10)]
    seen_extended = []

    def _fake_synth(self, prompt, anime_style, model_name, seed, total_frames, duration_sec, anchor_bgr=None, is_extended=False):
        seen_extended.append((is_extended, anchor_bgr is not None))
        return fake_frames, "mock_ltx"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        with patch("backend.generation.cloud_diffusion.CloudVideoDiffusionClient.synthesize_chunk_frames", side_effect=_fake_synth, autospec=True):
            ch_mp4 = render_chapter_autonomous(plan, 0, out_dir)

        assert ch_mp4.is_file()
        assert ch_mp4.stat().st_size > 10000
        meta = json.loads((out_dir / "chapter_00_meta.json").read_text())
        assert meta["engines_used"] == ["mock_ltx", "mock_ltx"]
        # Act 0 is T2V opening (False, False); Act 1 receives extracted real last frame (True, True)
        assert seen_extended == [(False, False), (True, True)]


def test_parse_release_catalog_and_frontend_relative_routes():
    """Verify GitHub release catalog parsing and dual Pages/FastAPI static routes."""
    body = json.dumps({"schema": "freecloud-gallery/v1", "items": [{"job_id": "fc_a", "video_url": "https://x/a.mp4"}]})
    items = parse_release_catalog(body)
    assert len(items) == 1
    assert items[0]["job_id"] == "fc_a"

    assert client.get("/").status_code == 200
    assert client.get("/css/style.css").status_code == 200
    assert client.get("/js/app.js").status_code == 200
