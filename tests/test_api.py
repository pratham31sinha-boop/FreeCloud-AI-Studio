"""Integration tests for FastAPI endpoints."""

import io
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "FreeCloud AI Studio"


def test_system_status_endpoint():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "storage" in data
    assert "orchestration" in data


def test_allocation_estimate_endpoint():
    response = client.get("/api/v1/allocation-estimate?duration_minutes=3.0")
    assert response.status_code == 200
    data = response.json()
    assert data["planned_workers"] == 6
    assert len(data["table"]) == 6


def test_comfyui_api_endpoints():
    health_res = client.get("/api/v1/comfyui/health")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["online"] is True

    nodes_res = client.get("/api/v1/comfyui/nodes")
    assert nodes_res.status_code == 200
    nodes_data = nodes_res.json()
    assert nodes_data["count"] > 100
    assert "CreateVideo" in nodes_data["sample_nodes"] or len(nodes_data["sample_nodes"]) > 0


def test_continuity_characters_endpoint():
    response = client.get("/api/v1/continuity/characters")
    assert response.status_code == 200
    characters = response.json()
    assert len(characters) >= 2
    char_ids = [c["character_id"] for c in characters]
    assert "jagdhatri_main" in char_ids
    assert "jash_lead" in char_ids


def test_video_upload_endpoint():
    dummy_video = io.BytesIO(b"fake_mp4_content_for_testing")
    files = {"file": ("test_scene.mp4", dummy_video, "video/mp4")}
    response = client.post("/api/v1/upload/video", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "file_id" in data


def test_job_submission_and_worker_workflow():
    # 1. Submit Job
    job_payload = {
        "story_prompt": "Jagdhatri confronts the syndicate in the docks.",
        "duration_minutes": 1.0,
        "anime_style": "ufotable_digital",
        "character_ids": ["jagdhatri_main", "jash_lead"],
        "continuity_identity_strict": True,
        "continuity_motion_tracking": True,
        "continuity_audio_sync": True,
    }
    create_res = client.post("/api/v1/jobs", json=job_payload)
    assert create_res.status_code == 201
    job = create_res.json()
    job_id = job["job_id"]
    assert job["target_worker_count"] == 4

    # 2. Register a test worker
    reg_res = client.post(
        "/api/v1/workers/register",
        json={"worker_id": "test_api_runner_01", "runner_type": "integration-test"},
    )
    assert reg_res.status_code == 200
    assert reg_res.json()["accepted"] is True

    # 3. Worker claims a task chunk
    claim_res = client.post("/api/v1/workers/claim-task?worker_id=test_api_runner_01")
    assert claim_res.status_code == 200
    lease = claim_res.json()
    assert lease["has_task"] is True
    chunk_id = lease["chunk_id"]

    # 4. Upload chunk artifact
    dummy_artifact = io.BytesIO(b"rendered_chunk_mp4_bytes")
    artifact_res = client.post(
        f"/api/v1/workers/test_api_runner_01/chunks/{chunk_id}/artifact?job_id={job_id}",
        files={"file": (f"{chunk_id}.mp4", dummy_artifact, "video/mp4")},
    )
    assert artifact_res.status_code == 200
    assert artifact_res.json()["status"] == "received"

    # 5. Report chunk completion
    comp_res = client.post(
        "/api/v1/workers/complete-chunk",
        json={
            "worker_id": "test_api_runner_01",
            "chunk_id": chunk_id,
            "job_id": job_id,
            "sha256_checksum": "dummy_checksum_123",
            "render_duration_sec": 1.5,
            "output_frames_count": 24,
            "continuity_metrics": {"identity_score": 0.92},
        },
    )
    assert comp_res.status_code == 200
    assert comp_res.json()["status"] == "recorded"

    # 6. Verify job status
    status_res = client.get(f"/api/v1/jobs/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["completed_chunks"] >= 1


def test_video_analysis_endpoint():
    # Test sample endpoint with 6s duration for fast test execution
    res = client.get("/api/v1/analysis/jagdhatri-sample?max_duration_sec=6.0")
    assert res.status_code == 200
    report = res.json()
    assert "video_path" in report
    assert report["total_frames"] > 0
    assert len(report["scenes"]) > 0
    assert report["scenes"][0]["camera_movement"] in [
        "static",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "zoom",
        "dynamic_action",
    ]
