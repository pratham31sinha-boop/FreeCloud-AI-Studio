"""Tests for Storage Manager."""

import io
from pathlib import Path
from backend.storage.manager import StorageManager


def test_storage_manager_operations(tmp_path: Path):
    manager = StorageManager(base_data_dir=tmp_path)

    # 1. Save uploaded file
    test_bytes = b"sample_video_bytes_12345"
    meta = manager.save_upload("sample.mp4", test_bytes)
    assert meta["file_id"].endswith(".mp4")
    assert Path(meta["path"]).is_file()
    assert meta["size_bytes"] == str(len(test_bytes))

    # 2. Save chunk artifact
    chunk_path = manager.save_chunk_artifact(
        job_id="job_001",
        chunk_id="chk_001",
        content=b"chunk_artifact_bytes",
    )
    assert chunk_path.is_file()
    retrieved_chunk = manager.get_chunk_artifact("job_001", "chk_001")
    assert retrieved_chunk == chunk_path

    # 3. Save final output video
    output_path = manager.save_output_video("job_001", b"final_mp4_bytes")
    assert output_path.is_file()
    retrieved_output = manager.get_output_video("job_001")
    assert retrieved_output == output_path

    # 4. Save and load character profile
    char_dir = manager.save_character_profile(
        "char_test_01",
        {"name": "Jagdhatri Test", "role": "Detective"},
        face_images=[("front.png", b"fake_png_data")],
    )
    assert char_dir.is_dir()
    loaded_profile = manager.load_character_profile("char_test_01")
    assert loaded_profile["name"] == "Jagdhatri Test"
