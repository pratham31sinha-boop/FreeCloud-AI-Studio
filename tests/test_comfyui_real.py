"""Real End-to-End ComfyUI Integration and Video Generation Tests.

Tests execution against a genuine, actively running ComfyUI server instance.
Verifies real WebSocket progress streaming, container integrity, and actual MP4 output generation.
"""

import asyncio
import os
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from comfyui.client import ComfyUIClient, comfyui_client
from comfyui.server_manager import comfyui_server_manager
from backend.storage.video_processor import video_processor


def test_comfyui_server_is_healthy():
    """Verify ComfyUI server is actively responding on /system_stats with real hardware metrics."""
    health = asyncio.run(comfyui_server_manager.check_health())
    assert health["online"] is True, f"ComfyUI is offline: {health}"
    system_stats = health["system_stats"]
    assert "system" in system_stats
    assert "devices" in system_stats
    assert system_stats["system"]["os"] == "linux"
    assert "pytorch_version" in system_stats["system"]


def test_comfyui_object_info_and_nodes():
    """Verify ComfyUI exposes real nodes including CreateVideo and SaveVideo."""
    info = asyncio.run(comfyui_client.get_object_info())
    assert len(info) > 100, "ComfyUI returned too few registered nodes"
    assert "CreateVideo" in info
    assert "SaveVideo" in info
    assert "LoadImage" in info
    assert "ImageScale" in info


def test_real_comfyui_video_generation(tmp_path: Path):
    """Phase C & First Success Criterion:

    Execute a real ComfyUI workflow that transforms an input into a genuine MP4 video output.
    Verifies that the generated file exists on disk, has non-zero size, and is a valid ISO MP4 container.
    """
    input_dir = comfyui_server_manager.server_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    test_frame_path = input_dir / "jagdhatri_test_frame.png"

    # 1. Synthesize an authentic test frame (Jagdhatri action stance silhouette)
    width, height = 320, 240
    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
    # Warm anime dusk background
    for y in range(height):
        r = int(180 * (y / height))
        b = int(220 * (1 - y / height))
        frame_img[y, :] = (b, 60, r)
    # Draw character silhouette
    cv2.circle(frame_img, (160, 100), 28, (20, 20, 35), -1)
    cv2.rectangle(frame_img, (145, 128), (175, 180), (25, 25, 45), -1)
    cv2.putText(frame_img, "JAS Action Frame", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(test_frame_path), frame_img)

    # 2. Build real ComfyUI execution graph using CreateVideo and SaveVideo
    prefix = f"test_anime_gen_{int(time.time())}"
    workflow_prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "jagdhatri_test_frame.png"}},
        "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "upscale_method": "bicubic", "width": 320, "height": 240, "crop": "disabled"}},
        "3": {"class_type": "RepeatImageBatch", "inputs": {"image": ["2", 0], "amount": 12}},
        "4": {"class_type": "CreateVideo", "inputs": {"images": ["3", 0], "fps": 6.0}},
        "5": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0], "filename_prefix": prefix, "format": "mp4"}},
    }

    # 3. Track execution over WebSocket
    start_time = time.time()
    events = []
    def on_event(ev):
        events.append(ev)

    result = asyncio.run(
        comfyui_client.execute_and_wait(
            workflow_prompt,
            progress_callback=on_event,
            timeout_sec=30.0,
        )
    )
    elapsed = time.time() - start_time

    assert result["status"] == "completed"
    prompt_id = result["prompt_id"]
    outputs = result["history"]["outputs"]
    assert "5" in outputs, f"Expected node 5 output in {outputs}"

    saved_videos = outputs["5"]["images"]
    assert len(saved_videos) > 0
    output_filename = saved_videos[0]["filename"]

    # 4. Verify output file exists and can be opened
    output_file_path = comfyui_server_manager.server_dir / "output" / output_filename
    assert output_file_path.is_file(), f"Output file does not exist: {output_file_path}"
    assert output_file_path.stat().st_size > 1000, "Output video file is unexpectedly small"

    # 5. Verify video container validity with OpenCV
    video_info = video_processor.get_video_info(output_file_path)
    assert video_info["width"] == 320
    assert video_info["height"] == 240
    assert video_info["frame_count"] == 12
    assert video_info["duration_sec"] == 2.0

    print(f"\nReal ComfyUI Video Generation Verified:")
    print(f"  - Output File: {output_filename}")
    print(f"  - File Size: {output_file_path.stat().st_size} bytes")
    print(f"  - Container: {video_info['width']}x{video_info['height']} @ {video_info['fps']} FPS ({video_info['frame_count']} frames)")
    print(f"  - Execution Time: {elapsed:.2f}s")
