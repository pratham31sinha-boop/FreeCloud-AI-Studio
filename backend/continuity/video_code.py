"""Video-as-Code Continuity State Extractor and Encoder.

Extracts a complete programmatic ('video in the form of coding') state manifest
from a completed video chunk along with its exact ending frame. This coded
manifest—covering camera kinematics, optical flow momentum, lighting/color LUTs,
identity fingerprints, perceptual hash grid, and audio phase—is passed to the
next video generator alongside the last frame image and extended scene prompt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _rgb_to_hex(bgr: Tuple[float, float, float]) -> str:
    b, g, r = [int(max(0, min(255, round(v)))) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _extract_dominant_palette(small_bgr: np.ndarray, k: int = 4) -> List[str]:
    """Fast deterministic palette extraction on a 160x90 frame (<1ms)."""
    h, w = small_bgr.shape[:2]
    grid = small_bgr.reshape(-1, 3)
    # Uniform spatial sample of 64 points + median/quartile color regions
    step = max(1, len(grid) // 64)
    sampled = grid[::step]
    # Split into k spatial brightness buckets for speed & zero scipy/sklearn dep
    luma = sampled[:, 0] * 0.114 + sampled[:, 1] * 0.587 + sampled[:, 2] * 0.299
    order = np.argsort(luma)
    chunks = np.array_split(sampled[order], k)
    palette = []
    for ch in chunks:
        if len(ch) > 0:
            mean_bgr = ch.mean(axis=0)
            palette.append(_rgb_to_hex((mean_bgr[0], mean_bgr[1], mean_bgr[2])))
    return palette or ["#1e293b", "#38bdf8", "#0f172a", "#94a3b8"]


def extract_video_code_from_frames(
    prev_frame: Optional[np.ndarray],
    last_frame: np.ndarray,
    job_id: str,
    chunk_index: int,
    chunk_id: str,
    model_engine: str,
    prompt: str,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    end_timestamp_sec: float,
    total_duration_sec: float,
    master_seed: int,
    character_ids: List[str],
    worker_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the complete Video-as-Code state dictionary from ending frames + state."""
    ws = worker_state or {}
    small_last = cv2.resize(last_frame, (160, 90), interpolation=cv2.INTER_AREA)

    # 1. Visual Color & Lighting Profile
    bgr_means = [round(float(v), 2) for v in small_last.mean(axis=(0, 1))]
    bgr_stds = [round(float(v), 2) for v in small_last.std(axis=(0, 1))]
    hsv = cv2.cvtColor(small_last, cv2.COLOR_BGR2HSV)
    hsv_means = [round(float(v), 2) for v in hsv.mean(axis=(0, 1))]
    gray_last = cv2.cvtColor(small_last, cv2.COLOR_BGR2GRAY)
    luma_mean = round(float(gray_last.mean()), 2)
    edges = cv2.Canny(gray_last, 80, 180)
    edge_density = round(float((edges > 0).mean()), 4)
    palette_hex = _extract_dominant_palette(small_last, k=4)

    # 8x8 perceptual luminance grid
    grid_8x8 = cv2.resize(gray_last, (8, 8), interpolation=cv2.INTER_AREA)
    perceptual_grid = [int(v) for v in grid_8x8.flatten().tolist()]

    # SHA-256 of last frame raw bytes for exact verification
    _, png_buf = cv2.imencode(".png", small_last)
    frame_hash = hashlib.sha256(png_buf.tobytes()).hexdigest()[:24]

    # 2. Optical Flow & Motion Momentum
    mean_dx = float(ws.get("end_flow_dx", 0.0))
    mean_dy = float(ws.get("end_flow_dy", 0.0))
    motion_energy = float(ws.get("motion_energy", 0.5))

    if prev_frame is not None and prev_frame.size > 0:
        try:
            small_prev = cv2.resize(prev_frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_last, None, 0.5, 2, 11, 2, 5, 1.1, 0
            )
            # Scale flow back to target pixel coordinates
            scale_x = width / 160.0
            scale_y = height / 90.0
            dx_scaled = flow[..., 0] * scale_x
            dy_scaled = flow[..., 1] * scale_y
            mean_dx = round(float(dx_scaled.mean()), 3)
            mean_dy = round(float(dy_scaled.mean()), 3)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            motion_energy = round(float(min(1.0, mag.mean() / 4.0)), 3)
        except Exception as exc:
            logger.debug("Fast optical flow fallback: %s", exc)

    dominant_dir_deg = round(math.degrees(math.atan2(mean_dy, mean_dx)), 2) if (mean_dx or mean_dy) else 0.0

    # 3. Camera Kinematics & Cyclic Phases
    cam_state = {
        "end_zoom": round(float(ws.get("end_zoom", 1.0 + 0.08 * (chunk_index + 1))), 4),
        "end_center_x": round(float(ws.get("end_center_x", width / 2.0)), 2),
        "end_center_y": round(float(ws.get("end_center_y", height / 2.0)), 2),
        "end_roll_deg": round(float(ws.get("end_roll_deg", 0.0)), 3),
        "pan_velocity_px_per_sec": {
            "vx": round(float(ws.get("pan_vx", mean_dx * fps)), 2),
            "vy": round(float(ws.get("pan_vy", mean_dy * fps)), 2),
        },
        "zoom_velocity_per_sec": round(float(ws.get("zoom_velocity", 0.015)), 5),
        "letterbox_bars": {
            "top": int(ws.get("bar_top", round(height * 0.055))),
            "bottom": int(ws.get("bar_bottom", round(height * 0.060))),
        },
    }

    motion_state = {
        "kinetic_theme": ws.get("kinetic_theme", "cinematic_locomotion"),
        "stride_phase_rad": round(float(ws.get("stride_phase_rad", 0.0)), 4),
        "breath_phase_rad": round(float(ws.get("breath_phase_rad", 0.0)), 4),
        "flare_phase_t": round(float(ws.get("flare_phase_t", end_timestamp_sec)), 3),
        "optical_flow_summary": {
            "mean_dx_px": mean_dx,
            "mean_dy_px": mean_dy,
            "dominant_direction_deg": dominant_dir_deg,
            "motion_energy": motion_energy,
        },
    }

    visual_state = {
        "bgr_channel_means": bgr_means,
        "bgr_channel_stds": bgr_stds,
        "hsv_means": hsv_means,
        "dominant_palette_hex": palette_hex,
        "brightness_luma": luma_mean,
        "edge_density": edge_density,
        "gamma": round(float(ws.get("gamma", 0.95)), 3),
        "contrast_gain": round(float(ws.get("contrast_gain", 1.05)), 3),
    }

    audio_state = {
        "theme": ws.get("audio_theme", "shinkai_sunset"),
        "sample_rate": int(ws.get("sample_rate", 44100)),
        "end_phase_rad": round(float(ws.get("audio_end_phase_rad", 0.0)), 4),
        "end_rpm_hz": round(float(ws.get("audio_end_rpm_hz", 120.0)), 2),
    }

    # 4. Programmatic Video-as-Code Representation ("video in the form of coding")
    next_seq = chunk_index + 1
    code_lines = [
        f"# FreeCloud Video-as-Code Continuity State (v1.0)",
        f"# Generated from Chunk {chunk_index} ({chunk_id}) -> Target Chunk {next_seq}",
        f"video_state = {{",
        f"    'source_model': '{model_engine}',",
        f"    'resolution': ({width}, {height}), 'fps': {fps},",
        f"    'end_time_sec': {end_timestamp_sec:.2f}, 'total_frames': {total_frames},",
        f"    'camera': {{'zoom': {cam_state['end_zoom']}, 'center': ({cam_state['end_center_x']}, {cam_state['end_center_y']}), 'roll_deg': {cam_state['end_roll_deg']}, 'vel': ({cam_state['pan_velocity_px_per_sec']['vx']}, {cam_state['pan_velocity_px_per_sec']['vy']})}},",
        f"    'flow': {{'dx': {mean_dx}, 'dy': {mean_dy}, 'dir_deg': {dominant_dir_deg}, 'energy': {motion_energy}}},",
        f"    'phases': {{'stride_rad': {motion_state['stride_phase_rad']}, 'breath_rad': {motion_state['breath_phase_rad']}, 'flare_t': {motion_state['flare_phase_t']}}},",
        f"    'lighting': {{'luma': {luma_mean}, 'bgr_mean': {bgr_means}, 'palette': {palette_hex}, 'gamma': {visual_state['gamma']}, 'gain': {visual_state['contrast_gain']}}},",
        f"    'identity': {{'master_seed': {master_seed}, 'characters': {character_ids}, 'frame_sig': '{frame_hash}'}},",
        f"    'audio': {{'theme': '{audio_state['theme']}', 'phase_rad': {audio_state['end_phase_rad']}, 'rpm_hz': {audio_state['end_rpm_hz']}}},",
        f"}}",
        f"extend_clip(anchor_img='/api/v1/jobs/{job_id}/anchor/{next_seq}', state=video_state, seamless=True)",
    ]

    return {
        "schema_version": "fc-video-code/v1",
        "job_id": job_id,
        "source_chunk_index": chunk_index,
        "source_chunk_id": chunk_id,
        "target_chunk_index": next_seq,
        "model_engine": model_engine,
        "resolution": {"width": width, "height": height, "fps": fps},
        "temporal_state": {
            "total_frames_rendered": total_frames,
            "end_timestamp_sec": round(end_timestamp_sec, 2),
            "global_progress_ratio": round(
                end_timestamp_sec / max(0.1, total_duration_sec), 3
            ),
        },
        "camera_kinematics": cam_state,
        "motion_and_physics": motion_state,
        "visual_color_and_lighting": visual_state,
        "identity_and_structure": {
            "master_seed": master_seed,
            "character_ids": character_ids,
            "last_frame_sig": frame_hash,
            "perceptual_grid_8x8": perceptual_grid,
        },
        "audio_continuity": audio_state,
        "scene_code_script": "\n".join(code_lines),
    }


def extract_handoff_bundle_from_mp4(
    artifact_path: Path,
    anchor_output_path: Path,
    code_output_path: Path,
    job_id: str,
    chunk_index: int,
    chunk_id: str,
    model_engine: str,
    prompt: str,
    end_timestamp_sec: float,
    total_duration_sec: float,
    master_seed: int,
    character_ids: List[str],
    worker_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Extract last frame image AND complete Video-as-Code state from an MP4 chunk."""
    try:
        cap = cv2.VideoCapture(str(artifact_path))
        if not cap.isOpened():
            return False, None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        if frame_count <= 0:
            cap.release()
            return False, None

        prev_frame = None
        if frame_count >= 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 2)
            ret_p, f_p = cap.read()
            if ret_p and f_p is not None:
                prev_frame = f_p

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
        ret, last_frame = cap.read()
        cap.release()

        if not ret or last_frame is None or last_frame.size == 0:
            return False, None

        anchor_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(anchor_output_path), last_frame)

        video_code = extract_video_code_from_frames(
            prev_frame=prev_frame,
            last_frame=last_frame,
            job_id=job_id,
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            model_engine=model_engine,
            prompt=prompt,
            width=width,
            height=height,
            fps=fps,
            total_frames=frame_count,
            end_timestamp_sec=end_timestamp_sec,
            total_duration_sec=total_duration_sec,
            master_seed=master_seed,
            character_ids=character_ids,
            worker_state=worker_state,
        )

        code_output_path.write_text(json.dumps(video_code, indent=2), encoding="utf-8")
        return True, video_code
    except Exception as exc:
        logger.error("Failed extracting Video-as-Code handoff from %s: %s", artifact_path, exc)
        return False, None


def build_extended_clip_prompt(
    original_prompt: str,
    next_scene_segment: str,
    prev_chunk_index: int,
    next_chunk_index: int,
    video_code: Optional[Dict[str, Any]],
) -> str:
    """Format explicit extended-clip instruction combining last frame + coded state + next prompt."""
    clean_orig = (original_prompt or "").strip()
    # Strip any earlier [EXTENDED CLIP ...] banner before appending new one
    if "[EXTENDED CLIP" in next_scene_segment:
        parts = next_scene_segment.split("Next Scene Continuation:")
        core_beat = parts[-1].strip() if len(parts) > 1 else next_scene_segment
    else:
        core_beat = next_scene_segment

    if video_code:
        vis = video_code.get("visual_color_and_lighting", {})
        flow = video_code.get("motion_and_physics", {}).get("optical_flow_summary", {})
        cam = video_code.get("camera_kinematics", {})
        src_model = video_code.get("model_engine", "wan_2_1")
        code_summary = (
            f"from_model={src_model}, end_zoom={cam.get('end_zoom', 1.0)}x, "
            f"flow=({flow.get('mean_dx_px', 0)},{flow.get('mean_dy_px', 0)}), "
            f"luma={vis.get('brightness_luma', 120)}, palette={vis.get('dominant_palette_hex', [])[:2]}"
        )
    else:
        code_summary = "seamless_anchor_continuation"

    return (
        f"[EXTENDED CLIP {next_chunk_index + 1} | Continuing seamlessly from Clip {prev_chunk_index + 1} "
        f"Last Frame Image & Coded Video State ({code_summary})] "
        f"Preserve 100% identity, lighting, camera momentum, and environment consistency from the preceding clip. "
        f"Next Scene Continuation: {core_beat} | Original Story Prompt: {clean_orig}"
    )
