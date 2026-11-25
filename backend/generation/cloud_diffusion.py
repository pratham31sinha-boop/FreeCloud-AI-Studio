"""Cloud neural video diffusion engine (LTX-Video Distilled, Wan 2.2 I2V, HunyuanVideo 1.5, FLUX.1-schnell).

Provides authentic generative AI motion synthesis for both opening Text-to-Video shots
and sequential Extend-Clip Image-to-Video continuation chained from previous chunk last frames,
with bidirectional DIS optical-flow interpolation and Lanczos4 1080p micro-sharpening.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("freecloud.cloud_diffusion")

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, static picture, frozen frame, inconsistent motion, blurry, "
    "jittery, distorted anatomy, watermark, overlaid text, frame border, lowres"
)


def clean_diffusion_prompt(
    prompt: str,
    anime_style: str = "",
    model_name: str = "",
) -> Tuple[str, str]:
    """Extract clean vivid action prompt and style enhancer without bracket boilerplate."""
    raw = (prompt or "").strip()
    # Extract explicit Next Scene Continuation first if present in Extend-Clip prompt
    m_next = re.search(
        r"Next Scene Continuation:\s*(.+?)(?:\s*\|\s*Original Story Prompt:|$)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    m_orig = re.search(r"Original Story Prompt:\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)

    if m_next:
        next_part = re.sub(r"Act \d+\s*\([^)]+\):\s*", "", m_next.group(1)).strip()
        orig_part = (m_orig.group(1).strip() if m_orig else "")
        core = f"{next_part}. {orig_part}".strip(". ")
    else:
        core = re.sub(r"\[[^\]]*\]", "", raw)
        core = re.sub(r"--\s*Act \d+\s*\([^)]+\):\s*", ", ", core)

    core = " ".join(core.split())[:300].strip()
    if not core:
        core = "Sleek futuristic silver hypercar speeding down a sunset coastal cliff highway"

    style_l = (anime_style or "").lower()
    core_l = core.lower()
    if any(k in style_l or k in core_l for k in ("anime", "ufotable", "shinkai", "ghibli", "mappa", "samurai", "katana", "ronin")):
        enhancer = (
            "cinematic anime animation, Ufotable and Makoto Shinkai studio sakuga quality, "
            "dynamic fluid motion, volumetric lighting, 24fps film grain, masterpiece, 8k"
        )
    elif "cyberpunk" in core_l or "neon" in core_l:
        enhancer = (
            "photorealistic cyberpunk cinema, wet neon asphalt reflections, dramatic camera tracking shot, "
            "dynamic motion blur, Unreal Engine 5 realism, anamorphic lens, 8k"
        )
    else:
        enhancer = (
            "photorealistic live-action cinema footage, RED V-Raptor 35mm anamorphic lens, "
            "smooth continuous forward physical motion, golden hour lighting, 8k UHD"
        )

    final_prompt = f"{core}, {enhancer}"
    return final_prompt, DEFAULT_NEGATIVE_PROMPT


def center_crop_aspect(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop frame cleanly to match target aspect ratio without stretching or bars."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
    target_ar = target_w / float(max(1, target_h))
    src_ar = w / float(h)
    if abs(src_ar - target_ar) < 0.02:
        return frame
    if src_ar > target_ar:
        new_w = max(2, int(round(h * target_ar)))
        x0 = max(0, (w - new_w) // 2)
        return frame[:, x0 : x0 + new_w]
    new_h = max(2, int(round(w / target_ar)))
    y0 = max(0, (h - new_h) // 2)
    return frame[y0 : y0 + new_h, :]


def _extract_video_path(res: Any) -> Optional[str]:
    """Safely normalize Gradio client prediction results (dict, tuple, list, str)."""
    if isinstance(res, (tuple, list)) and len(res) > 0:
        first = res[0]
        if isinstance(first, dict):
            return first.get("video") or first.get("path")
        if isinstance(first, str):
            return first
    elif isinstance(res, dict):
        return res.get("video") or res.get("path")
    elif isinstance(res, str):
        return res
    return None


def _decode_video_frames(video_path: Optional[str], min_frames: int = 8) -> Optional[List[np.ndarray]]:
    """Decode local MP4 file from Gradio result into BGR frames."""
    if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        return None
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            frames.append(frm)
    finally:
        cap.release()
    return frames if len(frames) >= min_frames else None


class CloudVideoDiffusionClient:
    """Orchestrates multi-engine ZeroGPU video diffusion with automatic quota failover."""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or None
        self.exhausted_engines: Set[str] = set()
        self.last_quota_error: Optional[str] = None

    def _make_client(self, space_id: str) -> Any:
        from gradio_client import Client

        if self.hf_token:
            return Client(space_id, token=self.hf_token)
        return Client(space_id)

    def _mark_error(self, engine_name: str, exc: Exception) -> None:
        msg = str(exc)
        logger.warning("[%s] Cloud diffusion error: %s", engine_name, msg[:260])
        lower = msg.lower()
        if any(k in lower for k in ("quota", "zerogpu", "exceeded", "429", "rate limit")):
            self.exhausted_engines.add(engine_name)
            self.last_quota_error = msg

    def _call_ltx(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        anchor_path: Optional[str] = None,
        width: int = 768,
        height: int = 448,
        duration_ui: float = 4.0,
    ) -> Optional[List[np.ndarray]]:
        """Run Lightricks/ltx-video-distilled T2V or Image-to-Video Extend Clip."""
        from gradio_client import handle_file

        engine_key = "ltx_i2v" if anchor_path else "ltx_t2v"
        t0 = time.time()
        try:
            client = self._make_client("Lightricks/ltx-video-distilled")
            w_adj = max(256, (int(width) // 32) * 32)
            h_adj = max(256, (int(height) // 32) * 32)
            if anchor_path:
                res = client.predict(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    input_image_filepath=handle_file(anchor_path),
                    input_video_filepath=None,
                    height_ui=h_adj,
                    width_ui=w_adj,
                    mode="image-to-video",
                    duration_ui=min(4.0, max(2.0, float(duration_ui))),
                    ui_frames_to_use=9,
                    seed_ui=seed % 1000000,
                    randomize_seed=False,
                    ui_guidance_scale=1,
                    improve_texture_flag=True,
                    api_name="/image_to_video",
                )
            else:
                res = client.predict(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    input_image_filepath=None,
                    input_video_filepath=None,
                    height_ui=h_adj,
                    width_ui=w_adj,
                    mode="text-to-video",
                    duration_ui=min(4.0, max(2.0, float(duration_ui))),
                    ui_frames_to_use=9,
                    seed_ui=seed % 1000000,
                    randomize_seed=False,
                    ui_guidance_scale=1,
                    improve_texture_flag=True,
                    api_name="/text_to_video",
                )
            frames = _decode_video_frames(_extract_video_path(res))
            if frames:
                logger.info("[%s] Generated %d real diffusion frames in %.2fs", engine_key, len(frames), time.time() - t0)
                return frames
        except Exception as exc:
            self._mark_error(engine_key, exc)
        return None

    def _call_wan_2_2_i2v(
        self,
        anchor_path: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        duration_seconds: float = 4.0,
    ) -> Optional[List[np.ndarray]]:
        """Run Wan 2.2 (zerogpu-aoti/wan2-2-fp8da-aoti-faster) Image-to-Video."""
        from gradio_client import handle_file

        t0 = time.time()
        try:
            client = self._make_client("zerogpu-aoti/wan2-2-fp8da-aoti-faster")
            res = client.predict(
                input_image=handle_file(anchor_path),
                prompt=prompt,
                steps=6,
                negative_prompt=negative_prompt,
                duration_seconds=min(4.0, max(2.5, float(duration_seconds))),
                guidance_scale=1.0,
                guidance_scale_2=1.0,
                seed=seed % 2147483647,
                randomize_seed=False,
                api_name="/generate_video",
            )
            frames = _decode_video_frames(_extract_video_path(res))
            if frames:
                logger.info("[wan_2_2_i2v] Generated %d real Wan 2.2 frames in %.2fs", len(frames), time.time() - t0)
                return frames
        except Exception as exc:
            self._mark_error("wan_2_2_i2v", exc)
        return None

    def _call_hunyuan_1_5_i2v(
        self,
        anchor_path: str,
        prompt: str,
        seed: int,
        length: int = 61,
    ) -> Optional[List[np.ndarray]]:
        """Run HunyuanVideo 1.5 (multimodalart/Hunyuan-Video-1-5) Image-to-Video."""
        from gradio_client import handle_file

        t0 = time.time()
        try:
            client = self._make_client("multimodalart/Hunyuan-Video-1-5")
            res = client.predict(
                input_image=handle_file(anchor_path),
                prompt=prompt,
                length=length,
                steps=6,
                shift=5.0,
                seed=seed % 2147483647,
                guidance=1.0,
                do_rewrite=True,
                api_name="/generate",
            )
            frames = _decode_video_frames(_extract_video_path(res))
            if frames:
                logger.info("[hunyuan_1_5_i2v] Generated %d Hunyuan 1.5 frames in %.2fs", len(frames), time.time() - t0)
                return frames
        except Exception as exc:
            self._mark_error("hunyuan_1_5_i2v", exc)
        return None

    def _synthesize_flux_opening_anchor(
        self,
        prompt: str,
        seed: int,
        width: int = 896,
        height: int = 512,
    ) -> Optional[str]:
        """Fallback T2I opening frame using FLUX.1-schnell so Wan2.2/Hunyuan can animate Clip 0 if LTX T2V hits quota."""
        t0 = time.time()
        try:
            client = self._make_client("black-forest-labs/FLUX.1-schnell")
            res = client.predict(
                prompt=prompt,
                seed=seed % 2147483647,
                randomize_seed=False,
                width=width,
                height=height,
                num_inference_steps=4,
                api_name="/infer",
            )
            img_path = None
            if isinstance(res, (tuple, list)) and len(res) > 0:
                first = res[0]
                img_path = first.get("path") if isinstance(first, dict) else (first if isinstance(first, str) else None)
            elif isinstance(res, dict):
                img_path = res.get("path")
            if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
                logger.info("[flux_t2i] Generated crisp opening keyframe in %.2fs", time.time() - t0)
                return img_path
        except Exception as exc:
            self._mark_error("flux_t2i", exc)
        return None

    def synthesize_clip(
        self,
        prompt: str,
        negative_prompt: str,
        model_name: str,
        seed: int,
        anchor_bgr: Optional[np.ndarray] = None,
        is_extended: bool = False,
        width: int = 768,
        height: int = 448,
        duration_ui: float = 4.0,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[str]]:
        """Generate one real AI diffusion clip (~4s) with failover across ZeroGPU spaces."""
        temp_anchor_path: Optional[str] = None
        try:
            if not is_extended or anchor_bgr is None:
                if "ltx_t2v" not in self.exhausted_engines:
                    frames = self._call_ltx(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        anchor_path=None,
                        width=width,
                        height=height,
                        duration_ui=duration_ui,
                    )
                    if frames:
                        return frames, "ltx_t2v"

                # Failover if LTX T2V unavailable: FLUX keyframe -> Wan 2.2 / Hunyuan I2V
                if "flux_t2i" not in self.exhausted_engines:
                    flux_path = self._synthesize_flux_opening_anchor(prompt, seed=seed, width=896, height=512)
                    if flux_path:
                        if "wan_2_2_i2v" not in self.exhausted_engines:
                            frames = self._call_wan_2_2_i2v(flux_path, prompt, negative_prompt, seed, duration_ui)
                            if frames:
                                return frames, "flux_wan_2_2"
                        if "hunyuan_1_5_i2v" not in self.exhausted_engines:
                            frames = self._call_hunyuan_1_5_i2v(flux_path, prompt, seed)
                            if frames:
                                return frames, "flux_hunyuan_1_5"
                return None, None

            # Extended Clip I2V: condition tightly on previous clip's actual last frame
            cropped = center_crop_aspect(anchor_bgr, width, height)
            resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
            temp_anchor_path = f"/tmp/fc_anchor_{uuid.uuid4().hex[:10]}.png"
            cv2.imwrite(temp_anchor_path, resized)

            model_l = (model_name or "").lower()
            if "wan" in model_l:
                order = ["wan_2_2_i2v", "ltx_i2v", "hunyuan_1_5_i2v"]
            elif "hunyuan" in model_l:
                order = ["hunyuan_1_5_i2v", "ltx_i2v", "wan_2_2_i2v"]
            else:
                order = ["ltx_i2v", "wan_2_2_i2v", "hunyuan_1_5_i2v"]

            for eng in order:
                if eng in self.exhausted_engines:
                    continue
                if eng == "ltx_i2v":
                    frames = self._call_ltx(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        anchor_path=temp_anchor_path,
                        width=width,
                        height=height,
                        duration_ui=duration_ui,
                    )
                elif eng == "wan_2_2_i2v":
                    frames = self._call_wan_2_2_i2v(
                        anchor_path=temp_anchor_path,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        duration_seconds=duration_ui,
                    )
                else:
                    frames = self._call_hunyuan_1_5_i2v(
                        anchor_path=temp_anchor_path,
                        prompt=prompt,
                        seed=seed,
                    )
                if frames:
                    return frames, eng
            return None, None
        finally:
            if temp_anchor_path and os.path.exists(temp_anchor_path):
                try:
                    os.remove(temp_anchor_path)
                except Exception:
                    pass

    def synthesize_chunk_frames(
        self,
        prompt: str,
        anime_style: str,
        model_name: str,
        seed: int,
        total_frames: int,
        duration_sec: float,
        anchor_bgr: Optional[np.ndarray] = None,
        is_extended: bool = False,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[str]]:
        """Generate authentic neural diffusion frames for a chunk, chaining a 2nd I2V sub-hop on >=8s chunks."""
        clean_p, neg_p = clean_diffusion_prompt(prompt, anime_style=anime_style, model_name=model_name)
        hop1_frames, eng1 = self.synthesize_clip(
            prompt=clean_p,
            negative_prompt=neg_p,
            model_name=model_name,
            seed=seed,
            anchor_bgr=anchor_bgr,
            is_extended=is_extended,
        )
        if not hop1_frames:
            return None, None

        all_frames = list(hop1_frames)
        used_engine = eng1 or "cloud_diffusion"

        # If chunk >= 8s, chain one seamless sub-hop I2V extension from hop1's last frame for double unique motion
        if duration_sec >= 8.0 and len(all_frames) < total_frames * 0.72:
            hop2_frames, eng2 = self.synthesize_clip(
                prompt=clean_p,
                negative_prompt=neg_p,
                model_name=model_name,
                seed=seed + 101,
                anchor_bgr=all_frames[-1],
                is_extended=True,
            )
            if hop2_frames and len(hop2_frames) > 2:
                # Match resolution to hop1 before appending (skip frame 0 duplicate of hop1[-1])
                th, tw = all_frames[0].shape[:2]
                for frm in hop2_frames[1:]:
                    if frm.shape[:2] != (th, tw):
                        frm = cv2.resize(center_crop_aspect(frm, tw, th), (tw, th), interpolation=cv2.INTER_LANCZOS4)
                    all_frames.append(frm)
                used_engine = f"{used_engine}+{eng2}"

        return all_frames, used_engine


def stream_diffusion_frames(
    diffusion_frames: List[np.ndarray],
    total_frames: int,
    width: int = 1920,
    height: int = 1080,
    seq_idx: int = 0,
    is_chapter_start: bool = False,
    is_chapter_end: bool = False,
    total_chunks: int = 4,
    chapter_title: Optional[str] = None,
    chapter_subtitle: Optional[str] = None,
    gamma_lut: Optional[np.ndarray] = None,
    on_frame: Optional[Callable[[int, np.ndarray], None]] = None,
) -> Dict[str, Any]:
    """Interpolate native diffusion frames using DIS Optical Flow, upscale Lanczos4 -> 1080p, sharpen, and stream."""
    if not diffusion_frames:
        raise ValueError("diffusion_frames cannot be empty")

    # Normalize aspect ratio to target (16:9) cleanly once per source frame
    norm_w = min(896, max(512, (diffusion_frames[0].shape[1] // 16) * 16))
    norm_h = max(288, int(round(norm_w * height / float(width))))
    norm_h = (norm_h // 8) * 8

    norm_frames: List[np.ndarray] = []
    for frm in diffusion_frames:
        cropped = center_crop_aspect(frm, width, height)
        if cropped.shape[1] != norm_w or cropped.shape[0] != norm_h:
            cropped = cv2.resize(cropped, (norm_w, norm_h), interpolation=cv2.INTER_LANCZOS4)
        norm_frames.append(cropped)

    n_src = len(norm_frames)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    fw, fh = max(192, norm_w // 2), max(108, norm_h // 2)
    grays = [cv2.cvtColor(cv2.resize(f, (fw, fh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in norm_frames]
    flows = [dis.calc(grays[i], grays[i + 1], None) for i in range(n_src - 1)]

    gy, gx = np.mgrid[0:norm_h, 0:norm_w].astype(np.float32)
    sx, sy = norm_w / float(fw), norm_h / float(fh)

    # Pre-render optional Documentary Chapter Slate once
    slate_roi = None
    slate_rect = None
    scale = width / 1920.0
    if is_chapter_start and chapter_title:
        sl_w = min(int(width * 0.62), 1180)
        sl_h = max(72, min(140, int(round(108 * scale))))
        sl_x = int(width * 0.045)
        sl_y = height - sl_h - int(48 * scale)
        slate_img = np.zeros((sl_h, sl_w, 3), dtype=np.uint8)
        slate_img[:] = (18, 16, 22)
        cv2.rectangle(slate_img, (0, 0), (8, sl_h), (70, 200, 255), -1)
        cv2.putText(slate_img, chapter_title, (24, int(sl_h * 0.44)), cv2.FONT_HERSHEY_SIMPLEX, max(0.65, 0.85 * scale), (245, 250, 255), 2, cv2.LINE_AA)
        if chapter_subtitle:
            cv2.putText(slate_img, chapter_subtitle, (24, int(sl_h * 0.80)), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, 0.58 * scale), (180, 225, 245), 1, cv2.LINE_AA)
        slate_roi = slate_img
        slate_rect = (sl_x, sl_y, sl_w, sl_h)

    for idx in range(total_frames):
        u = idx / float(max(1, total_frames - 1))
        pos = u * (n_src - 1)
        i0 = min(max(0, n_src - 2), int(pos))
        i1 = min(n_src - 1, i0 + 1)
        a = float(pos - i0)

        if i0 == i1 or a <= 0.04:
            blended = norm_frames[i0]
        elif a >= 0.96:
            blended = norm_frames[i1]
        else:
            fl = cv2.resize(flows[i0], (norm_w, norm_h), interpolation=cv2.INTER_LINEAR)
            w0 = cv2.remap(norm_frames[i0], gx - fl[:, :, 0] * (sx * a), gy - fl[:, :, 1] * (sy * a), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w1 = cv2.remap(norm_frames[i1], gx + fl[:, :, 0] * (sx * (1.0 - a)), gy + fl[:, :, 1] * (sy * (1.0 - a)), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            blended = cv2.addWeighted(w0, 1.0 - a, w1, a, 0.0)

        # Lanczos4 1080p upscale + gentle Unsharp Mask micro-detail polish
        if (norm_w, norm_h) != (width, height):
            frame = cv2.resize(blended, (width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            frame = blended.copy()
        blur = cv2.GaussianBlur(frame, (0, 0), 1.35)
        cv2.addWeighted(frame, 1.20, blur, -0.20, 0.0, dst=frame)

        if gamma_lut is not None:
            cv2.LUT(frame, gamma_lut, dst=frame)

        if slate_roi is not None and slate_rect is not None and 6 <= idx <= min(total_frames - 12, 96):
            sx0, sy0, sw, sh = slate_rect
            alpha = ((idx - 6) / 12.0 * 0.80) if idx < 18 else (max(0.0, (96 - idx) / 14.0) * 0.80 if idx > 82 else 0.80)
            if alpha > 0.01:
                roi_dst = frame[sy0 : sy0 + sh, sx0 : sx0 + sw]
                cv2.addWeighted(slate_roi, alpha, roi_dst, 1.0 - alpha, 0.0, dst=roi_dst)

        if total_chunks >= 8:
            if is_chapter_start and idx < 8:
                cv2.convertScaleAbs(frame, dst=frame, alpha=(idx + 1) / 8.0)
            elif is_chapter_end and idx >= total_frames - 8:
                cv2.convertScaleAbs(frame, dst=frame, alpha=max(0.08, (total_frames - idx) / 8.0))

        if on_frame is not None:
            on_frame(idx, frame)

    return {
        "end_zoom": 1.02,
        "end_center_x": round(width * 0.5, 2),
        "end_center_y": round(height * 0.5, 2),
        "pan_vx": 0.0,
        "pan_vy": 0.0,
        "stride_phase_rad": round((seq_idx * 1.2) % (2.0 * math.pi), 4),
        "breath_phase_rad": 0.0,
        "flare_phase_t": round(seq_idx * (total_frames / 24.0), 3),
    }
