"""High-end 1080p vector cel-shaded & 2.5D parallax animation engine.

Renders genuine frame-by-frame articulated character animation, sculpted bezier
anime anatomy, multi-plane parallax cities/sanctuaries, 3D-projected magic rings
& rotating spiral galaxies, cloth/hair wave physics, crescent sword slash waves,
branching lightning bolts, perspective speedlines, and additive volumetric bloom.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import cairo  # type: ignore

    HAS_CAIRO = True
except ImportError:
    cairo = None
    HAS_CAIRO = False


def classify_animation_scene(prompt: str) -> str:
    """Classify user prompt into a rich animated choreography archetype."""
    text = (prompt or "").lower()
    has_word = lambda *words: any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)

    if has_word("car", "hypercar", "supercar", "racing", "drift", "highway", "formula", "bike"):
        return "neon_hypercar"
    if has_word("cosmos", "space", "galaxy", "universe", "nebula", "planet", "black", "hole", "solar", "astronomy", "starship", "rocket"):
        return "cosmos_odyssey"
    if has_word("dragon", "elven", "guardian", "angel", "fairy", "magic", "spirit", "forest", "sanctuary", "wing", "wings"):
        return "celestial_dragon"
    return "anime_duel"


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _draw_lightning_bolt(
    ctx: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rng: np.random.Generator,
    color_rgb: Tuple[float, float, float] = (0.35, 0.92, 1.0),
    core_width: float = 3.2,
    jitter: float = 28.0,
    segments: int = 8,
    branch: bool = True,
) -> None:
    """Draw a multi-segment crackling plasma lightning arc with optional fork."""
    dx, dy = x1 - x0, y1 - y0
    pts = [(x0, y0)]
    for i in range(1, segments):
        t = i / float(segments)
        env = math.sin(math.pi * t)
        pts.append((
            x0 + dx * t + float(rng.uniform(-jitter, jitter)) * env,
            y0 + dy * t + float(rng.uniform(-jitter, jitter)) * env,
        ))
    pts.append((x1, y1))

    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.set_source_rgba(color_rgb[0], color_rgb[1], color_rgb[2], 0.44)
    ctx.set_line_width(core_width * 4.0)
    ctx.stroke_preserve()
    ctx.set_source_rgba(0.96, 0.99, 1.0, 0.96)
    ctx.set_line_width(core_width)
    ctx.stroke()

    if branch and len(pts) >= 4:
        mid = pts[len(pts) // 2]
        _draw_lightning_bolt(
            ctx,
            mid[0],
            mid[1],
            mid[0] + dx * 0.35 + float(rng.uniform(-55, 55)),
            mid[1] + dy * 0.35 + float(rng.uniform(-55, 55)),
            rng,
            color_rgb=color_rgb,
            core_width=core_width * 0.62,
            jitter=jitter * 0.6,
            segments=4,
            branch=False,
        )


def _draw_sculpted_limb(
    ctx: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    w0: float,
    w1: float,
    fill_rgb: Tuple[float, float, float],
    rim_rgb: Tuple[float, float, float],
    ink_rgb: Tuple[float, float, float] = (0.03, 0.03, 0.07),
) -> None:
    """Draw a smooth organic tapered limb capsule with rounded joint caps, ink outline & rim highlight."""
    dx, dy = x1 - x0, y1 - y0
    length = max(1e-3, math.hypot(dx, dy))
    nx, ny = -dy / length, dx / length

    r0, r1 = w0 * 0.5, w1 * 0.5
    ang = math.atan2(dy, dx)

    ctx.new_sub_path()
    ctx.arc(x0, y0, r0, ang + math.pi * 0.5, ang + math.pi * 1.5)
    ctx.curve_to(
        x0 + dx * 0.45 - nx * r0 * 1.08,
        y0 + dy * 0.45 - ny * r0 * 1.08,
        x1 - dx * 0.25 - nx * r1 * 1.05,
        y1 - dy * 0.25 - ny * r1 * 1.05,
        x1 - nx * r1,
        y1 - ny * r1,
    )
    ctx.arc(x1, y1, r1, ang - math.pi * 0.5, ang + math.pi * 0.5)
    ctx.curve_to(
        x1 - dx * 0.25 + nx * r1 * 1.05,
        y1 - dy * 0.25 + ny * r1 * 1.05,
        x0 + dx * 0.45 + nx * r0 * 1.08,
        y0 + dy * 0.45 + ny * r0 * 1.08,
        x0 + nx * r0,
        y0 + ny * r0,
    )
    ctx.close_path()

    ctx.set_source_rgb(*fill_rgb)
    ctx.fill_preserve()
    ctx.set_source_rgb(*ink_rgb)
    ctx.set_line_width(max(2.2, w0 * 0.13))
    ctx.stroke()

    # Cel specular rim arc along leading silhouette edge
    ctx.move_to(x0 + nx * (r0 * 0.88), y0 + ny * (r0 * 0.88))
    ctx.curve_to(
        x0 + dx * 0.45 + nx * (r0 * 0.96),
        y0 + dy * 0.45 + ny * (r0 * 0.96),
        x1 - dx * 0.25 + nx * (r1 * 0.94),
        y1 - dy * 0.25 + ny * (r1 * 0.94),
        x1 + nx * (r1 * 0.88),
        y1 + ny * (r1 * 0.88),
    )
    ctx.set_source_rgba(rim_rgb[0], rim_rgb[1], rim_rgb[2], 0.78)
    ctx.set_line_width(max(1.8, w1 * 0.20))
    ctx.stroke()


def _draw_magic_ring_3d(
    ctx: Any,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle: float,
    color_rgb: Tuple[float, float, float],
    alpha: float = 0.85,
) -> None:
    """Draw a tilted 3D-perspective glowing anime magic/cyber seal circle."""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(1.0, ry / max(1.0, rx))

    for r_scale, lw, a_mul in ((1.0, 4.0, 0.9), (0.86, 2.2, 0.65), (0.62, 2.8, 0.75)):
        r = rx * r_scale
        ctx.arc(0, 0, r, 0, 2.0 * math.pi)
        ctx.set_source_rgba(color_rgb[0], color_rgb[1], color_rgb[2], alpha * a_mul * 0.35)
        ctx.set_line_width(lw * 3.0)
        ctx.stroke_preserve()
        ctx.set_source_rgba(1.0, 0.98, 0.94, alpha * a_mul)
        ctx.set_line_width(lw)
        ctx.stroke()

    for poly_sides, rot_dir, r_ratio in ((6, 1.0, 0.86), (4, -1.35, 0.62)):
        rr = rx * r_ratio
        ctx.move_to(rr * math.cos(angle * rot_dir), rr * math.sin(angle * rot_dir))
        for k in range(1, poly_sides + 1):
            a = angle * rot_dir + (2.0 * math.pi * k) / poly_sides
            ctx.line_to(rr * math.cos(a), rr * math.sin(a))
        ctx.close_path()
        ctx.set_source_rgba(color_rgb[0], color_rgb[1], color_rgb[2], alpha * 0.78)
        ctx.set_line_width(2.2)
        ctx.stroke()

    for k in range(12):
        a = angle * 0.8 + (2.0 * math.pi * k) / 12.0
        ctx.move_to(rx * 0.88 * math.cos(a), rx * 0.88 * math.sin(a))
        ctx.line_to(rx * 0.98 * math.cos(a), rx * 0.98 * math.sin(a))
    ctx.set_source_rgba(1.0, 0.95, 0.85, alpha * 0.7)
    ctx.set_line_width(2.0)
    ctx.stroke()
    ctx.restore()


class AnimeSceneRenderer:
    """Renders fluid 1080p cel-shaded vector animation frames with Extend Clip continuity."""

    def __init__(self, width: int = 1920, height: int = 1080, fps: float = 24.0):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.scale = self.width / 1920.0

    def render_sequence(
        self,
        prompt: str,
        total_frames: int,
        seq_idx: int,
        total_chunks: int,
        chunk_id: str,
        duration_sec: float,
        model_name: str,
        is_chapter_start: bool = False,
        is_chapter_end: bool = False,
        chapter_title: Optional[str] = None,
        chapter_subtitle: Optional[str] = None,
        prev_video_code: Optional[Dict[str, Any]] = None,
        gamma_lut: Optional[np.ndarray] = None,
        on_frame: Optional[Callable[[int, np.ndarray], None]] = None,
    ) -> Dict[str, Any]:
        """Generate all frames cleanly, streaming BGR images to `on_frame` and returning handoff state."""
        if not HAS_CAIRO:
            return self._render_cv2_fallback(
                prompt=prompt,
                total_frames=total_frames,
                seq_idx=seq_idx,
                duration_sec=duration_sec,
                prev_video_code=prev_video_code,
                gamma_lut=gamma_lut,
                on_frame=on_frame,
            )

        w, h = self.width, self.height
        s = self.scale
        arch = classify_animation_scene(prompt)
        beat = seq_idx % 4

        prev_motion = (prev_video_code or {}).get("motion_and_physics", {})
        prev_cam = (prev_video_code or {}).get("camera_kinematics", {})
        init_stride = float(prev_motion.get("stride_phase_rad", 0.0))
        init_breath = float(prev_motion.get("breath_phase_rad", 0.0))
        init_t = float(prev_motion.get("flare_phase_t", seq_idx * duration_sec))
        init_scroll = float(prev_cam.get("end_center_x", 0.0))

        rng = np.random.default_rng((abs(hash(chunk_id)) + seq_idx * 131) % (2**31 - 1))

        num_stars = 200
        star_x = rng.uniform(0, w, num_stars).astype(np.float32)
        star_y = rng.uniform(0, h * 0.72, num_stars).astype(np.float32)
        star_depth = rng.uniform(0.2, 1.0, num_stars).astype(np.float32)
        star_phase = rng.uniform(0, 6.28, num_stars).astype(np.float32)

        num_parts = 75
        part_x = rng.uniform(0, w, num_parts).astype(np.float32)
        part_y = rng.uniform(0, h, num_parts).astype(np.float32)
        part_vx = rng.uniform(-16.0, -4.0, num_parts).astype(np.float32) * s
        part_vy = rng.uniform(-3.5, 4.5, num_parts).astype(np.float32) * s
        part_r = rng.uniform(2.0, 5.5, num_parts).astype(np.float32) * s

        g_n = 580
        g_r = (rng.power(0.7, g_n) * min(w, h) * 0.45).astype(np.float32)
        g_arm = rng.integers(0, 3, g_n)
        g_theta0 = (g_arm * (2.0 * math.pi / 3.0) + (g_r / (140.0 * s)) + rng.normal(0, 0.28, g_n)).astype(np.float32)
        g_hue = rng.uniform(0.0, 1.0, g_n).astype(np.float32)
        g_sz = rng.uniform(1.5, 4.5, g_n).astype(np.float32) * s

        slate_roi = None
        slate_rect = None
        bar_top = int(round(h * 0.052))
        bar_bot = int(round(h * 0.056))
        if is_chapter_start and chapter_title:
            sl_w = min(int(w * 0.62), 1180)
            sl_h = max(72, min(140, int(round(108 * s))))
            sl_x = int(w * 0.045)
            sl_y = h - bar_bot - sl_h - int(28 * s)
            slate_img = np.zeros((sl_h, sl_w, 3), dtype=np.uint8)
            slate_img[:] = (18, 16, 22)
            cv2.rectangle(slate_img, (0, 0), (8, sl_h), (70, 200, 255), -1)
            cv2.putText(slate_img, chapter_title, (24, int(sl_h * 0.44)), cv2.FONT_HERSHEY_SIMPLEX, max(0.65, 0.85 * s), (245, 250, 255), 2, cv2.LINE_AA)
            if chapter_subtitle:
                cv2.putText(slate_img, chapter_subtitle, (24, int(sl_h * 0.80)), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, 0.58 * s), (180, 225, 245), 1, cv2.LINE_AA)
            slate_roi = slate_img
            slate_rect = (sl_x, sl_y, sl_w, sl_h)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        lw_glow = max(64, w // 4)
        lh_glow = max(36, h // 4)

        end_stride = init_stride
        end_breath = init_breath
        end_t = init_t
        end_scroll = init_scroll

        for idx in range(total_frames):
            u = idx / max(1, total_frames - 1)
            dt = idx / max(1.0, self.fps)
            t_glob = init_t + dt
            scroll_x = init_scroll + dt * (290.0 if arch in ("anime_duel", "neon_hypercar") else 170.0) * s
            stride = init_stride + dt * (11.5 if arch == "anime_duel" else 6.8)
            breath = init_breath + dt * 3.6
            end_stride = stride
            end_breath = breath
            end_t = t_glob
            end_scroll = scroll_x

            cam_shake_x = 0.0
            cam_shake_y = 0.0
            if arch == "anime_duel" and beat in (1, 2) and (0.35 < (u % 0.5) < 0.48):
                cam_shake_x = math.sin(t_glob * 95.0) * 12.0 * s
                cam_shake_y = math.cos(t_glob * 82.0) * 8.0 * s

            ctx.save()
            ctx.translate(cam_shake_x, cam_shake_y)

            if arch == "anime_duel":
                self._draw_anime_duel_scene(ctx, w, h, s, idx, u, dt, t_glob, beat, scroll_x, stride, breath, star_x, star_y, star_depth, star_phase, part_x, part_y, part_vx, part_vy, part_r, rng)
            elif arch == "celestial_dragon":
                self._draw_celestial_dragon_scene(ctx, w, h, s, idx, u, dt, t_glob, beat, scroll_x, stride, breath, star_x, star_y, star_depth, star_phase, part_x, part_y, part_vx, part_vy, part_r, rng)
            elif arch == "cosmos_odyssey":
                self._draw_cosmos_odyssey_scene(ctx, w, h, s, idx, u, dt, t_glob, beat, scroll_x, stride, breath, star_x, star_y, star_depth, star_phase, g_r, g_theta0, g_hue, g_sz, rng)
            else:
                self._draw_neon_hypercar_scene(ctx, w, h, s, idx, u, dt, t_glob, beat, scroll_x, stride, breath, star_x, star_y, star_depth, star_phase, part_x, part_y, part_vx, part_vy, part_r, rng)

            ctx.restore()
            surface.flush()

            buf = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape((h, w, 4))
            frame = np.ascontiguousarray(buf[:, :, :3])

            small = cv2.resize(frame, (lw_glow, lh_glow), interpolation=cv2.INTER_AREA)
            bright = cv2.subtract(small, np.array([150, 150, 150], dtype=np.uint8))
            bloom_s = cv2.GaussianBlur(bright, (21, 21), 0)
            bloom_full = cv2.resize(bloom_s, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(frame, 1.0, bloom_full, 0.64, 0.0, dst=frame)

            if gamma_lut is not None:
                cv2.LUT(frame, gamma_lut, dst=frame)

            if slate_roi is not None and slate_rect is not None and 6 <= idx <= min(total_frames - 12, 96):
                sx, sy, sw, sh = slate_rect
                alpha = ((idx - 6) / 12.0 * 0.80) if idx < 18 else (max(0.0, (96 - idx) / 14.0) * 0.80 if idx > 82 else 0.80)
                if alpha > 0.01:
                    roi_dst = frame[sy : sy + sh, sx : sx + sw]
                    cv2.addWeighted(slate_roi, alpha, roi_dst, 1.0 - alpha, 0.0, dst=roi_dst)

            if total_chunks >= 8:
                if is_chapter_start and idx < 8:
                    cv2.convertScaleAbs(frame, dst=frame, alpha=(idx + 1) / 8.0)
                elif is_chapter_end and idx >= total_frames - 8:
                    cv2.convertScaleAbs(frame, dst=frame, alpha=max(0.08, (total_frames - idx) / 8.0))

            frame[:bar_top, :] = 0
            frame[-bar_bot:, :] = 0

            if on_frame is not None:
                on_frame(idx, frame)

        return {
            "kinetic_theme": arch,
            "end_zoom": 1.04,
            "end_center_x": round(end_scroll % 192000.0, 2),
            "end_center_y": round(h * 0.5, 2),
            "pan_vx": round(240.0 * s, 2),
            "pan_vy": 0.0,
            "stride_phase_rad": round(end_stride % (2.0 * math.pi), 4),
            "breath_phase_rad": round(end_breath % (2.0 * math.pi), 4),
            "flare_phase_t": round(end_t, 3),
        }

    # ------------------------------------------------------------------
    # SCENE 1: CYBER-RONIN / SAMURAI / ACTION ANIME DUEL
    # ------------------------------------------------------------------
    def _draw_anime_duel_scene(
        self,
        ctx: Any,
        w: int,
        h: int,
        s: float,
        idx: int,
        u: float,
        dt: float,
        t_glob: float,
        beat: int,
        scroll_x: float,
        stride: float,
        breath: float,
        star_x: np.ndarray,
        star_y: np.ndarray,
        star_depth: np.ndarray,
        star_phase: np.ndarray,
        part_x: np.ndarray,
        part_y: np.ndarray,
        part_vx: np.ndarray,
        part_vy: np.ndarray,
        part_r: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        # 1. Atmospheric dusk-crimson / neon-indigo anime sky dome
        sky = cairo.LinearGradient(0, 0, 0, h * 0.85)
        if beat == 2:
            sky.add_color_stop_rgb(0.0, 0.04, 0.02, 0.13)
            sky.add_color_stop_rgb(0.50, 0.24, 0.06, 0.22)
            sky.add_color_stop_rgb(1.0, 0.45, 0.12, 0.25)
        else:
            sky.add_color_stop_rgb(0.0, 0.02, 0.03, 0.11)
            sky.add_color_stop_rgb(0.55, 0.10, 0.07, 0.26)
            sky.add_color_stop_rgb(1.0, 0.16, 0.26, 0.45)
        ctx.set_source(sky)
        ctx.rectangle(0, 0, w, h)
        ctx.fill()

        # 2. Glowing Anime Full Moon + soft cloud bands
        moon_x = w * 0.72 - (scroll_x * 0.015) % (w * 0.20)
        moon_y = h * 0.25
        moon_r = 128.0 * s
        halo = cairo.RadialGradient(moon_x, moon_y, moon_r * 0.65, moon_x, moon_y, moon_r * 3.2)
        if beat == 2:
            halo.add_color_stop_rgba(0.0, 1.0, 0.36, 0.44, 0.72)
            halo.add_color_stop_rgba(1.0, 0.5, 0.08, 0.25, 0.0)
        else:
            halo.add_color_stop_rgba(0.0, 0.65, 0.92, 1.0, 0.62)
            halo.add_color_stop_rgba(1.0, 0.15, 0.35, 0.85, 0.0)
        ctx.set_source(halo)
        ctx.arc(moon_x, moon_y, moon_r * 3.2, 0, 2.0 * math.pi)
        ctx.fill()

        ctx.set_source_rgb(0.98, 0.96, 0.94 if beat == 2 else 1.0)
        ctx.arc(moon_x, moon_y, moon_r, 0, 2.0 * math.pi)
        ctx.fill()

        tw = 0.45 + 0.55 * np.sin(t_glob * 3.5 + star_phase)
        for i in range(90):
            sx = float((star_x[i] - scroll_x * 0.04 * star_depth[i]) % w)
            sy = float(star_y[i])
            ctx.set_source_rgba(0.84, 0.95, 1.0, float(tw[i]) * 0.78)
            ctx.arc(sx, sy, float(1.4 + star_depth[i] * 1.4) * s, 0, 6.28)
            ctx.fill()

        # Soft stylized billowing clouds drifting across sky
        for ci in range(4):
            cx = (w * (0.18 + 0.28 * ci) - scroll_x * (0.10 + 0.04 * ci)) % (w + 680 * s) - 340 * s
            cy = h * (0.17 + 0.07 * ci)
            cgrad = cairo.RadialGradient(cx, cy, 20 * s, cx, cy, 340 * s)
            cgrad.add_color_stop_rgba(0.0, 0.12, 0.10, 0.22, 0.55)
            cgrad.add_color_stop_rgba(1.0, 0.08, 0.07, 0.16, 0.0)
            ctx.save()
            ctx.translate(cx, cy)
            ctx.scale(2.8, 0.45)
            ctx.set_source(cgrad)
            ctx.arc(0, 0, 180.0 * s, 0, 6.28)
            ctx.fill()
            ctx.restore()

        # 3. Sweeping Volumetric Searchlights
        for li in range(3):
            lx_base = (w * (0.18 + 0.34 * li) - scroll_x * 0.25) % (w + 400 * s) - 200 * s
            sway = math.sin(t_glob * (1.2 + 0.3 * li) + li * 2.0) * 240.0 * s
            cone = cairo.LinearGradient(lx_base, h * 0.78, lx_base + sway, h * 0.05)
            cone.add_color_stop_rgba(0.0, 0.15, 0.85, 1.0, 0.28)
            cone.add_color_stop_rgba(1.0, 0.15, 0.85, 1.0, 0.0)
            ctx.set_source(cone)
            ctx.move_to(lx_base - 14 * s, h * 0.78)
            ctx.line_to(lx_base + 14 * s, h * 0.78)
            ctx.line_to(lx_base + sway + 170 * s, 0)
            ctx.line_to(lx_base + sway - 170 * s, 0)
            ctx.close_path()
            ctx.fill()

        # 4. Midground Cyber-Tokyo Skyline with Illuminated Windows & Pagoda Roof Eaves
        horiz_y = h * 0.76
        b_step = 135.0 * s
        num_b = int(w / b_step) + 3
        b_start = int(scroll_x * 0.48 // b_step)
        b_offset = (scroll_x * 0.48) % b_step
        for bi in range(num_b):
            seed_id = b_start + bi
            bx = bi * b_step - b_offset
            bh = (155.0 + ((seed_id * 73) % 230)) * s
            bw = b_step * 0.84
            by = horiz_y - bh
            ctx.set_source_rgb(0.06, 0.07, 0.14)
            ctx.rectangle(bx, by, bw, bh + 12 * s)
            ctx.fill()

            # Traditional curved Shinto pagoda roof accent every 4th building
            if seed_id % 4 == 1:
                ctx.set_source_rgb(0.10, 0.08, 0.16)
                ctx.move_to(bx - 18 * s, by + 6 * s)
                ctx.curve_to(bx + bw * 0.25, by - 26 * s, bx + bw * 0.75, by - 26 * s, bx + bw + 18 * s, by + 6 * s)
                ctx.close_path()
                ctx.fill()

            neon_c = (0.05, 0.92, 1.0) if (seed_id % 3 != 0) else (1.0, 0.24, 0.56)
            ctx.set_source_rgba(neon_c[0], neon_c[1], neon_c[2], 0.88)
            ctx.set_line_width(3.0 * s)
            ctx.move_to(bx + 2 * s, by)
            ctx.line_to(bx + bw - 2 * s, by)
            ctx.stroke()

            # Lit office windows (sparse flickering matrix)
            ctx.set_source_rgba(neon_c[0], neon_c[1], neon_c[2], 0.24)
            rows = int(bh // (28 * s))
            for r_i in range(1, min(rows, 8)):
                for c_i in range(3):
                    if (seed_id * 17 + r_i * 7 + c_i * 3) % 5 != 0:
                        ctx.rectangle(bx + (14 + c_i * 32) * s, by + r_i * 26 * s, 16 * s, 10 * s)
            ctx.fill()

        # 5. Wet Rain-Slicked Rooftop Surface & Mirrored Neon Puddles
        ground_y = h * 0.755
        g_grad = cairo.LinearGradient(0, ground_y, 0, h)
        g_grad.add_color_stop_rgb(0.0, 0.05, 0.07, 0.16)
        g_grad.add_color_stop_rgb(1.0, 0.02, 0.03, 0.09)
        ctx.set_source(g_grad)
        ctx.rectangle(0, ground_y, w, h - ground_y)
        ctx.fill()

        post_sp = 220.0 * s
        p_off = scroll_x % post_sp
        ctx.set_source_rgba(0.18, 0.85, 1.0, 0.65)
        ctx.set_line_width(4.0 * s)
        ctx.move_to(0, ground_y)
        ctx.line_to(w, ground_y)
        ctx.stroke()
        for pi in range(int(w / post_sp) + 2):
            px = pi * post_sp - p_off
            ctx.set_source_rgba(0.14, 0.18, 0.32, 0.92)
            ctx.rectangle(px, ground_y - 26 * s, 14 * s, 32 * s)
            ctx.fill()
            ctx.set_source_rgba(1.0, 0.28, 0.55, 0.92)
            ctx.arc(px + 7 * s, ground_y - 26 * s, 5.5 * s, 0, 6.28)
            ctx.fill()
            # Neon floor puddle smear
            ctx.set_source_rgba(0.15, 0.88, 1.0, 0.14)
            ctx.rectangle(px - 22 * s, ground_y + 22 * s, 58 * s, 5.0 * s)
            ctx.fill()

        # 6. Hero Trajectory + Arcane Surge Seal
        hero_x = w * (0.35 if beat in (0, 2) else 0.42) + math.sin(u * math.pi) * 85.0 * s
        jump_lift = max(0.0, math.sin(stride * 0.5)) * (135.0 * s if beat == 2 else 36.0 * s)
        hero_y = ground_y - 14.0 * s - jump_lift

        if beat in (1, 2):
            _draw_magic_ring_3d(
                ctx,
                hero_x + 40 * s,
                ground_y + 24 * s,
                215.0 * s,
                64.0 * s,
                t_glob * 2.5,
                (0.15, 0.92, 1.0) if beat == 1 else (1.0, 0.35, 0.62),
                alpha=0.76,
            )

        # 7. Articulated Cyber-Oni Adversary Drone
        drone_x = w * 0.77 + math.sin(t_glob * 2.2) * 32.0 * s
        drone_y = h * 0.47 + math.cos(t_glob * 3.1) * 44.0 * s
        if beat != 3 or u < 0.72:
            self._draw_cyber_drone(ctx, drone_x, drone_y, s, t_glob, beat, hit=(beat == 2 and abs(math.sin(t_glob * 6.0)) > 0.65), rng=rng)

        # Ground Cast Shadow
        shadow_r = max(48.0, 124.0 - jump_lift * 0.4) * s
        ctx.save()
        ctx.translate(hero_x, ground_y + 10 * s)
        ctx.scale(1.0, 0.22)
        ctx.set_source_rgba(0.0, 0.0, 0.02, 0.65)
        ctx.arc(0, 0, shadow_r, 0, 6.28)
        ctx.fill()
        ctx.restore()

        # 8. Sculpted Articulated Cyber-Ronin + Flowing Haori + Glowing Katana Wave
        self._draw_articulated_anime_warrior(ctx, hero_x, hero_y, s, t_glob, u, beat, stride, breath, rng)

        # 9. Rain & Burning Sparks
        rain_col = (0.72, 0.95, 1.0, 0.45) if beat != 2 else (1.0, 0.72, 0.45, 0.58)
        ctx.set_source_rgba(*rain_col)
        for i in range(len(part_x)):
            px = float((part_x[i] + part_vx[i] * idx * 1.8) % w)
            py = float((part_y[i] + (18.0 * s if beat != 2 else -5.0 * s) * idx + part_vy[i] * idx) % h)
            ctx.set_line_width(float(part_r[i] * 0.55))
            ctx.move_to(px, py)
            ctx.line_to(px - 22.0 * s, py + (38.0 * s if beat != 2 else -14.0 * s))
            ctx.stroke()

        # 10. Anime Speedlines on Action Beats
        if beat in (1, 2) or (beat == 3 and u < 0.65):
            cx, cy = hero_x + 90 * s, hero_y - 120 * s
            for k in range(32):
                ang = (2.0 * math.pi * k) / 32.0 + 0.12 * math.sin(t_glob * 18.0 + k)
                r0 = (560.0 + ((k * 97 + idx * 19) % 240)) * s
                r1 = 1280.0 * s
                ctx.set_source_rgba(0.7, 0.96, 1.0, 0.25)
                ctx.set_line_width((2.0 + (k % 3)) * s)
                ctx.move_to(cx + math.cos(ang) * r0, cy + math.sin(ang) * r0)
                ctx.line_to(cx + math.cos(ang) * r1, cy + math.sin(ang) * r1)
                ctx.stroke()

        if beat == 3 and 0.45 < u < 0.72:
            flash_k = math.sin(math.pi * (u - 0.45) / 0.27)
            ctx.set_source_rgba(1.0, 0.98, 0.92, 0.88 * flash_k)
            ctx.set_line_width(24.0 * s * flash_k)
            ctx.move_to(hero_x - 160 * s, hero_y + 120 * s)
            ctx.line_to(drone_x + 260 * s, hero_y - 240 * s)
            ctx.stroke()

    def _draw_articulated_anime_warrior(
        self,
        ctx: Any,
        hx: float,
        hy: float,
        s: float,
        t_glob: float,
        u: float,
        beat: int,
        stride: float,
        breath: float,
        rng: np.random.Generator,
    ) -> None:
        """Render sculpted anime hero rig with organic anatomy, haori cloak, blinking eyes & katana."""
        lean = 0.34 if beat == 0 else (0.52 if beat == 1 else (0.28 + 0.16 * math.sin(t_glob * 5.0)))
        bob = math.sin(stride * 2.0) * 9.0 * s
        hip_x = hx
        hip_y = hy - 146.0 * s + bob
        sh_x = hip_x + math.sin(lean) * 114.0 * s
        sh_y = hip_y - math.cos(lean) * 114.0 * s + math.sin(breath) * 3.0 * s
        head_x = sh_x + math.sin(lean * 0.7) * 40.0 * s
        head_y = sh_y - 48.0 * s

        suit_dark = (0.08, 0.09, 0.16)
        suit_mid = (0.16, 0.19, 0.32)
        rim_cyan = (0.25, 0.94, 1.0)
        accent_gold = (1.0, 0.72, 0.24)
        cape_red = (0.86, 0.15, 0.35)

        for leg_sign, col in ((-1.0, suit_dark), (1.0, suit_mid)):
            ph = stride + (0.0 if leg_sign > 0 else math.pi)
            thigh_a = lean + 0.56 * math.sin(ph)
            knee_x = hip_x + math.sin(thigh_a) * 78.0 * s
            knee_y = hip_y + math.cos(thigh_a) * 78.0 * s
            shin_a = thigh_a - (0.65 + 0.45 * max(0.0, -math.cos(ph)))
            foot_x = knee_x + math.sin(shin_a) * 76.0 * s
            foot_y = knee_y + math.cos(shin_a) * 76.0 * s
            _draw_sculpted_limb(ctx, hip_x, hip_y, knee_x, knee_y, 38 * s, 26 * s, col, rim_cyan)
            _draw_sculpted_limb(ctx, knee_x, knee_y, foot_x, foot_y, 26 * s, 18 * s, col, rim_cyan)
            # Sleek boot silhouette
            _draw_sculpted_limb(ctx, foot_x, foot_y, foot_x + 30 * s, foot_y + 4 * s, 18 * s, 10 * s, (0.12, 0.14, 0.22), accent_gold)

            if leg_sign < 0:
                # 6-node cubic-curved crimson scarf & flowing jacket coat tail
                cpts = [(sh_x - 8 * s, sh_y + 8 * s)]
                for ci in range(1, 7):
                    wave = math.sin(t_glob * 11.5 - ci * 0.72) * (18.0 + ci * 8.0) * s
                    cpts.append((sh_x - ci * 48.0 * s, sh_y + 18.0 * s + ci * 13.0 * s + wave))
                ctx.move_to(*cpts[0])
                for p in cpts[1:]:
                    ctx.line_to(*p)
                for p in reversed(cpts):
                    ctx.line_to(p[0] - 10 * s, p[1] + 48 * s)
                ctx.close_path()
                c_grad = cairo.LinearGradient(cpts[0][0], cpts[0][1], cpts[-1][0], cpts[-1][1])
                c_grad.add_color_stop_rgb(0.0, *cape_red)
                c_grad.add_color_stop_rgb(1.0, 0.28, 0.04, 0.16)
                ctx.set_source(c_grad)
                ctx.fill_preserve()
                ctx.set_source_rgba(1.0, 0.45, 0.60, 0.75)
                ctx.set_line_width(2.5 * s)
                ctx.stroke()

                # Sculpted Armored Jacket Torso + Flared Haori Waist Skirt
                _draw_sculpted_limb(ctx, hip_x, hip_y, sh_x, sh_y, 40 * s, 50 * s, (0.14, 0.16, 0.28), rim_cyan)
                # Shoulder Pauldron armor
                _draw_sculpted_limb(ctx, sh_x - 14 * s, sh_y - 6 * s, sh_x + 18 * s, sh_y + 8 * s, 30 * s, 20 * s, (0.20, 0.24, 0.42), accent_gold)
                # Chest neon core
                ctx.set_source_rgba(0.25, 0.95, 1.0, 0.92)
                ctx.arc((hip_x + sh_x) * 0.50 + 6 * s, (hip_y + sh_y) * 0.50, 8.5 * s, 0, 6.28)
                ctx.fill()

        # Off-hand balance arm
        l_ang = -lean - 0.8 + 0.35 * math.sin(stride + math.pi)
        l_elb_x = sh_x + math.sin(l_ang) * 44 * s
        l_elb_y = sh_y + math.cos(l_ang) * 44 * s
        l_hand_x = l_elb_x + math.sin(l_ang + 0.4) * 40 * s
        l_hand_y = l_elb_y + math.cos(l_ang + 0.4) * 40 * s
        _draw_sculpted_limb(ctx, sh_x, sh_y, l_elb_x, l_elb_y, 20 * s, 16 * s, suit_dark, rim_cyan)
        _draw_sculpted_limb(ctx, l_elb_x, l_elb_y, l_hand_x, l_hand_y, 16 * s, 13 * s, suit_dark, rim_cyan)

        # Sword Arm + Katana Choreography
        if beat == 0:
            sword_ang = -0.35 + 0.35 * math.sin(t_glob * 2.5)
        elif beat == 1:
            sword_ang = -1.1 + 2.6 * math.sin(t_glob * 5.2)
        elif beat == 2:
            sword_ang = -2.2 + 4.4 * ((u * 2.0) % 1.0)
        else:
            sword_ang = 0.85 - 2.1 * _clamp01((u - 0.3) * 2.5)

        elb_x = sh_x + math.cos(sword_ang * 0.6) * 52.0 * s
        elb_y = sh_y + math.sin(sword_ang * 0.6) * 52.0 * s + 12.0 * s
        hand_x = elb_x + math.cos(sword_ang) * 48.0 * s
        hand_y = elb_y + math.sin(sword_ang) * 48.0 * s
        _draw_sculpted_limb(ctx, sh_x, sh_y, elb_x, elb_y, 24 * s, 18 * s, suit_mid, rim_cyan)
        _draw_sculpted_limb(ctx, elb_x, elb_y, hand_x, hand_y, 18 * s, 15 * s, (0.85, 0.72, 0.66), rim_cyan)

        blade_len = 220.0 * s
        tip_x = hand_x + math.cos(sword_ang - 0.16) * blade_len
        tip_y = hand_y + math.sin(sword_ang - 0.16) * blade_len

        # Sweeping multi-stop crescent sword ribbon arc
        arc_span = 1.65 if beat in (1, 2) else 0.55
        if beat in (1, 2, 3):
            ctx.save()
            a_end = sword_ang - 0.16
            a_start = a_end - arc_span
            ctx.arc(hand_x, hand_y, blade_len * 1.02, min(a_start, a_end), max(a_start, a_end))
            ctx.set_source_rgba(0.15, 0.92, 1.0, 0.42)
            ctx.set_line_width(52.0 * s)
            ctx.stroke_preserve()
            ctx.set_source_rgba(0.95, 0.99, 1.0, 0.86)
            ctx.set_line_width(11.0 * s)
            ctx.stroke()
            ctx.restore()

        # Ornate gold Katana Tsuba & glowing curved blade
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(hand_x, hand_y)
        ctx.line_to(tip_x, tip_y)
        ctx.set_source_rgba(0.18, 0.94, 1.0, 0.60)
        ctx.set_line_width(15.0 * s)
        ctx.stroke_preserve()
        ctx.set_source_rgb(0.98, 0.99, 1.0)
        ctx.set_line_width(5.5 * s)
        ctx.stroke()

        ctx.set_source_rgb(*accent_gold)
        ctx.arc(hand_x, hand_y, 8.5 * s, 0, 6.28)
        ctx.fill()

        if beat in (1, 2):
            _draw_lightning_bolt(ctx, hand_x, hand_y, tip_x + 42 * s, tip_y - 28 * s, rng, (0.3, 0.95, 1.0), core_width=2.8 * s, jitter=24 * s, segments=6)

        # --- Sculpted Anime Profile Head, Jawline, Flowing Hair & Eye Motion Light Trail ---
        ctx.save()
        ctx.translate(head_x, head_y)
        ctx.rotate(lean * 0.22)

        # Sculpted anime head with pointed chin (not plain circle!)
        ctx.set_source_rgb(0.92, 0.78, 0.72)
        ctx.move_to(-18 * s, -22 * s)
        ctx.curve_to(22 * s, -30 * s, 28 * s, -4 * s, 24 * s, 14 * s)
        ctx.line_to(14 * s, 26 * s)
        ctx.line_to(-8 * s, 22 * s)
        ctx.line_to(-22 * s, 4 * s)
        ctx.close_path()
        ctx.fill()

        # Layered anime dark hair crown + windswept streaming hair tails
        ctx.set_source_rgb(0.11, 0.13, 0.24)
        for hi in range(5):
            w_hair = math.sin(t_glob * 12.0 - hi * 0.75) * 12.0 * s
            ctx.move_to(-10 * s, -16 * s + hi * 5 * s)
            ctx.curve_to(-55 * s, -26 * s + w_hair, -100 * s, -6 * s + w_hair * 1.35, -138 * s, 16 * s + w_hair * 1.55)
            ctx.line_to(-65 * s, 18 * s + hi * 5 * s)
            ctx.close_path()
            ctx.fill_preserve()
            ctx.set_source_rgba(0.32, 0.88, 1.0, 0.52)
            ctx.set_line_width(1.8 * s)
            ctx.stroke()
            ctx.set_source_rgb(0.11, 0.13, 0.24)

        # Anime glowing eye + comet-light energy trail on action beats!
        blink = 0.12 if ((t_glob % 3.1) > 2.96) else 1.0
        if beat in (1, 2):
            ctx.move_to(14 * s, -2 * s)
            ctx.curve_to(-28 * s, -12 * s, -72 * s, 4 * s, -110 * s, 12 * s)
            ctx.set_source_rgba(0.22, 0.96, 1.0, 0.72)
            ctx.set_line_width(5.5 * s)
            ctx.stroke()

        ctx.set_source_rgba(0.20, 0.96, 1.0, 0.98)
        ctx.save()
        ctx.translate(12 * s, -2 * s)
        ctx.scale(1.0, blink)
        ctx.move_to(-7 * s, 0)
        ctx.line_to(10 * s, -4.5 * s)
        ctx.line_to(8 * s, 4.5 * s)
        ctx.close_path()
        ctx.fill()
        ctx.restore()
        ctx.restore()

    def _draw_cyber_drone(
        self,
        ctx: Any,
        dx: float,
        dy: float,
        s: float,
        t_glob: float,
        beat: int,
        hit: bool,
        rng: np.random.Generator,
    ) -> None:
        """Render horned Cyber-Oni Sentinel Mecha with vernier wings & glowing crimson core."""
        ctx.save()
        ctx.translate(dx, dy)
        # Articulated hovering wing fins & Oni horns
        for side in (-1.0, 1.0):
            fl = math.sin(t_glob * 7.0) * 16.0 * s
            ctx.move_to(side * 28 * s, -10 * s)
            ctx.curve_to(side * 85 * s, -68 * s + fl, side * 125 * s, -42 * s + fl, side * 124 * s, 42 * s + fl * 0.5)
            ctx.close_path()
            ctx.set_source_rgb(0.14, 0.12, 0.22)
            ctx.fill_preserve()
            ctx.set_source_rgba(1.0, 0.26, 0.45, 0.92)
            ctx.set_line_width(3.0 * s)
            ctx.stroke()

            # Oni Crown Horns
            ctx.set_source_rgb(0.95, 0.76, 0.25)
            ctx.move_to(side * 16 * s, -42 * s)
            ctx.line_to(side * 42 * s, -94 * s)
            ctx.line_to(side * 28 * s, -36 * s)
            ctx.close_path()
            ctx.fill()

        ctx.set_source_rgb(0.08, 0.08, 0.14)
        ctx.arc(0, 0, 52.0 * s, 0, 6.28)
        ctx.fill_preserve()
        ctx.set_source_rgba(1.0, 0.28, 0.45, 0.88)
        ctx.set_line_width(4.0 * s)
        ctx.stroke()

        eye_scan = math.sin(t_glob * 5.0) * 18.0 * s
        ctx.set_source_rgba(1.0, 0.20, 0.35, 0.98)
        ctx.arc(eye_scan, -4 * s, 16.0 * s, 0, 6.28)
        ctx.fill()

        if hit:
            _draw_magic_ring_3d(ctx, -35 * s, 0, 95 * s, 115 * s, t_glob * 4.0, (1.0, 0.35, 0.48), alpha=0.92)
            _draw_lightning_bolt(ctx, -85 * s, -50 * s, 45 * s, 55 * s, rng, (1.0, 0.45, 0.35), 3.2 * s, 26 * s, 5)
        ctx.restore()

    # ------------------------------------------------------------------
    # SCENE 2: CELESTIAL DRAGON / ELVEN SPIRIT SANCTUARY
    # ------------------------------------------------------------------
    def _draw_celestial_dragon_scene(
        self,
        ctx: Any,
        w: int,
        h: int,
        s: float,
        idx: int,
        u: float,
        dt: float,
        t_glob: float,
        beat: int,
        scroll_x: float,
        stride: float,
        breath: float,
        star_x: np.ndarray,
        star_y: np.ndarray,
        star_depth: np.ndarray,
        star_phase: np.ndarray,
        part_x: np.ndarray,
        part_y: np.ndarray,
        part_vx: np.ndarray,
        part_vy: np.ndarray,
        part_r: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        sky = cairo.LinearGradient(0, 0, 0, h)
        sky.add_color_stop_rgb(0.0, 0.02, 0.05, 0.14)
        sky.add_color_stop_rgb(0.5, 0.05, 0.18, 0.28)
        sky.add_color_stop_rgb(1.0, 0.08, 0.30, 0.34)
        ctx.set_source(sky)
        ctx.rectangle(0, 0, w, h)
        ctx.fill()

        # Layered undulating Aurora curtains
        for ai in range(3):
            ctx.move_to(0, h * (0.20 + ai * 0.08))
            for x in range(0, w + 80, 80):
                ay = h * (0.22 + ai * 0.07) + math.sin(x * 0.004 + t_glob * 1.4 + ai * 1.6) * 58.0 * s
                ctx.line_to(x, ay)
            ctx.line_to(w, 0)
            ctx.line_to(0, 0)
            ctx.close_path()
            ctx.set_source_rgba(0.18 if ai % 2 == 0 else 0.45, 0.92, 0.72, 0.15)
            ctx.fill()

        # Sculpted Craggy Floating Islands with Waterfall Ribbons & Foam Plunge Clouds
        for ii in range(4):
            ix = (w * (0.15 + 0.28 * ii) - scroll_x * (0.35 + 0.12 * ii)) % (w + 480 * s) - 240 * s
            iy = h * (0.64 - 0.09 * (ii % 2)) + math.sin(t_glob * 1.8 + ii * 1.7) * 16.0 * s
            iw = (225.0 + ii * 38.0) * s

            # Jagged rock underside
            ctx.move_to(ix - iw * 0.52, iy + 6 * s)
            ctx.curve_to(ix - iw * 0.2, iy - 16 * s, ix + iw * 0.2, iy - 16 * s, ix + iw * 0.52, iy + 6 * s)
            ctx.line_to(ix + iw * 0.26, iy + iw * 0.55)
            ctx.line_to(ix + 10 * s, iy + iw * 0.72)
            ctx.line_to(ix - iw * 0.28, iy + iw * 0.48)
            ctx.close_path()
            ctx.set_source_rgb(0.06, 0.12, 0.18)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.25, 0.95, 0.78, 0.55)
            ctx.set_line_width(2.5 * s)
            ctx.stroke()

            # Cascading Waterfall + splash spray orbs
            wf_shift = (t_glob * 240.0 * s) % (40.0 * s)
            wfg = cairo.LinearGradient(ix, iy, ix, h)
            wfg.add_color_stop_rgba(0.0, 0.45, 0.98, 0.92, 0.62)
            wfg.add_color_stop_rgba(1.0, 0.25, 0.85, 0.92, 0.08)
            ctx.set_source(wfg)
            ctx.rectangle(ix - 14 * s, iy + 4 * s, 28 * s, h - iy)
            ctx.fill()
            for drop in range(6):
                dy_r = iy + ((drop * 45.0 * s + wf_shift) % (h * 0.35))
                ctx.set_source_rgba(0.88, 1.0, 0.98, 0.82)
                ctx.arc(ix + math.sin(drop + t_glob * 6.0) * 8 * s, dy_r, 4.5 * s, 0, 6.28)
                ctx.fill()

        drag_x = w * 0.55 + math.cos(t_glob * 1.1) * 120.0 * s
        drag_y = h * 0.44 + math.sin(t_glob * 1.8) * 65.0 * s
        _draw_magic_ring_3d(ctx, drag_x, drag_y, 265.0 * s, 235.0 * s, t_glob * 1.3, (0.35, 1.0, 0.78), alpha=0.72)

        seg_pts = []
        for si in range(11):
            px = drag_x - si * 48.0 * s
            py = drag_y + math.sin(t_glob * 5.2 - si * 0.58) * (32.0 + si * 3.5) * s
            seg_pts.append((px, py))

        # Dual Scalloped Feathered Wings
        wing_flap = math.sin(t_glob * 5.4)
        shoulder = seg_pts[2]
        for w_dir in (-1.0, 1.0):
            tip_x = shoulder[0] - 95.0 * s + w_dir * 45.0 * s
            tip_y = shoulder[1] - 265.0 * s * wing_flap - (125.0 * s if w_dir > 0 else 45.0 * s)
            mid_x = (shoulder[0] + tip_x) * 0.5 + 35 * s
            mid_y = (shoulder[1] + tip_y) * 0.5 - 55 * s
            ctx.move_to(*shoulder)
            ctx.curve_to(mid_x, mid_y, tip_x + 40 * s, tip_y + 20 * s, tip_x, tip_y)
            ctx.curve_to(tip_x - 85 * s, tip_y + 95 * s, shoulder[0] - 110 * s, shoulder[1] + 55 * s, *shoulder)
            ctx.close_path()
            wg = cairo.LinearGradient(shoulder[0], shoulder[1], tip_x, tip_y)
            wg.add_color_stop_rgba(0.0, 0.22, 0.95, 0.85, 0.85)
            wg.add_color_stop_rgba(1.0, 0.88, 0.98, 1.0, 0.38)
            ctx.set_source(wg)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.85, 1.0, 0.95, 0.92)
            ctx.set_line_width(3.0 * s)
            ctx.stroke()

        for si in range(len(seg_pts) - 1, 0, -1):
            r0 = max(9.0, 40.0 - si * 2.8) * s
            r1 = max(11.0, 40.0 - (si - 1) * 2.8) * s
            _draw_sculpted_limb(
                ctx,
                seg_pts[si][0],
                seg_pts[si][1],
                seg_pts[si - 1][0],
                seg_pts[si - 1][1],
                r0 * 1.4,
                r1 * 1.4,
                (0.09, 0.45, 0.52) if si % 2 == 0 else (0.14, 0.58, 0.62),
                (0.45, 1.0, 0.85),
            )

        hx, hy = seg_pts[0]
        ctx.save()
        ctx.translate(hx, hy)
        ctx.set_source_rgb(0.16, 0.68, 0.72)
        ctx.move_to(-28 * s, -30 * s)
        ctx.curve_to(25 * s, -34 * s, 68 * s, -22 * s, 82 * s, -6 * s)
        ctx.line_to(74 * s, 16 * s)
        ctx.line_to(-18 * s, 26 * s)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(0.6, 1.0, 0.92, 0.9)
        ctx.set_line_width(3.0 * s)
        ctx.stroke()

        # Twin golden dragon horns + flowing whisker/mane ribbon
        ctx.set_source_rgb(1.0, 0.84, 0.38)
        ctx.move_to(-8 * s, -22 * s)
        ctx.curve_to(-45 * s, -52 * s, -75 * s, -65 * s, -98 * s, -74 * s)
        ctx.line_to(-24 * s, -8 * s)
        ctx.close_path()
        ctx.fill()

        w_m = math.sin(t_glob * 8.0) * 18.0 * s
        ctx.set_source_rgba(0.65, 1.0, 0.92, 0.82)
        ctx.set_line_width(4.0 * s)
        ctx.move_to(52 * s, 6 * s)
        ctx.curve_to(-10 * s, 22 * s + w_m, -68 * s, 12 * s - w_m, -120 * s, 32 * s + w_m)
        ctx.stroke()

        ctx.set_source_rgb(1.0, 0.95, 0.45)
        ctx.arc(36 * s, -6 * s, 7.5 * s, 0, 6.28)
        ctx.fill()
        ctx.restore()

        if beat in (2, 3):
            b_grad = cairo.LinearGradient(hx + 70 * s, hy, w, hy + 40 * s)
            b_grad.add_color_stop_rgba(0.0, 0.92, 1.0, 0.95, 0.88)
            b_grad.add_color_stop_rgba(0.4, 0.25, 0.98, 0.85, 0.55)
            b_grad.add_color_stop_rgba(1.0, 0.2, 0.8, 1.0, 0.0)
            ctx.set_source(b_grad)
            pulse_bw = (32.0 + 14.0 * math.sin(t_glob * 16.0)) * s
            ctx.move_to(hx + 72 * s, hy - 8 * s)
            ctx.line_to(w, hy - pulse_bw * 2.2)
            ctx.line_to(w, hy + pulse_bw * 2.2)
            ctx.line_to(hx + 72 * s, hy + 8 * s)
            ctx.close_path()
            ctx.fill()

        for i in range(len(part_x)):
            fx = float((part_x[i] + math.sin(t_glob * 2.1 + i) * 65 * s + scroll_x * 0.2) % w)
            fy = float((part_y[i] + math.cos(t_glob * 1.7 + i * 0.4) * 55 * s) % h)
            ctx.set_source_rgba(0.55, 1.0, 0.82, 0.72)
            ctx.arc(fx, fy, float(part_r[i] * 0.95), 0, 6.28)
            ctx.fill()

    # ------------------------------------------------------------------
    # SCENE 3: ANIMATED 3D COSMOS ODYSSEY & BLACK HOLE / STARSHIP
    # ------------------------------------------------------------------
    def _draw_cosmos_odyssey_scene(
        self,
        ctx: Any,
        w: int,
        h: int,
        s: float,
        idx: int,
        u: float,
        dt: float,
        t_glob: float,
        beat: int,
        scroll_x: float,
        stride: float,
        breath: float,
        star_x: np.ndarray,
        star_y: np.ndarray,
        star_depth: np.ndarray,
        star_phase: np.ndarray,
        g_r: np.ndarray,
        g_theta0: np.ndarray,
        g_hue: np.ndarray,
        g_sz: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        sky = cairo.LinearGradient(0, 0, w, h)
        sky.add_color_stop_rgb(0.0, 0.01, 0.015, 0.06)
        sky.add_color_stop_rgb(0.5, 0.05, 0.03, 0.16)
        sky.add_color_stop_rgb(1.0, 0.02, 0.08, 0.18)
        ctx.set_source(sky)
        ctx.rectangle(0, 0, w, h)
        ctx.fill()

        # Layered glowing cosmic nebula gas clouds drifting behind galaxy
        for ni, ncol in enumerate(((0.18, 0.55, 0.95), (0.78, 0.22, 0.65), (0.15, 0.82, 0.78))):
            nx = w * (0.28 + 0.25 * ni) + math.sin(t_glob * 0.35 + ni * 2.1) * 90.0 * s
            ny = h * (0.35 + 0.14 * (ni % 2)) + math.cos(t_glob * 0.4 + ni) * 55.0 * s
            ng = cairo.RadialGradient(nx, ny, 25 * s, nx, ny, 380 * s)
            ng.add_color_stop_rgba(0.0, ncol[0], ncol[1], ncol[2], 0.28)
            ng.add_color_stop_rgba(1.0, ncol[0], ncol[1], ncol[2], 0.0)
            ctx.set_source(ng)
            ctx.arc(nx, ny, 380 * s, 0, 6.28)
            ctx.fill()

        gx, gy = w * 0.56, h * 0.46
        tilt = 0.46
        spin_rate = t_glob * 0.28
        angles = g_theta0 + spin_rate + (120.0 * s / np.maximum(g_r, 35.0 * s)) * 0.25
        px_arr = gx + g_r * np.cos(angles)
        py_arr = gy + g_r * np.sin(angles) * tilt

        core = cairo.RadialGradient(gx, gy, 8 * s, gx, gy, 290 * s)
        if beat == 2:
            core.add_color_stop_rgba(0.0, 1.0, 0.72, 0.28, 0.92)
            core.add_color_stop_rgba(0.45, 0.92, 0.32, 0.18, 0.48)
            core.add_color_stop_rgba(1.0, 0.2, 0.1, 0.4, 0.0)
        else:
            core.add_color_stop_rgba(0.0, 0.98, 0.95, 1.0, 0.88)
            core.add_color_stop_rgba(0.45, 0.28, 0.65, 1.0, 0.45)
            core.add_color_stop_rgba(1.0, 0.1, 0.1, 0.35, 0.0)
        ctx.set_source(core)
        ctx.save()
        ctx.translate(gx, gy)
        ctx.scale(1.3, tilt * 1.15)
        ctx.arc(0, 0, 265 * s, 0, 6.28)
        ctx.fill()
        ctx.restore()

        for i in range(len(g_r)):
            c_mix = float(g_hue[i])
            ctx.set_source_rgba(0.6 + 0.4 * c_mix, 0.78 + 0.18 * (1.0 - c_mix), 1.0 if c_mix < 0.7 else 0.72, 0.85)
            ctx.arc(float(px_arr[i]), float(py_arr[i]), float(g_sz[i]), 0, 6.28)
            ctx.fill()

        if beat == 2:
            bh_x, bh_y = w * 0.56, h * 0.48
            ctx.set_source_rgba(0.28, 0.78, 1.0, 0.32)
            for side in (-1.0, 1.0):
                ctx.move_to(bh_x - 22 * s, bh_y)
                ctx.line_to(bh_x + side * 60 * s, bh_y + side * 460 * s)
                ctx.line_to(bh_x + side * 140 * s, bh_y + side * 460 * s)
                ctx.line_to(bh_x + 22 * s, bh_y)
                ctx.close_path()
                ctx.fill()
            _draw_magic_ring_3d(ctx, bh_x, bh_y, 248 * s, 74 * s, t_glob * 3.8, (1.0, 0.68, 0.22), alpha=0.90)
            ctx.set_source_rgb(0.01, 0.01, 0.02)
            ctx.arc(bh_x, bh_y, 76.0 * s, 0, 6.28)
            ctx.fill_preserve()
            ctx.set_source_rgba(1.0, 0.85, 0.45, 0.95)
            ctx.set_line_width(4.5 * s)
            ctx.stroke()
        else:
            pl_x = w * 0.24 + math.sin(t_glob * 0.45) * 42.0 * s
            pl_y = h * 0.58 + math.cos(t_glob * 0.55) * 24.0 * s
            pl_r = 128.0 * s
            pg = cairo.RadialGradient(pl_x - pl_r * 0.38, pl_y - pl_r * 0.35, pl_r * 0.15, pl_x, pl_y, pl_r)
            pg.add_color_stop_rgb(0.0, 0.55, 0.92, 1.0)
            pg.add_color_stop_rgb(0.55, 0.14, 0.42, 0.75)
            pg.add_color_stop_rgb(1.0, 0.02, 0.04, 0.12)
            ctx.set_source(pg)
            ctx.arc(pl_x, pl_y, pl_r, 0, 6.28)
            ctx.fill()
            ctx.save()
            ctx.translate(pl_x, pl_y)
            ctx.rotate(-0.24)
            ctx.scale(1.0, 0.28)
            for rr, lw, aa in ((pl_r * 1.9, 14 * s, 0.45), (pl_r * 1.55, 8 * s, 0.65)):
                ctx.set_source_rgba(0.68, 0.94, 1.0, aa)
                ctx.set_line_width(lw)
                ctx.arc(0, 0, rr, 0, 6.28)
                ctx.stroke()
            ctx.restore()

        # Detailed Swept-Winged Anime Orbital Cruiser + Dual Plasma Ion Trails
        ship_x = w * 0.22 + (u * w * 0.52) if beat in (1, 3) else w * 0.36 + math.sin(t_glob * 1.4) * 90 * s
        ship_y = h * 0.36 - (u * h * 0.12) + math.sin(t_glob * 2.8) * 18 * s
        eg = cairo.LinearGradient(ship_x - 360 * s, ship_y, ship_x, ship_y)
        eg.add_color_stop_rgba(0.0, 0.1, 0.75, 1.0, 0.0)
        eg.add_color_stop_rgba(1.0, 0.75, 0.98, 1.0, 0.95)
        ctx.set_source(eg)
        ctx.set_line_width(14.0 * s)
        ctx.move_to(ship_x - 340 * s, ship_y + 18 * s)
        ctx.line_to(ship_x, ship_y)
        ctx.stroke()

        ctx.save()
        ctx.translate(ship_x, ship_y)
        ctx.set_source_rgb(0.84, 0.90, 0.98)
        ctx.move_to(92 * s, 0)
        ctx.line_to(12 * s, -16 * s)
        ctx.line_to(-46 * s, -48 * s)
        ctx.line_to(-22 * s, 0)
        ctx.line_to(-46 * s, 44 * s)
        ctx.line_to(12 * s, 14 * s)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(0.2, 0.9, 1.0, 0.85)
        ctx.set_line_width(2.5 * s)
        ctx.stroke()
        ctx.restore()

    # ------------------------------------------------------------------
    # SCENE 4: HIGH-SPEED CYBER-HYPERCAR NEON PURSUIT
    # ------------------------------------------------------------------
    def _draw_neon_hypercar_scene(
        self,
        ctx: Any,
        w: int,
        h: int,
        s: float,
        idx: int,
        u: float,
        dt: float,
        t_glob: float,
        beat: int,
        scroll_x: float,
        stride: float,
        breath: float,
        star_x: np.ndarray,
        star_y: np.ndarray,
        star_depth: np.ndarray,
        star_phase: np.ndarray,
        part_x: np.ndarray,
        part_y: np.ndarray,
        part_vx: np.ndarray,
        part_vy: np.ndarray,
        part_r: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        vy = h * 0.44
        vx = w * 0.50 + math.sin(t_glob * 1.3) * 85.0 * s
        sky = cairo.LinearGradient(0, 0, 0, vy)
        sky.add_color_stop_rgb(0.0, 0.03, 0.02, 0.10)
        sky.add_color_stop_rgb(1.0, 0.35, 0.08, 0.38)
        ctx.set_source(sky)
        ctx.rectangle(0, 0, w, vy)
        ctx.fill()

        # Synthwave Sun directly grounded on horizon + horizontal scanlines + flanking mountains
        ctx.save()
        ctx.set_source_rgba(1.0, 0.42, 0.25, 0.90)
        ctx.arc(vx, vy, 175 * s, math.pi, 2.0 * math.pi)
        ctx.fill()
        ctx.set_source_rgb(0.18, 0.04, 0.22)
        for band_y in (vy - 30 * s, vy - 62 * s, vy - 95 * s, vy - 125 * s):
            ctx.rectangle(vx - 190 * s, band_y, 380 * s, 9 * s)
        ctx.fill()
        ctx.restore()

        # Jagged Synthwave Horizon Mountains
        ctx.set_source_rgb(0.06, 0.03, 0.14)
        ctx.move_to(0, vy)
        for mx in range(0, w + 120, 120):
            mh = (42 + ((mx * 29) % 85)) * s if abs(mx - vx) > 190 * s else 12 * s
            ctx.line_to(mx, vy - mh)
        ctx.line_to(w, vy)
        ctx.close_path()
        ctx.fill()

        road = cairo.LinearGradient(0, vy, 0, h)
        road.add_color_stop_rgb(0.0, 0.05, 0.03, 0.12)
        road.add_color_stop_rgb(1.0, 0.02, 0.03, 0.08)
        ctx.set_source(road)
        ctx.rectangle(0, vy, w, h - vy)
        ctx.fill()

        for ri in range(12):
            z = ((ri / 12.0) - ((t_glob * 1.8) % (1.0 / 12.0))) % 1.0
            persp = max(0.04, z * z)
            ry = vy + persp * (h - vy)
            half_rw = persp * w * 0.85
            ctx.set_source_rgba(0.18, 0.92, 1.0, 0.3 + 0.55 * persp)
            ctx.set_line_width(max(1.5, 5.0 * persp * s))
            ctx.move_to(0, ry)
            ctx.line_to(w, ry)
            ctx.stroke()
            if ri % 2 == 0 and persp > 0.12:
                ctx.set_source_rgba(1.0, 0.22, 0.65, 0.62 * persp)
                ctx.set_line_width(6.0 * persp * s)
                ctx.move_to(vx - half_rw * 0.72, ry)
                ctx.line_to(vx - half_rw * 0.52, ry - 440 * persp * s)
                ctx.line_to(vx + half_rw * 0.52, ry - 440 * persp * s)
                ctx.line_to(vx + half_rw * 0.72, ry)
                ctx.stroke()

        for lx in (-0.75, -0.35, 0.0, 0.35, 0.75):
            ctx.set_source_rgba(0.25, 0.95, 1.0, 0.72)
            ctx.set_line_width(4.0 * s)
            ctx.move_to(vx, vy)
            ctx.line_to(vx + lx * w * 0.95, h)
            ctx.stroke()

        # Detailed Sculpted Widebody Cyber-Hypercar
        cx = w * 0.50 + math.sin(t_glob * 1.9) * 240.0 * s
        cy = h * 0.77 + math.sin(t_glob * 18.0) * 3.5 * s
        roll = math.cos(t_glob * 1.9) * 0.075

        ctx.save()
        ctx.translate(cx, cy)
        ctx.rotate(roll)

        # Widebody tires with neon rims
        for wx in (-165 * s, 165 * s):
            ctx.set_source_rgb(0.05, 0.05, 0.08)
            ctx.rectangle(wx - 32 * s, -44 * s, 64 * s, 96 * s)
            ctx.fill()
            ctx.set_source_rgba(0.15, 0.92, 1.0, 0.85)
            ctx.set_line_width(3.0 * s)
            ctx.arc(wx, 4 * s, 22 * s, 0, 6.28)
            ctx.stroke()

        # Lower aero diffuser & body chassis
        ctx.set_source_rgb(0.08, 0.10, 0.18)
        ctx.move_to(-215 * s, 28 * s)
        ctx.line_to(-195 * s, -36 * s)
        ctx.curve_to(-145 * s, -48 * s, -120 * s, -92 * s, -85 * s, -92 * s)
        ctx.line_to(85 * s, -92 * s)
        ctx.curve_to(120 * s, -92 * s, 145 * s, -48 * s, 195 * s, -36 * s)
        ctx.line_to(215 * s, 28 * s)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(0.28, 0.94, 1.0, 0.9)
        ctx.set_line_width(3.5 * s)
        ctx.stroke()

        # Glossy tinted coupe glass canopy
        ctx.set_source_rgb(0.03, 0.06, 0.14)
        ctx.move_to(-92 * s, -46 * s)
        ctx.line_to(-72 * s, -84 * s)
        ctx.line_to(72 * s, -84 * s)
        ctx.line_to(92 * s, -46 * s)
        ctx.close_path()
        ctx.fill()

        # Swan-neck GT Spoiler Wing + Pylons
        ctx.set_source_rgb(0.12, 0.12, 0.22)
        ctx.rectangle(-95 * s, -110 * s, 14 * s, 26 * s)
        ctx.rectangle(81 * s, -110 * s, 14 * s, 26 * s)
        ctx.rectangle(-218 * s, -118 * s, 436 * s, 15 * s)
        ctx.fill()

        # Continuous Neon LED Tail Light Blade + Nitro Flames
        ctx.set_source_rgb(1.0, 0.16, 0.36)
        ctx.rectangle(-182 * s, -34 * s, 364 * s, 12 * s)
        ctx.fill()
        flame_len = (46.0 + 38.0 * abs(math.sin(t_glob * 34.0))) * s
        for px in (-62 * s, 62 * s):
            fg = cairo.RadialGradient(px, 16 * s, 4 * s, px, 16 * s, flame_len)
            fg.add_color_stop_rgba(0.0, 0.95, 0.98, 1.0, 0.95)
            fg.add_color_stop_rgba(0.45, 0.18, 0.78, 1.0, 0.85)
            fg.add_color_stop_rgba(1.0, 1.0, 0.28, 0.12, 0.0)
            ctx.set_source(fg)
            ctx.arc(px, 16 * s, flame_len * 0.65, 0, 6.28)
            ctx.fill()
        ctx.restore()

    def _render_cv2_fallback(
        self,
        prompt: str,
        total_frames: int,
        seq_idx: int,
        duration_sec: float,
        prev_video_code: Optional[Dict[str, Any]],
        gamma_lut: Optional[np.ndarray],
        on_frame: Optional[Callable[[int, np.ndarray], None]],
    ) -> Dict[str, Any]:
        """Minimal OpenCV vector fallback only if PyCairo is unavailable."""
        w, h = self.width, self.height
        for idx in range(total_frames):
            t = idx / max(1.0, self.fps)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = (22, 14, 34)
            cx = int(w * (0.3 + 0.4 * math.sin(t * 1.4)))
            cy = int(h * 0.55)
            cv2.circle(frame, (cx, cy), int(h * 0.12), (255, 220, 90), -1, cv2.LINE_AA)
            if gamma_lut is not None:
                cv2.LUT(frame, gamma_lut, dst=frame)
            if on_frame is not None:
                on_frame(idx, frame)
        return {"kinetic_theme": classify_animation_scene(prompt), "end_zoom": 1.02}
