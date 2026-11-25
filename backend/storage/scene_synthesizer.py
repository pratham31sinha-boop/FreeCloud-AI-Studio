"""Independent High-Resolution 1080p Scene & Anchor Synthesizer.

Provides fast, self-contained 1080p scene anchor generation for every prompt
theme with:
1. Local bundled keyframe extraction + anime cel-shading (e.g. Jagdhatri).
2. Cached curated domain assets (instant on subsequent runs, 3s fail-fast online).
3. Zero-dependency offline crisp vector/shader 1080p artwork synthesizer
   (never falls back to featureless blurry gradients).
"""

from __future__ import annotations

import logging
import math
import re
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

THEME_IMAGE_CATALOG: Dict[str, str] = {
    "hypercar": "https://images.pexels.com/photos/28498801/pexels-photo-28498801.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "cyberpunk": "https://images.pexels.com/photos/3165335/pexels-photo-3165335.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "forest": "https://images.pexels.com/photos/1423600/pexels-photo-1423600.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "space": "https://images.pexels.com/photos/2150/sky-space-dark-galaxy.jpg?auto=compress&cs=tinysrgb&w=1920",
    "wildlife": "https://images.pexels.com/photos/145939/pexels-photo-145939.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "sunset": "https://images.pexels.com/photos/36717/amazing-animal-beautiful-beautifull.jpg?auto=compress&cs=tinysrgb&w=1920",
    "samurai": "https://images.pexels.com/photos/1440476/pexels-photo-1440476.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "martial": "https://images.pexels.com/photos/7045731/pexels-photo-7045731.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "train": "https://images.pexels.com/photos/1793503/pexels-photo-1793503.jpeg?auto=compress&cs=tinysrgb&w=1920",
}


def classify_prompt_theme(prompt: str) -> str:
    """Classify a prompt accurately using word boundaries to avoid substring false positives."""
    p = (prompt or "").lower()

    def has_word(*words: str) -> bool:
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        return bool(re.search(pattern, p))

    if has_word("jagdhatri", "jas", "jash", "swayambhu", "kolkata"):
        return "jagdhatri"
    if has_word("car", "supercar", "hypercar", "vehicle", "race", "racing", "highway", "drift", "driving"):
        return "hypercar"
    if has_word("cyberpunk", "neon", "tokyo", "cyborg", "hologram", "holograms", "hacker", "hoverbike", "futuristic"):
        return "cyberpunk"
    if has_word("elven", "forest", "bioluminescence", "bioluminescent", "cathedral", "woods", "enchanted", "spirit", "fairy", "wings"):
        return "forest"
    if has_word("samurai", "ronin", "katana", "shinobi", "ninja", "dojo", "bamboo", "blade", "swordswoman"):
        return "samurai"
    if has_word("astronaut", "space", "alien", "planet", "galaxy", "mars", "cosmos", "nebula", "orbital", "mecha", "colony"):
        return "space"
    if has_word("panther", "leopard", "tiger", "lion", "wolf", "jaguar", "predator", "feline"):
        return "wildlife"
    if has_word("train", "steam", "witch", "cottage", "ghibli"):
        return "train"
    if has_word("martial", "ambush", "operative", "fight", "arena", "warrior", "punch", "kick", "combat"):
        return "martial"
    if has_word("dragon", "fire", "canyon", "castle", "sunset", "shinkai", "sky", "comet", "lake", "cherry"):
        return "sunset"
    return "cyberpunk"


def _fast_anime_stylize(img: np.ndarray) -> np.ndarray:
    """Fast 1080p anime cel-shading stylization (<35ms)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    smooth_small = cv2.bilateralFilter(small, d=5, sigmaColor=50, sigmaSpace=50)
    smooth = cv2.resize(smooth_small, (w, h), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges_small = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 3), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5
    )
    edges = cv2.resize(edges_small, (w, h), interpolation=cv2.INTER_NEAREST)
    edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.28, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.06, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.bitwise_and(boosted, edge_mask)


def synthesize_crisp_scene_anchor(
    prompt: str,
    seed: int = 42,
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    """Generate a sharp, multi-layered 1080p scene illustration completely offline.

    Never returns a blurry empty gradient; always renders sky depth, geometric
    silhouettes, volumetric god-rays, and theme-appropriate focal visuals.
    """
    theme = classify_prompt_theme(prompt)
    rng = np.random.default_rng(abs(seed) % (2**31 - 1))

    # Theme color palettes (BGR bottom, BGR top, accent BGR)
    palettes: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]] = {
        "forest": ((45, 55, 12), (28, 18, 8), (220, 255, 110)),
        "cyberpunk": ((45, 12, 32), (18, 8, 22), (255, 220, 40)),
        "hypercar": ((25, 38, 72), (18, 12, 28), (80, 190, 255)),
        "space": ((30, 14, 12), (8, 4, 6), (255, 210, 140)),
        "samurai": ((24, 20, 58), (12, 10, 26), (90, 110, 255)),
        "wildlife": ((28, 42, 18), (12, 18, 10), (110, 235, 160)),
        "jagdhatri": ((35, 24, 52), (12, 12, 24), (240, 165, 60)),
        "martial": ((35, 24, 52), (12, 12, 24), (240, 165, 60)),
        "train": ((85, 60, 30), (190, 125, 65), (140, 220, 255)),
        "sunset": ((60, 45, 110), (110, 55, 35), (120, 210, 255)),
    }
    bot_c, top_c, acc_c = palettes.get(theme, palettes["cyberpunk"])

    # 1. Sky vertical gradient
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    sky = (np.array(top_c, dtype=np.float32) * (1.0 - t) + np.array(bot_c, dtype=np.float32) * t)
    canvas = np.broadcast_to(sky, (height, width, 3)).astype(np.uint8).copy()

    # 2. Celestial orb / moon / portal core / rising sun
    orb_x = int(width * (0.5 if theme in ("forest", "hypercar", "samurai") else 0.65))
    orb_y = int(height * 0.34)
    orb_r = int(height * 0.14)
    glow = np.zeros_like(canvas, dtype=np.float32)
    cv2.circle(glow, (orb_x, orb_y), orb_r * 2, acc_c, -1, lineType=cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (151, 151), 0)
    canvas = np.clip(canvas.astype(np.float32) + glow * 0.45, 0, 255).astype(np.uint8)
    cv2.circle(canvas, (orb_x, orb_y), orb_r, acc_c, -1, lineType=cv2.LINE_AA)

    # 3. Layered parallax horizon & silhouettes
    horizon_y = int(height * 0.68)
    if theme in ("cyberpunk", "jagdhatri", "martial"):
        # Cityscape / Warehouse catwalk towers
        for bx in range(0, width, int(width * 0.055)):
            bw = int(rng.integers(int(width * 0.03), int(width * 0.06)))
            bh = int(rng.integers(int(height * 0.18), int(height * 0.55)))
            top_y = horizon_y - bh
            col = (int(rng.integers(14, 28)), int(rng.integers(12, 24)), int(rng.integers(24, 42)))
            cv2.rectangle(canvas, (bx, top_y), (bx + bw, horizon_y), col, -1)
            cv2.rectangle(canvas, (bx, top_y), (bx + bw, horizon_y), acc_c, 1)
            # Lit neon windows
            for wy in range(top_y + 18, horizon_y - 12, 24):
                if rng.random() > 0.35:
                    cv2.line(canvas, (bx + 6, wy), (bx + bw - 6, wy), acc_c, 2)
    elif theme == "forest":
        # Ancient cathedral arches & towering tree trunks
        for tx in range(-60, width + 60, int(width * 0.09)):
            tw = int(rng.integers(40, 95))
            cv2.rectangle(canvas, (tx, 0), (tx + tw, height), (18, 28, 12), -1)
            cv2.ellipse(canvas, (tx + int(width * 0.045), int(height * 0.25)), (int(width * 0.05), int(height * 0.28)), 0, 180, 360, (24, 45, 22), 18, cv2.LINE_AA)
    else:
        # Mountain ridges
        pts = [(0, height), (0, horizon_y)]
        for mx in range(0, width + 120, 120):
            my = horizon_y - int(rng.integers(40, 220))
            pts.append((mx, my))
        pts.append((width, height))
        cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], (22, 20, 30))

    # 4. Crisp Ground / Perspective Floor Grid
    cv2.rectangle(canvas, (0, horizon_y), (width, height), (14, 14, 22), -1)
    for gx in range(-width, width * 2, int(width * 0.08)):
        cv2.line(canvas, (width // 2, horizon_y), (gx, height), tuple(int(c * 0.5) for c in acc_c), 1, cv2.LINE_AA)
    for gy_step in range(1, 7):
        y_line = horizon_y + int(((gy_step / 6.0) ** 1.6) * (height - horizon_y))
        cv2.line(canvas, (0, y_line), (width, y_line), tuple(int(c * 0.4) for c in acc_c), 1, cv2.LINE_AA)

    # 5. Focal Subject Silhouette / Geometry
    if theme == "forest":
        # Guardian Spirit silhouette with translucent wings of ethereal starlight
        cx, cy = width // 2, int(height * 0.56)
        wing_pts_l = np.array([(cx, cy), (cx - 260, cy - 220), (cx - 340, cy - 40), (cx - 140, cy + 40)], np.int32)
        wing_pts_r = np.array([(cx, cy), (cx + 260, cy - 220), (cx + 340, cy - 40), (cx + 140, cy + 40)], np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [wing_pts_l, wing_pts_r], acc_c, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
        cv2.polylines(canvas, [wing_pts_l, wing_pts_r], True, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy - 15), (32, 95), 0, 0, 360, (240, 255, 255), -1, cv2.LINE_AA)
    elif theme in ("jagdhatri", "martial", "samurai"):
        # Dynamic hero martial action silhouette with aura rim & katana/motion arc
        cx, cy = int(width * 0.48), int(height * 0.62)
        body = np.array([(cx - 40, cy + 160), (cx - 20, cy - 80), (cx + 55, cy - 110), (cx + 85, cy + 160)], np.int32)
        cv2.fillPoly(canvas, [body], (20, 22, 34), cv2.LINE_AA)
        cv2.circle(canvas, (cx + 15, cy - 145), 42, (20, 22, 34), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx + 15, cy - 145), 44, acc_c, 3, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx + 40, cy - 40), (320, 120), -18, 200, 340, acc_c, 6, cv2.LINE_AA)
    elif theme == "hypercar":
        cx, cy = width // 2, int(height * 0.72)
        car = np.array([(cx - 260, cy + 40), (cx - 180, cy - 60), (cx + 160, cy - 60), (cx + 270, cy + 40)], np.int32)
        cv2.fillPoly(canvas, [car], (30, 32, 40), cv2.LINE_AA)
        cv2.polylines(canvas, [car], True, acc_c, 3, cv2.LINE_AA)
        cv2.circle(canvas, (cx - 190, cy + 10), 22, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx + 190, cy + 10), 22, (255, 255, 255), -1, cv2.LINE_AA)

    # 6. Luminous particles / bioluminescence / stars
    for _ in range(140):
        px = int(rng.integers(0, width))
        py = int(rng.integers(0, int(height * 0.92)))
        pr = int(rng.integers(2, 6))
        cv2.circle(canvas, (px, py), pr, acc_c, -1, cv2.LINE_AA)

    # 7. Diagonal volumetric light shafts
    ray_layer = np.zeros_like(canvas)
    for rx in range(int(width * 0.15), int(width * 0.85), int(width * 0.16)):
        pts_ray = np.array([(rx, 0), (rx + 90, 0), (rx + 360, height), (rx + 180, height)], np.int32)
        cv2.fillPoly(ray_layer, [pts_ray], acc_c)
    ray_layer = cv2.GaussianBlur(ray_layer, (81, 81), 0)
    canvas = np.clip(canvas.astype(np.float32) + ray_layer.astype(np.float32) * 0.18, 0, 255).astype(np.uint8)
    return canvas


def resolve_scene_anchor(
    prompt: str,
    artifacts_dir: Path,
    seed: int = 42,
    width: int = 1920,
    height: int = 1080,
    seq_idx: int = 0,
) -> np.ndarray:
    """Resolve an authentic 1080p visual anchor for the prompt (local cache -> domain asset -> offline artwork)."""
    theme = classify_prompt_theme(prompt)
    cache_dir = artifacts_dir.parent / "cache" / "themes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_theme_img = cache_dir / f"theme_{theme}_{width}x{height}.png"

    # 1. Local Jagdhatri Keyframes (zero network, real character)
    if theme == "jagdhatri":
        kf_dir = artifacts_dir.parent / "analysis" / "jagdhatri_ep224" / "keyframes"
        if kf_dir.is_dir():
            kfs = sorted(kf_dir.glob("keyframe_*.jpg"))
            if kfs:
                chosen = kfs[seq_idx % len(kfs)]
                raw = cv2.imread(str(chosen))
                if raw is not None and raw.size > 0:
                    # Fit vertical keyframe into 16:9 widescreen with crisp blurred backdrop
                    fh, fw = raw.shape[:2]
                    bg = cv2.GaussianBlur(cv2.resize(raw, (width, height)), (51, 51), 0)
                    bg = (bg.astype(np.float32) * 0.38).astype(np.uint8)
                    scale = height / float(fh)
                    fg_w = min(width, int(fw * scale))
                    fg = cv2.resize(raw, (fg_w, height), interpolation=cv2.INTER_LINEAR)
                    x_off = (width - fg_w) // 2
                    bg[:, x_off : x_off + fg_w] = _fast_anime_stylize(fg)
                    return bg

    # 2. Cached domain theme image (0ms)
    if cached_theme_img.is_file() and cached_theme_img.stat().st_size > 10_000:
        img = cv2.imread(str(cached_theme_img))
        if img is not None and img.shape[0] > 0:
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            return img

    # 3. Fast curated domain fetch (max 3s timeout, saves to cache)
    url = THEME_IMAGE_CATALOG.get(theme)
    if url:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                raw_bytes = resp.read()
            if len(raw_bytes) > 10_000:
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None and decoded.shape[0] > 0:
                    resized = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(str(cached_theme_img), resized)
                    return resized
        except Exception as exc:
            logger.debug("Fast curated fetch skipped/failed for %s: %s", theme, exc)

    # 4. 100% Independent crisp procedural 1080p artwork fallback
    return synthesize_crisp_scene_anchor(prompt=prompt, seed=seed, width=width, height=height)
