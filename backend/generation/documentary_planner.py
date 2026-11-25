"""Multi-Chapter Documentary Planner for Long-Form Video Generation (2 to 15 Minutes).

Structures long-form prompts into distinct narrative chapters (each consisting of
sequentially linked Extend Clip + Video-as-Code shots) so chapters execute in
parallel across workers while preserving pixel-locked continuity within each chapter
and preventing cumulative zoom collapse or repetitive act loops.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List


def is_documentary_job(prompt: str, duration_minutes: float) -> bool:
    """Detect whether a job should engage multi-chapter documentary planning."""
    p = (prompt or "").lower()
    if duration_minutes >= 1.5:
        return True
    return bool(
        re.search(
            r"\b(documentary|docuseries|history of|chronicle|odyssey|universe|cosmos|planet earth|deep ocean)\b",
            p,
        )
    )


def classify_documentary_domain(prompt: str) -> str:
    p = (prompt or "").lower()
    if re.search(r"\b(cosmos|universe|space|galaxy|black hole|nebula|big bang|star|astronomy|solar|planet)\b", p):
        return "cosmos"
    if re.search(r"\b(ocean|sea|abyss|marine|whale|coral|deep|aquatic|reef|jellyfish)\b", p):
        return "ocean"
    return "earth"


DOCUMENTARY_CHAPTER_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
    "cosmos": [
        {"title": "CHAPTER I — PRIMORDIAL SPARK", "subtitle": "13.8 Billion Years Ago • Before Stars Ignited", "beat": "Vast cosmic microwave glow resolving into the first filaments of hydrogen and dark matter across newborn spacetime"},
        {"title": "CHAPTER II — NURSERIES OF LIGHT", "subtitle": "Pillars of Creation • Cosmic Gas & Ionized Dust", "beat": "Towering interstellar nebulae illuminated by ultraviolet winds from newborn supergiant stars"},
        {"title": "CHAPTER III — FORGES OF HEAVENS", "subtitle": "Nuclear Fusion • Inside Stellar Furnaces", "beat": "Blazing stellar coronas erupting in colossal plasma loops, forging carbon, oxygen, and iron"},
        {"title": "CHAPTER IV — THE SPIRAL ARCHITECTS", "subtitle": "200 Billion Suns • Galactic Dynamics", "beat": "Sweeping spiral arms of luminous star clusters rotating calmly around an incandescent galactic core"},
        {"title": "CHAPTER V — COLLISION & SUPERNOVA", "subtitle": "Violent Transmutation • Cosmic Rebirth", "beat": "Expanding shockwaves of a dying star scattering gold, platinum, and luminous stardust through void"},
        {"title": "CHAPTER VI — EVENT HORIZON", "subtitle": "Where Gravity Bends Time & Light", "beat": "Relativistic accretion disk swirling furiously around the dark shadow of a supermassive black hole"},
        {"title": "CHAPTER VII — PALE BLUE CRADLE", "subtitle": "An Oasis Adrift in Cosmic Night", "beat": "Fragile sunlit atmosphere wrapping a world suspended against the endless starry horizon"},
        {"title": "CHAPTER VIII — HORIZON OF TOMORROW", "subtitle": "Observing the Infinite • Looking Forward", "beat": "Deep-space vantage sweeping outward into unmapped galaxies glowing silently in eternity"},
    ],
    "ocean": [
        {"title": "CHAPTER I — THE BLUE FRONTIER", "subtitle": "71% of Earth • Surface in Perpetual Motion", "beat": "Endless sapphire ocean swells catching golden dawn sunlight across open pelagic waters"},
        {"title": "CHAPTER II — ARCHITECTS OF REEF", "subtitle": "Sunlit Gardens • Living Limestone Cities", "beat": "Vibrant coral ecosystems teeming with synchronized shoals of iridescent reef swimmers"},
        {"title": "CHAPTER III — CURRENTS OF LIFE", "subtitle": "Global Conveyor • Thermal Migrations", "beat": "Great ocean currents sweeping nutrient plumes across submerged seamounts and underwater canyons"},
        {"title": "CHAPTER IV — TWILIGHT DESCENDING", "subtitle": "200 Meters Deep • The Mesopelagic Zone", "beat": "Fading cobalt light giving way to translucent drifters descending through quiet mid-water darkness"},
        {"title": "CHAPTER V — BIOLUMINESCENT ABYSS", "subtitle": "Living Starlight in Eternal Midnight", "beat": "Ethereal jellyfish and deep-sea organisms pulsing with cold cyan and violet bioluminescence"},
        {"title": "CHAPTER VI — GIANTS OF THE DEEP", "subtitle": "Acoustic Titans Spanning Ocean Basins", "beat": "Majestic whales gliding effortlessly through cathedral shafts of underwater sunbeams"},
        {"title": "CHAPTER VII — THERMAL VENTS", "subtitle": "Volcanic Chimneys • Origins of Life", "beat": "Superheated mineral plumes erupting on the abyssal floor far below sunlit waves"},
        {"title": "CHAPTER VIII — STEWARDS OF THE TIDE", "subtitle": "Fragile Balance • Tomorrow's Seas", "beat": "Calm oceanic horizon glowing softly under twilight skies as waves roll onward"},
    ],
    "earth": [
        {"title": "CHAPTER I — AWAKENING PLANET", "subtitle": "Four Billion Years of Tectonic Sculpture", "beat": "Misty primeval mountain ridges emerging from swirling dawn clouds across untouched wilderness"},
        {"title": "CHAPTER II — THE GREEN LUNGS", "subtitle": "Ancient Canopies • Symbiosis in Canopy Mist", "beat": "Towering emerald forest cathedrals filtering golden shafts of morning humidity and mist"},
        {"title": "CHAPTER III — FROZEN SUMMITS", "subtitle": "Carved by Ice • High Altitude Frontiers", "beat": "Snow-clad alpine peaks catching crisp alpenglow above sweeping glacial valleys"},
        {"title": "CHAPTER IV — PULSE OF WATER", "subtitle": "Rivers Shaping Stone Across Millennia", "beat": "Rushing canyon rivers carving ancient strata beneath dramatic golden-hour skies"},
        {"title": "CHAPTER V — THE APEX PREDATORS", "subtitle": "Precision & Patience • Wild Survival", "beat": "Stealthy apex wildlife moving gracefully through sun-dappled grasslands and jungle shadow"},
        {"title": "CHAPTER VI — NIGHT UNDER AURORA", "subtitle": "Solar Winds Dancing Across Polar Skies", "beat": "Luminous emerald aurora ribbons undulating silently above quiet northern peaks"},
        {"title": "CHAPTER VII — HARMONY IN MOTION", "subtitle": "Seasonal Migrations • Planetary Rhythm", "beat": "Sweeping panoramic vistas uniting forest, mountain, sky, and living ecosystems"},
        {"title": "CHAPTER VIII — OUR SHARED HORIZON", "subtitle": "Preserving Wonder for Future Generations", "beat": "Radiant sunset horizon shining across thriving earth landscapes in timeless calm"},
    ],
}

SHOT_MICRO_ARCS: List[str] = [
    "Opening Panorama: Establishing majestic scale with slow, steady atmospheric reveal",
    "Immersion Track: Gliding deeper into the environment, highlighting intricate natural textures",
    "Kinetic Observation: Following dynamic light, particles, and fluid natural motion continuously",
    "Intimate Close Study: Dramatic lighting interplay illuminating subtle structural geometry",
    "Reflective Chapter Close: Expansive sweep building smoothly toward the chapter's quiet twilight transition",
]


def build_documentary_plan(
    story_prompt: str,
    total_seconds: float,
    total_chunks: int,
) -> List[Dict[str, Any]]:
    """Build per-chunk chapter structure, dependency DAG, and non-repeating visual beats."""
    domain = classify_documentary_domain(story_prompt)
    chapters = DOCUMENTARY_CHAPTER_TEMPLATES[domain]

    # Target up to 8 balanced chapters (or fewer for short durations)
    num_chapters = min(len(chapters), max(2, min(8, total_chunks // 2 if total_chunks >= 4 else 1)))
    if total_chunks < 4:
        num_chapters = 1

    base_per_ch = total_chunks // num_chapters
    rem = total_chunks % num_chapters

    plan: List[Dict[str, Any]] = []
    seq = 0
    for ch_idx in range(num_chapters):
        count = base_per_ch + (1 if ch_idx < rem else 0)
        ch_meta = chapters[ch_idx % len(chapters)]
        for local_i in range(count):
            is_start = (local_i == 0)
            is_end = (local_i == count - 1)
            micro_arc = SHOT_MICRO_ARCS[local_i % len(SHOT_MICRO_ARCS)]
            plan.append({
                "sequence_index": seq,
                "chapter_index": ch_idx,
                "chapter_title": ch_meta["title"],
                "chapter_subtitle": ch_meta["subtitle"],
                "chapter_beat": f"{ch_meta['beat']} — {micro_arc}",
                "is_chapter_start": is_start,
                "is_chapter_end": is_end,
                "depends_on_chunk_index": None if is_start else (seq - 1),
                "domain": domain,
            })
            seq += 1

    return plan
