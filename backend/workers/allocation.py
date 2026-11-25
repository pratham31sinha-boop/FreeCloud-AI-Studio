"""Worker allocation and task chunking policies.

Maps video generation duration to planned worker counts according to the
FreeCloud AI Studio architecture specification:

| Duration   | Workers |
| 1 minute   | 4       |
| 2 minutes  | 5       |
| 3 minutes  | 6       |
| 5 minutes  | 7       |
| 10 minutes | 8       |
| 15 minutes | 10      |
"""

from __future__ import annotations

import math
import re
import uuid
from typing import List, Tuple
from backend.workers.models import JobRequest, TaskChunk, TaskChunkStatus


# Exact planned table checkpoints across all stages (1s up to 15m)
ALLOCATION_TABLE: List[Tuple[float, int]] = [
    (0.05, 1),   # Stage 1: 1-3 seconds (single worker fast pass)
    (0.15, 2),   # Stage 2: 5-9 seconds
    (0.30, 2),   # Stage 3: 15 seconds
    (0.60, 3),   # Stage 4: 30 seconds
    (1.0, 4),    # Stage 5: 1 minute -> 4 workers
    (2.0, 5),    # Stage 6: 2 minutes -> 5 workers
    (3.0, 6),    # Stage 7: 3 minutes -> 6 workers
    (5.0, 7),    # Stage 8: 5 minutes -> 7 workers
    (10.0, 8),   # Stage 9: 10 minutes -> 8 workers
    (15.0, 10),  # Stage 10: 15 minutes -> 10 workers
]


def calculate_worker_allocation(duration_minutes: float) -> int:
    """Calculate target number of worker nodes for a given video duration in minutes.

    Architecture requirement: Minimum 4 cloud servers pooled together to combine
    computing power (each runner with 16GB RAM equivalent) for ultra-fast parallel rendering.
    - 1s - 1 min: 4 workers minimum
    - 2 min: 5 workers
    - 3 min: 6 workers
    - 5 min: 7 workers
    - 10 min: 8 workers
    - 15 min: 10 workers
    """
    if duration_minutes <= 1.0:
        return 4
    if duration_minutes <= 2.0:
        return 5
    if duration_minutes <= 3.0:
        return 6
    if duration_minutes <= 5.0:
        return 7
    if duration_minutes <= 10.0:
        return 8
    return 10


def generate_progressive_act_beats(story_prompt: str, total_acts: int = 4) -> List[str]:
    """Generate narrative act progression beats tailored to the subject and prompt."""
    p = story_prompt.lower()

    def has_word(*words: str) -> bool:
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        return bool(re.search(pattern, p))

    if has_word("car", "supercar", "hypercar", "vehicle", "race", "racing", "highway", "drift", "driving"):
        return [
            "Act 1 (Ignition & Acceleration): Sleek titanium hypercar roaring out of a mountain pass onto the coastal cliffside highway at golden hour sunset, twin LED headlights piercing through twilight mist, gleaming metallic reflections",
            "Act 2 (High-Speed Cruising): Accelerating at high velocity along the winding cliffside road, ocean waves crashing on rocks far below, tires gripping asphalt with high-speed motion blur, warm sun flare gleaming on windshield",
            "Act 3 (Apex Drift Maneuver): Aggressive dynamic drift around a dramatic cliffside hairpin curve, tire smoke billowing, glowing red brake calipers, camera tracking the centrifugal slide with intense kinetic momentum",
            "Act 4 (Hero Departure): Surging at full throttle down the wide illuminated coastal straightaway toward the vibrant neon-lit metropolis skyline on the horizon at dusk",
        ]
    elif has_word("jagdhatri", "jas", "jash", "swayambhu", "ambush", "warehouse", "operative", "martial"):
        return [
            "Act 1 (Tactical Infiltration): Jagdhatri (Codename: JAS) enters the dimly lit industrial warehouse, scanning catwalks and shadows with razor-sharp vigilance",
            "Act 2 (Ambush & Counter-Strike): Detecting hostile operatives in motion, executing a lightning-fast defensive martial arts spin-kick with sweeping momentum",
            "Act 3 (Perimeter Lock): Continuing seamlessly as Jash covers the outer exits, high-contrast muzzle flashes and neon warehouse spotlights illuminating the combat zone",
            "Act 4 (Classified Extraction): Securing the encrypted intelligence briefcase in unbroken poise while the warehouse dust settles into the night skyline",
        ]
    elif has_word("elven", "forest", "bioluminescence", "bioluminescent", "cathedral", "fairy", "wings", "guardian"):
        return [
            "Act 1 (Sanctuary Awakening): Ancient bioluminescent forest cathedral glowing under ethereal starlight shafts as the sacred guardian spirit begins to materialize",
            "Act 2 (Translucent Wings Unfurl): Expanding luminous wings of liquid starlight amidst drifting spores and ancient rune-carved mossy pillars",
            "Act 3 (Celestial Resonance): Rising in graceful harmonic suspension as emerald and cyan light waves ripple outward through the enchanted canopy",
            "Act 4 (Ascension Into Starlight): Surging skyward through the towering cathedral arches, scattering radiant trails of bioluminescent constellations",
        ]
    elif has_word("astronaut", "space", "alien", "planet", "galaxy", "mars", "cosmos", "rover", "nebula"):
        return [
            "Act 1 (Touchdown & First Steps): Astronaut stepping onto glowing alien planetary soil, gold-tinted reflective visor catching distant binary suns, atmospheric dust drifting",
            "Act 2 (Traversal): Walking past crystalline rock spires and bioluminescent geysers, scanning alien geological formations with high-tech scanner",
            "Act 3 (Discovery): Approaching an ancient floating alien monolith, geometric engravings pulsing with radiant blue energy as the astronaut reaches forward",
            "Act 4 (Ascension): The alien artifact unleashes a swirling column of celestial light, camera pulling up into the starry nebula cosmos",
        ]
    elif has_word("samurai", "cyberpunk", "ronin", "tokyo", "katana", "ninja", "blade"):
        return [
            "Act 1 (Atmospheric Entrance): Cybernetic ronin walking slowly through a rain-drenched neon Tokyo alleyway at night, trench coat billowing in the wind",
            "Act 2 (Drawing the Blade): Stepping through puddles reflecting vibrant holographic billboards, drawing a glowing energy katana with an electric hum",
            "Act 3 (Combat Climax): Dynamic spinning strike through falling rain, cutting holographic light beams with sparks and arcs of electric energy",
            "Act 4 (Silent Exit): Smoothly sheathing the glowing blade as rain falls, fading into the dark shadows of the towering neon skyline",
        ]
    elif has_word("panther", "leopard", "tiger", "lion", "cat", "feline", "jaguar"):
        return [
            "Act 1 (Emergence): The majestic predator stealthily emerges from deep misty rainforest shadows into the light, initiating a smooth predatory prowl, glowing emerald eyes scanning ahead",
            "Act 2 (Progression): Continuing forward through the terrain, stepping across glistening wet mossy riverstones, water droplets flying off paws, sleek fur rippling with continuous momentum",
            "Act 3 (Climactic Tension): Intense moment of focus, lowering into an aggressive stance, baring sharp fangs in a low territorial snarl, breath visible in humid night air, muscles tensed",
            "Act 4 (Grand Finale): Powerful surge forward into a majestic full-speed stride, accelerating past luminous bioluminescent flora into the deep jungle canopy",
        ]
    else:
        return [
            f"Act 1 (Introduction): Establishing shot introducing {story_prompt.strip()}, emerging into the scene with atmospheric presence and smooth initial movement",
            f"Act 2 (Development): Dynamic motion progression of {story_prompt.strip()}, traversing through the environment with continuous physical momentum",
            f"Act 3 (Climax): High-intensity dramatic moment with peak action, emotional focus, and dynamic kinetic energy",
            f"Act 4 (Resolution): Powerful concluding motion sequence surging toward the destination, panoramic cinematic camera pull-back",
        ]


RELAY_MODEL_SEQUENCE = ["wan_2_1", "minimax_h3", "hunyuan_video"]


def plan_task_chunks(
    job_id: str,
    request: JobRequest,
    target_chunk_sec: float = 15.0,
) -> List[TaskChunk]:
    """Break a video generation job into sequential task chunks for distributed workers.

    Each chunk corresponds to a continuous segment of time with its narrative prompt,
    character IDs, and continuity references. Supports single-model lock or
    sequential multi-model relay (Model 1 -> Video-as-Code + Last Frame -> Model 2).
    """
    total_seconds = max(10.0, float(request.duration_minutes * 60.0))
    worker_count = calculate_worker_allocation(request.duration_minutes)

    # Determine chunk count: at least equal to worker_count so all workers can be utilized,
    # or divided by target_chunk_sec if the video is longer.
    calculated_chunks = max(worker_count, math.ceil(total_seconds / target_chunk_sec))
    chunk_duration = total_seconds / calculated_chunks

    chunks: List[TaskChunk] = []
    current_time = 0.0

    from backend.generation.documentary_planner import (
        build_documentary_plan,
        is_documentary_job,
    )

    use_documentary = is_documentary_job(request.story_prompt, request.duration_minutes)
    doc_plan = (
        build_documentary_plan(request.story_prompt, total_seconds, calculated_chunks)
        if use_documentary
        else None
    )
    act_beats = generate_progressive_act_beats(request.story_prompt, total_acts=calculated_chunks)
    req_model = getattr(request, "model_name", "wan_2_1") or "wan_2_1"
    use_relay = bool(getattr(request, "model_relay", False) or req_model == "auto_relay")

    for idx in range(calculated_chunks):
        end_time = min(total_seconds, current_time + chunk_duration)
        chunk_model = RELAY_MODEL_SEQUENCE[idx % len(RELAY_MODEL_SEQUENCE)] if use_relay else req_model

        if doc_plan:
            p_item = doc_plan[idx]
            ch_idx = p_item["chapter_index"]
            ch_title = p_item["chapter_title"]
            ch_sub = p_item["chapter_subtitle"]
            is_ch_start = p_item["is_chapter_start"]
            is_ch_end = p_item["is_chapter_end"]
            dep_idx = p_item["depends_on_chunk_index"]
            beat_text = p_item["chapter_beat"]
            if is_ch_start:
                scene_prompt = (
                    f"[{ch_title} | Shot {idx + 1}/{calculated_chunks} | {current_time:.1f}s-{end_time:.1f}s] "
                    f"{request.story_prompt.strip()} -- {beat_text}"
                )
            else:
                scene_prompt = (
                    f"[EXTENDED CLIP {idx + 1}/{calculated_chunks} ({ch_title}) | "
                    f"Continuing seamlessly from Shot {idx} Last Frame Image & Coded Video State ({current_time:.1f}s-{end_time:.1f}s)] "
                    f"Next Scene Continuation: {beat_text} | Documentary Subject: {request.story_prompt.strip()}"
                )
        else:
            ch_idx = 0
            ch_title = None
            ch_sub = None
            is_ch_start = (idx == 0)
            is_ch_end = (idx == calculated_chunks - 1)
            dep_idx = None if idx == 0 else (idx - 1)
            beat_text = act_beats[idx % len(act_beats)]
            if idx == 0:
                scene_prompt = (
                    f"[Act {idx + 1}/{calculated_chunks} | {current_time:.1f}s-{end_time:.1f}s] "
                    f"{request.story_prompt.strip()} -- {beat_text}"
                )
            else:
                scene_prompt = (
                    f"[EXTENDED CLIP {idx + 1}/{calculated_chunks} | Continuing seamlessly from Clip {idx} "
                    f"Last Frame Image & Coded Video State ({current_time:.1f}s-{end_time:.1f}s)] "
                    f"Next Scene Continuation: {beat_text} | Original Story Prompt: {request.story_prompt.strip()}"
                )

        chunk = TaskChunk(
            chunk_id=f"{job_id}_chk_{idx + 1:03d}_{uuid.uuid4().hex[:6]}",
            job_id=job_id,
            sequence_index=idx,
            start_sec=round(current_time, 2),
            end_sec=round(end_time, 2),
            duration_sec=round(end_time - current_time, 2),
            prompt_segment=scene_prompt,
            anime_style=str(request.anime_style),
            model_name=chunk_model,
            anchor_frame_url=f"/api/v1/jobs/{job_id}/anchor/{idx}",
            is_extended_clip=(dep_idx is not None),
            depends_on_chunk_index=dep_idx,
            chapter_index=ch_idx,
            chapter_title=ch_title,
            chapter_subtitle=ch_sub,
            is_chapter_start=is_ch_start,
            is_chapter_end=is_ch_end,
            character_ids=list(request.character_ids),
            motion_reference_id=f"motion_{job_id}_seq_{idx:03d}" if request.continuity_motion_tracking else None,
            status=TaskChunkStatus.PENDING,
        )
        chunks.append(chunk)
        current_time = end_time

    return chunks

