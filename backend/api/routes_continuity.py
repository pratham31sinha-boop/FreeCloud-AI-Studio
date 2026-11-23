"""Continuity Engine API routes."""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from backend.continuity.audio_sync import audio_sync_engine
from backend.continuity.base import CharacterProfile, MotionSequence, StoryBeat, VoiceProfile
from backend.continuity.character_refs import character_ref_manager
from backend.continuity.motion_refs import motion_action_manager
from backend.continuity.quality_check import quality_checker
from backend.continuity.story_continuity import story_continuity_engine
from backend.continuity.voice_refs import voice_reference_manager

router = APIRouter(prefix="/continuity", tags=["Continuity Engine"])


class ScriptAnalysisRequest(BaseModel):
    script_text: str


class AudioSyncRequest(BaseModel):
    dialogue: str
    character_id: str
    start_sec: float = 0.0
    duration_sec: float = 3.0


@router.get("/characters", response_model=List[CharacterProfile])
def list_characters() -> List[CharacterProfile]:
    """List canonical character references (including Jagdhatri presets)."""
    return character_ref_manager.list_characters()


@router.get("/characters/{character_id}", response_model=CharacterProfile)
def get_character(character_id: str) -> CharacterProfile:
    """Retrieve details for a specific character identity reference."""
    profile = character_ref_manager.get_character(character_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character {character_id} not found in continuity registry",
        )
    return profile


@router.post("/characters", response_model=CharacterProfile, status_code=status.HTTP_201_CREATED)
def register_character(profile: CharacterProfile) -> CharacterProfile:
    """Register or update a character reference profile."""
    character_ref_manager.register_character(profile)
    return profile


@router.get("/motions", response_model=List[MotionSequence])
def list_motions() -> List[MotionSequence]:
    """List registered motion and choreography references."""
    return motion_action_manager.list_motions()


@router.get("/voices", response_model=List[VoiceProfile])
def list_voices() -> List[VoiceProfile]:
    """List registered character voice timbre references."""
    return voice_reference_manager.list_voices()


@router.post("/analyze-script", response_model=List[StoryBeat])
def analyze_script(req: ScriptAnalysisRequest) -> List[StoryBeat]:
    """Break a narrative script into sequential beats with character associations."""
    return story_continuity_engine.analyze_script(req.script_text)


@router.post("/audio-sync-plan")
def create_audio_sync_plan(req: AudioSyncRequest) -> Dict[str, Any]:
    """Generate phoneme/viseme timing alignment plan for lip-sync."""
    plan = audio_sync_engine.plan_synchronization(
        dialogue=req.dialogue,
        character_id=req.character_id,
        start_sec=req.start_sec,
        duration_sec=req.duration_sec,
    )
    return plan.model_dump()


@router.get("/quality-thresholds")
def get_quality_thresholds() -> Dict[str, Any]:
    """Return configured continuity assurance quality thresholds."""
    return {
        "min_identity_consistency": quality_checker.min_quality_score,
        "max_temporal_flicker": 0.35,
        "min_motion_fidelity": 0.65,
        "min_audio_sync_confidence": 0.75,
    }
