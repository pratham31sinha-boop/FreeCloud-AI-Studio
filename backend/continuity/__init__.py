"""Continuity Engine package for FreeCloud AI Studio."""

from backend.continuity.audio_sync import AudioSyncEngine, audio_sync_engine
from backend.continuity.base import (
    AppearanceSnapshot,
    AudioSyncPlan,
    CharacterProfile,
    IAudioSyncEngine,
    ICharacterAppearanceMemory,
    ICharacterReferenceManager,
    IMotionActionReferenceManager,
    IQualityChecker,
    ISceneHistoryTracker,
    IStoryContinuityEngine,
    IVoiceReferenceManager,
    MotionSequence,
    QualityReport,
    SceneState,
    StoryBeat,
    VoiceProfile,
)
from backend.continuity.character_memory import (
    CharacterAppearanceMemory,
    character_appearance_memory,
)
from backend.continuity.character_refs import (
    CharacterReferenceManager,
    character_ref_manager,
)
from backend.continuity.motion_refs import (
    MotionActionReferenceManager,
    motion_action_manager,
)
from backend.continuity.quality_check import QualityChecker, quality_checker
from backend.continuity.scene_history import (
    SceneHistoryTracker,
    scene_history_tracker,
)
from backend.continuity.story_continuity import (
    StoryContinuityEngine,
    story_continuity_engine,
)
from backend.continuity.voice_refs import (
    VoiceReferenceManager,
    voice_reference_manager,
)

__all__ = [
    "AppearanceSnapshot",
    "AudioSyncPlan",
    "CharacterProfile",
    "MotionSequence",
    "QualityReport",
    "SceneState",
    "StoryBeat",
    "VoiceProfile",
    "ICharacterReferenceManager",
    "ICharacterAppearanceMemory",
    "IMotionActionReferenceManager",
    "ISceneHistoryTracker",
    "IStoryContinuityEngine",
    "IVoiceReferenceManager",
    "IAudioSyncEngine",
    "IQualityChecker",
    "CharacterReferenceManager",
    "character_ref_manager",
    "CharacterAppearanceMemory",
    "character_appearance_memory",
    "MotionActionReferenceManager",
    "motion_action_manager",
    "SceneHistoryTracker",
    "scene_history_tracker",
    "StoryContinuityEngine",
    "story_continuity_engine",
    "VoiceReferenceManager",
    "voice_reference_manager",
    "AudioSyncEngine",
    "audio_sync_engine",
    "QualityChecker",
    "quality_checker",
]
