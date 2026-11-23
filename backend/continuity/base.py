"""Abstract interfaces and core data models for the FreeCloud Continuity Engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CharacterProfile(BaseModel):
    """Reference profile for a consistent character identity."""

    character_id: str
    name: str
    role: str
    anime_archetype: str
    physical_traits: Dict[str, str] = Field(default_factory=dict)
    clothing_styles: List[Dict[str, str]] = Field(default_factory=list)
    reference_images: List[str] = Field(default_factory=list)
    embedding_vector: Optional[List[float]] = None
    lora_trigger: Optional[str] = None
    created_at: float = 0.0


class AppearanceSnapshot(BaseModel):
    """Captured appearance state of a character at a specific scene/timestamp."""

    character_id: str
    scene_index: int
    timestamp_sec: float
    current_clothing: str
    facial_expression: str
    appearance_embedding: Optional[List[float]] = None
    is_injured_or_altered: bool = False
    alteration_notes: Optional[str] = None


class MotionSequence(BaseModel):
    """Motion and camera dynamics describing an action sequence."""

    motion_id: str
    scene_index: int
    start_sec: float
    end_sec: float
    action_type: str  # e.g., "martial_arts_defense", "investigative_walk", "dramatic_turn"
    camera_motion: str  # e.g., "tracking_shot", "close_up_pan", "static"
    pose_keypoints_url: Optional[str] = None
    intensity_score: float = 0.5


class SceneState(BaseModel):
    """Temporal environment and lighting continuity context for a scene."""

    scene_index: int
    location: str
    time_of_day: str  # e.g., "interior_daylight", "night_office", "rainy_street"
    lighting_palette: List[str] = Field(default_factory=list)
    atmospheric_effects: List[str] = Field(default_factory=list)
    preceding_scene_index: Optional[int] = None


class StoryBeat(BaseModel):
    """Narrative segment within the video timeline."""

    beat_index: int
    description: str
    speaking_character_id: Optional[str] = None
    dialogue: Optional[str] = None
    emotional_tone: str = "neutral"


class VoiceProfile(BaseModel):
    """Voice timbre and speech reference for consistent character audio."""

    character_id: str
    voice_name: str
    language_code: str = "en"
    pitch_adjustment: float = 0.0
    reference_sample_url: Optional[str] = None
    timbre_embedding_id: Optional[str] = None


class AudioSyncPlan(BaseModel):
    """Audio and dialogue synchronization plan."""

    character_id: str
    dialogue: str
    start_sec: float
    duration_sec: float
    viseme_sequence: List[Dict[str, Any]] = Field(default_factory=list)
    sync_confidence: float = 0.0


class QualityReport(BaseModel):
    """Comprehensive continuity and visual quality validation report."""

    job_id: str
    passed: bool
    identity_consistency_score: float = Field(
        description="Cosine similarity or perceptual consistency score (0.0 to 1.0)"
    )
    motion_fidelity_score: float = Field(
        description="Adherence to original action trajectory (0.0 to 1.0)"
    )
    temporal_flicker_score: float = Field(
        description="Measurement of inter-frame jitter (0.0=none, 1.0=severe)"
    )
    audio_sync_confidence: float = Field(
        description="Lip-sync and dialogue alignment metric (0.0 to 1.0)"
    )
    issues_detected: List[str] = Field(default_factory=list)
    remediation_recommendations: List[str] = Field(default_factory=list)


# --- Modular Abstract Interfaces ---


class ICharacterReferenceManager(ABC):
    """Interface for registering and querying fixed character identity references."""

    @abstractmethod
    def register_character(self, profile: CharacterProfile) -> bool:
        pass

    @abstractmethod
    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        pass

    @abstractmethod
    def list_characters(self) -> List[CharacterProfile]:
        pass


class ICharacterAppearanceMemory(ABC):
    """Interface for tracking dynamic appearance across shots to prevent drift."""

    @abstractmethod
    def record_snapshot(self, snapshot: AppearanceSnapshot) -> None:
        pass

    @abstractmethod
    def get_latest_snapshot(self, character_id: str) -> Optional[AppearanceSnapshot]:
        pass

    @abstractmethod
    def check_drift(
        self, character_id: str, current_embedding: List[float]
    ) -> float:
        """Returns drift score from reference base (0.0 = identical, 1.0 = divergence)."""
        pass


class IMotionActionReferenceManager(ABC):
    """Interface for storing and extracting motion and choreographic sequences."""

    @abstractmethod
    def register_motion(self, motion: MotionSequence) -> None:
        pass

    @abstractmethod
    def get_motion_for_scene(self, scene_index: int) -> Optional[MotionSequence]:
        pass


class ISceneHistoryTracker(ABC):
    """Interface for tracking environment, lighting, and timeline continuity."""

    @abstractmethod
    def record_scene_state(self, state: SceneState) -> None:
        pass

    @abstractmethod
    def get_scene_state(self, scene_index: int) -> Optional[SceneState]:
        pass


class IStoryContinuityEngine(ABC):
    """Interface for breaking story scripts into coherent sequential beats."""

    @abstractmethod
    def analyze_script(self, script_text: str) -> List[StoryBeat]:
        pass


class IVoiceReferenceManager(ABC):
    """Interface for storing voice reference models and pitch characteristics."""

    @abstractmethod
    def register_voice(self, profile: VoiceProfile) -> None:
        pass

    @abstractmethod
    def get_voice(self, character_id: str) -> Optional[VoiceProfile]:
        pass


class IAudioSyncEngine(ABC):
    """Interface for synchronizing character lip-sync and dialogue timing."""

    @abstractmethod
    def plan_synchronization(
        self, dialogue: str, character_id: str, start_sec: float, duration_sec: float
    ) -> AudioSyncPlan:
        pass


class IQualityChecker(ABC):
    """Interface for validating continuity, identity preservation, and quality thresholds."""

    @abstractmethod
    def evaluate_job(
        self, job_id: str, chunk_metrics: List[Dict[str, Any]]
    ) -> QualityReport:
        pass
