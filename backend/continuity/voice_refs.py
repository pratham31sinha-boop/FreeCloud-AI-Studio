"""Voice Reference Manager component for the Continuity Engine."""

from __future__ import annotations

from typing import Dict, List, Optional
from backend.continuity.base import IVoiceReferenceManager, VoiceProfile


class VoiceReferenceManager(IVoiceReferenceManager):
    """Stores speaker timbre profiles for audio voice synthesis."""

    def __init__(self):
        self._profiles: Dict[str, VoiceProfile] = {}
        self._seed_jagdhatri_voices()

    def register_voice(self, profile: VoiceProfile) -> None:
        self._profiles[profile.character_id] = profile

    def get_voice(self, character_id: str) -> Optional[VoiceProfile]:
        return self._profiles.get(character_id)

    def list_voices(self) -> List[VoiceProfile]:
        return list(self._profiles.values())

    def _seed_jagdhatri_voices(self) -> None:
        self.register_voice(
            VoiceProfile(
                character_id="jagdhatri_main",
                voice_name="Jagdhatri_Original_Bengali_Hindi",
                language_code="bn",
                pitch_adjustment=0.0,
                reference_sample_url="/data/characters/audio/jagdhatri_voice_ref.wav",
                timbre_embedding_id="timbre_jas_v1",
            )
        )
        self.register_voice(
            VoiceProfile(
                character_id="jash_lead",
                voice_name="Jash_Mukherjee_Voice",
                language_code="bn",
                pitch_adjustment=-0.05,
                reference_sample_url="/data/characters/audio/jash_voice_ref.wav",
                timbre_embedding_id="timbre_jash_v1",
            )
        )


voice_reference_manager = VoiceReferenceManager()
