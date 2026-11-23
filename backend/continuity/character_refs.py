"""Character Reference Manager component for the Continuity Engine."""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from backend.continuity.base import CharacterProfile, ICharacterReferenceManager


class CharacterReferenceManager(ICharacterReferenceManager):
    """Stores and retrieves canonical character reference profiles.

    Pre-seeded with reference models for the Jagdhatri anime conversion project.
    """

    def __init__(self):
        self._profiles: Dict[str, CharacterProfile] = {}
        self._seed_jagdhatri_presets()

    def register_character(self, profile: CharacterProfile) -> bool:
        if not profile.created_at:
            profile.created_at = time.time()
        self._profiles[profile.character_id] = profile
        return True

    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        return self._profiles.get(character_id)

    def list_characters(self) -> List[CharacterProfile]:
        return list(self._profiles.values())

    def _seed_jagdhatri_presets(self) -> None:
        """Seed reference profiles for Jagdhatri serial recreation."""
        jagdhatri = CharacterProfile(
            character_id="jagdhatri_main",
            name="Jagdhatri (JAS)",
            role="Special Crime Branch Officer / Domestic Persona",
            anime_archetype="Action Heroine & Dual-Identity Detective",
            physical_traits={
                "hair": "Long jet-black hair tied in functional tactical braid for action, sleek open hair for domestic scenes",
                "eyes": "Sharp amber-brown eyes, intense focused gaze in investigative scenes",
                "facial_features": "High cheekbones, resolute expression, minimal makeup, small silver studs",
                "build": "Athletic, agile, tall and commanding presence",
            },
            clothing_styles=[
                {
                    "style_name": "Tactical Officer",
                    "palette": "Dark navy tactical jacket, charcoal cargo trousers, combat boots",
                    "notes": "Action sequences, undercover investigations",
                },
                {
                    "style_name": "Traditional Saree",
                    "palette": "Red and gold Bengali silk saree, subtle jewellery",
                    "notes": "Family / domestic dramatic scenes",
                },
            ],
            reference_images=[
                "characters/jagdhatri/turnaround_front.png",
                "characters/jagdhatri/turnaround_side.png",
                "characters/jagdhatri/action_pose_01.png",
            ],
            embedding_vector=[0.12, 0.45, -0.23, 0.88, 0.31, 0.05, -0.62, 0.41],
            lora_trigger="jagdhatri_anime_v1",
            created_at=time.time(),
        )

        jash = CharacterProfile(
            character_id="jash_lead",
            name="Swayambhu Mukherjee (Jash)",
            role="Senior Crime Branch Officer",
            anime_archetype="Charismatic Detective & Action Lead",
            physical_traits={
                "hair": "Styled dark hair, slightly textured with side parting",
                "eyes": "Warm dark brown, observant and expressive",
                "facial_features": "Defined jawline, charismatic smile, confident stance",
                "build": "Broad shoulders, athletic frame",
            },
            clothing_styles=[
                {
                    "style_name": "Police Formal / Blazer",
                    "palette": "Tailored charcoal blazer, crisp white shirt, formal leather shoes",
                    "notes": "Interrogation and briefing scenes",
                },
                {
                    "style_name": "Field Action",
                    "palette": "Olive leather jacket, dark denim trousers, tactical watch",
                    "notes": "Pursuit and combat scenes",
                },
            ],
            reference_images=[
                "characters/jash/turnaround_front.png",
                "characters/jash/turnaround_profile.png",
            ],
            embedding_vector=[-0.15, 0.62, 0.33, -0.42, 0.54, 0.11, 0.28, -0.37],
            lora_trigger="jash_anime_v1",
            created_at=time.time(),
        )

        self.register_character(jagdhatri)
        self.register_character(jash)


# Global singleton instance
character_ref_manager = CharacterReferenceManager()
