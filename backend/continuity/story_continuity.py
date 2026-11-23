"""Story Continuity Engine.

Parses input stories and serial transcripts into structured beats with character
and dialogue associations to maintain narrative flow.
"""

from __future__ import annotations

import re
from typing import List
from backend.continuity.base import IStoryContinuityEngine, StoryBeat


class StoryContinuityEngine(IStoryContinuityEngine):
    """Deconstructs narrative prompts into sequential story beats."""

    def analyze_script(self, script_text: str) -> List[StoryBeat]:
        """Convert a story script or prompt into sequential narrative beats."""
        lines = [line.strip() for line in script_text.splitlines() if line.strip()]
        if not lines:
            return []

        beats: List[StoryBeat] = []
        dialogue_pattern = re.compile(r"^([A-Za-z0-9_\s]+)\s*:\s*(.+)$")

        beat_idx = 0
        for line in lines:
            # Check if this line is formatted as Character: Dialogue
            match = dialogue_pattern.match(line)
            if match:
                char_name, dialogue_text = match.groups()
                normalized_id = char_name.strip().lower().replace(" ", "_")
                beats.append(
                    StoryBeat(
                        beat_index=beat_idx,
                        description=f"{char_name} speaks: {dialogue_text}",
                        speaking_character_id=normalized_id,
                        dialogue=dialogue_text.strip(),
                        emotional_tone=self._detect_tone(dialogue_text),
                    )
                )
            else:
                # Narrative description beat
                beats.append(
                    StoryBeat(
                        beat_index=beat_idx,
                        description=line,
                        speaking_character_id=None,
                        dialogue=None,
                        emotional_tone=self._detect_tone(line),
                    )
                )
            beat_idx += 1

        return beats

    def _detect_tone(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ("fight", "strike", "attack", "run", "pursuit", "danger")):
            return "high_action"
        if any(w in lower for w in ("investigate", "secret", "clue", "shadow", "observe")):
            return "mysterious_tense"
        if any(w in lower for w in ("family", "home", "mother", "love", "smile", "peace")):
            return "warm_domestic"
        return "neutral_dramatic"


story_continuity_engine = StoryContinuityEngine()
