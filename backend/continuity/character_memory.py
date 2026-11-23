"""Character Appearance Memory module.

Tracks temporal appearance snapshots across shots and detects visual drift
between chunks assigned to separate worker nodes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
from backend.continuity.base import AppearanceSnapshot, ICharacterAppearanceMemory


class CharacterAppearanceMemory(ICharacterAppearanceMemory):
    """Stores temporal appearance history and evaluates cross-chunk visual drift."""

    def __init__(self):
        # Mapping from character_id to sequential snapshots
        self._history: Dict[str, List[AppearanceSnapshot]] = {}

    def record_snapshot(self, snapshot: AppearanceSnapshot) -> None:
        """Record the appearance state of a character at a specific shot/time."""
        if snapshot.character_id not in self._history:
            self._history[snapshot.character_id] = []
        self._history[snapshot.character_id].append(snapshot)
        # Keep snapshots sorted by timestamp
        self._history[snapshot.character_id].sort(key=lambda s: s.timestamp_sec)

    def get_latest_snapshot(self, character_id: str) -> Optional[AppearanceSnapshot]:
        """Retrieve the most recent appearance state for continuity chaining."""
        snapshots = self._history.get(character_id, [])
        return snapshots[-1] if snapshots else None

    def get_history_for_character(
        self, character_id: str
    ) -> List[AppearanceSnapshot]:
        return list(self._history.get(character_id, []))

    def check_drift(
        self, character_id: str, current_embedding: List[float]
    ) -> float:
        """Compute visual drift using cosine distance against recent reference embedding.

        Returns 0.0 (perfect match) to 1.0 (complete divergence).
        """
        latest = self.get_latest_snapshot(character_id)
        if not latest or not latest.appearance_embedding:
            return 0.0

        ref = latest.appearance_embedding
        curr = current_embedding

        if len(ref) != len(curr) or len(ref) == 0:
            return 0.5

        # Compute cosine similarity
        dot = sum(a * b for a, b in zip(ref, curr))
        norm_ref = math.sqrt(sum(a * a for a in ref))
        norm_curr = math.sqrt(sum(b * b for b in curr))

        if norm_ref == 0 or norm_curr == 0:
            return 0.5

        cosine_sim = dot / (norm_ref * norm_curr)
        # Bound between -1.0 and 1.0
        cosine_sim = max(-1.0, min(1.0, cosine_sim))

        # Convert similarity (1.0 -> 0.0 drift, -1.0 -> 1.0 drift)
        drift = (1.0 - cosine_sim) / 2.0
        return round(drift, 4)


character_appearance_memory = CharacterAppearanceMemory()
