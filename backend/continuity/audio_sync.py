"""Audio and Dialogue Synchronization component."""

from __future__ import annotations

from typing import Any, Dict, List
from backend.continuity.base import AudioSyncPlan, IAudioSyncEngine


class AudioSyncEngine(IAudioSyncEngine):
    """Calculates phoneme/viseme timing and generates alignment plans for lip-sync."""

    def plan_synchronization(
        self, dialogue: str, character_id: str, start_sec: float, duration_sec: float
    ) -> AudioSyncPlan:
        """Create a placeholder viseme timeline aligned with dialogue text."""
        words = dialogue.split()
        visemes: List[Dict[str, Any]] = []

        if words:
            step = duration_sec / len(words)
            for idx, word in enumerate(words):
                t = start_sec + (idx * step)
                viseme_shape = self._approximate_viseme(word)
                visemes.append(
                    {
                        "time_sec": round(t, 2),
                        "word": word,
                        "viseme": viseme_shape,
                        "mouth_openness": 0.6 if viseme_shape in ("AA", "O") else 0.3,
                    }
                )

        return AudioSyncPlan(
            character_id=character_id,
            dialogue=dialogue,
            start_sec=start_sec,
            duration_sec=duration_sec,
            viseme_sequence=visemes,
            sync_confidence=0.88 if words else 1.0,
        )

    def _approximate_viseme(self, word: str) -> str:
        w = word.lower()
        if any(c in w for c in ("o", "u")):
            return "O"
        if any(c in w for c in ("a", "e")):
            return "AA"
        if any(c in w for c in ("m", "b", "p")):
            return "MBP"
        return "REST"


audio_sync_engine = AudioSyncEngine()
