"""Scene History Tracker component for the Continuity Engine."""

from __future__ import annotations

from typing import Dict, List, Optional
from backend.continuity.base import ISceneHistoryTracker, SceneState


class SceneHistoryTracker(ISceneHistoryTracker):
    """Maintains sequential environment, lighting, and timeline history."""

    def __init__(self):
        self._scenes: Dict[int, SceneState] = {}

    def record_scene_state(self, state: SceneState) -> None:
        self._scenes[state.scene_index] = state

    def get_scene_state(self, scene_index: int) -> Optional[SceneState]:
        return self._scenes.get(scene_index)

    def list_scenes(self) -> List[SceneState]:
        return [self._scenes[k] for k in sorted(self._scenes.keys())]

    def check_lighting_continuity(
        self, prev_scene_idx: int, curr_scene_idx: int
    ) -> bool:
        """Verify whether lighting palette transitions logically between cuts."""
        prev = self.get_scene_state(prev_scene_idx)
        curr = self.get_scene_state(curr_scene_idx)
        if not prev or not curr:
            return True
        # If same location and time of day, palettes should share common base
        if prev.location == curr.location and prev.time_of_day == curr.time_of_day:
            overlap = set(prev.lighting_palette).intersection(set(curr.lighting_palette))
            return len(overlap) > 0
        return True


scene_history_tracker = SceneHistoryTracker()
