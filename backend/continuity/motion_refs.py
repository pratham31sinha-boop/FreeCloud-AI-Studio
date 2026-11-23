"""Motion and Action Reference component.

Preserves original movement sequences, martial arts action choreography,
and camera trajectories for Jagdhatri action scenes.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from backend.continuity.base import IMotionActionReferenceManager, MotionSequence


class MotionActionReferenceManager(IMotionActionReferenceManager):
    """Manages motion tracking sequences, action labels, and camera kinematics."""

    def __init__(self):
        self._motions: Dict[str, MotionSequence] = {}
        self._scene_motion_map: Dict[int, str] = {}
        self._seed_jagdhatri_action_sequences()

    def register_motion(self, motion: MotionSequence) -> None:
        self._motions[motion.motion_id] = motion
        self._scene_motion_map[motion.scene_index] = motion.motion_id

    def get_motion(self, motion_id: str) -> Optional[MotionSequence]:
        return self._motions.get(motion_id)

    def get_motion_for_scene(self, scene_index: int) -> Optional[MotionSequence]:
        motion_id = self._scene_motion_map.get(scene_index)
        if motion_id:
            return self._motions.get(motion_id)
        return None

    def list_motions(self) -> List[MotionSequence]:
        return list(self._motions.values())

    def _seed_jagdhatri_action_sequences(self) -> None:
        """Seed choreography profiles based on signature Jagdhatri serial action beats."""
        sequences = [
            MotionSequence(
                motion_id="jagdhatri_action_disarm_01",
                scene_index=0,
                start_sec=0.0,
                end_sec=15.0,
                action_type="martial_arts_disarm",
                camera_motion="dynamic_dolly_tracking",
                pose_keypoints_url="/data/characters/motions/jagdhatri_disarm.json",
                intensity_score=0.85,
            ),
            MotionSequence(
                motion_id="jagdhatri_action_counter_02",
                scene_index=1,
                start_sec=15.0,
                end_sec=30.0,
                action_type="rapid_block_and_throw",
                camera_motion="low_angle_whip_pan",
                pose_keypoints_url="/data/characters/motions/jagdhatri_counter.json",
                intensity_score=0.90,
            ),
            MotionSequence(
                motion_id="jash_pursuit_03",
                scene_index=2,
                start_sec=30.0,
                end_sec=45.0,
                action_type="tactical_sprint_and_cover",
                camera_motion="side_tracking_steadicam",
                pose_keypoints_url="/data/characters/motions/jash_pursuit.json",
                intensity_score=0.75,
            ),
        ]
        for seq in sequences:
            self.register_motion(seq)


motion_action_manager = MotionActionReferenceManager()
