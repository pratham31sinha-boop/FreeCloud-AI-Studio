"""Tests for Continuity Engine Modules and Interfaces."""

from backend.continuity.audio_sync import audio_sync_engine
from backend.continuity.base import (
    AppearanceSnapshot,
    CharacterProfile,
    MotionSequence,
    SceneState,
)
from backend.continuity.character_memory import character_appearance_memory
from backend.continuity.character_refs import character_ref_manager
from backend.continuity.motion_refs import motion_action_manager
from backend.continuity.quality_check import quality_checker
from backend.continuity.scene_history import scene_history_tracker
from backend.continuity.story_continuity import story_continuity_engine
from backend.continuity.voice_refs import voice_reference_manager


def test_character_reference_presets():
    """Verify Jagdhatri serial character reference presets are correctly seeded."""
    jagdhatri = character_ref_manager.get_character("jagdhatri_main")
    assert jagdhatri is not None
    assert "JAS" in jagdhatri.name
    assert "tactical" in jagdhatri.physical_traits["hair"].lower()
    assert jagdhatri.lora_trigger == "jagdhatri_anime_v1"

    jash = character_ref_manager.get_character("jash_lead")
    assert jash is not None
    assert "Swayambhu" in jash.name


def test_character_appearance_drift():
    """Verify visual drift calculation using embedding cosine distance."""
    char_id = "test_char_drift"
    base_embedding = [1.0, 0.0, 0.0, 0.0]

    snapshot = AppearanceSnapshot(
        character_id=char_id,
        scene_index=0,
        timestamp_sec=0.0,
        current_clothing="Tactical Navy",
        facial_expression="Determined",
        appearance_embedding=base_embedding,
    )
    character_appearance_memory.record_snapshot(snapshot)

    # Identical embedding should result in drift 0.0
    drift_identical = character_appearance_memory.check_drift(char_id, [1.0, 0.0, 0.0, 0.0])
    assert drift_identical == 0.0

    # Orthogonal embedding should produce moderate drift (~0.5)
    drift_orthogonal = character_appearance_memory.check_drift(char_id, [0.0, 1.0, 0.0, 0.0])
    assert drift_orthogonal == 0.5

    # Opposing vector should produce maximum drift (1.0)
    drift_opposing = character_appearance_memory.check_drift(char_id, [-1.0, 0.0, 0.0, 0.0])
    assert drift_opposing == 1.0


def test_motion_action_references():
    """Verify registration and lookup of action sequences."""
    motion = MotionSequence(
        motion_id="test_martial_arts_01",
        scene_index=4,
        start_sec=60.0,
        end_sec=75.0,
        action_type="high_kick_and_parry",
        camera_motion="arc_shot_around_character",
    )
    motion_action_manager.register_motion(motion)

    retrieved = motion_action_manager.get_motion_for_scene(4)
    assert retrieved is not None
    assert retrieved.action_type == "high_kick_and_parry"


def test_scene_history_lighting():
    """Verify environment lighting continuity checks."""
    s1 = SceneState(
        scene_index=1,
        location="Warehouse Interior",
        time_of_day="night",
        lighting_palette=["#0f172a", "#1e293b", "#38bdf8"],
    )
    s2 = SceneState(
        scene_index=2,
        location="Warehouse Interior",
        time_of_day="night",
        lighting_palette=["#0f172a", "#64748b"],
    )
    scene_history_tracker.record_scene_state(s1)
    scene_history_tracker.record_scene_state(s2)

    assert scene_history_tracker.check_lighting_continuity(1, 2) is True


def test_story_continuity_parsing():
    """Verify script deconstruction into narrative beats."""
    script = (
        "Jagdhatri: Stop right there!\n"
        "Jash: The perimeter is locked.\n"
        "A sudden explosion rocks the east wing."
    )
    beats = story_continuity_engine.analyze_script(script)
    assert len(beats) == 3
    assert beats[0].speaking_character_id == "jagdhatri"
    assert beats[0].dialogue == "Stop right there!"
    assert beats[1].speaking_character_id == "jash"
    assert beats[2].speaking_character_id is None


def test_audio_sync_planning():
    """Verify lip-sync alignment and viseme timeline generation."""
    plan = audio_sync_engine.plan_synchronization(
        dialogue="Hello Jagdhatri",
        character_id="jash_lead",
        start_sec=10.0,
        duration_sec=2.0,
    )
    assert plan.character_id == "jash_lead"
    assert len(plan.viseme_sequence) == 2
    assert plan.sync_confidence > 0.8


def test_quality_checker():
    """Verify quality scoring and threshold alerts."""
    # Good metrics
    report_pass = quality_checker.evaluate_job(
        "job_good",
        [
            {"identity_score": 0.88, "motion_score": 0.85, "flicker_score": 0.10, "audio_sync_score": 0.90}
        ],
    )
    assert report_pass.passed is True
    assert len(report_pass.issues_detected) == 0

    # Low identity score metrics
    report_fail = quality_checker.evaluate_job(
        "job_low_id",
        [
            {"identity_score": 0.55, "motion_score": 0.85, "flicker_score": 0.10, "audio_sync_score": 0.90}
        ],
    )
    assert report_fail.passed is False
    assert any("Identity consistency" in issue for issue in report_fail.issues_detected)
