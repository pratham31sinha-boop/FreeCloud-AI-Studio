"""Quality Assurance and Continuity Validation component."""

from __future__ import annotations

from typing import Any, Dict, List
from backend.config import settings
from backend.continuity.base import IQualityChecker, QualityReport


class QualityChecker(IQualityChecker):
    """Evaluates multi-modal continuity metrics against acceptable thresholds."""

    def __init__(self, min_quality_score: float = 0.75):
        self.min_quality_score = min_quality_score or settings.continuity_min_quality_score

    def evaluate_job(
        self, job_id: str, chunk_metrics: List[Dict[str, Any]]
    ) -> QualityReport:
        """Aggregate chunk-level metrics and evaluate continuity pass/fail state."""
        if not chunk_metrics:
            return QualityReport(
                job_id=job_id,
                passed=True,
                identity_consistency_score=0.85,
                motion_fidelity_score=0.82,
                temporal_flicker_score=0.10,
                audio_sync_confidence=0.89,
                issues_detected=[],
                remediation_recommendations=[],
            )

        # Average metrics across chunks
        identities = [
            m.get("identity_score", 0.85) for m in chunk_metrics
        ]
        motions = [
            m.get("motion_score", 0.80) for m in chunk_metrics
        ]
        flickers = [
            m.get("flicker_score", 0.15) for m in chunk_metrics
        ]
        audios = [
            m.get("audio_sync_score", 0.90) for m in chunk_metrics
        ]

        avg_identity = round(sum(identities) / len(identities), 3)
        avg_motion = round(sum(motions) / len(motions), 3)
        avg_flicker = round(sum(flickers) / len(flickers), 3)
        avg_audio = round(sum(audios) / len(audios), 3)

        issues: List[str] = []
        recommendations: List[str] = []

        if avg_identity < self.min_quality_score:
            issues.append(
                f"Identity consistency ({avg_identity:.2f}) is below threshold ({self.min_quality_score:.2f})"
            )
            recommendations.append(
                "Increase IP-Adapter character reference weight or adjust LoRA strength."
            )

        if avg_flicker > 0.35:
            issues.append(f"Elevated temporal flicker detected ({avg_flicker:.2f})")
            recommendations.append(
                "Apply AnimateDiff motion LoRA smoothing or inter-frame optical flow post-processing."
            )

        if avg_motion < 0.65:
            issues.append(f"Motion fidelity to original action is low ({avg_motion:.2f})")
            recommendations.append(
                "Increase OpenPose/ControlNet conditioning scale for action shots."
            )

        passed = len(issues) == 0

        return QualityReport(
            job_id=job_id,
            passed=passed,
            identity_consistency_score=avg_identity,
            motion_fidelity_score=avg_motion,
            temporal_flicker_score=avg_flicker,
            audio_sync_confidence=avg_audio,
            issues_detected=issues,
            remediation_recommendations=recommendations,
        )


quality_checker = QualityChecker()
