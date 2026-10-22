"""Decision Engine for SatVerify.

Classifies satellite imagery into REAL, FAKE, or INCONCLUSIVE outcomes
based on fused suspicion scores, uncertainty intervals, and evidence sufficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import get_settings
from core.score_fusion import FusionResult
from utils.logging_config import get_logger

logger = get_logger("core.decision")


@dataclass
class DecisionResult:
    """Final decision outcome and classification metadata."""

    verdict: str  # "REAL", "FAKE", or "INCONCLUSIVE"
    headline: str
    confidence_percentage: float
    fused_score: Optional[float]
    lower_threshold: float
    upper_threshold: float
    reason: str
    disclaimer: str
    has_conflicting_evidence: bool = False

    @property
    def badge_color(self) -> str:
        """UI badge color for the verdict."""
        if self.verdict == "REAL":
            return "#00C853"  # Vibrant forest green
        elif self.verdict == "FAKE":
            return "#D50000"  # Alert red
        return "#FFAB00"  # Amber for inconclusive / uncertain


class DecisionEngine:
    """Evaluates fused scores against uncertainty thresholds to produce tri-state verdicts."""

    def __init__(
        self,
        real_threshold: Optional[float] = None,
        fake_threshold: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.real_threshold = real_threshold or settings.DECISION_REAL_THRESHOLD
        self.fake_threshold = fake_threshold or settings.DECISION_FAKE_THRESHOLD

        if self.real_threshold >= self.fake_threshold:
            raise ValueError("real_threshold must be strictly less than fake_threshold.")

    def evaluate(self, fusion: FusionResult) -> DecisionResult:
        """Evaluate fusion output and determine the final authenticity verdict."""
        disclaimer = (
            "Automated multi-branch forensic assessment. This tool identifies mathematical and visual anomalies "
            "consistent with digital manipulation or synthetic generation. A REAL classification indicates absence "
            "of detected forensic anomalies, not legal or certified proof of biological carbon sequestration."
        )

        # 1. Check evidence sufficiency
        if not fusion.is_sufficient_evidence or fusion.fused_score is None:
            return DecisionResult(
                verdict="INCONCLUSIVE",
                headline="INCONCLUSIVE — Insufficient Forensic Evidence",
                confidence_percentage=50.0,
                fused_score=None,
                lower_threshold=self.real_threshold,
                upper_threshold=self.fake_threshold,
                reason=(
                    f"Evidence is insufficient ({fusion.available_branches_count} branch(es) available). "
                    "Cannot establish a reliable decision."
                ),
                disclaimer=disclaimer,
                has_conflicting_evidence=False,
            )

        score = fusion.fused_score

        # 2. Check for conflicting evidence among available branches
        valid_scores = [
            b.raw_score
            for b in fusion.branch_contributions
            if b.raw_score is not None
        ]
        has_conflict = False
        if len(valid_scores) >= 2:
            max_s, min_s = max(valid_scores), min(valid_scores)
            if max_s >= 0.75 and min_s <= 0.25:
                has_conflict = True

        # 3. Decision threshold comparison
        if score < self.real_threshold:
            confidence = round((1.0 - score) * 100.0, 1)
            return DecisionResult(
                verdict="REAL",
                headline="REAL — Authentic Terrain Candidate",
                confidence_percentage=confidence,
                fused_score=score,
                lower_threshold=self.real_threshold,
                upper_threshold=self.fake_threshold,
                reason=(
                    f"Fused suspicion score ({score:.3f}) is below decision threshold ({self.real_threshold:.2f}). "
                    "Forensic signals across compression, frequency spectrum, and micro-texture exhibit uniform, "
                    "natural patterns characteristic of authentic satellite sensor acquisitions."
                ),
                disclaimer=disclaimer,
                has_conflicting_evidence=has_conflict,
            )

        elif score >= self.fake_threshold:
            confidence = round(score * 100.0, 1)
            return DecisionResult(
                verdict="FAKE",
                headline="FAKE — Altered / Synthetic Image Candidate",
                confidence_percentage=confidence,
                fused_score=score,
                lower_threshold=self.real_threshold,
                upper_threshold=self.fake_threshold,
                reason=(
                    f"Fused suspicion score ({score:.3f}) exceeds or meets decision threshold ({self.fake_threshold:.2f}). "
                    "Significant forensic anomalies detected across multiple analytical branches, such as uneven "
                    "compression residuals, periodic spectral spikes, or boundary discontinuities."
                ),
                disclaimer=disclaimer,
                has_conflicting_evidence=has_conflict,
            )

        else:
            # Score falls within the uncertainty interval [real_threshold, fake_threshold)
            return DecisionResult(
                verdict="INCONCLUSIVE",
                headline="INCONCLUSIVE — Ambiguous / Conflicting Evidence",
                confidence_percentage=50.0,
                fused_score=score,
                lower_threshold=self.real_threshold,
                upper_threshold=self.fake_threshold,
                reason=(
                    f"Fused score ({score:.3f}) lies in the uncertainty margin "
                    f"[{self.real_threshold:.2f}, {self.fake_threshold:.2f}). "
                    + (
                        "Conflicting indicators detected between forensic branches; "
                        if has_conflict
                        else "Evidence is moderate and inconclusive; "
                    )
                    + "manual auditor verification and ground-truth validation are strongly recommended."
                ),
                disclaimer=disclaimer,
                has_conflicting_evidence=has_conflict,
            )
