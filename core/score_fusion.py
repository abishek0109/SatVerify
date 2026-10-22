"""Multi-branch weighted score fusion engine for SatVerify.

Dynamically renormalizes branch weights across available forensic signals,
prevents uncalibrated models from distorting decisions, and calculates
individual branch contributions to the final suspicion score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import get_settings
from forensics.base import ForensicResult
from utils.logging_config import get_logger

logger = get_logger("core.score_fusion")


@dataclass
class BranchContribution:
    """Breakdown of an individual forensic branch's role in the fusion score."""

    branch_id: str
    branch_name: str
    status: str
    raw_score: Optional[float]
    base_weight: float
    effective_weight: float
    weighted_contribution: float
    explanation: str


@dataclass
class FusionResult:
    """Complete output of the multi-branch score fusion process."""

    fused_score: Optional[float]  # [0.0, 1.0], None if insufficient evidence
    available_branches_count: int
    total_branches_count: int
    branch_contributions: List[BranchContribution] = field(default_factory=list)
    effective_weights: Dict[str, float] = field(default_factory=dict)
    summary_message: str = ""
    is_sufficient_evidence: bool = False


class WeightedScoreFuser:
    """Fuses heterogeneous forensic branch signals via dynamic re-normalized weighting."""

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None) -> None:
        self.settings = get_settings()
        self.base_weights = custom_weights or self.settings.BRANCH_WEIGHTS.copy()
        # Ensure base weights sum to 1.0
        total_base = sum(self.base_weights.values())
        if total_base > 0:
            self.base_weights = {k: v / total_base for k, v in self.base_weights.items()}

    def _map_branch_results(
        self, branch_results: List[ForensicResult]
    ) -> Dict[str, ForensicResult]:
        """Map individual forensic results into the presentation's fusion categories.

        Architecture categories:
        1. 'ela' -> ELA (22%)
        2. 'fft' -> FFT (22%)
        3. 'vit' -> ViT / ResNet50 (18%)
        4. 'structural' -> Structural Consistency (16%)
        5. 'texture_geometry' -> Mean of Texture & Geometry (12%)
        6. 'shadow_elevation' -> Mean of Shadow & Elevation (10%)
        """
        by_id: Dict[str, ForensicResult] = {r.branch_id: r for r in branch_results}
        mapped: Dict[str, Tuple[str, Optional[float], str, str]] = {}

        # 1. ELA
        ela_res = by_id.get("ela")
        if ela_res and ela_res.is_available:
            mapped["ela"] = (ela_res.branch_name, ela_res.score, "AVAILABLE", ela_res.explanation)
        else:
            mapped["ela"] = (
                "Error Level Analysis (ELA)",
                None,
                ela_res.status if ela_res else "UNAVAILABLE",
                ela_res.explanation if ela_res else "Branch not run.",
            )

        # 2. FFT
        fft_res = by_id.get("fft")
        if fft_res and fft_res.is_available:
            mapped["fft"] = (fft_res.branch_name, fft_res.score, "AVAILABLE", fft_res.explanation)
        else:
            mapped["fft"] = (
                "Fourier Frequency Analysis (FFT)",
                None,
                fft_res.status if fft_res else "UNAVAILABLE",
                fft_res.explanation if fft_res else "Branch not run.",
            )

        # 3. ViT / ResNet deep learning
        # Use whichever trained model is available (prefer ViT or ResNet)
        vit_res = by_id.get("vit")
        resnet_res = by_id.get("resnet50")
        dl_res = None
        if vit_res and vit_res.is_available:
            dl_res = vit_res
        elif resnet_res and resnet_res.is_available:
            dl_res = resnet_res

        if dl_res and dl_res.is_available:
            mapped["vit"] = (dl_res.branch_name, dl_res.score, "AVAILABLE", dl_res.explanation)
        else:
            mapped["vit"] = (
                "Vision Transformer / Deep Learning",
                None,
                "UNAVAILABLE",
                (
                    "Deep learning model excluded: uncalibrated ImageNet weights cannot "
                    "be used for satellite verification."
                ),
            )

        # 4. Structural Consistency
        struct_res = by_id.get("structural")
        if struct_res and struct_res.is_available:
            mapped["structural"] = (
                struct_res.branch_name,
                struct_res.score,
                "AVAILABLE",
                struct_res.explanation,
            )
        else:
            mapped["structural"] = (
                "Structural Consistency Analysis",
                None,
                struct_res.status if struct_res else "UNAVAILABLE",
                struct_res.explanation if struct_res else "Branch not run.",
            )

        # 5. Texture & Geometry
        tex_res = by_id.get("texture")
        geom_res = by_id.get("geometry")
        tg_scores = []
        if tex_res and tex_res.is_available and tex_res.score is not None:
            tg_scores.append(tex_res.score)
        if geom_res and geom_res.is_available and geom_res.score is not None:
            tg_scores.append(geom_res.score)

        if tg_scores:
            mean_tg = float(sum(tg_scores) / len(tg_scores))
            mapped["texture_geometry"] = (
                "Texture & Geometry Analysis",
                mean_tg,
                "AVAILABLE",
                f"Combined LBP micro-texture and spatial coherence signals ({len(tg_scores)} sub-branches).",
            )
        else:
            mapped["texture_geometry"] = (
                "Texture & Geometry Analysis",
                None,
                "UNAVAILABLE",
                "Texture and geometry sub-branches unavailable.",
            )

        # 6. Optional Shadow & Elevation
        sh_res = by_id.get("shadow")
        elev_res = by_id.get("elevation")
        phys_scores = []
        if sh_res and sh_res.is_available and sh_res.score is not None:
            phys_scores.append(sh_res.score)
        if elev_res and elev_res.is_available and elev_res.score is not None:
            phys_scores.append(elev_res.score)

        if phys_scores:
            mean_phys = float(sum(phys_scores) / len(phys_scores))
            mapped["shadow_elevation"] = (
                "Physical Shadow & Elevation Validation",
                mean_phys,
                "AVAILABLE",
                f"Physical plausibility evaluated from {len(phys_scores)} available physical modules.",
            )
        else:
            mapped["shadow_elevation"] = (
                "Physical Shadow & Elevation Validation",
                None,
                "UNAVAILABLE",
                "Physical solar/DEM metadata not provided.",
            )

        return mapped

    def fuse(self, branch_results: List[ForensicResult]) -> FusionResult:
        """Execute weighted fusion with dynamic re-normalization."""
        mapped_categories = self._map_branch_results(branch_results)

        # Identify which categories have valid scores
        available_categories = {
            cat_id: data
            for cat_id, data in mapped_categories.items()
            if data[1] is not None
        }

        min_required = self.settings.MIN_BRANCHES_REQUIRED
        is_sufficient = len(available_categories) >= min_required

        # Handle zero or insufficient evidence
        if not is_sufficient:
            contributions = []
            for cat_id, (name, score, status, expl) in mapped_categories.items():
                base_w = self.base_weights.get(cat_id, 0.0)
                contributions.append(
                    BranchContribution(
                        branch_id=cat_id,
                        branch_name=name,
                        status=status,
                        raw_score=score,
                        base_weight=round(base_w, 3),
                        effective_weight=0.0,
                        weighted_contribution=0.0,
                        explanation=expl,
                    )
                )

            return FusionResult(
                fused_score=None,
                available_branches_count=len(available_categories),
                total_branches_count=len(mapped_categories),
                branch_contributions=contributions,
                effective_weights={},
                summary_message=(
                    f"Insufficient forensic evidence: only {len(available_categories)} "
                    f"branch(es) available (minimum {min_required} required)."
                ),
                is_sufficient_evidence=False,
            )

        # Dynamic re-normalization of weights across available branches
        # Sum base weights of available branches
        sum_available_base = sum(
            self.base_weights.get(cat_id, 0.0) for cat_id in available_categories
        )

        effective_weights: Dict[str, float] = {}
        for cat_id in available_categories:
            base_w = self.base_weights.get(cat_id, 0.0)
            effective_weights[cat_id] = (
                base_w / sum_available_base if sum_available_base > 0 else (1.0 / len(available_categories))
            )

        # Calculate final fused score
        fused_sum = 0.0
        contributions: List[BranchContribution] = []

        for cat_id, (name, score, status, expl) in mapped_categories.items():
            base_w = self.base_weights.get(cat_id, 0.0)
            if cat_id in effective_weights and score is not None:
                eff_w = effective_weights[cat_id]
                contribution = score * eff_w
                fused_sum += contribution
                contributions.append(
                    BranchContribution(
                        branch_id=cat_id,
                        branch_name=name,
                        status="AVAILABLE",
                        raw_score=round(score, 3),
                        base_weight=round(base_w, 3),
                        effective_weight=round(eff_w, 3),
                        weighted_contribution=round(contribution, 3),
                        explanation=expl,
                    )
                )
            else:
                contributions.append(
                    BranchContribution(
                        branch_id=cat_id,
                        branch_name=name,
                        status=status,
                        raw_score=None,
                        base_weight=round(base_w, 3),
                        effective_weight=0.0,
                        weighted_contribution=0.0,
                        explanation=expl,
                    )
                )

        fused_score = round(min(max(fused_sum, 0.0), 1.0), 3)

        summary_msg = (
            f"Fused {len(available_categories)}/{len(mapped_categories)} branches. "
            f"Weights dynamically renormalized to 100% across active signals."
        )

        return FusionResult(
            fused_score=fused_score,
            available_branches_count=len(available_categories),
            total_branches_count=len(mapped_categories),
            branch_contributions=contributions,
            effective_weights={k: round(v, 4) for k, v in effective_weights.items()},
            summary_message=summary_msg,
            is_sufficient_evidence=True,
        )
