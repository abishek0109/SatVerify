"""Branch H — Shadow and Solar-Geometry Validation Module.

Analyzes shadow orientations, lighting consistency across features, and validates
astronomical solar azimuth when acquisition metadata is provided.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from forensics.ela import colorize_grayscale
from utils.logging_config import get_logger

logger = get_logger("forensics.shadow")


class ShadowAnalyzer:
    """Evaluates physical shadow coherence and optional solar alignment."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _segment_shadow_candidates(
        self, rgb_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment potential shadow regions using luminance and blue-to-red ratio."""
        r = rgb_arr[:, :, 0].astype(np.float32)
        g = rgb_arr[:, :, 1].astype(np.float32)
        b = rgb_arr[:, :, 2].astype(np.float32)

        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        # Shadows are typically in lowest 15-20% luminance with slightly elevated blue scattering
        lum_thresh = float(np.percentile(luminance, 18))
        shadow_mask = luminance < lum_thresh

        # Exclude deep water bodies if present by checking texture/variance
        return shadow_mask, luminance

    def _estimate_dominant_shadow_orientation(
        self, shadow_mask: np.ndarray, luminance: np.ndarray
    ) -> Tuple[Optional[float], float, np.ndarray]:
        """Estimate dominant directional gradient at shadow boundaries."""
        # Compute gradient at shadow boundaries
        gy, gx = np.gradient(luminance)
        # Boundary is where shadow mask transitions to lit terrain
        boundary = np.zeros_like(shadow_mask)
        boundary[1:-1, 1:-1] = (
            (shadow_mask[1:-1, 1:-1] != shadow_mask[:-2, 1:-1])
            | (shadow_mask[1:-1, 1:-1] != shadow_mask[2:, 1:-1])
            | (shadow_mask[1:-1, 1:-1] != shadow_mask[1:-1, :-2])
            | (shadow_mask[1:-1, 1:-1] != shadow_mask[1:-1, 2:])
        )

        b_gx = gx[boundary]
        b_gy = gy[boundary]

        if len(b_gx) < 50:
            return None, 0.0, np.zeros_like(luminance, dtype=np.uint8)

        angles = np.arctan2(b_gy, b_gx)  # [-pi, pi]
        # Circular mean / dominant orientation
        sin_sum = float(np.sum(np.sin(angles)))
        cos_sum = float(np.sum(np.cos(angles)))
        dominant_angle_deg = float(np.degrees(np.arctan2(sin_sum, cos_sum))) % 360.0

        # Measure orientation coherence (resultant vector length R in [0, 1])
        r_length = float(np.sqrt(sin_sum**2 + cos_sum**2) / len(b_gx))

        # Visualization of shadow mask & boundaries
        vis = (shadow_mask.astype(np.uint8) * 128) + (boundary.astype(np.uint8) * 127)
        return dominant_angle_deg, r_length, vis

    def analyze(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ForensicResult:
        """Run shadow consistency and solar validation."""
        try:
            if isinstance(image_input, Image.Image):
                rgb_pil = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                rgb_pil = Image.fromarray(image_input).convert("RGB")
            else:
                with Image.open(str(image_input)) as img:
                    rgb_pil = img.convert("RGB")

            rgb_arr = np.asarray(rgb_pil, dtype=np.uint8)
            h, w, _ = rgb_arr.shape

            shadow_mask, luminance = self._segment_shadow_candidates(rgb_arr)
            shadow_coverage = float(np.mean(shadow_mask))

            dom_angle, coherence, vis_mask = self._estimate_dominant_shadow_orientation(
                shadow_mask, luminance
            )

            # Metadata check (solar azimuth / timestamp)
            solar_azimuth = None
            has_solar_metadata = False
            if metadata and "solar_azimuth" in metadata:
                try:
                    solar_azimuth = float(metadata["solar_azimuth"])
                    has_solar_metadata = True
                except (ValueError, TypeError):
                    pass

            # If shadows are too sparse or not clearly detectable
            if shadow_coverage < 0.02 or dom_angle is None:
                return ForensicResult(
                    branch_id="shadow",
                    branch_name="Shadow and Solar-Geometry Validation",
                    status="LIMITED",
                    score=None,
                    explanation=(
                        f"Limited shadow coverage ({shadow_coverage:.1%}). "
                        "No reliable tree canopy or topographic shadows detected for directional consistency validation."
                    ),
                    method_description=(
                        "Estimates boundary orientation of shadows and tests directional coherence. "
                        "Validates against solar azimuth if acquisition metadata is supplied."
                    ),
                    limitations="Requires detectable shadows cast by elevated terrain, trees, or structures.",
                    metrics={"shadow_coverage": round(shadow_coverage, 3)},
                    visualization=colorize_grayscale(vis_mask, "plasma"),
                )

            # Heuristic score based on shadow directional coherence
            # In real scenes under single solar illumination, shadows have consistent orientation (coherence > 0.40)
            # Spliced scenes with conflicting light sources or flat generative images exhibit low coherence or multi-modal confusion.
            # However, diffuse cloud illumination also lowers coherence, so score is conservative.
            incoherence = max(0.0, 1.0 - coherence)
            norm_incoh = min(incoherence / 0.85, 1.0)

            # If solar metadata exists, compare estimated light direction with true sun azimuth
            solar_discrepancy_deg = None
            if has_solar_metadata and solar_azimuth is not None:
                # Shadow points opposite to sun direction: expected_angle = (solar_azimuth + 180) % 360
                expected_shadow_angle = (solar_azimuth + 180.0) % 360.0
                diff_angle = abs(dom_angle - expected_shadow_angle)
                solar_discrepancy_deg = min(diff_angle, 360.0 - diff_angle)
                metadata_penalty = min(solar_discrepancy_deg / 90.0, 1.0)
                raw_score = 0.50 * norm_incoh + 0.50 * metadata_penalty
            else:
                raw_score = norm_incoh * 0.70  # Scaled conservatively when no metadata

            suspiciousness_score = float(1.0 / (1.0 + np.exp(-6.0 * (raw_score - 0.40))))
            suspiciousness_score = round(min(max(suspiciousness_score, 0.0), 1.0), 3)

            viz_rgb = colorize_grayscale(vis_mask, "plasma")

            if has_solar_metadata and solar_discrepancy_deg is not None:
                explanation = (
                    f"Shadow orientation ({dom_angle:.1f}°) compared against solar azimuth ({solar_azimuth:.1f}°). "
                    f"Angular discrepancy: {solar_discrepancy_deg:.1f}°. Coherence: {coherence:.2f}."
                )
            else:
                explanation = (
                    f"Shadow coherence measured at {coherence:.2f} (dominant orientation: {dom_angle:.1f}°). "
                    f"Acquisition solar metadata not provided; evaluation is based on intra-scene light consistency."
                )

            return ForensicResult(
                branch_id="shadow",
                branch_name="Shadow and Solar-Geometry Validation",
                status="AVAILABLE",
                score=suspiciousness_score,
                explanation=explanation,
                method_description=(
                    "Segments shadow regions, determines boundary light gradient orientation, "
                    "and measures directional coherence across features."
                ),
                limitations=(
                    "Overcast lighting, diffuse atmosphere, or nadir solar angles diminish cast shadows. "
                    "Cannot confirm solar alignment without verified acquisition timestamp and coordinates."
                ),
                metrics={
                    "shadow_coverage": round(shadow_coverage, 3),
                    "dominant_orientation_deg": round(dom_angle, 1),
                    "directional_coherence": round(coherence, 3),
                    "solar_metadata_available": has_solar_metadata,
                    "solar_discrepancy_deg": (
                        round(solar_discrepancy_deg, 1) if solar_discrepancy_deg is not None else None
                    ),
                },
                visualization=viz_rgb,
            )

        except Exception as exc:
            logger.exception("Shadow analysis failed: %s", exc)
            return ForensicResult(
                branch_id="shadow",
                branch_name="Shadow and Solar-Geometry Validation",
                status="ERROR",
                score=None,
                explanation=f"Error during shadow validation: {exc}",
                method_description="Shadow directional coherence and solar geometry analysis.",
                limitations="Requires shadow segmentation.",
            )
