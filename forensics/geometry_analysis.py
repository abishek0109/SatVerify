"""Branch G — Geometry and Spatial Coherence Analysis Module.

Detects local geometric inconsistencies, seam-carving signatures, patch-boundary
clipping, and spatial color-gradient discontinuities.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from forensics.ela import colorize_grayscale
from utils.logging_config import get_logger

logger = get_logger("forensics.geometry")


class GeometryCoherenceAnalyzer:
    """Evaluates spatial coherence, color-channel alignments, and seam discontinuities."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _channel_correlation(self, rgb_arr: np.ndarray) -> Tuple[float, float, float]:
        """Compute Pearson correlation coefficients between color channels (R-G, G-B, R-B).

        Natural sensors exhibit high cross-channel correlation. In synthetic or green-boosted
        tampered imagery, green channel is often artificially decoupled from red and blue.
        """
        r = rgb_arr[:, :, 0].ravel().astype(np.float64)
        g = rgb_arr[:, :, 1].ravel().astype(np.float64)
        b = rgb_arr[:, :, 2].ravel().astype(np.float64)

        def corr(x, y):
            std_x, std_y = np.std(x), np.std(y)
            if std_x < 1e-4 or std_y < 1e-4:
                return 1.0
            return float(np.corrcoef(x, y)[0, 1])

        rg = corr(r, g)
        gb = corr(g, b)
        rb = corr(r, b)
        return rg, gb, rb

    def _patch_seam_detection(self, gray: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect rectangular or straight patch boundaries from horizontal and vertical grid profiles."""
        h, w = gray.shape
        # Horizontal and vertical differential profiles
        diff_y = np.abs(gray[1:, :] - gray[:-1, :])
        diff_x = np.abs(gray[:, 1:] - gray[:, :-1])

        # Row-wise and column-wise mean step-gradient
        row_steps = np.mean(diff_y, axis=1)
        col_steps = np.mean(diff_x, axis=0)

        # Check for isolated sharp peak lines (hallmark of rectangular patch pasting)
        def peak_ratio(steps: np.ndarray) -> float:
            if len(steps) < 4:
                return 0.0
            med = float(np.median(steps)) + 1e-4
            p99 = float(np.percentile(steps, 99))
            return p99 / med

        row_spike = peak_ratio(row_steps)
        col_spike = peak_ratio(col_steps)
        seam_score = max(row_spike, col_spike)

        # Create 2D boundary map
        seam_map = np.zeros((h, w), dtype=np.float32)
        seam_map[1:, :] += diff_y
        seam_map[:, 1:] += diff_x
        return seam_score, seam_map

    def analyze(
        self, image_input: Union[Image.Image, np.ndarray, str]
    ) -> ForensicResult:
        """Run geometry and spatial coherence analysis."""
        try:
            if isinstance(image_input, Image.Image):
                rgb_pil = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                rgb_pil = Image.fromarray(image_input).convert("RGB")
            else:
                with Image.open(str(image_input)) as img:
                    rgb_pil = img.convert("RGB")

            rgb_arr = np.asarray(rgb_pil, dtype=np.uint8)
            gray = (
                0.299 * rgb_arr[:, :, 0]
                + 0.587 * rgb_arr[:, :, 1]
                + 0.114 * rgb_arr[:, :, 2]
            ).astype(np.float32)

            h, w = gray.shape
            if h < 16 or w < 16:
                return ForensicResult(
                    branch_id="geometry",
                    branch_name="Geometry and Spatial Coherence Analysis",
                    status="ERROR",
                    score=None,
                    explanation="Image too small for geometric analysis.",
                    method_description="Seam and spectral cross-channel coherence.",
                )

            # 1. Color channel coherence
            rg_corr, gb_corr, rb_corr = self._channel_correlation(rgb_arr)
            # Artificial green-boosting lowers correlation between Green and Red/Blue
            min_corr = min(rg_corr, gb_corr, rb_corr)
            channel_decorrelation = max(0.0, 1.0 - min_corr)

            # 2. Patch seam detection
            seam_spike, seam_map = self._patch_seam_detection(gray)

            # 3. Heuristic Scoring
            # Seam spike > 2.5 indicates strong horizontal/vertical boundary lines
            # Channel decorrelation > 0.35 indicates abnormal color manipulation
            norm_seam = min(max(seam_spike - 1.5, 0.0) / 3.0, 1.0)
            norm_decorr = min(channel_decorrelation / 0.40, 1.0)

            raw_score = 0.55 * norm_seam + 0.45 * norm_decorr
            suspiciousness_score = float(1.0 / (1.0 + np.exp(-7.0 * (raw_score - 0.35))))
            suspiciousness_score = round(min(max(suspiciousness_score, 0.0), 1.0), 3)

            # 4. Visualization
            norm_seam_map = (
                (seam_map / max(float(np.percentile(seam_map, 98)), 1e-4)) * 255.0
            )
            norm_seam_map = np.clip(norm_seam_map, 0, 255).astype(np.uint8)
            viz_rgb = colorize_grayscale(norm_seam_map, "viridis")

            # 5. Explanation
            if suspiciousness_score >= 0.62:
                explanation = (
                    f"Spatial coherence anomalies detected: seam step ratio={seam_spike:.2f}, "
                    f"min channel corr={min_corr:.2f}. Indicates straight rectangular splicing seams "
                    f"or artificial green-channel coloration."
                )
            elif suspiciousness_score <= 0.38:
                explanation = (
                    f"Normal spatial coherence: cross-channel correlation is robust (R-G={rg_corr:.2f}, G-B={gb_corr:.2f}) "
                    f"with no significant linear rectangular seam artifacts."
                )
            else:
                explanation = (
                    f"Moderate spatial coherence (seam ratio={seam_spike:.2f}, min corr={min_corr:.2f}). "
                    f"Linear field boundaries or high-contrast roads may elevate boundary metrics."
                )

            return ForensicResult(
                branch_id="geometry",
                branch_name="Geometry and Spatial Coherence Analysis",
                status="AVAILABLE",
                score=suspiciousness_score,
                explanation=explanation,
                method_description=(
                    "Analyzes spatial seam step-gradients across rows/columns and evaluates cross-channel "
                    "correlation (R, G, B) to identify splicing seams and decoupled color manipulation."
                ),
                limitations=(
                    "Regular orthorectification crop boundaries or survey grid lines in genuine products "
                    "can produce linear step artifacts."
                ),
                metrics={
                    "rg_correlation": round(rg_corr, 3),
                    "gb_correlation": round(gb_corr, 3),
                    "rb_correlation": round(rb_corr, 3),
                    "seam_step_spike_ratio": round(seam_spike, 2),
                },
                visualization=viz_rgb,
            )

        except Exception as exc:
            logger.exception("Geometry analysis failed: %s", exc)
            return ForensicResult(
                branch_id="geometry",
                branch_name="Geometry and Spatial Coherence Analysis",
                status="ERROR",
                score=None,
                explanation=f"Error during geometry analysis: {exc}",
                method_description="Spatial coherence and color correlation analysis.",
                limitations="Requires multi-channel image.",
            )
