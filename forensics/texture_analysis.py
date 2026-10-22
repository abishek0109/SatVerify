"""Branch E — Local Binary Pattern (LBP) Texture Analysis Module.

Analyzes micro-texture distributions, measures regional entropy, and identifies
suspiciously repeated patterns, unnatural smoothness, or tiling in vegetation canopy.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from forensics.ela import colorize_grayscale
from utils.logging_config import get_logger

logger = get_logger("forensics.texture")


class TextureAnalyzer:
    """Performs Local Binary Pattern (LBP) texture and regional entropy analysis."""

    def __init__(
        self, radius: int = 2, n_points: int = 16, method: str = "uniform"
    ) -> None:
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.settings = get_settings()

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Compute LBP using skimage if available, else vectorized NumPy fallback."""
        try:
            from skimage.feature import local_binary_pattern

            lbp = local_binary_pattern(
                gray, P=self.n_points, R=self.radius, method=self.method
            )
            return lbp.astype(np.float32)
        except Exception:
            # Vectorized 8-neighbor LBP fallback for radius=1
            h, w = gray.shape
            padded = np.pad(gray, 1, mode="reflect")
            center = gray.astype(np.int16)
            lbp = np.zeros((h, w), dtype=np.uint8)

            shifts = [
                (-1, -1, 0),
                (-1, 0, 1),
                (-1, 1, 2),
                (0, 1, 3),
                (1, 1, 4),
                (1, 0, 5),
                (1, -1, 6),
                (0, -1, 7),
            ]
            for dy, dx, bit in shifts:
                neighbor = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w].astype(np.int16)
                lbp += ((neighbor >= center).astype(np.uint8)) << bit
            return lbp.astype(np.float32)

    def analyze(
        self, image_input: Union[Image.Image, np.ndarray, str]
    ) -> ForensicResult:
        """Compute LBP texture statistics and assess canopy authenticity."""
        try:
            # 1. Grayscale representation
            if isinstance(image_input, Image.Image):
                gray = np.asarray(image_input.convert("L"), dtype=np.float32)
            elif isinstance(image_input, np.ndarray):
                if image_input.ndim == 3:
                    gray = (
                        0.299 * image_input[:, :, 0]
                        + 0.587 * image_input[:, :, 1]
                        + 0.114 * image_input[:, :, 2]
                    ).astype(np.float32)
                else:
                    gray = image_input.astype(np.float32)
            else:
                with Image.open(str(image_input)) as img:
                    gray = np.asarray(img.convert("L"), dtype=np.float32)

            h, w = gray.shape
            if h < 16 or w < 16:
                return ForensicResult(
                    branch_id="texture",
                    branch_name="Local Binary Pattern (LBP) Texture Analysis",
                    status="ERROR",
                    score=None,
                    explanation="Image too small for reliable texture analysis.",
                    method_description="Local Binary Pattern (LBP) texture extraction.",
                )

            # 2. Compute LBP on integer array (skimage convention)
            gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
            lbp = self._compute_lbp(gray_uint8)

            # Normalize LBP for visualization
            lbp_min, lbp_max = float(lbp.min()), float(lbp.max())
            norm_lbp = (
                (lbp - lbp_min) / max(lbp_max - lbp_min, 1e-6) * 255.0
            ).astype(np.uint8)
            viz_rgb = colorize_grayscale(norm_lbp, "magma")

            # 3. Global Shannon Entropy of texture
            # Natural diverse terrain exhibits moderate-to-high entropy
            # AI synthesis or clone-stamped vegetation exhibits either unnatural flat uniformity or cloned histograms
            hist, _ = np.histogram(lbp.ravel(), bins=min(self.n_points + 2, 64), density=True)
            hist = hist[hist > 0]
            shannon_entropy = float(-np.sum(hist * np.log2(hist)))

            # 4. Regional patch diversity (4x4 grid)
            grid_r, grid_c = 4, 4
            patch_h, patch_w = h // grid_r, w // grid_c
            patch_entropies = []
            patch_stds = []

            for r in range(grid_r):
                for c in range(grid_c):
                    patch = lbp[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
                    p_hist, _ = np.histogram(patch.ravel(), bins=16, density=True)
                    p_hist = p_hist[p_hist > 0]
                    p_ent = float(-np.sum(p_hist * np.log2(p_hist))) if len(p_hist) > 0 else 0.0
                    patch_entropies.append(p_ent)
                    patch_stds.append(float(np.std(patch)))

            entropy_variance = float(np.var(patch_entropies))
            mean_patch_entropy = float(np.mean(patch_entropies))
            min_entropy = float(np.min(patch_entropies))

            # 5. Heuristic Suspiciousness Calculation:
            # - Synthetic/cloned canopy often shows extreme uniformity (low entropy < 2.0)
            #   or abrupt divergence between cloned and natural patches (high entropy_variance > 0.8)
            # Natural satellite canopy has high micro-texture entropy (~ 3.0 to 5.0).
            low_entropy_suspicion = max(0.0, (3.2 - mean_patch_entropy) / 2.0)
            variance_suspicion = min(entropy_variance / 0.70, 1.0)

            raw_score = 0.50 * low_entropy_suspicion + 0.50 * variance_suspicion
            suspiciousness_score = float(1.0 / (1.0 + np.exp(-7.0 * (raw_score - 0.35))))
            suspiciousness_score = round(min(max(suspiciousness_score, 0.0), 1.0), 3)

            # 6. Explanation
            if suspiciousness_score >= 0.62:
                explanation = (
                    f"Texture anomalies detected: mean entropy={mean_patch_entropy:.2f}, "
                    f"regional variance={entropy_variance:.2f}. Identified abnormally uniform "
                    f"canopy micro-texture or sharp textural divergence between patches."
                )
            elif suspiciousness_score <= 0.38:
                explanation = (
                    f"Consistent organic micro-texture: mean entropy={mean_patch_entropy:.2f}, "
                    f"entropy variance={entropy_variance:.2f}. Expected fractal texture variation "
                    f"consistent with natural forest and soil."
                )
            else:
                explanation = (
                    f"Moderate texture variability (entropy={mean_patch_entropy:.2f}). "
                    f"Texture patterns are inconclusive; monoculture plantations or calm bodies of water "
                    f"can produce naturally uniform readings."
                )

            return ForensicResult(
                branch_id="texture",
                branch_name="Local Binary Pattern (LBP) Texture Analysis",
                status="AVAILABLE",
                score=suspiciousness_score,
                explanation=explanation,
                method_description=(
                    f"Computes Local Binary Patterns (P={self.n_points}, R={self.radius}) "
                    f"and measures regional Shannon entropy to detect unnatural smoothing or tiling."
                ),
                limitations=(
                    "Agricultural monocultures, uniform crops, open water, and barren sand dunes "
                    "naturally exhibit low texture entropy and can trigger false positive alerts."
                ),
                metrics={
                    "global_entropy": round(shannon_entropy, 3),
                    "mean_patch_entropy": round(mean_patch_entropy, 3),
                    "min_patch_entropy": round(min_entropy, 3),
                    "entropy_variance": round(entropy_variance, 4),
                    "radius": self.radius,
                    "points": self.n_points,
                },
                visualization=viz_rgb,
            )

        except Exception as exc:
            logger.exception("LBP texture analysis failed: %s", exc)
            return ForensicResult(
                branch_id="texture",
                branch_name="Local Binary Pattern (LBP) Texture Analysis",
                status="ERROR",
                score=None,
                explanation=f"Error during texture analysis: {exc}",
                method_description="Local Binary Pattern micro-texture analysis.",
                limitations="Requires 2D intensity data.",
            )
