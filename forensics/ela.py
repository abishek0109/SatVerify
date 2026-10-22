"""Branch B — Error Level Analysis (ELA) Module.

Computes re-compression residuals to identify differential compression artifacts,
spliced boundaries, and uneven synthesis signatures in satellite imagery.
"""

from __future__ import annotations

import io
from typing import Optional, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from utils.logging_config import get_logger

logger = get_logger("forensics.ela")


def colorize_grayscale(gray: np.ndarray, colormap_name: str = "turbo") -> np.ndarray:
    """Colorize a normalized grayscale array using Matplotlib or NumPy fallback."""
    try:
        import matplotlib
        norm = gray.astype(np.float32) / 255.0
        cmap = matplotlib.colormaps[colormap_name]
        rgba = cmap(norm)
        rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
        return rgb
    except Exception:
        # Fallback pseudo-color: simple hot gradient (R=gray, G=gray/2, B=0)
        r = gray
        g = (gray * 0.5).astype(np.uint8)
        b = np.zeros_like(gray)
        return np.stack([r, g, b], axis=-1)


class ErrorLevelAnalyzer:
    """Performs Error Level Analysis on satellite imagery."""

    def __init__(
        self,
        quality: Optional[int] = None,
        amplification_factor: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.quality = quality or settings.ELA_JPEG_QUALITY
        self.amplification = amplification_factor or settings.ELA_AMPLIFICATION_FACTOR

    def analyze(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
    ) -> ForensicResult:
        """Run ELA analysis and return standardized ForensicResult."""
        try:
            # 1. Normalize input to RGB PIL Image
            if isinstance(image_input, Image.Image):
                original_pil = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                original_pil = Image.fromarray(image_input).convert("RGB")
            else:
                with Image.open(str(image_input)) as img:
                    original_pil = img.convert("RGB")

            # 2. In-memory JPEG recompression
            buffer = io.BytesIO()
            original_pil.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            recompressed_pil = Image.open(buffer).convert("RGB")

            # 3. Signed difference calculation (prevents uint8 underflow)
            orig_arr = np.asarray(original_pil, dtype=np.int16)
            recomp_arr = np.asarray(recompressed_pil, dtype=np.int16)
            diff = np.abs(orig_arr - recomp_arr)  # Shape (H, W, 3)

            # 4. Amplification
            amplified = np.clip(diff * self.amplification, 0, 255).astype(np.uint8)

            # 5. Extract statistics on single intensity channel
            diff_intensity = np.mean(diff, axis=2).astype(np.float32)  # (H, W)
            mean_error = float(np.mean(diff_intensity))
            std_error = float(np.std(diff_intensity))
            max_error = float(np.max(diff_intensity))

            # Regional block variance (e.g. 16x16 grid)
            h, w = diff_intensity.shape
            block_size = max(16, min(h // 16, w // 16))
            num_h = h // block_size
            num_w = w // block_size

            if num_h > 0 and num_w > 0:
                grid_blocks = diff_intensity[: num_h * block_size, : num_w * block_size]
                grid_blocks = grid_blocks.reshape(num_h, block_size, num_w, block_size)
                block_means = grid_blocks.mean(axis=(1, 3))
                regional_variance = float(np.var(block_means))
            else:
                regional_variance = 0.0

            # 6. Heuristic suspiciousness score
            # Real satellite images exhibit uniform compression residuals (std ~ 2 to 12).
            # Spliced / synthetic imagery introduces sharp local variance (std > 20) and high regional variance.
            # Base normalized spread:
            norm_std = min(std_error / 30.0, 1.0)
            norm_reg_var = min(regional_variance / 40.0, 1.0)
            raw_score = 0.65 * norm_std + 0.35 * norm_reg_var

            # Calibrate through sigmoid sharpening around 0.35
            suspiciousness_score = float(1.0 / (1.0 + np.exp(-8.0 * (raw_score - 0.35))))
            suspiciousness_score = round(min(max(suspiciousness_score, 0.0), 1.0), 3)

            # 7. Generate visualization
            # Colorize the amplified difference
            amp_gray = np.clip(np.mean(amplified, axis=2), 0, 255).astype(np.uint8)
            heatmap_rgb = colorize_grayscale(amp_gray, "turbo")

            # Explanation
            if suspiciousness_score >= 0.62:
                explanation = (
                    f"High compression residual variance (std={std_error:.2f}, regional var={regional_variance:.2f}). "
                    f"Uneven error distribution indicates potential splicing, copy-paste tampering, or generative synthesis."
                )
            elif suspiciousness_score <= 0.38:
                explanation = (
                    f"Uniform compression residuals (std={std_error:.2f}, regional var={regional_variance:.2f}). "
                    f"Consistent degradation across the scene matches expected sensor capture patterns."
                )
            else:
                explanation = (
                    f"Moderate residual variance (std={std_error:.2f}). "
                    f"Residual levels are inconclusive; natural high-contrast terrain may produce localized spikes."
                )

            return ForensicResult(
                branch_id="ela",
                branch_name="Error Level Analysis (ELA)",
                status="AVAILABLE",
                score=suspiciousness_score,
                explanation=explanation,
                method_description=(
                    "Recompresses image to JPEG (quality=90) and computes absolute pixel residuals. "
                    "Uniform residuals indicate uniform capture history, while variance hotspots suggest local manipulation."
                ),
                limitations=(
                    "Heuristic signal sensitive to image format, prior re-saving cycles, and natural high-contrast edges. "
                    "Bright residuals do not definitively prove fraud."
                ),
                metrics={
                    "mean_error": round(mean_error, 2),
                    "std_error": round(std_error, 2),
                    "max_error": round(max_error, 2),
                    "regional_variance": round(regional_variance, 2),
                    "recompression_quality": self.quality,
                    "amplification_factor": self.amplification,
                },
                visualization=heatmap_rgb,
            )

        except Exception as exc:
            logger.exception("ELA analysis failed: %s", exc)
            return ForensicResult(
                branch_id="ela",
                branch_name="Error Level Analysis (ELA)",
                status="ERROR",
                score=None,
                explanation=f"Error executing ELA analysis: {exc}",
                method_description="JPEG recompression residual variance computation.",
                limitations="Requires valid image decoding.",
            )
