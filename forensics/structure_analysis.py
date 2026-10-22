"""Branch F — Structural Consistency Analysis Module.

Detects structural irregularities, broken edge continuity (roads/rivers),
abrupt vegetation boundary discontinuities, and seam-line artifacts.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from forensics.ela import colorize_grayscale
from utils.logging_config import get_logger

logger = get_logger("forensics.structure")


class StructuralConsistencyAnalyzer:
    """Analyzes geometric edge coherence and boundary continuity in satellite imagery."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _compute_gradients(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Sobel-like gradients along X and Y axes."""
        # Use simple convolution kernels with numpy
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        try:
            from scipy.signal import convolve2d

            gx = convolve2d(gray, kx, mode="same", boundary="symm")
            gy = convolve2d(gray, ky, mode="same", boundary="symm")
        except Exception:
            # Fallback simple differences
            gx = np.zeros_like(gray)
            gy = np.zeros_like(gray)
            gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
            gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5

        grad_mag = np.sqrt(gx**2 + gy**2)
        return gx, gy, grad_mag

    def _vegetation_boundary_analysis(self, rgb_arr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Analyze green-channel excess vegetation index (ExG = 2G - R - B) and boundary sharpness."""
        r = rgb_arr[:, :, 0].astype(np.float32)
        g = rgb_arr[:, :, 1].astype(np.float32)
        b = rgb_arr[:, :, 2].astype(np.float32)

        # Excess Green Index (ExG)
        exg = 2.0 * g - r - b

        # Compute gradient of vegetation index to detect unnatural artificial hard boundaries
        _, _, exg_grad = self._compute_gradients(exg)

        # High 98th percentile gradient indicates unnatural cookie-cutter / cut-and-paste tree boundaries
        p98_grad = float(np.percentile(exg_grad, 98))
        return p98_grad, exg_grad

    def analyze(
        self, image_input: Union[Image.Image, np.ndarray, str]
    ) -> ForensicResult:
        """Run structural consistency checks."""
        try:
            # 1. Obtain RGB and Grayscale arrays
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
                    branch_id="structural",
                    branch_name="Structural Consistency Analysis",
                    status="ERROR",
                    score=None,
                    explanation="Image too small for structural analysis.",
                    method_description="Edge gradient and linear continuity analysis.",
                )

            # 2. Compute edge gradients
            gx, gy, grad_mag = self._compute_gradients(gray)

            # 3. Analyze Vegetation Boundaries (ExG gradient)
            exg_p98, exg_grad = self._vegetation_boundary_analysis(rgb_arr)

            # 4. Detect abrupt regional edge discontinuities (block-level edge density variation)
            grid_size = 16
            bh, bw = max(8, h // grid_size), max(8, w // grid_size)
            num_h, num_w = h // bh, w // bw

            edge_binary = grad_mag > np.percentile(grad_mag, 85)
            block_edge_densities = []

            for r in range(num_h):
                for c in range(num_w):
                    block = edge_binary[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                    block_edge_densities.append(float(np.mean(block)))

            edge_density_var = float(np.var(block_edge_densities))
            max_edge_density = float(np.max(block_edge_densities))
            min_edge_density = float(np.min(block_edge_densities))
            edge_disparity = max_edge_density - min_edge_density

            # 5. Linear feature continuity / discontinuity heuristic
            # In authentic scenes, roads and rivers show continuous gradient orientation.
            # In spliced/generative images, gradient orientation is randomized or fractured at patch borders.
            angles = np.arctan2(gy, gx)
            edge_angles = angles[edge_binary]
            if len(edge_angles) > 100:
                angle_hist, _ = np.histogram(edge_angles, bins=18, range=(-np.pi, np.pi))
                angle_entropy = float(
                    -np.sum(
                        (angle_hist / angle_hist.sum())
                        * np.log2((angle_hist / angle_hist.sum()) + 1e-8)
                    )
                )
            else:
                angle_entropy = 3.0

            # 6. Heuristic Scoring
            # High vegetation border gradient (cut-and-paste boundary) -> higher score
            # High block edge disparity (spliced sharp contrast edge) -> higher score
            norm_exg_boundary = min(max(exg_p98 - 30.0, 0.0) / 70.0, 1.0)
            norm_edge_disparity = min(edge_disparity / 0.60, 1.0)

            raw_score = 0.55 * norm_exg_boundary + 0.45 * norm_edge_disparity
            suspiciousness_score = float(1.0 / (1.0 + np.exp(-7.0 * (raw_score - 0.35))))
            suspiciousness_score = round(min(max(suspiciousness_score, 0.0), 1.0), 3)

            # 7. Visualization: Composite edge + vegetation gradient
            norm_grad = (
                (grad_mag / max(float(np.percentile(grad_mag, 99)), 1e-4)) * 255.0
            )
            norm_grad = np.clip(norm_grad, 0, 255).astype(np.uint8)
            viz_rgb = colorize_grayscale(norm_grad, "inferno")

            # 8. Explanation
            if suspiciousness_score >= 0.62:
                explanation = (
                    f"Structural boundary anomalies detected: vegetation edge gradient={exg_p98:.1f}, "
                    f"regional edge disparity={edge_disparity:.2f}. Indicates artificial cut-and-paste seams, "
                    f"discontinuous linear corridors, or spliced terrain patches."
                )
            elif suspiciousness_score <= 0.38:
                explanation = (
                    f"Continuous organic structures: natural vegetation gradients ({exg_p98:.1f}) and "
                    f"smooth boundary transitions. Linear features and terrain corridors exhibit coherent continuity."
                )
            else:
                explanation = (
                    f"Moderate structural continuity (vegetation gradient={exg_p98:.1f}). "
                    f"Features are borderline; human infrastructure (fences, logging roads, canals) can create sharp boundaries."
                )

            return ForensicResult(
                branch_id="structural",
                branch_name="Structural Consistency Analysis",
                status="AVAILABLE",
                score=suspiciousness_score,
                explanation=explanation,
                method_description=(
                    "Analyzes Sobel spatial gradient continuity, Excess Green (ExG) vegetation boundary sharpness, "
                    "and regional edge density disparities to detect spliced patches and disrupted infrastructure."
                ),
                limitations=(
                    "Legitimate land-use boundaries (agricultural field borders, clearings, asphalt roads, property fences) "
                    "produce sharp linear gradients that can resemble manipulation seams."
                ),
                metrics={
                    "vegetation_boundary_gradient": round(exg_p98, 2),
                    "edge_density_disparity": round(edge_disparity, 3),
                    "edge_density_variance": round(edge_density_var, 4),
                    "gradient_angle_entropy": round(angle_entropy, 3),
                },
                visualization=viz_rgb,
            )

        except Exception as exc:
            logger.exception("Structural consistency analysis failed: %s", exc)
            return ForensicResult(
                branch_id="structural",
                branch_name="Structural Consistency Analysis",
                status="ERROR",
                score=None,
                explanation=f"Error executing structural consistency analysis: {exc}",
                method_description="Sobel gradient and vegetation boundary analysis.",
                limitations="Requires spatial 2D array.",
            )
