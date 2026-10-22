"""Branch C — Fast Fourier Transform (FFT) Frequency Analysis Module.

Transforms images into the frequency domain to identify periodic artifacts,
checkerboard patterns, and synthetic generation fingerprints.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from forensics.ela import colorize_grayscale
from utils.logging_config import get_logger

logger = get_logger("forensics.fft")


class FrequencyDomainAnalyzer:
    """Analyzes spatial frequencies and spectral anomalies in satellite imagery."""

    def __init__(self, spike_sigma: float = 3.0, log_factor: float = 20.0) -> None:
        self.spike_sigma = spike_sigma
        self.log_factor = log_factor
        self.settings = get_settings()

    def _compute_radial_band_energies(
        self, magnitude: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Compute relative energy in low, mid, and high frequency bands."""
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = np.sqrt(cx**2 + cy**2)

        # Normalize radius to [0, 1]
        r_norm = r / max(max_r, 1e-6)

        total_energy = float(np.sum(magnitude)) + 1e-8
        low_band = float(np.sum(magnitude[r_norm < 0.15])) / total_energy
        mid_band = float(np.sum(magnitude[(r_norm >= 0.15) & (r_norm < 0.50)])) / total_energy
        high_band = float(np.sum(magnitude[r_norm >= 0.50])) / total_energy

        # Ratio of high to low energy
        high_to_low_ratio = high_band / max(low_band, 1e-5)
        return low_band, mid_band, high_band, high_to_low_ratio

    def analyze(
        self, image_input: Union[Image.Image, np.ndarray, str]
    ) -> ForensicResult:
        """Run 2D FFT analysis and compute spectral anomaly metrics."""
        try:
            # 1. Convert to grayscale float array
            if isinstance(image_input, Image.Image):
                gray = np.asarray(image_input.convert("L"), dtype=np.float32)
            elif isinstance(image_input, np.ndarray):
                if image_input.ndim == 3:
                    # RGB -> Gray
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
            if h < 8 or w < 8:
                return ForensicResult(
                    branch_id="fft",
                    branch_name="Fourier Frequency Analysis (FFT)",
                    status="ERROR",
                    score=None,
                    explanation="Image dimensions too small for FFT frequency analysis.",
                    method_description="2D Fast Fourier Transform.",
                )

            # 2. Compute 2D Fast Fourier Transform and shift zero-freq to center
            fft2 = np.fft.fft2(gray)
            fft_shifted = np.fft.fftshift(fft2)
            magnitude = np.abs(fft_shifted)

            # 3. Logarithmic magnitude spectrum for display
            log_spectrum = self.log_factor * np.log(magnitude + 1.0)
            norm_spectrum = (
                (log_spectrum - log_spectrum.min())
                / max(log_spectrum.max() - log_spectrum.min(), 1e-6)
                * 255.0
            ).astype(np.uint8)

            # 4. Colorize frequency spectrum for visual inspection
            viz_rgb = colorize_grayscale(norm_spectrum, "jet")

            # 5. Extract spectral statistics
            mean_mag = float(np.mean(magnitude))
            std_mag = float(np.std(magnitude))
            threshold = mean_mag + (self.spike_sigma * std_mag)

            # Outlier spikes (periodic frequencies / GAN grid harmonics)
            spike_mask = magnitude > threshold
            spike_count = int(np.count_nonzero(spike_mask))
            spike_ratio = float(spike_count / magnitude.size)

            # Radial frequency band distribution
            low_band, mid_band, high_band, high_to_low = self._compute_radial_band_energies(
                magnitude
            )

            # 6. Heuristic anomaly scoring
            # Natural terrain typically exhibits 1/f power drop-off (high frequency < 0.20 of total, low frequency dominant).
            # Synthetic generators (diffusion, GAN upsampling, patch splicing) produce:
            # - Elevated high-frequency noise floor (high_band > 0.25)
            # - Periodic discrete frequency spikes (spike_ratio > 0.0008)
            norm_spike = min(spike_ratio / 0.0012, 1.0)
            norm_high_freq = min(max((high_band - 0.15) / 0.25, 0.0), 1.0)

            raw_score = (0.60 * norm_spike) + (0.40 * norm_high_freq)

            # Calibrate through sigmoid around 0.35
            suspiciousness_score = float(1.0 / (1.0 + np.exp(-8.0 * (raw_score - 0.35))))
            suspiciousness_score = round(min(max(suspiciousness_score, 0.0), 1.0), 3)

            # 7. Formulate explanation
            if suspiciousness_score >= 0.62:
                explanation = (
                    f"Elevated spectral anomalies detected: spike ratio={spike_ratio:.5f}, "
                    f"high-freq energy={high_band:.1%}. Periodic grid harmonics or abnormal "
                    f"high-frequency noise suggest generative AI upsampling or synthetic textures."
                )
            elif suspiciousness_score <= 0.38:
                explanation = (
                    f"Natural power spectral density observed: smooth radial decay, "
                    f"low spike ratio ({spike_ratio:.5f}), expected 1/f falloff. Consistent with authentic satellite sensors."
                )
            else:
                explanation = (
                    f"Moderate frequency characteristics (high-freq={high_band:.1%}, spike ratio={spike_ratio:.5f}). "
                    f"Spectral signature is ambiguous; repetitive natural patterns (e.g. agricultural rows) may create minor spikes."
                )

            return ForensicResult(
                branch_id="fft",
                branch_name="Fourier Frequency Analysis (FFT)",
                status="AVAILABLE",
                score=suspiciousness_score,
                explanation=explanation,
                method_description=(
                    "Transforms image to frequency space via 2D Fast Fourier Transform. "
                    "Measures radial energy decay and identifies anomalous periodic grid spikes."
                ),
                limitations=(
                    "Regular natural terrain (manmade tree plantations, vineyard rows, urban grids) "
                    "can produce periodic spectral spikes that mimic generative artifacts."
                ),
                metrics={
                    "spike_ratio": round(spike_ratio, 6),
                    "spike_count": spike_count,
                    "low_band_energy": round(low_band, 3),
                    "mid_band_energy": round(mid_band, 3),
                    "high_band_energy": round(high_band, 3),
                    "high_to_low_ratio": round(high_to_low, 3),
                },
                visualization=viz_rgb,
            )

        except Exception as exc:
            logger.exception("FFT analysis failed: %s", exc)
            return ForensicResult(
                branch_id="fft",
                branch_name="Fourier Frequency Analysis (FFT)",
                status="ERROR",
                score=None,
                explanation=f"Error executing FFT frequency analysis: {exc}",
                method_description="2D Fast Fourier Transform analysis.",
                limitations="Requires numerical grayscale conversion.",
            )
