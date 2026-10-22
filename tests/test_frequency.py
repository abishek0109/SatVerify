"""Unit tests for Fourier Frequency Analysis (Branch C)."""

import numpy as np
from PIL import Image

from forensics.frequency_analysis import FrequencyDomainAnalyzer


def test_fft_produces_magnitude_spectrum_and_score():
    """Verify 2D FFT calculates power spectrum, spike ratios, and normalized scores."""
    analyzer = FrequencyDomainAnalyzer()

    # Image with high frequency noise
    arr = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    result = analyzer.analyze(arr)

    assert result.status == "AVAILABLE"
    assert result.branch_id == "fft"
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    # Visualization
    assert result.visualization is not None
    assert result.visualization.shape == (128, 128, 3)

    # Metrics
    assert "spike_ratio" in result.metrics
    assert "high_band_energy" in result.metrics
    assert 0.0 <= result.metrics["high_band_energy"] <= 1.0


def test_fft_smooth_natural_frequencies():
    """Verify smooth sinusoidal gradient produces expected low high-frequency energy."""
    analyzer = FrequencyDomainAnalyzer()
    x = np.linspace(0, np.pi * 2, 128)
    y = np.linspace(0, np.pi * 2, 128)
    xx, yy = np.meshgrid(x, y)
    smooth = ((np.sin(xx) + np.cos(yy) + 2.0) / 4.0 * 255.0).astype(np.uint8)

    result = analyzer.analyze(smooth)
    assert result.status == "AVAILABLE"
    assert result.metrics["high_band_energy"] < 0.20
