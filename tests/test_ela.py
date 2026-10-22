"""Unit tests for Error Level Analysis (Branch B)."""

import numpy as np
from PIL import Image

from forensics.ela import ErrorLevelAnalyzer


def test_ela_produces_valid_output_and_statistics():
    """Verify that ELA outputs valid image arrays, bounded scores, and metrics."""
    analyzer = ErrorLevelAnalyzer(quality=90, amplification_factor=10)

    # Synthetic image with texture
    arr = np.random.randint(40, 200, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    result = analyzer.analyze(img)
    assert result.status == "AVAILABLE"
    assert result.branch_id == "ela"
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    # Visualization checks
    assert result.visualization is not None
    assert result.visualization.shape == (128, 128, 3)
    assert result.visualization.dtype == np.uint8

    # Metric checks
    assert "std_error" in result.metrics
    assert "mean_error" in result.metrics
    assert result.metrics["std_error"] >= 0.0


def test_ela_handles_uniform_image():
    """Verify that a completely uniform image produces low residual error."""
    analyzer = ErrorLevelAnalyzer()
    img = Image.new("RGB", (128, 128), color=(100, 150, 100))

    result = analyzer.analyze(img)
    assert result.status == "AVAILABLE"
    assert result.score is not None
    # Uniform image has almost zero recompression residual variation
    assert result.score < 0.38
