"""Unit tests for Texture (Branch E), Structural (Branch F), and Geometry (Branch G)."""

import numpy as np
from PIL import Image

from forensics.geometry_analysis import GeometryCoherenceAnalyzer
from forensics.structure_analysis import StructuralConsistencyAnalyzer
from forensics.texture_analysis import TextureAnalyzer


def test_lbp_texture_analysis():
    """Verify LBP calculates entropy and regional variance metrics."""
    analyzer = TextureAnalyzer()
    arr = np.random.randint(50, 180, (128, 128), dtype=np.uint8)

    result = analyzer.analyze(arr)
    assert result.status == "AVAILABLE"
    assert result.branch_id == "texture"
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert "global_entropy" in result.metrics
    assert "mean_patch_entropy" in result.metrics
    assert result.visualization is not None


def test_structural_consistency_analysis():
    """Verify structural analysis detects edge disparities and gradients."""
    analyzer = StructuralConsistencyAnalyzer()

    # Image with sharp synthetic rectangular seam
    img_arr = np.full((128, 128, 3), 50, dtype=np.uint8)
    img_arr[32:96, 32:96, :] = 220  # White square in dark scene

    result = analyzer.analyze(img_arr)
    assert result.status == "AVAILABLE"
    assert result.branch_id == "structural"
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert "edge_density_disparity" in result.metrics
    assert result.visualization is not None


def test_geometry_coherence_analysis():
    """Verify geometry analysis measures cross-channel correlation and seams."""
    analyzer = GeometryCoherenceAnalyzer()

    # Normal correlated channels
    gray = np.random.randint(50, 200, (128, 128), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    result = analyzer.analyze(rgb)
    assert result.status == "AVAILABLE"
    assert result.branch_id == "geometry"
    assert result.score is not None
    # Identical channels have correlation 1.0 and should produce very low suspicion
    assert result.metrics["rg_correlation"] > 0.95
