"""End-to-end integration tests for SatVerifyPipeline."""

from pathlib import Path
from PIL import Image

from core.pipeline import SatVerifyPipeline


def test_pipeline_on_valid_fixture(tmp_path):
    """Verify that the full pipeline executes on an image and returns a valid PipelineReport."""
    # Create simple test image
    test_img_path = tmp_path / "valid_scene.jpg"
    img = Image.new("RGB", (128, 128), color=(45, 120, 55))
    img.save(test_img_path, format="JPEG")

    pipeline = SatVerifyPipeline()
    report = pipeline.analyze(test_img_path)

    # Schema checks
    assert report.analysis_id.startswith("SAT-")
    assert report.validation.is_valid is True
    assert report.bundle is not None
    assert len(report.branch_results) >= 7

    # Check key forensic branches
    assert "ela" in report.branch_results
    assert "fft" in report.branch_results
    assert "texture" in report.branch_results
    assert "structural" in report.branch_results

    # Check decision
    assert report.decision.verdict in ["REAL", "FAKE", "INCONCLUSIVE"]
    assert 0.0 <= report.decision.confidence_percentage <= 100.0
    assert report.execution_time_seconds > 0.0


def test_pipeline_handles_missing_model_checkpoint(tmp_path):
    """Verify that a missing CNN checkpoint doesn't crash the pipeline and is honestly reported."""
    test_img_path = tmp_path / "test_no_ckpt.jpg"
    img = Image.new("RGB", (64, 64), color=(30, 80, 40))
    img.save(test_img_path, format="JPEG")

    pipeline = SatVerifyPipeline()
    report = pipeline.analyze(test_img_path)

    # Check deep learning branches
    resnet_res = report.branch_results.get("resnet50")
    assert resnet_res is not None
    # If no trained weights exist, it must be UNAVAILABLE or ERROR
    assert resnet_res.status in ["UNAVAILABLE", "ERROR"]
    assert resnet_res.score is None


def test_pipeline_handles_corrupt_input(tmp_path):
    """Verify that corrupt data is caught at validation step without crashing."""
    corrupt_file = tmp_path / "broken.jpg"
    corrupt_file.write_bytes(b"CORRUPTED_NON_IMAGE_DATA")

    pipeline = SatVerifyPipeline()
    report = pipeline.analyze(corrupt_file)

    assert report.validation.is_valid is False
    assert report.decision.verdict == "INCONCLUSIVE"
    assert "Corrupted or invalid" in report.decision.reason
