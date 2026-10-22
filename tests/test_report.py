"""Unit tests for PDF Report generation."""

from pathlib import Path
from PIL import Image

from core.pipeline import SatVerifyPipeline
from services.report_service import PDFReportGenerator


def test_pdf_report_generation(tmp_path):
    """Verify that PDFReportGenerator produces non-empty valid PDF bytes."""
    test_img = tmp_path / "report_test.jpg"
    img = Image.new("RGB", (128, 128), color=(50, 100, 60))
    img.save(test_img, format="JPEG")

    pipeline = SatVerifyPipeline()
    report = pipeline.analyze(test_img)

    generator = PDFReportGenerator()
    out_pdf = tmp_path / "test_report.pdf"
    pdf_bytes = generator.generate(report, output_path=out_pdf)

    assert len(pdf_bytes) > 1000  # Multi-page PDF should be > 1KB
    assert pdf_bytes.startswith(b"%PDF")
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 1000
