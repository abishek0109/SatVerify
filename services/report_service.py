"""PDF Report Generation Service for SatVerify.

Produces multi-page, publication-quality forensic audit reports
summarizing authenticity verdicts, branch-level metrics, and visualizations.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config.settings import get_settings
from core.pipeline import PipelineReport
from utils.logging_config import get_logger

logger = get_logger("services.report")


class PDFReportGenerator:
    """Generates comprehensive PDF audit reports from PipelineReport data."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(
        self, report: PipelineReport, output_path: Optional[Path] = None
    ) -> bytes:
        """Generate a complete PDF report and return bytes or save to disk."""
        buf = io.BytesIO()

        try:
            with PdfPages(buf) as pdf:
                # -------------------------------------------------------------
                # PAGE 1: Executive Summary & Core Comparison
                # -------------------------------------------------------------
                fig1 = plt.figure(figsize=(8.5, 11), dpi=200)
                fig1.patch.set_facecolor("#FFFFFF")

                # Header Branding
                plt.text(0.08, 0.95, "SatVerify", fontsize=20, fontweight="bold", color="#0D1B2A")
                plt.text(0.08, 0.925, "AI Satellite Image Authenticity Audit Report", fontsize=11, color="#415A77")
                plt.text(0.08, 0.905, "Automated Forensics for Carbon Credit Market Integrity", fontsize=9, fontstyle="italic", color="#778DA9")

                # Metadata Box (Right)
                meta_text = (
                    f"Report ID: {report.analysis_id}\n"
                    f"Generated: {report.timestamp}\n"
                    f"Filename: {report.validation.metadata.get('filename', 'satellite_upload.jpg')}\n"
                    f"Dimensions: {report.validation.dimensions_str}\n"
                    f"Execution: {report.execution_time_seconds:.2f}s"
                )
                plt.text(0.60, 0.90, meta_text, fontsize=8.5, color="#1B263B", linespacing=1.4,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4F8", edgecolor="#D0DBE5"))

                # Horizontal Rule
                plt.axhline(0.88, xmin=0.08, xmax=0.92, color="#E0E6ED", linewidth=1.5)

                # Verdict Card
                v_color = "#00873E" if report.decision.verdict == "REAL" else ("#C62828" if report.decision.verdict == "FAKE" else "#E65100")
                bg_color = "#E8F5E9" if report.decision.verdict == "REAL" else ("#FFEBEE" if report.decision.verdict == "FAKE" else "#FFF8E1")

                score_display = (
                    f"Suspicion Score: {report.decision.fused_score:.3f} / 1.00"
                    if report.decision.fused_score is not None
                    else "Suspicion Score: N/A"
                )

                verdict_box = (
                    f"VERDICT: {report.decision.headline}\n"
                    f"Confidence: {report.decision.confidence_percentage:.1f}% | {score_display}\n\n"
                    f"Reasoning:\n{report.decision.reason}"
                )
                plt.text(
                    0.08, 0.77, verdict_box, fontsize=9.5, color=v_color, fontweight="bold", linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.8", facecolor=bg_color, edgecolor=v_color, linewidth=1.5)
                )

                # Branch Summary Table
                plt.text(0.08, 0.64, "Forensic Branch Contribution Summary", fontsize=11, fontweight="bold", color="#0D1B2A")

                # Table columns: Branch, Status, Raw Score, Effective Weight, Contribution
                headers = ["Forensic Branch", "Status", "Score", "Weight", "Contribution"]
                table_data = []
                for b in report.fusion.branch_contributions:
                    s_str = f"{b.raw_score:.3f}" if b.raw_score is not None else "Excluded"
                    w_str = f"{b.effective_weight:.1%}" if b.effective_weight > 0 else "0%"
                    c_str = f"{b.weighted_contribution:.3f}" if b.weighted_contribution > 0 else "-"
                    table_data.append([b.branch_name, b.status, s_str, w_str, c_str])

                table = plt.table(
                    cellText=table_data,
                    colLabels=headers,
                    colLoc="left",
                    cellLoc="left",
                    loc="center",
                    bbox=[0.08, 0.44, 0.84, 0.18],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8.5)
                table.auto_set_column_width(col=list(range(len(headers))))

                # Style table header cells
                for col_idx in range(len(headers)):
                    cell = table[(0, col_idx)]
                    cell.set_facecolor("#1B263B")
                    cell.get_text().set_color("#FFFFFF")
                    cell.get_text().set_fontweight("bold")

                # Four Visual Panels (Row across middle bottom)
                plt.text(0.08, 0.40, "Core Analytical Evidence Strip", fontsize=11, fontweight="bold", color="#0D1B2A")

                ax1 = fig1.add_axes([0.08, 0.20, 0.19, 0.17])
                if report.bundle:
                    ax1.imshow(report.bundle.pil_original)
                ax1.set_title("Input Image", fontsize=8.5, fontweight="bold")
                ax1.axis("off")

                ax2 = fig1.add_axes([0.30, 0.20, 0.19, 0.17])
                ela_res = report.branch_results.get("ela")
                if ela_res and ela_res.visualization is not None:
                    ax2.imshow(ela_res.visualization)
                ax2.set_title("ELA Residuals", fontsize=8.5, fontweight="bold")
                ax2.axis("off")

                ax3 = fig1.add_axes([0.52, 0.20, 0.19, 0.17])
                fft_res = report.branch_results.get("fft")
                if fft_res and fft_res.visualization is not None:
                    ax3.imshow(fft_res.visualization)
                ax3.set_title("FFT Spectrum", fontsize=8.5, fontweight="bold")
                ax3.axis("off")

                ax4 = fig1.add_axes([0.74, 0.20, 0.19, 0.17])
                struct_res = report.branch_results.get("structural")
                if struct_res and struct_res.visualization is not None:
                    ax4.imshow(struct_res.visualization)
                ax4.set_title("Structural Gradients", fontsize=8.5, fontweight="bold")
                ax4.axis("off")

                # Legal Disclaimer Footer
                plt.text(
                    0.08, 0.08,
                    f"LEGAL & SCIENTIFIC DISCLAIMER:\n{report.decision.disclaimer}",
                    fontsize=7.5, color="#5A6E85", linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#E2E8F0")
                )
                plt.text(0.08, 0.04, f"SatVerify v{self.settings.VERSION} | Page 1 of 2", fontsize=7.5, color="#778DA9")

                plt.axis("off")
                pdf.savefig(fig1)
                plt.close(fig1)

                # -------------------------------------------------------------
                # PAGE 2: Detailed Forensic Breakdown & Methodologies
                # -------------------------------------------------------------
                fig2 = plt.figure(figsize=(8.5, 11), dpi=200)
                fig2.patch.set_facecolor("#FFFFFF")

                plt.text(0.08, 0.95, "Technical Forensic Evidence & Method Limitations", fontsize=15, fontweight="bold", color="#0D1B2A")
                plt.axhline(0.93, xmin=0.08, xmax=0.92, color="#E0E6ED", linewidth=1.5)

                # Display 4 secondary forensic maps
                ax_p1 = fig2.add_axes([0.08, 0.70, 0.22, 0.18])
                tex_res = report.branch_results.get("texture")
                if tex_res and tex_res.visualization is not None:
                    ax_p1.imshow(tex_res.visualization)
                ax_p1.set_title("LBP Texture Map", fontsize=8.5, fontweight="bold")
                ax_p1.axis("off")

                plt.text(0.33, 0.84, "Branch E: Local Binary Patterns (LBP)", fontsize=9.5, fontweight="bold", color="#0D1B2A")
                plt.text(0.33, 0.73,
                         f"Status: {tex_res.status if tex_res else 'N/A'}\n"
                         f"Metrics: {tex_res.metrics if tex_res else 'N/A'}\n"
                         f"Explanation: {tex_res.explanation if tex_res else 'N/A'}",
                         fontsize=7.5, color="#334155", linespacing=1.3)

                ax_p2 = fig2.add_axes([0.08, 0.46, 0.22, 0.18])
                geom_res = report.branch_results.get("geometry")
                if geom_res and geom_res.visualization is not None:
                    ax_p2.imshow(geom_res.visualization)
                ax_p2.set_title("Spatial Coherence Map", fontsize=8.5, fontweight="bold")
                ax_p2.axis("off")

                plt.text(0.33, 0.60, "Branch G: Geometry & Seam Analysis", fontsize=9.5, fontweight="bold", color="#0D1B2A")
                plt.text(0.33, 0.49,
                         f"Status: {geom_res.status if geom_res else 'N/A'}\n"
                         f"Metrics: {geom_res.metrics if geom_res else 'N/A'}\n"
                         f"Explanation: {geom_res.explanation if geom_res else 'N/A'}",
                         fontsize=7.5, color="#334155", linespacing=1.3)

                ax_p3 = fig2.add_axes([0.08, 0.22, 0.22, 0.18])
                sh_res = report.branch_results.get("shadow")
                if sh_res and sh_res.visualization is not None:
                    ax_p3.imshow(sh_res.visualization)
                ax_p3.set_title("Shadow Boundary Map", fontsize=8.5, fontweight="bold")
                ax_p3.axis("off")

                plt.text(0.33, 0.36, "Branch H: Shadow & Solar Geometry", fontsize=9.5, fontweight="bold", color="#0D1B2A")
                plt.text(0.33, 0.25,
                         f"Status: {sh_res.status if sh_res else 'N/A'}\n"
                         f"Metrics: {sh_res.metrics if sh_res else 'N/A'}\n"
                         f"Explanation: {sh_res.explanation if sh_res else 'N/A'}",
                         fontsize=7.5, color="#334155", linespacing=1.3)

                # Methodology & Weighting Note
                notes = (
                    "WEIGHTED FUSION METHODOLOGY & POLICY:\n"
                    "• Base Weights: ELA (22%), FFT (22%), Deep Learning (18%), Structural (16%), "
                    "Texture/Geom (12%), Shadow/Elevation (10%).\n"
                    "• Dynamic Re-normalization: Unavailable branches are excluded and remaining weights normalized to 100%.\n"
                    "• Missing Checkpoints Policy: Untrained ImageNet heads are never presented as deepfake detectors."
                )
                plt.text(
                    0.08, 0.08, notes, fontsize=7.5, color="#1B263B", linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#EBF3FA", edgecolor="#BFDBFE")
                )
                plt.text(0.08, 0.04, f"SatVerify v{self.settings.VERSION} | Page 2 of 2", fontsize=7.5, color="#778DA9")

                plt.axis("off")
                pdf.savefig(fig2)
                plt.close(fig2)

            pdf_bytes = buf.getvalue()
            if output_path:
                output_path.write_bytes(pdf_bytes)
                logger.info("Saved PDF report to: %s", output_path)

            return pdf_bytes

        except Exception as exc:
            logger.exception("Failed to generate PDF report: %s", exc)
            raise RuntimeError(f"PDF generation failed: {exc}") from exc
        finally:
            buf.close()
