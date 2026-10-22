"""SatVerify — Deepfake Satellite Image Detection for Carbon Credit Markets.

A multi-branch geospatial image forensics application combining Error Level Analysis,
2D Fourier frequency domain analysis, Local Binary Pattern texture metrics,
structural boundary analysis, spatial coherence, shadow validation, and weighted score fusion.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from PIL import Image

from config.settings import get_settings
from core.pipeline import PipelineReport, SatVerifyPipeline
from explainability.visualizations import (
    create_branch_score_chart,
    create_comparison_figure,
    create_fusion_donut_chart,
)
from services.image_service import ImageService
from services.report_service import PDFReportGenerator
from utils.logging_config import get_logger

logger = get_logger("app")
settings = get_settings()

st.set_page_config(
    page_title="SatVerify — Deepfake Satellite Image Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles() -> None:
    """Inject modern, dark geospatial-forensic CSS styles."""
    st.markdown(
        """
        <style>
            /* Main background & typography */
            .main {
                background-color: #0B1118;
                color: #E2E8F0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .stApp {
                background: radial-gradient(circle at 15% 15%, #131E2C 0%, #0B1118 70%);
            }
            
            /* Header Hero Banner */
            .sat-hero {
                background: linear-gradient(135deg, #0E1A29 0%, #152538 50%, #0B1522 100%);
                border: 1px solid #1E344D;
                border-radius: 14px;
                padding: 1.5rem 1.8rem;
                margin-bottom: 1.4rem;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            }
            .sat-title {
                font-size: 2.1rem;
                font-weight: 800;
                letter-spacing: -0.5px;
                background: linear-gradient(90deg, #FFFFFF 0%, #69F0AE 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
            }
            .sat-tagline {
                font-size: 1.05rem;
                color: #94A3B8;
                margin-top: 0.3rem;
                font-weight: 500;
            }
            
            /* Verdict Badges */
            .verdict-card {
                padding: 1.25rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 1.2rem;
                border-width: 1.5px;
                border-style: solid;
            }
            .verdict-real {
                background: rgba(0, 200, 83, 0.08);
                border-color: #00C853;
                box-shadow: 0 0 25px rgba(0, 200, 83, 0.15);
            }
            .verdict-fake {
                background: rgba(213, 0, 0, 0.08);
                border-color: #D50000;
                box-shadow: 0 0 25px rgba(213, 0, 0, 0.2);
            }
            .verdict-inconclusive {
                background: rgba(255, 171, 0, 0.08);
                border-color: #FFAB00;
                box-shadow: 0 0 25px rgba(255, 171, 0, 0.15);
            }
            
            /* Forensic Card Containers */
            .forensic-card {
                background-color: #121A24;
                border: 1px solid #1E2D3D;
                border-radius: 10px;
                padding: 1.1rem;
                margin-bottom: 1rem;
            }
            .metric-stat {
                font-size: 1.7rem;
                font-weight: 700;
                color: #FFFFFF;
            }
            .metric-label {
                font-size: 0.85rem;
                color: #94A3B8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            /* Sidebar styling */
            section[data-testid="stSidebar"] {
                background-color: #0D1520 !important;
                border-right: 1px solid #1A2838;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #121C28;
                border: 1px solid #1E2E40;
                border-radius: 8px 8px 0 0;
                padding: 10px 18px;
                color: #94A3B8;
                font-weight: 600;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1A2B3E !important;
                color: #00E676 !important;
                border-bottom: 2px solid #00E676 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_pipeline() -> SatVerifyPipeline:
    """Initialize and cache the master forensic pipeline."""
    return SatVerifyPipeline()


def render_hero_banner() -> None:
    """Render the application hero title banner."""
    st.markdown(
        """
        <div class="sat-hero">
            <h1 class="sat-title">🛰️ SatVerify</h1>
            <div class="sat-tagline">AI Forensics for Trusted Carbon Markets — Deepfake Satellite Image Detection</div>
            <p style="color: #64748B; font-size: 0.88rem; margin: 0.5rem 0 0 0;">
                Multi-branch forensic validation: Error Level Analysis • 2D Fast Fourier Transform • 
                LBP Canopy Texture • Structural Continuity • Spatial Coherence • Solar Geometry
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls() -> Dict[str, float]:
    """Render sidebar navigation, threshold tuning, and project links."""
    st.sidebar.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/satellite/materialicons/48dp/2x/baseline_satellite_white_48dp.png",
        width=48,
    )
    st.sidebar.markdown("### **SatVerify Console**")
    st.sidebar.caption("v2.0.0 • Hackathon Edition")
    st.sidebar.divider()

    st.sidebar.markdown("#### ⚙️ **Decision Thresholds**")
    real_thr = st.sidebar.slider(
        "REAL Decision Threshold",
        min_value=0.20,
        max_value=0.50,
        value=settings.DECISION_REAL_THRESHOLD,
        step=0.01,
        help="Suspicion scores strictly below this value are classified as REAL (Authentic).",
    )
    fake_thr = st.sidebar.slider(
        "FAKE Decision Threshold",
        min_value=0.51,
        max_value=0.80,
        value=settings.DECISION_FAKE_THRESHOLD,
        step=0.01,
        help="Suspicion scores at or above this value are classified as FAKE (Altered).",
    )

    st.sidebar.info(
        f"**Uncertainty Margin:** Scores between [{real_thr:.2f}, {fake_thr:.2f}) "
        "produce an **INCONCLUSIVE** outcome."
    )

    st.sidebar.divider()
    st.sidebar.markdown("#### 🔬 **Enabled Pipeline Branches**")
    st.sidebar.markdown(
        """
        - 🟢 **Branch B:** Error Level Analysis (22%)
        - 🟢 **Branch C:** Fourier Transform FFT (22%)
        - ⚪ **Branch A/D:** Deep Learning CNN/ViT (18%)
        - 🟢 **Branch F:** Structural Consistency (16%)
        - 🟢 **Branch E/G:** Texture & Geometry (12%)
        - 🟡 **Branch H/I:** Shadow & Elevation (10%)
        """
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "SatVerify is an automated forensic demonstration. It highlights mathematical "
        "anomalies and should be corroborated with field audits."
    )

    return {"real_thr": real_thr, "fake_thr": fake_thr}


def render_verdict_banner(report: PipelineReport) -> None:
    """Render the high-impact verdict banner with calibrated scores and badges."""
    verdict = report.decision.verdict
    conf = report.decision.confidence_percentage
    score = report.decision.fused_score

    score_text = f"Suspicion Score: {score:.3f}" if score is not None else "Suspicion: N/A"

    if verdict == "REAL":
        v_class = "verdict-real"
        v_icon = "✅"
        v_title = "REAL — Authentic Terrain Candidate"
        v_desc = (
            "No significant forensic manipulation signatures detected across active analytical branches. "
            "Compression residuals, frequency roll-off, and micro-texture align with expected natural sensor captures."
        )
    elif verdict == "FAKE":
        v_class = "verdict-fake"
        v_icon = "⚠️"
        v_title = "FAKE — Altered / Synthetic Image Detected"
        v_desc = (
            "Significant forensic anomalies detected across multiple independent branches. "
            "Evidence indicates abnormal compression disparity, periodic generative harmonics, or spliced vegetation seams."
        )
    else:
        v_class = "verdict-inconclusive"
        v_icon = "⚖️"
        v_title = "INCONCLUSIVE — Ambiguous / Conflicting Evidence"
        v_desc = (
            "Forensic evidence is insufficient or contradictory. The fused score falls inside the calibrated "
            "uncertainty interval. A definitive REAL or FAKE decision cannot be reliably rendered."
        )

    st.markdown(
        f"""
        <div class="verdict-card {v_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h2 style="margin: 0; font-size: 1.55rem; font-weight: 800; color: #FFFFFF;">
                    {v_icon} {v_title}
                </h2>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; background: rgba(0,0,0,0.3); padding: 0.3rem 0.8rem; border-radius: 8px;">
                    {conf:.1f}% Confidence | {score_text}
                </div>
            </div>
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.45; color: #CBD5E1;">
                {report.decision.reason}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_analyzer(pipeline: SatVerifyPipeline, thresholds: Dict[str, float]) -> None:
    """Main image analysis and live forensics dashboard."""
    st.markdown("### **1. Satellite Image Upload & Acquisition**")

    # Sample Preset Fixtures
    fixtures_dir = Path("data") / "fixtures"
    sample_options = ["-- Select Preset Sample Fixture or Upload Below --"]
    sample_files: Dict[str, Path] = {}

    if fixtures_dir.exists():
        for r_file in sorted(fixtures_dir.glob("real/*.jpg")):
            name = f"🌲 Authentic Terrain Sample: {r_file.stem}"
            sample_options.append(name)
            sample_files[name] = r_file
        for f_file in sorted(fixtures_dir.glob("fake/*.jpg")):
            name = f"🚨 Tampered 'Paper Forest' Sample: {f_file.stem}"
            sample_options.append(name)
            sample_files[name] = f_file

    col_upload, col_preset = st.columns([1.2, 1.0])

    with col_preset:
        chosen_sample = st.selectbox(
            "Quick Demo: Load Preset Fixture",
            options=sample_options,
            help="Select a synthetic fixture sample to immediately evaluate the pipeline.",
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Satellite Image (JPEG, PNG, TIFF, BMP, WebP)",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
            help="Upload full-resolution optical satellite scenes.",
        )

    # Determine input source
    file_bytes: Optional[bytes] = None
    filename: str = "upload.jpg"

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
    elif chosen_sample != sample_options[0] and chosen_sample in sample_files:
        p = sample_files[chosen_sample]
        file_bytes = p.read_bytes()
        filename = p.name

    if file_bytes is None:
        st.info("👆 Upload a satellite image or choose a preset sample above to begin verification.")

        # Display Pipeline Overview Cards
        st.markdown("---")
        st.markdown("#### **How SatVerify Protects Carbon Credit Markets**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="forensic-card">
                    <h4 style="color: #69F0AE; margin-top:0;">🛡️ The "Paper Forest" Threat</h4>
                    <p style="font-size: 0.88rem; color: #94A3B8;">
                        Fraudulent carbon projects claim non-existent forest growth by digitally altering 
                        satellite imagery, copy-pasting tree textures, or boosting green color channels 
                        to simulate lush biomass.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="forensic-card">
                    <h4 style="color: #00B0FF; margin-top:0;">🔬 Multi-Branch Forensics</h4>
                    <p style="font-size: 0.88rem; color: #94A3B8;">
                        Instead of relying on a black-box model, SatVerify fuses physics-based Error Level 
                        Analysis, Fourier frequency harmonics, structural boundary continuity, and micro-texture 
                        entropy into a transparent score.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div class="forensic-card">
                    <h4 style="color: #FFD600; margin-top:0;">📄 Verifiable Audit Reports</h4>
                    <p style="font-size: 0.88rem; color: #94A3B8;">
                        Every inspection produces an exportable, tamper-evident PDF audit certificate 
                        containing analysis IDs, branch metrics, and full visual evidence strips for compliance auditors.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # Image metadata and preview
    st.markdown("---")
    st.markdown("### **2. Forensic Multi-Branch Inspection**")

    # Run analysis
    with st.spinner("Executing multi-branch forensic analysis pipeline..."):
        pipeline.decision_engine.real_threshold = thresholds["real_thr"]
        pipeline.decision_engine.fake_threshold = thresholds["fake_thr"]
        report = pipeline.analyze(file_bytes, original_filename=filename)

    # Render Verdict
    render_verdict_banner(report)

    # Key Metrics Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Analysis ID", report.analysis_id.split("-")[-1])
    with m2:
        st.metric("Dimensions", report.validation.dimensions_str)
    with m3:
        st.metric("Branches Active", f"{report.fusion.available_branches_count}/{report.fusion.total_branches_count}")
    with m4:
        f_score_disp = f"{report.decision.fused_score:.3f}" if report.decision.fused_score is not None else "N/A"
        st.metric("Fused Suspicion", f_score_disp)
    with m5:
        st.metric("Execution Time", f"{report.execution_time_seconds:.2f}s")

    # Visual Evidence Tabs
    tab_overview, tab_ela, tab_fft, tab_texture, tab_structure, tab_geometry, tab_charts, tab_report = st.tabs(
        [
            "👁️ Evidence Strip",
            "🔍 ELA Residuals",
            "📡 FFT Frequency",
            "🌿 LBP Texture",
            "📐 Structural & Seams",
            "🛰️ Geometry Coherence",
            "📊 Score Charts",
            "📄 Export Audit PDF",
        ]
    )

    with tab_overview:
        st.markdown("#### **Consolidated Analytical Comparison**")
        c_orig, c_ela, c_fft, c_str = st.columns(4)

        with c_orig:
            if report.bundle:
                st.image(report.bundle.pil_original, caption="Input Satellite Image", use_container_width=True)
        with c_ela:
            ela_r = report.branch_results.get("ela")
            if ela_r and ela_r.visualization is not None:
                st.image(ela_r.visualization, caption="ELA Residual Map (JPEG q=90)", use_container_width=True)
        with c_fft:
            fft_r = report.branch_results.get("fft")
            if fft_r and fft_r.visualization is not None:
                st.image(fft_r.visualization, caption="Fourier Magnitude Spectrum", use_container_width=True)
        with c_str:
            str_r = report.branch_results.get("structural")
            if str_r and str_r.visualization is not None:
                st.image(str_r.visualization, caption="Structural Edge Gradients", use_container_width=True)

        st.caption(
            "Comparing spatial texture against recompression residuals (ELA), periodic frequency spikes (FFT), "
            "and vegetation boundary sharpness (Structural)."
        )

    with tab_ela:
        st.markdown("#### **Branch B — Error Level Analysis (ELA)**")
        col_img, col_info = st.columns([1.2, 1.0])
        ela_r = report.branch_results.get("ela")
        with col_img:
            if ela_r and ela_r.visualization is not None:
                st.image(ela_r.visualization, caption="Error Level Heatmap (Turbo Colormap)", use_container_width=True)
        with col_info:
            if ela_r:
                score_str = f"{ela_r.score:.3f}" if ela_r.score is not None else "N/A"
                st.markdown(f"**Branch Status:** `{ela_r.status}`")
                st.markdown(f"**Suspicion Score:** `{score_str}` / 1.00")
                st.markdown(f"**Findings:** {ela_r.explanation}")
                st.markdown("**Quantitative Metrics:**")
                st.json(ela_r.metrics)
                st.info(f"**Methodology:** {ela_r.method_description}")
                st.warning(f"**Known Limitations:** {ela_r.limitations}")

    with tab_fft:
        st.markdown("#### **Branch C — Fourier Transform Frequency Analysis (FFT)**")
        col_img, col_info = st.columns([1.2, 1.0])
        fft_r = report.branch_results.get("fft")
        with col_img:
            if fft_r and fft_r.visualization is not None:
                st.image(fft_r.visualization, caption="Centered 2D Fast Fourier Transform Spectrum", use_container_width=True)
        with col_info:
            if fft_r:
                score_str = f"{fft_r.score:.3f}" if fft_r.score is not None else "N/A"
                st.markdown(f"**Branch Status:** `{fft_r.status}`")
                st.markdown(f"**Suspicion Score:** `{score_str}` / 1.00")
                st.markdown(f"**Findings:** {fft_r.explanation}")
                st.markdown("**Quantitative Metrics:**")
                st.json(fft_r.metrics)
                st.info(f"**Methodology:** {fft_r.method_description}")
                st.warning(f"**Known Limitations:** {fft_r.limitations}")

    with tab_texture:
        st.markdown("#### **Branch E — Local Binary Pattern (LBP) Micro-Texture**")
        col_img, col_info = st.columns([1.2, 1.0])
        tex_r = report.branch_results.get("texture")
        with col_img:
            if tex_r and tex_r.visualization is not None:
                st.image(tex_r.visualization, caption="Normalized LBP Feature Map", use_container_width=True)
        with col_info:
            if tex_r:
                score_str = f"{tex_r.score:.3f}" if tex_r.score is not None else "N/A"
                st.markdown(f"**Branch Status:** `{tex_r.status}`")
                st.markdown(f"**Suspicion Score:** `{score_str}` / 1.00")
                st.markdown(f"**Findings:** {tex_r.explanation}")
                st.markdown("**Quantitative Metrics:**")
                st.json(tex_r.metrics)
                st.info(f"**Methodology:** {tex_r.method_description}")
                st.warning(f"**Known Limitations:** {tex_r.limitations}")

    with tab_structure:
        st.markdown("#### **Branch F — Structural Consistency & Vegetation Seams**")
        col_img, col_info = st.columns([1.2, 1.0])
        str_r = report.branch_results.get("structural")
        with col_img:
            if str_r and str_r.visualization is not None:
                st.image(str_r.visualization, caption="Excess Green (ExG) and Sobel Boundary Map", use_container_width=True)
        with col_info:
            if str_r:
                score_str = f"{str_r.score:.3f}" if str_r.score is not None else "N/A"
                st.markdown(f"**Branch Status:** `{str_r.status}`")
                st.markdown(f"**Suspicion Score:** `{score_str}` / 1.00")
                st.markdown(f"**Findings:** {str_r.explanation}")
                st.markdown("**Quantitative Metrics:**")
                st.json(str_r.metrics)
                st.info(f"**Methodology:** {str_r.method_description}")
                st.warning(f"**Known Limitations:** {str_r.limitations}")

    with tab_geometry:
        st.markdown("#### **Branch G — Spatial Coherence & Cross-Channel Alignment**")
        col_img, col_info = st.columns([1.2, 1.0])
        geom_r = report.branch_results.get("geometry")
        with col_img:
            if geom_r and geom_r.visualization is not None:
                st.image(geom_r.visualization, caption="Differential Step Seam Map", use_container_width=True)
        with col_info:
            if geom_r:
                score_str = f"{geom_r.score:.3f}" if geom_r.score is not None else "N/A"
                st.markdown(f"**Branch Status:** `{geom_r.status}`")
                st.markdown(f"**Suspicion Score:** `{score_str}` / 1.00")
                st.markdown(f"**Findings:** {geom_r.explanation}")
                st.markdown("**Quantitative Metrics:**")
                st.json(geom_r.metrics)
                st.info(f"**Methodology:** {geom_r.method_description}")
                st.warning(f"**Known Limitations:** {geom_r.limitations}")

    with tab_charts:
        st.markdown("#### **Multi-Branch Fusion & Attribution Breakdown**")
        c_bar, c_pie = st.columns([1.4, 1.0])
        with c_bar:
            score_chart = create_branch_score_chart(report.fusion)
            st.image(score_chart, use_container_width=True)
        with c_pie:
            donut_chart = create_fusion_donut_chart(report.fusion)
            st.image(donut_chart, use_container_width=True)

        st.markdown("##### **Branch Fusion Calculation Table**")
        fusion_rows = []
        for b in report.fusion.branch_contributions:
            fusion_rows.append(
                {
                    "Branch Category": b.branch_name,
                    "Status": b.status,
                    "Raw Score": f"{b.raw_score:.3f}" if b.raw_score is not None else "Excluded",
                    "Base Weight": f"{b.base_weight:.1%}",
                    "Normalized Weight": f"{b.effective_weight:.1%}",
                    "Weighted Contribution": f"{b.weighted_contribution:.3f}" if b.weighted_contribution > 0 else "-",
                }
            )
        st.dataframe(fusion_rows, use_container_width=True)

    with tab_report:
        st.markdown("#### **Export Official Verification Audit PDF**")
        st.write(
            "Download an authoritative, publication-quality 2-page PDF audit certificate "
            "documenting analysis parameters, branch scores, visualizations, and disclaimers."
        )

        try:
            report_gen = PDFReportGenerator()
            pdf_bytes = report_gen.generate(report)
            pdf_filename = f"SatVerify_Audit_{report.analysis_id}.pdf"

            st.download_button(
                label=f"📥 Download Audit Report ({pdf_filename})",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
            )
            st.success(f"Audit certificate generated ({len(pdf_bytes) / 1024:.1f} KB). Ready for download.")
        except Exception as exc:
            st.error(f"Failed to generate PDF report: {exc}")


def page_architecture() -> None:
    """Explains model architecture, checkpoints, and transparency principles."""
    st.markdown("### **🧠 Model Architecture & Deep Learning Registry**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="forensic-card">
                <h4 style="color: #69F0AE; margin-top:0;">Branch A: ResNet50 Transfer Learning</h4>
                <p style="font-size: 0.9rem; color: #CBD5E1;">
                    <strong>Architecture:</strong> ResNet50 Backbone (ImageNet Pretrained) + 
                    GlobalAveragePooling2D + Dropout(0.4) + Dense(128, ReLU) + Dropout(0.2) + Dense(1, Sigmoid).
                </p>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    <strong>Input Dimensions:</strong> 224 × 224 × 3 with Caffe zero-centering preprocessing.<br>
                    <strong>Checkpoint:</strong> <code>checkpoints/satverify_resnet50_best.h5</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="forensic-card">
                <h4 style="color: #00B0FF; margin-top:0;">Branch D: Vision Transformer (ViT)</h4>
                <p style="font-size: 0.9rem; color: #CBD5E1;">
                    <strong>Architecture:</strong> HuggingFace <code>google/vit-base-patch16-224</code>.<br>
                    Self-attention over 16×16 spatial image patches for long-range spatial coherence.
                </p>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    <strong>Input Dimensions:</strong> 224 × 224 × 3 normalized.<br>
                    <strong>Checkpoint:</strong> <code>checkpoints/satverify_vit_best.pt</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### **The Model Honesty Principle**")
    st.warning(
        "CRITICAL REQUIREMENT: An ImageNet model without verified satellite deepfake fine-tuning "
        "is NEVER treated as an authenticity classifier. Untrained neural heads are flagged as "
        "UNAVAILABLE, and the pipeline gracefully relies on deterministic mathematical forensics "
        "(ELA, FFT, LBP, and Structural analysis) until domain weights are installed."
    )


def page_methodology() -> None:
    """Details forensic methods, limitations, and mathematical formulations."""
    st.markdown("### **📐 Forensic Methodology & Calibration Standards**")

    with st.expander("🔬 Error Level Analysis (ELA)"):
        st.markdown(
            """
            - **Hypothesis:** Authentic sensor acquisitions possess homogeneous compression history. Spliced or generative 
              pixels degrade at distinct rates upon JPEG recompression.
            - **Formula:** $\\Delta(x, y) = |I_{\\text{orig}}(x, y) - I_{\\text{JPEG-90}}(x, y)| \\times 10$
            - **Metric:** Standard deviation $\\sigma_{\\text{ELA}}$ and regional block variance.
            - **Limitation:** Can be affected by multi-generation resaving cycles.
            """
        )

    with st.expander("📡 Fourier Transform Frequency Analysis (FFT)"):
        st.markdown(
            """
            - **Hypothesis:** Convolutional upsampling operators leave periodic grid spikes in the 2D power spectrum, 
              violating the natural $1/f^\\alpha$ spectral decay.
            - **Formula:** $F(u, v) = \\sum_{x} \\sum_{y} f(x, y) e^{-j 2\\pi (ux/M + vy/N)}$
            - **Magnitude:** $M(u, v) = 20 \\times \\ln(1 + |F(u, v)|)$
            - **Limitation:** Regular agricultural rows can create natural directional spikes.
            """
        )

    with st.expander("🌿 Local Binary Pattern (LBP) Texture Analysis"):
        st.markdown(
            """
            - **Hypothesis:** Real forest canopies exhibit complex fractal micro-texture. AI synthesis often produces 
              unnatural micro-smoothing or repeated cloned patches.
            - **Formula:** $\\text{LBP}_{P, R} = \\sum_{p=0}^{P-1} s(g_p - g_c) 2^p$
            - **Metric:** Regional Shannon entropy $H = -\\sum p_i \\log_2 p_i$.
            """
        )


def page_about() -> None:
    """Contextualizes the paper forest problem in carbon credit markets."""
    st.markdown("### **🌍 The Carbon Credit Crisis & SatVerify**")
    st.markdown(
        """
        Carbon credit certification relies heavily on remote sensing to audit afforestation and reforestation claims.
        However, the advent of generative vision models (GANs, Diffusion) enables malicious actors to generate 
        fictitious **"paper forests"** that appear biologically healthy to casual visual inspection.

        **SatVerify's Mission:**
        1. **Forensic Transparency:** Decouple verification from opaque black boxes through explainable math.
        2. **Tri-State Realism:** Reject forced binary decisions; empower auditors with confidence intervals and INCONCLUSIVE ratings.
        3. **Compliance Readiness:** Provide exportable forensic audit certificates to uphold carbon market trust.
        """
    )


def main() -> None:
    """Application entrypoint."""
    inject_custom_styles()
    render_hero_banner()
    thresholds = render_sidebar_controls()
    pipeline = get_pipeline()

    # Main Navigation
    menu = st.radio(
        "Navigation",
        options=[
            "🛰️ Authenticity Analyzer",
            "🧠 Model Architecture",
            "📐 Methodology & Math",
            "🌍 Problem Statement & Impact",
        ],
        horizontal=True,
    )

    st.divider()

    if menu == "🛰️ Authenticity Analyzer":
        page_analyzer(pipeline, thresholds)
    elif menu == "🧠 Model Architecture":
        page_architecture()
    elif menu == "📐 Methodology & Math":
        page_methodology()
    elif menu == "🌍 Problem Statement & Impact":
        page_about()


if __name__ == "__main__":
    main()
