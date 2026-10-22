"""Visualization and chart generation utilities for explainable forensics."""

from __future__ import annotations

import io
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server and thread safety
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core.score_fusion import FusionResult


def overlay_heatmap_on_image(
    original_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Alpha-blend a colorized heatmap over an original RGB image array."""
    if original_rgb.shape[:2] != heatmap_rgb.shape[:2]:
        # Resize heatmap
        h, w = original_rgb.shape[:2]
        hm_pil = Image.fromarray(heatmap_rgb).resize((w, h), resample=Image.Resampling.BILINEAR)
        heatmap_rgb = np.asarray(hm_pil, dtype=np.uint8)

    orig_f = original_rgb.astype(np.float32)
    heat_f = heatmap_rgb.astype(np.float32)
    blended = np.clip((1.0 - alpha) * orig_f + alpha * heat_f, 0, 255).astype(np.uint8)
    return blended


def create_branch_score_chart(fusion: FusionResult) -> Image.Image:
    """Generate a modern horizontal bar chart displaying branch scores and weights."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor("#121820")
    ax.set_facecolor("#1A232E")

    branch_names = []
    scores = []
    colors = []
    labels = []

    for item in reversed(fusion.branch_contributions):
        # Shorten name for clean axis display
        display_name = item.branch_name.split("(")[0].strip()
        branch_names.append(display_name)

        if item.raw_score is not None:
            s = item.raw_score
            scores.append(s)
            if s >= 0.62:
                colors.append("#D50000")  # Red: suspicious
            elif s <= 0.38:
                colors.append("#00C853")  # Green: natural
            else:
                colors.append("#FFAB00")  # Amber: uncertain
            eff_w_pct = item.effective_weight * 100
            labels.append(f"{s:.2f} (wt {eff_w_pct:.0f}%)")
        else:
            scores.append(0.0)
            colors.append("#455A64")  # Gray: unavailable
            labels.append("Excluded / N/A")

    y_pos = np.arange(len(branch_names))
    bars = ax.barh(y_pos, scores, color=colors, height=0.55, edgecolor="#2E3C4D")

    # Add threshold reference lines
    ax.axvline(0.38, color="#00C853", linestyle="--", linewidth=1.2, alpha=0.7, label="REAL threshold (0.38)")
    ax.axvline(0.62, color="#D50000", linestyle="--", linewidth=1.2, alpha=0.7, label="FAKE threshold (0.62)")

    ax.set_xlim(0.0, 1.05)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(branch_names, color="#E0E6ED", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Suspicion Score (0.0 = Authentic, 1.0 = Altered/Synthetic)", color="#90A4AE", fontsize=9)
    ax.set_title("Forensic Branch Suspicion Breakdown", color="#FFFFFF", fontsize=11, fontweight="bold", pad=12)

    # Style axes and grid
    ax.tick_params(colors="#90A4AE", labelsize=8.5)
    ax.grid(axis="x", color="#2E3C4D", linestyle=":", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color("#2E3C4D")

    # Value annotations on bars
    for idx, (bar, label_text) in enumerate(zip(bars, labels)):
        w = bar.get_width()
        x_text = max(w + 0.02, 0.05) if w < 0.85 else w - 0.18
        t_color = "#FFFFFF" if w >= 0.85 else "#CFD8DC"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2.0, label_text,
                va="center", ha="left" if w < 0.85 else "center", color=t_color, fontsize=8.5, fontweight="bold")

    ax.legend(loc="upper right", facecolor="#121820", edgecolor="#2E3C4D", labelcolor="#CFD8DC", fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def create_fusion_donut_chart(fusion: FusionResult) -> Image.Image:
    """Generate a donut chart showing active branch weights."""
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    fig.patch.set_facecolor("#121820")
    ax.set_facecolor("#121820")

    active_items = [b for b in fusion.branch_contributions if b.effective_weight > 0]
    if not active_items:
        # Fallback empty figure
        ax.text(0.5, 0.5, "No Active Branches", color="#CFD8DC", ha="center", va="center")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    labels = [b.branch_name.split("(")[0].strip() for b in active_items]
    weights = [b.effective_weight for b in active_items]
    palette = ["#00E676", "#00B0FF", "#FFD600", "#FF6D00", "#E040FB", "#76FF03"]

    wedges, texts, autotexts = ax.pie(
        weights,
        labels=labels,
        autopct="%1.0f%%",
        pctdistance=0.75,
        colors=palette[: len(weights)],
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor="#121820", linewidth=1.5),
    )

    for t in texts:
        t.set_color("#CFD8DC")
        t.set_fontsize(8.5)
    for at in autotexts:
        at.set_color("#121820")
        at.set_fontweight("bold")
        at.set_fontsize(8.5)

    ax.set_title("Active Weight Distribution (100% Normalized)", color="#FFFFFF", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def create_comparison_figure(
    original_pil: Image.Image,
    ela_rgb: Optional[np.ndarray],
    fft_rgb: Optional[np.ndarray],
    struct_rgb: Optional[np.ndarray],
    verdict: str,
    fused_score: Optional[float],
) -> Image.Image:
    """Create a consolidated side-by-side comparison panel for reports and previews."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
    fig.patch.set_facecolor("#0F141C")

    # Panel 1: Original
    axes[0].imshow(original_pil)
    axes[0].set_title("Input Satellite Image", color="#FFFFFF", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: ELA
    if ela_rgb is not None:
        axes[1].imshow(ela_rgb)
    axes[1].set_title("Error Level Analysis (ELA)", color="#FFFFFF", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: FFT
    if fft_rgb is not None:
        axes[2].imshow(fft_rgb)
    axes[2].set_title("Fourier Spectrum (FFT)", color="#FFFFFF", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Structural
    if struct_rgb is not None:
        axes[3].imshow(struct_rgb)
    axes[3].set_title("Structural Anomaly Map", color="#FFFFFF", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    score_str = f"Suspicion Score: {fused_score:.2f}" if fused_score is not None else "Score: N/A"
    fig.suptitle(
        f"SatVerify Forensic Inspection — Verdict: {verdict} ({score_str})",
        color="#00E676" if verdict == "REAL" else ("#FF1744" if verdict == "FAKE" else "#FFAB00"),
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
