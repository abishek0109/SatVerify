"""Configuration settings and environment management for SatVerify.

Centralizes all application paths, branch weights, decision thresholds,
and forensic parameter definitions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class AppConfig:
    """Immutable application settings container."""

    # Project metadata
    APP_NAME: str = "SatVerify"
    APP_TAGLINE: str = "AI Forensics for Trusted Carbon Markets"
    VERSION: str = "2.0.0"

    # Base filesystem paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    CHECKPOINTS_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "checkpoints"
    )
    REPORTS_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "reports"
    )
    ARTIFACTS_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts"
    )
    DOCS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "docs")

    # Image upload limits and constraints
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    )
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MAX_IMAGE_DIMENSION: int = 4096  # Max width or height in pixels
    CNN_INPUT_SHAPE: Tuple[int, int] = (224, 224)

    # Forensic Branch Initial Weights (as specified in SatVerify architecture)
    # Total sum = 1.00 (100%)
    BRANCH_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "ela": 0.22,
            "fft": 0.22,
            "vit": 0.18,
            "structural": 0.16,
            "texture_geometry": 0.12,
            "shadow_elevation": 0.10,
        }
    )

    # Decision Engine Tri-State Thresholds
    # Scores are normalized in [0.0, 1.0], where higher represents higher anomaly/suspicion.
    DECISION_REAL_THRESHOLD: float = 0.38
    DECISION_FAKE_THRESHOLD: float = 0.62
    MIN_BRANCHES_REQUIRED: int = 2

    # ELA Parameters
    ELA_JPEG_QUALITY: int = 90
    ELA_AMPLIFICATION_FACTOR: int = 10

    # FFT Parameters
    FFT_LOG_FACTOR: float = 20.0
    FFT_SPIKE_SIGMA: float = 3.0

    # LBP Parameters
    LBP_RADIUS: int = 2
    LBP_POINTS: int = 16
    LBP_METHOD: str = "uniform"

    # Default checkpoint filename
    RESNET_CHECKPOINT_NAME: str = "satverify_resnet50_best.h5"
    VIT_CHECKPOINT_NAME: str = "satverify_vit_best.pt"

    def ensure_directories(self) -> None:
        """Create necessary project directories if they do not exist."""
        for directory in [
            self.DATA_DIR,
            self.CHECKPOINTS_DIR,
            self.REPORTS_DIR,
            self.ARTIFACTS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


_SETTINGS_INSTANCE: AppConfig | None = None


def get_settings() -> AppConfig:
    """Retrieve the singleton AppConfig instance."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        _SETTINGS_INSTANCE = AppConfig()
        _SETTINGS_INSTANCE.ensure_directories()
    return _SETTINGS_INSTANCE
