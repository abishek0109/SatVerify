"""Master SatVerify Analysis Pipeline.

Orchestrates image validation, preprocessing, multi-branch forensic execution,
weighted score fusion, and tri-state decision classification into a unified result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
from PIL import Image

from config.settings import get_settings
from core.decision_engine import DecisionEngine, DecisionResult
from core.preprocessing import ImagePreprocessor, PreprocessedImageBundle
from core.score_fusion import FusionResult, WeightedScoreFuser
from forensics.base import ForensicResult
from forensics.ela import ErrorLevelAnalyzer
from forensics.elevation_analysis import ElevationTerrainValidator
from forensics.frequency_analysis import FrequencyDomainAnalyzer
from forensics.geometry_analysis import GeometryCoherenceAnalyzer
from forensics.shadow_analysis import ShadowAnalyzer
from forensics.structure_analysis import StructuralConsistencyAnalyzer
from forensics.texture_analysis import TextureAnalyzer
from models.model_loader import get_model_registry
from utils.logging_config import get_logger
from utils.validation import ImageValidationResult, validate_image_file

logger = get_logger("core.pipeline")


@dataclass
class PipelineReport:
    """Complete, end-to-end outcome of a SatVerify analysis session."""

    analysis_id: str
    timestamp: str
    execution_time_seconds: float
    validation: ImageValidationResult
    bundle: Optional[PreprocessedImageBundle]
    branch_results: Dict[str, ForensicResult]
    fusion: FusionResult
    decision: DecisionResult
    metadata: Dict[str, Any] = field(default_factory=dict)


class SatVerifyPipeline:
    """Production-grade multi-branch image forensics pipeline for satellite verification."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.preprocessor = ImagePreprocessor()

        # Forensics modules
        self.ela_analyzer = ErrorLevelAnalyzer()
        self.fft_analyzer = FrequencyDomainAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        self.structure_analyzer = StructuralConsistencyAnalyzer()
        self.geometry_analyzer = GeometryCoherenceAnalyzer()
        self.shadow_analyzer = ShadowAnalyzer()
        self.elevation_validator = ElevationTerrainValidator()

        # Decision & Fusion
        self.fuser = WeightedScoreFuser()
        self.decision_engine = DecisionEngine()

    def analyze(
        self,
        file_input: Union[str, Path, bytes],
        original_filename: Optional[str] = None,
        dem_input: Optional[Union[np.ndarray, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineReport:
        """Run complete authenticity verification on a satellite image."""
        start_time = time.time()
        now_utc = datetime.now(timezone.utc)
        analysis_id = f"SAT-{now_utc.strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}"
        timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

        # 1. Image Validation
        validation = validate_image_file(file_input, original_filename=original_filename)
        if not validation.is_valid:
            logger.warning("Image validation failed for %s: %s", original_filename, validation.error_message)
            dummy_fusion = FusionResult(
                fused_score=None,
                available_branches_count=0,
                total_branches_count=6,
                is_sufficient_evidence=False,
                summary_message=f"Validation failed: {validation.error_message}",
            )
            dummy_decision = DecisionResult(
                verdict="INCONCLUSIVE",
                headline="INCONCLUSIVE — Invalid or Corrupted Image",
                confidence_percentage=0.0,
                fused_score=None,
                lower_threshold=self.settings.DECISION_REAL_THRESHOLD,
                upper_threshold=self.settings.DECISION_FAKE_THRESHOLD,
                reason=validation.error_message or "Invalid image file.",
                disclaimer="Validation rejected file before forensic execution.",
            )
            return PipelineReport(
                analysis_id=analysis_id,
                timestamp=timestamp_str,
                execution_time_seconds=round(time.time() - start_time, 3),
                validation=validation,
                bundle=None,
                branch_results={},
                fusion=dummy_fusion,
                decision=dummy_decision,
            )

        # 2. Preprocessing
        bundle = self.preprocessor.process(file_input, source_path=original_filename)

        # 3. Execute Forensic Branches Safely
        branch_results: Dict[str, ForensicResult] = {}

        # Branch B: ELA
        try:
            branch_results["ela"] = self.ela_analyzer.analyze(bundle.pil_original)
        except Exception as exc:
            logger.exception("Branch ELA crashed: %s", exc)
            branch_results["ela"] = ForensicResult(
                branch_id="ela",
                branch_name="Error Level Analysis (ELA)",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # Branch C: FFT
        try:
            branch_results["fft"] = self.fft_analyzer.analyze(bundle.np_grayscale)
        except Exception as exc:
            logger.exception("Branch FFT crashed: %s", exc)
            branch_results["fft"] = ForensicResult(
                branch_id="fft",
                branch_name="Fourier Frequency Analysis (FFT)",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # Branch E: LBP Texture
        try:
            branch_results["texture"] = self.texture_analyzer.analyze(bundle.np_grayscale)
        except Exception as exc:
            logger.exception("Branch Texture crashed: %s", exc)
            branch_results["texture"] = ForensicResult(
                branch_id="texture",
                branch_name="Local Binary Pattern (LBP) Texture Analysis",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # Branch F: Structural Consistency
        try:
            branch_results["structural"] = self.structure_analyzer.analyze(bundle.np_original)
        except Exception as exc:
            logger.exception("Branch Structural crashed: %s", exc)
            branch_results["structural"] = ForensicResult(
                branch_id="structural",
                branch_name="Structural Consistency Analysis",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # Branch G: Geometry & Spatial Coherence
        try:
            branch_results["geometry"] = self.geometry_analyzer.analyze(bundle.np_original)
        except Exception as exc:
            logger.exception("Branch Geometry crashed: %s", exc)
            branch_results["geometry"] = ForensicResult(
                branch_id="geometry",
                branch_name="Geometry and Spatial Coherence Analysis",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # Branch H: Shadow & Solar Geometry
        try:
            branch_results["shadow"] = self.shadow_analyzer.analyze(bundle.np_original, metadata=metadata)
        except Exception as exc:
            logger.exception("Branch Shadow crashed: %s", exc)
            branch_results["shadow"] = ForensicResult(
                branch_id="shadow",
                branch_name="Shadow and Solar-Geometry Validation",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # Branch I: Elevation & DEM
        try:
            branch_results["elevation"] = self.elevation_validator.analyze(
                bundle.np_original, dem_input=dem_input, georef_metadata=metadata
            )
        except Exception as exc:
            logger.exception("Branch Elevation crashed: %s", exc)
            branch_results["elevation"] = ForensicResult(
                branch_id="elevation",
                branch_name="Elevation and Terrain Validation",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # Branch A: ResNet50 Classifier
        try:
            resnet = get_model_registry().get_resnet_classifier()
            branch_results["resnet50"] = resnet.predict(bundle.np_resnet_input)
        except Exception as exc:
            logger.exception("Branch ResNet50 crashed: %s", exc)
            branch_results["resnet50"] = ForensicResult(
                branch_id="resnet50",
                branch_name="ResNet50 CNN Classifier",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # Branch D: Vision Transformer (ViT)
        try:
            vit = get_model_registry().get_vit_analyzer()
            branch_results["vit"] = vit.analyze(bundle.pil_resized_224)
        except Exception as exc:
            logger.exception("Branch ViT crashed: %s", exc)
            branch_results["vit"] = ForensicResult(
                branch_id="vit",
                branch_name="Vision Transformer (ViT) Analysis",
                status="ERROR",
                score=None,
                explanation=f"Error: {exc}",
            )

        # 4. Multi-Branch Score Fusion
        results_list = list(branch_results.values())
        fusion = self.fuser.fuse(results_list)

        # 5. Tri-State Decision Engine
        decision = self.decision_engine.evaluate(fusion)

        elapsed = round(time.time() - start_time, 3)
        logger.info(
            "Pipeline analysis %s completed in %s s -> Verdict: %s (score=%s)",
            analysis_id,
            elapsed,
            decision.verdict,
            decision.fused_score,
        )

        return PipelineReport(
            analysis_id=analysis_id,
            timestamp=timestamp_str,
            execution_time_seconds=elapsed,
            validation=validation,
            bundle=bundle,
            branch_results=branch_results,
            fusion=fusion,
            decision=decision,
            metadata=metadata or {},
        )
