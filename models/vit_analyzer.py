"""Branch D — Vision Transformer (ViT) Analysis Module.

Implements patch-level self-attention analysis for satellite imagery authenticity.
Supports HuggingFace `google/vit-base-patch16-224` and local fine-tuned checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from utils.logging_config import get_logger

logger = get_logger("models.vit")


class ViTAnalyzer:
    """Vision Transformer module for patch-level and global anomaly assessment."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.settings = get_settings()
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model: Any = None
        self.processor: Any = None
        self.device: str = "cpu"
        self.is_trained_checkpoint_loaded: bool = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ViT dependencies and optionally load fine-tuned weights."""
        try:
            import torch
            from transformers import AutoConfig, ViTForImageClassification, ViTImageProcessor

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Check if checkpoint exists
            if self.checkpoint_path and self.checkpoint_path.exists():
                config = AutoConfig.from_pretrained(self.model_name)
                config.num_labels = 2
                self.model = ViTForImageClassification.from_pretrained(
                    self.model_name, config=config, ignore_mismatched_sizes=True
                )
                self.processor = ViTImageProcessor.from_pretrained(self.model_name)

                # Load weights
                state = torch.load(str(self.checkpoint_path), map_location=self.device)
                state_dict = state.get("model_state_dict", state)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.is_trained_checkpoint_loaded = True
                logger.info("Loaded fine-tuned ViT checkpoint: %s", self.checkpoint_path)

        except ImportError:
            logger.info("PyTorch or Transformers not installed. ViT module will operate in standby.")
        except Exception as exc:
            logger.warning("Could not initialize ViT model: %s", exc)

    def analyze(self, image_pil: Image.Image) -> ForensicResult:
        """Run ViT inference or transparently report unavailability."""
        if self.model is None or not self.is_trained_checkpoint_loaded:
            return ForensicResult(
                branch_id="vit",
                branch_name="Vision Transformer (ViT) Analysis",
                status="UNAVAILABLE",
                score=None,
                explanation=(
                    "No fine-tuned satellite forensic ViT checkpoint loaded. Generic ImageNet ViT "
                    "predictions are not valid deepfake indicators and are excluded from score fusion."
                ),
                method_description=(
                    "Vision Transformer (patch size 16x16) self-attention modeling for long-range "
                    "spatial inconsistencies across forest canopy and land-use boundaries."
                ),
                limitations=(
                    "Pretrained ImageNet transformers classify semantic objects (e.g. animals, vehicles), "
                    "not generative noise or compression anomalies. Domain fine-tuning is required."
                ),
                metrics={"is_trained_checkpoint_loaded": False},
                visualization=None,
            )

        try:
            import torch

            encoded = self.processor(images=image_pil, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(self.device)

            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Class 1 = FAKE
            fake_prob = float(probs[0][1].item())
            score = round(min(max(fake_prob, 0.0), 1.0), 3)

            return ForensicResult(
                branch_id="vit",
                branch_name="Vision Transformer (ViT) Analysis",
                status="AVAILABLE",
                score=score,
                explanation=f"ViT model evaluated. Synthetic anomaly probability: {score:.1%}.",
                method_description="ViT patch-level self-attention analysis on satellite imagery.",
                limitations="Performance depends on training dataset fidelity.",
                metrics={
                    "is_trained_checkpoint_loaded": True,
                    "fake_probability": round(fake_prob, 4),
                },
                visualization=None,
            )

        except Exception as exc:
            logger.exception("ViT analysis failed: %s", exc)
            return ForensicResult(
                branch_id="vit",
                branch_name="Vision Transformer (ViT) Analysis",
                status="ERROR",
                score=None,
                explanation=f"Error executing ViT analysis: {exc}",
                method_description="Vision Transformer inference.",
                limitations="Requires GPU/CPU tensor execution.",
            )
