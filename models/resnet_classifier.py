"""Branch A — ResNet50 Deep Learning Binary Classifier.

Implements transfer-learning ResNet50 architecture with custom binary classification head
(GlobalAveragePooling -> Dropout -> Dense(128, ReLU) -> Dropout -> Dense(1, Sigmoid)).

Adheres strictly to the SatVerify transparency principle: an uncalibrated ImageNet backbone
without verified satellite deepfake weights is never presented as a genuine detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from utils.logging_config import get_logger

logger = get_logger("models.resnet")


class ResNet50Classifier:
    """ResNet50-based transfer learning classifier for satellite image authenticity."""

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        self.settings = get_settings()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model: Any = None
        self.framework: Optional[str] = None  # "tensorflow" or "torch"
        self.is_trained_checkpoint_loaded: bool = False
        self._initialize()

    def _initialize(self) -> None:
        """Attempt to initialize framework and load weights if available."""
        # Try TensorFlow / Keras first
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
            from tensorflow.keras.models import Model

            self.framework = "tensorflow"
            logger.info("TensorFlow available. Building ResNet50 architecture.")

            # Base model
            try:
                base_model = ResNet50(
                    weights="imagenet",
                    include_top=False,
                    input_shape=(224, 224, 3),
                )
                base_model.trainable = False
            except Exception as e:
                logger.warning("Could not download/load ImageNet weights: %s", e)
                base_model = ResNet50(
                    weights=None,
                    include_top=False,
                    input_shape=(224, 224, 3),
                )
                base_model.trainable = False

            # Custom classification head
            x = base_model.output
            x = GlobalAveragePooling2D(name="gap")(x)
            x = Dropout(0.4, name="dropout_1")(x)
            x = Dense(128, activation="relu", name="dense_128")(x)
            x = Dropout(0.2, name="dropout_2")(x)
            predictions = Dense(1, activation="sigmoid", name="authenticity_output")(x)

            self.model = Model(inputs=base_model.input, outputs=predictions)

        except ImportError:
            # Check PyTorch fallback
            try:
                import torch
                import torch.nn as nn
                from torchvision.models import ResNet50_Weights, resnet50

                self.framework = "torch"
                logger.info("PyTorch available. Building ResNet50 architecture.")

                class ResNet50Binary(nn.Module):
                    def __init__(self):
                        super().__init__()
                        try:
                            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
                        except Exception:
                            self.backbone = resnet50(weights=None)
                        in_features = self.backbone.fc.in_features
                        self.backbone.fc = nn.Identity()
                        self.head = nn.Sequential(
                            nn.Dropout(0.4),
                            nn.Linear(in_features, 128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, 1),
                            nn.Sigmoid(),
                        )

                    def forward(self, x):
                        feats = self.backbone(x)
                        return self.head(feats)

                self.model = ResNet50Binary()
                self.model.eval()

            except ImportError:
                self.framework = None
                logger.info("Neither TensorFlow nor PyTorch installed. Operating in heuristic mode.")

        # Attempt to load checkpoint if provided
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_weights(self.checkpoint_path)

    def _load_weights(self, path: Path) -> bool:
        """Load trained weights from disk."""
        try:
            if self.framework == "tensorflow" and self.model is not None:
                self.model.load_weights(str(path))
                self.is_trained_checkpoint_loaded = True
                logger.info("Loaded TensorFlow ResNet50 checkpoint from: %s", path)
                return True
            elif self.framework == "torch" and self.model is not None:
                import torch

                state = torch.load(str(path), map_location="cpu")
                self.model.load_state_dict(state.get("model_state_dict", state))
                self.is_trained_checkpoint_loaded = True
                logger.info("Loaded PyTorch ResNet50 checkpoint from: %s", path)
                return True
        except Exception as exc:
            logger.warning("Failed to load checkpoint from %s: %exc", path, exc)
            self.is_trained_checkpoint_loaded = False
        return False

    def predict(self, np_resnet_input: np.ndarray) -> ForensicResult:
        """Run ResNet50 inference.

        If a trained checkpoint is absent, this method transparently refuses to
        output ungrounded predictions, strictly adhering to scientific honesty.
        """
        # Case 1: Framework missing
        if self.framework is None or self.model is None:
            return ForensicResult(
                branch_id="resnet50",
                branch_name="ResNet50 CNN Classifier",
                status="UNAVAILABLE",
                score=None,
                explanation=(
                    "Deep learning framework (TensorFlow/PyTorch) not installed in runtime environment. "
                    "ResNet50 branch is excluded from score fusion."
                ),
                method_description="ResNet50 CNN with binary transfer-learning classification head.",
                limitations="Requires deep learning dependencies and GPU/CPU backend.",
                metrics={"framework": "none", "checkpoint_loaded": False},
            )

        # Case 2: Untrained / Base ImageNet weights without fine-tuned satellite weights
        if not self.is_trained_checkpoint_loaded:
            return ForensicResult(
                branch_id="resnet50",
                branch_name="ResNet50 CNN Classifier",
                status="UNAVAILABLE",
                score=None,
                explanation=(
                    "No trained satellite deepfake checkpoint loaded. An untrained classification head "
                    "cannot produce valid authenticity decisions. ResNet50 branch is excluded from fusion."
                ),
                method_description=(
                    "ResNet50 backbone + GlobalAveragePooling + Dropout + Dense(128, ReLU) + Dense(1, Sigmoid)."
                ),
                limitations=(
                    "ImageNet pretraining alone lacks domain sensitivity to satellite synthesis; "
                    "fine-tuned weights on verified Sentinel/Landsat and synthetic datasets are required."
                ),
                metrics={
                    "framework": self.framework,
                    "checkpoint_loaded": False,
                    "target_shape": (224, 224, 3),
                },
            )

        # Case 3: Trained checkpoint loaded — perform actual inference
        try:
            if self.framework == "tensorflow":
                pred = self.model.predict(np_resnet_input, verbose=0)
                raw_fake_score = float(pred[0][0])
            else:
                import torch

                # Convert (1, 224, 224, 3) to torch tensor (1, 3, 224, 224)
                arr = np.transpose(np_resnet_input, (0, 3, 1, 2)).astype(np.float32)
                tensor = torch.from_numpy(arr)
                with torch.no_grad():
                    pred = self.model(tensor)
                    raw_fake_score = float(pred[0][0].item())

            score = round(min(max(raw_fake_score, 0.0), 1.0), 3)

            explanation = (
                f"ResNet50 inference completed. Model fake probability: {score:.1%}. "
                f"Checkpoint: {self.checkpoint_path.name if self.checkpoint_path else 'default'}."
            )

            return ForensicResult(
                branch_id="resnet50",
                branch_name="ResNet50 CNN Classifier",
                status="AVAILABLE",
                score=score,
                explanation=explanation,
                method_description="ResNet50 CNN fine-tuned on satellite image tampering datasets.",
                limitations="Generalization depends on diversity and sensor distribution of training data.",
                metrics={
                    "framework": self.framework,
                    "checkpoint_loaded": True,
                    "raw_score": round(raw_fake_score, 4),
                },
            )

        except Exception as exc:
            logger.exception("ResNet50 prediction failed: %s", exc)
            return ForensicResult(
                branch_id="resnet50",
                branch_name="ResNet50 CNN Classifier",
                status="ERROR",
                score=None,
                explanation=f"Error executing ResNet50 inference: {exc}",
                method_description="ResNet50 CNN transfer learning.",
                limitations="Requires compatible input tensor shape.",
            )
