"""Model registry and centralized loader for SatVerify deep learning models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config.settings import get_settings
from models.resnet_classifier import ResNet50Classifier
from models.vit_analyzer import ViTAnalyzer
from utils.logging_config import get_logger

logger = get_logger("models.loader")


class ModelRegistry:
    """Manages cached instances of deep learning and forensic model wrappers."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._resnet_instance: Optional[ResNet50Classifier] = None
        self._vit_instance: Optional[ViTAnalyzer] = None

    def get_resnet_classifier(self) -> ResNet50Classifier:
        """Retrieve or instantiate cached ResNet50Classifier."""
        if self._resnet_instance is None:
            checkpoint = self.settings.CHECKPOINTS_DIR / self.settings.RESNET_CHECKPOINT_NAME
            self._resnet_instance = ResNet50Classifier(
                checkpoint_path=checkpoint if checkpoint.exists() else None
            )
        return self._resnet_instance

    def get_vit_analyzer(self) -> ViTAnalyzer:
        """Retrieve or instantiate cached ViTAnalyzer."""
        if self._vit_instance is None:
            checkpoint = self.settings.CHECKPOINTS_DIR / self.settings.VIT_CHECKPOINT_NAME
            self._vit_instance = ViTAnalyzer(
                checkpoint_path=checkpoint if checkpoint.exists() else None
            )
        return self._vit_instance


_REGISTRY_INSTANCE: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Retrieve singleton ModelRegistry instance."""
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = ModelRegistry()
    return _REGISTRY_INSTANCE
