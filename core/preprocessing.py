"""Image loading, format normalization, and deep learning preprocessing."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from utils.logging_config import get_logger

logger = get_logger("preprocessing")


@dataclass
class PreprocessedImageBundle:
    """Standardized representation of an input image across pipelines."""

    pil_original: Image.Image  # Full-resolution PIL RGB
    np_original: np.ndarray  # Full-resolution uint8 RGB array (H, W, 3)
    np_grayscale: np.ndarray  # Full-resolution uint8 grayscale array (H, W)
    pil_resized_224: Image.Image  # Resized to 224x224 for CNN/ViT
    np_resnet_input: np.ndarray  # Preprocessed array (1, 224, 224, 3) float32
    original_size: Tuple[int, int]  # (width, height)
    source_path: Optional[str] = None


class ImagePreprocessor:
    """Prepares images for forensic analysis and neural network inference."""

    def __init__(self, target_shape: Tuple[int, int] = (224, 224)) -> None:
        self.target_shape = target_shape
        self.settings = get_settings()

    def load_image(self, input_source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """Load an image from any supported source into a decoupled RGB PIL Image."""
        if isinstance(input_source, Image.Image):
            return input_source.convert("RGB").copy()

        if isinstance(input_source, bytes):
            with Image.open(io.BytesIO(input_source)) as img:
                return img.convert("RGB").copy()

        path = Path(input_source)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        try:
            with Image.open(path) as img:
                return img.convert("RGB").copy()
        except Exception as exc:
            raise RuntimeError(f"Failed to load image from '{path}': {exc}") from exc

    def preprocess_for_resnet50(self, pil_image_224: Image.Image) -> np.ndarray:
        """Apply ResNet50 ImageNet preprocessing (caffe style).

        Converts RGB -> BGR and subtracts ImageNet channel means:
        B: 103.939, G: 116.779, R: 123.68
        """
        arr = np.asarray(pil_image_224, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        # Convert RGB to BGR
        bgr = arr[:, :, ::-1].copy()

        # Zero-center with ImageNet means (caffe mode)
        bgr[:, :, 0] -= 103.939
        bgr[:, :, 1] -= 116.779
        bgr[:, :, 2] -= 123.68

        # Add batch dimension: (1, 224, 224, 3)
        return np.expand_dims(bgr, axis=0)

    def process(
        self, input_source: Union[str, Path, bytes, Image.Image], source_path: Optional[str] = None
    ) -> PreprocessedImageBundle:
        """Create a complete PreprocessedImageBundle from the input source."""
        pil_original = self.load_image(input_source)
        width, height = pil_original.size

        # Convert to numpy arrays
        np_original = np.asarray(pil_original, dtype=np.uint8)

        # Grayscale representation for FFT, structure, and LBP
        # Using standard luminosity weights: 0.299 R + 0.587 G + 0.114 B
        if np_original.ndim == 3 and np_original.shape[2] >= 3:
            gray_float = (
                0.299 * np_original[:, :, 0]
                + 0.587 * np_original[:, :, 1]
                + 0.114 * np_original[:, :, 2]
            )
            np_grayscale = np.clip(gray_float, 0, 255).astype(np.uint8)
        else:
            np_grayscale = np_original.copy()

        # Resized 224x224 PIL image
        pil_resized_224 = pil_original.resize(self.target_shape, resample=Image.Resampling.BILINEAR)
        np_resnet_input = self.preprocess_for_resnet50(pil_resized_224)

        effective_path = source_path
        if effective_path is None and isinstance(input_source, (str, Path)):
            effective_path = str(input_source)

        return PreprocessedImageBundle(
            pil_original=pil_original,
            np_original=np_original,
            np_grayscale=np_grayscale,
            pil_resized_224=pil_resized_224,
            np_resnet_input=np_resnet_input,
            original_size=(width, height),
            source_path=effective_path,
        )
