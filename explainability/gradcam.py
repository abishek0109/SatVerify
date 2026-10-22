"""Grad-CAM (Gradient-weighted Class Activation Mapping) Module.

Generates visual explanations highlighting regions that influenced CNN classification.
Handles missing weights and uninstalled frameworks with explicit status explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

from forensics.ela import colorize_grayscale
from utils.logging_config import get_logger

logger = get_logger("explainability.gradcam")


@dataclass
class GradCAMResult:
    """Outcome of Grad-CAM explainability computation."""

    is_available: bool
    status_message: str
    heatmap_raw: Optional[np.ndarray] = None  # (H, W) float32 in [0, 1]
    heatmap_rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8 RGB
    overlay_rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8 RGB blended with original
    target_layer_name: Optional[str] = None


class GradCAMExplainer:
    """Computes Grad-CAM activations for ResNet50 models."""

    def __init__(self, classifier: Any = None) -> None:
        self.classifier = classifier

    def generate(
        self,
        pil_image: Image.Image,
        np_resnet_input: Optional[np.ndarray] = None,
    ) -> GradCAMResult:
        """Compute Grad-CAM overlay or provide an informative fallback."""
        if self.classifier is None or not getattr(self.classifier, "is_trained_checkpoint_loaded", False):
            return GradCAMResult(
                is_available=False,
                status_message=(
                    "Grad-CAM unavailable: Requires a fine-tuned CNN model with valid gradient graphs. "
                    "Untrained ImageNet backbones are excluded to prevent misleading saliency artifacts. "
                    "Refer to the ELA, FFT, and Structural anomaly maps for explainable evidence."
                ),
            )

        # If trained model is present
        try:
            framework = getattr(self.classifier, "framework", None)
            model = getattr(self.classifier, "model", None)

            if framework == "tensorflow" and model is not None:
                import tensorflow as tf

                # Target last conv layer in ResNet50
                target_layer = None
                for layer in reversed(model.layers):
                    if "conv" in layer.name or isinstance(layer, tf.keras.layers.Conv2D):
                        target_layer = layer
                        break

                if target_layer is None:
                    return GradCAMResult(
                        is_available=False,
                        status_message="No suitable convolutional layer identified for Grad-CAM.",
                    )

                grad_model = tf.keras.models.Model(
                    inputs=[model.inputs],
                    outputs=[target_layer.output, model.output],
                )

                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(np_resnet_input)
                    loss = predictions[:, 0]

                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
                heatmap_np = heatmap.numpy()

            elif framework == "torch" and model is not None:
                import torch
                import torch.nn.functional as F

                # PyTorch ResNet50 hook
                activations = []
                gradients = []

                def forward_hook(module, input, output):
                    activations.append(output)

                def backward_hook(module, grad_in, grad_out):
                    gradients.append(grad_out[0])

                target_layer = model.backbone.layer4[-1]
                h_f = target_layer.register_forward_hook(forward_hook)
                h_b = target_layer.register_backward_hook(backward_hook)

                arr = np.transpose(np_resnet_input, (0, 3, 1, 2)).astype(np.float32)
                tensor = torch.from_numpy(arr)
                output = model(tensor)
                model.zero_grad()
                output.backward()

                h_f.remove()
                h_b.remove()

                grads = gradients[0].cpu().data.numpy()[0]
                acts = activations[0].cpu().data.numpy()[0]
                weights = np.mean(grads, axis=(1, 2))

                cam = np.zeros(acts.shape[1:], dtype=np.float32)
                for i, w in enumerate(weights):
                    cam += w * acts[i]

                cam = np.maximum(cam, 0)
                heatmap_np = cam / max(float(np.max(cam)), 1e-6)

            else:
                return GradCAMResult(
                    is_available=False,
                    status_message="Unrecognized model backend for gradient extraction.",
                )

            # Resize heatmap to original image dimensions
            orig_w, orig_h = pil_image.size
            heatmap_pil = Image.fromarray((heatmap_np * 255.0).astype(np.uint8)).resize(
                (orig_w, orig_h), resample=Image.Resampling.BILINEAR
            )
            heatmap_resized = np.asarray(heatmap_pil, dtype=np.float32) / 255.0

            # Colorize with turbo colormap
            heatmap_rgb = colorize_grayscale((heatmap_resized * 255).astype(np.uint8), "turbo")

            # Blend with original
            orig_rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
            alpha = 0.45
            overlay = np.clip(
                (1.0 - alpha) * orig_rgb + alpha * heatmap_rgb, 0, 255
            ).astype(np.uint8)

            return GradCAMResult(
                is_available=True,
                status_message="Grad-CAM visualization generated successfully.",
                heatmap_raw=heatmap_resized,
                heatmap_rgb=heatmap_rgb,
                overlay_rgb=overlay,
                target_layer_name=getattr(target_layer, "name", "last_conv"),
            )

        except Exception as exc:
            logger.exception("Grad-CAM generation failed: %s", exc)
            return GradCAMResult(
                is_available=False,
                status_message=f"Grad-CAM computation encountered an error: {exc}",
            )
