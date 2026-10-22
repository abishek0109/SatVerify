from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import cv2
import numpy as np
from PIL import Image


ELA_AMPLIFICATION_FACTOR: Final[int] = 10


def load_image_as_pil(image_path: str) -> Image.Image:
    """Load an image from disk and normalize it to RGB mode.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL.Image.Image: An RGB image copy detached from the file handle.

    Why:
        Satellite image uploads can arrive in many formats and channel layouts
        (RGB, RGBA, palette, grayscale). Converting to RGB up-front gives the
        model and forensic pipelines a consistent input contract.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to open image '{path}': {exc}") from exc


def perform_ela(image_path: str, quality: int = 90) -> np.ndarray:
    """Compute Error Level Analysis (ELA) for an image.

    Args:
        image_path: Path to the input image.
        quality: JPEG quality used during re-compression (1-100).

    Returns:
        np.ndarray: Amplified ELA image as uint8 RGB array.

    Why:
        Real satellite images (for example, products derived from Sentinel-2 or
        Landsat pipelines) often carry consistent compression behavior tied to
        physical sensor capture and downstream processing. Generative pipelines
        (GAN/Diffusion) synthesize pixels without that same acquisition chain,
        which can produce uneven re-compression residuals around textures,
        boundaries, and repeated patterns. ELA highlights these residual
        inconsistencies, making authenticity anomalies easier to inspect.
    """
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100.")

    original = load_image_as_pil(image_path)
    temp_jpeg_path: Path | None = None

    try:
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_jpeg_path = Path(tmp_file.name)

        # Re-compress the source image with a controlled JPEG quality.
        original.save(temp_jpeg_path, format="JPEG", quality=quality)

        recompressed = load_image_as_pil(str(temp_jpeg_path))

        # Use signed math before absolute difference to prevent uint underflow.
        original_array = np.asarray(original, dtype=np.int16)
        recompressed_array = np.asarray(recompressed, dtype=np.int16)
        diff = np.abs(original_array - recompressed_array)

        # Amplify forensic signal so subtle artifacts become visually inspectable.
        amplified = np.clip(diff * ELA_AMPLIFICATION_FACTOR, 0, 255).astype(np.uint8)
        return amplified

    except Exception as exc:
        raise RuntimeError(f"Failed to perform ELA for '{image_path}': {exc}") from exc

    finally:
        if temp_jpeg_path is not None:
            try:
                temp_jpeg_path.unlink(missing_ok=True)
            except Exception:
                # Best-effort cleanup only; never hide main forensic errors.
                pass


def perform_frequency_analysis(image_path: str) -> np.ndarray:
    """Create a Fourier heatmap highlighting spectral anomalies.

    Args:
        image_path: Path to the input image.

    Returns:
        np.ndarray: Colorized FFT magnitude heatmap (BGR uint8).

    Why:
        GAN-style generation commonly uses convolution and upsampling operators
        that can leave periodic, grid-like spectral traces ("GAN fingerprints")
        not obvious in pixel space. FFT transforms the image into frequency
        space, where these artifacts may appear as unusual spikes and structured
        bands. This makes hidden synthetic patterns easier to detect in
        satellite imagery authenticity checks.
    """
    try:
        image = load_image_as_pil(image_path)

        # Convert to a single intensity channel for stable frequency analysis.
        grayscale = np.asarray(image.convert("L"), dtype=np.float32)

        # FFT maps spatial patterns into frequency components.
        fft_result = np.fft.fft2(grayscale)
        fft_shifted = np.fft.fftshift(fft_result)

        # Log scaling compresses dynamic range for interpretable visualization.
        magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1.0)

        # Normalize to 8-bit display range.
        normalized = cv2.normalize(
            magnitude_spectrum, None, 0, 255, norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Color map improves human interpretability of spectral hotspots.
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return heatmap

    except Exception as exc:
        raise RuntimeError(
            f"Failed to perform frequency analysis for '{image_path}': {exc}"
        ) from exc


def compute_ela_score(image_path: str) -> float:
    """Compute a normalized fake-likelihood score from ELA residual variance.

    Args:
        image_path: Path to the input image.

    Returns:
        float: Fake-likelihood score in the range [0.0, 1.0].

    Why:
        Real satellite scenes captured by physical sensors (for example,
        Sentinel-2 and Landsat pipelines) tend to exhibit relatively uniform,
        low-variance ELA residuals because compression behavior is consistent
        across the frame. AI-generated imagery often shows higher residual
        variance because GAN/Diffusion synthesis introduces uneven pixel
        statistics around boundaries, texture transitions, and synthetic
        vegetation edges. A high ELA standard deviation is therefore a strong
        indicator of fake content. This is a zero-shot forensic signal, so it
        works without any task-specific training data.
    """
    try:
        ela_image = perform_ela(image_path)

        # Convert ELA output to grayscale so variance is measured on one channel.
        if ela_image.ndim == 3 and ela_image.shape[2] == 3:
            ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
        else:
            ela_gray = ela_image

        # Standard deviation captures spread/irregularity in compression residuals.
        std_value = float(np.std(ela_gray))

        # Map typical ELA std range (0-80) to a unit interval score.
        normalized_score = min(std_value / 80.0, 1.0)

        # Sharpen separation around the midpoint to make decisions less ambiguous.
        sharpened_score = 1.0 / (1.0 + np.exp(-10.0 * (normalized_score - 0.5)))
        return float(sharpened_score)
    except Exception as exc:
        raise RuntimeError(f"Failed to compute ELA score for '{image_path}': {exc}") from exc


def compute_fft_score(image_path: str) -> float:
    """Compute a normalized fake-likelihood score from FFT spike statistics.

    Args:
        image_path: Path to the input image.

    Returns:
        float: Fake-likelihood score in the range [0.0, 1.0].

    Why:
        GAN generators can leave periodic "fingerprints" in frequency space due
        to strided convolutions and upsampling, often visible as anomalous
        spikes. Diffusion outputs can also exhibit high-frequency noise
        structures that differ from the natural 1/f roll-off of real satellite
        sensor imagery. Authentic satellite images usually show smoother power
        spectral density trends, while unusual spike concentration indicates
        potential synthesis. This is a deterministic, explainable classical
        signal-processing method suitable for non-technical judging.
    """
    try:
        image = load_image_as_pil(image_path)

        # Convert to grayscale float for stable FFT operations.
        grayscale = np.asarray(image.convert("L"), dtype=np.float32)

        # Transform image into frequency domain and center low frequencies.
        fft_result = np.fft.fft2(grayscale)
        fft_shifted = np.fft.fftshift(fft_result)

        # Magnitude spectrum expresses energy at each frequency coordinate.
        magnitude = np.abs(fft_shifted)

        # Global mean/std define a dynamic threshold for abnormal spikes.
        mean_value = float(np.mean(magnitude))
        std_value = float(np.std(magnitude))
        threshold = mean_value + (3.0 * std_value)

        # Count how many frequencies are unusually strong outliers.
        spike_count = int(np.count_nonzero(magnitude > threshold))

        # Ratio normalizes out image-size differences.
        spike_ratio = float(spike_count / magnitude.size)

        # Map spike ratio (0 to 0.001+) into [0, 1].
        normalized_score = min(spike_ratio / 0.001, 1.0)

        # Sharpen separation near the midpoint for clearer fake confidence.
        sharpened_score = 1.0 / (1.0 + np.exp(-10.0 * (normalized_score - 0.5)))
        return float(sharpened_score)
    except Exception as exc:
        raise RuntimeError(f"Failed to compute FFT score for '{image_path}': {exc}") from exc
