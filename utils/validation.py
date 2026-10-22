"""Image file validation and metadata extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image, ExifTags

from config.settings import get_settings
from utils.logging_config import get_logger

logger = get_logger("validation")


@dataclass
class ImageValidationResult:
    """Outcome of validating an uploaded satellite image."""

    is_valid: bool
    error_message: Optional[str] = None
    width: int = 0
    height: int = 0
    format: str = ""
    mode: str = ""
    file_size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions_str(self) -> str:
        """Human-readable image dimensions."""
        return f"{self.width} × {self.height}" if self.is_valid else "N/A"


def extract_safe_exif(img: Image.Image) -> Dict[str, Any]:
    """Safely extract readable EXIF metadata without throwing exceptions."""
    exif_data: Dict[str, Any] = {}
    try:
        raw_exif = img.getexif()
        if not raw_exif:
            return exif_data

        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            # Omit huge byte payloads or thumbnail blobs
            if isinstance(value, (bytes, bytearray)) and len(value) > 64:
                continue
            exif_data[str(tag_name)] = str(value)
    except Exception as exc:
        logger.debug("Failed to extract EXIF metadata: %s", exc)
    return exif_data


def validate_image_file(
    file_input: Union[str, Path, bytes], original_filename: Optional[str] = None
) -> ImageValidationResult:
    """Validate that the input represents a valid, securely processable image.

    Args:
        file_input: Path to image file on disk or raw bytes.
        original_filename: Original filename if input is provided as raw bytes.

    Returns:
        ImageValidationResult containing validation status, dimensions, and metadata.
    """
    settings = get_settings()

    try:
        # 1. Determine size and filename
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                return ImageValidationResult(
                    is_valid=False, error_message=f"File not found: {path.name}"
                )
            file_size = path.stat().st_size
            filename = path.name
            target = path
        elif isinstance(file_input, bytes):
            file_size = len(file_input)
            filename = original_filename or "upload.bin"
            target = file_input
        else:
            return ImageValidationResult(
                is_valid=False, error_message="Unsupported input type for validation."
            )

        # 2. Check file size bounds
        if file_size == 0:
            return ImageValidationResult(
                is_valid=False, error_message="Uploaded file is empty (0 bytes)."
            )

        if file_size > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            return ImageValidationResult(
                is_valid=False,
                error_message=(
                    f"File size ({actual_mb:.1f} MB) exceeds maximum allowed limit ({max_mb:.0f} MB)."
                ),
            )

        # 3. Check file extension
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in settings.SUPPORTED_EXTENSIONS:
            return ImageValidationResult(
                is_valid=False,
                error_message=(
                    f"Unsupported image format '{suffix}'. "
                    f"Supported formats: {', '.join(settings.SUPPORTED_EXTENSIONS)}"
                ),
            )

        # 4. Attempt to open and verify image with PIL
        try:
            if isinstance(target, bytes):
                import io

                pil_img = Image.open(io.BytesIO(target))
            else:
                pil_img = Image.open(target)

            # verify() checks stream integrity
            img_format = pil_img.format or suffix.lstrip(".").upper()
            img_mode = pil_img.mode
            width, height = pil_img.size

            # Load pixels to confirm decoding doesn't raise decompression bombs/errors
            pil_img.load()

        except Exception as exc:
            return ImageValidationResult(
                is_valid=False,
                error_message=f"Corrupted or invalid image data: {exc}",
            )

        # 5. Check dimensions
        if width <= 0 or height <= 0:
            return ImageValidationResult(
                is_valid=False, error_message=f"Invalid image dimensions: {width}x{height}."
            )

        if (
            width > settings.MAX_IMAGE_DIMENSION
            or height > settings.MAX_IMAGE_DIMENSION
        ):
            return ImageValidationResult(
                is_valid=False,
                error_message=(
                    f"Image dimension ({width}x{height}) exceeds maximum limit "
                    f"({settings.MAX_IMAGE_DIMENSION}x{settings.MAX_IMAGE_DIMENSION})."
                ),
            )

        # 6. Extract safe EXIF
        exif = extract_safe_exif(pil_img)

        return ImageValidationResult(
            is_valid=True,
            error_message=None,
            width=width,
            height=height,
            format=img_format,
            mode=img_mode,
            file_size_bytes=file_size,
            metadata={
                "filename": filename,
                "aspect_ratio": round(width / max(height, 1), 3),
                "exif": exif,
            },
        )

    except Exception as exc:
        logger.exception("Unexpected error during image validation: %s", exc)
        return ImageValidationResult(
            is_valid=False,
            error_message=f"Image validation encountered an unexpected error: {exc}",
        )
