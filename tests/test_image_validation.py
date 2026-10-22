"""Unit tests for image validation and metadata extraction."""

import io
import pytest
import numpy as np
from PIL import Image

from utils.validation import validate_image_file


def test_valid_rgb_image():
    """Verify that a standard RGB image validates successfully with correct metadata."""
    img = Image.new("RGB", (256, 256), color=(40, 120, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    result = validate_image_file(raw_bytes, original_filename="test_forest.jpg")
    assert result.is_valid is True
    assert result.error_message is None
    assert result.width == 256
    assert result.height == 256
    assert result.format in ["JPEG", "JPG"]
    assert result.dimensions_str == "256 × 256"


def test_corrupted_image_rejected():
    """Verify that corrupt byte sequences are safely caught and rejected."""
    corrupted_bytes = b"NOT_AN_IMAGE_RANDOM_GARBAGE_PAYLOAD_12345678"
    result = validate_image_file(corrupted_bytes, original_filename="corrupted.jpg")
    assert result.is_valid is False
    assert result.error_message is not None
    assert "Corrupted or invalid image" in result.error_message


def test_empty_file_rejected():
    """Verify that empty files (0 bytes) are rejected."""
    result = validate_image_file(b"", original_filename="empty.png")
    assert result.is_valid is False
    assert "empty" in result.error_message.lower()


def test_unsupported_extension_rejected():
    """Verify that unsupported extensions are rejected."""
    img = Image.new("RGB", (64, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = validate_image_file(buf.getvalue(), original_filename="archive.zip")
    assert result.is_valid is False
    assert "Unsupported image format" in result.error_message
