"""Image handling and file lifecycle service."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

from PIL import Image

from config.settings import get_settings
from utils.logging_config import get_logger

logger = get_logger("services.image")


class ImageService:
    """Manages upload caching, temporary storage, and image conversion."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.temp_dir = Path(tempfile.gettempdir()) / "satverify_uploads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def save_upload_to_temp(
        self, file_bytes: bytes, original_filename: str
    ) -> Path:
        """Persist uploaded bytes to a secured temporary file path."""
        suffix = Path(original_filename).suffix.lower() or ".jpg"
        temp_path = self.temp_dir / f"sat_{uuid4().hex[:12]}{suffix}"
        temp_path.write_bytes(file_bytes)
        logger.debug("Saved temporary upload to: %s", temp_path)
        return temp_path

    def cleanup_file(self, path: Optional[Union[str, Path]]) -> None:
        """Safely remove a temporary file."""
        if path is None:
            return
        try:
            p = Path(path)
            if p.exists() and self.temp_dir in p.parents:
                p.unlink(missing_ok=True)
                logger.debug("Cleaned up temp file: %s", p)
        except Exception as exc:
            logger.debug("Failed cleanup for %s: %s", path, exc)
