"""Structured logging configuration for SatVerify."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured console logging for SatVerify."""
    root_logger = logging.getLogger("satverify")
    if root_logger.handlers:
        return  # Avoid duplicating handlers if already configured

    root_logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve a configured logger under the 'satverify' namespace."""
    configure_logging()
    if name:
        return logging.getLogger(f"satverify.{name}")
    return logging.getLogger("satverify")
