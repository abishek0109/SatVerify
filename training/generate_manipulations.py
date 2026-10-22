"""Synthetic Manipulation Generator for SatVerify.

Creates realistic, controlled fraudulent satellite imagery from authentic source images:
- Vegetation copy-paste (paper forest fabrication)
- Splicing from disparate scenes
- Green-channel boosting (artificial NDVI enhancement)
- Patch repetition / texture tiling
- Boundary smoothing and noise injection

Saves detailed provenance metadata linking manipulations to their ground-truth parameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from config.settings import get_settings
from utils.logging_config import get_logger

logger = get_logger("training.manipulations")


class ManipulationGenerator:
    """Generates synthetic deepfake and tampered satellite training samples."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.settings = get_settings()
        self.output_dir = output_dir or (self.settings.DATA_DIR / "synthetic_tampered")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_records: List[Dict[str, Any]] = []

    def copy_paste_vegetation(
        self,
        image_pil: Image.Image,
        patch_size: Tuple[int, int] = (64, 64),
        src_coords: Optional[Tuple[int, int]] = None,
        dst_coords: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Copy a vegetation patch from one region and paste it onto another terrain area."""
        arr = np.asarray(image_pil.convert("RGB")).copy()
        h, w, _ = arr.shape
        pw, ph = patch_size

        if src_coords is None:
            # Pick a source patch with relatively high green channel
            sx = int(np.random.randint(0, max(w - pw, 1)))
            sy = int(np.random.randint(0, max(h - ph, 1)))
        else:
            sx, sy = src_coords

        if dst_coords is None:
            dx = int(np.random.randint(0, max(w - pw, 1)))
            dy = int(np.random.randint(0, max(h - ph, 1)))
        else:
            dx, dy = dst_coords

        patch = arr[sy : sy + ph, sx : sx + pw].copy()
        # Paste patch onto target
        arr[dy : dy + ph, dx : dx + pw] = patch

        result_img = Image.fromarray(arr)
        meta = {
            "manipulation_type": "copy_paste_vegetation",
            "patch_size": [pw, ph],
            "src_coords": [sx, sy],
            "dst_coords": [dx, dy],
        }
        return result_img, meta

    def boost_green_channel(
        self,
        image_pil: Image.Image,
        boost_factor: float = 1.45,
        target_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Artificially enhance green color channel to fake high vegetation index (NDVI)."""
        arr = np.asarray(image_pil.convert("RGB"), dtype=np.float32).copy()
        h, w, _ = arr.shape

        if target_box is None:
            # Random rectangular region
            bx = int(np.random.randint(0, w // 2))
            by = int(np.random.randint(0, h // 2))
            bw = int(np.random.randint(w // 4, w - bx))
            bh = int(np.random.randint(h // 4, h - by))
            box = (bx, by, bx + bw, by + bh)
        else:
            box = target_box

        x1, y1, x2, y2 = box
        # Boost green channel in region
        arr[y1:y2, x1:x2, 1] = np.clip(arr[y1:y2, x1:x2, 1] * boost_factor, 0, 255)
        # Suppress red slightly to simulate chlorophyll absorption
        arr[y1:y2, x1:x2, 0] = np.clip(arr[y1:y2, x1:x2, 0] * 0.85, 0, 255)

        result_img = Image.fromarray(arr.astype(np.uint8))
        meta = {
            "manipulation_type": "green_channel_boost",
            "boost_factor": boost_factor,
            "region_box": list(box),
        }
        return result_img, meta

    def repeat_texture_patch(
        self,
        image_pil: Image.Image,
        patch_size: int = 48,
        num_repetitions: int = 4,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Repeat a single canopy texture tile across multiple locations to mimic stamp cloning."""
        arr = np.asarray(image_pil.convert("RGB")).copy()
        h, w, _ = arr.shape

        sx = int(np.random.randint(0, max(w - patch_size, 1)))
        sy = int(np.random.randint(0, max(h - patch_size, 1)))
        tile = arr[sy : sy + patch_size, sx : sx + patch_size].copy()

        paste_locs = []
        for _ in range(num_repetitions):
            px = int(np.random.randint(0, max(w - patch_size, 1)))
            py = int(np.random.randint(0, max(h - patch_size, 1)))
            arr[py : py + patch_size, px : px + patch_size] = tile
            paste_locs.append([px, py])

        result_img = Image.fromarray(arr)
        meta = {
            "manipulation_type": "texture_repetition",
            "patch_size": patch_size,
            "tile_source": [sx, sy],
            "paste_locations": paste_locs,
        }
        return result_img, meta

    def splice_scenes(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        seam_orientation: str = "vertical",
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Combine regions from two separate satellite images to create an artificial landscape."""
        arr_a = np.asarray(image_a.convert("RGB").resize((256, 256)))
        arr_b = np.asarray(image_b.convert("RGB").resize((256, 256)))

        spliced = arr_a.copy()
        split_pos = 128

        if seam_orientation == "vertical":
            spliced[:, split_pos:] = arr_b[:, split_pos:]
        else:
            spliced[split_pos:, :] = arr_b[split_pos:, :]

        result_img = Image.fromarray(spliced)
        meta = {
            "manipulation_type": "scene_splicing",
            "seam_orientation": seam_orientation,
            "split_position": split_pos,
        }
        return result_img, meta

    def save_metadata(self, filename: str = "manipulation_metadata.json") -> Path:
        """Persist metadata records to disk."""
        meta_file = self.output_dir / filename
        meta_file.write_text(json.dumps(self.metadata_records, indent=2))
        logger.info("Saved manipulation metadata to: %s", meta_file)
        return meta_file
