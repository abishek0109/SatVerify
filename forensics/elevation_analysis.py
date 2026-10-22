"""Branch I — Elevation and Terrain Validation Module.

Optional physical plausibility module that cross-references image characteristics
with Digital Elevation Model (DEM) data and georeferencing metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from config.settings import get_settings
from forensics.base import ForensicResult
from forensics.ela import colorize_grayscale
from utils.logging_config import get_logger

logger = get_logger("forensics.elevation")


class ElevationTerrainValidator:
    """Validates terrain plausibility against elevation models or slope constraints."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def analyze(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
        dem_input: Optional[Union[np.ndarray, str]] = None,
        georef_metadata: Optional[Dict[str, Any]] = None,
    ) -> ForensicResult:
        """Run elevation and terrain validation."""
        try:
            # Check if external DEM data is provided
            if dem_input is None:
                return ForensicResult(
                    branch_id="elevation",
                    branch_name="Elevation and Terrain Validation",
                    status="UNAVAILABLE",
                    score=None,
                    explanation=(
                        "Digital Elevation Model (DEM) and georeferencing metadata not supplied. "
                        "Topographic plausibility cross-validation is optional and bypassed."
                    ),
                    method_description=(
                        "Compares surface slope, aspect, and water accumulation from DEM reference models "
                        "against optical vegetation textures."
                    ),
                    limitations=(
                        "Requires georeferenced GeoTIFF imagery and calibrated DEM datasets (SRTM, Copernicus 30m, etc.)."
                    ),
                    metrics={
                        "dem_provided": False,
                        "georeferencing_present": bool(georef_metadata),
                    },
                    visualization=None,
                )

            # If DEM is provided as an array or file
            if isinstance(dem_input, str):
                try:
                    # Attempt loading via PIL or numpy if .npy/.tif
                    with Image.open(dem_input) as dem_img:
                        dem_arr = np.asarray(dem_img, dtype=np.float32)
                except Exception:
                    dem_arr = np.load(dem_input).astype(np.float32)
            else:
                dem_arr = dem_input.astype(np.float32)

            # Calculate terrain slope
            gy, gx = np.gradient(dem_arr)
            slope = np.arctan(np.sqrt(gx**2 + gy**2)) * (180.0 / np.pi)  # degrees
            mean_slope = float(np.mean(slope))
            max_slope = float(np.max(slope))

            # Cross-reference with image vegetation
            if isinstance(image_input, Image.Image):
                rgb_pil = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                rgb_pil = Image.fromarray(image_input).convert("RGB")
            else:
                with Image.open(str(image_input)) as img:
                    rgb_pil = img.convert("RGB")

            rgb_arr = np.asarray(rgb_pil, dtype=np.float32)
            exg = 2.0 * rgb_arr[:, :, 1] - rgb_arr[:, :, 0] - rgb_arr[:, :, 2]

            # Resize slope to match image shape if necessary
            if slope.shape != exg.shape:
                from PIL import Image as PImage

                slope_pil = PImage.fromarray(slope).resize(
                    (exg.shape[1], exg.shape[0]), resample=PImage.Resampling.BILINEAR
                )
                slope = np.asarray(slope_pil, dtype=np.float32)

            # Check vegetation on implausibly steep slopes (> 65 degrees usually cannot support dense forest canopy)
            dense_forest = exg > np.percentile(exg, 75)
            extreme_cliffs = slope > 65.0
            anomaly_overlap = np.logical_and(dense_forest, extreme_cliffs)
            overlap_ratio = float(np.mean(anomaly_overlap))

            # Score
            norm_anomaly = min(overlap_ratio / 0.05, 1.0)
            score = float(1.0 / (1.0 + np.exp(-6.0 * (norm_anomaly - 0.40))))
            score = round(min(max(score, 0.0), 1.0), 3)

            norm_slope = ((slope - slope.min()) / max(slope.max() - slope.min(), 1e-4) * 255.0).astype(
                np.uint8
            )
            viz_rgb = colorize_grayscale(norm_slope, "terrain")

            return ForensicResult(
                branch_id="elevation",
                branch_name="Elevation and Terrain Validation",
                status="AVAILABLE",
                score=score,
                explanation=(
                    f"DEM cross-reference evaluated. Mean slope: {mean_slope:.1f}°. "
                    f"Forest-on-extreme-cliff anomaly ratio: {overlap_ratio:.2%}."
                ),
                method_description="Cross-references optical canopy mask with DEM slope model.",
                limitations="Depends on DEM resolution and registration accuracy.",
                metrics={
                    "dem_provided": True,
                    "mean_slope_deg": round(mean_slope, 2),
                    "max_slope_deg": round(max_slope, 2),
                    "cliff_canopy_anomaly_ratio": round(overlap_ratio, 4),
                },
                visualization=viz_rgb,
            )

        except Exception as exc:
            logger.exception("Elevation analysis failed: %s", exc)
            return ForensicResult(
                branch_id="elevation",
                branch_name="Elevation and Terrain Validation",
                status="ERROR",
                score=None,
                explanation=f"Error evaluating DEM: {exc}",
                method_description="Digital Elevation Model cross-reference.",
                limitations="Requires registered DEM array.",
            )
