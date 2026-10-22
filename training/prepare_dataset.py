"""Dataset preparation and train/val/test splitting pipeline for SatVerify."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from config.settings import get_settings
from training.generate_manipulations import ManipulationGenerator
from utils.logging_config import get_logger

logger = get_logger("training.dataset")


class DatasetManager:
    """Organizes satellite imagery into leak-free training, validation, and test splits."""

    def __init__(self, data_root: Optional[Path] = None) -> None:
        self.settings = get_settings()
        self.data_root = data_root or self.settings.DATA_DIR

    def ensure_split_directories(self) -> None:
        """Create standard split folders: train, validation, test (real/fake)."""
        for split in ["train", "validation", "test"]:
            for label in ["real", "fake"]:
                (self.data_root / split / label).mkdir(parents=True, exist_ok=True)
        logger.info("Verified dataset split directories under: %s", self.data_root)

    def split_raw_dataset(
        self,
        raw_real_dir: Path,
        raw_fake_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Partition raw REAL and FAKE folders into train, validation, and test sets."""
        self.ensure_split_directories()
        random.seed(seed)

        def collect_files(folder: Path) -> List[Path]:
            return [
                f for f in folder.glob("*")
                if f.is_file() and f.suffix.lower() in self.settings.SUPPORTED_EXTENSIONS
            ]

        real_files = collect_files(raw_real_dir)
        fake_files = collect_files(raw_fake_dir)

        if not real_files or not fake_files:
            raise ValueError(f"Found {len(real_files)} real and {len(fake_files)} fake images. Need >= 1 of each.")

        for files, label in [(real_files, "real"), (fake_files, "fake")]:
            random.shuffle(files)
            n = len(files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_set = files[:n_train]
            val_set = files[n_train : n_train + n_val]
            test_set = files[n_train + n_val :]

            for split_name, file_list in [("train", train_set), ("validation", val_set), ("test", test_set)]:
                target_dir = self.data_root / split_name / label
                for src in file_list:
                    shutil.copy2(src, target_dir / src.name)

        logger.info(
            "Partitioned dataset: %s real, %s fake into train/val/test.",
            len(real_files), len(fake_files)
        )

    def generate_smoke_test_fixtures(self, num_samples: int = 8) -> None:
        """Generate synthetic procedural terrain fixtures for automated testing."""
        fixtures_dir = self.data_root / "fixtures"
        real_dir = fixtures_dir / "real"
        fake_dir = fixtures_dir / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        generator = ManipulationGenerator(output_dir=fake_dir)

        for i in range(num_samples):
            # 1. Procedural Authentic Terrain (Perlin-like smooth noise with green/brown forest hues)
            w, h = 256, 256
            x, y = np.meshgrid(np.linspace(0, 5, w), np.linspace(0, 5, h))
            # Natural spatial wave patterns
            elevation = np.sin(x) * np.cos(y) + 0.5 * np.sin(2 * x + y)
            elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())

            # Terrain coloring: Forests (dark green), Vegetation (medium green), Soil (brown)
            r = (elevation * 80 + 30).astype(np.uint8)
            g = (elevation * 120 + 70).astype(np.uint8)
            b = (elevation * 50 + 20).astype(np.uint8)

            # Add fine sensor noise
            noise = np.random.normal(0, 4, (h, w, 3))
            rgb = np.clip(np.stack([r, g, b], axis=-1) + noise, 0, 255).astype(np.uint8)
            real_pil = Image.fromarray(rgb)

            real_path = real_dir / f"fixture_real_{i:03d}.jpg"
            real_pil.save(real_path, quality=92)

            # 2. Procedural Tampered Terrain (Copy-paste or green boosted)
            if i % 2 == 0:
                fake_pil, _ = generator.copy_paste_vegetation(real_pil, patch_size=(64, 64))
            else:
                fake_pil, _ = generator.boost_green_channel(real_pil, boost_factor=1.6)

            fake_path = fake_dir / f"fixture_fake_{i:03d}.jpg"
            fake_pil.save(fake_path, quality=88)

        logger.info("Generated %d smoke-test fixtures in: %s", num_samples, fixtures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SatVerify datasets and fixtures.")
    parser.add_argument("--generate-fixtures", action="store_true", help="Generate synthetic fixtures for testing.")
    parser.add_argument("--samples", type=int, default=8, help="Number of fixture samples.")
    args = parser.parse_args()

    dm = DatasetManager()
    if args.generate_fixtures:
        dm.generate_smoke_test_fixtures(num_samples=args.samples)
    else:
        dm.ensure_split_directories()
        print("Dataset split directories ensured. Use --generate-fixtures to create test samples.")


if __name__ == "__main__":
    main()
