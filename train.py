from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor

from forensics import load_image_as_pil
from model import DeepfakeDetector


IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
CLASS_TO_INDEX: Dict[str, int] = {"REAL": 0, "FAKE": 1}


@dataclass(frozen=True)
class TrainingConfig:
    """Container for all train-time configuration values.

    Args:
        data_dir: Root dataset directory containing train/ and val/ splits.
        output_checkpoint: Destination path for the best model checkpoint.
        model_name: Hugging Face ViT model name.
        epochs: Number of full training epochs.
        batch_size: Number of images per batch.
        learning_rate: AdamW learning rate.
        num_workers: DataLoader worker process count.
        initial_threshold: Starting fake decision threshold.

    Returns:
        None

    Why:
        Centralizing configuration improves reproducibility and keeps command-line
        parsing and training logic cleanly separated.
    """

    data_dir: Path
    output_checkpoint: Path
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    initial_threshold: float


def parse_args() -> TrainingConfig:
    """Parse command-line options and build a validated training config.

    Args:
        None

    Returns:
        TrainingConfig: Parsed immutable training configuration.

    Why:
        A predictable CLI interface lets you re-run training quickly during the
        hackathon while keeping hyperparameters explicit and documented.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune SatVerify ViT model on REAL/FAKE satellite imagery."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "satellite_auth",
        help="Dataset root with train/REAL, train/FAKE, val/REAL, val/FAKE.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("checkpoints") / "satverify_best.pt",
        help="Path to save best model checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/vit-base-patch16-224",
        help="Base Hugging Face model identifier.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="AdamW learning rate."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader worker count."
    )
    parser.add_argument(
        "--initial-threshold",
        type=float,
        default=0.60,
        help="Initial FAKE decision threshold before validation calibration.",
    )

    args = parser.parse_args()
    return TrainingConfig(
        data_dir=args.data_dir,
        output_checkpoint=args.output_checkpoint,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        initial_threshold=args.initial_threshold,
    )


def collect_samples(split_dir: Path) -> List[Tuple[Path, int]]:
    """Collect labeled image paths from a split directory.

    Args:
        split_dir: Path to a split directory (for example train/ or val/).

    Returns:
        List[Tuple[Path, int]]: List of (image_path, class_index) pairs.

    Why:
        Explicit file collection makes dataset integrity checks transparent and
        avoids silent mislabeling from unexpected folder structures.
    """
    samples: List[Tuple[Path, int]] = []
    for class_name, class_index in CLASS_TO_INDEX.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder: '{class_dir}'. Expected REAL and FAKE."
            )

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_index))

    if not samples:
        raise RuntimeError(f"No images found in split directory '{split_dir}'.")
    return samples


class SatelliteAuthDataset(Dataset):
    """Dataset wrapper that preprocesses images with ViTImageProcessor."""

    def __init__(
        self, samples: Sequence[Tuple[Path, int]], processor: ViTImageProcessor
    ) -> None:
        """Initialize dataset.

        Args:
            samples: Sequence of (image_path, label_index) entries.
            processor: ViT processor for resizing/normalization.

        Returns:
            None

        Why:
            Keeping preprocessing aligned with the pretrained backbone is critical
            for stable transfer learning performance.
        """
        self.samples = list(samples)
        self.processor = processor

    def __len__(self) -> int:
        """Return dataset size.

        Args:
            None

        Returns:
            int: Number of samples.

        Why:
            PyTorch DataLoader requires an accurate sample count to batch data.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Read and preprocess one sample.

        Args:
            index: Dataset sample index.

        Returns:
            Dict[str, object]: Dictionary with pixel tensor, label, and path.

        Why:
            Returning file paths allows validation-time forensic scoring and
            threshold calibration on the same samples.
        """
        image_path, label = self.samples[index]
        image = load_image_as_pil(str(image_path))
        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "label": label, "path": str(image_path)}


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Collate dataset records into tensors/lists for a training step.

    Args:
        batch: Sequence of dataset output dictionaries.

    Returns:
        Dict[str, object]: Batched tensors and sample paths.

    Why:
        Custom collation preserves image paths required for forensic validation
        while still producing dense tensors for fast model execution.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])  # type: ignore[index]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)  # type: ignore[index]
    paths = [str(item["path"]) for item in batch]
    return {"pixel_values": pixel_values, "labels": labels, "paths": paths}


def compute_class_weights(samples: Sequence[Tuple[Path, int]], device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        samples: Training sample list.
        device: Device where the loss tensor should live.

    Returns:
        torch.Tensor: Class weights ordered by index [REAL, FAKE].

    Why:
        Satellite datasets can be skewed across real/fake classes. Weighted loss
        helps prevent the model from collapsing to the majority class.
    """
    counts = np.zeros(len(CLASS_TO_INDEX), dtype=np.float64)
    for _, label in samples:
        counts[label] += 1.0

    if np.any(counts == 0):
        raise RuntimeError("Each class must contain at least one training image.")

    # Inverse-frequency weighting normalized to keep average scale near 1.
    inv = 1.0 / counts
    normalized = inv / inv.mean()
    return torch.tensor(normalized, dtype=torch.float32, device=device)


def train_one_epoch(
    detector: DeepfakeDetector,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Run one full supervised training epoch.

    Args:
        detector: Detector wrapper containing ViT backbone.
        train_loader: DataLoader for training split.
        optimizer: Optimizer for parameter updates.
        criterion: Loss function (CrossEntropy with class weights).

    Returns:
        float: Mean training loss for the epoch.

    Why:
        Isolating the per-epoch training loop makes the overall pipeline easier
        to inspect, test, and extend with scheduler or mixed precision later.
    """
    detector.model.train()
    running_loss = 0.0
    step_count = 0

    for step_count, batch in enumerate(train_loader, start=1):
        pixel_values = batch["pixel_values"].to(detector.device)  # type: ignore[index]
        labels = batch["labels"].to(detector.device)  # type: ignore[index]

        optimizer.zero_grad()
        outputs = detector.model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(step_count, 1)


def evaluate_vit(
    detector: DeepfakeDetector, val_loader: DataLoader, criterion: nn.Module
) -> Tuple[float, List[float], List[int], List[str]]:
    """Evaluate ViT on validation data and collect per-sample probabilities.

    Args:
        detector: Detector wrapper containing ViT backbone.
        val_loader: DataLoader for validation split.
        criterion: Loss function for reporting validation loss.

    Returns:
        Tuple[float, List[float], List[int], List[str]]: Mean loss, raw ViT fake
        probabilities, ground-truth labels, and image paths.

    Why:
        Validation probabilities are needed to build ensemble scores and calibrate
        a decision threshold that reduces false positives on REAL images.
    """
    detector.model.eval()
    running_loss = 0.0
    step_count = 0
    vit_fake_probs: List[float] = []
    labels_all: List[int] = []
    paths_all: List[str] = []

    with torch.no_grad():
        for step_count, batch in enumerate(val_loader, start=1):
            pixel_values = batch["pixel_values"].to(detector.device)  # type: ignore[index]
            labels = batch["labels"].to(detector.device)  # type: ignore[index]
            paths = batch["paths"]  # type: ignore[index]

            outputs = detector.model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            running_loss += float(loss.item())

            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            fake_index = detector.label2id["FAKE"]

            for idx, prob_vector in enumerate(probabilities):
                vit_fake_probs.append(float(prob_vector[fake_index]))
                labels_all.append(int(labels[idx].item()))
                paths_all.append(str(paths[idx]))

    return running_loss / max(step_count, 1), vit_fake_probs, labels_all, paths_all


def build_ensemble_scores(vit_fake_probs: Sequence[float], paths: Sequence[str]) -> List[float]:
    """Combine ViT, ELA, and FFT signals into final fake probabilities.

    Args:
        vit_fake_probs: Raw ViT FAKE probabilities from validation forward pass.
        paths: Image paths matching the probability order.

    Returns:
        List[float]: Ensemble fake scores in [0.0, 1.0].

    Why:
        Ensemble scoring mirrors production inference behavior, ensuring threshold
        calibration is aligned with what Streamlit will actually use.
    """
    if len(vit_fake_probs) != len(paths):
        raise ValueError("vit_fake_probs and paths must have the same length.")

    scores: List[float] = []
    for vit_raw, image_path in zip(vit_fake_probs, paths):
        # Match runtime ViT score calibration used in model.predict().
        vit_calibrated = (float(vit_raw) * 0.5) + 0.25

        # Compute deterministic forensic fake signals from residual/frequency cues.
        ela_score = float(compute_safe_forensic("ela", image_path))
        fft_score = float(compute_safe_forensic("fft", image_path))

        # Weighted fusion: ViT (0.2), ELA (0.4), FFT (0.4).
        final_score = (0.2 * vit_calibrated) + (0.4 * ela_score) + (0.4 * fft_score)
        scores.append(float(final_score))
    return scores


def compute_safe_forensic(method: str, image_path: str) -> float:
    """Compute a forensic score with clear error context.

    Args:
        method: Either "ela" or "fft".
        image_path: Image path for forensic analysis.

    Returns:
        float: Forensic fake-likelihood in [0.0, 1.0].

    Why:
        Explicit method routing with contextual errors makes debugging data
        quality issues much faster during rapid hackathon iteration.
    """
    from forensics import compute_ela_score, compute_fft_score

    try:
        if method == "ela":
            return float(compute_ela_score(image_path))
        if method == "fft":
            return float(compute_fft_score(image_path))
        raise ValueError(f"Unsupported forensic method '{method}'.")
    except Exception as exc:
        raise RuntimeError(
            f"Failed forensic scoring ({method}) for image '{image_path}': {exc}"
        ) from exc


def calibrate_decision_threshold(
    fake_scores: Sequence[float], labels: Sequence[int]
) -> Tuple[float, Dict[str, float]]:
    """Find a threshold that reduces false positives while preserving recall.

    Args:
        fake_scores: Ensemble fake probabilities.
        labels: Ground-truth class indices (0=REAL, 1=FAKE).

    Returns:
        Tuple[float, Dict[str, float]]: Best threshold and calibration metrics.

    Why:
        The biggest practical issue is false alarms on real imagery. This search
        favors specificity (REAL correctly identified) while maintaining useful
        recall on FAKE samples.
    """
    if len(fake_scores) != len(labels):
        raise ValueError("fake_scores and labels must have the same length.")
    if not fake_scores:
        raise ValueError("Need at least one validation sample for calibration.")

    scores_np = np.asarray(fake_scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)

    best_threshold = 0.60
    best_metrics: Dict[str, float] = {
        "objective": -1e9,
        "specificity": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
    }

    for threshold in np.linspace(0.30, 0.90, 121):
        preds = (scores_np >= threshold).astype(np.int64)

        tp = int(np.sum((preds == 1) & (labels_np == 1)))
        tn = int(np.sum((preds == 0) & (labels_np == 0)))
        fp = int(np.sum((preds == 1) & (labels_np == 0)))
        fn = int(np.sum((preds == 0) & (labels_np == 1)))

        specificity = tn / max(tn + fp, 1)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

        # Weighted objective prioritizes low false positives on REAL samples.
        objective = (0.55 * specificity) + (0.35 * recall) + (0.10 * f1)

        # Penalize thresholds that collapse fake recall too aggressively.
        min_recall_target = 0.55
        if recall < min_recall_target:
            objective -= (min_recall_target - recall) * 0.25

        if objective > best_metrics["objective"]:
            best_threshold = float(threshold)
            best_metrics = {
                "objective": float(objective),
                "specificity": float(specificity),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
            }

    return best_threshold, best_metrics


def run_training(config: TrainingConfig) -> None:
    """Train detector, calibrate threshold, and save best checkpoint.

    Args:
        config: Complete training configuration.

    Returns:
        None

    Why:
        End-to-end automation from raw folders to calibrated checkpoint removes
        fragile manual steps and makes retraining repeatable under time pressure.
    """
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Dataset structure not found. Expected:\n"
            f"  {train_dir}\\REAL\n"
            f"  {train_dir}\\FAKE\n"
            f"  {val_dir}\\REAL\n"
            f"  {val_dir}\\FAKE"
        )

    detector = DeepfakeDetector(
        model_name=config.model_name, num_labels=2, decision_threshold=config.initial_threshold
    )
    processor = detector.processor

    train_samples = collect_samples(train_dir)
    val_samples = collect_samples(val_dir)

    train_dataset = SatelliteAuthDataset(train_samples, processor)
    val_dataset = SatelliteAuthDataset(val_samples, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )

    class_weights = compute_class_weights(train_samples, detector.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(detector.model.parameters(), lr=config.learning_rate)

    best_objective = -1e9
    best_state_dict = copy.deepcopy(detector.model.state_dict())
    best_threshold = detector.decision_threshold
    best_metrics: Dict[str, float] = {}

    print(f"Training on device: {detector.device}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(detector, train_loader, optimizer, criterion)
        val_loss, vit_probs, val_labels, val_paths = evaluate_vit(detector, val_loader, criterion)

        # Build validation ensemble scores using deterministic forensic branches.
        ensemble_scores = build_ensemble_scores(vit_probs, val_paths)
        threshold, metrics = calibrate_decision_threshold(ensemble_scores, val_labels)
        detector.decision_threshold = threshold

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"thr={threshold:.3f} | spec={metrics['specificity']:.3f} | "
            f"recall={metrics['recall']:.3f} | f1={metrics['f1']:.3f}"
        )

        if metrics["objective"] > best_objective:
            best_objective = metrics["objective"]
            best_state_dict = copy.deepcopy(detector.model.state_dict())
            best_threshold = threshold
            best_metrics = dict(metrics)

    # Restore and persist best-performing checkpoint.
    detector.model.load_state_dict(best_state_dict)
    detector.model.eval()
    detector.decision_threshold = best_threshold

    checkpoint_parent = config.output_checkpoint.parent
    checkpoint_parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": detector.model.state_dict(),
            "decision_threshold": detector.decision_threshold,
            "id2label": detector.id2label,
            "label2id": detector.label2id,
            "val_metrics": best_metrics,
            "config": {
                "model_name": config.model_name,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "data_dir": str(config.data_dir),
            },
        },
        config.output_checkpoint,
    )

    print(f"Saved best checkpoint to: {config.output_checkpoint}")
    print(
        "Best metrics | "
        f"specificity={best_metrics.get('specificity', 0.0):.3f} | "
        f"recall={best_metrics.get('recall', 0.0):.3f} | "
        f"f1={best_metrics.get('f1', 0.0):.3f} | "
        f"accuracy={best_metrics.get('accuracy', 0.0):.3f} | "
        f"decision_threshold={detector.decision_threshold:.3f}"
    )


def main() -> None:
    """Entrypoint for CLI training execution.

    Args:
        None

    Returns:
        None

    Why:
        A dedicated main function cleanly handles runtime exceptions and provides
        actionable error messages during hackathon debugging.
    """
    config = parse_args()
    try:
        run_training(config)
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
