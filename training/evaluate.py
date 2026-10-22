"""Evaluation and benchmarking module for SatVerify.

Calculates precision, recall, F1, specificity, confusion matrix,
and inconclusive rate across test splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.settings import get_settings
from core.pipeline import SatVerifyPipeline
from utils.logging_config import get_logger

logger = get_logger("training.evaluate")


class PipelineEvaluator:
    """Evaluates the full SatVerify pipeline against ground-truth datasets."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline = SatVerifyPipeline()

    def evaluate_directory(self, test_root: Path) -> Dict[str, Any]:
        """Run pipeline evaluation over REAL and FAKE folders."""
        real_dir = test_root / "real"
        fake_dir = test_root / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(f"Test split must contain 'real' and 'fake' folders: {test_root}")

        real_files = [f for f in real_dir.glob("*") if f.is_file() and f.suffix.lower() in self.settings.SUPPORTED_EXTENSIONS]
        fake_files = [f for f in fake_dir.glob("*") if f.is_file() and f.suffix.lower() in self.settings.SUPPORTED_EXTENSIONS]

        logger.info("Evaluating %d real and %d fake images...", len(real_files), len(fake_files))

        y_true: List[int] = []  # 0=REAL, 1=FAKE
        y_pred: List[Optional[int]] = []  # 0=REAL, 1=FAKE, None=INCONCLUSIVE
        scores: List[float] = []

        # Evaluate REAL images
        for f in real_files:
            rep = self.pipeline.analyze(f)
            y_true.append(0)
            if rep.decision.verdict == "REAL":
                y_pred.append(0)
            elif rep.decision.verdict == "FAKE":
                y_pred.append(1)
            else:
                y_pred.append(None)
            scores.append(rep.decision.fused_score if rep.decision.fused_score is not None else 0.5)

        # Evaluate FAKE images
        for f in fake_files:
            rep = self.pipeline.analyze(f)
            y_true.append(1)
            if rep.decision.verdict == "REAL":
                y_pred.append(0)
            elif rep.decision.verdict == "FAKE":
                y_pred.append(1)
            else:
                y_pred.append(None)
            scores.append(rep.decision.fused_score if rep.decision.fused_score is not None else 0.5)

        # Compute metrics
        total = len(y_true)
        inconclusive_count = sum(1 for p in y_pred if p is None)
        inconclusive_rate = inconclusive_count / max(total, 1)

        # Decided samples only
        decided_indices = [i for i, p in enumerate(y_pred) if p is not None]
        if decided_indices:
            decided_true = [y_true[i] for i in decided_indices]
            decided_pred = [y_pred[i] for i in decided_indices]

            tp = sum(1 for yt, yp in zip(decided_true, decided_pred) if yt == 1 and yp == 1)
            tn = sum(1 for yt, yp in zip(decided_true, decided_pred) if yt == 0 and yp == 0)
            fp = sum(1 for yt, yp in zip(decided_true, decided_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(decided_true, decided_pred) if yt == 1 and yp == 0)

            accuracy = (tp + tn) / len(decided_indices)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
        else:
            tp = tn = fp = fn = 0
            accuracy = precision = recall = specificity = f1 = 0.0

        results = {
            "total_samples": total,
            "real_samples": len(real_files),
            "fake_samples": len(fake_files),
            "inconclusive_count": inconclusive_count,
            "inconclusive_rate": round(inconclusive_rate, 4),
            "decided_samples": len(decided_indices),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

        logger.info("Evaluation Results: %s", results)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SatVerify pipeline.")
    parser.add_argument("--test-dir", type=Path, default=Path("data") / "fixtures", help="Path to test folder.")
    parser.add_argument("--output", type=Path, default=Path("reports") / "evaluation_report.json")
    args = parser.parse_args()

    evaluator = PipelineEvaluator()
    res = evaluator.evaluate_directory(args.test_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2))
    print(f"Evaluation report written to: {args.output}")


if __name__ == "__main__":
    main()
