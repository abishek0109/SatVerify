"""Base definitions and standard return types for forensic branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ForensicResult:
    """Standardized output container for any forensic branch analysis."""

    branch_id: str
    branch_name: str
    status: str  # "AVAILABLE", "UNAVAILABLE", "ERROR", "LIMITED"
    score: Optional[float] = None  # Normalized [0.0, 1.0], higher = more suspicious
    explanation: str = ""
    method_description: str = ""
    limitations: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    visualization: Optional[np.ndarray] = None  # RGB uint8 image array

    @property
    def is_available(self) -> bool:
        """Check if branch produced a valid score."""
        return self.status == "AVAILABLE" and self.score is not None
