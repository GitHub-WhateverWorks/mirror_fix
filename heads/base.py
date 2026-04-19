from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class HeadResult:
    name: str
    fixed_depth: np.ndarray
    aux_map: np.ndarray
    support_mask: np.ndarray
    changed_mask: np.ndarray
    applied: bool
    alpha: float = 0.0
    score: float = np.nan
    meta: dict[str, Any] = field(default_factory=dict)


class MirrorCorrectionHead:
    name = "base"

    def run(self, depth: np.ndarray, mask: np.ndarray) -> HeadResult:
        raise NotImplementedError