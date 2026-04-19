from __future__ import annotations

import cv2
import numpy as np

from .base import MirrorCorrectionHead, HeadResult

def ring_seed_value(depth: np.ndarray, mask: np.ndarray, kernel_size: int = 7) -> float:
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dil = cv2.dilate(mask_u8, kernel, iterations=1)
    ring = ((dil > 0) & (mask_u8 == 0))

    vals = depth[ring & np.isfinite(depth)]
    if vals.size > 0:
        return float(np.median(vals))

    vals = depth[(mask_u8 == 0) & np.isfinite(depth)]
    if vals.size > 0:
        return float(np.median(vals))

    vals = depth[np.isfinite(depth)]
    if vals.size > 0:
        return float(np.median(vals))

    return 0.0
class DiffusionFillHead(MirrorCorrectionHead):
    name = "diffusion"

    def __init__(
        self,
        min_mask_pixels: int = 50,
        iters: int = 200,
        step: float = 0.24,
    ):
        self.min_mask_pixels = min_mask_pixels
        self.iters = iters
        self.step = step

    def run(self, depth: np.ndarray, mask: np.ndarray) -> HeadResult:
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() < self.min_mask_pixels:
            return HeadResult(
                name=self.name,
                fixed_depth=depth.copy().astype(np.float32),
                aux_map=depth.copy().astype(np.float32),
                support_mask=np.zeros_like(mask, dtype=np.uint8),
                changed_mask=np.zeros_like(mask, dtype=np.uint8),
                applied=False,
                meta={"reason": "mask_too_small"},
            )

        depth = depth.astype(np.float32)
        known = (mask == 0)
        unknown = (mask > 0)

        fixed = depth.copy()

        # IMPORTANT: remove raw mirror depth from initialization
        seed_value = ring_seed_value(depth, mask, kernel_size=7)
        fixed[unknown] = seed_value

        for _ in range(self.iters):
            nbr = (
                np.roll(fixed, 1, axis=0) +
                np.roll(fixed, -1, axis=0) +
                np.roll(fixed, 1, axis=1) +
                np.roll(fixed, -1, axis=1)
            ) * 0.25

            updated = fixed + self.step * (nbr - fixed)
            fixed[unknown] = updated[unknown]
            fixed[known] = depth[known]

        changed_mask = mask.copy()

        return HeadResult(
            name=self.name,
            fixed_depth=fixed.astype(np.float32),
            aux_map=fixed.astype(np.float32),
            support_mask=known.astype(np.uint8),
            changed_mask=changed_mask.astype(np.uint8),
            applied=True,
            alpha=1.0,
            score=np.nan,
            meta={"iters": self.iters, "step": self.step, "seed_value": seed_value},
        )