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
def boundary_seed_map(depth: np.ndarray, mask: np.ndarray):
    mask_u8 = (mask > 0).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)
    dil = cv2.dilate(mask_u8, kernel, iterations=2)

    boundary = ((dil > 0) & (mask_u8 == 0)).astype(np.uint8)

    # distance transform gives nearest boundary index
    inv_boundary = 1 - boundary
    dist, labels = cv2.distanceTransformWithLabels(
        inv_boundary,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    seed_map = depth.copy()

    boundary_yx = np.argwhere(boundary > 0)

    if boundary_yx.shape[0] == 0:
        return depth.copy(), boundary

    H, W = mask.shape

    for y in range(H):
        for x in range(W):
            if mask_u8[y, x] == 0:
                continue

            idx = labels[y, x] - 1
            if idx < 0 or idx >= len(boundary_yx):
                continue

            by, bx = boundary_yx[idx]

            seed_map[y, x] = depth[by, bx]

    return seed_map, boundary
class DiffusionFillHead(MirrorCorrectionHead):
    name = "diffusion"

    def __init__(
        self,
        min_mask_pixels: int = 50,
        iters: int = 40,
        step: float = 0.20,
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

        kernel = np.ones((3, 3), np.float32)

        # Outside support ring
        dil = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        support_mask = ((dil > 0) & (mask == 0)).astype(np.uint8)

        if support_mask.sum() < 20:
            seed_value = ring_seed_value(depth, mask, kernel_size=7)
            fixed[unknown] = seed_value
            changed_mask = mask.copy()

            return HeadResult(
                name=self.name,
                fixed_depth=fixed.astype(np.float32),
                aux_map=fixed.astype(np.float32),
                support_mask=support_mask.astype(np.uint8),
                changed_mask=changed_mask.astype(np.uint8),
                applied=True,
                alpha=float(self.step),
                score=np.nan,
                meta={
                    "iters": self.iters,
                    "step": self.step,
                    "seed_value": seed_value,
                    "fill_mode": "fallback_seed",
                },
            )

        vals = depth[(support_mask > 0) & np.isfinite(depth)]
        seed_value = float(np.median(vals)) if vals.size > 0 else ring_seed_value(depth, mask, kernel_size=7)

        filled = depth.copy()

        # Set unknown to seed only as placeholder.
        # These placeholder pixels must NOT contribute to propagation until they become known.
        filled, support_mask = boundary_seed_map(depth, mask)

        known_fill = support_mask.astype(np.uint8)
        unknown_fill = mask.astype(np.uint8)

        for _ in range(self.iters):
            grow = cv2.dilate(known_fill, np.ones((3, 3), np.uint8), iterations=1)
            new_pixels = (grow > 0) & (unknown_fill > 0) & (known_fill == 0)

            if not new_pixels.any():
                break

            known_float = known_fill.astype(np.float32)

            # CRITICAL FIX:
            # Only already-known pixels contribute to the propagated value.
            neighbor_sum = cv2.filter2D(
                filled * known_float,
                ddepth=-1,
                kernel=kernel,
                borderType=cv2.BORDER_REPLICATE,
            )

            neighbor_count = cv2.filter2D(
                known_float,
                ddepth=-1,
                kernel=kernel,
                borderType=cv2.BORDER_REPLICATE,
            )

            propagated = neighbor_sum / np.maximum(neighbor_count, 1e-6)

            # Since new_pixels are being filled for the first time,
            # use mostly propagated value, not mostly seed value.
            filled[new_pixels] = (
                (1.0 - self.step) * filled[new_pixels]
                + self.step * propagated[new_pixels]
            )

            known_fill[new_pixels] = 1

        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        max_dist = float(dist.max())

        DIST_RATIO = 0.90
        MIN_W = 0.20
        MAX_W = 0.70

        if max_dist < 1e-6:
            apply_mask = mask.astype(bool)
            weight = np.zeros_like(depth, dtype=np.float32)
            weight[apply_mask] = 0.35
        else:
            max_fill_dist = max_dist * DIST_RATIO
            apply_mask = (mask > 0) & (dist <= max_fill_dist)

            t = np.zeros_like(depth, dtype=np.float32)
            t[apply_mask] = 1.0 - (dist[apply_mask] / (max_fill_dist + 1e-6))

            # Smooth falloff instead of linear/chunky falloff
            t[apply_mask] = t[apply_mask] * t[apply_mask] * (3.0 - 2.0 * t[apply_mask])

            weight = np.zeros_like(depth, dtype=np.float32)
            weight[apply_mask] = MIN_W + (MAX_W - MIN_W) * t[apply_mask]

            # CRITICAL: smooth the weight map so it does not look like chunks/rings
            weight = cv2.GaussianBlur(weight, (21, 21), 0)

            # Keep weight only inside mirror
            weight[mask == 0] = 0.0

        fixed = depth.copy()
        inside = mask > 0
        fixed[inside] = (
            (1.0 - weight[inside]) * depth[inside]
            + weight[inside] * filled[inside]
        )

        changed_mask = ((weight > 0.03) & (mask > 0)).astype(np.uint8)

        return HeadResult(
            name=self.name,
            fixed_depth=fixed.astype(np.float32),
            aux_map=fixed.astype(np.float32),
            support_mask=support_mask.astype(np.uint8),
            changed_mask=changed_mask.astype(np.uint8),
            applied=True,
            alpha=float(self.step),
            score=np.nan,
            meta={
                "iters": self.iters,
                "step": self.step,
                "seed_value": seed_value,
                "fill_mode": "outside_to_inside_propagation",
            },
        )