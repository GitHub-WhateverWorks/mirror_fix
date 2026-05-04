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
class GaussianFillHead(MirrorCorrectionHead):
    name = "gaussian"

    def __init__(
        self,
        min_mask_pixels: int = 50,
        blur_kernel: int = 5,
        iters: int = 12,
        alpha: float = 0.22,
    ):
        self.min_mask_pixels = min_mask_pixels
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.iters = iters
        self.alpha = alpha

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

        # Outside-to-inside Gaussian fill.
        # Instead of seeding the whole mirror with one median value, grow depth
        # from the outside boundary ring layer-by-layer into the mask.
        kernel = np.ones((3, 3), np.uint8)

        # Support = narrow outside ring around the predicted mirror.
        dil = cv2.dilate(mask, kernel, iterations=2)
        support_mask = ((dil > 0) & (mask == 0)).astype(np.uint8)

        seed_value = float(np.median(depth[(support_mask > 0) & np.isfinite(depth)]))

        filled = depth.copy()
        filled, support_mask = boundary_seed_map(depth, mask)

        # known_fill means pixels whose values are allowed to propagate.
        # Start from outside support only, then expand inward.
        known_fill = support_mask.copy()
        unknown_fill = mask.copy()

        for _ in range(self.iters):
            grow = cv2.dilate(known_fill, kernel, iterations=1)
            new_pixels = (grow > 0) & (unknown_fill > 0) & (known_fill == 0)

            if not new_pixels.any():
                break

            blurred = cv2.GaussianBlur(
                filled,
                (self.blur_kernel, self.blur_kernel),
                0,
            )

            filled[new_pixels] = (
                self.alpha * blurred[new_pixels]
                + (1.0 - self.alpha) * filled[new_pixels]
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
            support_mask=known.astype(np.uint8),
            changed_mask=changed_mask.astype(np.uint8),
            applied=True,
            alpha=float(self.alpha),
            score=np.nan,
            meta={"iters": self.iters, "blur_kernel": self.blur_kernel, "seed_value": seed_value},
        )