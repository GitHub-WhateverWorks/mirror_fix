from __future__ import annotations

import cv2
import numpy as np

from .base import MirrorCorrectionHead, HeadResult


class PlaneFitHead(MirrorCorrectionHead):
    name = "plane"

    def __init__(
        self,
        min_mask_pixels: int = 200,
        boundary_kernel: int = 9,
        robust_iters: int = 4,
        gate_residual_pct: float = 0.02,
        alpha_min: float = 0.5,
        alpha_max: float = 0.9,
        min_support_points: int = 50,
        require_two_sides: bool = True,
    ):
        self.min_mask_pixels = min_mask_pixels
        self.boundary_kernel = boundary_kernel
        self.robust_iters = robust_iters
        self.gate_residual_pct = gate_residual_pct
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.min_support_points = min_support_points
        self.require_two_sides = require_two_sides

    def robust_plane_fit_from_points(self, xs, ys, z):
        if len(xs) < self.min_support_points:
            return None
        xs = xs.astype(np.float64)
        ys = ys.astype(np.float64)
        z = z.astype(np.float64)

        A = np.stack([xs, ys, np.ones_like(xs)], axis=1)
        w = np.ones(len(xs), dtype=np.float64)
        coeffs = None

        for _ in range(self.robust_iters):
            Aw = A * w[:, None]
            zw = z * w
            coeffs, *_ = np.linalg.lstsq(Aw, zw, rcond=None)
            pred = A @ coeffs
            resid = np.abs(pred - z)
            med = np.median(resid)
            scale = 1.4826 * med + 1e-6
            w = 1.0 / (1.0 + (resid / (3.0 * scale)) ** 2)

        return coeffs.astype(np.float32)

    def fit_plane_map(self, depth: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        h, w = depth.shape[:2]
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return (coeffs[0] * xx + coeffs[1] * yy + coeffs[2]).astype(np.float32)

    def get_outer_wall_band(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        h, w = mask_u8.shape[:2]
        if mask_u8.sum() == 0:
            return np.zeros_like(mask_u8, dtype=np.uint8)

        kernel = np.ones((self.boundary_kernel, self.boundary_kernel), np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        band = ((dilated > 0) & (mask_u8 == 0)).astype(np.uint8)

        top_touch = (mask_u8[0, :] > 0)
        bot_touch = (mask_u8[h - 1, :] > 0)
        lef_touch = (mask_u8[:, 0] > 0)
        rig_touch = (mask_u8[:, w - 1] > 0)
        edge_margin = max(1, self.boundary_kernel // 2)

        if top_touch.any():
            band[0:min(edge_margin, h), :] = 0
        if bot_touch.any():
            band[max(0, h - edge_margin):h, :] = 0
        if lef_touch.any():
            band[:, 0:min(edge_margin, w)] = 0
        if rig_touch.any():
            band[:, max(0, w - edge_margin):w] = 0

        return band.astype(np.uint8)

    def band_side_coverage(self, mask: np.ndarray, band: np.ndarray):
        top_side = np.zeros_like(mask, dtype=np.uint8)
        bot_side = np.zeros_like(mask, dtype=np.uint8)
        lef_side = np.zeros_like(mask, dtype=np.uint8)
        rig_side = np.zeros_like(mask, dtype=np.uint8)

        top_side[1:, :] = mask[:-1, :]
        bot_side[:-1, :] = mask[1:, :]
        lef_side[:, 1:] = mask[:, :-1]
        rig_side[:, :-1] = mask[:, 1:]

        counts = {
            "top": int(np.logical_and(band > 0, top_side > 0).sum()),
            "bottom": int(np.logical_and(band > 0, bot_side > 0).sum()),
            "left": int(np.logical_and(band > 0, lef_side > 0).sum()),
            "right": int(np.logical_and(band > 0, rig_side > 0).sum()),
        }
        present = sum(v > 20 for v in counts.values())
        return counts, present

    def search_plane(self, depth: np.ndarray, mask: np.ndarray):
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() < self.min_mask_pixels:
            return depth.copy(), None, np.zeros_like(mask, dtype=np.uint8), {}

        band = self.get_outer_wall_band(mask)
        ys, xs = np.where(band > 0)
        if len(xs) == 0:
            return depth.copy(), None, band, {"side_counts": {}, "n_sides": 0}

        z = depth[ys, xs]
        finite = np.isfinite(z)
        xs, ys, z = xs[finite], ys[finite], z[finite]

        if len(z) >= self.min_support_points:
            z_lo = np.percentile(z, 5)
            z_hi = np.percentile(z, 95)
            keep = (z >= z_lo) & (z <= z_hi)
            xs, ys, z = xs[keep], ys[keep], z[keep]
            band2 = np.zeros_like(mask, dtype=np.uint8)
            band2[ys, xs] = 1
            band = band2

        side_counts, n_sides = self.band_side_coverage(mask, band)
        if self.require_two_sides and n_sides < 2:
            return depth.copy(), None, band, {"side_counts": side_counts, "n_sides": n_sides}

        if len(xs) < self.min_support_points:
            return depth.copy(), None, band, {"side_counts": side_counts, "n_sides": n_sides}

        coeffs = self.robust_plane_fit_from_points(xs, ys, z)
        if coeffs is None:
            return depth.copy(), None, band, {"side_counts": side_counts, "n_sides": n_sides}

        plane = self.fit_plane_map(depth, coeffs)
        return plane, coeffs, band, {"side_counts": side_counts, "n_sides": n_sides}

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

        plane_map, coeffs, band, meta = self.search_plane(depth, mask)
        if coeffs is None:
            return HeadResult(
                name=self.name,
                fixed_depth=depth.copy().astype(np.float32),
                aux_map=plane_map.astype(np.float32),
                support_mask=band.astype(np.uint8),
                changed_mask=np.zeros_like(mask, dtype=np.uint8),
                applied=False,
                score=np.nan,
                meta={**meta, "reason": "no_plane"},
            )

        mirror = mask > 0
        residual = np.abs(depth[mirror] - plane_map[mirror])
        local = depth[mirror]
        local_range = np.percentile(local, 95) - np.percentile(local, 5)
        local_range = float(max(local_range, 1e-6))
        gate_score = float(np.mean(residual) / local_range)

        if gate_score < self.gate_residual_pct:
            return HeadResult(
                name=self.name,
                fixed_depth=depth.copy().astype(np.float32),
                aux_map=plane_map.astype(np.float32),
                support_mask=band.astype(np.uint8),
                changed_mask=np.zeros_like(mask, dtype=np.uint8),
                applied=False,
                alpha=0.0,
                score=gate_score,
                meta={**meta, "reason": "gate_skip"},
            )

        score_hi = 3.0 * self.gate_residual_pct
        frac = np.clip((gate_score - self.gate_residual_pct) / (score_hi - self.gate_residual_pct + 1e-8), 0.0, 1.0)
        alpha = float(self.alpha_min + frac * (self.alpha_max - self.alpha_min))

        fixed = depth.copy().astype(np.float32)
        fixed[mirror] = (1.0 - alpha) * depth[mirror] + alpha * plane_map[mirror]

        changed_mask = np.zeros_like(mask, dtype=np.uint8)
        changed_mask[mirror] = 1

        return HeadResult(
            name=self.name,
            fixed_depth=fixed.astype(np.float32),
            aux_map=plane_map.astype(np.float32),
            support_mask=band.astype(np.uint8),
            changed_mask=changed_mask.astype(np.uint8),
            applied=True,
            alpha=alpha,
            score=gate_score,
            meta={**meta, "coeffs": coeffs.tolist()},
        )