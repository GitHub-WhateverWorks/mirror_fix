#!/usr/bin/env python3
"""
mirror_fix.py

Boundary-plane-fit mirror depth correction:
- load DA3 depth (.npy)
- load segmentation mask (.png or .npy)
- build a boundary band from the mirror mask
- fit a plane z = ax + by + c from boundary depth samples
- fill the whole mirror region with that plane
- save one large panel per image

Edit CONFIG only.
"""

from pathlib import Path
import numpy as np
import cv2


# =========================
# CONFIG
# =========================

DEPTH_DIR = "./depth_outputs"
MASK_DIR = "./seg_outputs/train"   # change to ./seg_outputs/test if needed
OUTPUT_DIR = "./mirror_fix_outputs"

MASK_THRESHOLD = 0.3

# boundary band settings
ERODE_KERNEL = 9     # must be odd-ish, controls inner boundary thickness
DILATE_KERNEL = 9    # optional outer boundary thickness
USE_OUTER_BAND = False  # False = use inner boundary only, True = use outer ring

# blend amount
ALPHA = 1.0          # 1.0 = fully replace mirror region with fitted plane

# safety
MIN_BOUNDARY_PIXELS = 50

# panel settings
PANEL_TEXT_SCALE = 0.8
PANEL_TEXT_THICKNESS = 2
PANEL_GAP = 12

# =========================


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    valid = arr[finite]
    lo = np.percentile(valid, 2.0)
    hi = np.percentile(valid, 98.0)

    if hi <= lo:
        lo = valid.min()
        hi = valid.max()

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return (arr * 255.0).astype(np.uint8)


def depth_to_color(depth: np.ndarray) -> np.ndarray:
    u8 = normalize_to_uint8(depth)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        PANEL_TEXT_SCALE,
        (255, 255, 255),
        PANEL_TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        PANEL_TEXT_SCALE,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def make_overlay(base_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4) -> np.ndarray:
    out = base_bgr.copy()
    color_layer = np.zeros_like(out)
    color_layer[mask > 0] = color
    out = cv2.addWeighted(out, 1.0, color_layer, alpha, 0)
    return out


def make_difference_map(raw_depth: np.ndarray, fixed_depth: np.ndarray) -> np.ndarray:
    diff = np.abs(fixed_depth.astype(np.float32) - raw_depth.astype(np.float32))
    diff_u8 = normalize_to_uint8(diff)
    return cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)


def load_mask(mask_path: Path) -> np.ndarray:
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path)

        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                mask = mask[..., 0]

        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0

        return (mask >= MASK_THRESHOLD).astype(np.uint8)

    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    return (img.astype(np.float32) / 255.0 >= MASK_THRESHOLD).astype(np.uint8)


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Minimal cleaning only.
    """
    return (mask > 0).astype(np.uint8)


def find_mask_for_stem(mask_dir: Path, stem: str) -> Path | None:
    candidates = [
        mask_dir / f"{stem}.npy",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}_mask.npy",
        mask_dir / f"{stem}_mask.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def get_boundary_band(mask: np.ndarray) -> np.ndarray:
    """
    Returns a thin boundary band.
    Default: inner boundary = mask - erode(mask)
    Optional: outer boundary = dilate(mask) - mask
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    if USE_OUTER_BAND:
        kernel = np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        band = ((dilated > 0) & (mask_u8 == 0)).astype(np.uint8)
    else:
        kernel = np.ones((ERODE_KERNEL, ERODE_KERNEL), np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        band = ((mask_u8 > 0) & (eroded == 0)).astype(np.uint8)

    return band


def fit_plane_from_band(depth: np.ndarray, band: np.ndarray):
    """
    Fit z = ax + by + c using least squares on boundary band pixels.
    Returns:
        plane_depth, success, coeffs
    """
    h, w = depth.shape
    ys, xs = np.where(band > 0)

    if len(xs) < MIN_BOUNDARY_PIXELS:
        return depth.copy(), False, None

    z = depth[ys, xs].astype(np.float32)
    finite = np.isfinite(z)
    xs = xs[finite]
    ys = ys[finite]
    z = z[finite]

    if len(z) < MIN_BOUNDARY_PIXELS:
        return depth.copy(), False, None

    # A @ [a, b, c] = z
    A = np.stack([
        xs.astype(np.float32),
        ys.astype(np.float32),
        np.ones_like(xs, dtype=np.float32)
    ], axis=1)

    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs

    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    plane = a * xx + b * yy + c

    return plane.astype(np.float32), True, coeffs


def plane_fill_fix(depth: np.ndarray, mask: np.ndarray):
    """
    Replace mirror region with fitted plane.
    Returns:
        fixed_depth
        plane_depth
        boundary_band
        changed_mask
        success
        coeffs
    """
    depth = depth.astype(np.float32)
    fixed = depth.copy()

    band = get_boundary_band(mask)
    plane, success, coeffs = fit_plane_from_band(depth, band)

    if not success:
        changed = np.zeros_like(mask, dtype=np.uint8)
        return fixed, depth.copy(), band, changed, False, coeffs

    mirror = mask > 0
    fixed[mirror] = (1.0 - ALPHA) * depth[mirror] + ALPHA * plane[mirror]

    changed = np.zeros_like(mask, dtype=np.uint8)
    changed[mirror] = 1

    return fixed, plane, band, changed, True, coeffs


def build_panel(
    stem: str,
    raw_mask: np.ndarray,
    clean_mask_img: np.ndarray,
    boundary_band: np.ndarray,
    changed_mask: np.ndarray,
    raw_depth: np.ndarray,
    plane_depth: np.ndarray,
    fixed_depth: np.ndarray,
    success: bool,
    coeffs,
) -> np.ndarray:
    raw_depth_vis = depth_to_color(raw_depth)
    plane_vis = depth_to_color(plane_depth)
    fixed_vis = depth_to_color(fixed_depth)
    diff_vis = make_difference_map(raw_depth, fixed_depth)

    raw_mask_vis = gray_to_bgr((raw_mask * 255).astype(np.uint8))
    clean_mask_vis = gray_to_bgr((clean_mask_img * 255).astype(np.uint8))
    band_vis = gray_to_bgr((boundary_band * 255).astype(np.uint8))
    changed_vis = gray_to_bgr((changed_mask * 255).astype(np.uint8))

    overlay_mask = make_overlay(raw_depth_vis, clean_mask_img, color=(0, 0, 255), alpha=0.35)
    overlay_band = make_overlay(raw_depth_vis, boundary_band, color=(0, 255, 255), alpha=0.5)
    overlay_changed = make_overlay(raw_depth_vis, changed_mask, color=(255, 255, 0), alpha=0.35)

    tiles = [
        add_title(raw_depth_vis, "Raw Depth"),
        add_title(raw_mask_vis, "Raw Mask"),
        add_title(clean_mask_vis, "Clean Mask"),
        add_title(overlay_mask, "Mask Overlay"),

        add_title(band_vis, "Boundary Band"),
        add_title(overlay_band, "Band Overlay"),
        add_title(plane_vis, "Fitted Plane"),
        add_title(fixed_vis, "Fixed Depth"),

        add_title(changed_vis, "Changed Region"),
        add_title(overlay_changed, "Changed Overlay"),
        add_title(diff_vis, "Abs Difference"),
        add_title(np.full_like(diff_vis, 255), "Info"),
    ]

    h, w = tiles[0].shape[:2]
    gap = PANEL_GAP
    white_v = np.full((h, gap, 3), 255, dtype=np.uint8)
    white_h = np.full((gap, w * 4 + gap * 3, 3), 255, dtype=np.uint8)

    row1 = np.hstack([tiles[0], white_v, tiles[1], white_v, tiles[2], white_v, tiles[3]])
    row2 = np.hstack([tiles[4], white_v, tiles[5], white_v, tiles[6], white_v, tiles[7]])
    row3 = np.hstack([tiles[8], white_v, tiles[9], white_v, tiles[10], white_v, tiles[11]])

    panel = np.vstack([row1, white_h, row2, white_h, row3])

    info = "plane fit: success" if success else "plane fit: FAILED"
    cv2.putText(panel, stem, (12, panel.shape[0] - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, info, (12, panel.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    if success and coeffs is not None:
        a, b, c = coeffs
        txt = f"z = {a:.4f}x + {b:.4f}y + {c:.4f}"
        cv2.putText(panel, txt, (w * 3 // 2, panel.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return panel


def main():
    depth_dir = Path(DEPTH_DIR)
    mask_dir = Path(MASK_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(depth_dir.glob("*.npy"))
    if not depth_files:
        raise RuntimeError(f"No depth .npy files found in {depth_dir}")

    print(f"[INFO] Found {len(depth_files)} depth files")

    for depth_path in depth_files:
        stem = depth_path.stem

        mask_path = find_mask_for_stem(mask_dir, stem)
        if mask_path is None:
            print(f"[SKIP] No mask found for {stem}")
            continue

        depth = np.load(depth_path).astype(np.float32)

        raw_mask = load_mask(mask_path)
        raw_mask = resize_mask_to_depth(raw_mask, depth)
        cleaned_mask = clean_mask(raw_mask)

        fixed_depth, plane_depth, boundary_band, changed_mask, success, coeffs = plane_fill_fix(depth, cleaned_mask)

        np.save(out_dir / f"{stem}_fixed.npy", fixed_depth)
        np.save(out_dir / f"{stem}_raw_mask.npy", raw_mask.astype(np.uint8))
        np.save(out_dir / f"{stem}_clean_mask.npy", cleaned_mask.astype(np.uint8))
        np.save(out_dir / f"{stem}_boundary_band.npy", boundary_band.astype(np.uint8))
        if success:
            np.save(out_dir / f"{stem}_plane.npy", plane_depth.astype(np.float32))

        panel = build_panel(
            stem=stem,
            raw_mask=raw_mask,
            clean_mask_img=cleaned_mask,
            boundary_band=boundary_band,
            changed_mask=changed_mask,
            raw_depth=depth,
            plane_depth=plane_depth,
            fixed_depth=fixed_depth,
            success=success,
            coeffs=coeffs,
        )
        cv2.imwrite(str(out_dir / f"{stem}_panel.png"), panel)

        print(f"[OK] {stem} | plane fit {'success' if success else 'failed'}")

    print("[DONE]")


if __name__ == "__main__":
    main()