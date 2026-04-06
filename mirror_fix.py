#!/usr/bin/env python3
"""
mirror_fix.py

Simple mirror-aware depth correction baseline:
- load DA3 depth (.npy)
- load segmentation mask (.png or .npy)
- apply minimal mask cleaning
- smooth the whole mirror region with masked blending
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

# smoothing strength
BLUR_KERNEL = 41   # must be odd
ALPHA = 1        # 0=no change, 1=fully smoothed inside mirror

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
    Intentionally minimal cleaning.
    Only binarize. No morphology, no component removal.
    """
    return (mask > 0).astype(np.uint8)


def masked_smooth_fix(depth: np.ndarray, mask: np.ndarray, alpha: float = ALPHA):
    """
    Smooth the entire mirror region with masked blending.
    Returns:
      fixed_depth
      smoothed_depth
      changed_mask
    """
    depth = depth.astype(np.float32)
    fixed = depth.copy()

    smoothed = cv2.GaussianBlur(depth, (BLUR_KERNEL, BLUR_KERNEL), 0)

    mirror = mask > 0
    fixed[mirror] = (1.0 - alpha) * depth[mirror] + alpha * smoothed[mirror]

    changed = np.zeros_like(mask, dtype=np.uint8)
    changed[mirror] = 1

    return fixed, smoothed, changed


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


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


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


def build_panel(
    stem: str,
    raw_mask: np.ndarray,
    clean_mask_img: np.ndarray,
    changed_mask: np.ndarray,
    raw_depth: np.ndarray,
    smoothed_depth: np.ndarray,
    fixed_depth: np.ndarray,
) -> np.ndarray:
    raw_depth_vis = depth_to_color(raw_depth)
    smoothed_vis = depth_to_color(smoothed_depth)
    fixed_vis = depth_to_color(fixed_depth)
    diff_vis = make_difference_map(raw_depth, fixed_depth)

    raw_mask_vis = gray_to_bgr((raw_mask * 255).astype(np.uint8))
    clean_mask_vis = gray_to_bgr((clean_mask_img * 255).astype(np.uint8))
    changed_mask_vis = gray_to_bgr((changed_mask * 255).astype(np.uint8))

    overlay_clean = make_overlay(raw_depth_vis, clean_mask_img, color=(0, 0, 255), alpha=0.35)
    overlay_changed = make_overlay(raw_depth_vis, changed_mask, color=(0, 255, 255), alpha=0.35)

    tiles = [
        add_title(raw_depth_vis, "Raw Depth"),
        add_title(raw_mask_vis, "Raw Mask"),
        add_title(clean_mask_vis, "Clean Mask"),
        add_title(overlay_clean, "Mask Overlay"),
        add_title(smoothed_vis, "Smoothed Depth"),
        add_title(changed_mask_vis, "Changed Region"),
        add_title(overlay_changed, "Changed Overlay"),
        add_title(fixed_vis, "Fixed Depth"),
    ]

    h, w = tiles[0].shape[:2]
    gap = PANEL_GAP
    white_v = np.full((h, gap, 3), 255, dtype=np.uint8)
    white_h = np.full((gap, w * 4 + gap * 3, 3), 255, dtype=np.uint8)

    row1 = np.hstack([tiles[0], white_v, tiles[1], white_v, tiles[2], white_v, tiles[3]])
    row2 = np.hstack([tiles[4], white_v, tiles[5], white_v, tiles[6], white_v, tiles[7]])

    panel = np.vstack([row1, white_h, row2])

    # optional third strip for difference map
    diff_tile = add_title(diff_vis, "Abs Difference")
    blank = np.full_like(diff_tile, 255)
    row3 = np.hstack([diff_tile, white_v, blank, white_v, blank, white_v, blank])
    panel = np.vstack([panel, white_h, row3])

    cv2.putText(
        panel,
        stem,
        (12, panel.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

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

        fixed_depth, smoothed_depth, changed_mask = masked_smooth_fix(depth, cleaned_mask, alpha=ALPHA)

        np.save(out_dir / f"{stem}_fixed.npy", fixed_depth)
        np.save(out_dir / f"{stem}_raw_mask.npy", raw_mask.astype(np.uint8))
        np.save(out_dir / f"{stem}_clean_mask.npy", cleaned_mask.astype(np.uint8))
        np.save(out_dir / f"{stem}_changed_mask.npy", changed_mask.astype(np.uint8))

        panel = build_panel(
            stem=stem,
            raw_mask=raw_mask,
            clean_mask_img=cleaned_mask,
            changed_mask=changed_mask,
            raw_depth=depth,
            smoothed_depth=smoothed_depth,
            fixed_depth=fixed_depth,
        )
        cv2.imwrite(str(out_dir / f"{stem}_panel.png"), panel)

        print(f"[OK] {stem}")

    print("[DONE]")


if __name__ == "__main__":
    main()