#!/usr/bin/env python3
"""
Run Depth Anything 3 (LOCAL) — no CLI args version
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import gc

from depth_anything_3.api import DepthAnything3


# =========================
# CONFIG
# =========================
MODEL_PATH = "./Depth-Anything-3/checkpoints/da3_large"
INPUT_PATH = "./img/MSD/MSD/train/image"
OUTPUT_DIR = "./depth_outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# safer than 504 for now
PROCESS_RES = 384
PROCESS_RES_METHOD = "upper_bound_resize"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# =========================


def normalize_depth(depth):
    depth = depth.astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    depth = (depth - d_min) / (d_max - d_min)
    return (depth * 255).astype(np.uint8)


def collect_images(path):
    p = Path(path)
    if p.is_file():
        return [p]

    images = sorted([x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS])
    if len(images) == 0:
        raise RuntimeError(f"No images found in {path}")
    return images


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading model from:", MODEL_PATH)
    print("[INFO] Device:", DEVICE)
    print("[INFO] Process resolution:", PROCESS_RES)

    model = DepthAnything3.from_pretrained(MODEL_PATH)
    model = model.to(DEVICE)
    model.eval()

    images = collect_images(INPUT_PATH)
    print(f"[INFO] Found {len(images)} images")

    with torch.no_grad():
        for idx, img_path in enumerate(images, 1):
            print(f"[INFO] [{idx}/{len(images)}] {img_path.name}")

            try:
                pred = model.inference(
                    [str(img_path)],
                    process_res=PROCESS_RES,
                    process_res_method=PROCESS_RES_METHOD,
                )

                depth = pred.depth[0]

                stem = img_path.stem
                np.save(output_dir / f"{stem}.npy", depth)

                depth_vis = normalize_depth(depth)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                cv2.imwrite(str(output_dir / f"{stem}.png"), depth_vis)

                print(f"[OK] Saved {stem}.npy and {stem}.png")

            except torch.OutOfMemoryError:
                print(f"[OOM] Failed on {img_path.name}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # help reduce fragmentation
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    print("[DONE]")


if __name__ == "__main__":
    main()