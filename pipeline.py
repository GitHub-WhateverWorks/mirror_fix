#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from heads import PlaneFitHead, GaussianFillHead, DiffusionFillHead


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PANELS = True
MIRROR_CLASS_IDX = 27
TOPK_DEBUG_CHANNELS = 3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RUN_PAIR_EVAL = True
RUN_MSD_EVAL = False

PAIR_DATASET_DIR = "./img/dataset_pairs"
PAIR_OUTPUT_DIR = "./eval_outputs_pairs_modular"
PAIR_MAX_IMAGES = 5
PAIR_SELECTED_IDS = ["0011"]

PAIR_ALIGN_METHOD = "ecc_affine"
PAIR_ALIGN_ECC_ITERS = 100
PAIR_ALIGN_ECC_EPS = 1e-5
PAIR_ALIGN_GAUSSIAN_BLUR = 5
PAIR_EVAL_USE_ALIGNMENT = True

PAIR_EVAL_ON_FULL = True
PAIR_EVAL_ON_USED_MASK = True
PAIR_EVAL_ON_CHANGED_MASK = True
PAIR_EVAL_MASK_ERODE_KERNEL = 5
PAIR_EVAL_BLUR_DEPTH = 0

MSD_IMAGE_DIR = "./img/MSD/MSD/train/image"
MSD_MASK_DIR = "./img/MSD/MSD/train/mask"
MSD_OUTPUT_JSON = "./eval_outputs_msd/segmentor_metrics.json"
MSD_MAX_IMAGES = 200

REPO_DIR = "/home/charlieof604/project2/dinov3"
SEG_WEIGHTS = "./dinov3_weights/dinov3_vit7b16_segmentor.pth"
SEG_BACKBONE_WEIGHTS = "./dinov3_weights/dinov3_vit7b16_pretrain_backbone.pth"
DA3_MODEL_PATH = "./Depth-Anything-3/checkpoints/da3_large"

WORK_SIZE = 384
PROCESS_RES = 384
PROCESS_RES_METHOD = "upper_bound_resize"

MASK_MODES_TO_RUN = ["argmax", "argmax_refine"]
HEADS_TO_RUN = ["plane", "gaussian", "diffusion"]

SEG_MIRROR_PROB_THRESHOLD = 0.01
SEG_PERCENTILE = 90.0
SEG_MIN_ABS_THRESH = 1e-5
ARGMAX_REFINE_PERCENTILE = 12.0
ARGMAX_REFINE_MIN_ABS_THRESH = 1e-5

FILL_MASK_HOLES = True
REMOVE_SMALL_COMPONENTS = True
MIN_COMPONENT_AREA = 800
USE_ERODED_PRED_MASK = True
PRED_ERODE_KERNEL = 9
PRED_ERODE_ITERS = 1


# =========================
# HELPERS
# =========================
def safe_mean(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape[:2]

    flood = mask_u8.copy()
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, floodfill_mask, (0, 0), 1)

    outside = flood
    holes = ((outside == 0) & (mask_u8 == 0)).astype(np.uint8)
    return np.maximum(mask_u8, holes).astype(np.uint8)


def collect_images(path: str, max_images: int | None = None):
    p = Path(path)
    if p.is_file():
        images = [p]
    else:
        images = sorted([x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS])
    if max_images is not None:
        images = images[:max_images]
    return images


def collect_pair_samples(root: str, max_images: int | None = None, selected_ids=None):
    base = Path(root)
    if not base.exists():
        raise RuntimeError(f"Pair dataset dir not found: {base}")
    dirs = sorted([d for d in base.iterdir() if d.is_dir()])

    if selected_ids is not None:
        wanted = {str(x) for x in selected_ids}
        dirs = [d for d in dirs if d.name in wanted]
        missing = sorted(wanted - {d.name for d in dirs})
        if missing:
            print(f"[WARN] Missing requested pair IDs: {missing}")

    if max_images is not None and selected_ids is None:
        dirs = dirs[:max_images]

    samples = []
    for d in dirs:
        mirror = None
        covered = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            m = d / f"mirror{ext}"
            c = d / f"covered{ext}"
            if m.exists():
                mirror = m
            if c.exists():
                covered = c
        if mirror is None or covered is None:
            print(f"[PAIR][SKIP] missing mirror/covered in {d}")
            continue
        samples.append({"id": d.name, "mirror": mirror, "covered": covered})
    return samples


def resize_depth_to_match(depth: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if depth.shape[:2] == (target_h, target_w):
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_mask_to_match(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if mask.shape[:2] == (target_h, target_w):
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def resize_prob_to_match(prob: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if prob.shape[:2] == (target_h, target_w):
        return prob.astype(np.float32)
    return cv2.resize(prob.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    valid = arr[finite]
    lo = np.percentile(valid, 2.0)
    hi = np.percentile(valid, 98.0)
    if hi <= lo:
        lo, hi = valid.min(), valid.max()
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return (arr * 255.0).astype(np.uint8)


def prob_to_uint8(prob: np.ndarray) -> np.ndarray:
    prob = prob.astype(np.float32)
    lo = float(np.min(prob))
    hi = float(np.max(prob))
    if hi <= lo + 1e-8:
        return np.zeros(prob.shape, dtype=np.uint8)
    x = (prob - lo) / (hi - lo)
    return (x * 255.0).astype(np.uint8)


def heatmap_color(prob: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(prob_to_uint8(prob), cv2.COLORMAP_TURBO)


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)


def overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.35) -> np.ndarray:
    out = base_bgr.copy()
    color_layer = np.zeros_like(out)
    color_layer[mask > 0] = color
    return cv2.addWeighted(out, 1.0, color_layer, alpha, 0)


def text_tile(lines, size=(384, 384)) -> np.ndarray:
    h, w = size
    tile = np.full((h, w, 3), 255, dtype=np.uint8)
    y = 28
    for line in lines:
        cv2.putText(tile, str(line), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
        if y > h - 10:
            break
    return tile


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 1
    return out


def erode_mask(mask: np.ndarray, kernel_size: int, iters: int) -> np.ndarray:
    if kernel_size <= 1 or iters <= 0:
        return (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=iters)


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    valid = (mask > 0) & np.isfinite(pred) & np.isfinite(gt)
    if valid.sum() == 0:
        return np.nan, np.nan
    pv = pred[valid]
    gv = gt[valid]
    mae = np.mean(np.abs(pv - gv))
    rmse = np.sqrt(np.mean((pv - gv) ** 2))
    return float(mae), float(rmse)


def binary_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = pred_mask > 0
    gt = gt_mask > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_area = pred.sum()
    gt_area = gt.sum()
    iou = inter / union if union > 0 else np.nan
    precision = inter / pred_area if pred_area > 0 else 0.0
    recall = inter / gt_area if gt_area > 0 else 0.0
    dice = (2 * inter) / (pred_area + gt_area) if (pred_area + gt_area) > 0 else np.nan
    return {
        "pred_pixels": int(pred_area),
        "gt_pixels": int(gt_area),
        "intersection": int(inter),
        "union": int(union),
        "iou": float(iou) if not np.isnan(iou) else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice) if not np.isnan(dice) else np.nan,
    }


def read_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8)


def find_mask_for_image(mask_dir: Path, image_stem: str):
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        p = mask_dir / f"{image_stem}{ext}"
        if p.exists():
            return p
    return None


def blur_depth_if_needed(depth: np.ndarray) -> np.ndarray:
    if PAIR_EVAL_BLUR_DEPTH and PAIR_EVAL_BLUR_DEPTH > 1:
        k = int(PAIR_EVAL_BLUR_DEPTH)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(depth.astype(np.float32), (k, k), 0)
    return depth.astype(np.float32)


def erode_eval_mask_if_needed(mask: np.ndarray) -> np.ndarray:
    k = int(PAIR_EVAL_MASK_ERODE_KERNEL)
    if k <= 1:
        return (mask > 0).astype(np.uint8)
    return erode_mask(mask, k, 1)


def align_covered_to_mirror(mirror_rgb: np.ndarray, covered_rgb: np.ndarray, covered_depth: np.ndarray):
    h, w = mirror_rgb.shape[:2]
    if covered_rgb.shape[:2] != (h, w):
        covered_rgb = cv2.resize(covered_rgb.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    covered_depth = resize_depth_to_match(covered_depth, (h, w))

    if (not PAIR_EVAL_USE_ALIGNMENT) or PAIR_ALIGN_METHOD == "none":
        return covered_rgb, covered_depth, {"align_ok": True, "align_method": "none", "cc": np.nan}

    gray1 = cv2.cvtColor(mirror_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(covered_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if PAIR_ALIGN_GAUSSIAN_BLUR and PAIR_ALIGN_GAUSSIAN_BLUR > 1:
        k = int(PAIR_ALIGN_GAUSSIAN_BLUR)
        if k % 2 == 0:
            k += 1
        gray1 = cv2.GaussianBlur(gray1, (k, k), 0)
        gray2 = cv2.GaussianBlur(gray2, (k, k), 0)

    if PAIR_ALIGN_METHOD == "ecc_translation":
        warp_mode = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
    elif PAIR_ALIGN_METHOD == "ecc_euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp = np.eye(2, 3, dtype=np.float32)
    elif PAIR_ALIGN_METHOD == "ecc_affine":
        warp_mode = cv2.MOTION_AFFINE
        warp = np.eye(2, 3, dtype=np.float32)
    else:
        return covered_rgb, covered_depth, {"align_ok": False, "align_method": PAIR_ALIGN_METHOD, "cc": np.nan, "error": "unknown method"}

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, PAIR_ALIGN_ECC_ITERS, PAIR_ALIGN_ECC_EPS)
    try:
        cc, warp = cv2.findTransformECC(gray1, gray2, warp, warp_mode, criteria)
        aligned_rgb = cv2.warpAffine(
            covered_rgb,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned_depth = cv2.warpAffine(
            covered_depth.astype(np.float32),
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned_rgb, aligned_depth.astype(np.float32), {
            "align_ok": True,
            "align_method": PAIR_ALIGN_METHOD,
            "cc": float(cc),
            "warp": warp.tolist(),
        }
    except cv2.error as e:
        return covered_rgb, covered_depth.astype(np.float32), {
            "align_ok": False,
            "align_method": PAIR_ALIGN_METHOD,
            "cc": np.nan,
            "error": str(e),
        }


# =========================
# MODELS
# =========================
sys.path.append(REPO_DIR)
from dinov3.eval.segmentation.inference import make_inference
from depth_anything_3.api import DepthAnything3


def load_segmentor():
    print("[MODEL] Loading segmentor...")
    model = torch.hub.load(
        REPO_DIR,
        "dinov3_vit7b16_ms",
        source="local",
        weights=SEG_WEIGHTS,
        backbone_weights=SEG_BACKBONE_WEIGHTS,
        torch_dtype=torch.bfloat16,
    )
    return model.to(DEVICE).eval()


def load_da3():
    print("[MODEL] Loading Depth Anything 3...")
    model = DepthAnything3.from_pretrained(DA3_MODEL_PATH)
    return model.to(DEVICE).eval()


# =========================
# SEGMENTOR
# =========================
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((WORK_SIZE, WORK_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def segment_with_debug(model, img: Image.Image, seg_mask_mode: str):
    orig_w, orig_h = img.size
    x = transform(img).unsqueeze(0).to(DEVICE)

    use_amp = DEVICE.startswith("cuda")
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                probs = make_inference(
                    x,
                    model,
                    inference_mode="whole",
                    decoder_head_type="m2f",
                    rescale_to=(orig_h, orig_w),
                    n_output_channels=150,
                    output_activation=partial(F.softmax, dim=1),
                )
        else:
            probs = make_inference(
                x,
                model,
                inference_mode="whole",
                decoder_head_type="m2f",
                rescale_to=(orig_h, orig_w),
                n_output_channels=150,
                output_activation=partial(F.softmax, dim=1),
            )

    probs = probs.detach().float().cpu()
    seg = probs.argmax(dim=1)[0].numpy().astype(np.int32)
    argmax_mask = (seg == MIRROR_CLASS_IDX).astype(np.uint8)
    mirror_prob = probs[0, MIRROR_CLASS_IDX].numpy().astype(np.float32)

    thr_used = None
    refine_region_pixels = int(argmax_mask.sum())

    if seg_mask_mode == "argmax":
        pred_mask = argmax_mask.copy()
    elif seg_mask_mode == "threshold":
        thr_used = float(SEG_MIRROR_PROB_THRESHOLD)
        pred_mask = (mirror_prob >= thr_used).astype(np.uint8)
    elif seg_mask_mode == "percentile":
        thr_used = float(np.percentile(mirror_prob, SEG_PERCENTILE))
        thr_used = max(thr_used, float(SEG_MIN_ABS_THRESH))
        pred_mask = (mirror_prob >= thr_used).astype(np.uint8)
    elif seg_mask_mode == "argmax_refine":
        if refine_region_pixels > 0:
            vals = mirror_prob[argmax_mask > 0]
            thr_used = float(np.percentile(vals, ARGMAX_REFINE_PERCENTILE))
            thr_used = max(thr_used, float(ARGMAX_REFINE_MIN_ABS_THRESH))
            pred_mask = ((argmax_mask > 0) & (mirror_prob >= thr_used)).astype(np.uint8)
        else:
            pred_mask = np.zeros_like(argmax_mask, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown seg_mask_mode: {seg_mask_mode}")

    avg_probs = probs[0].mean(dim=(1, 2)).numpy()
    max_probs = probs[0].amax(dim=(1, 2)).numpy()

    topk_avg = np.argsort(avg_probs)[-TOPK_DEBUG_CHANNELS:][::-1]
    topk_max = np.argsort(max_probs)[-TOPK_DEBUG_CHANNELS:][::-1]
    topk_channels = []
    seen = set()
    for idx in list(topk_max) + list(topk_avg):
        i = int(idx)
        if i not in seen:
            seen.add(i)
            topk_channels.append(i)
        if len(topk_channels) >= TOPK_DEBUG_CHANNELS:
            break

    topk_maps = []
    for idx in topk_channels:
        ch_prob = probs[0, idx].numpy().astype(np.float32)
        topk_maps.append({
            "class_idx": idx,
            "mean_prob": float(avg_probs[idx]),
            "max_prob": float(max_probs[idx]),
            "prob_map": ch_prob,
        })

    stats = {
        "seg_mask_mode": str(seg_mask_mode),
        "mirror_prob_min": float(mirror_prob.min()),
        "mirror_prob_max": float(mirror_prob.max()),
        "mirror_prob_mean": float(mirror_prob.mean()),
        "threshold_used": float(thr_used) if thr_used is not None else None,
        "argmax_pixels": int(argmax_mask.sum()),
        "pred_mask_pixels": int(pred_mask.sum()),
        "refine_region_pixels": refine_region_pixels,
        "topk_avg": [(int(i), float(avg_probs[i])) for i in np.argsort(avg_probs)[-5:][::-1]],
        "topk_max": [(int(i), float(max_probs[i])) for i in np.argsort(max_probs)[-5:][::-1]],
    }

    return {
        "argmax_mask": argmax_mask,
        "pred_mask": pred_mask,
        "mirror_prob": mirror_prob,
        "topk_maps": topk_maps,
        "stats": stats,
    }


def choose_raw_mask(seg_debug: dict, seg_mask_mode: str) -> np.ndarray:
    if seg_mask_mode == "argmax":
        return seg_debug["argmax_mask"].copy()
    if seg_mask_mode in ["threshold", "percentile", "argmax_refine"]:
        return seg_debug["pred_mask"].copy()
    raise ValueError(f"Unknown seg_mask_mode: {seg_mask_mode}")


# =========================
# DA3
# =========================
def run_da3(model, path: Path) -> np.ndarray:
    pred = model.inference(
        [str(path)],
        process_res=PROCESS_RES,
        process_res_method=PROCESS_RES_METHOD,
    )
    return pred.depth[0].astype(np.float32)


# =========================
# FACTORIES
# =========================
def build_heads():
    heads = []
    for name in HEADS_TO_RUN:
        if name == "plane":
            heads.append(PlaneFitHead())
        elif name == "gaussian":
            heads.append(GaussianFillHead(blur_kernel=9, iters=60, alpha=1.0))
        elif name == "diffusion":
            heads.append(DiffusionFillHead(iters=200, step=0.24))
        else:
            raise ValueError(f"Unknown head: {name}")
    return heads


# =========================
# PANELS
# =========================
def save_pair_panel(
    out_path: Path,
    pair_id: str,
    mask_mode: str,
    head_name: str,
    mirror_rgb: np.ndarray,
    covered_rgb_aligned: np.ndarray,
    mirror_prob: np.ndarray,
    argmax_mask: np.ndarray,
    pred_mask: np.ndarray,
    raw_pred_mask: np.ndarray,
    used_pred_mask: np.ndarray,
    changed_mask: np.ndarray,
    raw_depth: np.ndarray,
    fixed_depth: np.ndarray,
    covered_depth_aligned: np.ndarray,
    aux_map: np.ndarray,
    support_mask: np.ndarray,
    stats_lines: list[str],
):
    def add_title_big(img: np.ndarray, title: str) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        header_h = max(42, h // 14)
        cv2.rectangle(out, (0, 0), (w, header_h), (255, 255, 255), -1)
        cv2.putText(out, title, (10, int(header_h * 0.72)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        return out

    def depth_to_color_shared(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
        x = depth.astype(np.float32)
        if hi <= lo + 1e-8:
            u8 = np.zeros(x.shape, dtype=np.uint8)
        else:
            x = np.clip(x, lo, hi)
            x = (x - lo) / (hi - lo + 1e-8)
            u8 = (x * 255.0).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)

    mirror_bgr = cv2.cvtColor(mirror_rgb, cv2.COLOR_RGB2BGR)
    covered_bgr = cv2.cvtColor(covered_rgb_aligned, cv2.COLOR_RGB2BGR)

    valid_vals = np.concatenate([
        raw_depth[np.isfinite(raw_depth)].reshape(-1),
        fixed_depth[np.isfinite(fixed_depth)].reshape(-1),
        covered_depth_aligned[np.isfinite(covered_depth_aligned)].reshape(-1),
    ]) if (
        np.isfinite(raw_depth).any()
        and np.isfinite(fixed_depth).any()
        and np.isfinite(covered_depth_aligned).any()
    ) else np.array([0.0, 1.0], dtype=np.float32)

    lo = float(np.percentile(valid_vals, 2.0))
    hi = float(np.percentile(valid_vals, 98.0))
    if hi <= lo:
        lo = float(valid_vals.min())
        hi = float(valid_vals.max())
    if hi <= lo:
        hi = lo + 1.0

    raw_depth_vis = depth_to_color_shared(raw_depth, lo, hi)
    fixed_depth_vis = depth_to_color_shared(fixed_depth, lo, hi)
    covered_depth_vis = depth_to_color_shared(covered_depth_aligned, lo, hi)
    aux_map_vis = depth_to_color_shared(aux_map, lo, hi)

    raw_vs_cov = np.abs(raw_depth - covered_depth_aligned)
    fix_vs_cov = np.abs(fixed_depth - covered_depth_aligned)
    raw_vs_fix = np.abs(raw_depth - fixed_depth)

    raw_vs_cov_vis = cv2.applyColorMap(normalize_to_uint8(raw_vs_cov), cv2.COLORMAP_INFERNO)
    fix_vs_cov_vis = cv2.applyColorMap(normalize_to_uint8(fix_vs_cov), cv2.COLORMAP_INFERNO)
    raw_vs_fix_vis = cv2.applyColorMap(normalize_to_uint8(raw_vs_fix), cv2.COLORMAP_INFERNO)

    tiles = [
        add_title_big(mirror_bgr, f"{pair_id} Mirror RGB"),
        add_title_big(covered_bgr, f"{pair_id} Covered aligned"),
        add_title_big(heatmap_color(mirror_prob), f"Mirror Prob"),
        add_title_big(mask_to_bgr(argmax_mask), "Argmax Mask"),

        add_title_big(mask_to_bgr(pred_mask), f"Pred Mask ({mask_mode})"),
        add_title_big(mask_to_bgr(raw_pred_mask), "Raw Pred"),
        add_title_big(mask_to_bgr(used_pred_mask), "Used Mask"),
        add_title_big(mask_to_bgr(changed_mask), f"{head_name} Changed"),

        add_title_big(raw_depth_vis, "Raw Depth"),
        add_title_big(fixed_depth_vis, f"{head_name} Fixed"),
        add_title_big(covered_depth_vis, "Covered target"),
        add_title_big(aux_map_vis, f"{head_name} Aux"),

        add_title_big(mask_to_bgr(support_mask), f"{head_name} Support"),
        add_title_big(raw_vs_cov_vis, "|Raw-Covered|"),
        add_title_big(fix_vs_cov_vis, f"|{head_name}-Covered|"),
        add_title_big(raw_vs_fix_vis, "|Raw-Fixed|"),

        add_title_big(text_tile(stats_lines, size=mirror_bgr.shape[:2]), "Stats"),
    ]

    while len(tiles) < 20:
        tiles.append(np.full_like(tiles[0], 255))

    h, w = tiles[0].shape[:2]
    gap = 10
    white_v = np.full((h, gap, 3), 255, dtype=np.uint8)
    white_h = np.full((gap, w * 4 + gap * 3, 3), 255, dtype=np.uint8)

    rows = []
    for r in range(5):
        row_tiles = tiles[r * 4:(r + 1) * 4]
        row = np.hstack([row_tiles[0], white_v, row_tiles[1], white_v, row_tiles[2], white_v, row_tiles[3]])
        rows.append(row)

    panel = np.vstack([rows[0], white_h, rows[1], white_h, rows[2], white_h, rows[3], white_h, rows[4]])
    cv2.imwrite(str(out_path), panel)


# =========================
# EVAL
# =========================
def _append_metric(store, mae_raw, mae_fix, rmse_raw, rmse_fix):
    store["mae_raw"].append(mae_raw)
    store["mae_fix"].append(mae_fix)
    store["rmse_raw"].append(rmse_raw)
    store["rmse_fix"].append(rmse_fix)


def _summarize_metric(store):
    mae_raw = safe_mean(store["mae_raw"])
    mae_fix = safe_mean(store["mae_fix"])
    rmse_raw = safe_mean(store["rmse_raw"])
    rmse_fix = safe_mean(store["rmse_fix"])
    return {
        "mae_raw": mae_raw,
        "mae_fix": mae_fix,
        "rmse_raw": rmse_raw,
        "rmse_fix": rmse_fix,
        "delta_mae": mae_raw - mae_fix if not np.isnan(mae_raw) and not np.isnan(mae_fix) else np.nan,
        "delta_rmse": rmse_raw - rmse_fix if not np.isnan(rmse_raw) and not np.isnan(rmse_fix) else np.nan,
    }


def init_combo_store():
    return {
        "all_metrics": {"mae_raw": [], "mae_fix": [], "rmse_raw": [], "rmse_fix": []},
        "used_region_metrics": {"mae_raw": [], "mae_fix": [], "rmse_raw": [], "rmse_fix": []},
        "changed_region_metrics": {"mae_raw": [], "mae_fix": [], "rmse_raw": [], "rmse_fix": []},
        "per_image": [],
    }


def evaluate_pairs(seg_model, da3_model, heads):
    print("\n========== PAIRS: mirror/covered evaluation ==========")
    os.makedirs(PAIR_OUTPUT_DIR, exist_ok=True)

    samples = collect_pair_samples(PAIR_DATASET_DIR, PAIR_MAX_IMAGES, PAIR_SELECTED_IDS)
    if not samples:
        raise RuntimeError(f"No valid pair samples found in {PAIR_DATASET_DIR}")

    summaries = {}
    for mask_mode in MASK_MODES_TO_RUN:
        for head in heads:
            summaries[f"{mask_mode}__{head.name}"] = init_combo_store()

    for sample in samples:
        pair_id = sample["id"]
        print(f"[PAIR] {pair_id}")

        mirror_img = Image.open(sample["mirror"]).convert("RGB")
        covered_img = Image.open(sample["covered"]).convert("RGB")
        mirror_np = np.array(mirror_img)
        covered_np = np.array(covered_img)
        H, W = mirror_np.shape[:2]

        raw_depth = run_da3(da3_model, sample["mirror"])
        covered_depth = run_da3(da3_model, sample["covered"])
        raw_depth = resize_depth_to_match(raw_depth, (H, W))
        covered_depth = resize_depth_to_match(covered_depth, covered_np.shape[:2])

        covered_rgb_aligned, covered_depth_aligned, align_info = align_covered_to_mirror(
            mirror_np, covered_np, covered_depth
        )

        target_depth = blur_depth_if_needed(covered_depth_aligned)
        raw_eval = blur_depth_if_needed(raw_depth)
        full_mask = np.ones((H, W), dtype=np.uint8)

        for mask_mode in MASK_MODES_TO_RUN:
            seg_debug = segment_with_debug(seg_model, mirror_img, mask_mode)
            argmax_mask = resize_mask_to_match(seg_debug["argmax_mask"], (H, W))
            pred_mask = resize_mask_to_match(seg_debug["pred_mask"], (H, W))
            mirror_prob = resize_prob_to_match(seg_debug["mirror_prob"], (H, W))

            raw_pred_mask = choose_raw_mask({"argmax_mask": argmax_mask, "pred_mask": pred_mask}, mask_mode)
            if REMOVE_SMALL_COMPONENTS:
                raw_pred_mask = remove_small_components(raw_pred_mask, MIN_COMPONENT_AREA)
            if FILL_MASK_HOLES:
                raw_pred_mask = fill_mask_holes(raw_pred_mask)

            used_pred_mask = raw_pred_mask.copy()
            if USE_ERODED_PRED_MASK:
                used_pred_mask = erode_mask(used_pred_mask, PRED_ERODE_KERNEL, PRED_ERODE_ITERS)

            used_eval_mask = erode_eval_mask_if_needed(used_pred_mask)

            stats = seg_debug["stats"]

            for head in heads:
                combo_key = f"{mask_mode}__{head.name}"
                result = head.run(raw_depth, used_pred_mask)
                fix_eval = blur_depth_if_needed(result.fixed_depth)
                changed_eval_mask = erode_eval_mask_if_needed(result.changed_mask)

                mae_raw_all, rmse_raw_all = depth_metrics(raw_eval, target_depth, full_mask)
                mae_fix_all, rmse_fix_all = depth_metrics(fix_eval, target_depth, full_mask)
                _append_metric(summaries[combo_key]["all_metrics"], mae_raw_all, mae_fix_all, rmse_raw_all, rmse_fix_all)

                mae_raw_used, rmse_raw_used = (
                    depth_metrics(raw_eval, target_depth, used_eval_mask)
                    if PAIR_EVAL_ON_USED_MASK else (np.nan, np.nan)
                )
                mae_fix_used, rmse_fix_used = (
                    depth_metrics(fix_eval, target_depth, used_eval_mask)
                    if PAIR_EVAL_ON_USED_MASK else (np.nan, np.nan)
                )
                _append_metric(summaries[combo_key]["used_region_metrics"], mae_raw_used, mae_fix_used, rmse_raw_used, rmse_fix_used)

                mae_raw_changed, rmse_raw_changed = (
                    depth_metrics(raw_eval, target_depth, changed_eval_mask)
                    if PAIR_EVAL_ON_CHANGED_MASK else (np.nan, np.nan)
                )
                mae_fix_changed, rmse_fix_changed = (
                    depth_metrics(fix_eval, target_depth, changed_eval_mask)
                    if PAIR_EVAL_ON_CHANGED_MASK else (np.nan, np.nan)
                )
                _append_metric(summaries[combo_key]["changed_region_metrics"], mae_raw_changed, mae_fix_changed, rmse_raw_changed, rmse_fix_changed)

                raw_fix_mae = float(np.mean(np.abs(raw_depth - result.fixed_depth)))
                raw_cov_mae = float(np.mean(np.abs(raw_depth - covered_depth_aligned)))
                fix_cov_mae = float(np.mean(np.abs(result.fixed_depth - covered_depth_aligned)))

                print(
                    f"  [{mask_mode} + {head.name}] full MAE {mae_raw_all:.3f}->{mae_fix_all:.3f} "
                    f"(Δ {mae_raw_all - mae_fix_all:+.3f}) | used MAE {mae_raw_used:.3f}->{mae_fix_used:.3f} "
                    f"(Δ {mae_raw_used - mae_fix_used:+.3f}) | applied={result.applied}"
                )

                summaries[combo_key]["per_image"].append({
                    "id": pair_id,
                    "mask_mode": mask_mode,
                    "head": head.name,
                    "align_info": align_info,
                    "applied": bool(result.applied),
                    "alpha": float(result.alpha),
                    "score": float(result.score) if not np.isnan(result.score) else np.nan,
                    "argmax_pixels": stats["argmax_pixels"],
                    "pred_mask_pixels": stats["pred_mask_pixels"],
                    "raw_pred_pixels": int((raw_pred_mask > 0).sum()),
                    "used_pred_pixels": int((used_pred_mask > 0).sum()),
                    "changed_pixels": int((result.changed_mask > 0).sum()),
                    "support_pixels": int((result.support_mask > 0).sum()),
                    "full_mae_raw": mae_raw_all,
                    "full_mae_fix": mae_fix_all,
                    "used_mae_raw": mae_raw_used,
                    "used_mae_fix": mae_fix_used,
                    "changed_mae_raw": mae_raw_changed,
                    "changed_mae_fix": mae_fix_changed,
                    "raw_fix_mae": raw_fix_mae,
                    "raw_cov_mae": raw_cov_mae,
                    "fix_cov_mae": fix_cov_mae,
                    "meta": result.meta,
                })

                if SAVE_PANELS:
                    stats_lines = [
                        f"pair_id: {pair_id}",
                        f"mask_mode: {mask_mode}",
                        f"head: {head.name}",
                        f"mirror min/mean/max: {stats['mirror_prob_min']:.5f}/{stats['mirror_prob_mean']:.5f}/{stats['mirror_prob_max']:.5f}",
                        f"argmax pixels: {stats['argmax_pixels']}",
                        f"pred pixels: {stats['pred_mask_pixels']}",
                        f"used pred pixels: {int((used_pred_mask > 0).sum())}",
                        f"changed pixels: {int((result.changed_mask > 0).sum())}",
                        f"support pixels: {int((result.support_mask > 0).sum())}",
                        f"align: {align_info.get('align_method')} ok={align_info.get('align_ok')} cc={align_info.get('cc')}",
                        f"RAW-COV MAE: {raw_cov_mae:.3f}",
                        f"FIX-COV MAE: {fix_cov_mae:.3f}",
                        f"RAW-FIX MAE: {raw_fix_mae:.3f}",
                        f"FULL MAE raw->fix: {mae_raw_all:.3f}->{mae_fix_all:.3f}",
                        f"USED MAE raw->fix: {mae_raw_used:.3f}->{mae_fix_used:.3f}",
                        f"CHANGED MAE raw->fix: {mae_raw_changed:.3f}->{mae_fix_changed:.3f}",
                        f"applied: {result.applied}",
                        f"alpha: {result.alpha:.3f}",
                        f"score: {result.score}",
                    ]
                    panel_path = Path(PAIR_OUTPUT_DIR) / f"{pair_id}_{mask_mode}_{head.name}_panel.png"
                    save_pair_panel(
                        panel_path,
                        pair_id,
                        mask_mode,
                        head.name,
                        mirror_np,
                        covered_rgb_aligned,
                        mirror_prob,
                        argmax_mask,
                        pred_mask,
                        raw_pred_mask,
                        used_pred_mask,
                        result.changed_mask,
                        raw_depth,
                        result.fixed_depth,
                        covered_depth_aligned,
                        result.aux_map,
                        result.support_mask,
                        stats_lines,
                    )

        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    final_summary = {}
    leaderboard = []

    for mask_mode in MASK_MODES_TO_RUN:
        for head in heads:
            combo_key = f"{mask_mode}__{head.name}"
            combo_summary = {
                "processed": len(summaries[combo_key]["per_image"]),
                "mask_mode": mask_mode,
                "head": head.name,
                "all_images": _summarize_metric(summaries[combo_key]["all_metrics"]),
                "used_mask_region": _summarize_metric(summaries[combo_key]["used_region_metrics"]),
                "changed_region": _summarize_metric(summaries[combo_key]["changed_region_metrics"]),
                "per_image": summaries[combo_key]["per_image"],
            }
            final_summary[combo_key] = combo_summary

            out_json = Path(PAIR_OUTPUT_DIR) / f"pair_eval_summary_{mask_mode}_{head.name}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(combo_summary, f, indent=2)

            leaderboard.append({
                "combo": combo_key,
                "mask_mode": mask_mode,
                "head": head.name,
                "full_delta_mae": combo_summary["all_images"]["delta_mae"],
                "used_delta_mae": combo_summary["used_mask_region"]["delta_mae"],
                "changed_delta_mae": combo_summary["changed_region"]["delta_mae"],
            })

    leaderboard.sort(key=lambda x: (-np.nan_to_num(x["used_delta_mae"], nan=-1e18), -np.nan_to_num(x["full_delta_mae"], nan=-1e18)))

    aggregate = {
        "processed_pairs": len(samples),
        "mask_modes": MASK_MODES_TO_RUN,
        "heads": [h.name for h in heads],
        "leaderboard": leaderboard,
        "combinations": {k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in final_summary.items()},
    }

    aggregate_json = Path(PAIR_OUTPUT_DIR) / "pair_eval_summary_all_combos.json"
    with open(aggregate_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    return final_summary, aggregate


def evaluate_segmentor_on_msd(seg_model):
    print("\n========== MSD: segmentor mask evaluation ==========")
    image_dir = Path(MSD_IMAGE_DIR)
    mask_dir = Path(MSD_MASK_DIR)
    os.makedirs(Path(MSD_OUTPUT_JSON).parent, exist_ok=True)

    images = collect_images(str(image_dir), MSD_MAX_IMAGES)
    if len(images) == 0:
        raise RuntimeError(f"No MSD images found in {image_dir}")
    if not mask_dir.exists():
        raise RuntimeError(f"MSD mask directory not found: {mask_dir}")

    all_results = {}

    for mask_mode in MASK_MODES_TO_RUN:
        ious, precs, recs, dices = [], [], [], []
        per_image = []
        missing_masks = 0
        processed = 0

        for img_path in images:
            mask_path = find_mask_for_image(mask_dir, img_path.stem)
            if mask_path is None:
                missing_masks += 1
                continue

            img = Image.open(img_path).convert("RGB")
            gt_mask = read_binary_mask(mask_path)
            seg_debug = segment_with_debug(seg_model, img, mask_mode)
            pred_mask = choose_raw_mask(seg_debug, mask_mode)
            pred_mask = resize_mask_to_match(pred_mask, gt_mask.shape[:2])

            if REMOVE_SMALL_COMPONENTS:
                pred_mask = remove_small_components(pred_mask, MIN_COMPONENT_AREA)
            if FILL_MASK_HOLES:
                pred_mask = fill_mask_holes(pred_mask)
            if USE_ERODED_PRED_MASK:
                pred_mask = erode_mask(pred_mask, PRED_ERODE_KERNEL, PRED_ERODE_ITERS)

            m = binary_mask_metrics(pred_mask, gt_mask)
            ious.append(m["iou"])
            precs.append(m["precision"])
            recs.append(m["recall"])
            dices.append(m["dice"])

            processed += 1
            per_image.append({
                "stem": img_path.stem,
                **m,
                "mirror_prob_max": seg_debug["stats"]["mirror_prob_max"],
                "mirror_prob_mean": seg_debug["stats"]["mirror_prob_mean"],
            })

        all_results[mask_mode] = {
            "processed": processed,
            "missing_masks": missing_masks,
            "mean_iou": safe_mean(ious),
            "mean_precision": safe_mean(precs),
            "mean_recall": safe_mean(recs),
            "mean_dice": safe_mean(dices),
            "per_image": per_image,
        }

    with open(MSD_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] RUN_PAIR_EVAL={RUN_PAIR_EVAL}")
    print(f"[INFO] RUN_MSD_EVAL={RUN_MSD_EVAL}")
    print(f"[INFO] MASK_MODES_TO_RUN={MASK_MODES_TO_RUN}")
    print(f"[INFO] HEADS_TO_RUN={HEADS_TO_RUN}")

    if not RUN_PAIR_EVAL and not RUN_MSD_EVAL:
        raise RuntimeError("Both RUN_PAIR_EVAL and RUN_MSD_EVAL are False. Nothing to do.")

    seg_model = load_segmentor()
    da3_model = load_da3() if RUN_PAIR_EVAL else None
    heads = build_heads()

    pair_summary = None
    aggregate = None
    msd_summary = None

    if RUN_PAIR_EVAL:
        pair_summary, aggregate = evaluate_pairs(seg_model, da3_model, heads)

    if RUN_MSD_EVAL:
        msd_summary = evaluate_segmentor_on_msd(seg_model)

    print("\n========== DONE ==========")
    if aggregate is not None:
        print("\nLeaderboard (sorted by used-mask delta MAE):")
        print(json.dumps(aggregate["leaderboard"], indent=2))
    if msd_summary is not None:
        print("\nMSD segmentor summary:")
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in msd_summary.items()}, indent=2))


if __name__ == "__main__":
    main()