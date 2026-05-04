#!/usr/bin/env python3
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_JSON = Path("eval_outputs_pairs_modular/pair_eval_summary_all_combos.json")
DETAIL_JSON = Path("eval_outputs_pairs_modular/pair_depth_metrics_detailed.json")
OUT_DIR = Path("eval_outputs_pairs_modular/plots")


def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def combo_label(combo: str) -> str:
    return (
        combo.replace("base_threshold", "base_th")
        .replace("plane_fit", "plane")
        .replace("__", "\n")
    )


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_leaderboard(summary):
    leaderboard = summary["leaderboard"]

    labels = [combo_label(x["combo"]) for x in leaderboard]
    used_add = [safe_float(x["used_delta_absolute_depth_deviation"]) for x in leaderboard]
    full_add = [safe_float(x["full_delta_absolute_depth_deviation"]) for x in leaderboard]
    used_rmse = [safe_float(x["used_delta_rmse"]) for x in leaderboard]

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x, used_add)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Used-region Mean Average Error improvement")
    plt.title("Mirror Recovery: Used-region Mean Average Error Improvement")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "leaderboard_used_add.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    width = 0.38
    plt.bar(x - width / 2, full_add, width, label="Full image")
    plt.bar(x + width / 2, used_add, width, label="Used mask")
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Mean Average Error improvement")
    plt.title("Full-image vs Used-mask Mean Average Error Improvement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "full_vs_used_add.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, used_rmse)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Used-region RMSE improvement")
    plt.title("Mirror Recovery: Used-region RMSE Improvement")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "leaderboard_used_rmse.png", dpi=200)
    plt.close()


def extract_detail_rows(detail):
    rows = []

    for _, item in detail.items():
        pair_id = item.get("pair_id")
        mask = item.get("mask_mode_display")
        method = item.get("method_display")
        combo = f"{mask}__{method}"

        regions = item.get("regions", {})
        used = regions.get("used_mirror_mask", {})
        changed = regions.get("changed_region", {})

        raw_add = safe_float(used.get("raw_absolute_depth_deviation"))
        fix_add = safe_float(used.get("fixed_absolute_depth_deviation"))
        delta_add = safe_float(used.get("delta_absolute_depth_deviation"))

        raw_rmse = safe_float(used.get("raw_rmse"))
        fix_rmse = safe_float(used.get("fixed_rmse"))
        delta_rmse = safe_float(used.get("delta_rmse"))

        used_pixels = safe_float(item.get("used_pred_pixels"))
        changed_pixels = safe_float(item.get("changed_pixels"))

        if used_pixels and used_pixels > 0:
            changed_ratio = changed_pixels / used_pixels
        else:
            changed_ratio = np.nan

        if not math.isnan(raw_add) and raw_add > 1e-8:
            rel_improve = delta_add / raw_add * 100.0
        else:
            rel_improve = np.nan

        rows.append({
            "pair_id": pair_id,
            "combo": combo,
            "mask": mask,
            "method": method,
            "applied": bool(item.get("applied")),
            "raw_add": raw_add,
            "fix_add": fix_add,
            "delta_add": delta_add,
            "raw_rmse": raw_rmse,
            "fix_rmse": fix_rmse,
            "delta_rmse": delta_rmse,
            "used_pixels": used_pixels,
            "changed_pixels": changed_pixels,
            "changed_ratio": changed_ratio,
            "relative_improvement_pct": rel_improve,
        })

    return rows


def group_by_combo(rows):
    combos = sorted(set(r["combo"] for r in rows))
    out = {}

    for combo in combos:
        rs = [r for r in rows if r["combo"] == combo]

        applied_rows = [
            r for r in rs
            if r["applied"] and np.isfinite(r["delta_add"])
        ]

        success_rows = [
            r for r in applied_rows
            if r["delta_add"] > 0
        ]

        all_delta = [r["delta_add"] for r in applied_rows if np.isfinite(r["delta_add"])]
        success_delta = [r["delta_add"] for r in success_rows if np.isfinite(r["delta_add"])]

        all_rel = [
            r["relative_improvement_pct"]
            for r in applied_rows
            if np.isfinite(r["relative_improvement_pct"])
        ]
        success_rel = [
            r["relative_improvement_pct"]
            for r in success_rows
            if np.isfinite(r["relative_improvement_pct"])
        ]

        all_changed = [
            r["changed_ratio"]
            for r in applied_rows
            if np.isfinite(r["changed_ratio"])
        ]
        success_changed = [
            r["changed_ratio"]
            for r in success_rows
            if np.isfinite(r["changed_ratio"])
        ]

        applied_count = len(applied_rows)
        success_count = len(success_rows)

        out[combo] = {
            "n_total": len(rs),
            "n_applied": applied_count,
            "n_success": success_count,

            "activation_rate": applied_count / len(rs) if rs else np.nan,
            "success_rate_all": success_count / len(rs) if rs else np.nan,
            "success_rate_applied": success_count / applied_count if applied_count > 0 else np.nan,

            # all applied rows
            "mean_delta_add_all_applied": float(np.mean(all_delta)) if all_delta else np.nan,
            "mean_relative_improvement_pct_all_applied": float(np.mean(all_rel)) if all_rel else np.nan,
            "mean_changed_ratio_all_applied": float(np.mean(all_changed)) if all_changed else np.nan,

            # success-only rows: use these for main ADD/improvement plots
            "mean_delta_add_success_only": float(np.mean(success_delta)) if success_delta else np.nan,
            "mean_relative_improvement_pct_success_only": float(np.mean(success_rel)) if success_rel else np.nan,
            "mean_changed_ratio_success_only": float(np.mean(success_changed)) if success_changed else np.nan,
        }

    return out

def plot_group_stats(grouped):
    combos = list(grouped.keys())
    labels = [combo_label(c) for c in combos]

    mean_rel = [grouped[c]["mean_relative_improvement_pct_success_only"] for c in combos]
    activation = [grouped[c]["activation_rate"] * 100 for c in combos]
    success = [grouped[c]["success_rate_applied"] * 100 for c in combos]
    changed = [grouped[c]["mean_changed_ratio_success_only"] * 100 for c in combos]

    x = np.arange(len(combos))

    plt.figure(figsize=(10, 5))
    plt.bar(x, mean_rel)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Mean relative Mean Average Error improvement (%)")
    plt.title("Mean Relative Improvement by Method (Successful Cases Only)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_relative_improvement_pct.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    width = 0.38
    plt.bar(x - width / 2, activation, width, label="Applied")
    plt.bar(x + width / 2, success, width, label="Improved")
    plt.xticks(x, labels)
    plt.ylabel("Rate (%)")
    plt.title("Activation Rate vs Success Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "activation_vs_success_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, changed)
    for i, v in enumerate(changed):
        if np.isfinite(v):
            plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    # Zoom y-axis around actual data range
    valid_changed = [v for v in changed if np.isfinite(v)]

    if valid_changed:
        ymin = min(valid_changed)
        ymax = max(valid_changed)

        margin = max(1.0, (ymax - ymin) * 0.15)

        plt.ylim(
            max(0, ymin - margin),
            min(100, ymax + margin),
        )

    plt.xticks(x, labels)
    plt.ylabel("Changed pixels / used mask (%)")
    plt.title("Changed-pixel Coverage (Successful Cases Only)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "changed_pixel_coverage.png", dpi=200)
    plt.close()


def plot_per_pair_heatmap(rows):
    combos = sorted(set(r["combo"] for r in rows))
    pairs = sorted(set(r["pair_id"] for r in rows))

    mat = np.full((len(pairs), len(combos)), np.nan)

    pair_to_i = {p: i for i, p in enumerate(pairs)}
    combo_to_j = {c: j for j, c in enumerate(combos)}

    for r in rows:
        i = pair_to_i[r["pair_id"]]
        j = combo_to_j[r["combo"]]
        mat[i, j] = r["delta_add"]

    plt.figure(figsize=(max(8, len(combos) * 1.3), max(5, len(pairs) * 0.45)))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="Used-region Mean Average Error improvement")
    plt.xticks(np.arange(len(combos)), [combo_label(c) for c in combos], rotation=0)
    plt.yticks(np.arange(len(pairs)), pairs)
    plt.xlabel("Method")
    plt.ylabel("Pair ID")
    plt.title("Per-pair Used-region Mean Average Error Improvement")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_pair_add_heatmap.png", dpi=200)
    plt.close()


def save_group_table(grouped):
    out_path = OUT_DIR / "group_summary.csv"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "combo,n_total,n_applied,n_success,activation_rate,"
            "success_rate_all,success_rate_applied,"
            "mean_delta_add_all_applied,mean_delta_add_success_only,"
            "mean_relative_improvement_pct_all_applied,"
            "mean_relative_improvement_pct_success_only,"
            "mean_changed_ratio_all_applied,"
            "mean_changed_ratio_success_only\n"
        )

        for combo, g in grouped.items():
            f.write(
                f"{combo},{g['n_total']},{g['n_applied']},{g['n_success']},"
                f"{g['activation_rate']},{g['success_rate_all']},"
                f"{g['success_rate_applied']},"
                f"{g['mean_delta_add_all_applied']},"
                f"{g['mean_delta_add_success_only']},"
                f"{g['mean_relative_improvement_pct_all_applied']},"
                f"{g['mean_relative_improvement_pct_success_only']},"
                f"{g['mean_changed_ratio_all_applied']},"
                f"{g['mean_changed_ratio_success_only']}\n"
            )

    print(f"Saved table: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_json(SUMMARY_JSON)
    detail = load_json(DETAIL_JSON)

    rows = extract_detail_rows(detail)
    grouped = group_by_combo(rows)

    plot_success_only_leaderboard(rows)

    plot_group_stats(grouped)
    plot_per_pair_heatmap(rows)
    save_group_table(grouped)

    print(f"Saved plots to: {OUT_DIR}")
    print("Generated:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()