# utils/label_stats.py
#
# New standalone script for AIBThings 2026 reviewer response.
# Author: Amir — Bachelor Thesis follow-up, GUC 2025
#
# Computes intention class distribution statistics directly from the
# precomputed annotations_with_intent.feather files, with no model or
# GPU involved. Answers R5's request to "quantify heuristic-label
# reliability, provide class distributions" — this script provides the
# distribution half; the confusion matrix (already in eval.py) provides
# a complementary view of reliability via matched TP predictions.
#
# Usage:
#   PYTHONPATH=/content/MultiTask-ViT-AV2 python utils/label_stats.py \
#       --data_root /content/local_data --splits train val

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import pandas as pd
import numpy as np

from utils.constants import INTENTIONS_MAP_REV, VEHICLE_CATEGORIES
from datasets.av2_dataset import ScenarioValidator


def compute_split_label_stats(data_root_dir: str, split_name: str) -> dict:
    """
    Scans every log in a split, reads its annotations_with_intent.feather,
    and tabulates intention class counts across all vehicle annotations
    (not just one per track — every timestamp-level box counts once,
    matching what the intention head is actually trained/evaluated on).

    Returns:
        dict with per-class counts, percentages, total, and per-log
        vehicle counts (useful for sanity-checking dataset balance).
    """
    split_dir = Path(data_root_dir) / split_name
    if not split_dir.is_dir():
        print(f"  WARNING: split dir not found: {split_dir}")
        return {}

    validator = ScenarioValidator(str(split_dir), skip_known_corrupted=False)
    valid_scenarios = validator.find_valid_scenarios()

    if not valid_scenarios:
        print(f"  WARNING: no valid scenarios found in {split_dir}")
        return {}

    print(f"  Found {len(valid_scenarios)} logs in '{split_name}' split.")

    class_counts = {i: 0 for i in range(8)}
    total_boxes = 0
    missing_files = 0
    per_log_counts = {}

    for scenario_info in valid_scenarios:
        log_dir = Path(scenario_info.log_dir)
        log_id = log_dir.name
        intent_file = log_dir / "annotations_with_intent.feather"

        if not intent_file.is_file():
            missing_files += 1
            continue

        df = pd.read_feather(intent_file)

        # Match exactly what prepare_gt_for_frame() filters on:
        # vehicle categories only, valid heuristic label (!= -1)
        df_valid = df[
            (df['category'].isin(VEHICLE_CATEGORIES)) &
            (df['heuristic_intent'] != -1)
        ]

        log_total = len(df_valid)
        per_log_counts[log_id] = log_total
        total_boxes += log_total

        vc = df_valid['heuristic_intent'].value_counts()
        for cls_idx, count in vc.items():
            cls_idx = int(cls_idx)
            if cls_idx in class_counts:
                class_counts[cls_idx] += int(count)

    if missing_files > 0:
        print(f"  WARNING: {missing_files} logs missing "
              f"annotations_with_intent.feather (skipped).")

    if total_boxes == 0:
        print(f"  WARNING: no valid annotated boxes found in '{split_name}'.")
        return {}

    class_stats = {}
    for cls_idx in range(8):
        cls_name = INTENTIONS_MAP_REV.get(cls_idx, f"Class_{cls_idx}")
        count = class_counts[cls_idx]
        pct = 100.0 * count / total_boxes
        class_stats[cls_name] = {
            "count": count,
            "percentage": round(pct, 2),
        }

    per_log_values = list(per_log_counts.values())

    return {
        "split": split_name,
        "num_logs": len(valid_scenarios),
        "num_logs_missing_intent_file": missing_files,
        "total_annotated_vehicle_boxes": total_boxes,
        "class_distribution": class_stats,
        "per_log_box_count": {
            "mean": round(float(np.mean(per_log_values)), 1) if per_log_values else 0,
            "std": round(float(np.std(per_log_values)), 1) if per_log_values else 0,
            "min": int(np.min(per_log_values)) if per_log_values else 0,
            "max": int(np.max(per_log_values)) if per_log_values else 0,
        },
    }


def print_split_stats(stats: dict) -> None:
    if not stats:
        return

    print(f"\n{'='*60}")
    print(f"  Intention Label Distribution — {stats['split']} split")
    print(f"{'='*60}")
    print(f"  Logs:                    {stats['num_logs']}")
    if stats['num_logs_missing_intent_file'] > 0:
        print(f"  Logs missing intent file: {stats['num_logs_missing_intent_file']}")
    print(f"  Total annotated boxes:   {stats['total_annotated_vehicle_boxes']}")
    print(f"\n  Class distribution:")
    print(f"  {'Class':<22} {'Count':>10} {'%':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*8}")

    # Sort by count descending — makes imbalance immediately visible
    sorted_classes = sorted(
        stats['class_distribution'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    for cls_name, cls_stats in sorted_classes:
        print(f"  {cls_name:<22} {cls_stats['count']:>10} {cls_stats['percentage']:>7.2f}%")

    plc = stats['per_log_box_count']
    print(f"\n  Per-log box count: mean={plc['mean']}, std={plc['std']}, "
          f"min={plc['min']}, max={plc['max']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute intention label class distribution stats."
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory containing train/ and val/ split folders."
    )
    parser.add_argument(
        "--splits", nargs='+', default=["val"],
        help="Splits to analyze. Default: val."
    )
    parser.add_argument(
        "--output_json", type=str, default="",
        help="Optional path to save combined stats as JSON."
    )
    args = parser.parse_args()

    all_stats = {}
    for split_name in args.splits:
        print(f"\nProcessing split: {split_name}")
        stats = compute_split_label_stats(args.data_root, split_name)
        if stats:
            print_split_stats(stats)
            all_stats[split_name] = stats

    if args.output_json and all_stats:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nSaved combined stats to: {out_path}")