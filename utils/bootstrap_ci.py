# utils/bootstrap_ci.py
#
# New standalone script for AIBThings 2026 reviewer response.
# Author: Amir — Bachelor Thesis follow-up, GUC 2025
#
# Computes bootstrap confidence intervals for detection, intention, and
# trajectory metrics from the per-sample results already saved by eval.py
# (via the save_results_path config option). No GPU, no model, no
# re-inference — pure CPU resampling of results already collected.
#
# Directly (partially) answers R5's comment:
#   "All models are trained only once with a single random seed, although
#    several reported improvements are very small... Multiple independent
#    runs, confidence intervals, and statistical significance tests are
#    required before attributing these differences to the proposed
#    trajectory supervision."
#
# IMPORTANT SCOPE CAVEAT (must be stated explicitly in the paper):
#   This bootstrap resamples VALIDATION SEQUENCES from a SINGLE trained
#   model's fixed predictions. It quantifies within-run sampling
#   uncertainty — "how much would this metric plausibly vary if the
#   validation set had been drawn slightly differently" — NOT cross-seed
#   training variance ("would a different training run give a different
#   model"). It is a partial, honest substitute for multi-seed training,
#   not a replacement for it. Confidence intervals from this method can
#   be narrower than the true uncertainty a multi-seed study would reveal,
#   since they don't capture variance from random initialization,
#   data shuffling order, or other stochastic elements of training itself.
#
# Method (standard nonparametric bootstrap, e.g. Efron & Tibshirani 1993):
#   1. Load the N per-sample results (one entry per validation sequence)
#      for a given model version.
#   2. Resample N sequences WITH REPLACEMENT from this pool.
#   3. Recompute each metric (mAP, accuracy, F1, minADE, etc.) on the
#      resampled pool, using the EXACT SAME metric functions as eval.py
#      (compute_detection_ap, compute_intention_metrics,
#      accumulate_trajectory_metrics), so results are directly comparable
#      to the numbers already reported.
#   4. Repeat B times (default 1000).
#   5. Report the 2.5th and 97.5th percentiles across the B resampled
#      values as a 95% confidence interval, plus the mean and std.
#
# Usage:
#   PYTHONPATH=/content/MultiTask-ViT-AV2 python utils/bootstrap_ci.py \
#       --results v1_results.pt v2_results.pt v3_single_results.pt \
#       --results_dir /content/drive/MyDrive/Amir_Dataset/EvalResults \
#       --n_bootstrap 1000 \
#       --output_json /content/drive/MyDrive/Amir_Dataset/EvalResults/bootstrap_ci.json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import numpy as np
import torch

from utils.metrics import (
    compute_detection_ap,
    compute_intention_metrics,
    accumulate_trajectory_metrics,
)
from utils.utils import compute_axis_aligned_iou

try:
    from utils.utils import compute_rotated_iou
    ROTATED_IOU_AVAILABLE = True
except ImportError:
    compute_rotated_iou = None
    ROTATED_IOU_AVAILABLE = False


# =============================================================================
# resample_and_compute
#
# Draws one bootstrap resample of the sample-level results (and, in
# lockstep, the correspondingly-indexed cached IoU matrices and trajectory
# metric dicts) and recomputes every metric on it.
# =============================================================================

def resample_and_compute(
    all_sample_results: list,
    all_cached_iou_matrices: list,
    all_traj_metric_results: list,
    e2e_traj_raw_results: list,
    use_rotated_iou: bool,
    rng: random.Random,
) -> dict:
    N = len(all_sample_results)
    if N == 0:
        return {}

    # Resample sample-level results (detection + intention) with replacement
    idx = [rng.randrange(N) for _ in range(N)]
    resampled_samples = [all_sample_results[i] for i in idx]
    resampled_iou = (
        [all_cached_iou_matrices[i] for i in idx]
        if all_cached_iou_matrices else None
    )

    iou_func = compute_rotated_iou if (use_rotated_iou and ROTATED_IOU_AVAILABLE) else compute_axis_aligned_iou

    det_metrics = compute_detection_ap(
        resampled_samples, iou_func, use_rotated_iou,
        precomputed_iou_matrices=resampled_iou,
    )
    intent_metrics = compute_intention_metrics(
        resampled_samples, iou_func, use_rotated_iou,
        precomputed_iou_matrices=resampled_iou,
    )

    out = {**det_metrics}
    out['intention_accuracy']     = intent_metrics.get('intention_accuracy', 0.0)
    out['intention_f1_macro']     = intent_metrics.get('intention_f1_macro', 0.0)
    out['intention_f1_weighted']  = intent_metrics.get('intention_f1_weighted', 0.0)

    # Oracle trajectory — independent pool (one entry per BATCH, not per
    # sequence, since the oracle path only evaluates gt_list[0] per batch —
    # see eval.py). Resample this pool separately at its own size.
    if all_traj_metric_results:
        M = len(all_traj_metric_results)
        traj_idx = [rng.randrange(M) for _ in range(M)]
        resampled_traj = [all_traj_metric_results[i] for i in traj_idx]
        traj_agg = accumulate_trajectory_metrics(resampled_traj)
        out['oracle_minADE'] = traj_agg['minADE']
        out['oracle_minFDE'] = traj_agg['minFDE']
        out['oracle_MR']     = traj_agg['MR']

    # End-to-end trajectory — also its own independent pool (one entry per
    # matched-scene result, see evaluate_trajectory_end_to_end in eval.py).
    if e2e_traj_raw_results:
        K = len(e2e_traj_raw_results)
        e2e_idx = [rng.randrange(K) for _ in range(K)]
        resampled_e2e = [e2e_traj_raw_results[i] for i in e2e_idx]
        e2e_agg = accumulate_trajectory_metrics(resampled_e2e)
        out['e2e_minADE'] = e2e_agg['minADE']
        out['e2e_minFDE'] = e2e_agg['minFDE']
        out['e2e_MR']     = e2e_agg['MR']

    return out


# =============================================================================
# compute_ci_for_file
# =============================================================================

def compute_ci_for_file(
    filepath: Path,
    n_bootstrap: int,
    seed: int = 42,
) -> dict:
    print(f"\nLoading: {filepath}")
    data = torch.load(filepath, map_location='cpu', weights_only=False)

    model_version           = data.get('model_version', filepath.stem)
    all_sample_results      = data.get('all_sample_results', [])
    all_cached_iou_matrices = data.get('all_cached_iou_matrices', [])
    all_traj_metric_results = data.get('all_traj_metric_results', [])
    e2e_traj_raw_results    = data.get('e2e_traj_raw_results', [])
    aggregated_metrics      = data.get('aggregated_metrics', {})

    # Infer whether this run used rotated IoU from the presence of a
    # 'distance_binned_mAP' entry is not reliable; instead check if any
    # cached IoU matrix exists and assume the same rotated/axis-aligned
    # setting was used consistently for the whole run. Since we cannot
    # recover the exact flag from the saved file, we default to
    # axis-aligned recomputation UNLESS overridden via --rotated below,
    # matching whichever mode the caller specifies per file.
    N = len(all_sample_results)
    M = len(all_traj_metric_results)
    K = len(e2e_traj_raw_results)
    print(f"  Model version: {model_version}")
    print(f"  N samples (det/intent): {N}")
    print(f"  M oracle traj batches:  {M}")
    print(f"  K end-to-end traj entries: {K}")

    if N == 0:
        print("  WARNING: no sample results found in this file — skipping.")
        return {}

    return {
        'model_version': model_version,
        'all_sample_results': all_sample_results,
        'all_cached_iou_matrices': all_cached_iou_matrices,
        'all_traj_metric_results': all_traj_metric_results,
        'e2e_traj_raw_results': e2e_traj_raw_results,
        'point_estimate': aggregated_metrics,
    }


def bootstrap_metrics(
    loaded: dict,
    n_bootstrap: int,
    use_rotated_iou: bool,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)

    bootstrap_samples = []
    for b in range(n_bootstrap):
        result = resample_and_compute(
            loaded['all_sample_results'],
            loaded['all_cached_iou_matrices'],
            loaded['all_traj_metric_results'],
            loaded['e2e_traj_raw_results'],
            use_rotated_iou,
            rng,
        )
        bootstrap_samples.append(result)
        if (b + 1) % 100 == 0:
            print(f"    Bootstrap {b + 1}/{n_bootstrap}")

    # Collect all metric keys seen across bootstrap samples
    all_keys = set()
    for r in bootstrap_samples:
        all_keys.update(r.keys())

    ci_results = {}
    for key in sorted(all_keys):
        values = [r[key] for r in bootstrap_samples if key in r]
        if not values:
            continue
        values = np.array(values)
        ci_results[key] = {
            'point_estimate': loaded['point_estimate'].get(key, float(np.mean(values))),
            'bootstrap_mean': float(np.mean(values)),
            'bootstrap_std':  float(np.std(values)),
            'ci_lower_2.5%':  float(np.percentile(values, 2.5)),
            'ci_upper_97.5%': float(np.percentile(values, 97.5)),
        }

    return ci_results


def print_ci_table(model_version: str, ci_results: dict) -> None:
    print(f"\n{'='*80}")
    print(f"  Bootstrap 95% Confidence Intervals — {model_version}")
    print(f"{'='*80}")
    print(f"  {'Metric':<22} {'Point est.':>12} {'Boot mean':>12} "
          f"{'95% CI':>22}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*22}")
    for key, stats in ci_results.items():
        ci_str = f"[{stats['ci_lower_2.5%']:.4f}, {stats['ci_upper_97.5%']:.4f}]"
        print(f"  {key:<22} {stats['point_estimate']:>12.4f} "
              f"{stats['bootstrap_mean']:>12.4f} {ci_str:>22}")
    print()


def check_overlap(name_a, ci_a, name_b, ci_b, key):
    if key not in ci_a or key not in ci_b:
        return None
    lo_a, hi_a = ci_a[key]['ci_lower_2.5%'], ci_a[key]['ci_upper_97.5%']
    lo_b, hi_b = ci_b[key]['ci_lower_2.5%'], ci_b[key]['ci_upper_97.5%']
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    return {
        'metric': key,
        f'{name_a}_ci': [lo_a, hi_a],
        f'{name_b}_ci': [lo_b, hi_b],
        'overlap': overlap,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals from saved eval results."
    )
    parser.add_argument(
        "--results", nargs='+', required=True,
        help="Filenames (relative to --results_dir) of saved eval .pt files."
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory containing the saved eval .pt files."
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples. Default: 1000."
    )
    parser.add_argument(
        "--rotated_iou", action='store_true',
        help="Recompute detection/intention metrics using rotated IoU "
             "(must match how the original eval run was configured)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible bootstrap resampling."
    )
    parser.add_argument(
        "--output_json", type=str, default="",
        help="Optional path to save all CI results as JSON."
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_ci_results = {}

    for fname in args.results:
        filepath = results_dir / fname
        if not filepath.is_file():
            print(f"WARNING: file not found, skipping: {filepath}")
            continue

        loaded = compute_ci_for_file(filepath, args.n_bootstrap, args.seed)
        if not loaded:
            continue

        print(f"  Running {args.n_bootstrap} bootstrap resamples "
              f"(rotated_iou={args.rotated_iou})...")
        ci_results = bootstrap_metrics(
            loaded, args.n_bootstrap, args.rotated_iou, args.seed
        )
        print_ci_table(loaded['model_version'], ci_results)

        all_ci_results[loaded['model_version']] = ci_results

    # --- Overlap check between consecutive versions (V1 vs V2, V2 vs V3) ---
    # Directly useful for the paper: "does the CI for this improvement
    # overlap zero difference" style reasoning made explicit per metric.
    versions = list(all_ci_results.keys())
    if len(versions) >= 2:
        print(f"\n{'='*80}")
        print(f"  CI Overlap Analysis (non-overlapping CIs suggest a")
        print(f"  more robust difference; overlapping CIs suggest the")
        print(f"  difference may not exceed sampling noise)")
        print(f"{'='*80}\n")

        overlap_results = []
        for i in range(len(versions) - 1):
            v_a, v_b = versions[i], versions[i + 1]
            ci_a, ci_b = all_ci_results[v_a], all_ci_results[v_b]
            common_keys = set(ci_a.keys()) & set(ci_b.keys())
            for key in sorted(common_keys):
                res = check_overlap(v_a, ci_a, v_b, ci_b, key)
                if res:
                    overlap_results.append(res)
                    flag = "OVERLAP" if res['overlap'] else "no overlap"
                    print(f"  {v_a} vs {v_b} — {key:<22}: {flag}")

        all_ci_results['_overlap_analysis'] = overlap_results

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_ci_results, f, indent=2)
        print(f"\nSaved all CI results to: {out_path}")