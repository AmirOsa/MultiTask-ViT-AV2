# utils/metrics.py
#
# New file for IntentTrajNet-AV2
# Author: [Your Name] — Bachelor Thesis, GUC 2025
#
# Centralises all evaluation metrics for all three models:
#   Detection:   mAP at multiple IoU thresholds (from Nadeem's eval_vit.py)
#   Intention:   Accuracy, F1 macro, F1 weighted, F1 per class
#   Trajectory:  minADE, minFDE, Miss Rate
#
# ## NEW — CHANGES FOR AIBThings 2026 REVISION (runtime optimization):
#   compute_detection_ap() and compute_intention_metrics() now accept an
#   optional `precomputed_iou_matrices` argument (default: None). When
#   provided, the pred-vs-GT IoU matrix for each sample is looked up
#   instead of being recomputed via iou_func(). This does NOT change any
#   computed value — it eliminates redundant recomputation of the same
#   IoU matrix, which previously happened independently inside
#   compute_detection_ap (once per IoU threshold, ×5), compute_intention_metrics,
#   compute_distance_binned_map (in eval.py), and the confusion-matrix
#   block (in eval.py). When precomputed_iou_matrices is None (the
#   default, used by any other caller), behavior is byte-for-byte
#   identical to before this change.
#
#   A secondary fix within compute_detection_ap(): the IoU matrix is now
#   computed at most once per sample even in the no-cache fallback path
#   (previously it was recomputed once per IoU threshold — 5× redundant
#   — despite the matrix not depending on the threshold at all, only the
#   `>= iou_t` comparison does). This is a pure redundancy removal and
#   does not change results.
#
# Trajectory metrics:
#   SOURCED: minADE, minFDE, MR definitions from Abdulbaki thesis Section 3.7
#   and AV2 benchmark specification (Wilson et al., 2023).
#
#   minADE: minimum Average Displacement Error
#       For each vehicle, find the predicted mode closest to GT.
#       Compute the average L2 distance between that mode and GT
#       across all timesteps. Average over all vehicles.
#       SOURCED: Abdulbaki thesis Table 3.1 definition.
#
#   minFDE: minimum Final Displacement Error
#       For each vehicle, find the predicted mode closest to GT at
#       the FINAL timestep. Compute L2 distance at that final step.
#       Average over all vehicles.
#       SOURCED: Abdulbaki thesis Table 3.1 definition.
#
#   MR: Miss Rate
#       Fraction of vehicles where the best predicted mode's final
#       position is more than 2.0 metres from the GT final position.
#       SOURCED: Abdulbaki thesis Table 3.1 — "distance between the
#       ground-truth endpoint and the best-predicted endpoint exceeds
#       2.0 meters."

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from utils.constants import (
    DETECTION_IOU_THRESHOLDS,
    IOU_THRESHOLD_FOR_INTENTION_MATCH,
    NUM_INTENTION_CLASSES,
    INTENTIONS_MAP_REV,
    TRAJECTORY_NUM_MODES,
    TRAJECTORY_FUTURE_STEPS,
)
from utils.utils import calculate_ap, compute_axis_aligned_iou

# Miss rate threshold in metres
# SOURCED: Abdulbaki thesis Section 3.7 — 2.0 metres
MISS_RATE_THRESHOLD_M = 2.0


# =============================================================================
# Trajectory metrics — UNCHANGED
# =============================================================================

def compute_trajectory_metrics(
    y_hat: torch.Tensor,
    pi: torch.Tensor,
    gt_traj: torch.Tensor,
    gt_mask: torch.Tensor,
) -> dict:
    """
    Compute minADE, minFDE, and Miss Rate for trajectory predictions.

    All three metrics follow the "best-of-K" (K=6) convention —
    for each vehicle we find the predicted mode closest to GT and
    evaluate only that mode. This is the standard evaluation protocol
    for multi-modal trajectory prediction.

    SOURCED: metric definitions from Abdulbaki thesis Section 3.7
    and AV2 benchmark specification.

    Args:
        y_hat:   [F, N, H, 4]  — trajectory predictions (µx,µy,bx,by)
        pi:      [N, F]         — mode logits (not used for metrics,
                                  metrics use best-of-K not argmax mode)
        gt_traj: [N, H, 2]     — GT future positions (x, y) in ego frame
        gt_mask: [N, H]         — True where GT is valid

    Returns:
        dict with:
            minADE:    float — minimum Average Displacement Error (metres)
            minFDE:    float — minimum Final Displacement Error (metres)
            MR:        float — Miss Rate (fraction of vehicles missed)
            N_vehicles: int — number of vehicles evaluated
    """
    device = y_hat.device
    N = y_hat.shape[1]
    num_modes  = y_hat.shape[0]
    H = y_hat.shape[2]

    if N == 0:
        return {
            "minADE": 0.0,
            "minFDE": 0.0,
            "MR": 0.0,
            "N_vehicles": 0
        }

    # Extract predicted positions (µx, µy) only — ignore scale
    pred_pos = y_hat[..., :2]
    # [F, N, H, 2]

    # Expand GT for comparison across all F modes
    gt_expanded = gt_traj.unsqueeze(0).expand(num_modes, N, H, 2)
    # [F, N, H, 2]

    mask_expanded = gt_mask.unsqueeze(0).expand(num_modes, N, H).float()
    # [F, N, H]

    # L2 distance at each timestep per mode per vehicle
    l2_per_step = torch.norm(pred_pos - gt_expanded, dim=-1)
    # [F, N, H]

    # Zero out padded timesteps
    l2_per_step_masked = l2_per_step * mask_expanded
    # [F, N, H]

    # --- minADE ---
    # For each vehicle, compute mean L2 per mode over valid timesteps
    # then take the minimum across modes
    # SOURCED: Abdulbaki thesis Table 3.1
    valid_counts = gt_mask.float().sum(dim=-1).clamp(min=1.0)
    # [N] — number of valid timesteps per vehicle

    # Sum L2 over timesteps per mode per vehicle
    l2_sum_per_mode = l2_per_step_masked.sum(dim=-1)
    # [F, N]

    # Average over valid timesteps
    ade_per_mode = l2_sum_per_mode / valid_counts.unsqueeze(0)
    # [F, N]

    # Take minimum across modes per vehicle
    min_ade_per_vehicle, best_mode_ade = ade_per_mode.min(dim=0)
    # [N]

    minADE = min_ade_per_vehicle.mean().item()

    # --- minFDE ---
    # For each vehicle, find the last valid timestep
    # L2 distance at that timestep for the best mode
    # SOURCED: Abdulbaki thesis Table 3.1
    #
    # Find last valid timestep index per vehicle
    last_valid_idx = (gt_mask.float() * torch.arange(
        H, device=device, dtype=torch.float32
    ).unsqueeze(0)).argmax(dim=1)
    # [N] — index of last valid timestep

    # Gather L2 at last valid timestep for each mode
    last_idx_expanded = last_valid_idx.view(1, N, 1).expand(num_modes, N, 1)
    l2_at_last = l2_per_step.gather(2, last_idx_expanded).squeeze(-1)
    # [num_modes, N]

    # Minimum across modes
    min_fde_per_vehicle, best_mode_fde = l2_at_last.min(dim=0)
    # [N]

    minFDE = min_fde_per_vehicle.mean().item()

    # --- Miss Rate ---
    # Fraction of vehicles where best FDE > 2.0 metres
    # SOURCED: Abdulbaki thesis Section 3.7 — threshold = 2.0 metres
    missed = (min_fde_per_vehicle > MISS_RATE_THRESHOLD_M).float()
    MR = missed.mean().item()

    return {
        "minADE": minADE,
        "minFDE": minFDE,
        "MR": MR,
        "N_vehicles": N
    }


def accumulate_trajectory_metrics(
    results_list: list,
) -> dict:
    """
    Accumulate trajectory metrics over multiple batches/scenes.

    Args:
        results_list: list of dicts from compute_trajectory_metrics()

    Returns:
        dict with weighted-average minADE, minFDE, MR
    """
    if not results_list:
        return {"minADE": 0.0, "minFDE": 0.0, "MR": 0.0, "N_vehicles": 0}

    total_N = sum(r["N_vehicles"] for r in results_list)
    if total_N == 0:
        return {"minADE": 0.0, "minFDE": 0.0, "MR": 0.0, "N_vehicles": 0}

    # Weighted average by number of vehicles
    minADE = sum(r["minADE"] * r["N_vehicles"] for r in results_list) / total_N
    minFDE = sum(r["minFDE"] * r["N_vehicles"] for r in results_list) / total_N
    MR = sum(r["MR"] * r["N_vehicles"] for r in results_list) / total_N

    return {
        "minADE": minADE,
        "minFDE": minFDE,
        "MR": MR,
        "N_vehicles": total_N
    }


# =============================================================================
# Detection metrics
# ## CHANGED — added optional precomputed_iou_matrices param + removed
# redundant per-threshold IoU recomputation (see module header for details).
# =============================================================================

def compute_detection_ap(
    all_sample_results: list,
    iou_func,
    use_rotated_iou: bool = False,
    precomputed_iou_matrices: list | None = None,   ## NEW
) -> dict:
    """
    Compute mAP at multiple IoU thresholds.

    Adapted from Nadeem's eval_vit.py — same logic, extracted to reusable function.
    SOURCED: AP computation from Nadeem's eval_vit.py using calculate_ap()
    from utils.py (VOC Pascal style).

    Args:
        all_sample_results: list of dicts, each with:
            pred_scores:       [M] — detection confidence scores
            pred_boxes_xywha:  [M, 5] — predicted boxes
            gt_boxes_xywha:    [N, 5] — GT boxes

        iou_func: IoU function (axis-aligned or rotated). Only used when
            precomputed_iou_matrices is None.
        use_rotated_iou: bool. Only used when precomputed_iou_matrices is None.
        precomputed_iou_matrices: ## NEW — optional list, same length and
            order as all_sample_results. Entry i is either a [num_pred_i,
            num_gt_i] tensor of pred-vs-GT IoU values for sample i (in the
            ORIGINAL, unsorted pred order matching sample['pred_boxes_xywha']),
            or None if that sample has zero preds or zero GT boxes. When
            provided, IoU is looked up instead of recomputed. When None
            (default), behavior is identical to the original function.

    Returns:
        dict: mAP at each IoU threshold
    """
    ap_per_iou = {iou_t: [] for iou_t in DETECTION_IOU_THRESHOLDS}

    for idx, sample in enumerate(all_sample_results):   ## CHANGED — enumerate for cache lookup
        pred_scores = sample['pred_scores']
        pred_boxes = sample['pred_boxes_xywha']
        gt_boxes = sample['gt_boxes_xywha']
        num_gt = gt_boxes.shape[0]
        num_pred = pred_boxes.shape[0]

        # ## NEW — compute (or fetch) the full pred-vs-GT IoU matrix ONCE
        # per sample, reused across all 5 IoU thresholds below. This
        # matrix does not depend on the threshold — only the `>= iou_t`
        # comparison does — so recomputing it per threshold (as the
        # original code did) was pure redundant work with no effect on
        # results. Fixed here regardless of whether a cache is supplied.
        full_iou_matrix = None
        if num_pred > 0 and num_gt > 0:
            if precomputed_iou_matrices is not None:
                full_iou_matrix = precomputed_iou_matrices[idx]
            elif use_rotated_iou:
                full_iou_matrix = iou_func(pred_boxes.float(), gt_boxes.float())
            else:
                full_iou_matrix = iou_func(
                    pred_boxes[:, :4].float(), gt_boxes[:, :4].float()
                )

        sort_idx = torch.argsort(pred_scores, descending=True) if num_pred > 0 else None

        for iou_t in DETECTION_IOU_THRESHOLDS:
            if num_pred == 0:
                ap_per_iou[iou_t].append(1.0 if num_gt == 0 else 0.0)
                continue
            if num_gt == 0:
                ap_per_iou[iou_t].append(0.0)
                continue

            # Reorder rows by score descending — same effect as the
            # original code's pred_boxes_sorted-based matrix, just via
            # indexing into the cached/full matrix instead of recomputing.
            iou_matrix = full_iou_matrix[sort_idx]   ## CHANGED

            gt_matched = torch.zeros(num_gt, dtype=torch.bool)
            tp_flags = torch.zeros(num_pred, dtype=torch.bool)

            for i in range(num_pred):
                ious = iou_matrix[i, :]
                if ious.numel() == 0:
                    continue
                best_iou, best_gt_idx = torch.max(ious, dim=0)
                if best_iou >= iou_t and not gt_matched[best_gt_idx]:
                    tp_flags[i] = True
                    gt_matched[best_gt_idx] = True

            tp_cumsum = torch.cumsum(tp_flags.float(), dim=0)
            recall = tp_cumsum / (num_gt + 1e-9)
            precision = tp_cumsum / (
                torch.arange(1, num_pred + 1).float() + 1e-9
            )
            ap = calculate_ap(recall.numpy(), precision.numpy())
            ap_per_iou[iou_t].append(ap)

    return {
        f"mAP@{iou_t}": np.mean(aps) if aps else 0.0
        for iou_t, aps in ap_per_iou.items()
    }


# =============================================================================
# Intention metrics
# ## CHANGED — added optional precomputed_iou_matrices param
# =============================================================================

def compute_intention_metrics(
    all_sample_results: list,
    iou_func,
    use_rotated_iou: bool = False,
    precomputed_iou_matrices: list | None = None,   ## NEW
) -> dict:
    """
    Compute intention prediction accuracy and F1 scores.

    Only evaluates intention on True Positive detections (IoU ≥ threshold).
    SOURCED: matching logic from Nadeem's eval_vit.py.

    Args:
        all_sample_results: list of dicts, each with:
            pred_scores:      [M] — detection scores
            pred_boxes_xywha: [M, 5] — predicted boxes
            pred_intentions:  [M] — predicted intention class indices
            gt_boxes_xywha:   [N, 5] — GT boxes
            gt_intentions:    [N] — GT intention class indices

        precomputed_iou_matrices: ## NEW — same format/semantics as in
            compute_detection_ap(). When provided, skips recomputing IoU.

    Returns:
        dict with accuracy, F1 macro, F1 weighted, F1 per class
    """
    matched_pred = []
    matched_gt = []

    for idx, sample in enumerate(all_sample_results):   ## CHANGED — enumerate for cache lookup
        pred_scores = sample['pred_scores']
        pred_boxes = sample['pred_boxes_xywha']
        pred_intentions = sample['pred_intentions']
        gt_boxes = sample['gt_boxes_xywha']
        gt_intentions = sample['gt_intentions']

        num_gt = gt_boxes.shape[0]
        num_pred = pred_boxes.shape[0]

        if num_gt == 0 or num_pred == 0:
            continue

        if precomputed_iou_matrices is not None:   ## NEW
            iou_matrix = precomputed_iou_matrices[idx]
        elif use_rotated_iou:
            iou_matrix = iou_func(pred_boxes.float(), gt_boxes.float())
        else:
            iou_matrix = iou_func(
                pred_boxes[:, :4].float(), gt_boxes[:, :4].float()
            )

        gt_matched = torch.zeros(num_gt, dtype=torch.bool)
        sort_idx = torch.argsort(pred_scores, descending=True)

        for i in range(num_pred):
            orig_idx = sort_idx[i]
            ious = iou_matrix[orig_idx, :]
            if ious.numel() == 0:
                continue
            best_iou, best_gt_idx = torch.max(ious, dim=0)
            if best_iou >= IOU_THRESHOLD_FOR_INTENTION_MATCH:
                if not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    matched_pred.append(pred_intentions[orig_idx].item())
                    matched_gt.append(gt_intentions[best_gt_idx].item())

    if not matched_pred:
        return {
            "intention_accuracy": 0.0,
            "intention_f1_macro": 0.0,
            "intention_f1_weighted": 0.0,
            "intention_f1_per_class": {},
            "n_matched": 0
        }

    labels = list(range(NUM_INTENTION_CLASSES))
    accuracy = accuracy_score(matched_gt, matched_pred)
    f1_macro = f1_score(
        matched_gt, matched_pred,
        labels=labels, average='macro', zero_division=0
    )
    f1_weighted = f1_score(
        matched_gt, matched_pred,
        labels=labels, average='weighted', zero_division=0
    )
    f1_per_class = f1_score(
        matched_gt, matched_pred,
        labels=labels, average=None, zero_division=0
    )

    return {
        "intention_accuracy": accuracy,
        "intention_f1_macro": f1_macro,
        "intention_f1_weighted": f1_weighted,
        "intention_f1_per_class": {
            INTENTIONS_MAP_REV.get(i, f"Class_{i}"): f1_per_class[i]
            for i in labels
        },
        "n_matched": len(matched_pred)
    }


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """
    Print all metrics in a clean formatted table.

    Args:
        metrics: combined dict from detection + intention + trajectory metrics
        model_name: label for the model (e.g. "V1", "V2", "V3")
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*50}")

    # Detection
    print("\nDetection (mAP):")
    for k, v in metrics.items():
        if k.startswith("mAP"):
            print(f"  {k}: {v:.4f}")

    # Intention
    print("\nIntention Prediction:")
    if "intention_accuracy" in metrics:
        print(f"  Accuracy:    {metrics['intention_accuracy']:.4f}")
        print(f"  F1 (Macro):  {metrics['intention_f1_macro']:.4f}")
        print(f"  F1 (Wt.):    {metrics['intention_f1_weighted']:.4f}")
        print(f"  Matched TPs: {metrics.get('n_matched', 0)}")
        if "intention_f1_per_class" in metrics:
            print("  F1 per class:")
            for cls_name, f1 in metrics['intention_f1_per_class'].items():
                print(f"    {cls_name:<22}: {f1:.4f}")

    # Trajectory
    print("\nTrajectory Prediction:")
    if "minADE" in metrics:
        print(f"  minADE:  {metrics['minADE']:.4f} m")
        print(f"  minFDE:  {metrics['minFDE']:.4f} m")
        print(f"  MR:      {metrics['MR']:.4f}")
        print(f"  N vehicles evaluated: {metrics.get('N_vehicles', 0)}")
    else:
        print("  Not applicable (V1 mode)")

    print(f"{'='*50}\n")