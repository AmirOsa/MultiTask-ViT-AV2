# training/eval.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
# Original file: eval_vit.py
#
# [... all original header comments unchanged ...]
#
# ## NEW ADDITIONS FOR REVIEWER RESPONSE (AIBThings 2026 revision):
#  14. [NEW] End-to-end trajectory evaluation — samples trajectory features at
#      post-NMS PREDICTED box centers instead of GT box centers, so reported
#      metrics reflect real detection-to-trajectory error propagation rather
#      than oracle-detection performance. Runs as an independent second pass
#      over the val set; does not alter the existing teacher-forced (oracle)
#      trajectory evaluation in any way.
#  15. [NEW] Agent-density latency/memory sweep — benchmarks the trajectory
#      head alone (not the full model) across a range of agent counts N,
#      to isolate the cost of social attention as scene density grows.
#  16. [NEW] Result-saving hook — optionally dumps per-sample detection,
#      intention, and trajectory results to disk so bootstrap confidence
#      intervals can be computed afterward by a separate script
#      (utils/bootstrap_ci.py) without re-running inference.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.config_loader import load_config, get_config_arg, get_nested
from utils.constants import (
    GRID_HEIGHT_PX, GRID_WIDTH_PX,
    ANCHOR_CONFIGS_PAPER,
    LIDAR_TOTAL_CHANNELS, MAP_CHANNELS,
    TRAJECTORY_FUTURE_STEPS,
    TRAJECTORY_NUM_MODES,
    INTENTIONS_MAP_REV,
    NUM_INTENTION_CLASSES,
    BEV_X_MIN, BEV_X_MAX,
    BEV_Y_MIN, BEV_Y_MAX,
    VOXEL_SIZE_M,               # NEW — needed for box->feature-map pixel conversion
    BEV_PIXEL_OFFSET_X,         # NEW
    BEV_PIXEL_OFFSET_Y,         # NEW
)
from datasets.av2_dataset import ArgoverseIntentNetDataset, collate_fn
from datasets.parquet_dataset import ParquetTrajectoryDataset, parquet_collate_fn
from models.model_mt import IntentNetViT_MT
from models.backbone import BasicBlock
from utils.utils import (
    generate_anchors, decode_box_predictions,
    apply_nms, compute_axis_aligned_iou
)
from utils.metrics import (
    compute_detection_ap,
    compute_intention_metrics,
    compute_trajectory_metrics,
    accumulate_trajectory_metrics,
    print_metrics,
)

try:
    from utils.utils import compute_rotated_iou
    ROTATED_IOU_AVAILABLE = True
except ImportError:
    compute_rotated_iou = None
    ROTATED_IOU_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


# =============================================================================
# Agent-local frame transformation
# SOURCED: Abdulbaki thesis Section 3.6
# =============================================================================

def transform_to_agent_local(
    traj_ego: torch.Tensor,
    boxes_xywha: torch.Tensor,
) -> torch.Tensor:
    """
    Transform ego-frame trajectory positions to agent-local frame.
    SOURCED: Abdulbaki thesis Section 3.6
    """
    N       = traj_ego.shape[0]
    cx      = boxes_xywha[:, 0]
    cy      = boxes_xywha[:, 1]
    heading = boxes_xywha[:, 4]

    agent_pos = torch.stack([cx, cy], dim=-1).unsqueeze(1)
    relative  = traj_ego - agent_pos

    cos_h = torch.cos(-heading).view(N, 1)
    sin_h = torch.sin(-heading).view(N, 1)

    local_x = cos_h * relative[..., 0] - sin_h * relative[..., 1]
    local_y = sin_h * relative[..., 0] + cos_h * relative[..., 1]

    return torch.stack([local_x, local_y], dim=-1)


# =============================================================================
# ## NEW — box-to-feature-map-pixel conversion
# Standalone replica of IntentNetViT_MT._boxes_to_feature_map_pixels() so
# eval.py can drive the trajectory head directly with PREDICTED boxes
# without needing any change to model_mt.py.
# =============================================================================

def boxes_to_feature_map_pixels(
    boxes_xywha: torch.Tensor,
    feature_map_h: int,
    feature_map_w: int,
) -> torch.Tensor:
    """
    Convert box centres (ego-frame metres) to feature map pixel coords.
    Identical formula to IntentNetViT_MT._boxes_to_feature_map_pixels().

    Args:
        boxes_xywha:   [N, 5] boxes (cx_m, cy_m, w, l, heading) in ego frame
        feature_map_h: height of the backbone feature map (e.g. 50)
        feature_map_w: width of the backbone feature map (e.g. 90)

    Returns:
        centers_px: [N, 2] — (col, row) on the feature map
    """
    if boxes_xywha.shape[0] == 0:
        return torch.zeros(0, 2, device=boxes_xywha.device)

    cx_m = boxes_xywha[:, 0]
    cy_m = boxes_xywha[:, 1]

    bev_pixel_x = BEV_PIXEL_OFFSET_X + cy_m / VOXEL_SIZE_M
    bev_pixel_y = BEV_PIXEL_OFFSET_Y - cx_m / VOXEL_SIZE_M

    feature_stride = GRID_HEIGHT_PX // feature_map_h
    fm_pixel_x = (bev_pixel_x / feature_stride).clamp(0, feature_map_w - 1)
    fm_pixel_y = (bev_pixel_y / feature_stride).clamp(0, feature_map_h - 1)

    return torch.stack([fm_pixel_x, fm_pixel_y], dim=-1)


# =============================================================================
# ## NEW — greedy IoU matching, shared by end-to-end trajectory eval and
# reused with the same semantics as the existing confusion-matrix matching
# loop (one predicted box can match at most one GT box, highest score first).
# =============================================================================

def greedy_match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_func,
    use_rotated: bool,
    iou_threshold: float = 0.5,
):
    """
    Greedy one-to-one matching between predicted and GT boxes.

    Returns:
        matched_pred_idx: LongTensor [M] — indices into pred_boxes
        matched_gt_idx:   LongTensor [M] — indices into gt_boxes
    """
    num_pred = pred_boxes.shape[0]
    num_gt   = gt_boxes.shape[0]

    if num_pred == 0 or num_gt == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    if use_rotated and ROTATED_IOU_AVAILABLE:
        iou_mat = iou_func(pred_boxes.float(), gt_boxes.float())
    else:
        iou_mat = iou_func(pred_boxes[:, :4].float(), gt_boxes[:, :4].float())

    gt_matched = torch.zeros(num_gt, dtype=torch.bool)
    sort_idx   = torch.argsort(pred_scores, descending=True)

    matched_pred_idx = []
    matched_gt_idx    = []

    for i in range(num_pred):
        orig_idx = sort_idx[i]
        ious     = iou_mat[orig_idx, :]
        if ious.numel() == 0:
            continue
        best_iou, best_gt_idx = torch.max(ious, dim=0)
        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            gt_matched[best_gt_idx] = True
            matched_pred_idx.append(orig_idx.item())
            matched_gt_idx.append(best_gt_idx.item())

    return (
        torch.tensor(matched_pred_idx, dtype=torch.long),
        torch.tensor(matched_gt_idx, dtype=torch.long),
    )


# =============================================================================
# Distance-binned mAP
# (unchanged)
# =============================================================================

def compute_distance_binned_map(all_sample_results, iou_func,
                                use_rotated, bins=None):
    """Compute mAP@0.5 broken down by GT box distance from ego vehicle."""
    if bins is None:
        bins = [
            (0,   20,  '0-20m'),
            (20,  40,  '20-40m'),
            (40,  60,  '40-60m'),
            (60,  100, '60-100m'),
        ]

    results_per_bin = {label: [] for _, _, label in bins}

    for sample in all_sample_results:
        pred_scores = sample['pred_scores']
        pred_boxes  = sample['pred_boxes_xywha']
        gt_boxes    = sample['gt_boxes_xywha']
        num_gt      = gt_boxes.shape[0]
        num_pred    = pred_boxes.shape[0]

        if num_gt == 0:
            continue

        gt_dist = torch.sqrt(gt_boxes[:, 0]**2 + gt_boxes[:, 1]**2)

        # ── Filter GT boxes to BEV bounds only ──
        in_bev_mask = (
            (gt_boxes[:, 0] >= BEV_X_MIN) &
            (gt_boxes[:, 0] <= BEV_X_MAX) &
            (gt_boxes[:, 1] >= BEV_Y_MIN) &
            (gt_boxes[:, 1] <= BEV_Y_MAX)
        )
        gt_boxes = gt_boxes[in_bev_mask]
        gt_dist  = gt_dist[in_bev_mask]
        num_gt   = gt_boxes.shape[0]

        if num_gt == 0:
            continue
        # ── End of filter ──

        for min_d, max_d, label in bins:
            bin_mask     = (gt_dist >= min_d) & (gt_dist < max_d)
            gt_boxes_bin = gt_boxes[bin_mask]
            num_gt_bin   = gt_boxes_bin.shape[0]

            if num_gt_bin == 0:
                continue
            if num_pred == 0:
                results_per_bin[label].append(0.0)
                continue

            if use_rotated and ROTATED_IOU_AVAILABLE:
                iou_matrix = iou_func(pred_boxes.float(), gt_boxes_bin.float())
            else:
                iou_matrix = iou_func(
                    pred_boxes[:, :4].float(), gt_boxes_bin[:, :4].float()
                )

            sort_idx   = torch.argsort(pred_scores, descending=True)
            gt_matched = torch.zeros(num_gt_bin, dtype=torch.bool)
            tp_flags   = torch.zeros(num_pred, dtype=torch.bool)

            for i in range(num_pred):
                ious = iou_matrix[sort_idx[i], :]
                if ious.numel() == 0:
                    continue
                best_iou, best_gt_idx = torch.max(ious, dim=0)
                if best_iou >= 0.5 and not gt_matched[best_gt_idx]:
                    tp_flags[i]             = True
                    gt_matched[best_gt_idx] = True

            tp_cumsum = torch.cumsum(tp_flags.float(), dim=0)
            recall    = tp_cumsum / (num_gt_bin + 1e-9)
            precision = tp_cumsum / (torch.arange(1, num_pred + 1).float() + 1e-9)

            from utils.utils import calculate_ap
            ap = calculate_ap(recall.numpy(), precision.numpy())
            results_per_bin[label].append(ap)

    return {
        label: float(np.mean(aps)) if aps else 0.0
        for label, aps in results_per_bin.items()
    }


# =============================================================================
# Parquet trajectory evaluation — NEW for V4/V5 (unchanged)
# =============================================================================

def evaluate_trajectory_parquet(
    model,
    parquet_val_dir: str,
    sensor_val_dir: str,
    device,
    use_agent_local: bool = True,
    eval_focal_only: bool = True,
    eval_all_agents: bool = True,
) -> dict:
    """
    Evaluate trajectory on parquet val scenarios.
    (unchanged from original — see previous version for full docstring)
    """
    print(f"\nEvaluating trajectory on parquet val scenarios...")
    print(f"  Parquet dir: {parquet_val_dir}")
    print(f"  Focal only:  {eval_focal_only}")
    print(f"  All agents:  {eval_all_agents}")

    try:
        parquet_val_dataset = ParquetTrajectoryDataset(
            parquet_dir=parquet_val_dir,
            sensor_dir=sensor_val_dir,
            is_train=False,
        )
        parquet_val_loader = DataLoader(
            parquet_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=parquet_collate_fn,
        )
        print(f"  Parquet val scenarios: {len(parquet_val_dataset)}")
    except Exception as e:
        print(f"  ERROR loading parquet val dataset: {e}")
        return {}

    focal_traj_results = []
    all_agent_traj_results = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(parquet_val_loader, desc="Parquet eval", unit="scenario"):
            if batch is None:
                continue

            try:
                lidar_bev = batch["lidar_bev"].to(device)
                map_bev   = batch["map_bev"].to(device)

                gt_boxes      = batch["gt_boxes"][0].to(device)
                agent_history = batch["agent_history"][0].to(device)
                gt_traj_focal = batch["gt_traj_focal"][0].to(device)
                gt_mask_focal = batch["gt_mask_focal"][0].to(device)
                gt_traj_all   = batch["gt_traj_all"][0].to(device)
                gt_mask_all   = batch["gt_mask_all"][0].to(device)
                focal_idx     = batch["focal_idx"][0]

                if gt_boxes.shape[0] == 0:
                    continue

                outputs = model.forward_traj_only(
                    lidar_bev=lidar_bev,
                    map_bev=map_bev,
                    gt_boxes=gt_boxes,
                    agent_history=agent_history,
                )

                y_hat = outputs.get("y_hat")
                pi    = outputs.get("pi")

                if y_hat is None or y_hat.shape[1] == 0:
                    continue

                N = y_hat.shape[1]

                if eval_focal_only and focal_idx < N:
                    y_hat_focal = y_hat[:, focal_idx:focal_idx+1, :, :]
                    pi_focal = pi[focal_idx:focal_idx+1, :]

                    gt_traj_f = gt_traj_focal.unsqueeze(0)
                    gt_mask_f = gt_mask_focal.unsqueeze(0)

                    if use_agent_local:
                        focal_box = gt_boxes[focal_idx:focal_idx+1]

                        gt_traj_local = transform_to_agent_local(
                            gt_traj_f, focal_box
                        )

                        pred_local_modes = []
                        for f in range(y_hat_focal.shape[0]):
                            local_f = transform_to_agent_local(
                                y_hat_focal[f, :, :, :2],
                                focal_box
                            )
                            pred_local_modes.append(local_f)
                        pred_pos_local = torch.stack(pred_local_modes, dim=0)

                        y_hat_eval = y_hat_focal.clone()
                        y_hat_eval[..., :2] = pred_pos_local
                        gt_traj_eval = gt_traj_local
                    else:
                        y_hat_eval   = y_hat_focal
                        gt_traj_eval = gt_traj_f

                    focal_metrics = compute_trajectory_metrics(
                        y_hat=y_hat_eval,
                        pi=pi_focal,
                        gt_traj=gt_traj_eval,
                        gt_mask=gt_mask_f,
                    )
                    focal_traj_results.append(focal_metrics)

                if eval_all_agents and N > 0:
                    N_eval = min(N, gt_traj_all.shape[0])

                    if use_agent_local and gt_boxes.shape[0] >= N_eval:
                        gt_boxes_eval = gt_boxes[:N_eval]

                        in_bev_mask = (
                            (gt_boxes_eval[:, 0] >= BEV_X_MIN) &
                            (gt_boxes_eval[:, 0] <= BEV_X_MAX) &
                            (gt_boxes_eval[:, 1] >= BEV_Y_MIN) &
                            (gt_boxes_eval[:, 1] <= BEV_Y_MAX)
                        )
                        in_bev_idx = torch.where(in_bev_mask)[0]
                        N_in_bev = len(in_bev_idx)

                        if N_in_bev > 0:
                            gt_traj_inbev  = gt_traj_all[:N_eval][in_bev_idx]
                            gt_mask_inbev  = gt_mask_all[:N_eval][in_bev_idx]
                            gt_boxes_inbev = gt_boxes_eval[in_bev_idx]

                            gt_traj_local = transform_to_agent_local(
                                gt_traj_inbev, gt_boxes_inbev
                            )

                            pred_local_modes = []
                            for f in range(y_hat.shape[0]):
                                local_f = transform_to_agent_local(
                                    y_hat[f, in_bev_idx, :, :2],
                                    gt_boxes_inbev
                                )
                                pred_local_modes.append(local_f)
                            pred_pos_local = torch.stack(pred_local_modes, dim=0)

                            y_hat_all_eval = y_hat[:, in_bev_idx].clone()
                            y_hat_all_eval[..., :2] = pred_pos_local

                            all_metrics = compute_trajectory_metrics(
                                y_hat=y_hat_all_eval,
                                pi=pi[in_bev_idx],
                                gt_traj=gt_traj_local,
                                gt_mask=gt_mask_inbev,
                            )
                            all_agent_traj_results.append(all_metrics)
                    else:
                        all_metrics = compute_trajectory_metrics(
                            y_hat=y_hat[:, :N_eval],
                            pi=pi[:N_eval],
                            gt_traj=gt_traj_all[:N_eval],
                            gt_mask=gt_mask_all[:N_eval],
                        )
                        all_agent_traj_results.append(all_metrics)

            except Exception as e:
                print(f"  Error in parquet eval batch: {e}")
                continue

    results = {}

    if focal_traj_results:
        focal_agg = accumulate_trajectory_metrics(focal_traj_results)
        results['focal_minADE'] = focal_agg['minADE']
        results['focal_minFDE'] = focal_agg['minFDE']
        results['focal_MR']     = focal_agg['MR']
        results['focal_N']      = focal_agg['N_vehicles']
        print(f"\nParquet Trajectory (focal agent, MF protocol):")
        print(f"  minADE: {focal_agg['minADE']:.4f} m")
        print(f"  minFDE: {focal_agg['minFDE']:.4f} m")
        print(f"  MR:     {focal_agg['MR']:.4f}")
        print(f"  N:      {focal_agg['N_vehicles']}")

    if all_agent_traj_results:
        all_agg = accumulate_trajectory_metrics(all_agent_traj_results)
        results['all_minADE'] = all_agg['minADE']
        results['all_minFDE'] = all_agg['minFDE']
        results['all_MR']     = all_agg['MR']
        results['all_N']      = all_agg['N_vehicles']
        print(f"\nParquet Trajectory (all agents):")
        print(f"  minADE: {all_agg['minADE']:.4f} m")
        print(f"  minFDE: {all_agg['minFDE']:.4f} m")
        print(f"  MR:     {all_agg['MR']:.4f}")
        print(f"  N:      {all_agg['N_vehicles']}")

    return results


# =============================================================================
# ## NEW — End-to-end (non-oracle) trajectory evaluation
#
# Runs as an INDEPENDENT second pass over the validation set. Detection is
# run and NMS'd exactly as in the main sensor loop, predicted boxes are
# matched to GT via IoU, and the trajectory head is queried directly at the
# PREDICTED box locations/params (not GT). This measures real detection ->
# trajectory error propagation instead of oracle-detection performance,
# directly addressing R5's point that teacher-forced evaluation does not
# reflect deployed behaviour.
#
# The existing oracle/teacher-forced trajectory evaluation inside the main
# sensor loop is completely untouched by this function.
# =============================================================================

def evaluate_trajectory_end_to_end(
    model,
    val_loader,
    anchors: torch.Tensor,
    device,
    confidence_threshold: float,
    nms_iou_threshold: float,
    use_rotated_nms: bool,
    use_rotated_iou_for_matching: bool,
    use_agent_local_traj: bool,
    iou_match_threshold: float = 0.5,
) -> tuple:
    """
    Evaluate trajectory prediction using PREDICTED (post-NMS) box locations
    to sample the trajectory head, instead of ground-truth box centres.

    Returns:
        agg_metrics: dict — minADE / minFDE / MR / N_vehicles, aggregated
        raw_results: list of per-batch dicts from compute_trajectory_metrics,
                     kept for later bootstrap CI computation
    """
    print("\nRunning END-TO-END trajectory evaluation "
          "(predicted detections, not GT teacher forcing)...")

    if getattr(model, 'traj_head', None) is None:
        print("  Model has no trajectory head — skipping.")
        return {}, []

    iou_func = compute_rotated_iou if (
        use_rotated_iou_for_matching and ROTATED_IOU_AVAILABLE
    ) else compute_axis_aligned_iou

    feature_map_h = model.feature_map_h
    feature_map_w = model.feature_map_w

    PARKED_CLASS       = 6
    MIN_DISPLACEMENT_M = 0.5

    raw_results = []

    model.eval()
    with torch.inference_mode():
        pbar = tqdm(val_loader, desc="End-to-end traj eval", unit="batch")

        for batch_data in pbar:
            if batch_data is None:
                continue

            gt_list_cpu = batch_data["gt_list"]
            batch_size  = batch_data["lidar_bev"].shape[0]

            try:
                lidar_bev = batch_data["lidar_bev"].to(device, non_blocking=True)
                map_bev   = batch_data["map_bev"].to(device, non_blocking=True)

                # Detection + intention only — trajectory head is queried
                # manually below at predicted box locations.
                outputs = model(
                    lidar_bev, map_bev,
                    gt_list=None,
                    use_gt_boxes_for_traj=False,
                    agent_history=None,
                    run_traj_head=False,
                )

                det_cls_logits = outputs["det_cls_logits"]
                det_box_preds  = outputs["det_box_preds"]
                feature_map    = outputs["feature_map"]  # [B, 512, 50, 90]

                for b_idx in range(batch_size):
                    gt_b = gt_list_cpu[b_idx]
                    if gt_b is None or gt_b['boxes_xywha'].shape[0] == 0:
                        continue

                    # --- Decode + NMS predicted boxes for this scene ---
                    cls_s = det_cls_logits[b_idx]
                    box_s = det_box_preds[b_idx]
                    scores = torch.sigmoid(cls_s)
                    if scores.ndim > 1:
                        scores = scores.squeeze(-1)

                    keep = torch.where(scores >= confidence_threshold)[0]
                    if keep.numel() == 0:
                        continue

                    sc_f   = scores[keep]
                    anch_f = anchors[keep]
                    box_f  = box_s[keep]

                    boxes_abs = decode_box_predictions(box_f, anch_f)
                    nms_keep  = apply_nms(
                        boxes_abs, sc_f, nms_iou_threshold,
                        use_rotated=use_rotated_nms
                    )
                    if nms_keep.numel() == 0:
                        continue

                    pred_boxes  = boxes_abs[nms_keep]       # [M, 5]
                    pred_scores = sc_f[nms_keep]            # [M]

                    # --- Match predicted boxes to GT boxes ---
                    gt_boxes      = gt_b['boxes_xywha'].to(device)
                    gt_intentions = gt_b['intentions'].to(device)

                    matched_pred_idx, matched_gt_idx = greedy_match_predictions_to_gt(
                        pred_boxes.cpu(), pred_scores.cpu(), gt_boxes.cpu(),
                        iou_func, use_rotated_iou_for_matching, iou_match_threshold
                    )
                    if matched_pred_idx.numel() == 0:
                        continue

                    matched_pred_idx = matched_pred_idx.to(device)
                    matched_gt_idx   = matched_gt_idx.to(device)

                    # --- Active-agent filter, applied on the GT side ---
                    # (same filter used for training/oracle eval, so results
                    # are directly comparable)
                    if 'future_traj_ego' not in gt_b:
                        continue

                    gt_traj_all = gt_b['future_traj_ego'].to(device)
                    gt_mask_all = gt_b['future_traj_mask'].to(device)

                    gt_traj_matched  = gt_traj_all[matched_gt_idx]
                    gt_mask_matched  = gt_mask_all[matched_gt_idx]
                    gt_intent_matched = gt_intentions[matched_gt_idx]

                    intent_mask = (gt_intent_matched != PARKED_CLASS)
                    displacements = (
                        gt_traj_matched.norm(dim=-1) * gt_mask_matched.float()
                    ).max(dim=-1).values
                    disp_mask   = displacements > MIN_DISPLACEMENT_M
                    active_mask = intent_mask & disp_mask

                    if not active_mask.any():
                        continue

                    final_pred_boxes = pred_boxes[matched_pred_idx][active_mask]
                    final_gt_traj    = gt_traj_matched[active_mask]
                    final_gt_mask    = gt_mask_matched[active_mask]

                    # --- Query trajectory head directly at PREDICTED boxes ---
                    pred_centers_px = boxes_to_feature_map_pixels(
                        final_pred_boxes, feature_map_h, feature_map_w
                    )

                    y_hat, pi = model.traj_head(
                        feature_map=feature_map[b_idx:b_idx+1],
                        map_bev=map_bev[b_idx:b_idx+1],
                        box_centers_px=pred_centers_px,
                        box_params_m=final_pred_boxes,
                        use_gt_boxes=False,
                    )
                    # y_hat: [F, N_active, H, 4] — already relative to the
                    # PREDICTED box params fed in above (the decoder was
                    # trained to output positions relative to whatever box
                    # params it is given), so no further frame transform is
                    # applied to y_hat itself.

                    if y_hat is None or y_hat.shape[1] == 0:
                        continue

                    # --- Transform GT into the PREDICTED box's frame ---
                    # This is the step that captures error propagation from
                    # detection localisation/heading error into trajectory
                    # error — we deliberately do NOT use the GT box here.
                    if use_agent_local_traj:
                        gt_traj_eval = transform_to_agent_local(
                            final_gt_traj, final_pred_boxes
                        )
                    else:
                        gt_traj_eval = final_gt_traj

                    metrics = compute_trajectory_metrics(
                        y_hat=y_hat,
                        pi=pi,
                        gt_traj=gt_traj_eval,
                        gt_mask=final_gt_mask,
                    )
                    raw_results.append(metrics)

            except Exception as e:
                print(f"  Error in end-to-end traj eval batch: {e}")
                continue

    if not raw_results:
        print("  No matched active agents found — end-to-end trajectory "
              "metrics unavailable.")
        return {}, []

    agg_metrics = accumulate_trajectory_metrics(raw_results)
    print(f"\nEnd-to-End Trajectory (predicted detections):")
    print(f"  minADE: {agg_metrics['minADE']:.4f} m")
    print(f"  minFDE: {agg_metrics['minFDE']:.4f} m")
    print(f"  MR:     {agg_metrics['MR']:.4f}")
    print(f"  N:      {agg_metrics['N_vehicles']}")

    return agg_metrics, raw_results


# =============================================================================
# ## NEW — Agent-density latency / memory benchmark
#
# Benchmarks the TRAJECTORY HEAD ALONE (fixed backbone feature map) across
# a range of agent counts N, to isolate the cost of social attention (V3)
# as scene density grows — directly addressing R3's point that identical
# reported MACs/latency for the full model do not isolate this cost.
# =============================================================================

def benchmark_trajectory_head_by_agent_count(
    model,
    device,
    agent_counts: list = None,
    num_warmup: int = 5,
    num_trials: int = 30,
) -> dict:
    """
    Times model.traj_head forward passes for increasing numbers of agents N,
    holding the backbone feature map fixed (randomly initialised, same
    across all N). Reports latency and peak CUDA memory per N.

    Returns:
        dict mapping N -> {'latency_ms_mean', 'latency_ms_std', 'peak_mem_mb'}
    """
    if getattr(model, 'traj_head', None) is None:
        print("Model has no trajectory head — skipping agent-density benchmark.")
        return {}

    if agent_counts is None:
        agent_counts = [1, 5, 10, 20, 30, 50, 75, 100]

    print(f"\nBenchmarking trajectory head across agent counts: {agent_counts}")

    feature_map_h = model.feature_map_h
    feature_map_w = model.feature_map_w
    feature_channels = model.feature_channels

    dummy_feature_map = torch.randn(
        1, feature_channels, feature_map_h, feature_map_w, device=device
    )
    dummy_map_bev = torch.randn(
        1, MAP_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX, device=device
    )

    results = {}
    model.eval()

    with torch.no_grad():
        for N in agent_counts:
            dummy_centers_px = torch.stack([
                torch.rand(N, device=device) * (feature_map_w - 1),
                torch.rand(N, device=device) * (feature_map_h - 1),
            ], dim=-1)

            dummy_box_params = torch.zeros(N, 5, device=device)
            dummy_box_params[:, 0] = torch.rand(N, device=device) * 40 - 10  # cx
            dummy_box_params[:, 1] = torch.rand(N, device=device) * 40 - 20  # cy
            dummy_box_params[:, 2] = 2.0   # w
            dummy_box_params[:, 3] = 4.5   # l
            dummy_box_params[:, 4] = torch.rand(N, device=device) * 3.14 - 1.57  # heading

            # Warmup
            for _ in range(num_warmup):
                model.traj_head(
                    feature_map=dummy_feature_map,
                    map_bev=dummy_map_bev,
                    box_centers_px=dummy_centers_px,
                    box_params_m=dummy_box_params,
                    use_gt_boxes=False,
                )
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)

            times = []
            for _ in range(num_trials):
                t_start = time.time()
                model.traj_head(
                    feature_map=dummy_feature_map,
                    map_bev=dummy_map_bev,
                    box_centers_px=dummy_centers_px,
                    box_params_m=dummy_box_params,
                    use_gt_boxes=False,
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - t_start)

            mean_ms = float(np.mean(times) * 1000)
            std_ms  = float(np.std(times) * 1000)

            peak_mem_mb = 0.0
            if device.type == 'cuda':
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            results[N] = {
                'latency_ms_mean': mean_ms,
                'latency_ms_std':  std_ms,
                'peak_mem_mb':     peak_mem_mb,
            }
            print(
                f"  N={N:>4}: {mean_ms:6.2f} ± {std_ms:5.2f} ms   "
                f"peak mem: {peak_mem_mb:7.1f} MB"
            )

    return results


# =============================================================================
# Main evaluation function
# =============================================================================

def main_eval():
    """Main evaluation function for IntentTrajNet-AV2."""

    # =========================================================================
    # Load config
    # =========================================================================
    config_path = get_config_arg()
    cfg = load_config(config_path)

    # =========================================================================
    # Read settings from config
    # =========================================================================
    MODEL_VERSION  = get_nested(cfg, 'model', 'version', default='V2')
    USE_TRAJECTORY = get_nested(cfg, 'model', 'use_trajectory', default=False)
    backbone_type  = get_nested(cfg, 'model', 'backbone', 'type', default='vit')
    decoder_type   = get_nested(cfg, 'model', 'trajectory', 'decoder_type', default='mlp')

    VAL_DATA_DIR = get_nested(cfg, 'data', 'val_dir', default='')

    # Parquet eval settings — for V4/V5
    PARQUET_VAL_DIR     = get_nested(cfg, 'data', 'parquet_val_dir',  default='')
    EVAL_FOCAL_ONLY     = get_nested(cfg, 'eval', 'eval_focal_only',  default=False)
    EVAL_ALL_AGENTS     = get_nested(cfg, 'eval', 'eval_all_agents',  default=False)
    USE_PARQUET_EVAL    = bool(PARQUET_VAL_DIR) and (EVAL_FOCAL_ONLY or EVAL_ALL_AGENTS)

    # ── NEW eval config flags — all default to sensible values so existing
    #    configs (without these keys) still run, just with the new analyses
    #    enabled by default since that's what this revision needs. ──
    EVAL_END_TO_END_TRAJ    = get_nested(cfg, 'eval', 'eval_end_to_end_traj',    default=True)
    BENCHMARK_AGENT_DENSITY = get_nested(cfg, 'eval', 'benchmark_agent_density', default=True)
    SAVE_RESULTS_PATH       = get_nested(cfg, 'eval', 'save_results_path',      default='')
    AGENT_DENSITY_COUNTS    = get_nested(
        cfg, 'eval', 'agent_density_counts',
        default=[1, 5, 10, 20, 30, 50, 75, 100]
    )

    CHECKPOINT_PATH = get_nested(
        cfg, 'checkpoints', 'save_dir', default=''
    ) + '/' + get_nested(
        cfg, 'checkpoints', 'filename',
        default=f'MultiTask_{MODEL_VERSION}.pth'
    )

    CONFIDENCE_THRESHOLD  = get_nested(cfg, 'eval', 'confidence_threshold', default=0.1)
    NMS_IOU_THRESHOLD     = get_nested(cfg, 'eval', 'nms_iou_threshold',    default=0.2)
    EVAL_USE_ROTATED_IOU  = get_nested(cfg, 'eval', 'use_rotated_iou',      default=False)
    EVAL_USE_ROTATED_NMS  = get_nested(cfg, 'eval', 'use_rotated_nms',      default=False)
    EVAL_USE_AGENT_LOCAL_TRAJ = get_nested(
        cfg, 'eval', 'use_agent_local_traj', default=True
    )

    INFERENCE_BATCH_SIZE = get_nested(cfg, 'eval',     'batch_size',   default=8)
    NUM_WORKERS          = get_nested(cfg, 'training', 'num_workers',  default=0)

    vit_model_name = get_nested(
        cfg, 'model', 'backbone', 'vit_model_name_lidar',
        default='vit_small_patch8_224'
    )
    fusion_stride = get_nested(cfg, 'model', 'backbone', 'fusion_block_stride', default=1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  IntentTrajNet-AV2 Evaluation — {MODEL_VERSION}")
    print(f"  Config: {config_path}")
    print(f"{'='*60}")
    print(f"  Device:                {DEVICE}")
    print(f"  Val data:              {VAL_DATA_DIR}")
    print(f"  Checkpoint:            {CHECKPOINT_PATH}")
    print(f"  Use trajectory:        {USE_TRAJECTORY}")
    print(f"  Decoder type:          {decoder_type}")
    print(f"  Parquet eval:          {USE_PARQUET_EVAL}")
    print(f"  Rotated IoU (mAP):     {EVAL_USE_ROTATED_IOU}")
    print(f"  Rotated NMS:           {EVAL_USE_ROTATED_NMS}")
    print(f"  Agent-local traj:      {EVAL_USE_AGENT_LOCAL_TRAJ}")
    print(f"  End-to-end traj eval:  {EVAL_END_TO_END_TRAJ}")          # NEW
    print(f"  Agent-density bench:   {BENCHMARK_AGENT_DENSITY}")       # NEW
    print(f"  Save results to:       {SAVE_RESULTS_PATH or '(disabled)'}")  # NEW
    print(f"{'='*60}\n")

    # =========================================================================
    # Load checkpoint
    # =========================================================================
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.is_file():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return

    print("Loading checkpoint...")
    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
        )
        print("Checkpoint loaded.\n")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return

    # =========================================================================
    # Model — read decoder_type from checkpoint for correct init
    # =========================================================================
    saved_backbone_cfg = checkpoint.get('backbone_cfg', {})
    use_trajectory     = checkpoint.get('use_trajectory', USE_TRAJECTORY)
    saved_decoder_type = checkpoint.get('decoder_type', decoder_type)

    saved_backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
    saved_backbone_cfg.setdefault('lidar_input_channels', LIDAR_TOTAL_CHANNELS)
    saved_backbone_cfg.setdefault('map_input_channels', MAP_CHANNELS)
    saved_backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch8_224')
    saved_backbone_cfg.setdefault('vit_model_name_map',   'vit_small_patch8_224')
    saved_backbone_cfg.setdefault('pretrained_lidar', False)
    saved_backbone_cfg.setdefault('pretrained_map',   False)
    saved_backbone_cfg.setdefault('fusion_block_planes', 512)
    saved_backbone_cfg.setdefault('res_block_type', BasicBlock)

    traj_head_cfg = {}
    if use_trajectory:
        traj_head_cfg = {
            'box_feat_dim': 5,
            'mlp_dropout':  0.0,
        }
    if saved_decoder_type == 'transformer':
        traj_head_cfg.update({
            'gru_hidden':         get_nested(cfg, 'model', 'trajectory', 'gru_hidden',         default=64),
            'num_heads':          get_nested(cfg, 'model', 'trajectory', 'num_heads',           default=8),
            'num_decoder_layers': get_nested(cfg, 'model', 'trajectory', 'num_decoder_layers',  default=2),
            'social_heads':       get_nested(cfg, 'model', 'trajectory', 'social_heads',        default=4),
            'social_layers':      get_nested(cfg, 'model', 'trajectory', 'social_layers',       default=1),
            'dropout':            get_nested(cfg, 'model', 'trajectory', 'dropout',             default=0.1),
        })

    saved_backbone_type = checkpoint.get('backbone_cfg', {}).get('type', backbone_type)
    saved_backbone_cfg.pop('type', None)

    try:
        model = IntentNetViT_MT(
            backbone_type=saved_backbone_type,
            backbone_cfg=saved_backbone_cfg,
            use_trajectory=use_trajectory,
            decoder_type=saved_decoder_type,
            trajectory_head_cfg=traj_head_cfg,
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded and set to eval mode.\n")
    except Exception as e:
        print(f"ERROR instantiating model: {e}")
        return

    # =========================================================================
    # Sensor Dataset — detection + intention + auxiliary trajectory eval
    # =========================================================================
    val_data_path = Path(VAL_DATA_DIR)
    if not val_data_path.is_dir():
        print(f"ERROR: Val data not found: {VAL_DATA_DIR}")
        return

    print("Initializing sensor eval dataset...")
    try:
        val_dataset = ArgoverseIntentNetDataset(
            data_dir=VAL_DATA_DIR, is_train=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=(DEVICE.type == 'cuda')
        )
        print(f"Sensor val DataLoader: {len(val_loader)} batches.\n")
    except Exception as e:
        print(f"ERROR initializing sensor val dataset: {e}")
        return

    # =========================================================================
    # Anchors
    # =========================================================================
    print("Generating anchors...")
    if backbone_type == 'swin':
        FEATURE_MAP_STRIDE = 8
    else:
        try:
            vit_patch_stride = int(vit_model_name.split('_patch')[-1].split('_')[0])
        except ValueError:
            vit_patch_stride = 8
        FEATURE_MAP_STRIDE = vit_patch_stride * fusion_stride

    anchors = generate_anchors(
        bev_height=GRID_HEIGHT_PX,
        bev_width=GRID_WIDTH_PX,
        feature_map_stride=FEATURE_MAP_STRIDE,
        anchor_configs=ANCHOR_CONFIGS_PAPER
    ).to(DEVICE)
    print(f"Anchors: {anchors.shape}")

    if EVAL_USE_ROTATED_IOU and ROTATED_IOU_AVAILABLE:
        iou_func = compute_rotated_iou
        print("Using rotated IoU.\n")
    else:
        iou_func = compute_axis_aligned_iou
        print("Using axis-aligned IoU.\n")

    if EVAL_USE_ROTATED_NMS and not ROTATED_IOU_AVAILABLE:
        print("WARNING: Rotated NMS requested but unavailable. Using axis-aligned.\n")
        EVAL_USE_ROTATED_NMS = False

    # =========================================================================
    # Sensor inference loop — detection + intention + auxiliary (oracle)
    # trajectory. UNCHANGED from original.
    # =========================================================================
    print("Running sensor inference...")
    all_sample_results      = []
    all_traj_metric_results = []

    with torch.inference_mode():
        pbar = tqdm(val_loader, desc=f"Sensor Eval [{MODEL_VERSION}]", unit="batch")

        for batch_data in pbar:
            if batch_data is None:
                continue

            batch_size  = batch_data["lidar_bev"].shape[0]
            gt_list_cpu = batch_data["gt_list"]

            try:
                lidar_bev = batch_data["lidar_bev"].to(DEVICE, non_blocking=True)
                map_bev   = batch_data["map_bev"].to(DEVICE, non_blocking=True)

                run_traj = use_trajectory and (saved_decoder_type in ('mlp', 'social_mlp'))

                outputs = model(
                    lidar_bev, map_bev,
                    gt_list=gt_list_cpu,
                    use_gt_boxes_for_traj=True,
                    agent_history=None,
                    run_traj_head=run_traj,
                )

                det_cls_logits   = outputs["det_cls_logits"]
                det_box_preds    = outputs["det_box_preds"]
                intention_logits = outputs["intention_logits"]
                y_hat = outputs.get("y_hat")
                pi    = outputs.get("pi")

                for b_idx in range(batch_size):
                    sample_pred = {
                        'pred_scores':      torch.empty(0, device='cpu'),
                        'pred_boxes_xywha': torch.empty((0, 5), device='cpu'),
                        'pred_intentions':  torch.empty(0, dtype=torch.long, device='cpu')
                    }
                    try:
                        cls_s = det_cls_logits[b_idx]
                        box_s = det_box_preds[b_idx]
                        int_s = intention_logits[b_idx]

                        scores = torch.sigmoid(cls_s)
                        if scores.ndim > 1:
                            scores = scores.squeeze(-1)

                        keep = torch.where(scores >= CONFIDENCE_THRESHOLD)[0]
                        if keep.numel() > 0:
                            sc_f   = scores[keep]
                            anch_f = anchors[keep]
                            box_f  = box_s[keep]
                            int_f  = int_s[keep]

                            boxes_abs = decode_box_predictions(box_f, anch_f)
                            nms_keep  = apply_nms(
                                boxes_abs, sc_f, NMS_IOU_THRESHOLD,
                                use_rotated=EVAL_USE_ROTATED_NMS
                            )

                            if nms_keep.numel() > 0:
                                sample_pred['pred_scores']      = sc_f[nms_keep].cpu()
                                sample_pred['pred_boxes_xywha'] = boxes_abs[nms_keep].cpu()
                                sample_pred['pred_intentions']  = torch.argmax(
                                    int_f[nms_keep], dim=-1
                                ).cpu()
                    except Exception as e:
                        print(f"Error post-processing b_idx={b_idx}: {e}")

                    gt = gt_list_cpu[b_idx]
                    all_sample_results.append({
                        **sample_pred,
                        'gt_boxes_xywha': gt.get('boxes_xywha', torch.empty((0, 5))),
                        'gt_intentions':  gt.get('intentions',  torch.empty(0, dtype=torch.long))
                    })

                # Auxiliary (oracle) trajectory metrics — UNCHANGED
                if run_traj and y_hat is not None:
                    gt_b0 = gt_list_cpu[0]
                    if gt_b0 is not None and 'future_traj_ego' in gt_b0:
                        gt_traj  = gt_b0['future_traj_ego'].to(DEVICE)
                        gt_mask  = gt_b0['future_traj_mask'].to(DEVICE)
                        gt_boxes = gt_b0['boxes_xywha'].to(DEVICE)

                        N_pred = y_hat.shape[1]
                        N_gt   = gt_traj.shape[0]
                        N_eval = min(N_pred, N_gt)

                        if N_eval > 0:
                            intentions_eval = gt_b0['intentions'].to(DEVICE)[:N_eval]
                            PARKED_CLASS = 6
                            intent_mask = (intentions_eval != PARKED_CLASS)

                            MIN_DISPLACEMENT_M = 0.5
                            traj_ego_eval  = gt_traj[:N_eval]
                            traj_mask_eval = gt_mask[:N_eval].float()
                            displacements  = (
                                traj_ego_eval.norm(dim=-1) * traj_mask_eval
                            ).max(dim=-1).values
                            disp_mask = displacements > MIN_DISPLACEMENT_M

                            moving_mask = intent_mask & disp_mask
                            if not moving_mask.any():
                                continue

                            gt_traj  = gt_traj[:N_eval][moving_mask]
                            gt_mask  = gt_mask[:N_eval][moving_mask]
                            gt_boxes = gt_boxes[:N_eval][moving_mask]
                            y_hat    = y_hat[:, :N_eval][:, moving_mask]
                            if pi is not None:
                                pi = pi[:N_eval][moving_mask]
                            N_eval   = moving_mask.sum().item()

                            gt_boxes_eval = gt_boxes
                            in_bev_mask = (
                                (gt_boxes_eval[:, 0] >= BEV_X_MIN) &
                                (gt_boxes_eval[:, 0] <= BEV_X_MAX) &
                                (gt_boxes_eval[:, 1] >= BEV_Y_MIN) &
                                (gt_boxes_eval[:, 1] <= BEV_Y_MAX)
                            )
                            in_bev_idx = torch.where(in_bev_mask)[0]
                            N_in_bev = len(in_bev_idx)

                            if N_in_bev > 0 and EVAL_USE_AGENT_LOCAL_TRAJ:
                                gt_traj_inbev  = gt_traj[:N_eval][in_bev_idx]
                                gt_mask_inbev  = gt_mask[:N_eval][in_bev_idx]
                                gt_boxes_inbev = gt_boxes_eval[in_bev_idx]

                                gt_traj_local = transform_to_agent_local(
                                    gt_traj_inbev, gt_boxes_inbev
                                )

                                F_modes = y_hat.shape[0]
                                pred_local_modes = []
                                for f in range(F_modes):
                                    local_f = transform_to_agent_local(
                                        y_hat[f, in_bev_idx, :, :2],
                                        gt_boxes_inbev
                                    )
                                    pred_local_modes.append(local_f)
                                pred_pos_local = torch.stack(pred_local_modes, dim=0)

                                y_hat_local = y_hat[:, in_bev_idx].clone()
                                y_hat_local[..., :2] = pred_pos_local

                                traj_m = compute_trajectory_metrics(
                                    y_hat=y_hat_local,
                                    pi=(pi[in_bev_idx] if pi is not None else None),
                                    gt_traj=gt_traj_local,
                                    gt_mask=gt_mask_inbev,
                                )
                                all_traj_metric_results.append(traj_m)

            except Exception as e:
                print(f"ERROR in sensor eval batch: {e}")

    print(f"\nCollected {len(all_sample_results)} sample results.")

    # =========================================================================
    # Compute and print sensor metrics — UNCHANGED
    # =========================================================================
    print("\nComputing detection mAP...")
    det_metrics = compute_detection_ap(
        all_sample_results, iou_func, EVAL_USE_ROTATED_IOU
    )

    print("Computing intention metrics...")
    intent_metrics = compute_intention_metrics(
        all_sample_results, iou_func, EVAL_USE_ROTATED_IOU
    )

    traj_metrics = {}
    if run_traj and all_traj_metric_results:
        print("Aggregating auxiliary (oracle) trajectory metrics...")
        traj_metrics = accumulate_trajectory_metrics(all_traj_metric_results)

    all_metrics = {**det_metrics, **intent_metrics, **traj_metrics}
    print_metrics(all_metrics, model_name=f"IntentTrajNet {MODEL_VERSION}")

    if run_traj:
        frame = 'agent-local' if EVAL_USE_AGENT_LOCAL_TRAJ else 'ego'
        print(f"  Auxiliary trajectory frame: {frame} (all agents, sensor GT, "
              f"teacher-forced/oracle detections)")

    # =========================================================================
    # Distance-binned mAP — UNCHANGED
    # =========================================================================
    print("\nDistance-binned mAP@0.5:")
    try:
        dist_map = compute_distance_binned_map(
            all_sample_results, iou_func, EVAL_USE_ROTATED_IOU
        )
        for label, ap in dist_map.items():
            print(f"  mAP@0.5 {label:<10}: {ap:.4f}")
        all_metrics['distance_binned_mAP'] = dist_map   # NEW — kept for saving
    except Exception as e:
        print(f"  Distance-binned mAP failed: {e}")

    # =========================================================================
    # Intention confusion matrix — UNCHANGED
    # =========================================================================
    confusion_matrix_result = None   # NEW — captured for saving
    if SKLEARN_AVAILABLE:
        print("\nIntention Confusion Matrix (rows=GT, cols=Predicted):")
        try:
            from utils.constants import IOU_THRESHOLD_FOR_INTENTION_MATCH
            matched_pred_all = []
            matched_gt_all   = []

            for sample in all_sample_results:
                pred_scores  = sample['pred_scores']
                pred_boxes   = sample['pred_boxes_xywha']
                pred_intents = sample['pred_intentions']
                gt_boxes     = sample['gt_boxes_xywha']
                gt_intents   = sample['gt_intentions']

                if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
                    continue

                if EVAL_USE_ROTATED_IOU and ROTATED_IOU_AVAILABLE:
                    iou_mat = iou_func(pred_boxes.float(), gt_boxes.float())
                else:
                    iou_mat = iou_func(
                        pred_boxes[:, :4].float(), gt_boxes[:, :4].float()
                    )

                gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
                sort_idx   = torch.argsort(pred_scores, descending=True)

                for i in range(pred_boxes.shape[0]):
                    orig_idx = sort_idx[i]
                    ious     = iou_mat[orig_idx, :]
                    if ious.numel() == 0:
                        continue
                    best_iou, best_gt_idx = torch.max(ious, dim=0)
                    if (best_iou >= IOU_THRESHOLD_FOR_INTENTION_MATCH
                            and not gt_matched[best_gt_idx]):
                        gt_matched[best_gt_idx] = True
                        matched_pred_all.append(pred_intents[orig_idx].item())
                        matched_gt_all.append(gt_intents[best_gt_idx].item())

            if matched_pred_all:
                cm = confusion_matrix(
                    matched_gt_all, matched_pred_all,
                    labels=list(range(NUM_INTENTION_CLASSES))
                )
                confusion_matrix_result = cm   # NEW
                class_names = [
                    INTENTIONS_MAP_REV.get(i, f'C{i}')[:8]
                    for i in range(NUM_INTENTION_CLASSES)
                ]
                header = f"{'':>12}" + ''.join(f'{n:>9}' for n in class_names)
                print(header)
                for i, row in enumerate(cm):
                    row_str = ''.join(f'{v:>9}' for v in row)
                    print(f"  {class_names[i]:<10}: {row_str}")
        except Exception as e:
            print(f"  Confusion matrix failed: {e}")

    # =========================================================================
    # ## NEW — End-to-end (non-oracle) trajectory evaluation
    # Independent second pass over val_loader. See function docstring above.
    # =========================================================================
    e2e_traj_metrics = {}
    e2e_traj_raw = []
    if EVAL_END_TO_END_TRAJ and use_trajectory and saved_decoder_type in ('mlp', 'social_mlp'):
        print(f"\n{'='*60}")
        print(f"  End-to-End Trajectory Evaluation (predicted detections)")
        print(f"{'='*60}")
        e2e_traj_metrics, e2e_traj_raw = evaluate_trajectory_end_to_end(
            model=model,
            val_loader=val_loader,
            anchors=anchors,
            device=DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            use_rotated_nms=EVAL_USE_ROTATED_NMS,
            use_rotated_iou_for_matching=EVAL_USE_ROTATED_IOU,
            use_agent_local_traj=EVAL_USE_AGENT_LOCAL_TRAJ,
        )
        if e2e_traj_metrics:
            all_metrics['e2e_minADE']    = e2e_traj_metrics['minADE']
            all_metrics['e2e_minFDE']    = e2e_traj_metrics['minFDE']
            all_metrics['e2e_MR']        = e2e_traj_metrics['MR']
            all_metrics['e2e_N_vehicles'] = e2e_traj_metrics['N_vehicles']

    # =========================================================================
    # Parquet trajectory evaluation — for V4/V5, also used for V3 re-eval
    # UNCHANGED
    # =========================================================================
    parquet_metrics = {}
    if USE_PARQUET_EVAL:
        print(f"\n{'='*60}")
        print(f"  Parquet Trajectory Evaluation (MF Protocol)")
        print(f"{'='*60}")
        parquet_metrics = evaluate_trajectory_parquet(
            model=model,
            parquet_val_dir=PARQUET_VAL_DIR,
            sensor_val_dir=VAL_DATA_DIR,
            device=DEVICE,
            use_agent_local=EVAL_USE_AGENT_LOCAL_TRAJ,
            eval_focal_only=EVAL_FOCAL_ONLY,
            eval_all_agents=EVAL_ALL_AGENTS,
        )
        all_metrics.update(parquet_metrics)

    # =========================================================================
    # Computational analysis — UNCHANGED (full-model latency @ batch=1)
    # =========================================================================
    print("\nComputational Analysis:")
    try:
        model.eval()
        dummy_lidar = torch.randn(
            1, LIDAR_TOTAL_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX
        ).to(DEVICE)
        dummy_map = torch.randn(1, MAP_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params/1e6:.1f}M")

        if THOP_AVAILABLE:
            macs, _ = thop_profile(
                model, inputs=(dummy_lidar, dummy_map), verbose=False
            )
            print(f"  MACs:       {macs/1e9:.1f}G")
        else:
            print("  MACs:       thop not installed (pip install thop)")

        times = []
        with torch.no_grad():
            for _ in range(5):
                model(dummy_lidar, dummy_map)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            for _ in range(50):
                t_start = time.time()
                model(dummy_lidar, dummy_map)
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - t_start)

        mean_ms = np.mean(times) * 1000
        std_ms  = np.std(times)  * 1000
        print(f"  Latency:    {mean_ms:.1f}ms ± {std_ms:.1f}ms "
              f"(batch=1, {DEVICE.type})")

    except Exception as e:
        print(f"  Computational analysis failed: {e}")

    # =========================================================================
    # ## NEW — Agent-density latency / memory sweep (trajectory head only)
    # =========================================================================
    agent_density_results = {}
    if BENCHMARK_AGENT_DENSITY and use_trajectory and saved_decoder_type in ('mlp', 'social_mlp'):
        print(f"\n{'='*60}")
        print(f"  Agent-Density Benchmark (trajectory head only)")
        print(f"{'='*60}")
        agent_density_results = benchmark_trajectory_head_by_agent_count(
            model=model,
            device=DEVICE,
            agent_counts=AGENT_DENSITY_COUNTS,
        )

    # =========================================================================
    # ## NEW — Save per-sample results for later bootstrap CI computation
    # =========================================================================
    if SAVE_RESULTS_PATH:
        try:
            save_path = Path(SAVE_RESULTS_PATH)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_version':            MODEL_VERSION,
                'config_path':              str(config_path),
                'checkpoint_path':          str(CHECKPOINT_PATH),
                'all_sample_results':       all_sample_results,
                'all_traj_metric_results':  all_traj_metric_results,   # oracle
                'e2e_traj_raw_results':     e2e_traj_raw,              # end-to-end
                'confusion_matrix':         confusion_matrix_result,
                'agent_density_results':    agent_density_results,
                'aggregated_metrics':       all_metrics,
            }, save_path)
            print(f"\nSaved per-sample results for bootstrap CI to: {save_path}")
        except Exception as e:
            print(f"  WARNING: could not save results to {SAVE_RESULTS_PATH}: {e}")

    print(f"\n--- Evaluation Finished [{MODEL_VERSION}] ---")
    return all_metrics


if __name__ == '__main__':
    main_eval()