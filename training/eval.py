# training/eval.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
# Original file: eval_vit.py
#
# Modifications:
#   1. Updated all import paths to match new repo structure
#   2. Replaced IntentNetViT with IntentNetViT_MT
#   3. Added YAML config loading
#   4. Added trajectory metrics (minADE, minFDE, MR) for V2/V3
#   5. Extracted metrics to utils/metrics.py
#   6. All original detection and intention eval logic unchanged
#   7. [NEW] Agent-local frame transformation for trajectory metrics
#      SOURCED: Abdulbaki thesis Section 3.6
#   8. [NEW] Rotated NMS flag — passes use_rotated to apply_nms()
#      SOURCED: Nadeem thesis Section 5.2.3
#   9. [NEW] Computational analysis — FLOPs, parameters, latency
#      SOURCED: Nadeem thesis Section 5.2.4
#  10. [NEW] Confusion matrix for intention prediction error analysis
#  11. [NEW] Distance-binned mAP at 0-20m, 20-40m, 40-60m, 60-100m

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
)
from datasets.av2_dataset import ArgoverseIntentNetDataset, collate_fn
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
# [NEW] Agent-local frame transformation
# SOURCED: Abdulbaki thesis Section 3.6 — per-agent rotation matrix
# =============================================================================

def transform_to_agent_local(
    traj_ego: torch.Tensor,
    boxes_xywha: torch.Tensor,
) -> torch.Tensor:
    """
    Transform ego-frame trajectory positions to agent-local frame.

    Each agent's trajectory is expressed relative to its own current
    position and heading. This makes displacement errors independent
    of how far the vehicle is from the ego vehicle.

    Makes trajectory metrics directly comparable to:
      - Abdulbaki HiVT baseline (agent-local frame, minADE=1.56m)
      - V3 (trained natively in agent-local frame)

    SOURCED: Abdulbaki thesis Section 3.6
      "Positions are transformed into a local coordinate system by
       subtracting the origin and applying a per-agent rotation matrix
       R_i, computed based on each agent's heading at t=49."

    Args:
        traj_ego:    [N, T, 2] — future positions in ego frame (x, y)
        boxes_xywha: [N, 5]   — current GT boxes (cx, cy, w, l, heading)

    Returns:
        [N, T, 2] — positions in each agent's local frame
                    origin    = agent current position
                    x-axis    = agent current heading direction
    """
    N       = traj_ego.shape[0]
    cx      = boxes_xywha[:, 0]   # [N]
    cy      = boxes_xywha[:, 1]   # [N]
    heading = boxes_xywha[:, 4]   # [N]

    # Step 1 — translate: subtract each agent's current position
    agent_pos = torch.stack([cx, cy], dim=-1).unsqueeze(1)  # [N, 1, 2]
    relative  = traj_ego - agent_pos                        # [N, T, 2]

    # Step 2 — rotate: by negative heading into agent's local frame
    cos_h = torch.cos(-heading).view(N, 1)   # [N, 1]
    sin_h = torch.sin(-heading).view(N, 1)   # [N, 1]

    local_x = cos_h * relative[..., 0] - sin_h * relative[..., 1]
    local_y = sin_h * relative[..., 0] + cos_h * relative[..., 1]

    return torch.stack([local_x, local_y], dim=-1)  # [N, T, 2]


# =============================================================================
# [NEW] Distance-binned mAP
# =============================================================================

def compute_distance_binned_map(all_sample_results, iou_func,
                                 use_rotated, bins=None):
    """
    Compute mAP@0.5 broken down by GT box distance from ego vehicle.

    Helps identify whether the model struggles at close or far ranges.

    Args:
        all_sample_results: list of sample result dicts
        iou_func:           IoU function to use
        use_rotated:        whether using rotated IoU
        bins:               list of (min_m, max_m, label) tuples

    Returns:
        dict: {label: ap_value}
    """
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

        # Distance of each GT box from ego (ego is at origin in ego frame)
        gt_dist = torch.sqrt(
            gt_boxes[:, 0]**2 + gt_boxes[:, 1]**2
        )  # [N_gt]

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
                iou_matrix = iou_func(
                    pred_boxes.float(), gt_boxes_bin.float()
                )
            else:
                iou_matrix = iou_func(
                    pred_boxes[:, :4].float(),
                    gt_boxes_bin[:, :4].float()
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
            precision = tp_cumsum / (
                torch.arange(1, num_pred + 1).float() + 1e-9
            )

            from utils.utils import calculate_ap
            ap = calculate_ap(recall.numpy(), precision.numpy())
            results_per_bin[label].append(ap)

    return {
        label: float(np.mean(aps)) if aps else 0.0
        for label, aps in results_per_bin.items()
    }


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

    VAL_DATA_DIR = get_nested(cfg, 'data', 'val_dir', default='')

    CHECKPOINT_PATH = get_nested(
        cfg, 'checkpoints', 'save_dir',
        default='/content/drive/MyDrive/Amir_Dataset/IntentTrajNet_checkpoints'
    ) + '/' + get_nested(
        cfg, 'checkpoints', 'filename',
        default=f'intenttrajnet_{MODEL_VERSION.lower()}.pth'
    )

    CONFIDENCE_THRESHOLD = get_nested(
        cfg, 'eval', 'confidence_threshold', default=0.1
    )
    NMS_IOU_THRESHOLD = get_nested(
        cfg, 'eval', 'nms_iou_threshold', default=0.2
    )
    EVAL_USE_ROTATED_IOU = get_nested(
        cfg, 'eval', 'use_rotated_iou', default=False
    )
    # [NEW] Rotated NMS — SOURCED: Nadeem thesis Section 5.2.3
    EVAL_USE_ROTATED_NMS = get_nested(
        cfg, 'eval', 'use_rotated_nms', default=False
    )
    # [NEW] Agent-local trajectory frame
    # SOURCED: Abdulbaki thesis Section 3.6
    # True  → comparable to HiVT and V3
    # False → original ego-frame (for reference only)
    EVAL_USE_AGENT_LOCAL_TRAJ = get_nested(
        cfg, 'eval', 'use_agent_local_traj', default=True
    )

    INFERENCE_BATCH_SIZE = get_nested(cfg, 'eval', 'batch_size', default=8)
    NUM_WORKERS          = get_nested(cfg, 'training', 'num_workers', default=0)

    vit_model_name = get_nested(
        cfg, 'model', 'backbone', 'vit_model_name_lidar',
        default='vit_small_patch8_224'
    )
    fusion_stride = get_nested(
        cfg, 'model', 'backbone', 'fusion_block_stride', default=1
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Print configuration
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  IntentTrajNet-AV2 Evaluation — {MODEL_VERSION}")
    print(f"  Config: {config_path}")
    print(f"{'='*60}")
    print(f"  Device:                {DEVICE}")
    print(f"  Val data:              {VAL_DATA_DIR}")
    print(f"  Checkpoint:            {CHECKPOINT_PATH}")
    print(f"  Use trajectory:        {USE_TRAJECTORY}")
    print(f"  Rotated IoU (mAP):     {EVAL_USE_ROTATED_IOU}")
    print(f"  Rotated NMS:           {EVAL_USE_ROTATED_NMS}")
    print(f"  Agent-local traj:      {EVAL_USE_AGENT_LOCAL_TRAJ}")
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
    # Model
    # =========================================================================
    saved_backbone_cfg = checkpoint.get('backbone_cfg', {})
    use_trajectory     = checkpoint.get('use_trajectory', USE_TRAJECTORY)

    saved_backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
    saved_backbone_cfg.setdefault('lidar_input_channels', LIDAR_TOTAL_CHANNELS)
    saved_backbone_cfg.setdefault('map_input_channels', MAP_CHANNELS)
    saved_backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch8_224')
    saved_backbone_cfg.setdefault('vit_model_name_map',   'vit_small_patch8_224')
    saved_backbone_cfg.setdefault('pretrained_lidar', False)
    saved_backbone_cfg.setdefault('pretrained_map',   False)
    saved_backbone_cfg.setdefault('fusion_block_planes', 512)
    saved_backbone_cfg.setdefault('res_block_type', BasicBlock)

    try:
        model = IntentNetViT_MT(
            backbone_cfg=saved_backbone_cfg,
            use_trajectory=use_trajectory,
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded and set to eval mode.\n")
    except Exception as e:
        print(f"ERROR instantiating model: {e}")
        return

    # =========================================================================
    # Dataset
    # =========================================================================
    val_data_path = Path(VAL_DATA_DIR)
    if not val_data_path.is_dir():
        print(f"ERROR: Val data not found: {VAL_DATA_DIR}")
        return

    print("Initializing eval dataset...")
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
        print(f"Val DataLoader: {len(val_loader)} batches.\n")
    except Exception as e:
        print(f"ERROR initializing dataset: {e}")
        return

    # =========================================================================
    # Anchors
    # =========================================================================
    print("Generating anchors...")
    if backbone_type == 'swin':
        FEATURE_MAP_STRIDE = 8
    else:
        try:
            vit_patch_stride = int(
                vit_model_name.split('_patch')[-1].split('_')[0]
            )
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

    # IoU function for mAP and intention matching
    if EVAL_USE_ROTATED_IOU and ROTATED_IOU_AVAILABLE:
        iou_func = compute_rotated_iou
        print("Using rotated IoU for mAP and intention matching.\n")
    else:
        iou_func = compute_axis_aligned_iou
        print("Using axis-aligned IoU for mAP and intention matching.\n")

    if EVAL_USE_ROTATED_NMS and not ROTATED_IOU_AVAILABLE:
        print("WARNING: Rotated NMS requested but rotated IoU unavailable.")
        print("         Falling back to axis-aligned NMS.\n")
        EVAL_USE_ROTATED_NMS = False

    # =========================================================================
    # Inference loop
    # =========================================================================
    print("Running inference...")
    all_sample_results      = []
    all_traj_metric_results = []

    with torch.inference_mode():
        pbar = tqdm(val_loader, desc=f"Eval [{MODEL_VERSION}]", unit="batch")

        for batch_data in pbar:
            if batch_data is None:
                continue

            batch_size  = batch_data["lidar_bev"].shape[0]
            gt_list_cpu = batch_data["gt_list"]

            try:
                lidar_bev = batch_data["lidar_bev"].to(DEVICE, non_blocking=True)
                map_bev   = batch_data["map_bev"].to(DEVICE, non_blocking=True)

                outputs = model(
                    lidar_bev, map_bev,
                    gt_list=gt_list_cpu,
                    use_gt_boxes_for_traj=True
                    # ASSUMED: GT boxes at eval time — upper bound on
                    # trajectory performance.
                    # SOURCED: DeTra (Casas et al. 2024) — standard practice.
                    # Consistent with Abdulbaki's use of GT agent tracks.
                )

                det_cls_logits   = outputs["det_cls_logits"]
                det_box_preds    = outputs["det_box_preds"]
                intention_logits = outputs["intention_logits"]
                y_hat = outputs.get("y_hat")
                pi    = outputs.get("pi")

                # ─────────────────────────────────────────────────────────────
                # Per-sample post-processing
                # SOURCED: Nadeem's eval_vit.py — logic unchanged
                # ─────────────────────────────────────────────────────────────
                for b_idx in range(batch_size):
                    sample_pred = {
                        'pred_scores':      torch.empty(0, device='cpu'),
                        'pred_boxes_xywha': torch.empty((0, 5), device='cpu'),
                        'pred_intentions':  torch.empty(
                            0, dtype=torch.long, device='cpu'
                        )
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

                            # [NEW] Rotated NMS option
                            # SOURCED: Nadeem thesis Section 5.2.3
                            nms_keep = apply_nms(
                                boxes_abs, sc_f, NMS_IOU_THRESHOLD,
                                use_rotated=EVAL_USE_ROTATED_NMS
                            )

                            if nms_keep.numel() > 0:
                                sample_pred['pred_scores'] = (
                                    sc_f[nms_keep].cpu()
                                )
                                sample_pred['pred_boxes_xywha'] = (
                                    boxes_abs[nms_keep].cpu()
                                )
                                sample_pred['pred_intentions'] = (
                                    torch.argmax(
                                        int_f[nms_keep], dim=-1
                                    ).cpu()
                                )
                    except Exception as e:
                        print(f"Error post-processing b_idx={b_idx}: {e}")

                    gt = gt_list_cpu[b_idx]
                    all_sample_results.append({
                        **sample_pred,
                        'gt_boxes_xywha': gt.get(
                            'boxes_xywha', torch.empty((0, 5))
                        ),
                        'gt_intentions': gt.get(
                            'intentions',
                            torch.empty(0, dtype=torch.long)
                        )
                    })

                # ─────────────────────────────────────────────────────────────
                # Trajectory metrics (V2/V3 only)
                # ─────────────────────────────────────────────────────────────
                if use_trajectory and y_hat is not None:
                    gt_b0 = gt_list_cpu[0]
                    if gt_b0 is not None and 'future_traj_ego' in gt_b0:
                        gt_traj  = gt_b0['future_traj_ego'].to(DEVICE)
                        gt_mask  = gt_b0['future_traj_mask'].to(DEVICE)
                        gt_boxes = gt_b0['boxes_xywha'].to(DEVICE)

                        N_pred = y_hat.shape[1]
                        N_gt   = gt_traj.shape[0]
                        N_eval = min(N_pred, N_gt)

                        if N_eval > 0:
                            boxes_ok = (gt_boxes.shape[0] >= N_eval)

                            if EVAL_USE_AGENT_LOCAL_TRAJ and boxes_ok:
                                # ─────────────────────────────────────────────
                                # [NEW] Transform to agent-local frame
                                # SOURCED: Abdulbaki thesis Section 3.6
                                # ─────────────────────────────────────────────

                                # Transform GT trajectory: [N, 60, 2]
                                gt_traj_local = transform_to_agent_local(
                                    gt_traj[:N_eval],
                                    gt_boxes[:N_eval]
                                )

                                # Transform predictions per mode
                                # y_hat: [F, N, 60, 4]
                                F = y_hat.shape[0]
                                pred_local_modes = []
                                for f in range(F):
                                    local_f = transform_to_agent_local(
                                        y_hat[f, :N_eval, :, :2],
                                        gt_boxes[:N_eval]
                                    )
                                    pred_local_modes.append(local_f)

                                pred_pos_local = torch.stack(
                                    pred_local_modes, dim=0
                                )  # [F, N, 60, 2]

                                # Rebuild y_hat with local positions
                                # scale (bx, by) unchanged — only positions
                                y_hat_local = y_hat[:, :N_eval].clone()
                                y_hat_local[..., :2] = pred_pos_local

                                traj_m = compute_trajectory_metrics(
                                    y_hat=y_hat_local,
                                    pi=(pi[:N_eval]
                                        if pi is not None else None),
                                    gt_traj=gt_traj_local,
                                    gt_mask=gt_mask[:N_eval],
                                )
                            else:
                                # Original ego-frame evaluation (fallback)
                                traj_m = compute_trajectory_metrics(
                                    y_hat=y_hat[:, :N_eval],
                                    pi=(pi[:N_eval]
                                        if pi is not None else None),
                                    gt_traj=gt_traj[:N_eval],
                                    gt_mask=gt_mask[:N_eval],
                                )

                            all_traj_metric_results.append(traj_m)

            except Exception as e:
                print(f"ERROR in eval batch: {e}")

    print(f"\nCollected {len(all_sample_results)} sample results.")

    # =========================================================================
    # Compute and print main metrics
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
    if use_trajectory and all_traj_metric_results:
        print("Aggregating trajectory metrics...")
        traj_metrics = accumulate_trajectory_metrics(all_traj_metric_results)

    all_metrics = {**det_metrics, **intent_metrics, **traj_metrics}
    print_metrics(all_metrics, model_name=f"IntentTrajNet {MODEL_VERSION}")

    # Trajectory frame note
    if use_trajectory:
        frame = 'agent-local' if EVAL_USE_AGENT_LOCAL_TRAJ else 'ego'
        print(f"  Trajectory frame: {frame}")
        if EVAL_USE_AGENT_LOCAL_TRAJ:
            print(
                "  Note: agent-local frame is directly comparable to "
                "Abdulbaki HiVT (minADE=1.56m) and V3"
            )

    # =========================================================================
    # [NEW] Distance-binned mAP
    # =========================================================================
    print("\nDistance-binned mAP@0.5:")
    try:
        dist_map = compute_distance_binned_map(
            all_sample_results, iou_func, EVAL_USE_ROTATED_IOU
        )
        for label, ap in dist_map.items():
            print(f"  mAP@0.5 {label:<10}: {ap:.4f}")
    except Exception as e:
        print(f"  Distance-binned mAP failed: {e}")

    # =========================================================================
    # [NEW] Intention confusion matrix
    # =========================================================================
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
                    iou_mat = iou_func(
                        pred_boxes.float(), gt_boxes.float()
                    )
                else:
                    iou_mat = iou_func(
                        pred_boxes[:, :4].float(),
                        gt_boxes[:, :4].float()
                    )

                gt_matched = torch.zeros(
                    gt_boxes.shape[0], dtype=torch.bool
                )
                sort_idx = torch.argsort(pred_scores, descending=True)

                for i in range(pred_boxes.shape[0]):
                    orig_idx = sort_idx[i]
                    ious = iou_mat[orig_idx, :]
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
                class_names = [
                    INTENTIONS_MAP_REV.get(i, f'C{i}')[:8]
                    for i in range(NUM_INTENTION_CLASSES)
                ]
                header = f"{'':>12}" + ''.join(
                    f'{n:>9}' for n in class_names
                )
                print(header)
                for i, row in enumerate(cm):
                    row_str = ''.join(f'{v:>9}' for v in row)
                    print(f"  {class_names[i]:<10}: {row_str}")
        except Exception as e:
            print(f"  Confusion matrix failed: {e}")
    else:
        print("\nSkipping confusion matrix (sklearn not available)")

    # =========================================================================
    # [NEW] Computational analysis
    # SOURCED: Nadeem thesis Section 5.2.4
    # =========================================================================
    print("\nComputational Analysis:")
    try:
        model.eval()
        dummy_lidar = torch.randn(
            1, LIDAR_TOTAL_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX
        ).to(DEVICE)
        dummy_map = torch.randn(
            1, MAP_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX
        ).to(DEVICE)

        # Parameter count (always available)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params/1e6:.1f}M")

        # FLOPs (requires thop)
        if THOP_AVAILABLE:
            macs, _ = thop_profile(
                model, inputs=(dummy_lidar, dummy_map), verbose=False
            )
            print(f"  MACs:       {macs/1e9:.1f}G")
        else:
            print("  MACs:       thop not installed (pip install thop)")

        # Latency measurement
        times = []
        with torch.no_grad():
            # Warm up GPU
            for _ in range(5):
                model(dummy_lidar, dummy_map)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            # Measure
            for _ in range(50):
                t_start = time.time()
                model(dummy_lidar, dummy_map)
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - t_start)

        mean_ms = np.mean(times) * 1000
        std_ms  = np.std(times) * 1000
        print(f"  Latency:    {mean_ms:.1f}ms ± {std_ms:.1f}ms "
              f"(batch=1, {DEVICE.type})")

    except Exception as e:
        print(f"  Computational analysis failed: {e}")

    print(f"\n--- Evaluation Finished [{MODEL_VERSION}] ---")
    return all_metrics


if __name__ == '__main__':
    main_eval()