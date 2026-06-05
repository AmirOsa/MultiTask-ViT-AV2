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
#   7. Agent-local frame transformation for trajectory metrics
#      SOURCED: Abdulbaki thesis Section 3.6
#   8. Rotated NMS flag
#   9. Computational analysis — FLOPs, parameters, latency
#  10. Confusion matrix for intention prediction error analysis
#  11. Distance-binned mAP
#  12. [NEW V4/V5] Parquet evaluation protocol:
#      - Load ParquetTrajectoryDataset for val scenarios
#      - Evaluate trajectory on focal agent only (MF protocol)
#      - Also evaluate on all agents (comparable to V2/V3)
#      - decoder_type read from checkpoint for correct model init
#  13. [NEW V4/V5] V3 re-evaluation on parquet scenarios:
#      - Run any model checkpoint on 48 parquet val scenarios
#      - Match by log_id and timestamp
#      - Report focal-agent minADE for cross-model comparison

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
# Distance-binned mAP
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
        # Only evaluate vehicles the model was trained to detect
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
# Parquet trajectory evaluation — NEW for V4/V5
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

    Used for V4/V5 primary trajectory evaluation (MF protocol).
    Also used for V3 re-evaluation to enable cross-model comparison.

    Evaluates on:
        - Focal agent only (MF protocol, comparable to Abdulbaki)
        - All agents (comparable to V2/V3 auxiliary trajectory)

    Args:
        model:           trained IntentNetViT_MT model
        parquet_val_dir: path to val parquet scenarios (48 scenarios)
        sensor_val_dir:  path to sensor val logs (for BEV loading)
        device:          torch device
        use_agent_local: transform to agent-local frame for metrics
        eval_focal_only: compute focal-agent minADE
        eval_all_agents: compute all-agent minADE

    Returns:
        dict with focal and all-agent trajectory metrics
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

                # Forward pass — trajectory head
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

                # ─────────────────────────────────────────────────────────
                # Focal agent evaluation (MF protocol)
                # ─────────────────────────────────────────────────────────
                if eval_focal_only and focal_idx < N:
                    y_hat_focal = y_hat[:, focal_idx:focal_idx+1, :, :]
                    # [F, 1, 60, 4]
                    pi_focal = pi[focal_idx:focal_idx+1, :]
                    # [1, F]

                    gt_traj_f = gt_traj_focal.unsqueeze(0)
                    # [1, 60, 2]
                    gt_mask_f = gt_mask_focal.unsqueeze(0)
                    # [1, 60]

                    if use_agent_local:
                        # Transform to agent-local frame
                        # SOURCED: Abdulbaki thesis Section 3.6
                        focal_box = gt_boxes[focal_idx:focal_idx+1]
                        # [1, 5]

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
                        # [F, 1, 60, 2]

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

                # ─────────────────────────────────────────────────────────
                # All-agent evaluation (comparable to V2/V3)
                # ─────────────────────────────────────────────────────────
                if eval_all_agents and N > 0:
                    N_eval = min(N, gt_traj_all.shape[0])

                    if use_agent_local and gt_boxes.shape[0] >= N_eval:
                        gt_boxes_eval = gt_boxes[:N_eval]

                        # Filter to in-BEV vehicles
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

    # Parquet eval settings — NEW for V4/V5
    PARQUET_VAL_DIR     = get_nested(cfg, 'data', 'parquet_val_dir',  default='')
    EVAL_FOCAL_ONLY     = get_nested(cfg, 'eval', 'eval_focal_only',  default=False)
    EVAL_ALL_AGENTS     = get_nested(cfg, 'eval', 'eval_all_agents',  default=False)
    USE_PARQUET_EVAL    = bool(PARQUET_VAL_DIR) and (EVAL_FOCAL_ONLY or EVAL_ALL_AGENTS)

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
    # Read decoder_type from checkpoint — important for V4/V5
    # so we init the right decoder even if config doesn't specify it
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

    # Trajectory head config for transformer decoder
    traj_head_cfg = {}
    if use_trajectory:
        traj_head_cfg = {
            'box_feat_dim': 5,  # cx, cy, w, l, heading
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

    # Read backbone_type from checkpoint
    saved_backbone_type = checkpoint.get('backbone_cfg', {}).get('type', backbone_type)
    # Remove 'type' key before passing to constructor
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
    # Sensor inference loop — detection + intention + auxiliary trajectory
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

                # For V4/V5: skip trajectory head in sensor eval
                # Trajectory is evaluated separately via parquet protocol
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

                # Auxiliary trajectory metrics (V2/V3 MLP decoder only)
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
                            # ── Filter to active agents only ──
                            # Step 1: Exclude explicitly parked vehicles
                            # SOURCED: IntentNet (Casas et al. 2018)
                            intentions_eval = gt_b0['intentions'].to(DEVICE)[:N_eval]
                            PARKED_CLASS = 6
                            intent_mask = (intentions_eval != PARKED_CLASS)

                            # Step 2: Exclude barely-moving agents
                            # Must move at least 0.5m over prediction horizon
                            # Automatically catches already-stopped vehicles
                            # SOURCED: DeTra velocity threshold (Casas et al. 2024)
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
    # Compute and print sensor metrics
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
        print("Aggregating auxiliary trajectory metrics...")
        traj_metrics = accumulate_trajectory_metrics(all_traj_metric_results)

    all_metrics = {**det_metrics, **intent_metrics, **traj_metrics}
    print_metrics(all_metrics, model_name=f"IntentTrajNet {MODEL_VERSION}")

    if run_traj:
        frame = 'agent-local' if EVAL_USE_AGENT_LOCAL_TRAJ else 'ego'
        print(f"  Auxiliary trajectory frame: {frame} (all agents, sensor GT)")

    # =========================================================================
    # Distance-binned mAP
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
    # Intention confusion matrix
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
    # Parquet trajectory evaluation — NEW for V4/V5
    # Also used for V3 re-evaluation
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
    # Computational analysis
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

    print(f"\n--- Evaluation Finished [{MODEL_VERSION}] ---")
    return all_metrics


if __name__ == '__main__':
    main_eval()