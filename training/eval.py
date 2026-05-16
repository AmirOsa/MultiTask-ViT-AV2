# training/eval.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
# Original file: eval_vit.py
#
# Modifications:
#   1. Updated all import paths to match new repo structure
#   2. Replaced IntentNetViT with IntentNetViT_MT
#   3. Added YAML config loading — run with:
#      python training/eval.py --config configs/v2_mlp.yaml
#   4. Added trajectory metrics (minADE, minFDE, MR) for V2/V3
#   5. Extracted metrics to utils/metrics.py
#   6. All original detection and intention eval logic unchanged

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from utils.config_loader import load_config, get_config_arg, get_nested
from utils.constants import (
    GRID_HEIGHT_PX, GRID_WIDTH_PX,
    ANCHOR_CONFIGS_PAPER,
    LIDAR_TOTAL_CHANNELS, MAP_CHANNELS,
    TRAJECTORY_FUTURE_STEPS,
    TRAJECTORY_NUM_MODES,
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


def main_eval():
    """Main evaluation function for IntentTrajNet-AV2."""

    # =========================================================================
    # Load config
    # =========================================================================
    config_path = get_config_arg()
    # Usage: python training/eval.py --config configs/v2_mlp.yaml
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
    INFERENCE_BATCH_SIZE = get_nested(cfg, 'eval', 'batch_size', default=8)
    NUM_WORKERS = get_nested(cfg, 'training', 'num_workers', default=0)

    # Backbone details for anchor generation
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
    print(f"\n{'='*55}")
    print(f"  IntentTrajNet-AV2 Evaluation — {MODEL_VERSION}")
    print(f"  Config: {config_path}")
    print(f"{'='*55}")
    print(f"  Device:          {DEVICE}")
    print(f"  Val data:        {VAL_DATA_DIR}")
    print(f"  Checkpoint:      {CHECKPOINT_PATH}")
    print(f"  Use trajectory:  {USE_TRAJECTORY}")
    print(f"{'='*55}\n")

    # =========================================================================
    # Load checkpoint
    # =========================================================================
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.is_file():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return

    print(f"Loading checkpoint...")
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
    use_trajectory = checkpoint.get('use_trajectory', USE_TRAJECTORY)

    # Apply defaults
    saved_backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
    saved_backbone_cfg.setdefault('lidar_input_channels', LIDAR_TOTAL_CHANNELS)
    saved_backbone_cfg.setdefault('map_input_channels', MAP_CHANNELS)
    saved_backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch8_224')
    saved_backbone_cfg.setdefault('vit_model_name_map', 'vit_small_patch8_224')
    saved_backbone_cfg.setdefault('pretrained_lidar', False)
    saved_backbone_cfg.setdefault('pretrained_map', False)
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

    # IoU function
    if EVAL_USE_ROTATED_IOU and ROTATED_IOU_AVAILABLE:
        iou_func = compute_rotated_iou
        print("Using rotated IoU.\n")
    else:
        iou_func = compute_axis_aligned_iou
        print("Using axis-aligned IoU.\n")

    # =========================================================================
    # Inference loop
    # =========================================================================
    print("Running inference...")
    all_sample_results = []
    all_traj_metric_results = []

    with torch.inference_mode():
        pbar = tqdm(val_loader, desc=f"Eval [{MODEL_VERSION}]", unit="batch")

        for batch_data in pbar:
            if batch_data is None:
                continue

            batch_size = batch_data["lidar_bev"].shape[0]
            gt_list_cpu = batch_data["gt_list"]

            try:
                lidar_bev = batch_data["lidar_bev"].to(DEVICE, non_blocking=True)
                map_bev   = batch_data["map_bev"].to(DEVICE, non_blocking=True)

                outputs = model(
                    lidar_bev, map_bev,
                    gt_list=gt_list_cpu,
                    use_gt_boxes_for_traj=True
                    # ASSUMED: GT boxes at eval time — upper bound on
                    # trajectory performance. See eval.py comments.
                )

                det_cls_logits = outputs["det_cls_logits"]
                det_box_preds  = outputs["det_box_preds"]
                intention_logits = outputs["intention_logits"]
                y_hat = outputs.get("y_hat")
                pi    = outputs.get("pi")

                # Per-sample post-processing
                # SOURCED: Nadeem's eval_vit.py — unchanged
                for b_idx in range(batch_size):
                    sample_pred = {
                        'pred_scores':       torch.empty(0, device='cpu'),
                        'pred_boxes_xywha':  torch.empty((0, 5), device='cpu'),
                        'pred_intentions':   torch.empty(
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
                            nms_keep  = apply_nms(
                                boxes_abs, sc_f, NMS_IOU_THRESHOLD
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
                        print(f"Error post-processing sample: {e}")

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

                # Trajectory metrics (V2/V3 only)
                if use_trajectory and y_hat is not None:
                    gt_b0 = gt_list_cpu[0]
                    if gt_b0 is not None and 'future_traj_ego' in gt_b0:
                        gt_traj = gt_b0['future_traj_ego'].to(DEVICE)
                        gt_mask = gt_b0['future_traj_mask'].to(DEVICE)

                        # Align N between trajectory prediction and GT
                        # y_hat: [F, N_pred, 60, 4]
                        # gt_traj: [N_gt, 60, 2]
                        # N_pred and N_gt may differ due to filtering
                        N_pred = y_hat.shape[1]
                        N_gt   = gt_traj.shape[0]
                        N_eval = min(N_pred, N_gt)

                        if N_eval > 0:
                            traj_m = compute_trajectory_metrics(
                                y_hat=y_hat[:, :N_eval],
                                pi=pi[:N_eval] if pi is not None else None,
                                gt_traj=gt_traj[:N_eval],
                                gt_mask=gt_mask[:N_eval],
                            )
                            all_traj_metric_results.append(traj_m)

            except Exception as e:
                print(f"ERROR in eval batch: {e}")

    print(f"\nCollected {len(all_sample_results)} sample results.")

    # =========================================================================
    # Compute and print metrics
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

    print(f"--- Evaluation Finished [{MODEL_VERSION}] ---")
    return all_metrics


if __name__ == '__main__':
    main_eval()