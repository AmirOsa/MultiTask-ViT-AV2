from __future__ import annotations
# training/train.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
# Original file: train_vit.py
#
# Modifications:
#   1. Updated all import paths to match new repo structure
#   2. Replaced IntentNetViT with IntentNetViT_MT
#   3. Replaced DetectionIntentionLoss with MultiTaskLoss
#   4. Added YAML config loading — run with:
#      python training/train.py --config configs/v4_transformer_traj.yaml
#   5. Added trajectory loss logging for V2/V3
#   6. Added trajectory GT extraction from gt_list
#   7. Added differential learning rates for V3/V4/V5 (backbone vs heads)
#   8. Added decoder_type reading from config (for V4/V5)
#   9. Added dual-dataset training loop (for V4/V5):
#      - Sensor dataloader: det+intent every iteration (922 batches/epoch)
#      - Parquet dataloader: trajectory every ~70 iterations (13 batches/epoch)
#      - Both update shared backbone simultaneously
#  10. Fixed epoch checkpoint naming — MODEL_VERSION not hardcoded V1
#  11. All original training logic unchanged (AdamW, ReduceLROnPlateau,
#      NaN detection, progress bar, checkpoint saving)
#  12. Added parquet val trajectory loss monitoring per epoch
#      to detect overfitting early (train vs val gap tracking)
#  13. MODIFICATION 13 — Full batch trajectory supervision fix:
#      Previously only gt_list[0] contributed trajectory loss (922 scenes/epoch)
#      Now all B scenes contribute via padding and masking (7,375 scenes/epoch)
#      — same as detection and intention. Standard approach used in literature.
#      Three changes: collate_fn in av2_dataset.py builds
#      padded tensors [B, N_max, H, 2] with agent_mask [B, N_max]; model
#      forward() loops over all B scenes; train.py uses padded tensors for GT.
#
#      NOTE FOR REPRODUCIBILITY:
#      The results reported in the submitted thesis (June 2026) were produced
#      with single-scene trajectory supervision (gt_list[0] only), giving the
#      trajectory head 922 scenes/epoch vs 7,375 for detection and intention.
#      Training with this implementation will produce different results:
#        - Trajectory metrics (minADE, minFDE, MR) will improve
#        - Detection/intention behaviour depends on decoder architecture
#      To reproduce original thesis results exactly, replace the trajectory
#      GT block below with:
#        gt_traj_ego = gt_list[0]['future_traj_ego'].to(DEVICE)
#        gt_mask     = gt_list[0]['future_traj_mask'].to(DEVICE)
#        gt_boxes    = gt_list[0]['boxes_xywha'].to(DEVICE)
#      and remove padded tensor keys from collate_fn in av2_dataset.py.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.config_loader import load_config, get_config_arg, get_nested
from utils.constants import (
    GRID_HEIGHT_PX, GRID_WIDTH_PX,
    NUM_INTENTION_CLASSES,
    ANCHOR_CONFIGS_PAPER,
    DOMINANT_CLASSES_FOR_DOWNSAMPLING,
    INTENTION_DOWNSAMPLE_RATIO,
    LIDAR_TOTAL_CHANNELS, MAP_CHANNELS,
    TRAJECTORY_LAMBDA,
    TRAJECTORY_FUTURE_STEPS,
    TRAJECTORY_NUM_MODES,
)
from datasets.av2_dataset import ArgoverseIntentNetDataset, collate_fn
from datasets.parquet_dataset import ParquetTrajectoryDataset, parquet_collate_fn
from models.model_mt import IntentNetViT_MT
from models.backbone import BasicBlock
from training.loss import MultiTaskLoss
from utils.utils import generate_anchors


# =============================================================================
# transform_to_agent_local
# Transforms trajectory GT from ego frame to agent-local frame.
# SOURCED: Abdulbaki thesis Section 3.8
# =============================================================================
def transform_to_agent_local(traj_ego, boxes_xywha):
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
# MODIFICATION 13: collect_batch_boxes_for_forward
# RETAINED FOR REFERENCE — no longer called in training loop
#
# This was an intermediate implementation of full batch trajectory supervision
# using concatenation across scenes. It has been superseded by the padding
# and masking approach in collate_fn (av2_dataset.py) which is standard
# in literature.
# =============================================================================
def collect_batch_boxes_for_forward(gt_list, device):
    """
    Collect GT boxes from all B scenes for trajectory head forward pass.

    Returns:
        all_boxes:          [N_total, 5] — all GT boxes across all B scenes
        scene_offsets:      list of (start, end) index pairs per scene
        feature_map_b_list: list of ints — which batch element each vehicle belongs to
    """
    all_boxes_list     = []
    scene_offsets      = []
    feature_map_b_list = []
    offset = 0

    for b, gt in enumerate(gt_list):
        if gt is None or gt['boxes_xywha'].shape[0] == 0:
            scene_offsets.append((offset, offset))
            continue

        boxes = gt['boxes_xywha'].to(device)  # [N_b, 5]
        N_b   = boxes.shape[0]

        all_boxes_list.append(boxes)
        feature_map_b_list.extend([b] * N_b)
        scene_offsets.append((offset, offset + N_b))
        offset += N_b

    if not all_boxes_list:
        return None, scene_offsets, []

    all_boxes = torch.cat(all_boxes_list, dim=0)  # [N_total, 5]
    return all_boxes, scene_offsets, feature_map_b_list


# =============================================================================
# MODIFICATION 13: collect_batch_trajectory_gt
# RETAINED FOR REFERENCE — no longer called in training loop
#
# This was an intermediate implementation of full batch trajectory supervision
# using concatenation across scenes. It has been superseded by the padding
# and masking approach in collate_fn (av2_dataset.py) which is standard
# in literature. The padding and masking approach is implemented directly
# in the training loop below using batch_data["traj_padded"] and
# batch_data["agent_mask"].
# =============================================================================
def collect_batch_trajectory_gt(gt_list, device, future_steps=TRAJECTORY_FUTURE_STEPS):
    """
    Collect trajectory GT from all B scenes in a batch.

    Previously only gt_list[0] was used, giving 8x less trajectory
    supervision than detection and intention. This function collects
    GT from all B scenes so every scene contributes every batch.

    SOURCED: padding approach standard in literature.

    Returns:
        all_gt_traj_local: [N_total, H, 2] — agent-local trajectories, all scenes
        all_gt_mask:       [N_total, H]    — validity mask
        all_boxes:         [N_total, 5]    — boxes for all active vehicles
        N_total:           total active vehicles across all B scenes (0 if none)
    """
    all_traj_list  = []
    all_mask_list  = []
    all_boxes_list = []

    PARKED_CLASS       = 6
    MIN_DISPLACEMENT_M = 0.5

    for b, gt in enumerate(gt_list):
        if gt is None:
            continue
        if 'future_traj_ego' not in gt:
            continue

        traj_ego   = gt['future_traj_ego'].to(device)   # [N_b, H, 2]
        traj_mask  = gt['future_traj_mask'].to(device)   # [N_b, H]
        boxes      = gt['boxes_xywha'].to(device)        # [N_b, 5]
        intentions = gt['intentions'].to(device)         # [N_b]

        N_b = traj_ego.shape[0]
        if N_b == 0:
            continue

        # Exclude parked vehicles
        intent_mask = (intentions != PARKED_CLASS)

        # Exclude barely-moving vehicles (< 0.5m displacement)
        displacements = (
            traj_ego.norm(dim=-1) * traj_mask.float()
        ).max(dim=-1).values
        disp_mask = displacements > MIN_DISPLACEMENT_M

        moving_mask = intent_mask & disp_mask

        if not moving_mask.any():
            continue

        traj_ego_filtered  = traj_ego[moving_mask]   # [N_active, H, 2]
        traj_mask_filtered = traj_mask[moving_mask]  # [N_active, H]
        boxes_filtered     = boxes[moving_mask]       # [N_active, 5]

        # Transform ego frame → agent-local frame
        # SOURCED: Abdulbaki thesis Section 3.8
        traj_local = transform_to_agent_local(traj_ego_filtered, boxes_filtered)
        # [N_active, H, 2]

        all_traj_list.append(traj_local)
        all_mask_list.append(traj_mask_filtered)
        all_boxes_list.append(boxes_filtered)

    if not all_traj_list:
        return None, None, None, 0

    all_gt_traj_local = torch.cat(all_traj_list,  dim=0)
    all_gt_mask       = torch.cat(all_mask_list,  dim=0)
    all_boxes         = torch.cat(all_boxes_list, dim=0)
    N_total           = all_gt_traj_local.shape[0]

    return all_gt_traj_local, all_gt_mask, all_boxes, N_total


# =============================================================================
# evaluate_val_traj_loss
# Computes trajectory loss on parquet val set to detect overfitting.
# Called at end of each epoch when USE_PARQUET is True.
# =============================================================================
def evaluate_val_traj_loss(
    model,
    parquet_val_loader,
    loss_fn,
    device,
    decoder_type,
):
    """
    Compute mean trajectory loss on parquet val scenarios.
    Used to monitor train vs val gap for overfitting detection.

    Returns:
        mean val trajectory loss (float), or None if no valid batches
    """
    model.eval()
    val_traj_losses = []

    with torch.no_grad():
        for val_batch in parquet_val_loader:
            if val_batch is None:
                continue
            try:
                p_lidar      = val_batch["lidar_bev"].to(device)
                p_map        = val_batch["map_bev"].to(device)
                p_gt_boxes   = val_batch["gt_boxes"][0].to(device)
                p_history    = val_batch["agent_history"][0].to(device)
                p_traj_focal = val_batch["gt_traj_focal"][0].to(device)
                p_mask_focal = val_batch["gt_mask_focal"][0].to(device)
                p_focal_idx  = val_batch["focal_idx"][0]

                if p_gt_boxes.shape[0] == 0:
                    continue

                parquet_outputs = model.forward_traj_only(
                    lidar_bev=p_lidar,
                    map_bev=p_map,
                    gt_boxes=p_gt_boxes,
                    agent_history=p_history,
                )

                y_hat = parquet_outputs.get("y_hat")
                pi    = parquet_outputs.get("pi")

                if y_hat is None or y_hat.shape[1] == 0:
                    continue

                if p_focal_idx >= y_hat.shape[1]:
                    continue

                y_hat_focal = y_hat[:, p_focal_idx:p_focal_idx+1, :, :]
                pi_focal    = pi[p_focal_idx:p_focal_idx+1, :]
                gt_traj     = p_traj_focal.unsqueeze(0)
                gt_mask     = p_mask_focal.unsqueeze(0)

                focal_box     = p_gt_boxes[p_focal_idx:p_focal_idx+1]
                gt_traj_local = transform_to_agent_local(gt_traj, focal_box)

                traj_out = loss_fn.traj_loss_fn(
                    y_hat=y_hat_focal,
                    pi=pi_focal,
                    gt_traj=gt_traj_local,
                    gt_mask=gt_mask,
                )

                loss_val = traj_out["loss"].item()
                if not (np.isnan(loss_val) or np.isinf(loss_val)):
                    val_traj_losses.append(loss_val)

            except Exception:
                continue

    model.train()

    if val_traj_losses:
        return float(np.mean(val_traj_losses))
    return None


if __name__ == '__main__':

    # =========================================================================
    # Load config from YAML
    # =========================================================================
    config_path = get_config_arg()
    cfg = load_config(config_path)

    # =========================================================================
    # Read all settings from config
    # =========================================================================
    MODEL_VERSION  = get_nested(cfg, 'model', 'version', default='V2')
    USE_TRAJECTORY = get_nested(cfg, 'model', 'use_trajectory', default=False)

    backbone_type    = get_nested(cfg, 'model', 'backbone', 'type', default='vit')
    pretrained       = get_nested(cfg, 'model', 'backbone', 'pretrained', default=False)
    swin_model_name  = get_nested(
        cfg, 'model', 'backbone', 'swin_model_name',
        default='swin_tiny_patch4_window7_224'
    )
    window_size      = get_nested(cfg, 'model', 'backbone', 'window_size', default=5)
    vit_model_name   = get_nested(
        cfg, 'model', 'backbone', 'vit_model_name_lidar',
        default='vit_small_patch8_224'
    )
    pretrained_lidar = get_nested(cfg, 'model', 'backbone', 'pretrained_lidar', default=False)
    pretrained_map   = get_nested(cfg, 'model', 'backbone', 'pretrained_map',   default=False)

    decoder_type = get_nested(cfg, 'model', 'trajectory', 'decoder_type', default='mlp')

    gru_hidden         = get_nested(cfg, 'model', 'trajectory', 'gru_hidden',         default=64)
    num_heads          = get_nested(cfg, 'model', 'trajectory', 'num_heads',           default=8)
    num_decoder_layers = get_nested(cfg, 'model', 'trajectory', 'num_decoder_layers',  default=2)
    social_heads       = get_nested(cfg, 'model', 'trajectory', 'social_heads',        default=4)
    social_layers      = get_nested(cfg, 'model', 'trajectory', 'social_layers',       default=1)
    traj_dropout       = get_nested(cfg, 'model', 'trajectory', 'dropout',             default=0.1)

    TRAIN_DATA_DIR   = get_nested(cfg, 'data', 'train_dir',   default='')
    VAL_DATA_DIR     = get_nested(cfg, 'data', 'val_dir',     default='')
    USE_FUTURE_TRAJ  = get_nested(cfg, 'data', 'future_traj', default=False)

    PARQUET_TRAIN_DIR = get_nested(cfg, 'data', 'parquet_train_dir', default='')
    PARQUET_VAL_DIR   = get_nested(cfg, 'data', 'parquet_val_dir',   default='')

    USE_PARQUET = bool(PARQUET_TRAIN_DIR)

    TRAIN_BATCH_SIZE = get_nested(cfg, 'training', 'batch_size',   default=8)
    NUM_EPOCHS       = get_nested(cfg, 'training', 'num_epochs',    default=10)
    NUM_WORKERS      = get_nested(cfg, 'training', 'num_workers',   default=0)

    LR_BACKBONE  = get_nested(cfg, 'training', 'optimizer', 'lr_backbone', default=None)
    LR_HEADS     = get_nested(cfg, 'training', 'optimizer', 'lr_heads',    default=None)
    LR           = get_nested(cfg, 'training', 'optimizer', 'lr',          default=1e-4)
    WEIGHT_DECAY = get_nested(cfg, 'training', 'optimizer', 'weight_decay',default=1e-4)

    USE_ROTATED_IOU  = get_nested(cfg, 'loss', 'use_rotated_iou',            default=False)
    APPLY_DOWNSAMPLE = get_nested(cfg, 'loss', 'apply_intention_downsampling',default=True)
    DOWNSAMPLE_RATIO = get_nested(
        cfg, 'loss', 'intention_downsample_ratio', default=INTENTION_DOWNSAMPLE_RATIO
    )
    BOX_WEIGHT    = get_nested(cfg, 'loss', 'box_weight',    default=1.0)
    CLS_WEIGHT    = get_nested(cfg, 'loss', 'cls_weight',    default=1.0)
    INTENT_WEIGHT = get_nested(cfg, 'loss', 'intent_weight', default=0.5)
    TRAJ_LAMBDA   = get_nested(cfg, 'loss', 'traj_lambda',   default=TRAJECTORY_LAMBDA)

    MODEL_SAVE_DIR = get_nested(
        cfg, 'checkpoints', 'save_dir',
        default='/content/drive/MyDrive/Bachelor Thesis/Checkpoints'
    )
    SAVE_FILENAME = get_nested(
        cfg, 'checkpoints', 'filename',
        default=f'MultiTask_{MODEL_VERSION}.pth'
    )
    PRETRAINED_CHECKPOINT = get_nested(
        cfg, 'checkpoints', 'pretrained_v1', default=''
    ) or get_nested(cfg, 'checkpoints', 'pretrained_v2', default='') \
      or get_nested(cfg, 'checkpoints', 'pretrained_v3', default='')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Build backbone config dict
    # =========================================================================
    if backbone_type == 'swin':
        BACKBONE_CFG = {
            'type': 'swin',
            'swin_model_name': swin_model_name,
            'pretrained': pretrained,
            'window_size': window_size,
            'out_channels': get_nested(cfg, 'model', 'backbone', 'out_channels', default=512),
            'img_size': (GRID_HEIGHT_PX, GRID_WIDTH_PX),
            'lidar_input_channels': LIDAR_TOTAL_CHANNELS,
            'map_input_channels': MAP_CHANNELS,
        }
        FEATURE_MAP_STRIDE = 8
    else:
        BACKBONE_CFG = {
            'lidar_input_channels': LIDAR_TOTAL_CHANNELS,
            'map_input_channels': MAP_CHANNELS,
            'vit_model_name_lidar': vit_model_name,
            'vit_model_name_map': get_nested(
                cfg, 'model', 'backbone', 'vit_model_name_map',
                default='vit_small_patch8_224'
            ),
            'pretrained_lidar': pretrained_lidar,
            'pretrained_map': pretrained_map,
            'img_size': (GRID_HEIGHT_PX, GRID_WIDTH_PX),
            'drop_path_rate_lidar': get_nested(
                cfg, 'model', 'backbone', 'drop_path_rate_lidar', default=0.1
            ),
            'drop_path_rate_map': get_nested(
                cfg, 'model', 'backbone', 'drop_path_rate_map', default=0.1
            ),
            'lidar_adapter_out_channels': get_nested(
                cfg, 'model', 'backbone', 'lidar_adapter_out_channels', default=192
            ),
            'map_adapter_out_channels': get_nested(
                cfg, 'model', 'backbone', 'map_adapter_out_channels', default=192
            ),
            'fusion_block_planes': get_nested(
                cfg, 'model', 'backbone', 'fusion_block_planes', default=512
            ),
            'fusion_block_layers': get_nested(
                cfg, 'model', 'backbone', 'fusion_block_layers', default=2
            ),
            'fusion_block_kernel_size': get_nested(
                cfg, 'model', 'backbone', 'fusion_block_kernel_size', default=3
            ),
            'fusion_block_stride': get_nested(
                cfg, 'model', 'backbone', 'fusion_block_stride', default=1
            ),
            'res_block_type': BasicBlock
        }
        try:
            vit_patch_stride = int(vit_model_name.split('_patch')[-1].split('_')[0])
        except ValueError:
            vit_patch_stride = 8
        FEATURE_MAP_STRIDE = (
            vit_patch_stride * BACKBONE_CFG.get('fusion_block_stride', 1)
        )

    # =========================================================================
    # Trajectory head config
    # =========================================================================
    TRAJECTORY_HEAD_CFG = {}
    if USE_TRAJECTORY:
        TRAJECTORY_HEAD_CFG = {
            'mlp_dropout': get_nested(
                cfg, 'model', 'trajectory', 'mlp_dropout', default=0.0
            ),
            'box_feat_dim': 5,  # cx, cy, w, l, heading 
        }
    if decoder_type == 'transformer':
        TRAJECTORY_HEAD_CFG.update({
            'gru_hidden':         gru_hidden,
            'num_heads':          num_heads,
            'num_decoder_layers': num_decoder_layers,
            'social_heads':       social_heads,
            'social_layers':      social_layers,
            'dropout':            traj_dropout,
        })

    # =========================================================================
    # Print configuration summary
    # =========================================================================
    print(f"\n{'='*55}")
    print(f"  IntentTrajNet-AV2 Training — {MODEL_VERSION}")
    print(f"  Config: {config_path}")
    print(f"{'='*55}")
    print(f"  Device:            {DEVICE}")
    print(f"  Backbone:          {backbone_type}")
    print(f"  Decoder:           {decoder_type}")
    print(f"  Use trajectory:    {USE_TRAJECTORY}")
    print(f"  Use parquet:       {USE_PARQUET}")
    if USE_TRAJECTORY:
        print(f"  Traj lambda:       {TRAJ_LAMBDA}")
    print(f"  Batch size:        {TRAIN_BATCH_SIZE}")
    print(f"  Epochs:            {NUM_EPOCHS}")
    if LR_BACKBONE:
        print(f"  LR backbone:       {LR_BACKBONE}")
        print(f"  LR heads:          {LR_HEADS}")
    else:
        print(f"  LR:                {LR}")
    print(f"  Feature map stride:{FEATURE_MAP_STRIDE}")
    print(f"{'='*55}\n")

    # =========================================================================
    # Sensor Dataset and DataLoader
    # =========================================================================
    train_data_path = Path(TRAIN_DATA_DIR)
    if not train_data_path.is_dir():
        print(f"ERROR: Training data not found: {TRAIN_DATA_DIR}")
        exit()

    print("Initializing sensor training dataset...")
    try:
        train_dataset = ArgoverseIntentNetDataset(
            data_dir=TRAIN_DATA_DIR, is_train=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=(DEVICE.type == 'cuda')
        )
        if len(train_loader) == 0:
            print("ERROR: Sensor DataLoader is empty.")
            exit()
        print(f"Sensor DataLoader: {len(train_loader)} batches.\n")
    except Exception as e:
        print(f"ERROR initializing sensor dataset: {e}")
        exit()

    # =========================================================================
    # Parquet Train Dataset and DataLoader
    # =========================================================================
    parquet_loader = None
    parquet_iter   = None
    PARQUET_FREQ   = 1

    if USE_PARQUET:
        parquet_dir = Path(PARQUET_TRAIN_DIR)
        if not parquet_dir.is_dir():
            print(f"ERROR: Parquet dir not found: {PARQUET_TRAIN_DIR}")
            exit()

        print("Initializing parquet trajectory dataset...")
        try:
            parquet_dataset = ParquetTrajectoryDataset(
                parquet_dir=PARQUET_TRAIN_DIR,
                sensor_dir=TRAIN_DATA_DIR,
                is_train=True,
            )
            parquet_loader = DataLoader(
                parquet_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                collate_fn=parquet_collate_fn,
            )
            if len(parquet_loader) == 0:
                print("ERROR: Parquet DataLoader is empty.")
                exit()

            n_sensor  = len(train_loader)
            n_parquet = len(parquet_loader)
            PARQUET_FREQ = max(1, n_sensor // n_parquet)

            parquet_iter = iter(parquet_loader)
            print(
                f"Parquet DataLoader: {n_parquet} scenarios. "
                f"Injecting every {PARQUET_FREQ} sensor batches.\n"
            )
        except Exception as e:
            print(f"ERROR initializing parquet dataset: {e}")
            exit()

    # =========================================================================
    # Parquet Val Dataset and DataLoader
    # Used for overfitting detection — val trajectory loss per epoch
    # =========================================================================
    parquet_val_loader = None

    if USE_PARQUET and PARQUET_VAL_DIR and USE_TRAJECTORY:
        parquet_val_path = Path(PARQUET_VAL_DIR)
        if parquet_val_path.is_dir():
            print("Initializing parquet val dataset for overfitting monitoring...")
            try:
                parquet_val_dataset = ParquetTrajectoryDataset(
                    parquet_dir=PARQUET_VAL_DIR,
                    sensor_dir=VAL_DATA_DIR,
                    is_train=False,
                )
                parquet_val_loader = DataLoader(
                    parquet_val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=parquet_collate_fn,
                )
                print(
                    f"Parquet Val DataLoader: "
                    f"{len(parquet_val_loader)} scenarios.\n"
                )
            except Exception as e:
                print(f"WARNING: Could not load parquet val dataset: {e}")
                parquet_val_loader = None
        else:
            print(
                f"WARNING: parquet_val_dir not found: {PARQUET_VAL_DIR}. "
                f"Overfitting monitoring disabled.\n"
            )

    # =========================================================================
    # Model
    # =========================================================================
    print("Initializing model...")
    model = IntentNetViT_MT(
        backbone_type=backbone_type,
        backbone_cfg=BACKBONE_CFG,
        use_trajectory=USE_TRAJECTORY,
        decoder_type=decoder_type,
        trajectory_head_cfg=TRAJECTORY_HEAD_CFG,
    ).to(DEVICE)

    if PRETRAINED_CHECKPOINT and Path(PRETRAINED_CHECKPOINT).is_file():
        print(f"Loading pretrained backbone from: {PRETRAINED_CHECKPOINT}")
        model.load_pretrained_backbone(PRETRAINED_CHECKPOINT)

    print(f"Model initialized.\n")

    # =========================================================================
    # Loss function
    # =========================================================================
    print("Initializing loss function...")
    loss_fn = MultiTaskLoss(
        use_rotated_iou=USE_ROTATED_IOU,
        apply_intention_downsampling=APPLY_DOWNSAMPLE,
        box_weight=BOX_WEIGHT,
        cls_weight=CLS_WEIGHT,
        intent_weight=INTENT_WEIGHT,
        traj_lambda=TRAJ_LAMBDA,
        use_trajectory_loss=USE_TRAJECTORY,
    ).to(DEVICE)
    print(f"Loss initialized.\n")

    # =========================================================================
    # Optimizer
    # =========================================================================
    if LR_BACKBONE and LR_HEADS and backbone_type == 'swin':
        backbone_params = list(model.backbone.parameters())
        backbone_ids    = set(id(p) for p in backbone_params)
        head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': LR_BACKBONE},
            {'params': head_params,     'lr': LR_HEADS},
        ], weight_decay=WEIGHT_DECAY)
        print(
            f"Optimizer: AdamW differential LR "
            f"(backbone={LR_BACKBONE}, heads={LR_HEADS})"
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        print(f"Optimizer: AdamW lr={LR}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # =========================================================================
    # Anchors
    # =========================================================================
    print("\nGenerating anchors...")
    anchors = generate_anchors(
        bev_height=GRID_HEIGHT_PX,
        bev_width=GRID_WIDTH_PX,
        feature_map_stride=FEATURE_MAP_STRIDE,
        anchor_configs=ANCHOR_CONFIGS_PAPER
    ).to(DEVICE)
    print(f"Anchors: {anchors.shape}\n")

    # =========================================================================
    # Resume from checkpoint
    # =========================================================================
    RESUME_CHECKPOINT = get_nested(cfg, 'checkpoints', 'resume', default='')
    start_epoch = 0

    if RESUME_CHECKPOINT and Path(RESUME_CHECKPOINT).is_file():
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        print(f"Resuming from epoch {start_epoch + 1}\n")
    else:
        print("Starting from scratch.\n")

    # =========================================================================
    # Overfitting tracking history
    # =========================================================================
    train_traj_history = []
    val_traj_history   = []

    # =========================================================================
    # Training loop
    # =========================================================================
    print(f"--- Starting Training [{MODEL_VERSION}] ---\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()

        epoch_loss         = 0.0
        epoch_cls_loss     = 0.0
        epoch_box_loss     = 0.0
        epoch_intent_loss  = 0.0
        epoch_traj_loss    = 0.0
        batches_done       = 0

        if USE_PARQUET and parquet_loader is not None:
            parquet_iter = iter(parquet_loader)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [{MODEL_VERSION}]",
            unit="batch"
        )

        for batch_idx, batch_data in enumerate(pbar):
            if batch_data is None:
                continue

            lidar_bev = batch_data["lidar_bev"].to(DEVICE, non_blocking=True)
            map_bev   = batch_data["map_bev"].to(DEVICE, non_blocking=True)
            gt_list   = batch_data["gt_list"]

            optimizer.zero_grad()

            if not USE_PARQUET:
                outputs = model(
                    lidar_bev, map_bev,
                    gt_list=gt_list,
                    use_gt_boxes_for_traj=True,
                    agent_history=None,
                    run_traj_head=True,
                    boxes_padded=batch_data["boxes_padded"].to(DEVICE) if "boxes_padded" in batch_data else None,
                    agent_mask=batch_data["agent_mask"].to(DEVICE) if "agent_mask" in batch_data else None,
                )

                det_cls_logits   = outputs["det_cls_logits"]
                det_box_preds    = outputs["det_box_preds"]
                intention_logits = outputs["intention_logits"]
                y_hat = outputs.get("y_hat")
                pi    = outputs.get("pi")

                if (torch.isnan(det_cls_logits).any() or
                        torch.isnan(det_box_preds).any() or
                        torch.isnan(intention_logits).any()):
                    print(f"Warning: NaN in model output at batch {batch_idx+1}. Skipping.")
                    continue

                gt_traj = None
                gt_mask = None
                if USE_TRAJECTORY and y_hat is not None:
                    # ==========================================================
                    # MODIFICATION 13: full batch trajectory supervision
                    # using padding and masking — standard in literature
                    #
                    # NOTE FOR REPRODUCIBILITY: thesis results (June 2026) used
                    # single-scene supervision (gt_list[0] only). See file header
                    # for full reproducibility instructions.
                    # ==========================================================
                    gt_traj = None
                    gt_mask = None

                    if "traj_padded" in batch_data:
                        traj_padded      = batch_data["traj_padded"].to(DEVICE)      # [B, N_max, H, 2]
                        traj_mask_padded = batch_data["traj_mask_padded"].to(DEVICE) # [B, N_max, H]
                        agent_mask_batch = batch_data["agent_mask"].to(DEVICE)       # [B, N_max]
                        boxes_padded_b   = batch_data["boxes_padded"].to(DEVICE)     # [B, N_max, 5]
                        intentions_batch = batch_data["intentions_padded"].to(DEVICE)# [B, N_max]

                        B_     = traj_padded.shape[0]
                        N_max_ = traj_padded.shape[1]
                        H_     = traj_padded.shape[2]

                        # Flatten batch dimension
                        traj_flat       = traj_padded.view(B_ * N_max_, H_, 2)
                        traj_mask_flat  = traj_mask_padded.view(B_ * N_max_, H_)
                        boxes_flat      = boxes_padded_b.view(B_ * N_max_, 5)
                        intentions_flat = intentions_batch.view(B_ * N_max_)
                        agent_mask_flat = agent_mask_batch.view(B_ * N_max_)

                        # Keep only real vehicles — exclude padding
                        traj_real       = traj_flat[agent_mask_flat]
                        traj_mask_real  = traj_mask_flat[agent_mask_flat]
                        boxes_real      = boxes_flat[agent_mask_flat]
                        intentions_real = intentions_flat[agent_mask_flat]

                        N_real = traj_real.shape[0]

                        if N_real > 0:
                            # Filter: exclude parked and barely-moving vehicles
                            PARKED_CLASS       = 6
                            MIN_DISPLACEMENT_M = 0.5

                            intent_mask   = (intentions_real != PARKED_CLASS)
                            displacements = (
                                traj_real.norm(dim=-1) * traj_mask_real.float()
                            ).max(dim=-1).values
                            disp_mask   = displacements > MIN_DISPLACEMENT_M
                            moving_mask = intent_mask & disp_mask

                            if moving_mask.any():
                                # Transform ego frame → agent-local frame
                                # SOURCED: Abdulbaki thesis Section 3.8
                                gt_traj = transform_to_agent_local(
                                    traj_real[moving_mask],
                                    boxes_real[moving_mask]
                                )
                                gt_mask = traj_mask_real[moving_mask]

                                # Align N between y_hat and GT
                                N_pred = y_hat.shape[1]
                                N_gt   = gt_traj.shape[0]
                                N_min  = min(N_pred, N_gt)

                                if N_min > 0:
                                    gt_traj = gt_traj[:N_min]
                                    gt_mask = gt_mask[:N_min]
                                    y_hat   = y_hat[:, :N_min]
                                    if pi is not None:
                                        pi = pi[:N_min]
                                else:
                                    gt_traj = None
                                    gt_mask = None

                loss_dict = loss_fn(
                    cls_logits=det_cls_logits,
                    box_preds=det_box_preds,
                    intention_logits=intention_logits,
                    anchors=anchors,
                    gt_list=gt_list,
                    y_hat=y_hat,
                    pi=pi,
                    gt_traj=gt_traj,
                    gt_mask=gt_mask,
                )
                loss = loss_dict["loss"]

            else:
                sensor_outputs = model.forward_det_intent_only(
                    lidar_bev=lidar_bev,
                    map_bev=map_bev,
                    gt_list=gt_list,
                )

                det_cls_logits   = sensor_outputs["det_cls_logits"]
                det_box_preds    = sensor_outputs["det_box_preds"]
                intention_logits = sensor_outputs["intention_logits"]

                if (torch.isnan(det_cls_logits).any() or
                        torch.isnan(det_box_preds).any() or
                        torch.isnan(intention_logits).any()):
                    print(f"Warning: NaN in sensor outputs at batch {batch_idx+1}. Skipping.")
                    continue

                det_intent_loss_dict = loss_fn.det_intent_loss(
                    cls_logits=det_cls_logits,
                    box_preds=det_box_preds,
                    intention_logits=intention_logits,
                    anchors=anchors,
                    gt_list=gt_list,
                )
                loss = det_intent_loss_dict["loss"]

                loss_dict = {
                    "loss":            loss,
                    "cls_loss":        det_intent_loss_dict["cls_loss"],
                    "box_loss":        det_intent_loss_dict["box_loss"],
                    "intent_loss":     det_intent_loss_dict["intent_loss"],
                    "traj_loss":       torch.tensor(0.0, device=DEVICE),
                    "num_pos_anchors": det_intent_loss_dict["num_pos_anchors"],
                }

                if (batch_idx % PARQUET_FREQ == 0 and
                        parquet_iter is not None):
                    try:
                        parquet_batch = next(parquet_iter)
                    except StopIteration:
                        parquet_iter = iter(parquet_loader)
                        try:
                            parquet_batch = next(parquet_iter)
                        except StopIteration:
                            parquet_batch = None

                    if parquet_batch is not None:
                        try:
                            p_lidar      = parquet_batch["lidar_bev"].to(DEVICE)
                            p_map        = parquet_batch["map_bev"].to(DEVICE)
                            p_gt_boxes   = parquet_batch["gt_boxes"][0].to(DEVICE)
                            p_history    = parquet_batch["agent_history"][0].to(DEVICE)
                            p_traj_focal = parquet_batch["gt_traj_focal"][0].to(DEVICE)
                            p_mask_focal = parquet_batch["gt_mask_focal"][0].to(DEVICE)
                            p_focal_idx  = parquet_batch["focal_idx"][0]

                            if p_gt_boxes.shape[0] > 0:
                                parquet_outputs = model.forward_traj_only(
                                    lidar_bev=p_lidar,
                                    map_bev=p_map,
                                    gt_boxes=p_gt_boxes,
                                    agent_history=p_history,
                                )

                                y_hat_parquet = parquet_outputs.get("y_hat")
                                pi_parquet    = parquet_outputs.get("pi")

                                if (y_hat_parquet is not None and
                                        y_hat_parquet.shape[1] > 0):
                                    if p_focal_idx < y_hat_parquet.shape[1]:
                                        y_hat_focal = y_hat_parquet[
                                            :, p_focal_idx:p_focal_idx+1, :, :
                                        ]
                                        pi_focal = pi_parquet[
                                            p_focal_idx:p_focal_idx+1, :
                                        ]
                                        gt_traj_focal_batch = p_traj_focal.unsqueeze(0)
                                        gt_mask_focal_batch = p_mask_focal.unsqueeze(0)
                                        focal_box = p_gt_boxes[
                                            p_focal_idx:p_focal_idx+1
                                        ]
                                        gt_traj_local = transform_to_agent_local(
                                            gt_traj_focal_batch,
                                            focal_box
                                        )
                                        traj_loss_out = loss_fn.traj_loss_fn(
                                            y_hat=y_hat_focal,
                                            pi=pi_focal,
                                            gt_traj=gt_traj_local,
                                            gt_mask=gt_mask_focal_batch,
                                        )
                                        traj_loss = traj_loss_out["loss"]

                                        if not (torch.isnan(traj_loss) or
                                                torch.isinf(traj_loss)):
                                            loss = loss + TRAJ_LAMBDA * traj_loss
                                            loss_dict["traj_loss"] = traj_loss.detach()

                        except Exception as e:
                            print(
                                f"Warning: Parquet batch error at "
                                f"batch {batch_idx}: {e}"
                            )

            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx+1}. Skipping.")
                continue

            loss.backward()
            optimizer.step()

            epoch_loss        += loss.item()
            epoch_cls_loss    += loss_dict["cls_loss"].item()
            epoch_box_loss    += loss_dict["box_loss"].item()
            epoch_intent_loss += loss_dict["intent_loss"].item()
            traj_l = loss_dict.get("traj_loss", 0.0)
            epoch_traj_loss += (
                traj_l.item() if isinstance(traj_l, torch.Tensor) else traj_l
            )
            batches_done += 1

            postfix = {
                'Loss': f"{loss.item():.4f}",
                'Cls':  f"{loss_dict['cls_loss'].item():.3f}",
                'Box':  f"{loss_dict['box_loss'].item():.3f}",
                'Int':  f"{loss_dict['intent_loss'].item():.3f}",
            }
            if USE_TRAJECTORY or USE_PARQUET:
                traj_val = traj_l.item() if isinstance(traj_l, torch.Tensor) else traj_l
                postfix['Traj'] = f"{traj_val:.3f}"
            num_pos = loss_dict.get('num_pos_anchors', 'N/A')
            postfix['#Pos'] = (
                num_pos.item() if isinstance(num_pos, torch.Tensor) else num_pos
            )
            pbar.set_postfix(postfix)

        # =====================================================================
        # Epoch summary and checkpoint
        # =====================================================================
        if batches_done > 0:
            avg_loss   = epoch_loss        / batches_done
            avg_cls    = epoch_cls_loss    / batches_done
            avg_box    = epoch_box_loss    / batches_done
            avg_intent = epoch_intent_loss / batches_done
            avg_traj   = epoch_traj_loss   / batches_done

            summary = (
                f"Epoch {epoch+1} | "
                f"Loss={avg_loss:.4f} "
                f"Cls={avg_cls:.4f} "
                f"Box={avg_box:.4f} "
                f"Intent={avg_intent:.4f}"
            )
            if USE_TRAJECTORY or USE_PARQUET:
                summary += f" Traj={avg_traj:.4f}"
            summary += f" | LR={optimizer.param_groups[0]['lr']:.1e}"
            print(summary)

            scheduler.step(avg_loss)

            # Save epoch checkpoint
            save_dir = Path(MODEL_SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            epoch_save_path = (
                save_dir / f"MultiTask_{MODEL_VERSION}_epoch{epoch+1}.pth"
            )
            torch.save({
                'epoch':                epoch + 1,
                'model_version':        MODEL_VERSION,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'backbone_cfg':         BACKBONE_CFG,
                'use_trajectory':       USE_TRAJECTORY,
                'decoder_type':         decoder_type,
                'traj_lambda':          TRAJ_LAMBDA,
                'config_path':          str(config_path),
            }, epoch_save_path)
            print(f"Checkpoint saved: {epoch_save_path}")

            # =================================================================
            # Val trajectory loss for overfitting detection
            # =================================================================
            if (USE_PARQUET and
                    parquet_val_loader is not None and
                    USE_TRAJECTORY and
                    loss_fn.traj_loss_fn is not None):

                print("Computing val trajectory loss...")
                val_traj_loss = evaluate_val_traj_loss(
                    model=model,
                    parquet_val_loader=parquet_val_loader,
                    loss_fn=loss_fn,
                    device=DEVICE,
                    decoder_type=decoder_type,
                )

                if val_traj_loss is not None:
                    train_traj_history.append(avg_traj)
                    val_traj_history.append(val_traj_loss)

                    gap  = val_traj_loss - avg_traj
                    flag = " ⚠️  OVERFITTING WARNING" if gap > 1.0 else ""

                    print(
                        f"  Train Traj Loss: {avg_traj:.4f} | "
                        f"Val Traj Loss: {val_traj_loss:.4f} | "
                        f"Gap: {gap:.4f}{flag}"
                    )

                    # Print full history table
                    print("\n  Trajectory Loss History:")
                    print(f"  {'Epoch':>5} | {'Train':>8} | {'Val':>8} | {'Gap':>8}")
                    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
                    for i, (tr, vl) in enumerate(
                        zip(train_traj_history, val_traj_history)
                    ):
                        g    = vl - tr
                        warn = " ⚠️" if g > 1.0 else ""
                        print(
                            f"  {i+1:>5} | {tr:>8.4f} | {vl:>8.4f} | "
                            f"{g:>8.4f}{warn}"
                        )
                    print()

        else:
            print(f"Epoch {epoch+1}: No batches processed.")

    # =========================================================================
    # Save final checkpoint
    # =========================================================================
    print("\n--- Training Finished ---")
    save_dir = Path(MODEL_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / SAVE_FILENAME

    torch.save({
        'epoch':                NUM_EPOCHS,
        'model_version':        MODEL_VERSION,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'backbone_cfg':         BACKBONE_CFG,
        'use_trajectory':       USE_TRAJECTORY,
        'decoder_type':         decoder_type,
        'traj_lambda':          TRAJ_LAMBDA,
        'config_path':          str(config_path),
    }, save_path)

    print(f"Saved: {save_path}")