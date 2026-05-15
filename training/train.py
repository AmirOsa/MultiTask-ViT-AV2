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
#      python training/train.py --config configs/v2_mlp.yaml
#   5. Added trajectory loss logging for V2/V3
#   6. Added trajectory GT extraction from gt_list
#   7. Added differential learning rates for V3 (backbone vs heads)
#   8. All original training logic unchanged (AdamW, ReduceLROnPlateau,
#      NaN detection, progress bar, checkpoint saving)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

# MODIFICATION: import config loader
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
from models.model_mt import IntentNetViT_MT
from models.backbone import BasicBlock
from training.loss import MultiTaskLoss
from utils.utils import generate_anchors


if __name__ == '__main__':

    # =========================================================================
    # Load config from YAML
    # =========================================================================
    config_path = get_config_arg()
    # Usage: python training/train.py --config configs/v2_mlp.yaml
    cfg = load_config(config_path)

    # =========================================================================
    # Read all settings from config
    # =========================================================================

    # Model
    MODEL_VERSION = get_nested(cfg, 'model', 'version', default='V2')
    USE_TRAJECTORY = get_nested(cfg, 'model', 'use_trajectory', default=False)

    # Backbone
    backbone_type = get_nested(cfg, 'model', 'backbone', 'type', default='vit')
    pretrained = get_nested(cfg, 'model', 'backbone', 'pretrained', default=False)
    # For V3 Swin backbone
    swin_model_name = get_nested(
        cfg, 'model', 'backbone', 'swin_model_name',
        default='swin_tiny_patch4_window7_224'
    )
    window_size = get_nested(cfg, 'model', 'backbone', 'window_size', default=5)

    # For V1/V2 ViT backbone
    vit_model_name = get_nested(
        cfg, 'model', 'backbone', 'vit_model_name_lidar',
        default='vit_small_patch8_224'
    )
    pretrained_lidar = get_nested(
        cfg, 'model', 'backbone', 'pretrained_lidar', default=False
    )
    pretrained_map = get_nested(
        cfg, 'model', 'backbone', 'pretrained_map', default=False
    )

    # Data
    TRAIN_DATA_DIR = get_nested(cfg, 'data', 'train_dir', default='')
    USE_FUTURE_TRAJ = get_nested(cfg, 'data', 'future_traj', default=False)

    # Training
    TRAIN_BATCH_SIZE = get_nested(cfg, 'training', 'batch_size', default=8)
    NUM_EPOCHS = get_nested(cfg, 'training', 'num_epochs', default=10)
    NUM_WORKERS = get_nested(cfg, 'training', 'num_workers', default=0)

    # Optimizer — V3 uses differential LR for pretrained backbone
    LR_BACKBONE = get_nested(
        cfg, 'training', 'optimizer', 'lr_backbone', default=None
    )
    LR_HEADS = get_nested(
        cfg, 'training', 'optimizer', 'lr_heads', default=None
    )
    LR = get_nested(
        cfg, 'training', 'optimizer', 'lr', default=1e-4
    )
    WEIGHT_DECAY = get_nested(
        cfg, 'training', 'optimizer', 'weight_decay', default=1e-4
    )

    # Loss
    USE_ROTATED_IOU = get_nested(cfg, 'loss', 'use_rotated_iou', default=False)
    APPLY_DOWNSAMPLE = get_nested(
        cfg, 'loss', 'apply_intention_downsampling', default=True
    )
    DOWNSAMPLE_RATIO = get_nested(
        cfg, 'loss', 'intention_downsample_ratio',
        default=INTENTION_DOWNSAMPLE_RATIO
    )
    BOX_WEIGHT = get_nested(cfg, 'loss', 'box_weight', default=1.0)
    CLS_WEIGHT = get_nested(cfg, 'loss', 'cls_weight', default=1.0)
    INTENT_WEIGHT = get_nested(cfg, 'loss', 'intent_weight', default=0.5)
    TRAJ_LAMBDA = get_nested(cfg, 'loss', 'traj_lambda', default=TRAJECTORY_LAMBDA)

    # Checkpoints
    MODEL_SAVE_DIR = get_nested(
        cfg, 'checkpoints', 'save_dir',
        default='/content/drive/MyDrive/Amir_Dataset/IntentTrajNet_checkpoints'
    )
    SAVE_FILENAME = get_nested(
        cfg, 'checkpoints', 'filename',
        default=f'intenttrajnet_{MODEL_VERSION.lower()}.pth'
    )
    PRETRAINED_CHECKPOINT = get_nested(
        cfg, 'checkpoints', 'pretrained_v1', default=''
    ) or get_nested(cfg, 'checkpoints', 'pretrained_v2', default='')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Build backbone config dict
    # =========================================================================
    if backbone_type == 'swin':
        # V3 — Swin backbone config
        BACKBONE_CFG = {
            'type': 'swin',
            'swin_model_name': swin_model_name,
            'pretrained': pretrained,
            'window_size': window_size,
            'out_channels': get_nested(
                cfg, 'model', 'backbone', 'out_channels', default=512
            ),
            'img_size': (GRID_HEIGHT_PX, GRID_WIDTH_PX),
            'lidar_input_channels': LIDAR_TOTAL_CHANNELS,
            'map_input_channels': MAP_CHANNELS,
        }
        # Feature map stride for Swin with patch4
        FEATURE_MAP_STRIDE = 8
        # We upsample Swin output to 50×90 which corresponds to stride 8
    else:
        # V1/V2 — ViT backbone config
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
                cfg, 'model', 'backbone', 'lidar_adapter_out_channels',
                default=192
            ),
            'map_adapter_out_channels': get_nested(
                cfg, 'model', 'backbone', 'map_adapter_out_channels',
                default=192
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
        # Compute feature map stride from ViT patch size
        try:
            vit_patch_stride = int(
                vit_model_name.split('_patch')[-1].split('_')[0]
            )
        except ValueError:
            vit_patch_stride = 8
        FEATURE_MAP_STRIDE = (
            vit_patch_stride * BACKBONE_CFG.get('fusion_block_stride', 1)
        )

    # =========================================================================
    # Print configuration summary
    # =========================================================================
    print(f"\n{'='*55}")
    print(f"  IntentTrajNet-AV2 Training — {MODEL_VERSION}")
    print(f"  Config: {config_path}")
    print(f"{'='*55}")
    print(f"  Device:            {DEVICE}")
    print(f"  Backbone:          {backbone_type}")
    print(f"  Use trajectory:    {USE_TRAJECTORY}")
    if USE_TRAJECTORY:
        print(f"  Traj lambda:       {TRAJ_LAMBDA} (NEEDS TEST)")
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
    # Dataset and DataLoader
    # =========================================================================
    train_data_path = Path(TRAIN_DATA_DIR)
    if not train_data_path.is_dir():
        print(f"ERROR: Training data not found: {TRAIN_DATA_DIR}")
        exit()

    print("Initializing training dataset...")
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
            print("ERROR: DataLoader is empty.")
            exit()
        print(f"DataLoader: {len(train_loader)} batches.\n")
    except Exception as e:
        print(f"ERROR initializing dataset: {e}")
        exit()

    # =========================================================================
    # Model
    # =========================================================================
    print("Initializing model...")
    model = IntentNetViT_MT(
        backbone_cfg=BACKBONE_CFG,
        use_trajectory=USE_TRAJECTORY,
    ).to(DEVICE)

    # Load pretrained backbone if specified
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
    # V3 uses differential LR: lower for pretrained backbone, higher for heads
    # SOURCED: standard practice for fine-tuning pretrained models
    # =========================================================================
    if LR_BACKBONE and LR_HEADS and backbone_type == 'swin':
        # Separate parameter groups for backbone vs heads
        backbone_params = list(model.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        head_params = [
            p for p in model.parameters()
            if id(p) not in backbone_ids
        ]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': LR_BACKBONE},
            {'params': head_params,    'lr': LR_HEADS},
        ], weight_decay=WEIGHT_DECAY)
        print(
            f"Optimizer: AdamW with differential LR "
            f"(backbone={LR_BACKBONE}, heads={LR_HEADS})"
        )
    else:
        # Single LR for all parameters (V1 and V2)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        print(f"Optimizer: AdamW lr={LR}")

    # SOURCED: ReduceLROnPlateau — Nadeem thesis Section 3.4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # =========================================================================
    # Anchors
    # SOURCED: generate_anchors() — Nadeem's original
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
    # Resume from checkpoint if specified                                              #remove this block after finishing training
    # =========================================================================
    RESUME_CHECKPOINT = get_nested(cfg, 'checkpoints', 'resume', default='')
    start_epoch = 0

    if RESUME_CHECKPOINT and Path(RESUME_CHECKPOINT).is_file():
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print(f"Resuming from epoch {start_epoch + 1}\n")
    else:
        print("Starting from scratch.\n")

    # =========================================================================
    # Training loop
    # =========================================================================
    print(f"--- Starting Training [{MODEL_VERSION}] ---\n")

    for epoch in range(start_epoch, NUM_EPOCHS):                                 #remove (start_epoch,) after finishing training
        model.train()

        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_box_loss = 0.0
        epoch_intent_loss = 0.0
        epoch_traj_loss = 0.0
        batches_done = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [{MODEL_VERSION}]",
            unit="batch"
        )

        for batch_idx, batch_data in enumerate(pbar):
            if batch_data is None:
                continue

            lidar_bev = batch_data["lidar_bev"].to(DEVICE, non_blocking=True)
            map_bev = batch_data["map_bev"].to(DEVICE, non_blocking=True)
            gt_list = batch_data["gt_list"]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                lidar_bev, map_bev,
                gt_list=gt_list,
                use_gt_boxes_for_traj=True
            )

            det_cls_logits = outputs["det_cls_logits"]
            det_box_preds = outputs["det_box_preds"]
            intention_logits = outputs["intention_logits"]
            y_hat = outputs.get("y_hat")
            pi = outputs.get("pi")

            # NaN check
            if (torch.isnan(det_cls_logits).any() or
                    torch.isnan(det_box_preds).any() or
                    torch.isnan(intention_logits).any()):
                print(f"Warning: NaN in model output at batch {batch_idx+1}. Skipping.")
                continue

            # Extract trajectory GT for V2/V3
            gt_traj = None
            gt_mask = None
            if USE_TRAJECTORY and y_hat is not None:
                if gt_list[0] is not None and 'future_traj_ego' in gt_list[0]:
                    gt_traj = gt_list[0]['future_traj_ego'].to(DEVICE)
                    gt_mask = gt_list[0]['future_traj_mask'].to(DEVICE)

            # Loss
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

            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx+1}. Skipping.")
                continue

            loss.backward()
            optimizer.step()

            # Accumulate
            epoch_loss += loss.item()
            epoch_cls_loss += loss_dict["cls_loss"].item()
            epoch_box_loss += loss_dict["box_loss"].item()
            epoch_intent_loss += loss_dict["intent_loss"].item()
            traj_l = loss_dict.get("traj_loss", 0.0)
            epoch_traj_loss += (
                traj_l.item() if isinstance(traj_l, torch.Tensor) else traj_l
            )
            batches_done += 1

            # Progress bar
            postfix = {
                'Loss': f"{loss.item():.4f}",
                'Cls':  f"{loss_dict['cls_loss'].item():.3f}",
                'Box':  f"{loss_dict['box_loss'].item():.3f}",
                'Int':  f"{loss_dict['intent_loss'].item():.3f}",
            }
            if USE_TRAJECTORY:
                traj_val = traj_l.item() if isinstance(traj_l, torch.Tensor) else traj_l
                postfix['Traj'] = f"{traj_val:.3f}"
            num_pos = loss_dict.get('num_pos_anchors', 'N/A')
            postfix['#Pos'] = (
                num_pos.item() if isinstance(num_pos, torch.Tensor) else num_pos
            )
            pbar.set_postfix(postfix)

        # Epoch summary
        if batches_done > 0:
            avg_loss   = epoch_loss   / batches_done
            avg_cls    = epoch_cls_loss   / batches_done
            avg_box    = epoch_box_loss   / batches_done
            avg_intent = epoch_intent_loss / batches_done
            avg_traj   = epoch_traj_loss  / batches_done

            summary = (
                f"Epoch {epoch+1} | "
                f"Loss={avg_loss:.4f} "
                f"Cls={avg_cls:.4f} "
                f"Box={avg_box:.4f} "
                f"Intent={avg_intent:.4f}"
            )
            if USE_TRAJECTORY:
                summary += f" Traj={avg_traj:.4f}"
            summary += f" | LR={optimizer.param_groups[0]['lr']:.1e}"
            print(summary)

            scheduler.step(avg_loss)

            # ── NEW: save checkpoint after every epoch ──────────────────
            save_dir = Path(MODEL_SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            epoch_save_path = save_dir / f"MultiTask_V2_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_version': MODEL_VERSION,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'backbone_cfg': BACKBONE_CFG,
                'use_trajectory': USE_TRAJECTORY,
                'traj_lambda': TRAJ_LAMBDA,
                'config_path': str(config_path),
            }, epoch_save_path)
            print(f"Checkpoint saved: {epoch_save_path}")
            # ────────────────────────────────────────────────────────────

        else:
            print(f"Epoch {epoch+1}: No batches processed.")

    # =========================================================================
    # Save checkpoint
    # =========================================================================
    print("\n--- Training Finished ---")
    save_dir = Path(MODEL_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / SAVE_FILENAME

    torch.save({
        'epoch': NUM_EPOCHS,
        'model_version': MODEL_VERSION,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'backbone_cfg': BACKBONE_CFG,
        'use_trajectory': USE_TRAJECTORY,
        'traj_lambda': TRAJ_LAMBDA,
        'config_path': str(config_path),
    }, save_path)

    print(f"Saved: {save_path}")