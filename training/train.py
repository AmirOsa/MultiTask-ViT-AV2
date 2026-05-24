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
#   8. Added decoder_type reading from config (NEW for V4/V5)
#   9. Added dual-dataset training loop (NEW for V4/V5):
#      - Sensor dataloader: det+intent every iteration (922 batches/epoch)
#      - Parquet dataloader: trajectory every ~70 iterations (13 batches/epoch)
#      - Both update shared backbone simultaneously
#  10. Fixed epoch checkpoint naming — MODEL_VERSION not hardcoded V1
#  11. All original training logic unchanged (AdamW, ReduceLROnPlateau,
#      NaN detection, progress bar, checkpoint saving)

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


if __name__ == '__main__':

    # =========================================================================
    # Load config from YAML
    # =========================================================================
    config_path = get_config_arg()
    cfg = load_config(config_path)

    # =========================================================================
    # Read all settings from config
    # =========================================================================

    # Model
    MODEL_VERSION  = get_nested(cfg, 'model', 'version', default='V2')
    USE_TRAJECTORY = get_nested(cfg, 'model', 'use_trajectory', default=False)

    # Backbone
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

    # Decoder type — NEW for V4/V5
    decoder_type = get_nested(cfg, 'model', 'trajectory', 'decoder_type', default='mlp')
    # 'mlp'         → V2, V3, V3-traj
    # 'transformer' → V4, V5

    # Trajectory head hyperparameters (V4/V5 transformer decoder)
    gru_hidden         = get_nested(cfg, 'model', 'trajectory', 'gru_hidden',         default=64)
    num_heads          = get_nested(cfg, 'model', 'trajectory', 'num_heads',           default=8)
    num_decoder_layers = get_nested(cfg, 'model', 'trajectory', 'num_decoder_layers',  default=2)
    social_heads       = get_nested(cfg, 'model', 'trajectory', 'social_heads',        default=4)
    social_layers      = get_nested(cfg, 'model', 'trajectory', 'social_layers',       default=1)
    traj_dropout       = get_nested(cfg, 'model', 'trajectory', 'dropout',             default=0.1)

    # Data
    TRAIN_DATA_DIR   = get_nested(cfg, 'data', 'train_dir',   default='')
    USE_FUTURE_TRAJ  = get_nested(cfg, 'data', 'future_traj', default=False)

    # Parquet data — NEW for V4/V5
    PARQUET_TRAIN_DIR = get_nested(cfg, 'data', 'parquet_train_dir', default='')
    USE_PARQUET       = (decoder_type == 'transformer') and bool(PARQUET_TRAIN_DIR)
    # True only for V4/V5 with transformer decoder and parquet dir configured

    # Training
    TRAIN_BATCH_SIZE = get_nested(cfg, 'training', 'batch_size',   default=8)
    NUM_EPOCHS       = get_nested(cfg, 'training', 'num_epochs',    default=10)
    NUM_WORKERS      = get_nested(cfg, 'training', 'num_workers',   default=0)

    # Optimizer
    LR_BACKBONE  = get_nested(cfg, 'training', 'optimizer', 'lr_backbone', default=None)
    LR_HEADS     = get_nested(cfg, 'training', 'optimizer', 'lr_heads',    default=None)
    LR           = get_nested(cfg, 'training', 'optimizer', 'lr',          default=1e-4)
    WEIGHT_DECAY = get_nested(cfg, 'training', 'optimizer', 'weight_decay',default=1e-4)

    # Loss
    USE_ROTATED_IOU  = get_nested(cfg, 'loss', 'use_rotated_iou',            default=False)
    APPLY_DOWNSAMPLE = get_nested(cfg, 'loss', 'apply_intention_downsampling',default=True)
    DOWNSAMPLE_RATIO = get_nested(
        cfg, 'loss', 'intention_downsample_ratio', default=INTENTION_DOWNSAMPLE_RATIO
    )
    BOX_WEIGHT    = get_nested(cfg, 'loss', 'box_weight',    default=1.0)
    CLS_WEIGHT    = get_nested(cfg, 'loss', 'cls_weight',    default=1.0)
    INTENT_WEIGHT = get_nested(cfg, 'loss', 'intent_weight', default=0.5)
    TRAJ_LAMBDA   = get_nested(cfg, 'loss', 'traj_lambda',   default=TRAJECTORY_LAMBDA)

    # Checkpoints
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
    # Trajectory head config (V4/V5 only)
    # =========================================================================
    TRAJECTORY_HEAD_CFG = {}
    if decoder_type == 'transformer':
        TRAJECTORY_HEAD_CFG = {
            'gru_hidden':         gru_hidden,
            'num_heads':          num_heads,
            'num_decoder_layers': num_decoder_layers,
            'social_heads':       social_heads,
            'social_layers':      social_layers,
            'dropout':            traj_dropout,
        }

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
    # Same as V1/V2/V3 — used for detection + intention training
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
    # Parquet Dataset and DataLoader — NEW for V4/V5
    # Used for trajectory training only
    # =========================================================================
    parquet_loader = None
    parquet_iter   = None
    PARQUET_FREQ   = 1  # use parquet batch every N sensor batches

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
                # Batch size 1 — N varies per scenario
                # Larger batches require padding logic not yet implemented
                shuffle=True,
                num_workers=0,
                collate_fn=parquet_collate_fn,
            )
            if len(parquet_loader) == 0:
                print("ERROR: Parquet DataLoader is empty.")
                exit()

            # How often to inject a parquet batch
            # Target: parquet dataset completes ~1 pass per epoch
            # sensor_batches / parquet_batches = 922 / 13 ≈ 70
            n_sensor = len(train_loader)
            n_parquet = len(parquet_loader)
            PARQUET_FREQ = max(1, n_sensor // n_parquet)
            # Every PARQUET_FREQ sensor batches, inject one parquet batch
            # This ensures both datasets complete ~1 pass per epoch

            parquet_iter = iter(parquet_loader)
            print(
                f"Parquet DataLoader: {n_parquet} scenarios. "
                f"Injecting every {PARQUET_FREQ} sensor batches.\n"
            )
        except Exception as e:
            print(f"ERROR initializing parquet dataset: {e}")
            exit()

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
    # V3/V4/V5: differential LR — lower for pretrained Swin backbone
    # SOURCED: Howard & Ruder (ACL 2018) — discriminative fine-tuning
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

    # SOURCED: ReduceLROnPlateau — Nadeem thesis Section 3.4
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
    # Resume from checkpoint if specified
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

        # Reset parquet iterator each epoch
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

            # =================================================================
            # V1/V2/V3 training path — single dataset, all heads
            # =================================================================
            if not USE_PARQUET:
                # Standard forward pass — all heads run
                outputs = model(
                    lidar_bev, map_bev,
                    gt_list=gt_list,
                    use_gt_boxes_for_traj=True,
                    agent_history=None,
                    run_traj_head=True,
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

                # Extract trajectory GT from sensor annotations
                gt_traj = None
                gt_mask = None
                if USE_TRAJECTORY and y_hat is not None:
                    if gt_list[0] is not None and 'future_traj_ego' in gt_list[0]:
                        gt_traj = gt_list[0]['future_traj_ego'].to(DEVICE)
                        gt_mask = gt_list[0]['future_traj_mask'].to(DEVICE)

                # Combined loss
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

            # =================================================================
            # V4/V5 dual-dataset training path
            # =================================================================
            else:
                # --- Sensor batch: detection + intention only ---
                # run_traj_head=False skips trajectory head for this batch
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

                # Detection + intention loss only
                det_intent_loss_dict = loss_fn.det_intent_loss(
                    cls_logits=det_cls_logits,
                    box_preds=det_box_preds,
                    intention_logits=intention_logits,
                    anchors=anchors,
                    gt_list=gt_list,
                )
                loss = det_intent_loss_dict["loss"]

                # Build loss_dict for logging
                loss_dict = {
                    "loss":           loss,
                    "cls_loss":       det_intent_loss_dict["cls_loss"],
                    "box_loss":       det_intent_loss_dict["box_loss"],
                    "intent_loss":    det_intent_loss_dict["intent_loss"],
                    "traj_loss":      torch.tensor(0.0, device=DEVICE),
                    "num_pos_anchors": det_intent_loss_dict["num_pos_anchors"],
                }

                # --- Parquet batch: trajectory only ---
                # Injected every PARQUET_FREQ sensor batches
                if (batch_idx % PARQUET_FREQ == 0 and
                        parquet_iter is not None):
                    try:
                        parquet_batch = next(parquet_iter)
                    except StopIteration:
                        # Reset parquet iterator when exhausted
                        parquet_iter = iter(parquet_loader)
                        try:
                            parquet_batch = next(parquet_iter)
                        except StopIteration:
                            parquet_batch = None

                    if parquet_batch is not None:
                        try:
                            p_lidar = parquet_batch["lidar_bev"].to(DEVICE)
                            p_map   = parquet_batch["map_bev"].to(DEVICE)

                            # Get focal agent data for batch element 0
                            p_gt_boxes    = parquet_batch["gt_boxes"][0].to(DEVICE)
                            p_history     = parquet_batch["agent_history"][0].to(DEVICE)
                            p_traj_focal  = parquet_batch["gt_traj_focal"][0].to(DEVICE)
                            p_mask_focal  = parquet_batch["gt_mask_focal"][0].to(DEVICE)
                            p_focal_idx   = parquet_batch["focal_idx"][0]

                            if p_gt_boxes.shape[0] > 0:
                                # Forward pass — trajectory head only
                                parquet_outputs = model.forward_traj_only(
                                    lidar_bev=p_lidar,
                                    map_bev=p_map,
                                    gt_boxes=p_gt_boxes,
                                    agent_history=p_history,
                                )

                                y_hat_parquet = parquet_outputs.get("y_hat")
                                pi_parquet    = parquet_outputs.get("pi")

                                if y_hat_parquet is not None and y_hat_parquet.shape[1] > 0:
                                    # Extract focal agent predictions only
                                    # Training loss on focal agent only
                                    # SOURCED: AV2 MF standard protocol
                                    if p_focal_idx < y_hat_parquet.shape[1]:
                                        y_hat_focal = y_hat_parquet[
                                            :, p_focal_idx:p_focal_idx+1, :, :
                                        ]
                                        # [F, 1, 60, 4]
                                        pi_focal = pi_parquet[
                                            p_focal_idx:p_focal_idx+1, :
                                        ]
                                        # [1, F]

                                        gt_traj_focal_batch = p_traj_focal.unsqueeze(0)
                                        # [1, 60, 2]
                                        gt_mask_focal_batch = p_mask_focal.unsqueeze(0)
                                        # [1, 60]

                                        # Trajectory loss on focal agent
                                        traj_loss_out = loss_fn.traj_loss_fn(
                                            y_hat=y_hat_focal,
                                            pi=pi_focal,
                                            gt_traj=gt_traj_focal_batch,
                                            gt_mask=gt_mask_focal_batch,
                                        )
                                        traj_loss = traj_loss_out["loss"]

                                        if not (torch.isnan(traj_loss) or
                                                torch.isinf(traj_loss)):
                                            # Add weighted trajectory loss
                                            # to detection+intention loss
                                            loss = loss + TRAJ_LAMBDA * traj_loss
                                            loss_dict["traj_loss"] = traj_loss.detach()

                        except Exception as e:
                            print(f"Warning: Parquet batch error at batch {batch_idx}: {e}")

            # =================================================================
            # Backward pass — same for both training paths
            # =================================================================
            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx+1}. Skipping.")
                continue

            loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_loss        += loss.item()
            epoch_cls_loss    += loss_dict["cls_loss"].item()
            epoch_box_loss    += loss_dict["box_loss"].item()
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
            epoch_save_path = save_dir / f"MultiTask_{MODEL_VERSION}_epoch{epoch+1}.pth"
            torch.save({
                'epoch':               epoch + 1,
                'model_version':       MODEL_VERSION,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'backbone_cfg':        BACKBONE_CFG,
                'use_trajectory':      USE_TRAJECTORY,
                'decoder_type':        decoder_type,
                'traj_lambda':         TRAJ_LAMBDA,
                'config_path':         str(config_path),
            }, epoch_save_path)
            print(f"Checkpoint saved: {epoch_save_path}")

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
        'epoch':               NUM_EPOCHS,
        'model_version':       MODEL_VERSION,
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'backbone_cfg':        BACKBONE_CFG,
        'use_trajectory':      USE_TRAJECTORY,
        'decoder_type':        decoder_type,
        'traj_lambda':         TRAJ_LAMBDA,
        'config_path':         str(config_path),
    }, save_path)

    print(f"Saved: {save_path}")