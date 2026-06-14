# models/model_mt.py
#
# New file for IntentTrajNet-AV2
# Author: Amir — Bachelor Thesis, GUC 2025
#
# Assembles the full multi-task model by combining backbone + heads.
# Supports all five model versions through config:
#
#   V1: backbone='vit', use_trajectory=False
#       Detection + intention only. Identical to Nadeem's IntentNetViT.
#
#   V2: backbone='vit', use_trajectory=True, decoder_type='mlp'
#       Adds MLP trajectory decoder on top of same ViT backbone.
#
#   V3: backbone='swin', use_trajectory=True, decoder_type='mlp'
#       Swaps backbone to Swin-T pretrained. MLP decoder unchanged.
#       ONLY change vs V2 — isolates backbone contribution.
#
#   V3-traj: backbone='swin', use_trajectory=True, decoder_type='mlp'
#       Same as V3 but trained on parquet scenarios.
#       Ablation baseline for V4 trajectory comparison.
#       Confirms MLP decoder performance on same data as V4.
#
#   V4: backbone='swin', use_trajectory=True, decoder_type='transformer'
#       Full trajectory system upgrade:
#         - GRU history encoder [N, 50, 5] → [N, 64]
#         - Social attention across N agents
#         - Transformer cross-attention decoder
#       Dual-dataset training: sensor → det+intent, parquet → trajectory
#       SOURCED: heterogeneous multi-task learning — UniDet, OmniDet
#
#   V5: backbone='swin', use_trajectory=True, decoder_type='transformer'
#       V4 + supervision fixes:
#         - Inverse frequency class weights (C1)
#         - 6s intention horizon (C2)
#         - Velocity-based heading (C3)
#
# Backbone selection:
#   'vit'  → TwoStreamViTBackbone (Nadeem's original, trained from scratch)
#   'swin' → SwinBackbone (Swin-T pretrained ImageNet, new for V3+)
#
# Decoder selection (use_trajectory=True only):
#   'mlp'         → BEVTrajectoryDecoder (V2, V3, V3-traj)
#   'transformer' → TransformerTrajectoryDecoder (V4, V5)
#
# All heads (DetectionHead, IntentionHead, TrajectoryHead) are backbone-agnostic.
# Both backbones output [B, 512, 50, 90] — heads work unchanged for all versions.
#
# Dual-dataset training (V4/V5):
#   Two dataloaders run simultaneously:
#     Sensor dataloader  → 7375 sequences → det+intent loss every iteration
#     Parquet dataloader → 100 scenarios  → trajectory loss every ~70 iterations
#   Both losses update the shared backbone.
#   run_traj_head=False skips trajectory for sensor batches.
#   run_traj_head=True  runs trajectory for parquet batches.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from models.backbone import TwoStreamViTBackbone, SwinBackbone
from models.heads import DetectionHead, IntentionHead, TrajectoryHead
from utils.constants import (
    LIDAR_TOTAL_CHANNELS,
    MAP_CHANNELS,
    GRID_HEIGHT_PX,
    GRID_WIDTH_PX,
    NUM_ANCHORS_PER_LOC,
    NUM_INTENTION_CLASSES,
    TRAJECTORY_FUTURE_STEPS,
    TRAJECTORY_NUM_MODES,
    VOXEL_SIZE_M,
    BEV_PIXEL_OFFSET_X,
    BEV_PIXEL_OFFSET_Y,
    AGENT_HISTORY_STEPS,
    AGENT_HISTORY_FEATURES,
)
from utils.utils import generate_anchors


class IntentNetViT_MT(nn.Module):
    """
    Multi-task model for joint vehicle detection, intention prediction,
    and trajectory forecasting from LiDAR BEV.

    Supports V1 through V5 via backbone_type, decoder_type, and
    use_trajectory flags. All versions share the same heads — only
    the backbone and trajectory decoder change across versions.

    Both backbones output [B, 512, 50, 90] — heads are backbone-agnostic.
    Both decoders output [F, N, 60, 4] — loss and eval code unchanged.

    Dual-dataset training (V4/V5):
        forward_det_intent_only() → sensor batches, det+intent only
        forward_traj_only()       → parquet batches, trajectory only
        Both methods call forward() with appropriate flags.
        Both update the shared backbone through their respective losses.

    Args:
        backbone_type:       'vit' for V1/V2, 'swin' for V3/V4/V5
        backbone_cfg:        dict of kwargs for selected backbone class
        use_trajectory:      False for V1, True for V2-V5
        decoder_type:        'mlp' for V2/V3/V3-traj, 'transformer' for V4/V5
        trajectory_head_cfg: optional kwargs for TrajectoryHead
    """

    def __init__(
        self,
        backbone_type: str = 'vit',
        # 'vit'  → TwoStreamViTBackbone (V1, V2)
        # 'swin' → SwinBackbone          (V3, V4, V5)
        backbone_cfg: dict | None = None,
        use_trajectory: bool = True,
        decoder_type: str = 'mlp',
        # 'mlp'         → BEVTrajectoryDecoder (V2, V3, V3-traj)
        # 'transformer' → TransformerTrajectoryDecoder (V4, V5)
        trajectory_head_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        self.use_trajectory = use_trajectory
        self.backbone_type = backbone_type
        self.decoder_type = decoder_type

        if backbone_cfg is None:
            backbone_cfg = {}

        # Remove 'type' key — used for routing only, not passed to constructors
        backbone_cfg.pop('type', None)

        # =====================================================================
        # Backbone — selected by backbone_type
        # =====================================================================
        if backbone_type == 'swin':
            # -----------------------------------------------------------------
            # SwinBackbone — V3, V4, V5
            # Swin-T pretrained on ImageNet
            # SOURCED: Liu et al., ICCV 2021 Best Paper
            # -----------------------------------------------------------------
            swin_cfg = {
                'lidar_input_channels': backbone_cfg.get(
                    'lidar_input_channels', LIDAR_TOTAL_CHANNELS
                ),
                'map_input_channels': backbone_cfg.get(
                    'map_input_channels', MAP_CHANNELS
                ),
                'pretrained': backbone_cfg.get('pretrained', True),
                # True → ImageNet pretrained weights via timm
                # SOURCED: RangeViT (CVPR 2023) — pretrained ViTs transfer
                # to LiDAR despite domain gap
                'out_channels': backbone_cfg.get('out_channels', 512),
                # 512 to match TwoStreamViTBackbone output
                # SOURCED: Nadeem thesis Section 3.3
                'img_size': backbone_cfg.get(
                    'img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX)
                ),
                'window_size': backbone_cfg.get('window_size', 5),
                # window_size=5: 100/5=20 ✓, 180/5=36 ✓
                # Confirmed working by forward pass test
                'swin_model_name': backbone_cfg.get(
                    'swin_model_name', 'swin_tiny_patch4_window7_224'
                ),
            }
            self.backbone = SwinBackbone(**swin_cfg)
            print(f"Backbone: SwinBackbone (pretrained={swin_cfg['pretrained']})")

        else:
            # -----------------------------------------------------------------
            # TwoStreamViTBackbone — V1, V2
            # ViT trained from scratch — Nadeem's original
            # SOURCED: Nadeem thesis Section 3.3
            # -----------------------------------------------------------------
            vit_cfg = {}
            vit_cfg['vit_model_name_lidar'] = backbone_cfg.get(
                'vit_model_name_lidar', 'vit_small_patch8_224'
            )
            vit_cfg['vit_model_name_map'] = backbone_cfg.get(
                'vit_model_name_map', 'vit_small_patch8_224'
            )
            vit_cfg['pretrained_lidar'] = backbone_cfg.get(
                'pretrained_lidar', False
            )
            vit_cfg['pretrained_map'] = backbone_cfg.get(
                'pretrained_map', False
            )
            vit_cfg['img_size'] = backbone_cfg.get(
                'img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX)
            )
            vit_cfg['lidar_adapter_out_channels'] = backbone_cfg.get(
                'lidar_adapter_out_channels', 192
            )
            vit_cfg['map_adapter_out_channels'] = backbone_cfg.get(
                'map_adapter_out_channels', 192
            )
            vit_cfg['fusion_block_planes'] = backbone_cfg.get(
                'fusion_block_planes', 512
            )
            vit_cfg['fusion_block_layers'] = backbone_cfg.get(
                'fusion_block_layers', 2
            )
            vit_cfg['fusion_block_kernel_size'] = backbone_cfg.get(
                'fusion_block_kernel_size', 3
            )
            vit_cfg['fusion_block_stride'] = backbone_cfg.get(
                'fusion_block_stride', 1
            )
            # Pass through any extra keys (e.g. res_block_type, drop_path_rate)
            for k, v in backbone_cfg.items():
                if k not in vit_cfg:
                    vit_cfg[k] = v

            self.backbone = TwoStreamViTBackbone(**vit_cfg)
            print(
                f"Backbone: TwoStreamViTBackbone "
                f"({vit_cfg['vit_model_name_lidar']})"
            )

        # Both backbones output [B, 512, 50, 90]
        self.feature_channels = self.backbone.final_feature_channels
        # Always 512 — heads are backbone-agnostic
        self.feature_map_h = GRID_HEIGHT_PX // 8   # 50
        self.feature_map_w = GRID_WIDTH_PX // 8    # 90

        # =====================================================================
        # Detection Head — unchanged from Nadeem, all versions
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        self.det_head = DetectionHead(
            in_channels=self.feature_channels
        )

        # =====================================================================
        # Intention Head — unchanged from Nadeem, all versions
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        self.intention_head = IntentionHead(
            in_channels=self.feature_channels
        )

        # =====================================================================
        # Trajectory Head — V2 onwards only
        #
        # V1:       not created (use_trajectory=False)
        # V2/V3:    MLP decoder (BEVTrajectoryDecoder)
        #           No history, no social attention
        # V3-traj:  MLP decoder, trained on parquet scenarios
        #           Ablation baseline for V4 trajectory comparison
        # V4/V5:    Transformer decoder (TransformerTrajectoryDecoder)
        #           GRU history encoder + social attention + cross-attention
        #
        # decoder_type is passed to TrajectoryHead which routes to
        # the correct decoder class in trajectory_decoder.py
        # =====================================================================
        if self.use_trajectory:
            if trajectory_head_cfg is None:
                trajectory_head_cfg = {}
            self.traj_head = TrajectoryHead(
                backbone_channels=self.feature_channels,
                feature_map_h=self.feature_map_h,
                feature_map_w=self.feature_map_w,
                decoder_type=decoder_type,
                # 'mlp'         → V2, V3, V3-traj
                # 'transformer' → V4, V5
                **trajectory_head_cfg
            )
            print(f"TrajectoryHead: ENABLED ({decoder_type} decoder)")
        else:
            self.traj_head = None
            print("TrajectoryHead: DISABLED (V1 mode)")

        # =====================================================================
        # Pre-generate anchors
        # Fixed during training — not learned parameters
        # SOURCED: generate_anchors() — Nadeem's original
        # =====================================================================
        anchors = generate_anchors()
        self.register_buffer('anchors', anchors)
        # Registered as buffer: moves to GPU with .to(device)
        # and saved with checkpoint

        print(
            f"\nIntentNetViT_MT Initialized:"
            f"\n  Backbone:         {backbone_type.upper()}"
            f"\n  Decoder:          {decoder_type}"
            f"\n  Feature channels: {self.feature_channels}"
            f"\n  Feature map:      {self.feature_map_h}×{self.feature_map_w}"
            f"\n  Trajectory head:  {'ENABLED' if use_trajectory else 'DISABLED'}"
            f"\n  Anchors:          {anchors.shape[0]} total"
        )

    # =========================================================================
    # _boxes_to_feature_map_pixels
    # Unchanged from original — coordinate conversion for trajectory head
    # =========================================================================

    def _boxes_to_feature_map_pixels(
        self,
        boxes_xywha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert GT box centres from ego-frame metres to feature map pixel coords.

        BEV coordinate convention (from Nadeem's utils.py):
            pixel_x (col) = BEV_PIXEL_OFFSET_X + y_ego / VOXEL_SIZE_M
            pixel_y (row) = BEV_PIXEL_OFFSET_Y - x_ego / VOXEL_SIZE_M

        Feature map is 8× smaller than BEV image (stride=8 for both ViT and Swin
        since SwinBackbone upsamples to [B, 512, 50, 90] matching ViT output).

        SOURCED: coordinate mapping from utils.py — Nadeem's original convention.

        Args:
            boxes_xywha: [N, 5] GT boxes (cx_m, cy_m, w, l, heading) in ego frame

        Returns:
            centers_px: [N, 2] box centres in feature map pixel coords (col, row)
        """
        if boxes_xywha.shape[0] == 0:
            return torch.zeros(0, 2, device=boxes_xywha.device)

        cx_m = boxes_xywha[:, 0]   # forward in ego frame
        cy_m = boxes_xywha[:, 1]   # lateral in ego frame

        bev_pixel_x = BEV_PIXEL_OFFSET_X + cy_m / VOXEL_SIZE_M   # column
        bev_pixel_y = BEV_PIXEL_OFFSET_Y - cx_m / VOXEL_SIZE_M   # row

        feature_stride = GRID_HEIGHT_PX // self.feature_map_h     # 8
        fm_pixel_x = (bev_pixel_x / feature_stride).clamp(
            0, self.feature_map_w - 1
        )
        fm_pixel_y = (bev_pixel_y / feature_stride).clamp(
            0, self.feature_map_h - 1
        )

        return torch.stack([fm_pixel_x, fm_pixel_y], dim=-1)
        # [N, 2] — (col, row) on 50×90 feature map

    # =========================================================================
    # forward — main forward pass for all versions
    # =========================================================================

    def forward(
        self,
        lidar_bev: torch.Tensor,
        map_bev: torch.Tensor,
        gt_list: list | None = None,
        use_gt_boxes_for_traj: bool = True,
        agent_history: torch.Tensor | None = None,
        # [N, 50, 5] — agent history from parquet files
        # None for V2/V3 (MLP decoder — history not used)
        # Required for V4/V5 (transformer decoder uses GRU history encoder)
        # SOURCED: history features — Abdulbaki thesis Section 3.6
        run_traj_head: bool = True,
        # Controls trajectory head execution for dual-dataset training:
        # True  → run trajectory head (V2/V3 normal, parquet batches V4/V5)
        # False → skip trajectory head (sensor batches in V4/V5 dual training)
        # Skipping trajectory head for sensor batches means det+intent losses
        # only update the backbone for those batches — correct behaviour.
        # ── MODIFICATION 13: accepted but not used inside forward ──────────────
        boxes_padded: torch.Tensor | None = None,
        # [B, N_max, 5] — from collate_fn, not used directly in forward
        # kept for API consistency with train.py call
        agent_mask: torch.Tensor | None = None,
        # [B, N_max] — from collate_fn, not used directly in forward
        # train.py uses these for GT collection, model uses gt_list for sampling
    ) -> dict:
        """
        Forward pass — identical interface for all versions V1-V5.

        V1/V2/V3 usage (single dataset):
            forward(lidar_bev, map_bev, gt_list=gt_list)
            All heads run every batch.

        V4/V5 dual-dataset usage:
            Sensor batch:  forward(..., run_traj_head=False)
                           Only det+intent heads contribute to loss.
            Parquet batch: forward(..., agent_history=history, run_traj_head=True)
                           Only trajectory head contributes to loss.
            Both batches update the shared backbone.

        Args:
            lidar_bev:             [B, 290, 400, 720]
            map_bev:               [B,   9, 400, 720]
            gt_list:               list of GT dicts per batch element
            use_gt_boxes_for_traj: True during training (teacher forcing)
                                   SOURCED: DeTra (Casas et al., 2024)
            agent_history:         [N, 50, 5] from parquet — V4/V5 only
            run_traj_head:         False for sensor batches in dual training

        Returns:
            dict with:
                det_cls_logits:   [B, 22500, 1]  — detection objectness
                det_box_preds:    [B, 22500, 6]  — box regression deltas
                intention_logits: [B, 22500, 8]  — intention class logits
                anchors:          [22500, 5]      — anchor boxes
                y_hat:            [F, N, 60, 4]  — trajectories or None
                pi:               [N, F]          — mode logits or None
                traj_gt_boxes:    [N, 5]          — boxes used for sampling
                feature_map:      [B, 512, 50, 90] — exposed for traj loss
        """
        B = lidar_bev.shape[0]
        device = lidar_bev.device

        # =====================================================================
        # Step 1: Backbone — shared for ALL tasks and ALL versions
        # Input:  lidar_bev [B, 290, 400, 720]
        #         map_bev   [B,   9, 400, 720]
        # Output: feature_map [B, 512, 50, 90]
        # =====================================================================
        feature_map = self.backbone(lidar_bev, map_bev)
        # [B, 512, 50, 90]

        # =====================================================================
        # Step 2: Detection head — unchanged for all versions V1-V5
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        det_cls_logits, det_box_preds = self.det_head(feature_map)
        # det_cls_logits: [B, 22500, 1]
        # det_box_preds:  [B, 22500, 6]

        # =====================================================================
        # Step 3: Intention head — unchanged for all versions V1-V5
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        intention_logits = self.intention_head(feature_map)
        # [B, 22500, 8]

        # =====================================================================
        # Step 4: Trajectory head
        #
        # Skipped when:
        #   - V1 (use_trajectory=False, traj_head is None)
        #   - Sensor batches in V4/V5 dual-dataset training (run_traj_head=False)
        #
        # Runs when:
        #   - V2/V3 normal training (MLP decoder, all agents, sensor GT)
        #   - V3-traj ablation (MLP decoder, parquet scenarios)
        #   - V4/V5 parquet batches (transformer decoder, focal agent)
        # =====================================================================
        y_hat = None
        pi = None
        traj_gt_boxes = None

        if self.use_trajectory and self.traj_head is not None and run_traj_head:

            if use_gt_boxes_for_traj and gt_list is not None:
                # --- Teacher forcing: use GT box locations ---
                # MODIFICATION 13: loop over all B scenes, sample from correct
                # feature_map[b] per scene, run social attention within each scene.
                # Previously only gt_list[0] was used — now all B scenes contribute.
                # SOURCED: padding approach — HiVT, DeTra, Social-LSTM.

                all_y_hat_list = []
                all_pi_list    = []
                all_boxes_list = []

                for b in range(B):
                    gt_b = gt_list[b]
                    if gt_b is None:
                        continue

                    boxes_b = gt_b['boxes_xywha'].to(device)  # [N_b, 5]
                    if boxes_b.shape[0] == 0:
                        continue

                    # Convert box centres to feature map pixel coords
                    box_centers_px = self._boxes_to_feature_map_pixels(boxes_b)
                    # [N_b, 2]

                    box_params_m = boxes_b[:, :5]
                    # [N_b, 5]

                    # Sample from correct feature map slice for scene b
                    # KEY FIX: each scene uses its own feature_map[b]
                    # not feature_map[0] as in the original
                    feature_map_b = feature_map[b:b+1]  # [1, 512, 50, 90]
                    map_bev_b     = map_bev[b:b+1]      # [1, 9, 400, 720]

                    # Run trajectory head for scene b only
                    # Social attention runs within scene b — correct behaviour
                    # Vehicles from different scenes do not attend to each other
                    y_hat_b, pi_b = self.traj_head(
                        feature_map=feature_map_b,
                        map_bev=map_bev_b,
                        box_centers_px=box_centers_px,
                        box_params_m=box_params_m,
                        agent_history=agent_history,
                        use_gt_boxes=True,
                    )
                    # y_hat_b: [F, N_b, H, 4]
                    # pi_b:    [N_b, F]

                    all_y_hat_list.append(y_hat_b)
                    all_pi_list.append(pi_b)
                    all_boxes_list.append(boxes_b)

                if all_y_hat_list:
                    # Concatenate across all scenes
                    y_hat         = torch.cat(all_y_hat_list, dim=1)  # [F, N_total, H, 4]
                    pi            = torch.cat(all_pi_list,    dim=0)  # [N_total, F]
                    traj_gt_boxes = torch.cat(all_boxes_list, dim=0)  # [N_total, 5]
                else:
                    F_modes = TRAJECTORY_NUM_MODES
                    H       = TRAJECTORY_FUTURE_STEPS
                    y_hat         = torch.zeros(F_modes, 0, H, 4, device=device)
                    pi            = torch.zeros(0, F_modes,       device=device)
                    traj_gt_boxes = torch.zeros(0, 5,             device=device)

            else:
                # --- Inference mode ---
                # After NMS, detected box centres are used for trajectory.
                # Handled post-NMS in eval.py.
                y_hat = None
                pi = None

        return {
            # Detection outputs — same as Nadeem's original, all versions
            "det_cls_logits":   det_cls_logits,    # [B, 22500, 1]
            "det_box_preds":    det_box_preds,      # [B, 22500, 6]

            # Intention outputs — same as Nadeem's original, all versions
            "intention_logits": intention_logits,   # [B, 22500, 8]

            # Anchors — needed by loss function
            "anchors":          self.anchors,       # [22500, 5]

            # Trajectory outputs — V2+ only, None for V1
            # Also None for sensor batches in V4/V5 dual-dataset training
            "y_hat":            y_hat,              # [F, N, 60, 4] or None
            "pi":               pi,                 # [N, F] or None

            # GT boxes used for trajectory sampling (for loss alignment)
            "traj_gt_boxes":    traj_gt_boxes,      # [N, 5] or None

            # Expose feature map — needed for trajectory loss in parquet batches
            # Not used in V1/V2/V3 loss computation
            "feature_map":      feature_map,        # [B, 512, 50, 90]
        }

    # =========================================================================
    # Convenience methods for dual-dataset training (V4/V5)
    # =========================================================================

    def forward_det_intent_only(
        self,
        lidar_bev: torch.Tensor,
        map_bev: torch.Tensor,
        gt_list: list | None = None,
    ) -> dict:
        """
        Sensor batch forward — detection + intention only.
        Trajectory head is skipped (run_traj_head=False).

        Used every iteration for sensor batches in V4/V5 dual training.
        Detection and intention losses update the shared backbone.

        SOURCED: heterogeneous multi-task learning — UniDet (Zhou et al.,
        2021), OmniDet (Rashed et al., 2021) use per-task dataloaders
        with a shared backbone updated by all task losses.
        """
        return self.forward(
            lidar_bev=lidar_bev,
            map_bev=map_bev,
            gt_list=gt_list,
            use_gt_boxes_for_traj=False,
            agent_history=None,
            run_traj_head=False,
        )

    def forward_traj_only(
        self,
        lidar_bev: torch.Tensor,
        map_bev: torch.Tensor,
        gt_boxes: torch.Tensor,
        agent_history: torch.Tensor,
    ) -> dict:
        """
        Parquet batch forward — trajectory head only.

        Backbone + all heads run, but only trajectory loss is computed
        in train.py for this batch. Detection and intention heads run
        but their outputs are not used for loss in parquet batches.

        Used every ~70 iterations for parquet batches in V4/V5 training.
        Trajectory loss updates the shared backbone, GRU encoder,
        social attention, and transformer decoder.

        Args:
            lidar_bev:     [B, 290, 400, 720] — matched sensor sequence
            map_bev:       [B,   9, 400, 720] — matched sensor sequence
            gt_boxes:      [N, 5] — agent boxes from parquet (ego frame)
            agent_history: [N, 50, 5] — agent history from parquet
                           SOURCED: Abdulbaki thesis Section 3.6

        Returns:
            dict with y_hat [F, N, 60, 4], pi [N, F], feature_map
        """
        # Wrap gt_boxes as gt_list format expected by forward()
        gt_list = [{'boxes_xywha': gt_boxes}]

        return self.forward(
            lidar_bev=lidar_bev,
            map_bev=map_bev,
            gt_list=gt_list,
            use_gt_boxes_for_traj=True,
            agent_history=agent_history,
            run_traj_head=True,
        )

    # =========================================================================
    # load_pretrained_backbone
    # =========================================================================

    def load_pretrained_backbone(self, checkpoint_path: str) -> None:
        """
        Load backbone weights from a previous checkpoint.

        For V3/V4/V5: not needed — Swin loads ImageNet pretrained weights
        via timm automatically when pretrained=True.

        For V2 resuming from V1: loads ViT backbone weights from V1
        checkpoint, trajectory head starts from random initialisation.

        For V4 initialising from V3: loads Swin backbone from V3 checkpoint,
        new trajectory head (GRU + social + transformer) starts from scratch.

        SOURCED: standard transfer learning practice.
        Howard & Ruder (ACL 2018) — discriminative fine-tuning.

        Args:
            checkpoint_path: path to .pth checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        backbone_state = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if k.startswith('backbone.')
        }

        if backbone_state:
            missing, unexpected = self.backbone.load_state_dict(
                backbone_state, strict=False
            )
            print(f"Loaded backbone weights from: {checkpoint_path}")
            if missing:
                print(f"  Missing keys:    {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print(
                f"Warning: No backbone weights found with 'backbone.' prefix "
                f"in {checkpoint_path}."
            )