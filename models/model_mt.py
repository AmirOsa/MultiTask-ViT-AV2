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
#   V2: backbone='vit', use_trajectory=True
#       Adds MLP trajectory decoder on top of same ViT backbone.
#
#   V3: backbone='swin', use_trajectory=True
#       Swaps backbone to Swin-T pretrained. MLP decoder unchanged.
#       ONLY change vs V2 — isolates backbone contribution.
#
#   V4: backbone='swin', use_trajectory=True, decoder_type='transformer'
#       Adds transformer trajectory decoder + agent history.
#       (decoder_type handled in trajectory_decoder.py — not yet implemented)
#
#   V5: backbone='swin', use_trajectory=True, decoder_type='transformer'
#       Adds class weights + 6s horizon + velocity heading.
#       (supervision changes in loss.py and dataset.py)
#
# Backbone selection:
#   'vit'  → TwoStreamViTBackbone (Nadeem's original, trained from scratch)
#   'swin' → SwinBackbone (Swin-T pretrained ImageNet, new for V3+)
#
# All heads (DetectionHead, IntentionHead, TrajectoryHead) are backbone-agnostic.
# Both backbones output [B, 512, 50, 90] — heads work unchanged for all versions.

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
)
from utils.utils import generate_anchors


class IntentNetViT_MT(nn.Module):
    """
    Multi-task model for joint vehicle detection, intention prediction,
    and trajectory forecasting from LiDAR BEV.

    Supports V1 through V5 via backbone_type and use_trajectory flags.
    All versions share the same heads — only the backbone changes between
    V1/V2 (ViT) and V3/V4/V5 (Swin-T pretrained).

    Output feature map is always [B, 512, 50, 90] regardless of backbone,
    so DetectionHead, IntentionHead, and TrajectoryHead are unchanged.

    Args:
        backbone_type:      'vit' for V1/V2, 'swin' for V3/V4/V5
        backbone_cfg:       dict of kwargs for the selected backbone class
        use_trajectory:     False for V1, True for V2-V5
        trajectory_head_cfg: optional kwargs for TrajectoryHead
    """

    def __init__(
        self,
        backbone_type: str = 'vit',
        # 'vit'  → TwoStreamViTBackbone (V1, V2)
        # 'swin' → SwinBackbone          (V3, V4, V5)
        backbone_cfg: dict | None = None,
        use_trajectory: bool = True,
        trajectory_head_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        self.use_trajectory = use_trajectory
        self.backbone_type = backbone_type

        if backbone_cfg is None:
            backbone_cfg = {}

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
                # NEEDS TEST: 5 divides 100 and 180 evenly
                # (feature map = 400/4=100, 720/4=180 with patch_size=4)
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
        # V1: not created (use_trajectory=False)
        # V2, V3: MLP decoder (BEVTrajectoryDecoder)
        # V4, V5: transformer decoder (added in trajectory_decoder.py later)
        # =====================================================================
        if self.use_trajectory:
            if trajectory_head_cfg is None:
                trajectory_head_cfg = {}
            self.traj_head = TrajectoryHead(
                backbone_channels=self.feature_channels,
                feature_map_h=self.feature_map_h,
                feature_map_w=self.feature_map_w,
                **trajectory_head_cfg
            )
            print("TrajectoryHead: ENABLED")
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

        print(
            f"\nIntentNetViT_MT Initialized:"
            f"\n  Version:          {backbone_type.upper()} backbone"
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
    # forward
    # =========================================================================

    def forward(
        self,
        lidar_bev: torch.Tensor,
        map_bev: torch.Tensor,
        gt_list: list | None = None,
        use_gt_boxes_for_traj: bool = True,
    ) -> dict:
        """
        Forward pass — identical interface for all versions V1-V5.

        Args:
            lidar_bev:  [B, 290, 400, 720]
            map_bev:    [B,   9, 400, 720]
            gt_list:    list of GT dicts per batch element (training only)
            use_gt_boxes_for_traj: True during training (teacher forcing)

        Returns:
            dict with:
                det_cls_logits:   [B, 22500, 1]
                det_box_preds:    [B, 22500, 6]
                intention_logits: [B, 22500, 8]
                anchors:          [22500, 5]
                y_hat:            [F, N, 60, 4] or None (V1)
                pi:               [N, F] or None (V1)
                traj_gt_boxes:    [N, 5] or None (V1)
        """
        B = lidar_bev.shape[0]
        device = lidar_bev.device

        # =====================================================================
        # Step 1: Backbone
        # Both TwoStreamViTBackbone and SwinBackbone output [B, 512, 50, 90]
        # =====================================================================
        feature_map = self.backbone(lidar_bev, map_bev)
        # [B, 512, 50, 90]

        # =====================================================================
        # Step 2: Detection head — unchanged for all versions
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        det_cls_logits, det_box_preds = self.det_head(feature_map)
        # det_cls_logits: [B, 22500, 1]
        # det_box_preds:  [B, 22500, 6]

        # =====================================================================
        # Step 3: Intention head — unchanged for all versions
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        intention_logits = self.intention_head(feature_map)
        # [B, 22500, 8]

        # =====================================================================
        # Step 4: Trajectory head — V2 onwards only
        # V1: skipped (traj_head is None)
        # V2, V3: MLP decoder
        # V4, V5: transformer decoder (not yet implemented)
        # =====================================================================
        y_hat = None
        pi = None
        traj_gt_boxes = None

        if self.use_trajectory and self.traj_head is not None:

            if use_gt_boxes_for_traj and gt_list is not None:
                # Teacher forcing — use GT box centres for BEV feature sampling
                # SOURCED: DeTra (Casas et al., 2024) — standard practice to
                # decouple trajectory learning from detection errors in early training
                gt_boxes_b0 = (
                    gt_list[0]['boxes_xywha'].to(device)
                    if gt_list[0] is not None
                    else torch.zeros(0, 5, device=device)
                )
                traj_gt_boxes = gt_boxes_b0

                if gt_boxes_b0.shape[0] > 0:
                    box_centers_px = self._boxes_to_feature_map_pixels(
                        gt_boxes_b0
                    )
                    # [N, 2] — (col, row) on 50×90 feature map

                    box_params_m = gt_boxes_b0[:, :4]
                    # [N, 4] — (cx_m, cy_m, w_m, l_m)

                    y_hat, pi = self.traj_head(
                        feature_map=feature_map,
                        box_centers_px=box_centers_px,
                        box_params_m=box_params_m,
                        use_gt_boxes=True
                    )
                    # y_hat: [F, N, 60, 4]
                    # pi:    [N, F]
                else:
                    F_modes = TRAJECTORY_NUM_MODES
                    H = TRAJECTORY_FUTURE_STEPS
                    y_hat = torch.zeros(F_modes, 0, H, 4, device=device)
                    pi = torch.zeros(0, F_modes, device=device)
            else:
                # Inference mode — trajectory prediction from predicted boxes
                # Handled post-NMS in eval.py
                y_hat = None
                pi = None

        return {
            "det_cls_logits":   det_cls_logits,    # [B, 22500, 1]
            "det_box_preds":    det_box_preds,      # [B, 22500, 6]
            "intention_logits": intention_logits,   # [B, 22500, 8]
            "anchors":          self.anchors,       # [22500, 5]
            "y_hat":            y_hat,              # [F, N, 60, 4] or None
            "pi":               pi,                 # [N, F] or None
            "traj_gt_boxes":    traj_gt_boxes       # [N, 5] or None
        }

    # =========================================================================
    # load_pretrained_backbone
    # =========================================================================

    def load_pretrained_backbone(self, checkpoint_path: str) -> None:
        """
        Load backbone weights from a previous checkpoint.

        For V3: not typically needed — Swin loads ImageNet pretrained weights
        via timm automatically when pretrained=True.

        For V2 resuming from V1: loads ViT backbone weights from V1 checkpoint,
        trajectory head starts from random initialisation.

        SOURCED: standard transfer learning practice.
        Howard & Ruder (ACL 2018) — fine-tuning pretrained models.

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