# models/model_mt.py
#
# New file for IntentTrajNet-AV2
# Author: [Your Name] — Bachelor Thesis, GUC 2025
#
# This file assembles the full multi-task model by combining:
#   - TwoStreamViTBackbone from Nadeem's model_vit.py (imported, not copied)
#   - DetectionHead and IntentionHead from models/heads.py (unchanged from Nadeem)
#   - TrajectoryHead from models/heads.py (new)
#
# Design principle:
#   TwoStreamViTBackbone is imported directly from the original repo
#   rather than copied. This means any fixes to the backbone automatically
#   apply here. The backbone is treated as a black box — input is
#   (lidar_bev, map_bev), output is [B, 512, 50, 90] feature map.
#
# V1 mode (use_trajectory=False):
#   Identical to Nadeem's IntentNetViT — only detection and intention heads.
#   Used as the baseline for comparison.
#
# V2 mode (use_trajectory=True, backbone='vit'):
#   Adds TrajectoryHead on top of the same ViT backbone.
#   Trajectory predictions from 1-second implicit BEV history.
#
# V3 mode (use_trajectory=True, backbone='swin'):
#   Swaps backbone to Swin-T (pretrained).
#   Trajectory decoder upgraded to transformer (cross-attention over BEV).
#   Added in backbone.py — referenced here via config.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import backbone from Nadeem's original — no code duplication
# Importing from our backbone.py
from models.backbone import TwoStreamViTBackbone, SwinBackbone

# Import heads from our new heads.py
from models.heads import DetectionHead, IntentionHead, TrajectoryHead

# Import constants
from utils.constants import (
    LIDAR_TOTAL_CHANNELS,
    MAP_CHANNELS,
    GRID_HEIGHT_PX,
    GRID_WIDTH_PX,
    NUM_ANCHORS_PER_LOC,
    NUM_INTENTION_CLASSES,
    TRAJECTORY_FUTURE_STEPS,   # SOURCED: Abdulbaki thesis Section 3.1
    TRAJECTORY_NUM_MODES,      # SOURCED: Abdulbaki thesis Section 3.4.3
    VOXEL_SIZE_M,
    BEV_PIXEL_OFFSET_X,
    BEV_PIXEL_OFFSET_Y,
)
from utils.utils import generate_anchors


class IntentNetViT_MT(nn.Module):
    """
    Multi-task Vision Transformer model for joint vehicle detection,
    intention prediction, and trajectory forecasting from LiDAR BEV.

    Extends Nadeem's IntentNetViT by adding a trajectory prediction head
    on top of the shared backbone feature map.

    Three operating modes controlled by config:
        V1: use_trajectory=False — detection + intention only (baseline)
        V2: use_trajectory=True, backbone='vit' — + MLP trajectory decoder
        V3: use_trajectory=True, backbone='swin' — + transformer decoder
            (backbone swap handled in backbone.py, referenced via config)

    Architecture:
        Input:  lidar_bev [B, 290, 400, 720]
                map_bev   [B,   9, 400, 720]
                    ↓
        TwoStreamViTBackbone
                    ↓
        feature_map [B, 512, 50, 90]   ← shared by all three heads
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    DetHead    IntentHead   TrajHead (V2/V3 only)
    [B,22500,1] [B,22500,8] [F,N,60,4] + [N,F]
    [B,22500,6]

    SOURCED: backbone architecture from Nadeem's thesis Section 3.3
    SOURCED: detection and intention heads unchanged from Nadeem
    SOURCED: trajectory head adapted from HiVT (Abdulbaki thesis Section 3.4.3)
    """

    def __init__(
        self,
        # --- Backbone config ---
        backbone_cfg: dict | None = None,
        # Dict of kwargs passed to TwoStreamViTBackbone.
        # Same format as Nadeem's IntentNetViT for compatibility.
        # SOURCED: parameter names from Nadeem's model_vit.py

        # --- Trajectory head config ---
        use_trajectory: bool = True,
        # False → V1 mode (no trajectory head)
        # True  → V2/V3 mode (trajectory head active)

        trajectory_head_cfg: dict | None = None,
        # Optional kwargs for TrajectoryHead
        # Currently unused — TrajectoryHead reads from constants

    ) -> None:
        super().__init__()

        self.use_trajectory = use_trajectory

        # =====================================================================
        # Backbone — TwoStreamViTBackbone
        # Imported from Nadeem's model_vit.py, not modified.
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        if backbone_cfg is None:
            backbone_cfg = {}

        # Apply same defaults as Nadeem's IntentNetViT
        # SOURCED: default values from IntentNetViT.__init__ in model_vit.py
        backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch8_224')
        backbone_cfg.setdefault('vit_model_name_map', 'vit_small_patch8_224')
        backbone_cfg.setdefault('pretrained_lidar', False)
        backbone_cfg.setdefault('pretrained_map', False)
        backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
        backbone_cfg.setdefault('lidar_adapter_out_channels', 192)
        backbone_cfg.setdefault('map_adapter_out_channels', 192)
        backbone_cfg.setdefault('fusion_block_planes', 512)
        backbone_cfg.setdefault('fusion_block_layers', 2)
        backbone_cfg.setdefault('fusion_block_kernel_size', 3)
        backbone_cfg.setdefault('fusion_block_stride', 1)

        self.backbone = TwoStreamViTBackbone(**backbone_cfg)
        self.feature_channels = self.backbone.final_feature_channels
        # SOURCED: 512 — Nadeem thesis Section 3.3

        # Feature map spatial dimensions
        # SOURCED: 50×90 = 400/8 × 720/8 — Nadeem thesis Section 3.3
        self.feature_map_h = GRID_HEIGHT_PX // 8
        self.feature_map_w = GRID_WIDTH_PX // 8

        # =====================================================================
        # Detection Head — unchanged from Nadeem
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        self.det_head = DetectionHead(
            in_channels=self.feature_channels
        )

        # =====================================================================
        # Intention Head — unchanged from Nadeem
        # SOURCED: Nadeem thesis Section 3.3
        # =====================================================================
        self.intention_head = IntentionHead(
            in_channels=self.feature_channels
        )

        # =====================================================================
        # Trajectory Head — NEW for V2 and V3
        # Only created when use_trajectory=True
        # SOURCED: adapted from HiVT MLPDecoder (Abdulbaki thesis Section 3.4.3)
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
            print("TrajectoryHead: ENABLED (V2/V3 mode)")
        else:
            self.traj_head = None
            print("TrajectoryHead: DISABLED (V1 mode)")

        # =====================================================================
        # Pre-generate anchors
        # SOURCED: generate_anchors() from utils.py — Nadeem's original
        # Anchors are fixed during training, not learned parameters
        # =====================================================================
        anchors = generate_anchors()
        self.register_buffer('anchors', anchors)
        # Registered as buffer so it moves to GPU with .to(device)
        # and is saved with the model checkpoint

        print(
            f"\nIntentNetViT_MT Initialized:"
            f"\n  Backbone: {backbone_cfg['vit_model_name_lidar']}"
            f"\n  Feature channels: {self.feature_channels}"
            f"\n  Feature map: {self.feature_map_h}×{self.feature_map_w}"
            f"\n  Detection head: ENABLED"
            f"\n  Intention head: ENABLED"
            f"\n  Trajectory head: {'ENABLED' if use_trajectory else 'DISABLED'}"
            f"\n  Anchors: {anchors.shape[0]} total"
        )

    def _boxes_to_feature_map_pixels(
        self,
        boxes_xywha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert GT box centres from ego-frame metres to feature map pixel coords.

        The BEV grid maps ego-frame metres to pixel coordinates using:
            pixel_x = BEV_PIXEL_OFFSET_X + y_ego / VOXEL_SIZE_M
            pixel_y = BEV_PIXEL_OFFSET_Y - x_ego / VOXEL_SIZE_M

        The feature map is 8× smaller than the BEV image (stride=8).
        So feature map pixel = BEV pixel / 8.

        SOURCED: coordinate mapping from utils.py create_intentnet_lidar_bev()
        and world_to_bev_pixel() — Nadeem's original coordinate convention.

        Args:
            boxes_xywha: [N, 5] GT boxes in ego frame (cx_m, cy_m, w, l, heading)

        Returns:
            centers_px: [N, 2] box centres in feature map pixel coords (col, row)
                        col ∈ [0, feature_map_w-1]
                        row ∈ [0, feature_map_h-1]
        """
        if boxes_xywha.shape[0] == 0:
            return torch.zeros(0, 2, device=boxes_xywha.device)

        # Extract ego-frame centre positions in metres
        cx_m = boxes_xywha[:, 0]  # [N] — forward/backward in ego frame
        cy_m = boxes_xywha[:, 1]  # [N] — left/right in ego frame

        # Convert to BEV pixel coordinates
        # SOURCED: pixel_x maps to y_ego (lateral), pixel_y maps to x_ego (forward)
        # This is the convention used throughout Nadeem's BEV generation code
        bev_pixel_x = BEV_PIXEL_OFFSET_X + cy_m / VOXEL_SIZE_M  # column
        bev_pixel_y = BEV_PIXEL_OFFSET_Y - cx_m / VOXEL_SIZE_M  # row

        # Scale from BEV pixels to feature map pixels (divide by stride=8)
        # SOURCED: stride=8 from patch_size=8 in vit_small_patch8_224
        feature_stride = GRID_HEIGHT_PX // self.feature_map_h
        # = 400 / 50 = 8
        fm_pixel_x = bev_pixel_x / feature_stride  # column on feature map
        fm_pixel_y = bev_pixel_y / feature_stride  # row on feature map

        # Clamp to valid feature map range
        fm_pixel_x = fm_pixel_x.clamp(0, self.feature_map_w - 1)
        fm_pixel_y = fm_pixel_y.clamp(0, self.feature_map_h - 1)

        # Stack as [N, 2] (col, row)
        centers_px = torch.stack([fm_pixel_x, fm_pixel_y], dim=-1)
        return centers_px

    def forward(
        self,
        lidar_bev: torch.Tensor,
        map_bev: torch.Tensor,
        gt_list: list | None = None,
        use_gt_boxes_for_traj: bool = True,
    ) -> dict:
        """
        Forward pass of IntentNetViT_MT.

        Args:
            lidar_bev:  [B, 290, 400, 720] — LiDAR BEV tensor
            map_bev:    [B,   9, 400, 720] — Map BEV tensor
            gt_list:    list of GT dicts per batch element
                        Required during training for teacher forcing.
                        None during inference.
            use_gt_boxes_for_traj: bool
                        True  → use GT box centres for trajectory sampling
                                 (teacher forcing during training)
                        False → use predicted box centres after NMS
                                 (inference mode)
                        SOURCED: DeTra (Casas et al., 2024) — GT boxes
                        during training decouples trajectory learning
                        from detection errors.

        Returns:
            dict with keys:
                det_cls_logits:    [B, 22500, 1]    — detection objectness
                det_box_preds:     [B, 22500, 6]    — box regression deltas
                intention_logits:  [B, 22500, 8]    — intention class logits
                anchors:           [22500, 5]        — anchor boxes
                y_hat:             [F, N, 60, 4]    — trajectories (V2/V3 only)
                pi:                [N, F]            — mode logits (V2/V3 only)
                traj_gt_boxes:     [N, 5]            — boxes used for sampling
                None for y_hat and pi when use_trajectory=False (V1 mode)
        """
        B = lidar_bev.shape[0]
        device = lidar_bev.device

        # =====================================================================
        # Step 1: Shared backbone forward pass
        # Input:  lidar_bev [B, 290, 400, 720]
        #         map_bev   [B,   9, 400, 720]
        # Output: feature_map [B, 512, 50, 90]
        # SOURCED: TwoStreamViTBackbone.forward() — Nadeem's model_vit.py
        # =====================================================================
        feature_map = self.backbone(lidar_bev, map_bev)
        # [B, 512, 50, 90]

        # =====================================================================
        # Step 2: Detection head
        # SOURCED: Nadeem's original — unchanged
        # =====================================================================
        det_cls_logits, det_box_preds = self.det_head(feature_map)
        # det_cls_logits: [B, 22500, 1]
        # det_box_preds:  [B, 22500, 6]

        # =====================================================================
        # Step 3: Intention head
        # SOURCED: Nadeem's original — unchanged
        # =====================================================================
        intention_logits = self.intention_head(feature_map)
        # [B, 22500, 8]

        # =====================================================================
        # Step 4: Trajectory head (V2/V3 only)
        # NEW — not present in Nadeem's IntentNetViT
        # =====================================================================
        y_hat = None
        pi = None
        traj_gt_boxes = None

        if self.use_trajectory and self.traj_head is not None:

            if use_gt_boxes_for_traj and gt_list is not None:
                # --- Teacher forcing: use GT box locations ---
                # During training we use the actual GT box centres to sample
                # BEV features. This decouples trajectory learning from
                # detection performance — in early training the detector is
                # weak and would give bad box locations, which would corrupt
                # the trajectory training signal.
                # SOURCED: DeTra (Casas et al., 2024) — standard practice.

                # Collect GT boxes from all batch elements
                # We process all vehicles from all frames together
                all_gt_boxes = []
                for b in range(B):
                    if gt_list[b] is not None and 'boxes_xywha' in gt_list[b]:
                        boxes_b = gt_list[b]['boxes_xywha'].to(device)
                        if boxes_b.shape[0] > 0:
                            all_gt_boxes.append(boxes_b)

                if len(all_gt_boxes) > 0:
                    # Stack all GT boxes across the batch
                    # Note: we use first batch element's feature map
                    # for trajectory since N varies per element
                    # For V2 we process batch element 0
                    # This is a known simplification for single-scene inference
                    # ASSUMED: single-scene processing for trajectory head
                    # Full batch processing is a V3 improvement
                    gt_boxes_b0 = gt_list[0]['boxes_xywha'].to(device) \
                        if gt_list[0] is not None else torch.zeros(0, 5, device=device)

                    traj_gt_boxes = gt_boxes_b0

                    if gt_boxes_b0.shape[0] > 0:
                        # Convert GT box centres to feature map pixel coords
                        box_centers_px = self._boxes_to_feature_map_pixels(
                            gt_boxes_b0
                        )
                        # [N, 2] — (col, row) on 50×90 feature map

                        # Box params for concatenation with BEV features
                        # Use (cx_m, cy_m, w_m, l_m) — first 4 columns
                        box_params_m = gt_boxes_b0[:, :4]
                        # [N, 4]

                        # Run trajectory head
                        # Takes feature map + box locations
                        # Returns [F, N, 60, 4] and [N, F]
                        y_hat, pi = self.traj_head(
                            feature_map=feature_map,
                            box_centers_px=box_centers_px,
                            box_params_m=box_params_m,
                            use_gt_boxes=True
                        )
                    else:
                        # No GT boxes for this batch element
                        F_modes = TRAJECTORY_NUM_MODES
                        H = TRAJECTORY_FUTURE_STEPS
                        y_hat = torch.zeros(F_modes, 0, H, 4, device=device)
                        pi = torch.zeros(0, F_modes, device=device)

            else:
                # --- Inference mode: use predicted box locations ---
                # After NMS, the detected box centres are used to sample
                # BEV features for trajectory prediction.
                # This is handled by the inference pipeline, not here.
                # For now, return None and handle in eval.py
                # ASSUMED: inference trajectory prediction is handled
                # post-NMS in the evaluation pipeline
                y_hat = None
                pi = None

        return {
            # Detection outputs — same as Nadeem's original
            "det_cls_logits": det_cls_logits,   # [B, 22500, 1]
            "det_box_preds": det_box_preds,      # [B, 22500, 6]

            # Intention outputs — same as Nadeem's original
            "intention_logits": intention_logits, # [B, 22500, 8]

            # Anchors — needed by loss function
            "anchors": self.anchors,              # [22500, 5]

            # Trajectory outputs — new for V2/V3, None for V1
            "y_hat": y_hat,    # [F, N, 60, 4] or None
            "pi": pi,          # [N, F] or None

            # GT boxes used for trajectory sampling (for loss alignment)
            "traj_gt_boxes": traj_gt_boxes  # [N, 5] or None
        }

    def load_pretrained_backbone(self, checkpoint_path: str) -> None:
        """
        Load backbone weights from Nadeem's pretrained checkpoint.

        This allows V2 to start from Nadeem's trained backbone rather
        than training from scratch. Only backbone weights are loaded —
        the new trajectory head starts from random initialisation.

        SOURCED: approach of loading partial checkpoint is standard
        transfer learning practice. The backbone is the shared component
        whose weights benefit most from pretraining.

        Args:
            checkpoint_path: path to Nadeem's vit_model.pth checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter to only backbone weights
        backbone_state = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if k.startswith('backbone.')
        }

        if backbone_state:
            missing, unexpected = self.backbone.load_state_dict(
                backbone_state, strict=False
            )
            print(f"Loaded backbone from: {checkpoint_path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            # Try loading full model weights and filtering
            backbone_state_full = {
                k: v for k, v in state_dict.items()
                if 'backbone' in k or 'vit_lidar' in k or 'vit_map' in k
            }
            print(
                f"Warning: Could not find backbone weights with 'backbone.' prefix. "
                f"Found {len(backbone_state_full)} potentially relevant keys."
            )