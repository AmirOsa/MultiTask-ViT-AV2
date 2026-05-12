# models/heads.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
#
# Modifications:
#   1. Updated import paths to match new repo structure
#   2. Added TrajectoryHead class — new, no equivalent in Nadeem's original
#   3. DetectionHead and IntentionHead are completely unchanged
#
# TrajectoryHead design:
#   Takes the shared BEV feature map [B, 512, 50, 90] and detected box
#   locations, extracts per-agent feature vectors via bilinear sampling
#   (F.grid_sample), concatenates box parameters, and feeds into
#   BEVTrajectoryDecoder to produce 6-mode trajectory predictions.
#
#   Bilinear sampling approach:
#   SOURCED: standard RoI feature extraction from two-stage detectors.
#   Introduced in Faster R-CNN (Ren et al., NeurIPS 2015) and refined
#   as RoIAlign in Mask R-CNN (He et al., ICCV 2017).
#
#   Teacher forcing during training:
#   SOURCED: standard practice in two-stage detection+prediction pipelines
#   including DeTra (Casas et al., Waabi 2024).

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import (
    NUM_ANCHORS_PER_LOC,       # 5 anchors per feature map location
    NUM_INTENTION_CLASSES,     # 8 intention classes
    GRID_HEIGHT_PX,            # 400 — input BEV height
    GRID_WIDTH_PX,             # 720 — input BEV width
    TRAJECTORY_DECODER_HIDDEN, # NEEDS TEST: hidden size for decoder
)
from models.trajectory_decoder import BEVTrajectoryDecoder


# =============================================================================
# DetectionHead
# Unchanged from Nadeem's original
# =============================================================================

class DetectionHead(nn.Module):
    """
    Detection head — predicts objectness and box parameters per anchor.
    Unchanged from Nadeem's original model_vit.py.

    Input:  [B, 512, 50, 90] feature map from backbone
    Output: objectness [B, 22500, 1]
            box_params [B, 22500, 6]  (dx, dy, dw, dl, sin_dh, cos_dh)

    22500 = 5 anchors × 50 × 90 feature map locations
    SOURCED: architecture from Nadeem's thesis Section 3.3
    """

    def __init__(self, in_channels: int = 512) -> None:
        super().__init__()
        num_outputs = NUM_ANCHORS_PER_LOC * 7
        # 7 = 1 objectness + 6 box params per anchor
        # SOURCED: Nadeem thesis Section 3.3

        self.conv = nn.Conv2d(
            in_channels,
            num_outputs,
            kernel_size=3,
            padding=1
        )

    def forward(self, feature_map: torch.Tensor):
        """
        Args:
            feature_map: [B, 512, 50, 90]
        Returns:
            objectness: [B, 22500, 1]
            box_params: [B, 22500, 6]
        """
        B = feature_map.shape[0]
        out = self.conv(feature_map)
        # [B, 35, 50, 90]  (35 = 5 anchors × 7 outputs)

        out = out.permute(0, 2, 3, 1).contiguous()
        # [B, 50, 90, 35]

        out = out.view(B, -1, 7)
        # [B, 22500, 7]

        objectness = out[..., :1]
        # [B, 22500, 1]

        box_params = out[..., 1:]
        # [B, 22500, 6]

        return objectness, box_params


# =============================================================================
# IntentionHead
# Unchanged from Nadeem's original
# =============================================================================

class IntentionHead(nn.Module):
    """
    Intention head — predicts 8-class intention logits per anchor.
    Unchanged from Nadeem's original model_vit.py.

    Input:  [B, 512, 50, 90] feature map from backbone
    Output: [B, 22500, 8]  (8 intention class logits per anchor)

    SOURCED: architecture from Nadeem's thesis Section 3.3
    """

    def __init__(self, in_channels: int = 512) -> None:
        super().__init__()
        num_outputs = NUM_ANCHORS_PER_LOC * NUM_INTENTION_CLASSES
        # 5 anchors × 8 classes = 40

        self.conv = nn.Conv2d(
            in_channels,
            num_outputs,
            kernel_size=3,
            padding=1
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, 512, 50, 90]
        Returns:
            intention_logits: [B, 22500, 8]
        """
        B = feature_map.shape[0]
        out = self.conv(feature_map)
        # [B, 40, 50, 90]  (40 = 5 anchors × 8 classes)

        out = out.permute(0, 2, 3, 1).contiguous()
        # [B, 50, 90, 40]

        out = out.view(B, -1, NUM_INTENTION_CLASSES)
        # [B, 22500, 8]

        return out


# =============================================================================
# TrajectoryHead
# NEW — no equivalent in Nadeem's original
# =============================================================================

class TrajectoryHead(nn.Module):
    """
    Trajectory head — predicts 6-mode future trajectories per detected vehicle.

    NEW class added for IntentTrajNet-AV2 multi-task learning.

    How it works:
        1. Receive the BEV feature map [B, 512, 50, 90] from backbone
        2. Receive N detected vehicle box centres in BEV pixel coordinates
        3. Normalise pixel coords to [-1, 1] for F.grid_sample
        4. Bilinear sample feature map at each vehicle location → [N, 512]
        5. Concatenate box params (cx_norm, cy_norm, w_m, h_m) → [N, 516]
        6. Feed into BEVTrajectoryDecoder → [F, N, 60, 4] and [N, F]

    Why bilinear sampling:
        SOURCED: RoI feature extraction standard since Faster R-CNN
        (Ren et al., NeurIPS 2015). RoIAlign (He et al., ICCV 2017)
        showed bilinear interpolation avoids quantisation errors
        that arise from nearest-neighbour grid snapping.

    Why teacher forcing during training:
        SOURCED: standard in two-stage detection+prediction pipelines.
        DeTra (Casas et al., 2024) uses GT locations during training
        to allow trajectory prediction to be learned without being
        degraded by detection errors in early training stages.

    Args:
        backbone_channels: number of channels in feature map (512)
        feature_map_h:     feature map height (50 = 400 / 8)
        feature_map_w:     feature map width  (90 = 720 / 8)
    """

    def __init__(
        self,
        backbone_channels: int = 512,
        # SOURCED: 512 from TwoStreamViTBackbone (Nadeem thesis Section 3.3)

        feature_map_h: int = 50,
        # SOURCED: 400px / stride 8 = 50 (Nadeem thesis Section 3.3)

        feature_map_w: int = 90,
        # SOURCED: 720px / stride 8 = 90 (Nadeem thesis Section 3.3)

        box_feat_dim: int = 4,
        # 4 box parameters: normalised cx, normalised cy, width_m, height_m
        # ASSUMED: standard box feature concatenation. Provides explicit
        # geometric context to the decoder alongside the BEV features.

    ) -> None:
        super().__init__()

        self.backbone_channels = backbone_channels
        self.feature_map_h = feature_map_h
        self.feature_map_w = feature_map_w
        self.box_feat_dim = box_feat_dim

        # Total feature dimension fed into decoder
        # 512 BEV channels + 4 box params = 516
        self.feat_dim = backbone_channels + box_feat_dim
        # SOURCED: 512 from backbone. Box params are 4 additional scalars.

        # Trajectory decoder — adapted from HiVT MLPDecoder
        # Takes [N, 516] → outputs [F, N, 60, 4] and [N, F]
        self.decoder = BEVTrajectoryDecoder(
            feat_dim=self.feat_dim,
            # SOURCED: 516 = 512 backbone + 4 box params
        )

    def _sample_features(
        self,
        feature_map: torch.Tensor,
        box_centers_px: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear sample the feature map at each detected vehicle location.

        F.grid_sample requires coordinates in [-1, 1] where:
            -1 = left/top edge of feature map
            +1 = right/bottom edge of feature map

        The normalisation formula is:
            norm = (pixel / (size - 1)) * 2 - 1

        Args:
            feature_map:    [B, 512, 50, 90]
            box_centers_px: [N, 2] — (col, row) in feature map pixel coords
                            col is x (width dimension, 0-89)
                            row is y (height dimension, 0-49)

        Returns:
            agent_features: [N, 512]
        """
        N = box_centers_px.shape[0]

        if N == 0:
            # No detections — return empty tensor
            return torch.zeros(
                0, self.backbone_channels,
                device=feature_map.device,
                dtype=feature_map.dtype
            )

        # Normalise pixel coordinates to [-1, 1]
        # col (x): divide by (width - 1), scale to [-1, 1]
        # row (y): divide by (height - 1), scale to [-1, 1]
        norm_x = (box_centers_px[:, 0] / (self.feature_map_w - 1)) * 2 - 1
        # [N] — normalised column coordinate
        norm_y = (box_centers_px[:, 1] / (self.feature_map_h - 1)) * 2 - 1
        # [N] — normalised row coordinate

        # F.grid_sample expects grid shape [B, H_out, W_out, 2]
        # We want one sample per agent: H_out=1, W_out=1
        # grid[..., 0] = x coordinate, grid[..., 1] = y coordinate
        grid = torch.stack([norm_x, norm_y], dim=-1)
        # [N, 2]
        grid = grid.view(1, N, 1, 2)
        # [1, N, 1, 2] — batch=1, N sampling locations, 1 point each

        # Expand feature map to match batch dimension
        # F.grid_sample expects [B, C, H, W] but we're processing
        # all N agents from a single frame together
        # feature_map: [B, 512, 50, 90] — take first element if B=1
        # Note: in training we process one frame at a time for trajectory
        # (different frames have different N values)
        feat = feature_map[0:1]
        # [1, 512, 50, 90]

        # Bilinear sampling
        # Output: [1, 512, N, 1]
        sampled = F.grid_sample(
            feat,                    # [1, 512, 50, 90]
            grid,                    # [1, N, 1, 2]
            mode='bilinear',         # smooth interpolation between pixels
            padding_mode='border',   # clamp out-of-bounds to border values
            align_corners=True       # corners correspond to -1 and +1
        )
        # [1, 512, N, 1]

        # Reshape to [N, 512]
        agent_features = sampled.squeeze(0).squeeze(-1).permute(1, 0)
        # [1, 512, N, 1] → [512, N] → [N, 512]

        return agent_features
        # [N, 512]

    def forward(
        self,
        feature_map: torch.Tensor,
        box_centers_px: torch.Tensor,
        box_params_m: torch.Tensor,
        use_gt_boxes: bool = True,
    ) -> tuple:
        """
        Forward pass of TrajectoryHead.

        Args:
            feature_map:    [B, 512, 50, 90]
                            Shared feature map from backbone.

            box_centers_px: [N, 2]
                            Vehicle box centres in feature map pixel coords.
                            (col, row) format where col ∈ [0, 89], row ∈ [0, 49]
                            During training: GT box centres (teacher forcing).
                            During inference: predicted box centres after NMS.

            box_params_m:   [N, 4]
                            Box parameters in metres: (cx_m, cy_m, w_m, l_m)
                            Concatenated to BEV features to give decoder
                            explicit geometric context.

            use_gt_boxes:   bool
                            True during training (teacher forcing with GT boxes).
                            False during inference (use predicted boxes).
                            SOURCED: DeTra (Casas et al., 2024) — GT boxes
                            during training to decouple trajectory learning
                            from detection errors.

        Returns:
            y_hat: [F, N, 60, 4]
                   6-mode trajectory predictions.
                   (µx, µy, bx, by) per timestep in ego frame metres.
                   SOURCED: output format from HiVT MLPDecoder.

            pi: [N, F]
                Raw mode logits. softmax(pi) = mode probabilities.
                SOURCED: HiVT MLPDecoder forward() return value.
        """
        N = box_centers_px.shape[0]

        if N == 0:
            # No vehicles detected — return empty tensors
            # This can happen for frames with no vehicles in the BEV
            F_modes = self.decoder.num_modes
            H = self.decoder.future_steps
            device = feature_map.device
            return (
                torch.zeros(F_modes, 0, H, 4, device=device),
                torch.zeros(0, F_modes, device=device)
            )

        # --- Step 1: Bilinear sample features at box locations ---
        # [N, 512]
        agent_bev_feat = self._sample_features(feature_map, box_centers_px)

        # --- Step 2: Concatenate box parameters ---
        # Box params give the decoder explicit geometric context:
        # where exactly is this vehicle and how large is it.
        # Without this the decoder only sees the 512 BEV features
        # without knowing which specific vehicle they belong to.
        # ASSUMED: concatenating box params is standard in two-stage
        # detection+trajectory pipelines. Provides position and size
        # information that may not be fully encoded in the BEV feature.
        agent_feat = torch.cat([agent_bev_feat, box_params_m], dim=-1)
        # [N, 512] cat [N, 4] → [N, 516]

        # --- Step 3: Decode trajectories ---
        # Feed [N, 516] into BEVTrajectoryDecoder
        # Returns [F, N, 60, 4] and [N, F]
        y_hat, pi = self.decoder(agent_feat)

        return y_hat, pi
        # y_hat: [F, N, 60, 4] — trajectory predictions
        # pi:    [N, F]         — mode logits