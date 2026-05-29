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
# V2/V3 TrajectoryHead:
#   - Bilinear sample BEV feature at box location → [N, 512]
#   - Concatenate box params → [N, 516]
#   - Feed into BEVTrajectoryDecoder (MLP) → [F, N, 60, 4]
#
# V4/V5 TrajectoryHead (decoder_type='transformer'):
#   - Bilinear sample BEV feature at box location → [N, 512]
#   - Load agent history [N, 50, 5] from parquet
#   - Feed into TransformerTrajectoryDecoder:
#       GRU encodes history → [N, 64]
#       Concatenate with BEV → [N, 576]
#       Social attention across N agents
#       Mode queries cross-attend to full BEV map
#   → [F, N, 60, 4] + [N, F]

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import (
    NUM_ANCHORS_PER_LOC,
    NUM_INTENTION_CLASSES,
    GRID_HEIGHT_PX,
    GRID_WIDTH_PX,
    TRAJECTORY_DECODER_HIDDEN,
    TRAJECTORY_FUTURE_STEPS,
    TRAJECTORY_NUM_MODES,
    AGENT_HISTORY_FEATURES,
    AGENT_HISTORY_STEPS,
)
from models.trajectory_decoder import (
    BEVTrajectoryDecoder,
    TransformerTrajectoryDecoder,
)


# =============================================================================
# DetectionHead — unchanged from Nadeem
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
        self.conv = nn.Conv2d(
            in_channels, num_outputs, kernel_size=3, padding=1
        )

    def forward(self, feature_map: torch.Tensor):
        B = feature_map.shape[0]
        out = self.conv(feature_map)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(B, -1, 7)
        objectness = out[..., :1]
        box_params = out[..., 1:]
        return objectness, box_params


# =============================================================================
# IntentionHead — unchanged from Nadeem
# =============================================================================

class IntentionHead(nn.Module):
    """
    Intention head — predicts 8-class intention logits per anchor.
    Unchanged from Nadeem's original model_vit.py.

    Input:  [B, 512, 50, 90] feature map from backbone
    Output: [B, 22500, 8]

    SOURCED: architecture from Nadeem's thesis Section 3.3
    """

    def __init__(self, in_channels: int = 512) -> None:
        super().__init__()
        num_outputs = NUM_ANCHORS_PER_LOC * NUM_INTENTION_CLASSES
        self.conv = nn.Conv2d(
            in_channels, num_outputs, kernel_size=3, padding=1
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        B = feature_map.shape[0]
        out = self.conv(feature_map)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(B, -1, NUM_INTENTION_CLASSES)
        return out


# =============================================================================
# TrajectoryHead — updated for V4/V5
# =============================================================================

class TrajectoryHead(nn.Module):
    """
    Trajectory head — predicts 6-mode future trajectories per detected vehicle.

    Supports two decoder types controlled by decoder_type parameter:

    'mlp' (V2, V3, V3-traj ablation):
        - Bilinear sample BEV at box location → [N, 512]
        - Concatenate box params → [N, 516]
        - BEVTrajectoryDecoder (MLP) → [F, N, 60, 4]
        - No history, no social attention

    'transformer' (V4, V5):
        - Bilinear sample BEV at box location → [N, 512]
        - GRU encode history [N, 50, 5] → [N, 64]
        - Concatenate → [N, 576]
        - Social attention across N agents
        - Mode queries cross-attend to full BEV map
        - TransformerTrajectoryDecoder → [F, N, 60, 4]

    Both decoders produce identical output shapes so downstream
    loss and eval code works unchanged.

    Args:
        backbone_channels: 512 — SOURCED: Nadeem thesis Section 3.3
        feature_map_h:     50  — SOURCED: 400/8
        feature_map_w:     90  — SOURCED: 720/8
        decoder_type:      'mlp' or 'transformer'
        box_feat_dim:      4   — (cx_norm, cy_norm, w_m, h_m)
                                 only used for MLP decoder
    """

    def __init__(
        self,
        backbone_channels: int = 512,
        feature_map_h: int = 50,
        feature_map_w: int = 90,
        decoder_type: str = 'mlp',
        # 'mlp'         → V2, V3, V3-traj
        # 'transformer' → V4, V5
        box_feat_dim: int = 4,
        # Only used for MLP decoder concatenation
        # 4 = (cx_m, cy_m, w_m, l_m)
        mlp_dropout: float = 0.0,

        # Transformer decoder hyperparameters
        # All confirmed by 3-epoch ablation
        gru_hidden: int = 64,
        # ASSUMED: confirmed by ablation {32, 64, 128}
        num_heads: int = 8,
        # ASSUMED: confirmed by ablation {4, 8}
        num_decoder_layers: int = 2,
        # ASSUMED: confirmed by ablation {1, 2, 3}
        social_heads: int = 4,
        # ASSUMED: confirmed by ablation {2, 4}
        social_layers: int = 1,
        # ASSUMED: lightweight for 100 scenarios
        dropout: float = 0.1,
        # Regularization for small dataset
    ) -> None:
        super().__init__()

        self.backbone_channels = backbone_channels
        self.feature_map_h = feature_map_h
        self.feature_map_w = feature_map_w
        self.decoder_type = decoder_type

        # =====================================================================
        # Decoder — selected by decoder_type
        # =====================================================================
        if decoder_type == 'transformer':
            # -----------------------------------------------------------------
            # V4/V5 — Full transformer system
            # GRU history + social attention + cross-attention decoder
            # -----------------------------------------------------------------
            self.decoder = TransformerTrajectoryDecoder(
                bev_channels=backbone_channels,
                # 512 — SOURCED: Nadeem thesis Section 3.3
                gru_hidden=gru_hidden,
                # 64 — ASSUMED: confirmed by ablation
                hidden_size=TRAJECTORY_DECODER_HIDDEN,
                # 256 — SOURCED: V2 ablation
                num_heads=num_heads,
                # 8 — ASSUMED: confirmed by ablation
                num_decoder_layers=num_decoder_layers,
                # 2 — ASSUMED: confirmed by ablation
                future_steps=TRAJECTORY_FUTURE_STEPS,
                # 60 — SOURCED: Abdulbaki Section 3.1
                num_modes=TRAJECTORY_NUM_MODES,
                # 6 — SOURCED: Abdulbaki Section 3.4.3
                min_scale=0.001,
                # SOURCED: HiVT decoder.py
                social_heads=social_heads,
                # 4 — ASSUMED: confirmed by ablation
                social_layers=social_layers,
                # 1 — ASSUMED: lightweight
                dropout=dropout,
            )
            print(
                f"TrajectoryHead: transformer decoder "
                f"(gru_hidden={gru_hidden}, "
                f"num_heads={num_heads}, "
                f"decoder_layers={num_decoder_layers}, "
                f"social_heads={social_heads})"
            )

        else:
            # -----------------------------------------------------------------
            # V2/V3/V3-traj — MLP decoder
            # Simple BEV feature sampling + MLP
            # SOURCED: adapted from HiVT MLPDecoder
            # -----------------------------------------------------------------
            mlp_feat_dim = backbone_channels + box_feat_dim
            # 512 + 4 = 516
            self.decoder = BEVTrajectoryDecoder(
                feat_dim=mlp_feat_dim,
                hidden_size=TRAJECTORY_DECODER_HIDDEN,
                future_steps=TRAJECTORY_FUTURE_STEPS,
                num_modes=TRAJECTORY_NUM_MODES,
                min_scale=0.001,
                dropout=mlp_dropout,
            )
            print(f"TrajectoryHead: MLP decoder (feat_dim={mlp_feat_dim})")

        self.box_feat_dim = box_feat_dim

    def _sample_features(
        self,
        feature_map: torch.Tensor,
        box_centers_px: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear sample the feature map at each detected vehicle location.

        Same for both MLP and transformer decoders.
        SOURCED: RoIAlign (He et al., ICCV 2017) — bilinear interpolation
        avoids quantisation errors from nearest-neighbour snapping.

        Args:
            feature_map:    [B, 512, 50, 90]
            box_centers_px: [N, 2] — (col, row) in feature map pixel coords

        Returns:
            agent_features: [N, 512]
        """
        N = box_centers_px.shape[0]

        if N == 0:
            return torch.zeros(
                0, self.backbone_channels,
                device=feature_map.device,
                dtype=feature_map.dtype
            )

        # Normalise to [-1, 1] for F.grid_sample
        norm_x = (box_centers_px[:, 0] / (self.feature_map_w - 1)) * 2 - 1
        norm_y = (box_centers_px[:, 1] / (self.feature_map_h - 1)) * 2 - 1

        grid = torch.stack([norm_x, norm_y], dim=-1)
        grid = grid.view(1, N, 1, 2)
        # [1, N, 1, 2]

        feat = feature_map[0:1]
        # [1, 512, 50, 90]

        sampled = F.grid_sample(
            feat, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        # [1, 512, N, 1]

        agent_features = sampled.squeeze(0).squeeze(-1).permute(1, 0)
        # [N, 512]

        return agent_features

    def forward(
        self,
        feature_map: torch.Tensor,
        box_centers_px: torch.Tensor,
        box_params_m: torch.Tensor,
        agent_history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        use_gt_boxes: bool = True,
    ) -> tuple:
        """
        Forward pass of TrajectoryHead.

        Args:
            feature_map:    [B, 512, 50, 90]
                            Shared feature map from backbone.

            box_centers_px: [N, 2]
                            Vehicle box centres in feature map pixel coords.
                            (col, row) format.
                            During training: GT box centres (teacher forcing).
                            During inference: predicted box centres after NMS.

            box_params_m:   [N, 4]
                            Box parameters: (cx_m, cy_m, w_m, l_m)
                            Used by MLP decoder only.

            agent_history:  [N, 50, 5] or None
                            Agent history from parquet files.
                            Required for transformer decoder (V4/V5).
                            None for MLP decoder (V2/V3).
                            SOURCED: Abdulbaki thesis Section 3.6

            history_mask:   [N, 50] or None
                            True where history is valid.
                            Not yet used — future improvement.

            use_gt_boxes:   bool
                            True during training (teacher forcing).
                            False during inference.
                            SOURCED: DeTra (Casas et al., 2024)

        Returns:
            y_hat: [F, N, 60, 4] — trajectory predictions (µx,µy,bx,by)
            pi:    [N, F]         — mode logits
        """
        N = box_centers_px.shape[0]

        if N == 0:
            F_modes = TRAJECTORY_NUM_MODES
            H = TRAJECTORY_FUTURE_STEPS
            device = feature_map.device
            return (
                torch.zeros(F_modes, 0, H, 4, device=device),
                torch.zeros(0, F_modes, device=device)
            )

        # --- Step 1: Bilinear sample features at box locations ---
        # Same for both decoder types
        agent_bev_feat = self._sample_features(feature_map, box_centers_px)
        # [N, 512]

        # --- Step 2: Decode trajectories ---
        if self.decoder_type == 'transformer':
            # Transformer decoder needs full BEV map + history
            if agent_history is None:
                # Fallback — zero history if not provided
                # Should not happen in normal V4/V5 training
                agent_history = torch.zeros(
                    N, AGENT_HISTORY_STEPS, AGENT_HISTORY_FEATURES,
                    device=feature_map.device,
                    dtype=feature_map.dtype
                )
                print(
                    "WARNING: agent_history is None for transformer decoder. "
                    "Using zero history — check data pipeline."
                )

            y_hat, pi = self.decoder(
                bev_feature_map=feature_map,
                # Full BEV map for cross-attention
                agent_bev_features=agent_bev_feat,
                # [N, 512] sampled features
                agent_history=agent_history,
                # [N, 50, 5] from parquet
                history_mask=history_mask,
            )

        else:
            # MLP decoder — concatenate box params and decode
            agent_feat = torch.cat([agent_bev_feat, box_params_m], dim=-1)
            # [N, 512] cat [N, 4] → [N, 516]

            y_hat, pi = self.decoder(agent_feat)

        return y_hat, pi
        # y_hat: [F, N, 60, 4]
        # pi:    [N, F]