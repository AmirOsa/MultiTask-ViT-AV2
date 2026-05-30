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
# V2 TrajectoryHead (decoder_type='mlp'):
#   - Bilinear sample BEV feature at box location → [N, 512]
#   - Concatenate box params (cx,cy,w,l,heading) → [N, 517]
#   - BEVTrajectoryDecoder (MLP) → [F, N, 30, 4]
#   - No social attention, no map sampling
#
# V3 TrajectoryHead (decoder_type='social_mlp'):
#   - Bilinear sample BEV feature at box location → [N, 512]
#   - Bilinear sample Map BEV at box location → [N, 9]
#   - Concatenate box params (cx,cy,w,l,heading) → [N, 526]
#   - Linear(526→256) + LayerNorm + ReLU → [N, 256]
#   - Social Attention across N agents → [N, 256]
#   - BEVTrajectoryDecoder (MLP) → [F, N, 30, 4]
#   SOURCED: DeTra object self-attention (Casas et al. 2024)
#            HiVT global interaction module (Zhou et al. 2022)

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
    MAP_CHANNELS,
)
from models.trajectory_decoder import (
    BEVTrajectoryDecoder,
    TransformerTrajectoryDecoder,
    SocialAttention,
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
# TrajectoryHead — V2 and V3
# =============================================================================

class TrajectoryHead(nn.Module):
    """
    Trajectory head — predicts 6-mode future trajectories per detected vehicle.

    Supports three decoder types controlled by decoder_type parameter:

    'mlp' (V2):
        - Bilinear sample BEV feature at box location → [N, 512]
        - Concatenate box params (cx,cy,w,l,heading) → [N, 517]
        - BEVTrajectoryDecoder (MLP) → [F, N, 30, 4]
        - No social attention, no explicit map sampling

    'social_mlp' (V3):
        - Bilinear sample BEV feature at box location → [N, 512]
        - Bilinear sample Map BEV at box location → [N, 9]
        - Concatenate box params (cx,cy,w,l,heading) → [N, 526]
        - Linear(526→256) + LayerNorm + ReLU → [N, 256]
        - Social Attention across N agents → [N, 256]
        - BEVTrajectoryDecoder (MLP) → [F, N, 30, 4]
        SOURCED: DeTra object self-attention (Casas et al. 2024)
                 HiVT global interaction module (Zhou et al. 2022)

    'transformer' (old V3 — parquet based, kept for reference):
        - Full transformer system with GRU + social + cross-attention

    Both MLP decoders produce identical output shapes so downstream
    loss and eval code works unchanged.

    Args:
        backbone_channels: 512 — SOURCED: Nadeem thesis Section 3.3
        feature_map_h:     50  — SOURCED: 400/8
        feature_map_w:     90  — SOURCED: 720/8
        decoder_type:      'mlp', 'social_mlp', or 'transformer'
        box_feat_dim:      5   — (cx_m, cy_m, w_m, l_m, heading)
                                 heading included for turn prediction
                                 SOURCED: DeTra pose (x, y, θ)
    """

    def __init__(
        self,
        backbone_channels: int = 512,
        feature_map_h: int = 50,
        feature_map_w: int = 90,
        decoder_type: str = 'mlp',
        # 'mlp'         → V2
        # 'social_mlp'  → V3
        # 'transformer' → old parquet-based V3
        box_feat_dim: int = 5,
        # Box parameters passed to trajectory decoder
        # 5 = (cx_m, cy_m, w_m, l_m, heading)
        # heading included — critical for turn prediction
        # SOURCED: DeTra pose representation (x, y, θ)
        mlp_dropout: float = 0.0,

        # Transformer decoder hyperparameters (kept for backward compatibility)
        gru_hidden: int = 64,
        num_heads: int = 8,
        num_decoder_layers: int = 2,
        social_heads: int = 4,
        social_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.backbone_channels = backbone_channels
        self.feature_map_h = feature_map_h
        self.feature_map_w = feature_map_w
        self.decoder_type = decoder_type
        self.box_feat_dim = box_feat_dim

        # =====================================================================
        # Decoder — selected by decoder_type
        # =====================================================================
        if decoder_type == 'transformer':
            # -----------------------------------------------------------------
            # Old parquet-based V3 — kept for backward compatibility
            # GRU history + social attention + cross-attention decoder
            # -----------------------------------------------------------------
            self.decoder = TransformerTrajectoryDecoder(
                bev_channels=backbone_channels,
                gru_hidden=gru_hidden,
                hidden_size=TRAJECTORY_DECODER_HIDDEN,
                num_heads=num_heads,
                num_decoder_layers=num_decoder_layers,
                future_steps=TRAJECTORY_FUTURE_STEPS,
                num_modes=TRAJECTORY_NUM_MODES,
                min_scale=0.001,
                social_heads=social_heads,
                social_layers=social_layers,
                dropout=dropout,
            )
            print(
                f"TrajectoryHead: transformer decoder "
                f"(gru_hidden={gru_hidden}, "
                f"num_heads={num_heads}, "
                f"decoder_layers={num_decoder_layers}, "
                f"social_heads={social_heads})"
            )

        elif decoder_type == 'social_mlp':
            # -----------------------------------------------------------------
            # V3 — Social + Map-Aware MLP decoder
            # BEV feature + map BEV + box params → projection →
            # social attention → MLP decoder
            # SOURCED: DeTra object self-attention (Casas et al. 2024)
            #          HiVT global interaction module (Zhou et al. 2022)
            # -----------------------------------------------------------------
            social_feat_dim = backbone_channels + MAP_CHANNELS + box_feat_dim
            # 512 + 9 + 5 = 526

            self.input_proj = nn.Sequential(
                nn.Linear(social_feat_dim, TRAJECTORY_DECODER_HIDDEN),
                nn.LayerNorm(TRAJECTORY_DECODER_HIDDEN),
                nn.ReLU(inplace=True),
            )
            # Projects [N, 526] → [N, 256]
            # Projection necessary to match SocialAttention hidden_size
            # and reduce dimensionality before attention

            self.social_attn = SocialAttention(
                hidden_size=TRAJECTORY_DECODER_HIDDEN,
                # 256 — matches projected feature dimension
                num_heads=4,
                # 4 heads — 256/4=64 per head ✓
                # SOURCED: DeTra social_heads=4 in ablation
                num_layers=1,
                # Lightweight — single layer sufficient for N~10-30 agents
                dropout=mlp_dropout,
            )

            self.decoder = BEVTrajectoryDecoder(
                feat_dim=TRAJECTORY_DECODER_HIDDEN,
                # 256 — after projection and social attention
                hidden_size=TRAJECTORY_DECODER_HIDDEN,
                future_steps=TRAJECTORY_FUTURE_STEPS,
                num_modes=TRAJECTORY_NUM_MODES,
                min_scale=0.001,
                dropout=mlp_dropout,
            )
            print(
                f"TrajectoryHead: social_mlp decoder "
                f"(feat_dim={social_feat_dim}→{TRAJECTORY_DECODER_HIDDEN}, "
                f"social_heads=4, map_sampling=True)"
            )

        else:
            # -----------------------------------------------------------------
            # V2 — MLP decoder
            # Simple BEV feature sampling + box params + MLP
            # SOURCED: adapted from HiVT MLPDecoder (Zhou et al. 2022)
            # -----------------------------------------------------------------
            mlp_feat_dim = backbone_channels + box_feat_dim
            # 512 + 5 = 517
            self.decoder = BEVTrajectoryDecoder(
                feat_dim=mlp_feat_dim,
                hidden_size=TRAJECTORY_DECODER_HIDDEN,
                future_steps=TRAJECTORY_FUTURE_STEPS,
                num_modes=TRAJECTORY_NUM_MODES,
                min_scale=0.001,
                dropout=mlp_dropout,
            )
            print(f"TrajectoryHead: MLP decoder (feat_dim={mlp_feat_dim})")

    def _sample_features(
        self,
        feature_map: torch.Tensor,
        box_centers_px: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear sample the feature map at each detected vehicle location.

        Works for any number of input channels — used for both LiDAR BEV
        feature map [B, 512, 50, 90] and Map BEV [B, 9, 400, 720].

        SOURCED: RoIAlign (He et al., ICCV 2017) — bilinear interpolation
        avoids quantisation errors from nearest-neighbour snapping.

        Args:
            feature_map:    [B, C, H, W] — any channel count C
            box_centers_px: [N, 2] — (col, row) in feature map pixel coords

        Returns:
            agent_features: [N, C]
        """
        N = box_centers_px.shape[0]
        C = feature_map.shape[1]
        H = feature_map.shape[2]
        W = feature_map.shape[3]

        if N == 0:
            return torch.zeros(
                0, C,
                device=feature_map.device,
                dtype=feature_map.dtype
            )

        # Normalise to [-1, 1] for F.grid_sample
        norm_x = (box_centers_px[:, 0] / (W - 1)) * 2 - 1
        norm_y = (box_centers_px[:, 1] / (H - 1)) * 2 - 1

        grid = torch.stack([norm_x, norm_y], dim=-1)
        grid = grid.view(1, N, 1, 2)
        # [1, N, 1, 2]

        feat = feature_map[0:1]
        # [1, C, H, W]

        sampled = F.grid_sample(
            feat, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        # [1, C, N, 1]

        agent_features = sampled.squeeze(0).squeeze(-1).permute(1, 0)
        # [N, C]

        return agent_features

    def forward(
        self,
        feature_map: torch.Tensor,
        map_bev: torch.Tensor | None = None,
        box_centers_px: torch.Tensor = None,
        box_params_m: torch.Tensor = None,
        agent_history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        use_gt_boxes: bool = True,
    ) -> tuple:
        """
        Forward pass of TrajectoryHead.

        Args:
            feature_map:    [B, 512, 50, 90]
                            Shared LiDAR+map feature map from backbone.

            map_bev:        [B, 9, 400, 720] or None
                            Raw HD map BEV — used by social_mlp (V3) only
                            for explicit map context at agent locations.
                            None for V2 mlp decoder.

            box_centers_px: [N, 2]
                            Vehicle box centres in feature map pixel coords.
                            (col, row) format.
                            During training: GT box centres (teacher forcing).
                            SOURCED: DeTra (Casas et al., 2024)

            box_params_m:   [N, 5]
                            Box parameters: (cx_m, cy_m, w_m, l_m, heading)
                            heading included for turn prediction.
                            SOURCED: DeTra pose representation (x, y, θ)

            agent_history:  [N, 50, 5] or None
                            Agent history from parquet files.
                            Required for transformer decoder only.
                            None for mlp and social_mlp decoders.

            history_mask:   [N, 50] or None
                            True where history is valid.
                            Not yet used — future improvement.

            use_gt_boxes:   bool
                            True during training (teacher forcing).
                            False during inference.

        Returns:
            y_hat: [F, N, 30, 4] — trajectory predictions (µx,µy,bx,by)
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

        # --- Step 1: Sample LiDAR+map BEV features at agent locations ---
        # Same for all decoder types
        agent_bev_feat = self._sample_features(feature_map, box_centers_px)
        # [N, 512]

        # --- Step 2: Decode trajectories ---
        if self.decoder_type == 'transformer':
            # -----------------------------------------------------------------
            # Old transformer decoder — parquet based
            # -----------------------------------------------------------------
            if agent_history is None:
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
                agent_bev_features=agent_bev_feat,
                agent_history=agent_history,
                history_mask=history_mask,
            )

        elif self.decoder_type == 'social_mlp':
            # -----------------------------------------------------------------
            # V3 — Social + Map-Aware MLP decoder
            # -----------------------------------------------------------------

            # Sample raw map BEV at agent locations for explicit map context
            if map_bev is not None:
                # Map BEV is full resolution [B, 9, 400, 720]
                # Need to convert box_centers_px from feature map coords
                # (50×90) to map BEV coords (400×720) by multiplying by stride=8
                map_centers_px = box_centers_px * 8.0
                # [N, 2] — now in 400×720 map BEV pixel coords
                agent_map_feat = self._sample_features(
                    map_bev, map_centers_px
                )
                # [N, 9]
            else:
                agent_map_feat = torch.zeros(
                    N, MAP_CHANNELS,
                    device=feature_map.device,
                    dtype=feature_map.dtype
                )

            # Concatenate BEV feature + map feature + box params
            agent_feat = torch.cat(
                [agent_bev_feat, agent_map_feat, box_params_m], dim=-1
            )
            # [N, 512+9+5] = [N, 526]

            # Project to hidden size
            agent_feat = self.input_proj(agent_feat)
            # [N, 256]

            # Social attention — agents attend to each other
            # SOURCED: DeTra object self-attention
            #          HiVT global interaction module
            agent_feat = self.social_attn(agent_feat)
            # [N, 256]

            # MLP decoder
            y_hat, pi = self.decoder(agent_feat)
            # y_hat: [F, N, 30, 4]
            # pi:    [N, F]

        else:
            # -----------------------------------------------------------------
            # V2 — MLP decoder
            # -----------------------------------------------------------------
            agent_feat = torch.cat([agent_bev_feat, box_params_m], dim=-1)
            # [N, 512+5] = [N, 517]

            y_hat, pi = self.decoder(agent_feat)
            # y_hat: [F, N, 30, 4]
            # pi:    [N, F]

        return y_hat, pi