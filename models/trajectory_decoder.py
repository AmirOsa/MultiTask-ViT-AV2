# models/trajectory_decoder.py
#
# Contains two trajectory decoders:
#
#   BEVTrajectoryDecoder  — MLP decoder, used in V2 and V3
#   Adapted from HiVT MLPDecoder (Zhou et al., CVPR 2022)
#   SOURCED: HiVT repo — models/decoder.py
#
#   TransformerTrajectoryDecoder — NEW for V4 and V5
#   Transformer cross-attention decoder with:
#     - GRU history encoder [N, 50, 5] → [N, 64]
#     - Social attention across N agents
#     - Mode queries cross-attending to full BEV feature map
#   SOURCED conceptually: Abdulbaki thesis Section 5.3 — transformer
#   decoder explicitly listed as future work.
#   SOURCED: Vaswani et al. (2017) — Attention Is All You Need
#   SOURCED: history features — Abdulbaki thesis Section 3.6
#
# Modifications to BEVTrajectoryDecoder vs original:
#   1. Replaced dual-input (local+global embed) with single agent feature
#   2. Added learnable mode_queries replacing global_embed
#   3. Added agent_proj layer (feat_dim=516 → hidden_size=256)
#   4. hidden_size changed from 64 → 256
#   5. BUG FIX: renamed F → num_modes to avoid shadowing F.elu_()

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import (
    TRAJECTORY_FUTURE_STEPS,       # SOURCED: Abdulbaki thesis Section 3.1
    TRAJECTORY_NUM_MODES,          # SOURCED: Abdulbaki thesis Section 3.4.3
    TRAJECTORY_MIN_SCALE,          # SOURCED: HiVT decoder.py
    TRAJECTORY_DECODER_HIDDEN,     # Confirmed by V2 ablation — 256
    AGENT_HISTORY_STEPS,           # SOURCED: Abdulbaki thesis Section 3.1
    AGENT_HISTORY_FEATURES,        # SOURCED: Abdulbaki thesis Section 3.6
)


def _init_weights(module: nn.Module) -> None:
    """
    Xavier uniform initialisation for linear layers.
    SOURCED: identical to init_weights() in HiVT utils.py
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =============================================================================
# BEVTrajectoryDecoder — MLP decoder for V2 and V3
# Unchanged from original except variable rename fix
# =============================================================================

class BEVTrajectoryDecoder(nn.Module):
    """
    Multi-modal MLP trajectory decoder for V2 and V3.

    Adapted from HiVT MLPDecoder (Zhou et al., CVPR 2022).
    SOURCED: HiVT repo models/decoder.py

    Input:  agent_feat [N, feat_dim=516]
            (512 BEV feature + 4 box params)
    Output: y_hat [num_modes, N, H, 4] — Laplace (µx,µy,bx,by)
            pi    [N, num_modes]        — mode logits
    """

    def __init__(
        self,
        feat_dim: int = 516,
        hidden_size: int = TRAJECTORY_DECODER_HIDDEN,
        future_steps: int = TRAJECTORY_FUTURE_STEPS,
        num_modes: int = TRAJECTORY_NUM_MODES,
        uncertain: bool = True,
        min_scale: float = TRAJECTORY_MIN_SCALE,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        # Learnable mode queries — replace HiVT's global_embed
        self.mode_queries = nn.Parameter(
            torch.randn(num_modes, hidden_size)
        )

        # Project agent feature to hidden_size
        self.agent_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Aggregation MLP
        self.aggr_embed = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Location head
        self.loc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, future_steps * 2),
        )

        # Scale head
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, future_steps * 2),
            )

        # Mode probability head
        self.pi = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.apply(_init_weights)
        nn.init.normal_(self.mode_queries, mean=0.0, std=0.1)

    def forward(
        self,
        agent_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_feat: [N, feat_dim]
        Returns:
            y_hat: [num_modes, N, H, 4]
            pi:    [N, num_modes]
        """
        N = agent_feat.shape[0]
        num_modes = self.num_modes

        agent_hidden = self.agent_proj(agent_feat)
        # [N, hidden_size]

        mode_q = self.mode_queries.unsqueeze(1).expand(
            num_modes, N, self.hidden_size
        )
        # [num_modes, N, hidden_size]

        agent_expanded = agent_hidden.unsqueeze(0).expand(
            num_modes, N, self.hidden_size
        )
        # [num_modes, N, hidden_size]

        pi_input = torch.cat([mode_q, agent_expanded], dim=-1)
        pi = self.pi(pi_input).squeeze(-1).t()
        # [N, num_modes]

        aggr_input = torch.cat([mode_q, agent_expanded], dim=-1)
        out = self.aggr_embed(aggr_input)
        # [num_modes, N, hidden_size]

        loc = self.loc(out).view(num_modes, N, self.future_steps, 2)
        # [num_modes, N, H, 2]

        if self.uncertain:
            scale = self.scale(out).view(num_modes, N, self.future_steps, 2)
            scale = F.elu_(scale, alpha=1.0) + 1.0 + self.min_scale
            y_hat = torch.cat([loc, scale], dim=-1)
            # [num_modes, N, H, 4]
        else:
            y_hat = loc

        return y_hat, pi


# =============================================================================
# GRUHistoryEncoder — NEW for V4/V5
# Encodes 5-second agent history [N, 50, 5] → [N, 64]
# =============================================================================

class GRUHistoryEncoder(nn.Module):
    """
    GRU-based agent history encoder for V4 and V5.

    Encodes 5 seconds of past motion per agent into a fixed-size
    history vector. The GRU naturally emphasizes recent timesteps
    over older ones — appropriate for trajectory prediction where
    recent motion is more predictive than distant past.

    Input:  history [N, history_steps, history_features]
            = [N, 50, 5] where 5 = (x, y, vx, vy, heading)
    Output: history_vector [N, hidden_size]

    Why GRU over transformer for history encoding:
        - 50 timesteps is short — GRU handles well without attention overhead
        - Natural recency bias matches trajectory prediction needs
        - Fewer parameters — less overfitting risk on 100 scenarios
        - Proven in Social-LSTM, GRIP, and other trajectory papers
        SOURCED: Social-LSTM (Alahi et al., CVPR 2016)

    Args:
        input_size:   5 — SOURCED: Abdulbaki thesis Section 3.6
                      f^t_i = [x, y, vx, vy, heading] ∈ R^5
        hidden_size:  64 — ASSUMED: lightweight encoder
                      Confirmed by 3-epoch ablation over {32, 64, 128}
        num_layers:   1 — standard for short sequences
    """

    def __init__(
        self,
        input_size: int = AGENT_HISTORY_FEATURES,
        # SOURCED: 5 — Abdulbaki thesis Section 3.6
        hidden_size: int = 64,
        # ASSUMED: confirmed by ablation
        num_layers: int = 1,
        # Standard for short 50-step sequences
        dropout: float = 0.1,
        # Regularization — important with only 100 training scenarios
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            # GRU dropout only applies between layers, not useful for 1 layer
        )

        # Layer norm on output for training stability
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            history:      [N, 50, 5] — agent history sequences
                          positions and velocities in city frame
                          SOURCED: Abdulbaki thesis Section 3.6
            history_mask: [N, 50] — True where history is valid
                          False where agent wasn't observed
                          (not yet used — future improvement)

        Returns:
            history_vec: [N, hidden_size] — encoded history per agent
        """
        N = history.shape[0]

        if N == 0:
            return torch.zeros(
                0, self.hidden_size,
                device=history.device,
                dtype=history.dtype
            )

        # GRU forward — take final hidden state
        # Output: (all_hidden [N, 50, hidden], final_hidden [1, N, hidden])
        _, final_hidden = self.gru(history)
        # final_hidden: [num_layers, N, hidden_size]

        # Take last layer's hidden state
        history_vec = final_hidden[-1]
        # [N, hidden_size]

        return self.norm(history_vec)
        # [N, hidden_size]


# =============================================================================
# SocialAttention — NEW for V4/V5
# Self-attention across N agents for social context
# =============================================================================

class SocialAttention(nn.Module):
    """
    Social attention layer — agents attend to each other.

    Each agent's feature vector attends to every other agent's
    feature vector. This encodes social context:
        - Is there a vehicle blocking the path?
        - Is oncoming traffic preventing a turn?
        - Are vehicles moving together in a platoon?

    SOURCED conceptually: HiVT (Zhou et al., CVPR 2022) — global
    interaction module via graph transformer. We use simpler
    self-attention without graph structure.

    SOURCED: Vaswani et al. (2017) — self-attention mechanism.

    Args:
        hidden_size: 256 — matches agent feature dimension
        num_heads:   4 — ASSUMED: smaller than decoder heads
                     N is small (~10-30 agents), 4 heads sufficient
                     Confirmed by 3-epoch ablation over {2, 4}
        num_layers:  1 — ASSUMED: lightweight, avoids overfitting
                     on 100 training scenarios
        dropout:     0.1 — regularization
    """

    def __init__(
        self,
        hidden_size: int = TRAJECTORY_DECODER_HIDDEN,
        # 256 — matches agent feature dimension
        num_heads: int = 4,
        # ASSUMED: confirmed by ablation
        # must divide hidden_size evenly: 256/4=64 ✓
        num_layers: int = 1,
        # ASSUMED: lightweight for small dataset
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # Standard transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            # Standard 4× expansion
            dropout=dropout,
            batch_first=True,
            # [N, seq, feat] format
            norm_first=True,
            # Pre-norm for training stability
            # SOURCED: Pre-LN transformer — Wang et al. (2019)
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        agent_features: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            agent_features:   [N, hidden_size] — per-agent features
            key_padding_mask: [N] — True for padding agents (not used yet)

        Returns:
            social_features: [N, hidden_size] — socially aware features
        """
        N = agent_features.shape[0]

        if N == 0:
            return agent_features

        if N == 1:
            # Single agent — no social context possible
            # Return unchanged with norm applied
            return self.norm(agent_features)

        # Add sequence dimension for transformer
        # [N, hidden_size] → [1, N, hidden_size]
        # We treat N agents as a sequence of length N
        x = agent_features.unsqueeze(0)
        # [1, N, hidden_size]

        # Self-attention across all N agents
        x = self.transformer(x)
        # [1, N, hidden_size]

        # Remove sequence dimension
        x = x.squeeze(0)
        # [N, hidden_size]

        return self.norm(x)
        # [N, hidden_size]


# =============================================================================
# TransformerTrajectoryDecoder — NEW for V4/V5
# Full trajectory system with GRU + social attention + cross-attention decoder
# =============================================================================

class TransformerTrajectoryDecoder(nn.Module):
    """
    Full transformer-based trajectory decoder for V4 and V5.

    Three-stage pipeline:
        1. GRU history encoder: [N, 50, 5] → [N, 64]
        2. Social attention: [N, 576] → [N, 576] (agents see each other)
        3. Cross-attention decoder: mode queries × BEV map → trajectories

    Architecture:
        BEV feature (sampled):  [N, 512]
        History vector (GRU):   [N,  64]
        Combined:               [N, 576]
        Projection:             [N, 256]  ← matches hidden_size
        Social attention:       [N, 256]  ← agents aware of each other
        Mode queries:           [F, 256]  ← F=6 learnable vectors
        Cross-attention:        [F, N, 256]  ← modes attend to BEV map
        Location head:          [F, N, 60, 2]
        Scale head:             [F, N, 60, 2]
        Output:                 [F, N, 60, 4] + [N, F]

    Why cross-attention over full BEV vs single sampled point (V2/V3):
        Single point: decoder sees only local features at vehicle location
        Full BEV map: decoder sees lane boundaries, intersections, obstacles
        A left turn prediction needs to know there's an intersection ahead
        — impossible from a single sampled point.
        SOURCED: cross-attention for trajectory — Wayformer (Nayakanti et al.,
        ICRA 2023); MotionDiffuser (Jiang et al., ICRA 2023)

    Args:
        feat_dim:         576 — BEV(512) + history(64)
        hidden_size:      256 — SOURCED: V2 ablation
        num_heads:        8   — ASSUMED: confirmed by ablation
        num_decoder_layers: 2 — ASSUMED: confirmed by ablation
        future_steps:     60  — SOURCED: Abdulbaki Section 3.1
        num_modes:        6   — SOURCED: Abdulbaki Section 3.4.3
        min_scale:        1e-3 — SOURCED: HiVT decoder.py
        gru_hidden:       64  — ASSUMED: confirmed by ablation
        social_heads:     4   — ASSUMED: confirmed by ablation
        social_layers:    1   — ASSUMED: confirmed by ablation
        dropout:          0.1 — regularization for small dataset
    """

    def __init__(
        self,
        bev_channels: int = 512,
        # SOURCED: backbone output channels — Nadeem thesis Section 3.3
        gru_hidden: int = 64,
        # ASSUMED: confirmed by ablation {32, 64, 128}
        hidden_size: int = TRAJECTORY_DECODER_HIDDEN,
        # SOURCED: V2 ablation confirmed 256
        num_heads: int = 8,
        # ASSUMED: confirmed by ablation {4, 8}
        # must divide hidden_size: 256/8=32 ✓
        num_decoder_layers: int = 2,
        # ASSUMED: conservative, confirmed by ablation {1, 2, 3}
        future_steps: int = TRAJECTORY_FUTURE_STEPS,
        # SOURCED: Abdulbaki thesis Section 3.1 — 60 steps = 6s
        num_modes: int = TRAJECTORY_NUM_MODES,
        # SOURCED: Abdulbaki thesis Section 3.4.3
        min_scale: float = TRAJECTORY_MIN_SCALE,
        # SOURCED: HiVT decoder.py
        social_heads: int = 4,
        # ASSUMED: confirmed by ablation {2, 4}
        social_layers: int = 1,
        # ASSUMED: lightweight for 100 scenarios
        dropout: float = 0.1,
        # Regularization — important with only 100 training scenarios
        uncertain: bool = True,
        # Laplace uncertainty — SOURCED: Abdulbaki Section 3.4.3
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.min_scale = min_scale
        self.uncertain = uncertain
        self.bev_channels = bev_channels
        self.gru_hidden = gru_hidden

        # Combined feature dimension
        self.feat_dim = bev_channels + gru_hidden
        # 512 + 64 = 576

        # =====================================================================
        # Stage 1 — GRU history encoder
        # [N, 50, 5] → [N, 64]
        # =====================================================================
        self.history_encoder = GRUHistoryEncoder(
            input_size=AGENT_HISTORY_FEATURES,
            # SOURCED: 5 — Abdulbaki thesis Section 3.6
            hidden_size=gru_hidden,
            num_layers=1,
            dropout=dropout,
        )

        # =====================================================================
        # Stage 2 — Project combined feature to hidden_size
        # [N, 576] → [N, 256]
        # Necessary to match hidden_size before social attention and decoder
        # =====================================================================
        self.input_proj = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # =====================================================================
        # Stage 3 — Social attention
        # [N, 256] → [N, 256] (agents attend to each other)
        # =====================================================================
        self.social_attention = SocialAttention(
            hidden_size=hidden_size,
            num_heads=social_heads,
            num_layers=social_layers,
            dropout=dropout,
        )

        # =====================================================================
        # Stage 4 — BEV feature map projection
        # Projects BEV map channels to hidden_size for cross-attention keys/values
        # [B, 512, 50, 90] → [B, 4500, 256]
        # 4500 = 50 × 90 spatial locations
        # =====================================================================
        self.bev_proj = nn.Sequential(
            nn.Linear(bev_channels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
        )

        # =====================================================================
        # Stage 5 — Learnable mode queries
        # [num_modes, hidden_size] = [6, 256]
        # Each mode query specialises in a different trajectory type
        # SOURCED conceptually: DeTra (Casas et al., 2024)
        # AutoBots (Girgis et al., 2021)
        # =====================================================================
        self.mode_queries = nn.Parameter(
            torch.randn(num_modes, hidden_size)
        )

        # =====================================================================
        # Stage 6 — Transformer decoder layers
        # Mode queries cross-attend to BEV feature map
        # Each layer: self-attention among modes + cross-attention to BEV
        # =====================================================================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            # Pre-norm for stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )

        # =====================================================================
        # Output heads — same structure as BEVTrajectoryDecoder
        # =====================================================================

        # Location head: [num_modes, N, hidden_size] → [num_modes, N, H, 2]
        self.loc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, future_steps * 2),
        )

        # Scale head: [num_modes, N, hidden_size] → [num_modes, N, H, 2]
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, future_steps * 2),
            )

        # Mode probability head
        self.pi = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # Initialise weights
        self.apply(_init_weights)
        nn.init.normal_(self.mode_queries, mean=0.0, std=0.1)

    def forward(
        self,
        bev_feature_map: torch.Tensor,
        agent_bev_features: torch.Tensor,
        agent_history: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of TransformerTrajectoryDecoder.

        Args:
            bev_feature_map:    [B, 512, 50, 90]
                                Full BEV feature map from backbone.
                                Used as memory for cross-attention.

            agent_bev_features: [N, 512]
                                Per-agent BEV features sampled at box locations.
                                Same as V2/V3 bilinear sampling.

            agent_history:      [N, 50, 5]
                                Agent history: (x, y, vx, vy, heading)
                                per timestep for last 50 timesteps.
                                SOURCED: Abdulbaki thesis Section 3.6

            history_mask:       [N, 50] — True where history valid
                                Not yet used — future improvement

        Returns:
            y_hat: [num_modes, N, H, 4]
                   Trajectory predictions in Laplace parameterisation.
                   (µx, µy, bx, by) per timestep.

            pi:    [N, num_modes]
                   Mode logits. softmax(pi) = mode probabilities.
        """
        N = agent_bev_features.shape[0]
        num_modes = self.num_modes
        hidden_size = self.hidden_size

        if N == 0:
            device = bev_feature_map.device
            return (
                torch.zeros(num_modes, 0, self.future_steps, 4, device=device),
                torch.zeros(0, num_modes, device=device)
            )

        # =================================================================
        # Stage 1 — Encode agent history
        # [N, 50, 5] → [N, 64]
        # =================================================================
        history_vec = self.history_encoder(agent_history, history_mask)
        # [N, gru_hidden=64]

        # =================================================================
        # Stage 2 — Concatenate BEV feature + history
        # [N, 512] cat [N, 64] → [N, 576]
        # =================================================================
        combined = torch.cat([agent_bev_features, history_vec], dim=-1)
        # [N, feat_dim=576]

        # =================================================================
        # Stage 3 — Project to hidden_size
        # [N, 576] → [N, 256]
        # =================================================================
        agent_feat = self.input_proj(combined)
        # [N, hidden_size=256]

        # =================================================================
        # Stage 4 — Social attention
        # Agents attend to each other
        # [N, 256] → [N, 256]
        # =================================================================
        agent_feat = self.social_attention(agent_feat)
        # [N, hidden_size=256]

        # =================================================================
        # Stage 5 — Prepare BEV feature map as cross-attention memory
        # [B, 512, 50, 90] → [4500, 256]
        # We use first batch element — trajectory processes one scene
        # =================================================================
        B = bev_feature_map.shape[0]
        bev = bev_feature_map[0]
        # [512, 50, 90]

        # Flatten spatial dimensions
        bev_flat = bev.permute(1, 2, 0).reshape(-1, self.bev_channels)
        # [4500, 512]

        # Project to hidden_size
        bev_memory = self.bev_proj(bev_flat)
        # [4500, 256]

        # =================================================================
        # Stage 6 — Transformer decoder
        # For each agent: mode queries cross-attend to BEV + agent context
        #
        # We process each agent independently through the decoder.
        # Mode queries start as [num_modes, hidden_size] and are
        # conditioned on agent_feat via addition before cross-attention.
        #
        # This is different from standard sequence-to-sequence decoder —
        # we decode num_modes trajectories per agent, conditioned on
        # the agent's social-aware feature and the full BEV map.
        # =================================================================

        # Expand mode queries for all agents
        mode_q = self.mode_queries.unsqueeze(0).expand(N, num_modes, self.hidden_size)
        # [N, num_modes, hidden_size]

        # Condition mode queries on agent feature
        # Each mode starts with shared agent context
        agent_cond = agent_feat.unsqueeze(1).expand(N, num_modes, self.hidden_size)
        # [N, num_modes, hidden_size]

        tgt = mode_q + agent_cond
        # [N, num_modes, hidden_size]
        # mode_q gives mode diversity, agent_cond gives agent-specific context

        # BEV memory expanded for all agents
        memory = bev_memory.unsqueeze(0).expand(N, -1, self.hidden_size)
        # [N, 4500, hidden_size]

        # Cross-attention: mode queries attend to BEV feature map
        # tgt:    [N, num_modes, hidden_size] — queries
        # memory: [N, 4500, hidden_size]      — keys and values from BEV
        decoder_out = self.transformer_decoder(tgt, memory)
        # [N, num_modes, hidden_size]

        # Rearrange to [num_modes, N, hidden_size] for output heads
        decoder_out = decoder_out.permute(1, 0, 2)
        # [num_modes, N, hidden_size]

        # =================================================================
        # Stage 7 — Output heads
        # =================================================================

        # Mode probabilities
        pi = self.pi(decoder_out).squeeze(-1).t()
        # [num_modes, N, 1] → [num_modes, N] → [N, num_modes]

        # Location predictions
        loc = self.loc(decoder_out).view(
            num_modes, N, self.future_steps, 2
        )
        # [num_modes, N, H, 2]

        # Scale predictions
        if self.uncertain:
            scale = self.scale(decoder_out).view(
                num_modes, N, self.future_steps, 2
            )
            scale = F.elu_(scale, alpha=1.0) + 1.0 + self.min_scale
            y_hat = torch.cat([loc, scale], dim=-1)
            # [num_modes, N, H, 4]
        else:
            y_hat = loc

        return y_hat, pi
        # y_hat: [num_modes, N, H, 4]
        # pi:    [N, num_modes]