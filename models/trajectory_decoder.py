# models/trajectory_decoder.py
#
# Adapted from HiVT — Hierarchical Vector Transformer
# Original: Zhou et al., CVPR 2022
# Original repo: https://github.com/ZikangZhou/HiVT
# Original file: models/decoder.py — MLPDecoder class
#
# Modifications:
#   1. Replaced dual-input interface (local_embed + global_embed) with a
#      single agent feature vector [N, feat_dim] sampled from the BEV
#      feature map. This is because we have no graph encoder producing
#      separate local and global embeddings.
#   2. Added F learnable mode_queries [F, hidden_size] that replace
#      HiVT's global_embed. These are learned parameters that encourage
#      the F trajectory modes to specialise in different future behaviours.
#      SOURCED conceptually: DeTra (Casas et al., 2024) uses learnable
#      mode queries from the start of the network. AutoBots (Girgis et al.,
#      2021) pioneered mode queries for multi-modal trajectory prediction.
#   3. Added agent_proj layer to project [N, feat_dim] → [N, hidden_size]
#      before passing to aggr_embed. In HiVT both inputs were already dk=64.
#      Here feat_dim=516 >> hidden_size so projection is necessary.
#   4. hidden_size changed from 64 (HiVT default) to TRAJECTORY_DECODER_HIDDEN
#      (default 256). The richer BEV features justify a wider hidden layer.
#      NEEDS TEST: ablation over {128, 256, 512} recommended.
#   5. All core logic (aggr_embed, loc, scale, pi, Laplace parameterisation,
#      output shapes [F,N,H,4] and [N,F]) is unchanged from HiVT MLPDecoder.
#
# SOURCED values:
#   future_steps = 60    — Abdulbaki thesis Section 3.1
#   num_modes    = 6     — Abdulbaki thesis Section 3.4.3 + HiVT code
#   min_scale    = 1e-3  — HiVT decoder.py MLPDecoder.__init__
#   Laplace dist — Abdulbaki thesis Section 3.4.3
#   Output shape [F,N,H,4] — HiVT MLPDecoder, (µx,µy,bx,by) per step

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import (
    TRAJECTORY_FUTURE_STEPS,       # SOURCED: Abdulbaki thesis Section 3.1
    TRAJECTORY_NUM_MODES,          # SOURCED: Abdulbaki thesis Section 3.4.3
    TRAJECTORY_MIN_SCALE,          # SOURCED: HiVT decoder.py
    TRAJECTORY_DECODER_HIDDEN,     # NEEDS TEST: ablation {128, 256, 512}
)


def _init_weights(module: nn.Module) -> None:
    """
    Initialises linear layer weights with Xavier uniform and biases to zero.
    SOURCED: identical to init_weights() in HiVT utils.py.
    Applied via self.apply(_init_weights) in __init__.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BEVTrajectoryDecoder(nn.Module):
    """
    Multi-modal trajectory decoder for BEV-based agent features.

    Adapted from HiVT MLPDecoder (Zhou et al., CVPR 2022).

    Input:
        agent_feat [N, feat_dim]
            Per-agent feature vector produced by TrajectoryHead
            (bilinear sample from BEV feature map + box params).
            feat_dim = backbone_channels + 4 = 512 + 4 = 516.

    Output:
        y_hat [F, N, H, 4]
            F=6 trajectory modes, N agents, H=60 future timesteps,
            4 values per step = (µx, µy, bx, by) in Laplace parameterisation.
            µx, µy: predicted position.
            bx, by: predicted scale (uncertainty).
            SOURCED: output format from HiVT MLPDecoder and
            Abdulbaki thesis Section 3.4.3.

        pi [N, F]
            Raw logits for mode probabilities.
            softmax(pi) gives the probability of each of the 6 modes.
            SOURCED: HiVT MLPDecoder forward() return value.

    How mode_queries replace HiVT's global_embed:
        In HiVT, global_embed [F, N, dk] comes from the global interaction
        module — it encodes social context between agents via a graph.
        We have no graph encoder. Instead, mode_queries [F, hidden_size]
        are F learned parameter vectors, one per trajectory mode.
        During forward(), they are expanded to [F, N, hidden_size] and
        combined with the agent feature. The model learns to specialise
        each mode query for a different type of future behaviour
        (e.g. straight, left turn, right turn, stop, etc.).
    """

    def __init__(
        self,
        feat_dim: int = 516,
        # 512 backbone channels + 4 box params (cx, cy, w, h)
        # SOURCED: 512 from TwoStreamViTBackbone output (Nadeem thesis)
        # + 4 box params concatenated in TrajectoryHead

        hidden_size: int = TRAJECTORY_DECODER_HIDDEN,
        # NEEDS TEST: ablation over {128, 256, 512}
        # 256 is the starting point — wider than HiVT's 64 to match
        # richer 516-dim input. Compressing 516 → 64 would discard
        # too much information from the BEV features.

        future_steps: int = TRAJECTORY_FUTURE_STEPS,
        # SOURCED: 60 — Abdulbaki thesis Section 3.1
        # 6 seconds × 10Hz = 60 timesteps

        num_modes: int = TRAJECTORY_NUM_MODES,
        # SOURCED: 6 — Abdulbaki thesis Section 3.4.3 + HiVT code

        uncertain: bool = True,
        # If True, predict Laplace scale (bx, by) alongside location.
        # Output is [F, N, H, 4] = (µx, µy, bx, by).
        # If False, output is [F, N, H, 2] = (µx, µy) only.
        # SOURCED: HiVT MLPDecoder uses uncertain=True by default.

        min_scale: float = TRAJECTORY_MIN_SCALE,
        # SOURCED: 1e-3 — HiVT decoder.py MLPDecoder.__init__
        # Prevents Laplace scale from collapsing to zero.

    ) -> None:
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        # ---------------------------------------------------------------------
        # Learnable mode query vectors
        # Shape: [F, hidden_size] = [6, 256]
        #
        # Each of the F vectors represents one trajectory "mode" or hypothesis.
        # During forward(), they are expanded to [F, N, hidden_size] so every
        # agent gets all F mode queries applied.
        #
        # MODIFICATION vs HiVT: in HiVT, global_embed [F, N, dk] comes from
        # the global interaction encoder. Here it is a learned parameter.
        # SOURCED conceptually: DeTra (Casas et al., 2024) uses learnable
        # mode queries. AutoBots (Girgis et al., 2021) pioneered this approach.
        # ---------------------------------------------------------------------
        self.mode_queries = nn.Parameter(
            torch.randn(num_modes, hidden_size)
        )

        # ---------------------------------------------------------------------
        # Agent feature projection
        # [N, feat_dim=516] → [N, hidden_size=256]
        #
        # MODIFICATION vs HiVT: HiVT's local_embed was already dk=64,
        # matching hidden_size. Our BEV features are 516-dimensional
        # so we need a projection layer first.
        # ---------------------------------------------------------------------
        self.agent_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------------
        # Aggregation MLP
        # Fuses mode query + projected agent feature → per-mode hidden vector
        # Input:  [F, N, hidden_size + hidden_size] = [F, N, 512]
        # Output: [F, N, hidden_size] = [F, N, 256]
        #
        # SOURCED: identical structure to HiVT MLPDecoder.aggr_embed
        # Input was (global_embed + local_embed), output was hidden_size.
        # Here (mode_query + agent_proj) replaces (global + local).
        # ---------------------------------------------------------------------
        self.aggr_embed = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------------
        # Location head
        # Predicts µx, µy for all future timesteps in one shot.
        # Input:  [F, N, hidden_size]
        # Output: [F, N, future_steps * 2] → reshape → [F, N, future_steps, 2]
        #
        # SOURCED: identical to HiVT MLPDecoder.loc
        # ---------------------------------------------------------------------
        self.loc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, future_steps * 2),
        )

        # ---------------------------------------------------------------------
        # Scale head (uncertainty)
        # Predicts bx, by (Laplace scale) for all future timesteps.
        # Input:  [F, N, hidden_size]
        # Output: [F, N, future_steps * 2] → reshape → [F, N, future_steps, 2]
        #
        # SOURCED: identical to HiVT MLPDecoder.scale
        # Only created when uncertain=True.
        # ---------------------------------------------------------------------
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, future_steps * 2),
            )

        # ---------------------------------------------------------------------
        # Mode probability head
        # Predicts one logit per mode per agent.
        # Input:  [F, N, hidden_size + hidden_size] (mode_query + agent_proj)
        # Output: [F, N, 1] → squeeze → [N, F]
        #
        # SOURCED: identical to HiVT MLPDecoder.pi
        # softmax(pi) gives probability of each mode being the correct future.
        # ---------------------------------------------------------------------
        self.pi = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

        # Initialise all weights with Xavier uniform
        # SOURCED: same as HiVT which calls self.apply(init_weights)
        self.apply(_init_weights)

        # Re-initialise mode_queries after apply() since apply()
        # would have tried to initialise it as a Linear (it's a Parameter)
        nn.init.normal_(self.mode_queries, mean=0.0, std=0.1)

    def forward(
        self,
        agent_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            agent_feat: [N, feat_dim]
                Per-agent BEV feature + box params.
                N = number of detected vehicles in this batch element.

        Returns:
            y_hat: [F, N, H, 4]
                Trajectory predictions in Laplace parameterisation.
                F=6 modes, N agents, H=60 timesteps, 4=(µx,µy,bx,by).
                Positions are in ego frame (metres).
                SOURCED: output format from HiVT MLPDecoder.

            pi: [N, F]
                Raw mode logits. Apply softmax to get probabilities.
                SOURCED: HiVT MLPDecoder forward() return value.
        """
        N = agent_feat.shape[0]
        F = self.num_modes

        # --- Step 1: Project agent feature ---
        # [N, feat_dim] → [N, hidden_size]
        # MODIFICATION: added vs HiVT (HiVT local_embed was already hidden_size)
        agent_hidden = self.agent_proj(agent_feat)  # [N, hidden_size]

        # --- Step 2: Expand mode queries for all agents ---
        # mode_queries: [F, hidden_size]
        # Expand to:    [F, N, hidden_size]
        # Each agent gets all F mode queries applied independently.
        mode_q = self.mode_queries.unsqueeze(1).expand(F, N, self.hidden_size)
        # [F, hidden_size] → [F, 1, hidden_size] → [F, N, hidden_size]

        # --- Step 3: Expand agent feature for all modes ---
        # agent_hidden: [N, hidden_size]
        # Expand to:    [F, N, hidden_size]
        agent_expanded = agent_hidden.unsqueeze(0).expand(F, N, self.hidden_size)
        # [N, hidden_size] → [1, N, hidden_size] → [F, N, hidden_size]

        # --- Step 4: Compute mode probabilities ---
        # Concatenate mode_query + agent_feature: [F, N, 2*hidden_size]
        # Pass through pi MLP: [F, N, 1]
        # Squeeze and transpose: [N, F]
        # SOURCED: identical logic to HiVT MLPDecoder.forward() pi computation
        pi_input = torch.cat([mode_q, agent_expanded], dim=-1)
        # [F, N, 2*hidden_size]
        pi = self.pi(pi_input).squeeze(-1).t()
        # [F, N, 1] → [F, N] → [N, F]

        # --- Step 5: Aggregate mode query + agent feature ---
        # Concatenate: [F, N, 2*hidden_size]
        # Pass through aggr_embed: [F, N, hidden_size]
        # SOURCED: identical to HiVT aggr_embed(cat(global_embed, local_embed))
        aggr_input = torch.cat([mode_q, agent_expanded], dim=-1)
        # [F, N, 2*hidden_size]
        out = self.aggr_embed(aggr_input)
        # [F, N, hidden_size]

        # --- Step 6: Predict locations ---
        # [F, N, hidden_size] → [F, N, future_steps*2] → [F, N, future_steps, 2]
        # SOURCED: identical to HiVT MLPDecoder loc head
        loc = self.loc(out).view(F, N, self.future_steps, 2)
        # [F, N, H, 2] where H = future_steps = 60

        # --- Step 7: Predict scales (uncertainty) and build output ---
        # SOURCED: identical to HiVT MLPDecoder scale head and output assembly
        if self.uncertain:
            scale = self.scale(out).view(F, N, self.future_steps, 2)
            # [F, N, H, 2]

            # ELU activation + 1.0 + min_scale ensures scale > min_scale > 0
            # This prevents the Laplace distribution from becoming infinitely
            # peaked (scale → 0 would cause NaN in log-likelihood loss)
            # SOURCED: identical to HiVT — F.elu_(scale) + 1.0 + min_scale
            scale = F.elu_(scale, alpha=1.0) + 1.0 + self.min_scale
            # [F, N, H, 2]

            # Concatenate loc and scale along last dimension
            # Output: [F, N, H, 4] = (µx, µy, bx, by) per timestep
            y_hat = torch.cat([loc, scale], dim=-1)
            # [F, N, H, 4]
        else:
            # Without uncertainty: output only locations
            y_hat = loc
            # [F, N, H, 2]

        return y_hat, pi
        # y_hat: [F, N, H, 4]  — trajectory predictions
        # pi:    [N, F]         — mode logits