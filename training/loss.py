# training/loss.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
#
# Modifications:
#   1. Updated import paths to match new repo structure
#   2. DetectionIntentionLoss — completely unchanged from Nadeem's original
#   3. Added TrajectoryLoss — new class implementing WTA Laplace NLL loss
#      SOURCED: loss formulation from Abdulbaki thesis Section 3.8
#   4. Added MultiTaskLoss — new wrapper combining all three losses
#      L_total = L_det_intent + λ × L_trajectory
#      NEEDS TEST: λ = TRAJECTORY_LAMBDA, ablation over {0.01, 0.1, 0.5, 1.0}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

# MODIFICATION: updated import paths
from utils.constants import (
    DOMINANT_CLASSES_FOR_DOWNSAMPLING,
    INTENTION_DOWNSAMPLE_RATIO,
    TRAJECTORY_LAMBDA,          # NEEDS TEST: ablation {0.01, 0.1, 0.5, 1.0}
    TRAJECTORY_NUM_MODES,       # SOURCED: Abdulbaki thesis Section 3.4.3
    TRAJECTORY_FUTURE_STEPS,    # SOURCED: Abdulbaki thesis Section 3.1
)
from utils.utils import compute_axis_aligned_iou, compute_rotated_iou


# =============================================================================
# DetectionIntentionLoss
# Completely unchanged from Nadeem's original loss.py
# =============================================================================

class DetectionIntentionLoss(nn.Module):
    def __init__(self,
                 iou_threshold=0.6,
                 neg_iou_threshold=0.45,
                 box_weight=1.0,
                 cls_weight=1.0,
                 intent_weight=0.5,
                 intention_class_weights=None,
                 use_rotated_iou=False,
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0,
                 smooth_l1_beta=1.0 / 9.0,
                 apply_intention_downsampling=True,
                 dominant_intentions=DOMINANT_CLASSES_FOR_DOWNSAMPLING,
                 intention_downsample_ratio=INTENTION_DOWNSAMPLE_RATIO
                 ):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.intent_weight = intent_weight
        self.use_rotated_iou = use_rotated_iou
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta

        self.apply_intention_downsampling = apply_intention_downsampling
        self.dominant_intentions = set(dominant_intentions)
        self.intention_downsample_keep_prob = 1.0 - intention_downsample_ratio

        effective_intention_weights = None
        if not self.apply_intention_downsampling and intention_class_weights is not None:
            effective_intention_weights = intention_class_weights

        self.register_buffer('final_intention_class_weights', effective_intention_weights)
        self.intention_criterion = nn.CrossEntropyLoss(
            weight=self.final_intention_class_weights,
            reduction='none'
        )

        print(
            f"Loss Initialized: Use Rotated IoU: {self.use_rotated_iou}, "
            f"Apply Intention Downsampling: {self.apply_intention_downsampling}, "
            f"Downsample Ratio: {intention_downsample_ratio if self.apply_intention_downsampling else 'N/A'}, "
            f"Intent Weight: {self.intent_weight}"
        )

    def forward(self, cls_logits, box_preds, intention_logits, anchors, gt_list):
        B = cls_logits.shape[0]
        N_total_anchors = anchors.shape[0]
        device = cls_logits.device
        anchors = anchors.to(device)

        cls_targets = torch.full((B, N_total_anchors), -1, dtype=torch.long, device=device)
        box_targets = torch.zeros((B, N_total_anchors, 6), dtype=torch.float32, device=device)
        intention_targets = torch.full((B, N_total_anchors), -1, dtype=torch.long, device=device)

        for b in range(B):
            if not isinstance(gt_list[b], dict) or \
               'boxes_xywha' not in gt_list[b] or \
               'intentions' not in gt_list[b]:
                cls_targets[b, :] = 0
                continue

            gt_boxes_item = gt_list[b]['boxes_xywha'].to(device)
            gt_intentions_item = gt_list[b]['intentions'].to(device)
            num_gt = gt_boxes_item.shape[0]

            if num_gt == 0:
                cls_targets[b, :] = 0
                continue

            iou_func = compute_rotated_iou if self.use_rotated_iou else compute_axis_aligned_iou
            iou_args = (anchors, gt_boxes_item)

            try:
                iou_matrix = iou_func(*iou_args)
            except Exception as e:
                print(f"Error in iou_func: {e}")
                raise e

            max_iou_per_anchor, max_iou_gt_idx_per_anchor = iou_matrix.max(dim=1)

            neg_mask_item = max_iou_per_anchor < self.neg_iou_threshold
            cls_targets[b, neg_mask_item] = 0

            pos_mask_item = max_iou_per_anchor >= self.iou_threshold
            cls_targets[b, pos_mask_item] = 1

            if num_gt > 0:
                _, max_iou_anchor_idx_per_gt = iou_matrix.max(dim=0)
                for gt_idx_force in range(num_gt):
                    anchor_idx_force = max_iou_anchor_idx_per_gt[gt_idx_force]
                    if not pos_mask_item[anchor_idx_force] and \
                       iou_matrix[anchor_idx_force, gt_idx_force] >= self.neg_iou_threshold:
                        pos_mask_item[anchor_idx_force] = True
                        cls_targets[b, anchor_idx_force] = 1

            final_pos_mask_item = (cls_targets[b, :] == 1)
            assigned_gt_indices = max_iou_gt_idx_per_anchor[final_pos_mask_item]
            pos_anchor_indices_item = torch.where(final_pos_mask_item)[0]

            if pos_anchor_indices_item.numel() > 0:
                current_assigned_anchors = anchors[pos_anchor_indices_item]
                current_assigned_gt_boxes = gt_boxes_item[assigned_gt_indices]
                current_assigned_gt_intentions = gt_intentions_item[assigned_gt_indices]

                eps = 1e-6
                delta_x = (current_assigned_gt_boxes[:, 0] - current_assigned_anchors[:, 0]) / \
                          (current_assigned_anchors[:, 2] + eps)
                delta_y = (current_assigned_gt_boxes[:, 1] - current_assigned_anchors[:, 1]) / \
                          (current_assigned_anchors[:, 3] + eps)
                delta_w = torch.log(
                    current_assigned_gt_boxes[:, 2] / (current_assigned_anchors[:, 2] + eps) + eps
                )
                delta_l = torch.log(
                    current_assigned_gt_boxes[:, 3] / (current_assigned_anchors[:, 3] + eps) + eps
                )
                delta_h_sin = torch.sin(
                    current_assigned_gt_boxes[:, 4] - current_assigned_anchors[:, 4]
                )
                delta_h_cos = torch.cos(
                    current_assigned_gt_boxes[:, 4] - current_assigned_anchors[:, 4]
                )

                box_targets[b, pos_anchor_indices_item, :] = torch.stack(
                    [delta_x, delta_y, delta_w, delta_l, delta_h_sin, delta_h_cos], dim=1
                )
                intention_targets[b, pos_anchor_indices_item] = current_assigned_gt_intentions

        cls_logits_flat = cls_logits.reshape(-1, 1)
        box_preds_flat = box_preds.reshape(-1, 6)
        intention_logits_flat = intention_logits.reshape(-1, intention_logits.shape[-1])

        cls_targets_flat = cls_targets.reshape(-1)
        box_targets_flat = box_targets.reshape(-1, 6)
        intention_targets_flat = intention_targets.reshape(-1)

        valid_cls_mask_flat = cls_targets_flat >= 0
        pos_targets_mask_flat = cls_targets_flat == 1
        num_pos_total_batch = pos_targets_mask_flat.sum()

        cls_loss = torch.tensor(0.0, device=device)
        if valid_cls_mask_flat.any():
            masked_cls_logits = cls_logits_flat[valid_cls_mask_flat]
            masked_cls_targets = cls_targets_flat[valid_cls_mask_flat].float()
            if masked_cls_logits.ndim > masked_cls_targets.ndim:
                masked_cls_targets = masked_cls_targets.unsqueeze(1)
            cls_loss = sigmoid_focal_loss(
                masked_cls_logits, masked_cls_targets,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum"
            )
            cls_loss = cls_loss / max(1, num_pos_total_batch)

        box_loss = torch.tensor(0.0, device=device)
        if num_pos_total_batch > 0:
            masked_box_preds = box_preds_flat[pos_targets_mask_flat]
            masked_box_targets = box_targets_flat[pos_targets_mask_flat]
            box_loss = F.smooth_l1_loss(
                masked_box_preds, masked_box_targets,
                beta=self.smooth_l1_beta, reduction="sum"
            )
            box_loss = box_loss / max(1, num_pos_total_batch)

        intent_loss = torch.tensor(0.0, device=device)
        if num_pos_total_batch > 0:
            intent_logits_pos = intention_logits_flat[pos_targets_mask_flat]
            intent_targets_pos = intention_targets_flat[pos_targets_mask_flat]

            if intent_targets_pos.numel() > 0:
                intent_loss_per_anchor = self.intention_criterion(
                    intent_logits_pos, intent_targets_pos
                )

                if self.apply_intention_downsampling:
                    with torch.no_grad():
                        downsample_mask = torch.ones_like(
                            intent_targets_pos, dtype=torch.float32
                        )
                        for dominant_idx in self.dominant_intentions:
                            is_dominant = (intent_targets_pos == dominant_idx)
                            if is_dominant.any():
                                num_dominant = is_dominant.sum().item()
                                rand_vals = torch.rand(num_dominant, device=device)
                                keep = rand_vals < self.intention_downsample_keep_prob
                                downsample_mask[is_dominant] = keep.float()

                    intent_loss = (intent_loss_per_anchor * downsample_mask).sum()
                    effective_num = downsample_mask.sum()
                    intent_loss = intent_loss / max(1, effective_num)
                else:
                    intent_loss = intent_loss_per_anchor.sum() / \
                                  max(1, intent_targets_pos.numel())

        total_loss = (
            self.cls_weight * cls_loss +
            self.box_weight * box_loss +
            self.intent_weight * intent_loss
        )

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print(
                f"NaN/Inf in loss! "
                f"Cls: {cls_loss.item():.4f}, "
                f"Box: {box_loss.item():.4f}, "
                f"Intent: {intent_loss.item():.4f}"
            )
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "cls_loss": torch.tensor(0.0, device=device),
                "box_loss": torch.tensor(0.0, device=device),
                "intent_loss": torch.tensor(0.0, device=device),
                "num_pos_anchors": num_pos_total_batch.item()
            }

        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "intent_loss": intent_loss.detach(),
            "num_pos_anchors": num_pos_total_batch.item()
        }


# =============================================================================
# TrajectoryLoss
# NEW — no equivalent in Nadeem's original
# =============================================================================

class TrajectoryLoss(nn.Module):
    """
    Winner-Takes-All Laplace NLL trajectory loss.

    NEW class added for IntentTrajNet-AV2 multi-task learning.

    Two components:
        1. Regression loss (WTA Laplace NLL)
           For each vehicle, find which of the F=6 predicted modes is
           closest to the GT trajectory. Train only that mode using
           Laplace negative log-likelihood. Ignore the other 5 modes.
           SOURCED: Abdulbaki thesis Section 3.8, equation:
           L_reg = -1/NH Σ log P(R^T(p^t - p^T) | µ̂^t, b̂^t)
           where P(·|·) is the Laplace PDF.

        2. Classification loss (Cross-Entropy on mode probabilities)
           Train the mode probability logits (pi) so the model assigns
           high probability to the winning mode.
           SOURCED: Abdulbaki thesis Section 3.8, L_cls term.

    Combined: L_traj = L_reg_WTA + L_cls
    SOURCED: Abdulbaki thesis Section 3.8, L = L_reg + L_cls

    Winner-Takes-All:
        Only the best mode is trained on regression. This prevents all
        modes from collapsing to the same prediction (mode collapse).
        SOURCED: variety loss — Thiede & Brahma, ICCV 2019
        "Analyzing the Variety Loss in the Context of Probabilistic
        Trajectory Prediction" — cited in Abdulbaki thesis reference [22].

    Laplace distribution:
        P(x | µ, b) = (1/2b) exp(-|x - µ| / b)
        NLL = log(2b) + |x - µ| / b
        Using Laplace instead of Gaussian allows heavier tails —
        the model is not over-penalised for occasional large errors.
        SOURCED: Abdulbaki thesis Section 3.4.3 — "mixture of Laplace
        distributions" stated explicitly.
    """

    def __init__(
        self,
        num_modes: int = TRAJECTORY_NUM_MODES,
        # SOURCED: 6 — Abdulbaki thesis Section 3.4.3
        future_steps: int = TRAJECTORY_FUTURE_STEPS,
        # SOURCED: 60 — Abdulbaki thesis Section 3.1
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps

    def _laplace_nll(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Laplace negative log-likelihood loss for one trajectory mode.

        Laplace NLL formula:
            NLL = log(2b) + |µ - GT| / b

        where:
            µ = predicted position  (from pred[..., :2])
            b = predicted scale     (from pred[..., 2:])
            GT = ground truth position

        SOURCED: Abdulbaki thesis Section 3.4.3 and Section 3.8
        SOURCED: HiVT decoder.py — same Laplace formulation used in
        the variety loss training objective.

        Args:
            pred: [N, H, 4]  — predicted (µx, µy, bx, by) per timestep
            gt:   [N, H, 2]  — GT (x, y) positions per timestep
            mask: [N, H]     — True where GT is valid (not padded)

        Returns:
            scalar mean NLL loss over all valid (vehicle, timestep) pairs
        """
        # Extract predicted mean and scale
        mu = pred[..., :2]     # [N, H, 2] — predicted positions
        b = pred[..., 2:]      # [N, H, 2] — predicted scales (uncertainty)

        # Laplace NLL: log(2b) + |µ - GT| / b
        # SOURCED: Abdulbaki thesis Section 3.8 regression loss equation
        nll = torch.log(2 * b) + torch.abs(mu - gt) / b
        # [N, H, 2]

        # Sum over x and y dimensions
        nll = nll.sum(dim=-1)
        # [N, H]

        # Apply validity mask — ignore padded timesteps
        # mask is True where data is real, False where zero-padded
        nll = nll * mask.float()
        # [N, H]

        # Average over valid timesteps
        # Use mask.float().sum() to count only valid entries
        num_valid = mask.float().sum().clamp(min=1.0)
        return nll.sum() / num_valid

    def forward(
        self,
        y_hat: torch.Tensor,
        pi: torch.Tensor,
        gt_traj: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> dict:
        """
        Compute WTA trajectory loss.

        Args:
            y_hat:   [F, N, H, 4]
                     Predicted trajectories from BEVTrajectoryDecoder.
                     F=6 modes, N vehicles, H=60 timesteps, 4=(µx,µy,bx,by).

            pi:      [N, F]
                     Raw mode logits from BEVTrajectoryDecoder.
                     softmax(pi) gives mode probabilities.

            gt_traj: [N, H, 2]
                     GT future positions in ego frame (x, y per timestep).
                     From future_traj_ego in GT dict.

            gt_mask: [N, H]
                     True where GT position is valid, False where padded.
                     From future_traj_mask in GT dict.

        Returns:
            dict with keys:
                loss          — total trajectory loss (reg + cls)
                reg_loss      — WTA Laplace NLL regression loss
                cls_loss      — mode classification cross-entropy loss
                best_mode_idx — index of winning mode per vehicle [N]
        """
        device = y_hat.device
        N = y_hat.shape[1]  # number of vehicles
        num_modes  = self.num_modes
        H = self.future_steps

        # Handle empty batch — no vehicles detected
        if N == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "loss": zero,
                "reg_loss": zero.detach(),
                "cls_loss": zero.detach(),
                "best_mode_idx": torch.zeros(0, dtype=torch.long, device=device)
            }

        # === Part 1: Find the winning mode (WTA) ===
        #
        # For each vehicle, compute the L2 distance from each predicted
        # mode to the GT trajectory. The mode with the smallest total
        # distance is the "winner".
        #
        # We use L2 distance for mode selection but Laplace NLL for
        # the actual regression loss. This is the standard WTA approach.
        # SOURCED: Abdulbaki thesis Section 3.8 — variety loss with
        # best-of-K selection.
        # SOURCED: Thiede & Brahma, ICCV 2019 — variety loss formulation.

        # Extract predicted positions (µx, µy) from all modes
        pred_positions = y_hat[..., :2]
        # [F, N, H, 2]

        # GT positions expanded for comparison across all F modes
        gt_expanded = gt_traj.unsqueeze(0).expand(num_modes, N, H, 2)
        # [F, N, H, 2]

        # Mask expanded for all modes
        mask_expanded = gt_mask.unsqueeze(0).expand(num_modes, N, H)
        # [F, N, H]

        # L2 distance at each timestep, summed over x and y
        l2_per_step = torch.norm(pred_positions - gt_expanded, dim=-1)
        # [F, N, H]

        # Zero out padded timesteps before summing
        l2_per_step = l2_per_step * mask_expanded.float()
        # [F, N, H]

        # Sum L2 distance over all valid timesteps per mode per vehicle
        l2_per_mode = l2_per_step.sum(dim=-1)
        # [F, N]

        # Find the winning mode — lowest total L2 distance
        best_mode_idx = l2_per_mode.argmin(dim=0)
        # [N] — index of best mode for each vehicle

        # === Part 2: WTA Regression Loss (Laplace NLL) ===
        #
        # Compute Laplace NLL only for the winning mode.
        # Gather the winning mode's predictions for each vehicle.
        # SOURCED: Abdulbaki thesis Section 3.8 regression loss.

        # Gather winning mode predictions for each vehicle
        # y_hat:          [F, N, H, 4]
        # best_mode_idx:  [N]
        # We need:        [N, H, 4] — the winning mode for each vehicle

        best_mode_expanded = best_mode_idx.view(1, N, 1, 1).expand(1, N, H, 4)
        best_pred = y_hat.gather(0, best_mode_expanded).squeeze(0)
        # [N, H, 4]

        # Compute Laplace NLL on winning mode only
        reg_loss = self._laplace_nll(best_pred, gt_traj, gt_mask)
        # scalar

        # === Part 3: Mode Classification Loss (Cross-Entropy) ===
        #
        # Train the mode probability scores to be high for the winner.
        # This teaches the model to correctly identify which future
        # is most likely.
        # SOURCED: Abdulbaki thesis Section 3.8 — L_cls term.

        # Cross-entropy loss between mode logits and winning mode index
        cls_loss = F.cross_entropy(pi, best_mode_idx)
        # pi: [N, F] — mode logits
        # best_mode_idx: [N] — winning mode index per vehicle
        # scalar

        # === Combined trajectory loss ===
        # SOURCED: Abdulbaki thesis Section 3.8 — L = L_reg + L_cls
        traj_loss = reg_loss + cls_loss

        # NaN/Inf check
        if torch.isnan(traj_loss) or torch.isinf(traj_loss):
            print(
                f"NaN/Inf in trajectory loss! "
                f"Reg: {reg_loss.item():.4f}, "
                f"Cls: {cls_loss.item():.4f}"
            )
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "loss": zero,
                "reg_loss": zero.detach(),
                "cls_loss": zero.detach(),
                "best_mode_idx": best_mode_idx
            }

        return {
            "loss": traj_loss,
            "reg_loss": reg_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "best_mode_idx": best_mode_idx
        }


# =============================================================================
# MultiTaskLoss
# NEW — combines DetectionIntentionLoss and TrajectoryLoss
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for IntentTrajNet-AV2.

    NEW class combining detection, intention, and trajectory losses.

    Formula:
        L_total = L_det_intent + λ × L_trajectory

    Where:
        L_det_intent = cls_weight × L_cls
                     + box_weight × L_box
                     + intent_weight × L_intent
        L_trajectory = L_reg_WTA + L_cls_modes
        λ = TRAJECTORY_LAMBDA

    NEEDS TEST: λ ablation over {0.01, 0.1, 0.5, 1.0}
    Run 3 epochs on 5 logs for each λ value.
    Select λ where:
        - trajectory loss visibly decreases over 3 epochs
        - detection mAP does not drop more than ~5% vs V1

    λ too small (0.01): trajectory loss ignored, no trajectory learning
    λ too large (1.0):  trajectory loss dominates, detection may degrade
    λ = 0.1: conservative starting point

    Reference for multi-task loss weighting:
        Caruana, R. (1997). Multitask Learning.
        Machine Learning, 28(1), 41-75.
        — showed related auxiliary tasks improve shared representations
          even when the auxiliary task uses a down-weighted loss.
    """

    def __init__(
        self,
        # Detection + intention loss params — passed through unchanged
        iou_threshold: float = 0.6,
        neg_iou_threshold: float = 0.45,
        box_weight: float = 1.0,
        cls_weight: float = 1.0,
        intent_weight: float = 0.5,
        use_rotated_iou: bool = False,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        smooth_l1_beta: float = 1.0 / 9.0,
        apply_intention_downsampling: bool = True,

        # Trajectory loss params
        traj_lambda: float = TRAJECTORY_LAMBDA,
        # NEEDS TEST: ablation over {0.01, 0.1, 0.5, 1.0}
        # Controls weight of trajectory loss relative to det+intent loss

        num_modes: int = TRAJECTORY_NUM_MODES,
        # SOURCED: 6 — Abdulbaki thesis Section 3.4.3

        future_steps: int = TRAJECTORY_FUTURE_STEPS,
        # SOURCED: 60 — Abdulbaki thesis Section 3.1

        use_trajectory_loss: bool = True,
        # Set to False for V1 (no trajectory head)
        # Set to True for V2 and V3
    ) -> None:
        super().__init__()

        self.traj_lambda = traj_lambda
        self.use_trajectory_loss = use_trajectory_loss

        # Detection + intention loss — unchanged from Nadeem's original
        self.det_intent_loss = DetectionIntentionLoss(
            iou_threshold=iou_threshold,
            neg_iou_threshold=neg_iou_threshold,
            box_weight=box_weight,
            cls_weight=cls_weight,
            intent_weight=intent_weight,
            use_rotated_iou=use_rotated_iou,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            smooth_l1_beta=smooth_l1_beta,
            apply_intention_downsampling=apply_intention_downsampling,
        )

        # Trajectory loss — new
        if use_trajectory_loss:
            self.traj_loss_fn = TrajectoryLoss(
                num_modes=num_modes,
                future_steps=future_steps,
            )
            print(
                f"MultiTaskLoss: Trajectory loss ENABLED. "
                f"λ = {traj_lambda} (NEEDS TEST: ablation recommended)"
            )
        else:
            self.traj_loss_fn = None
            print("MultiTaskLoss: Trajectory loss DISABLED (V1 mode).")

    def forward(
        self,
        # Detection + intention inputs — same as Nadeem's original
        cls_logits: torch.Tensor,
        box_preds: torch.Tensor,
        intention_logits: torch.Tensor,
        anchors: torch.Tensor,
        gt_list: list,

        # Trajectory inputs — new, only used when use_trajectory_loss=True
        y_hat: torch.Tensor = None,
        # [F, N, H, 4] from TrajectoryHead — None if V1
        pi: torch.Tensor = None,
        # [N, F] from TrajectoryHead — None if V1
        gt_traj: torch.Tensor = None,
        # [N, H, 2] from GT dict future_traj_ego — None if V1
        gt_mask: torch.Tensor = None,
        # [N, H] from GT dict future_traj_mask — None if V1
    ) -> dict:
        """
        Compute combined multi-task loss.

        For V1: only det_intent_loss is computed (use_trajectory_loss=False).
        For V2 and V3: all three losses are computed and combined.

        Returns:
            dict with keys:
                loss           — total combined loss (used for backprop)
                cls_loss       — detection focal loss
                box_loss       — box regression Smooth L1 loss
                intent_loss    — intention cross-entropy loss
                traj_loss      — trajectory WTA loss (0.0 if V1)
                traj_reg_loss  — trajectory regression component
                traj_cls_loss  — trajectory mode classification component
                num_pos_anchors — number of positive anchors in batch
        """
        device = cls_logits.device

        # --- Detection + intention loss ---
        # Unchanged from Nadeem's original
        det_intent_out = self.det_intent_loss(
            cls_logits, box_preds, intention_logits, anchors, gt_list
        )

        total_loss = det_intent_out["loss"]

        # --- Trajectory loss ---
        # Only computed for V2 and V3
        traj_loss_val = torch.tensor(0.0, device=device)
        traj_reg_loss_val = torch.tensor(0.0, device=device)
        traj_cls_loss_val = torch.tensor(0.0, device=device)

        if self.use_trajectory_loss and y_hat is not None and gt_traj is not None:
            traj_out = self.traj_loss_fn(y_hat, pi, gt_traj, gt_mask)

            traj_loss_val = traj_out["loss"]
            traj_reg_loss_val = traj_out["reg_loss"]
            traj_cls_loss_val = traj_out["cls_loss"]

            # Add trajectory loss to total with λ weighting
            # NEEDS TEST: λ ablation over {0.01, 0.1, 0.5, 1.0}
            total_loss = total_loss + self.traj_lambda * traj_loss_val

        # Final NaN/Inf check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(
                f"NaN/Inf in total loss! "
                f"Det+Intent: {det_intent_out['loss'].item():.4f}, "
                f"Traj: {traj_loss_val.item():.4f}"
            )
            return {
                "loss": torch.tensor(
                    0.0, device=device, requires_grad=True
                ),
                "cls_loss": torch.tensor(0.0, device=device),
                "box_loss": torch.tensor(0.0, device=device),
                "intent_loss": torch.tensor(0.0, device=device),
                "traj_loss": torch.tensor(0.0, device=device),
                "traj_reg_loss": torch.tensor(0.0, device=device),
                "traj_cls_loss": torch.tensor(0.0, device=device),
                "num_pos_anchors": det_intent_out["num_pos_anchors"]
            }

        return {
            "loss": total_loss,
            "cls_loss": det_intent_out["cls_loss"],
            "box_loss": det_intent_out["box_loss"],
            "intent_loss": det_intent_out["intent_loss"],
            "traj_loss": traj_loss_val.detach()
            if isinstance(traj_loss_val, torch.Tensor)
            else traj_loss_val,
            "traj_reg_loss": traj_reg_loss_val,
            "traj_cls_loss": traj_cls_loss_val,
            "num_pos_anchors": det_intent_out["num_pos_anchors"]
        }