# training/gradient_conflict_analysis.py
#
# New standalone script for AIBThings 2026 reviewer response.
# Author: Amir — Bachelor Thesis follow-up, GUC 2025
#
# Analyzes task-gradient conflicts in the shared backbone, directly
# answering R3's request:
#   "The paper does not analyze negative transfer through task-gradient
#    conflicts. Gradient similarity, task-specific loss curves, or
#    representation-level analysis would help explain why detection and
#    intention improvements do not occur simultaneously."
#
# Method:
#   For a handful of validation batches, run one forward pass through the
#   shared backbone, then compute FOUR SEPARATE backward passes — one per
#   task loss (cls_loss, box_loss, intent_loss, traj_loss) — each time
#   collecting the gradient with respect to the SHARED BACKBONE parameters
#   only (model.backbone.parameters()). No optimizer step is taken; the
#   model is never updated. We then compute pairwise cosine similarity
#   between the flattened gradient vectors for each task pair.
#
#   Interpretation:
#     cosine similarity near +1  -> gradients point in a similar
#                                    direction; tasks are cooperative
#                                    (improving one likely helps the other)
#     cosine similarity near  0  -> gradients are roughly orthogonal;
#                                    tasks are largely independent
#     cosine similarity near -1  -> gradients point in opposite
#                                    directions; tasks are in direct
#                                    conflict (negative transfer:
#                                    improving one actively hurts the other)
#
# SOURCED: gradient cosine similarity as a measure of task conflict is
# a standard diagnostic in multi-task learning literature, e.g.
# "Gradient Surgery for Multi-Task Learning" (Yu et al., NeurIPS 2020)
# uses exactly this pairwise cosine similarity test to detect conflicting
# gradients before applying gradient projection.
#
# Usage:
#   PYTHONPATH=/content/MultiTask-ViT-AV2 python \
#       training/gradient_conflict_analysis.py \
#       --config configs/v3_social_mlp.yaml \
#       --num_batches 15

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from utils.config_loader import load_config, get_nested
from utils.constants import (
    GRID_HEIGHT_PX, GRID_WIDTH_PX,
    ANCHOR_CONFIGS_PAPER,
    LIDAR_TOTAL_CHANNELS, MAP_CHANNELS,
    INTENTION_DOWNSAMPLE_RATIO,
    DOMINANT_CLASSES_FOR_DOWNSAMPLING,
)
from datasets.av2_dataset import ArgoverseIntentNetDataset, collate_fn
from models.model_mt import IntentNetViT_MT
from models.backbone import BasicBlock
from training.loss import DetectionIntentionLoss, TrajectoryLoss
from utils.utils import generate_anchors


# =============================================================================
# transform_to_agent_local
# Copied from training/train.py — same GT transform used for trajectory loss.
# =============================================================================

def transform_to_agent_local(traj_ego, boxes_xywha):
    N       = traj_ego.shape[0]
    cx      = boxes_xywha[:, 0]
    cy      = boxes_xywha[:, 1]
    heading = boxes_xywha[:, 4]
    agent_pos = torch.stack([cx, cy], dim=-1).unsqueeze(1)
    relative  = traj_ego - agent_pos
    cos_h = torch.cos(-heading).view(N, 1)
    sin_h = torch.sin(-heading).view(N, 1)
    local_x = cos_h * relative[..., 0] - sin_h * relative[..., 1]
    local_y = sin_h * relative[..., 0] + cos_h * relative[..., 1]
    return torch.stack([local_x, local_y], dim=-1)


# =============================================================================
# load_model_from_checkpoint
# Same logic as eval.py / agent_density_benchmark.py, extracted here too
# so this script is fully standalone.
# =============================================================================

def load_model_from_checkpoint(cfg: dict, device: torch.device):
    MODEL_VERSION = get_nested(cfg, 'model', 'version', default='V2')
    backbone_type = get_nested(cfg, 'model', 'backbone', 'type', default='vit')
    decoder_type  = get_nested(cfg, 'model', 'trajectory', 'decoder_type', default='mlp')

    checkpoint_path = get_nested(
        cfg, 'checkpoints', 'save_dir', default=''
    ) + '/' + get_nested(
        cfg, 'checkpoints', 'filename',
        default=f'MultiTask_{MODEL_VERSION}.pth'
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_backbone_cfg = checkpoint.get('backbone_cfg', {})
    use_trajectory      = checkpoint.get('use_trajectory', False)
    saved_decoder_type  = checkpoint.get('decoder_type', decoder_type)

    saved_backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
    saved_backbone_cfg.setdefault('lidar_input_channels', LIDAR_TOTAL_CHANNELS)
    saved_backbone_cfg.setdefault('map_input_channels', MAP_CHANNELS)
    saved_backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch8_224')
    saved_backbone_cfg.setdefault('vit_model_name_map',   'vit_small_patch8_224')
    saved_backbone_cfg.setdefault('pretrained_lidar', False)
    saved_backbone_cfg.setdefault('pretrained_map',   False)
    saved_backbone_cfg.setdefault('fusion_block_planes', 512)
    saved_backbone_cfg.setdefault('res_block_type', BasicBlock)

    traj_head_cfg = {}
    if use_trajectory:
        traj_head_cfg = {'box_feat_dim': 5, 'mlp_dropout': 0.0}
    if saved_decoder_type == 'transformer':
        traj_head_cfg.update({
            'gru_hidden':         get_nested(cfg, 'model', 'trajectory', 'gru_hidden',        default=64),
            'num_heads':          get_nested(cfg, 'model', 'trajectory', 'num_heads',          default=8),
            'num_decoder_layers': get_nested(cfg, 'model', 'trajectory', 'num_decoder_layers', default=2),
            'social_heads':       get_nested(cfg, 'model', 'trajectory', 'social_heads',       default=4),
            'social_layers':      get_nested(cfg, 'model', 'trajectory', 'social_layers',      default=1),
            'dropout':            get_nested(cfg, 'model', 'trajectory', 'dropout',            default=0.1),
        })

    saved_backbone_type = checkpoint.get('backbone_cfg', {}).get('type', backbone_type)
    saved_backbone_cfg.pop('type', None)

    model = IntentNetViT_MT(
        backbone_type=saved_backbone_type,
        backbone_cfg=saved_backbone_cfg,
        use_trajectory=use_trajectory,
        decoder_type=saved_decoder_type,
        trajectory_head_cfg=traj_head_cfg,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded: {MODEL_VERSION} ({saved_decoder_type} decoder, "
          f"use_trajectory={use_trajectory})\n")
    return model, MODEL_VERSION, saved_decoder_type, use_trajectory


# =============================================================================
# get_backbone_grad_vector
#
# After a .backward() call, flattens and concatenates the .grad of every
# backbone parameter into a single 1D vector. Returns None if no gradient
# was produced at all (e.g. a loss term that happens to be exactly zero
# for this batch, such as traj_loss when there are no active agents).
# =============================================================================

def get_backbone_grad_vector(model: nn.Module) -> torch.Tensor | None:
    grads = []
    for p in model.backbone.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().reshape(-1))
        else:
            # Parameter received no gradient this backward pass — treat
            # its contribution as zero, so shapes stay consistent across
            # tasks even if a task's loss graph doesn't touch every
            # backbone parameter (unlikely here, since backbone feeds
            # all heads, but handled defensively).
            grads.append(torch.zeros(p.numel(), device=p.device))
    if not grads:
        return None
    return torch.cat(grads)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None:
        return float('nan')
    a_norm = a.norm()
    b_norm = b.norm()
    if a_norm < 1e-12 or b_norm < 1e-12:
        return float('nan')
    return (torch.dot(a, b) / (a_norm * b_norm)).item()


# =============================================================================
# compute_task_losses
#
# Runs one forward pass and returns each task's loss as a SEPARATE scalar
# tensor, still attached to the autograd graph, so each can be backward()'d
# independently. This mirrors the loss computation in training/loss.py's
# DetectionIntentionLoss and TrajectoryLoss, but keeps the four components
# (cls, box, intent, traj) unsummed so we can isolate their gradients.
# =============================================================================

def compute_task_losses(
    model, det_intent_loss_fn, traj_loss_fn,
    lidar_bev, map_bev, gt_list, anchors,
    use_trajectory: bool, device,
):
    outputs = model(
        lidar_bev, map_bev,
        gt_list=gt_list,
        use_gt_boxes_for_traj=True,
        agent_history=None,
        run_traj_head=use_trajectory,
    )

    det_cls_logits   = outputs["det_cls_logits"]
    det_box_preds    = outputs["det_box_preds"]
    intention_logits = outputs["intention_logits"]
    y_hat = outputs.get("y_hat")
    pi    = outputs.get("pi")

    # --- Detection + intention losses, computed via the SAME anchor
    # matching logic as training, but we need cls/box/intent as separate
    # tensors rather than the combined scalar DetectionIntentionLoss
    # normally returns. Easiest correct way: call the loss module three
    # times isn't right either (matching is stochastic-free but the
    # weighted sum inside forward() combines them). Instead we replicate
    # by calling the same forward() and reading its returned components
    # BEFORE they were weighted-summed — but DetectionIntentionLoss only
    # returns the already-summed weighted total, not the raw components
    # as differentiable tensors (they are  detached  for logging).
    #
    # To get REAL differentiable per-task losses without duplicating all
    # of DetectionIntentionLoss's anchor-matching logic, we instead call
    # the loss module three times with the OTHER two weights set to zero
    # via direct attribute overrides, restoring them after. This reuses
    # 100% of the existing, validated anchor-matching code with no
    # duplication, at the cost of 3x forward-loss-compute (matching is
    # cheap CPU/GPU tensor ops, not a bottleneck here).
    orig_weights = (
        det_intent_loss_fn.cls_weight,
        det_intent_loss_fn.box_weight,
        det_intent_loss_fn.intent_weight,
    )

    # --- cls_loss only ---
    det_intent_loss_fn.cls_weight, det_intent_loss_fn.box_weight, det_intent_loss_fn.intent_weight = 1.0, 0.0, 0.0
    cls_only = det_intent_loss_fn(det_cls_logits, det_box_preds, intention_logits, anchors, gt_list)
    cls_loss = cls_only["loss"]

    # --- box_loss only ---
    det_intent_loss_fn.cls_weight, det_intent_loss_fn.box_weight, det_intent_loss_fn.intent_weight = 0.0, 1.0, 0.0
    box_only = det_intent_loss_fn(det_cls_logits, det_box_preds, intention_logits, anchors, gt_list)
    box_loss = box_only["loss"]

    # --- intent_loss only ---
    det_intent_loss_fn.cls_weight, det_intent_loss_fn.box_weight, det_intent_loss_fn.intent_weight = 0.0, 0.0, 1.0
    intent_only = det_intent_loss_fn(det_cls_logits, det_box_preds, intention_logits, anchors, gt_list)
    intent_loss = intent_only["loss"]

    # Restore original weights (harmless here since we don't reuse this
    # loss_fn instance for anything else afterward, but good hygiene)
    det_intent_loss_fn.cls_weight, det_intent_loss_fn.box_weight, det_intent_loss_fn.intent_weight = orig_weights

    # --- traj_loss, using the SAME active-agent filtering as train.py ---
    traj_loss = None
    if use_trajectory and y_hat is not None and y_hat.shape[1] > 0:
        gt_b0 = gt_list[0]
        if gt_b0 is not None and 'future_traj_ego' in gt_b0:
            traj_ego  = gt_b0['future_traj_ego'].to(device)
            traj_mask = gt_b0['future_traj_mask'].to(device)
            boxes     = gt_b0['boxes_xywha'].to(device)
            intentions = gt_b0['intentions'].to(device)

            PARKED_CLASS = 6
            MIN_DISPLACEMENT_M = 0.5
            intent_mask = (intentions != PARKED_CLASS)
            displacements = (traj_ego.norm(dim=-1) * traj_mask.float()).max(dim=-1).values
            disp_mask = displacements > MIN_DISPLACEMENT_M
            moving_mask = intent_mask & disp_mask

            if moving_mask.any():
                gt_traj_local = transform_to_agent_local(
                    traj_ego[moving_mask], boxes[moving_mask]
                )
                gt_mask_active = traj_mask[moving_mask]

                N_pred = y_hat.shape[1]
                N_gt   = gt_traj_local.shape[0]
                N_min  = min(N_pred, N_gt)

                if N_min > 0:
                    traj_out = traj_loss_fn(
                        y_hat=y_hat[:, :N_min],
                        pi=pi[:N_min] if pi is not None else None,
                        gt_traj=gt_traj_local[:N_min],
                        gt_mask=gt_mask_active[:N_min],
                    )
                    traj_loss = traj_out["loss"]

    return cls_loss, box_loss, intent_loss, traj_loss


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-conflict analysis across task losses."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=15,
                        help="Number of val batches to analyze.")
    parser.add_argument("--output_json", type=str, default="",
                        help="Optional path to save results as JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, model_version, decoder_type, use_trajectory = load_model_from_checkpoint(cfg, device)
    model.eval()   # eval mode for BatchNorm/Dropout determinism — gradients
                    # still flow normally, this only affects layer behavior,
                    # not whether backward() works.

    VAL_DATA_DIR = get_nested(cfg, 'data', 'val_dir', default='')
    val_dataset = ArgoverseIntentNetDataset(data_dir=VAL_DATA_DIR, is_train=False)
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=True,   # shuffle=True so batches
        num_workers=0, collate_fn=collate_fn,       # aren't all from one log
    )

    vit_model_name = get_nested(cfg, 'model', 'backbone', 'vit_model_name_lidar', default='vit_small_patch8_224')
    fusion_stride  = get_nested(cfg, 'model', 'backbone', 'fusion_block_stride', default=1)
    try:
        vit_patch_stride = int(vit_model_name.split('_patch')[-1].split('_')[0])
    except ValueError:
        vit_patch_stride = 8
    FEATURE_MAP_STRIDE = vit_patch_stride * fusion_stride

    anchors = generate_anchors(
        bev_height=GRID_HEIGHT_PX, bev_width=GRID_WIDTH_PX,
        feature_map_stride=FEATURE_MAP_STRIDE, anchor_configs=ANCHOR_CONFIGS_PAPER,
    ).to(device)

    det_intent_loss_fn = DetectionIntentionLoss(
        apply_intention_downsampling=True,
        dominant_intentions=DOMINANT_CLASSES_FOR_DOWNSAMPLING,
        intention_downsample_ratio=INTENTION_DOWNSAMPLE_RATIO,
    ).to(device)
    traj_loss_fn = TrajectoryLoss().to(device) if use_trajectory else None

    task_names = ['cls', 'box', 'intent']
    if use_trajectory:
        task_names.append('traj')

    pair_names = [
        (a, b) for i, a in enumerate(task_names) for b in task_names[i+1:]
    ]

    similarities_per_pair = {f"{a}_vs_{b}": [] for a, b in pair_names}
    batches_used = 0

    print(f"Running gradient-conflict analysis over up to {args.num_batches} "
          f"batches for tasks: {task_names}\n")

    for batch_data in val_loader:
        if batches_used >= args.num_batches:
            break
        if batch_data is None:
            continue

        lidar_bev = batch_data["lidar_bev"].to(device)
        map_bev   = batch_data["map_bev"].to(device)
        gt_list   = batch_data["gt_list"]

        try:
            cls_loss, box_loss, intent_loss, traj_loss = compute_task_losses(
                model, det_intent_loss_fn, traj_loss_fn,
                lidar_bev, map_bev, gt_list, anchors,
                use_trajectory, device,
            )
        except Exception as e:
            print(f"  Skipping batch due to error: {e}")
            continue

        task_losses = {'cls': cls_loss, 'box': box_loss, 'intent': intent_loss}
        if use_trajectory:
            task_losses['traj'] = traj_loss

        # Skip this batch entirely if any required task loss is missing
        # (e.g. traj_loss is None because no active agents in this batch)
        if any(v is None for v in task_losses.values()):
            continue

        # --- Compute one gradient vector per task, on the SAME forward
        # pass's graph. We must retain_graph=True on all but the last
        # backward() call, since each call otherwise frees the shared
        # backbone activations needed by the other tasks' backward passes.
        task_grads = {}
        task_list = list(task_losses.items())
        for i, (name, loss_val) in enumerate(task_list):
            model.zero_grad(set_to_none=True)
            is_last = (i == len(task_list) - 1)
            loss_val.backward(retain_graph=not is_last)
            task_grads[name] = get_backbone_grad_vector(model)

        model.zero_grad(set_to_none=True)

        # --- Pairwise cosine similarity for this batch ---
        for a, b in pair_names:
            sim = cosine_similarity(task_grads[a], task_grads[b])
            if not np.isnan(sim):
                similarities_per_pair[f"{a}_vs_{b}"].append(sim)

        batches_used += 1
        print(f"  Batch {batches_used}/{args.num_batches} processed.")

    print(f"\n{'='*60}")
    print(f"  Gradient Conflict Analysis Results ({model_version})")
    print(f"  Batches used: {batches_used}")
    print(f"{'='*60}")
    print(f"\n  {'Task pair':<20} {'Mean cos-sim':>14} {'Std':>10} {'N':>6}")
    print(f"  {'-'*20} {'-'*14} {'-'*10} {'-'*6}")

    results = {}
    for pair_name, sims in similarities_per_pair.items():
        if sims:
            mean_sim = float(np.mean(sims))
            std_sim  = float(np.std(sims))
            n        = len(sims)
        else:
            mean_sim, std_sim, n = float('nan'), float('nan'), 0
        results[pair_name] = {'mean_cosine_sim': mean_sim, 'std': std_sim, 'n_batches': n}
        print(f"  {pair_name:<20} {mean_sim:>14.4f} {std_sim:>10.4f} {n:>6}")

    print(f"\n  Interpretation: >0 cooperative, ~0 independent, <0 conflicting "
          f"(negative transfer)\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                'model_version': model_version,
                'decoder_type': decoder_type,
                'batches_used': batches_used,
                'task_pair_results': results,
            }, f, indent=2)
        print(f"Saved results to: {out_path}")


if __name__ == '__main__':
    main()