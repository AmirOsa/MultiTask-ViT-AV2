# training/agent_density_benchmark.py
#
# New standalone script for AIBThings 2026 reviewer response.
# Author: Amir — Bachelor Thesis follow-up, GUC 2025
#
# Extracted from eval.py's benchmark_trajectory_head_by_agent_count() so
# it can be run independently, without paying the cost of the full
# rotated-IoU sensor evaluation loop. Loads a checkpoint, builds the
# model exactly as eval.py does, and benchmarks the trajectory head
# alone across a range of agent counts N — directly answering R3's
# point that identical reported MACs/latency for the full model do not
# isolate the cost of social attention (V3) as scene density grows.
#
# Usage:
#   PYTHONPATH=/content/MultiTask-ViT-AV2 python \
#       training/agent_density_benchmark.py --config configs/v3_social_mlp.yaml

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch

from utils.config_loader import load_config, get_config_arg, get_nested
from utils.constants import GRID_HEIGHT_PX, GRID_WIDTH_PX, LIDAR_TOTAL_CHANNELS, MAP_CHANNELS
from models.model_mt import IntentNetViT_MT
from models.backbone import BasicBlock

# Reuse the benchmark function directly from eval.py — importing this
# module does NOT trigger main_eval(), since it's guarded by
# `if __name__ == '__main__':` in eval.py.
from training.eval import benchmark_trajectory_head_by_agent_count


def load_model_from_checkpoint(cfg: dict, device: torch.device):
    """
    Same model-loading logic as eval.py's main_eval(), extracted so this
    script doesn't need to run the rest of the eval pipeline.
    """
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
    model.eval()

    print(f"Model loaded: {MODEL_VERSION} ({saved_decoder_type} decoder)\n")
    return model, MODEL_VERSION, saved_decoder_type


if __name__ == '__main__':
    config_path = get_config_arg()
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, model_version, decoder_type = load_model_from_checkpoint(cfg, device)

    if model.traj_head is None:
        print(f"{model_version} has no trajectory head — nothing to benchmark. Exiting.")
        sys.exit(0)

    agent_counts = get_nested(
        cfg, 'eval', 'agent_density_counts',
        default=[1, 5, 10, 20, 30, 50, 75, 100]
    )

    results = benchmark_trajectory_head_by_agent_count(
        model=model,
        device=device,
        agent_counts=agent_counts,
    )

    save_path = get_nested(cfg, 'eval', 'agent_density_save_path', default='')
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"\nSaved agent-density results to: {out_path}")