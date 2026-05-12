# utils/config_loader.py
#
# New file for IntentTrajNet-AV2
# Author: [Your Name] — Bachelor Thesis, GUC 2025
#
# Loads YAML config files and provides a clean dict interface.
# Used by train.py and eval.py to read model/training configuration.

import yaml
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load a YAML config file and return as a nested dict.

    Args:
        config_path: path to .yaml config file

    Returns:
        config dict with all settings
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    print(f"Loaded config: {config_path}")
    return config


def get_config_arg() -> str:
    """
    Parse --config argument from command line.

    Usage:
        python training/train.py --config configs/v2_mlp.yaml
        python training/eval.py  --config configs/v1_baseline.yaml

    Returns:
        path to config file as string
    """
    parser = argparse.ArgumentParser(description="IntentTrajNet-AV2")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/v2_mlp.yaml)"
    )
    args = parser.parse_args()
    return args.config


def get_nested(config: dict, *keys, default=None):
    """
    Safely get a nested value from config dict.

    Example:
        lr = get_nested(cfg, 'training', 'optimizer', 'lr', default=1e-4)

    Args:
        config: nested config dict
        *keys:  sequence of keys to traverse
        default: value to return if key not found

    Returns:
        value at config[keys[0]][keys[1]]... or default
    """
    val = config
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key, default)
        else:
            return default
    return val if val is not None else default