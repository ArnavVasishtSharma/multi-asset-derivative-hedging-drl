"""
utils/config.py
----------------
YAML configuration loader with CLI override support.

Usage:
    from utils.config import load_config

    # Load from YAML
    cfg = load_config("../configs/novelty1.yaml")

    # Load with CLI overrides
    cfg = load_config("../configs/novelty1.yaml", overrides={"lr_actor": 5e-5, "timesteps": 2_000_000})

    # Access values
    print(cfg["training"]["timesteps"])
"""

import os
import logging
from typing import Optional, Dict, Any

import yaml

log = logging.getLogger(__name__)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    merged = base.copy()
    for k, v in overrides.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _flatten_to_nested(flat: Dict[str, Any]) -> dict:
    """
    Convert flat key=value overrides to nested dict.

    Supports dot-notation: {"training.lr": 1e-4} → {"training": {"lr": 1e-4}}
    Also supports plain keys at the top level.
    """
    nested: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested


def load_config(
    yaml_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Load a YAML configuration file and merge with optional overrides.

    Parameters
    ----------
    yaml_path : path to YAML config file
    overrides : flat dict of key=value or dot-notation overrides

    Returns
    -------
    config : dict with fully merged configuration
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if overrides:
        nested = _flatten_to_nested(overrides)
        config = _deep_merge(config, nested)

    log.info(f"Loaded config from {yaml_path} ({len(config)} top-level keys)")
    return config


def config_to_flat(config: dict, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten a nested config dict to dot-notation keys.

    Useful for logging to W&B or printing.
    """
    flat = {}
    for k, v in config.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(config_to_flat(v, key))
        else:
            flat[key] = v
    return flat
