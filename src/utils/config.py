# Create config utility module
import os
import yaml
from typing import Dict, Any, Optional
from omegaconf import OmegaConf

def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and apply overrides.

    Args:
        config_path: Path to configuration file (default: configs/default.yaml)
        overrides: Dictionary of config values to override

    Returns:
        Dict containing configuration
    """
    default_config_path = os.path.join('configs', 'default.yaml')

    with open(default_config_path, 'r') as f:
        default_config = yaml.safe_load(f)

    config = OmegaConf.create(default_config)

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        custom_conf = OmegaConf.create(custom_config)
        config = OmegaConf.merge(config, custom_conf)

    if overrides:
        override_conf = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_conf)

    resolved_config = OmegaConf.to_container(config, resolve=True)

    return resolved_config


