#!/usr/bin/env python
"""
Configuration Manager for DREMA Suite.
Loads YAML configurations and provides hierarchical dictionary and attribute-style access.
"""

import os
import yaml
from typing import Any, Dict, Optional


class ConfigDict(dict):
    """A dictionary supporting attribute-style access (e.g., cfg.perception.voxel_size)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k] = ConfigDict(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Configuration has no parameter '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[name] = value

    def get_nested(self, path: str, default: Any = None) -> Any:
        """Retrieves a nested key using dot notation (e.g. 'perception.tracking.subsample')."""
        parts = path.split(".")
        curr = self
        for p in parts:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
            else:
                return default
        return curr

    def set_nested(self, path: str, value: Any) -> None:
        """Sets a nested key using dot notation."""
        parts = path.split(".")
        curr = self
        for p in parts[:-1]:
            if p not in curr or not isinstance(curr[p], dict):
                curr[p] = ConfigDict()
            curr = curr[p]
        curr[parts[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Converts back to pure Python dict."""
        res = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                res[k] = v.to_dict()
            else:
                res[k] = v
        return res


def load_config(config_path: str = "configs/drema_default.yaml") -> ConfigDict:
    """Loads a YAML configuration file into a ConfigDict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return ConfigDict(data)
