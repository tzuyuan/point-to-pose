# src/point2pose/core/build.py
from typing import Mapping, Any
from point2pose.core.module_registry import REGISTER, TRACKER, STATE, SAMPLER, OPTIMIZER
import point2pose.modules


def build_from_cfg(cfg: Mapping[str, Any], registry):
    """
    cfg = {'type': name of the module. it is required to match the registered name. 'params': {...}}
    """
    mod_type = cfg["type"]
    params = cfg.get("params", {}) or {}
    cls = registry.get(mod_type)
    return cls(params)
