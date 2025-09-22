# src/point2pose/core/registry.py
from typing import Callable, Dict, Type


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable] = {}

    def register_module(self, key: str):
        def decorator(cls_or_fn):
            if key in self._items:
                raise KeyError(f"{self.name}: duplicate key '{key}'")
            self._items[key] = cls_or_fn
            return cls_or_fn

        return decorator

    def get(self, key: str):
        if key not in self._items:
            raise KeyError(f"{self.name}: '{key}' not found")
        return self._items[key]


REGISTER = Registry("register")
TRACKER = Registry("tracker")
STATE = Registry("state")
SAMPLER = Registry("sampler")
OPTIM = Registry("optimizer")
