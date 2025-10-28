from typing import Callable, Dict, Type


class ModuleRegistry:
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


REGISTER = ModuleRegistry("register")
TRACKER = ModuleRegistry("tracker")
STATE = ModuleRegistry("state")
SAMPLER = ModuleRegistry("sampler")
SEGMENTER = ModuleRegistry("segmenter")
CRITERION = ModuleRegistry("criterion")
OPTIMIZER = ModuleRegistry("optimizer")
