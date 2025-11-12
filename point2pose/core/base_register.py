from abc import ABC, abstractmethod
import os


class Register(ABC):
    def __init__(self, config=None):
        self.config = config
        self.debug_level = config.get("debug_level", 0)
        self.debug_dir = config.get("debug_dir", None)

        self.type = config.get("type", "base_register")

        if self.debug_level > 0 and self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

    @abstractmethod
    def register(self, source_pcd, target_pcd, init_pose=None):
        raise NotImplementedError(
            "Registrator method must be implemented by subclasses."
        )
