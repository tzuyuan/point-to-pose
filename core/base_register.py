from abc import ABC, abstractmethod


class Register(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def register(self, src_pcd, tgt_pcd, init_pose=None):
        raise NotImplementedError(
            "Registrator method must be implemented by subclasses."
        )
