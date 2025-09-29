from abc import ABC, abstractmethod


class BaseSource(ABC):
    """Abstract base for all input sources (live or dataset)."""

    @abstractmethod
    def open(self): ...
    @abstractmethod
    def close(self): ...
    @abstractmethod
    def __iter__(self): ...


class BaseDatasetSource(BaseSource):
    """Finite dataset-like source."""

    @abstractmethod
    def __len__(self): ...
    def reset(self): ...
    def seek(self, index: int): ...


class BaseLiveSource(BaseSource):
    """Streaming source (potentially infinite)."""

    def start(self):
        """
        Alias function to start the source.
        """
        self.open()

    def stop(self):
        """
        Alias function to stop the source.
        """
        self.close()
