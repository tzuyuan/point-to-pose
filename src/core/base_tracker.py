from abc import ABC, abstractmethod


class Tracker(ABC):
    def __init__(self, config):
        self.name = "base_tracker"

    @abstractmethod
    def track_once(self, frame):
        """
        Track the given the frame once.
        """
        raise NotImplementedError("Track method must be implemented by subclasses.")

    @abstractmethod
    def add_query_points(self, frame_id, new_points):
        """
        Add new points to be tracked.
        """
        raise NotImplementedError(
            "Add points method must be implemented by subclasses."
        )

    # @abstractmethod
    # def get_query_features(self):
    #     """
    #     Get the features of the query points.
    #     """
    #     raise NotImplementedError(
    #         "Get query features method must be implemented by subclasses."
    #     )
