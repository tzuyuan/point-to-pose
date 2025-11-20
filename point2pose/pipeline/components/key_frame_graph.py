class KeyFrameGraph:
    """
    Optimizer for keyframes only (Pose Graph Optimization).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Initialize GTSAM graph or similar here

    def add_keyframe(self, pose, landmarks):
        """
        Add a new keyframe and its observed landmarks to the global graph.
        """
        pass

    def optimize(self):
        """
        Perform global optimization.
        """
        pass
