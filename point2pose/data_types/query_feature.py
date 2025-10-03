from typing import Sequence, Tuple, NamedTuple
import torch


class QueryFeatures(NamedTuple):
    """Query features used to compute trajectories.

    These are sampled from the query frames and are a full descriptor of the
    tracked points. They can be acquired from a query image and then reused in a
    separate video.

    Attributes:
      lowres: Low-resolution features, one for each resolution; each has shape
        [batch, num_query_points, 256]
      hires: High-resolution features, one for each resolution; each has shape
        [batch, num_query_points, 64]
      resolutions: Resolutions used for trajectory computation.  There will be one
        entry for the initialization, and then an entry for each PIPs refinement
        resolution.
    """

    lowres: Sequence[torch.Tensor]
    hires: Sequence[torch.Tensor]
    resolutions: Sequence[Tuple[int, int]]
