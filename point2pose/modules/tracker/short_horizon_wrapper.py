import numpy as np

from point2pose.core.base_tracker import Tracker
from point2pose.core.module_registry import TRACKER


@TRACKER.register_module("short_horizon")
class ShortHorizonWrapper(Tracker):
    """
    Ablation wrapper that destroys long-range point identity.

    Wraps a base tracker (e.g. TAPIR) and permanently kills any tracked
    point that has been invisible for K consecutive frames. Killed points
    are reported with visible=False, tracks=NaN, uncertainty=inf forever
    after, even if the inner tracker re-localizes them. This isolates the
    contribution of long-range identity persistence from the rest of the
    pipeline (RANSAC, TSDF refinement, factor graph, recovery manager),
    while keeping the underlying tracker model and feature representation
    constant.

    Index identity is preserved (required by the global track2obj_map),
    so killed slots remain in place; the existing FrontEnd / KeyFrameManager
    resampling path replenishes coverage by adding new query points.
    """

    def __init__(self, config):
        super().__init__(config)
        self.K = int(config["K"])
        self.name = f"short_horizon_K{self.K}"

        inner_cfg = config["inner"]
        inner_cls = TRACKER.get(inner_cfg["type"])
        self.inner = inner_cls(inner_cfg.get("params", {}) or {})

        self._invisible_count = np.zeros(0, dtype=np.int32)
        self._killed = np.zeros(0, dtype=bool)

    def _ensure_capacity(self, n):
        cur = self._invisible_count.shape[0]
        if cur < n:
            grow = n - cur
            self._invisible_count = np.concatenate(
                [self._invisible_count, np.zeros(grow, dtype=np.int32)]
            )
            self._killed = np.concatenate(
                [self._killed, np.zeros(grow, dtype=bool)]
            )

    def initialize(self, frame):
        return self.inner.initialize(frame)

    def track_once(self, frame):
        tracks, unc, vis = self.inner.track_once(frame)
        n = tracks.shape[0]
        self._ensure_capacity(n)

        vis = vis.astype(bool).copy()
        self._invisible_count[:n] = np.where(
            vis, 0, self._invisible_count[:n] + 1
        )
        newly_killed = (~self._killed[:n]) & (self._invisible_count[:n] >= self.K)
        self._killed[:n] |= newly_killed

        killed_mask = self._killed[:n]
        if killed_mask.any():
            vis[killed_mask] = False
            tracks = tracks.copy()
            tracks[killed_mask] = np.nan
            unc = unc.copy()
            unc[killed_mask] = np.inf

        n_killed_total = int(self._killed[:n].sum())
        n_killed_new = int(newly_killed.sum())
        if n_killed_new > 0:
            print(
                f"[SH(K={self.K})] frame {getattr(frame, 'id', '?')}: "
                f"+{n_killed_new} killed (total {n_killed_total}/{n})"
            )

        return tracks, unc, vis

    def add_query_points(self, frame, new_points):
        new_idx = self.inner.add_query_points(frame, new_points)
        n_added = len(new_idx)
        if n_added > 0:
            self._invisible_count = np.concatenate(
                [self._invisible_count, np.zeros(n_added, dtype=np.int32)]
            )
            self._killed = np.concatenate(
                [self._killed, np.zeros(n_added, dtype=bool)]
            )
        return new_idx
