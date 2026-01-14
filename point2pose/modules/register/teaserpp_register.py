import numpy as np

from point2pose.core.base_register import Register
from point2pose.core.module_registry import REGISTER
from point2pose.utils.transform import transform_pts


@REGISTER.register_module("teaserpp")
class TeaserPPRegister(Register):
    """
    TEASER++ robust point cloud registration (known correspondences).

    Returns stats with:
      - clique_idx: indices (into original correspondences) of the inlier max clique
      - inlier_idx: indices (into original correspondences) of the translation inliers
      - clique_mask: (N,) bool mask for clique_idx
      - inlier_mask: (N,) bool mask for inlier_idx
    """

    def __init__(self, config=None):
        super().__init__(config or {})
        cfg = config or {}

        self._noise_bound = float(cfg.get("noise_bound", 0.1))
        self._cbar2 = float(cfg.get("cbar2", 1.0))
        self._estimate_scaling = bool(cfg.get("estimate_scaling", False))

        self._rot_alg = cfg.get("rotation_estimation_algorithm", "GNC_TLS")
        self._rot_gnc_factor = float(cfg.get("rotation_gnc_factor", 1.4))
        self._rot_max_iters = int(cfg.get("rotation_max_iterations", 100))
        self._rot_cost_thres = float(cfg.get("rotation_cost_threshold", 1e-12))

        self._min_inliers = int(cfg.get("min_inliers", 3))

        self._teaser = None

    def _lazy_import(self):
        if self._teaser is None:
            import teaserpp_python

            self._teaser = teaserpp_python
        return self._teaser

    @staticmethod
    def _safe_int_list(x):
        """Convert TEASER++ returned container to a plain Python list[int]."""
        if x is None:
            return []
        try:
            return [int(v) for v in list(x)]
        except Exception:
            # last resort: try numpy
            return [int(v) for v in np.asarray(x).reshape(-1)]

    def register(self, src_pcd, tgt_pcd, init_pose=None, **kwargs):
        stats = {}

        if src_pcd is None or tgt_pcd is None:
            return np.eye(4), {"valid": False, "reason": "empty_input"}

        src_pcd = np.asarray(src_pcd, dtype=np.float64)
        tgt_pcd = np.asarray(tgt_pcd, dtype=np.float64)

        if src_pcd.ndim != 2 or src_pcd.shape[1] != 3 or tgt_pcd.shape != src_pcd.shape:
            raise ValueError(
                f"Expected src/tgt shape (N,3) with same N. Got {src_pcd.shape}, {tgt_pcd.shape}"
            )

        N = src_pcd.shape[0]
        if N < self._min_inliers:
            stats.update({"valid": False, "reason": "too_few_points", "N": int(N)})
            return np.eye(4), stats

        # Apply init pose to source points if provided (same convention as your SVD register)
        p0 = (
            transform_pts(init_pose, src_pcd)
            if init_pose is not None
            else src_pcd.copy()
        )

        # TEASER++ wants 3xN
        src = np.ascontiguousarray(p0.T)  # (3,N)
        dst = np.ascontiguousarray(tgt_pcd.T)  # (3,N)

        teaserpp = self._lazy_import()

        params = teaserpp.RobustRegistrationSolver.Params()
        params.noise_bound = self._noise_bound
        params.cbar2 = self._cbar2
        params.estimate_scaling = self._estimate_scaling

        # rotation algorithm enum
        if isinstance(self._rot_alg, str):
            alg = self._rot_alg.upper().replace("-", "_")
            if alg in ["GNC_TLS", "GNCTLS"]:
                params.rotation_estimation_algorithm = (
                    teaserpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
                )
            elif alg in ["FGR", "FAST_GLOBAL_REGISTRATION"]:
                params.rotation_estimation_algorithm = (
                    teaserpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.FGR
                )
            else:
                raise ValueError(
                    f"Unknown rotation_estimation_algorithm: {self._rot_alg}"
                )
        else:
            params.rotation_estimation_algorithm = self._rot_alg

        params.rotation_gnc_factor = self._rot_gnc_factor
        params.rotation_max_iterations = self._rot_max_iters
        params.rotation_cost_threshold = self._rot_cost_thres

        solver = teaserpp.RobustRegistrationSolver(params)
        solver.solve(src, dst)
        sol = solver.getSolution()

        # Pose
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(sol.rotation, dtype=np.float64)
        T[:3, 3] = np.asarray(sol.translation, dtype=np.float64).reshape(3)

        # --- Inlier extraction ---
        # clique indices (max clique) are indices into original correspondences
        clique_idx = []
        inlier_idx = []

        # Not all bindings expose all methods; guard with hasattr.
        if hasattr(solver, "getInlierMaxClique"):
            clique_idx = self._safe_int_list(solver.getInlierMaxClique())

        # Translation inliers indices (into original correspondences)
        if hasattr(solver, "getTranslationInliers"):
            inlier_idx = self._safe_int_list(solver.getTranslationInliers())

        # Some builds provide "map" + "mask" instead; reconstruct if needed:
        # - map: indices of the inlier max clique in original measurements
        # - mask: boolean mask over that clique indicating which survived translation TLS
        if (
            (not inlier_idx)
            and hasattr(solver, "getTranslationInliersMap")
            and hasattr(solver, "getTranslationInliersMask")
        ):
            clique_map = self._safe_int_list(solver.getTranslationInliersMap())
            mask_over_clique = np.asarray(
                solver.getTranslationInliersMask(), dtype=bool
            ).reshape(-1)
            # Build final inliers in original indexing
            if len(clique_map) == len(mask_over_clique):
                inlier_idx = [
                    clique_map[i] for i, m in enumerate(mask_over_clique) if m
                ]
            # If clique_idx not provided, use clique_map
            if not clique_idx:
                clique_idx = clique_map

        # Build masks (length N)
        clique_mask = np.zeros(N, dtype=bool)
        inlier_mask = np.zeros(N, dtype=bool)

        if len(clique_idx) > 0:
            clique_idx_arr = np.asarray(clique_idx, dtype=int)
            clique_idx_arr = clique_idx_arr[
                (clique_idx_arr >= 0) & (clique_idx_arr < N)
            ]
            clique_mask[clique_idx_arr] = True

        if len(inlier_idx) > 0:
            inlier_idx_arr = np.asarray(inlier_idx, dtype=int)
            inlier_idx_arr = inlier_idx_arr[
                (inlier_idx_arr >= 0) & (inlier_idx_arr < N)
            ]
            inlier_mask[inlier_idx_arr] = True

        # Populate stats
        stats["valid"] = bool(getattr(sol, "valid", True))
        stats["scale"] = float(getattr(sol, "scale", 1.0))
        stats["rotation"] = T[:3, :3].copy()
        stats["translation"] = T[:3, 3].copy()
        stats["N"] = int(N)

        stats["clique_idx"] = clique_idx
        stats["inliers"] = np.array(inlier_idx)
        stats["clique_mask"] = clique_mask
        stats["inlier_mask"] = inlier_mask
        stats["num_clique"] = int(clique_mask.sum())
        stats["num_inliers"] = int(inlier_mask.sum())

        stats["residuals"] = np.linalg.norm(src.T - dst.T, axis=1)

        if self.debug_level > 1:
            print(
                f"[TeaserPPRegister] clique: {stats['num_clique']}  inliers: {stats['num_inliers']}  scale: {stats['scale']}"
            )

        # Compose back with init_pose if we used it
        if init_pose is not None:
            T = T @ init_pose

        return T, stats
