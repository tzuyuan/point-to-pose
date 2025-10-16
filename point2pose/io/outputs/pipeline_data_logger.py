import os
import time
import numpy as np

import h5py


class EndRunLogger:
    """
    Collect everything in memory; write once at the end.
    Ragged arrays are stored in packed form: data_flat + offsets (CSR-style).
    """

    def __init__(self):
        # scalar-per-row
        self.timestamps = []
        self.frame_ids = []
        self.obj_ids = []

        # fixed 4x4 pose per row
        self.poses_4x4 = []  # list of (4,4) arrays

        # registration scalar stats (optional)
        self.reg_iter = []
        self.reg_thr = []
        self.res_mean = []
        self.res_median = []
        self.res_max = []
        self.num_inliers = []
        self.total_points = []
        self.mean_residual_inliers = []
        self.mean_residual_outliers = []

        # ragged fields (store as python lists now; pack at finalize)
        self.track2d_list = []  # each (Ni,2)
        self.track3d_list = []  # each (Mi,3) optional, can be None
        self.keypoints_list = []  # each (Ki,3)
        self.uncertainties_list = []  # each (Ki,)
        self.inliers_list = []  # each (Ki,) bool

    @staticmethod
    def _pack_ragged(list_of_arrays, expected_cols=None, dtype=np.float64):
        """
        Returns (data_flat, offsets, cols)
        - data_flat: 1D array of concatenated rows (row-major)
        - offsets:   int64 array of length (n_rows+1)
        - cols:      number of columns (metadata; None->1)
        """
        n = len(list_of_arrays)
        cols = 1 if expected_cols is None else int(expected_cols)
        sizes = np.empty(n, dtype=np.int64)
        # compute total elements
        total_elems = 0
        for i, arr in enumerate(list_of_arrays):
            if arr is None:
                sizes[i] = 0
                continue
            a = np.asarray(arr)
            if a.ndim == 2:
                if expected_cols is not None and a.shape[1] != expected_cols:
                    raise ValueError(
                        f"Ragged column mismatch: got {a.shape[1]} vs expected {expected_cols}"
                    )
                k = a.shape[0] * (a.shape[1])
                sizes[i] = k
                total_elems += k
            elif a.ndim == 1:
                if expected_cols not in (None, 1):
                    raise ValueError("Expected 2D ragged but got 1D")
                sizes[i] = a.shape[0]
                total_elems += a.shape[0]
            else:
                raise ValueError("Unsupported ragged rank")
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])

        # dtype handling
        if dtype == np.bool_:
            data_flat = np.zeros(total_elems, dtype=np.bool_)
        else:
            data_flat = np.zeros(total_elems, dtype=dtype)

        # fill
        pos = 0
        for arr in list_of_arrays:
            if arr is None:
                continue
            a = np.asarray(arr)
            data_flat[pos : pos + a.size] = a.reshape(-1)
            pos += a.size

        return data_flat, offsets, cols

    def add_row(
        self,
        frame_id: int,
        obj_id: int,
        pose_4x4: np.ndarray,
        *,
        track2d=None,  # (N,2)
        track3d=None,  # (M,3) optional
        keypoints=None,  # (K,3)
        uncertainties=None,  # (K,)
        inliers=None,  # (K,) bool
        reg_stats: dict | None = None,
        timestamp: float | None = None,
    ):
        self.timestamps.append(time.time() if timestamp is None else float(timestamp))
        self.frame_ids.append(int(frame_id))
        self.obj_ids.append(int(obj_id))
        self.poses_4x4.append(np.asarray(pose_4x4, dtype=np.float64))

        rs = reg_stats or {}
        residuals = np.asarray(rs.get("residuals", []), dtype=float)
        inl = np.asarray(rs.get("inliers", []), dtype=bool)

        self.reg_iter.append(int(rs.get("iter", -1)))
        self.reg_thr.append(float(rs.get("thr", -1.0)))
        self.res_mean.append(float(np.mean(residuals)) if residuals.size else -1.0)
        self.res_median.append(float(np.median(residuals)) if residuals.size else -1.0)
        self.res_max.append(float(np.max(residuals)) if residuals.size else -1.0)
        self.num_inliers.append(int(np.sum(inl)) if inl.size else -1)
        self.total_points.append(int(residuals.size))
        if residuals.size and inl.size == residuals.size:
            self.mean_residual_inliers.append(
                float(np.mean(residuals[inl])) if inl.any() else -1.0
            )
            self.mean_residual_outliers.append(
                float(np.mean(residuals[~inl])) if (~inl).any() else -1.0
            )
        else:
            self.mean_residual_inliers.append(-1.0)
            self.mean_residual_outliers.append(-1.0)

        self.track2d_list.append(None if track2d is None else np.asarray(track2d))
        self.track3d_list.append(None if track3d is None else np.asarray(track3d))
        self.keypoints_list.append(None if keypoints is None else np.asarray(keypoints))
        self.uncertainties_list.append(
            None if uncertainties is None else np.asarray(uncertainties)
        )
        self.inliers_list.append(
            None if inliers is None else np.asarray(inliers, dtype=bool)
        )

    def finalize_npz(self, path: str):
        """
        Write a single .npz with packed ragged arrays + scalars/fixed fields.
        Fastest + simplest. Great if you live in Python/NumPy. MATLAB can also load .npz via third-party or Python bridge.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # pack ragged
        t2d_flat, t2d_off, t2d_cols = self._pack_ragged(
            self.track2d_list, expected_cols=2, dtype=np.float64
        )
        t3d_flat, t3d_off, t3d_cols = self._pack_ragged(
            self.track3d_list, expected_cols=3, dtype=np.float64
        )
        kp_flat, kp_off, kp_cols = self._pack_ragged(
            self.keypoints_list, expected_cols=3, dtype=np.float64
        )
        un_flat, un_off, un_cols = self._pack_ragged(
            self.uncertainties_list, expected_cols=1, dtype=np.float64
        )
        in_flat, in_off, in_cols = self._pack_ragged(
            self.inliers_list, expected_cols=1, dtype=np.bool_
        )

        np.savez_compressed(
            path,
            timestamp=np.asarray(self.timestamps, dtype=np.float64),
            frame_id=np.asarray(self.frame_ids, dtype=np.int64),
            obj_id=np.asarray(self.obj_ids, dtype=np.int64),
            pose_4x4=np.stack(self.poses_4x4, axis=0),  # (R,4,4)
            reg_iter=np.asarray(self.reg_iter, dtype=np.int64),
            reg_thr=np.asarray(self.reg_thr, dtype=np.float64),
            res_mean=np.asarray(self.res_mean, dtype=np.float64),
            res_median=np.asarray(self.res_median, dtype=np.float64),
            res_max=np.asarray(self.res_max, dtype=np.float64),
            num_inliers=np.asarray(self.num_inliers, dtype=np.int64),
            total_points=np.asarray(self.total_points, dtype=np.int64),
            mean_residual_inliers=np.asarray(
                self.mean_residual_inliers, dtype=np.float64
            ),
            mean_residual_outliers=np.asarray(
                self.mean_residual_outliers, dtype=np.float64
            ),
            # packed ragged
            track2d_flat=t2d_flat,
            track2d_off=t2d_off,
            track2d_cols=np.int64(t2d_cols),
            track3d_flat=t3d_flat,
            track3d_off=t3d_off,
            track3d_cols=np.int64(t3d_cols),
            keypoints_flat=kp_flat,
            keypoints_off=kp_off,
            keypoints_cols=np.int64(kp_cols),
            uncertainties_flat=un_flat,
            uncertainties_off=un_off,
            uncertainties_cols=np.int64(un_cols),
            inliers_flat=in_flat,
            inliers_off=in_off,
            inliers_cols=np.int64(in_cols),
        )

    def finalize_h5(self, path: str):
        """
        Single-shot HDF5 write (no resizes). Use if you want HDF5 specifically.
        """

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with h5py.File(path, "w") as f:
            # scalars/fixed
            f.create_dataset(
                "timestamp", data=np.asarray(self.timestamps, dtype=np.float64)
            )
            f.create_dataset(
                "frame_id", data=np.asarray(self.frame_ids, dtype=np.int64)
            )
            f.create_dataset("obj_id", data=np.asarray(self.obj_ids, dtype=np.int64))
            f.create_dataset("pose_4x4", data=np.stack(self.poses_4x4, axis=0))

            f.create_dataset("reg_iter", data=np.asarray(self.reg_iter, dtype=np.int64))
            f.create_dataset("reg_thr", data=np.asarray(self.reg_thr, dtype=np.float64))
            f.create_dataset(
                "res_mean", data=np.asarray(self.res_mean, dtype=np.float64)
            )
            f.create_dataset(
                "res_median", data=np.asarray(self.res_median, dtype=np.float64)
            )
            f.create_dataset("res_max", data=np.asarray(self.res_max, dtype=np.float64))
            f.create_dataset(
                "num_inliers", data=np.asarray(self.num_inliers, dtype=np.int64)
            )
            f.create_dataset(
                "total_points", data=np.asarray(self.total_points, dtype=np.int64)
            )
            f.create_dataset(
                "mean_residual_inliers",
                data=np.asarray(self.mean_residual_inliers, dtype=np.float64),
            )
            f.create_dataset(
                "mean_residual_outliers",
                data=np.asarray(self.mean_residual_outliers, dtype=np.float64),
            )

            # packed ragged
            def pack_and_write(name, lst, cols, dtype):
                flat, off, cols_val = self._pack_ragged(
                    lst, expected_cols=cols, dtype=dtype
                )
                f.create_dataset(f"{name}_flat", data=flat)
                f.create_dataset(f"{name}_off", data=off)
                f.attrs[f"{name}_cols"] = int(cols_val)

            pack_and_write("track2d", self.track2d_list, 2, np.float64)
            pack_and_write("track3d", self.track3d_list, 3, np.float64)
            pack_and_write("keypoints", self.keypoints_list, 3, np.float64)
            pack_and_write("uncertainties", self.uncertainties_list, 1, np.float64)
            pack_and_write("inliers", self.inliers_list, 1, np.bool_)
