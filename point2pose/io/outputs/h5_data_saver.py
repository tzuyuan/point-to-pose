import os
import h5py
import numpy as np
from scipy.io import savemat

from typing import Dict, Any, Tuple, Union


DTypeSpec = Union[np.dtype, Tuple[np.dtype, Tuple[int, ...]]]


class H5DataSaver:
    """
    Append-optimized HDF5 writer for mixed scalar, fixed-shape, and ragged fields.

    fixed_fields: dict[str, DTypeSpec]
        - Value is either:
            * np.dtype (scalar per row), e.g. np.float64
            * (np.dtype, shape) for fixed-shape arrays per row, e.g. (np.float64, (4,4))
    ragged_fields: dict[str, np.dtype]
        - Value is base dtype for variable-length (ragged) arrays; each row can have different length.

    Example:
        saver = H5DataSaver(
            "run.h5",
            fixed_fields={
                "timestamp": np.float64,
                "frame_id": np.int64,
                "pose": (np.float64, (4,4)),      # fixed 4x4
                "t": (np.float64, (3,)),          # fixed 3-vector
            },
            ragged_fields={
                "residuals": np.float64,          # variable length
                "inliers": np.bool_,
            },
            compression="gzip",
            flush_every=100,
        )
        saver.append({"timestamp": 0.0, "frame_id": 0, "pose": np.eye(4), "t": [0,0,0],
                      "residuals": [0.1, 0.2], "inliers": [True, False]})
        saver.close()
    """

    def __init__(
        self,
        h5_path: str,
        fixed_fields: Dict[str, DTypeSpec],
        ragged_fields: Dict[str, np.dtype] = None,
        *,
        compression: str = "gzip",
        chunks: bool = True,
        flush_every: int = 100,
        file_mode: str = "a",
        libver: str = "latest",
    ):
        self.h5_path = h5_path
        os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)

        self.fixed_fields: Dict[str, DTypeSpec] = dict(fixed_fields or {})
        self.ragged_fields: Dict[str, np.dtype] = dict(ragged_fields or {})
        self.compression = compression
        self.chunks = chunks
        self.flush_every = max(1, int(flush_every))
        self._append_count_since_flush = 0
        self._row_count_cache = None

        self._f = h5py.File(self.h5_path, file_mode, libver=libver)
        self._ensure_datasets()

    # ---------- Public API ----------

    def append(self, row: Dict[str, Any]) -> int:
        """
        Append a single row. Returns row index written.
        - Missing required fields -> KeyError
        - Fixed-shape fields are checked/reshaped to declared shape
        """
        n = self._length()

        # 1) Resize all datasets by +1
        for name in self.fixed_fields.keys():
            self._f[name].resize((n + 1,) + self._field_shape(name))
        for name in self.ragged_fields.keys():
            self._f[name].resize((n + 1,))

        # 2) Write fixed fields
        for name, spec in self.fixed_fields.items():
            if name not in row:
                raise KeyError(f"Missing required fixed field '{name}' in row.")
            dtype, shape = self._parse_spec(spec)
            val = np.asarray(row[name], dtype=dtype)
            if shape:  # fixed-shape array
                try:
                    val = val.reshape(shape)
                except Exception as e:
                    raise ValueError(
                        f"Field '{name}' expected shape {shape}, got {val.shape}"
                    ) from e
            self._f[name][n] = val

        # 3) Write ragged fields (vlen)
        for name, base_dtype in self.ragged_fields.items():
            arr = row.get(name, None)
            if arr is None:
                arr = np.asarray([], dtype=base_dtype)
            else:
                arr = np.asarray(arr, dtype=base_dtype)
            self._f[name][n] = arr

        # 4) Flush periodically
        self._row_count_cache = n + 1
        self._append_count_since_flush += 1
        if self._append_count_since_flush >= self.flush_every:
            self.flush()

        return n

    def length(self) -> int:
        """Number of appended rows."""
        return self._length()

    def flush(self):
        self._f.flush()
        self._append_count_since_flush = 0

    def close(self):
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def to_dict_of_arrays(self) -> Dict[str, np.ndarray]:
        """
        Load entire file into memory. Ragged fields return dtype=object arrays.
        """
        out = {}
        for name in self._f.keys():
            out[name] = self._f[name][...]
        return out

    # ---------- Private helpers ----------

    def _parse_spec(self, spec: DTypeSpec) -> Tuple[np.dtype, Tuple[int, ...]]:
        if isinstance(spec, tuple):
            dtype, shape = spec
            return (np.dtype(dtype), tuple(shape))
        else:
            return (np.dtype(spec), ())

    def _field_shape(self, name: str) -> Tuple[int, ...]:
        spec = self.fixed_fields[name]
        _, shape = self._parse_spec(spec)
        return shape

    def _ensure_datasets(self):
        # Fixed: scalars or fixed-shape arrays per row
        for name, spec in self.fixed_fields.items():
            dtype, shape = self._parse_spec(spec)
            if name not in self._f:
                self._f.create_dataset(
                    name,
                    shape=(0,) + shape,  # e.g. (0,), (0,3), (0,4,4)
                    maxshape=(None,) + shape,
                    dtype=dtype,
                    chunks=self.chunks,
                    compression=self.compression,
                )

        # Ragged: variable-length vectors per row
        for name, base_dtype in self.ragged_fields.items():
            if name not in self._f:
                vlen_dtype = h5py.vlen_dtype(np.dtype(base_dtype))
                self._f.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=vlen_dtype,
                    chunks=self.chunks,
                    compression=self.compression,
                )

    def _length(self) -> int:
        if self._row_count_cache is not None:
            return self._row_count_cache
        any_name = next(iter(self._f.keys()), None)
        self._row_count_cache = 0 if any_name is None else self._f[any_name].shape[0]
        return self._row_count_cache


# -------- Convenience: registration-logger-specific factory --------


def make_registration_h5_saver(h5_path: str, flush_every: int = 100) -> H5DataSaver:
    """
    Saver preset for your registration stats. Add more fixed fields if needed:
    e.g., ("pose", (np.float64, (4,4))) or ("rpy", (np.float64, (3,)))
    """
    fixed_fields: Dict[str, DTypeSpec] = {
        "timestamp": np.float64,
        "frame_id": np.int64,
        "obj_id": np.int64,
        "num_points": np.int64,
        "iter": np.int64,
        "thr": np.float64,
        "res_mean": np.float64,
        "res_median": np.float64,
        "res_max": np.float64,
        "num_inliers": np.int64,
        "total_points": np.int64,
        "mean_residual_inliers": np.float64,
        "mean_residual_outliers": np.float64,
        # Uncomment to store pose/angles per frame as fixed fields:
        # "pose": (np.float64, (4,4)),
        # "rpy": (np.float64, (3,)),
    }
    ragged_fields = {
        "residuals": np.float64,
        "inliers": np.bool_,
    }
    return H5DataSaver(
        h5_path,
        fixed_fields=fixed_fields,
        ragged_fields=ragged_fields,
        compression="gzip",
        flush_every=flush_every,
    )


# -------- Optional: export helpers --------


def export_h5_to_npz(h5_path: str, npz_path: str):
    with h5py.File(h5_path, "r") as f:
        out = {k: f[k][...] for k in f.keys()}
    np.savez_compressed(npz_path, **out)


def export_h5_to_mat(h5_path: str, mat_path: str):
    """
    Writes a v7.2 MAT (not HDF5-based). For a v7.3 MAT, open HDF5 in MATLAB and save there.
    """

    with h5py.File(h5_path, "r") as f:
        out = {k: f[k][...] for k in f.keys()}
    savemat(mat_path, out)
