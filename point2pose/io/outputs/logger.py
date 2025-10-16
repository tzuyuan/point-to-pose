import os
import atexit
import signal
from typing import Dict, Any, Iterable, Set, Tuple
import numpy as np
import h5py
from scipy.io import savemat


class DataLogger:
    """
    Ultra-low-overhead, end-of-run saver:
      - Buffers rows in memory using Python lists (O(1) append)
      - Saves once at exit to NPZ (+ MAT cell arrays for ragged)
      - Optional HDF5 if desired (off by default)

    Ragged fields are packed as:
      <name>_data:    concatenated 1-D array
      <name>_offsets: int64 start indices for each row
      <name>_lengths: int64 lengths per row
      <name>_cell:    (MAT only) cell array, one vector per row

    Fixed-shape fields are stacked to proper NumPy arrays (N, ...).
    If stacking fails due to inconsistent shapes, we fall back to object array.

    Usage:
        saver = DataLogger(
            out_dir="logs",
            base_name="meata_data",
            ragged_fields={"residuals","inliers"},
            also_save_h5=False,
        )
        saver.log({...})  # per-frame
        # automatic save at process exit, or call saver.save_now() yourself
    """

    def __init__(
        self,
        out_dir: str,
        base_name: str = "meata_data",
        ragged_fields: Iterable[str] = (),
        *,
        also_save_h5: bool = False,
        h5_compression: str = "gzip",
        h5_path_override: str | None = None,
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.base_name = base_name
        self.ragged_fields: Set[str] = set(ragged_fields or [])
        self.also_save_h5 = also_save_h5
        self.h5_compression = h5_compression
        self.h5_path_override = h5_path_override

        # Internal buffers: dict[name] -> list[values]
        self._buf: Dict[str, list] = {}
        self._keys_frozen = False  # freeze field set after first log
        self._saved = False

        # Ensure automatic save at exit / signals
        atexit.register(self._atexit_hook)
        try:
            signal.signal(signal.SIGINT, self._signal_hook)
            signal.signal(signal.SIGTERM, self._signal_hook)
        except Exception:
            # Some environments (e.g., notebooks) may not allow it—ignore.
            pass

    # ---------------- Public API ----------------

    def log(self, row: Dict[str, Any]) -> None:
        """
        Append a row of stats. Unknown keys at first call define the schema.
        For subsequent calls, missing keys get filled with None, and extra keys are added.
        """
        if not self._keys_frozen:
            # First rows define the initial key set, but we allow drift (add keys if appear)
            for k in row.keys():
                if k not in self._buf:
                    self._buf[k] = []
        else:
            # Ensure any new keys are added (rare, but robust)
            for k in row.keys():
                if k not in self._buf:
                    self._buf[k] = [None] * self._n_rows()

        # Fill missing keys with None so lengths match
        for k in self._buf.keys():
            self._buf[k].append(row.get(k, None))

        # Freeze after first write for a tiny speedup (skip dict key creation next times)
        if not self._keys_frozen:
            self._keys_frozen = True

    def save_now(self) -> Tuple[str, str | None]:
        """
        Force a save immediately. Returns (npz_path, mat_path).
        If also_save_h5=True, a third HDF5 file is written as well.
        """
        if self._saved:
            # Already saved once; allow re-save by design but no need usually
            pass

        npz_path = os.path.join(self.out_dir, f"{self.base_name}.npz")
        mat_path = os.path.join(self.out_dir, f"{self.base_name}.mat")
        h5_path = self.h5_path_override or os.path.join(
            self.out_dir, f"{self.base_name}.h5"
        )

        packed_npz: Dict[str, Any] = {}
        packed_mat: Dict[str, Any] = {}

        # 1) Fixed-shape: try to stack; if fails, fallback to object array
        fixed_keys = [k for k in self._buf.keys() if k not in self.ragged_fields]
        for k in fixed_keys:
            vals = self._buf[k]
            # Convert simple Python scalars to arrays for consistent stacking
            vals = [np.asarray(v) if not isinstance(v, np.ndarray) else v for v in vals]

            try:
                arr = np.stack(vals, axis=0)
            except Exception:
                # Not consistent shapes—fallback to object array
                arr = np.empty((len(vals),), dtype=object)
                for i, v in enumerate(vals):
                    arr[i] = v
            packed_npz[k] = arr
            packed_mat[k] = arr

        # 2) Ragged: pack as flat + offsets + lengths; also build MATLAB cell
        for k in self.ragged_fields:
            seq = self._buf.get(k, [])
            # Normalize each element to a 1-D np array (empty if None)
            norm: list[np.ndarray] = []
            for v in seq:
                if v is None:
                    norm.append(np.asarray([], dtype=float))
                else:
                    a = np.asarray(v)
                    if a.ndim == 0:
                        a = a.reshape(1)
                    elif a.ndim > 1:
                        a = a.reshape(-1)
                    norm.append(a)

            # Determine dtype from first non-empty
            dt = None
            for a in norm:
                if a.size > 0:
                    dt = a.dtype
                    break
            if dt is None:
                dt = np.float64

            lengths = np.fromiter(
                (a.size for a in norm), dtype=np.int64, count=len(norm)
            )
            offsets = np.empty_like(lengths)
            offsets[0] = 0
            if len(lengths) > 1:
                np.cumsum(lengths[:-1], out=offsets[1:])

            total = int(lengths.sum())
            data = np.empty((total,), dtype=dt)
            # Fill concat buffer
            pos = 0
            for a in norm:
                n = a.size
                if n:
                    data[pos : pos + n] = a
                pos += n

            # NPZ/MAT packed
            packed_npz[f"{k}_data"] = data
            packed_npz[f"{k}_offsets"] = offsets
            packed_npz[f"{k}_lengths"] = lengths

            packed_mat[f"{k}_data"] = data
            packed_mat[f"{k}_offsets"] = offsets
            packed_mat[f"{k}_lengths"] = lengths

            # For MATLAB convenience: also export a cell array (<name>_cell)
            # scipy.savemat converts Python lists-of-arrays to cell arrays.
            packed_mat[f"{k}_cell"] = [a for a in norm]

        # 3) Write NPZ
        np.savez_compressed(npz_path, **packed_npz)

        # 4) Write MAT (v7.2 via savemat)
        # Note: For very large cells, MATLAB load time can be high; the packed form is faster to parse.
        savemat(mat_path, self._mat_sanitize_dict(packed_mat))

        # 5) Optional: HDF5 (fixed fields as datasets; ragged as vlen datasets)
        if self.also_save_h5:
            with h5py.File(h5_path, "w") as f:
                # fixed fields
                for k in fixed_keys:
                    arr = packed_npz[k]
                    f.create_dataset(
                        k,
                        data=arr,
                        compression=self.h5_compression if arr.size > 0 else None,
                    )
                # ragged fields as vlen
                for k in self.ragged_fields:
                    # reconstruct per-row arrays from packed to write as vlen
                    data = packed_npz[f"{k}_data"]
                    offsets = packed_npz[f"{k}_offsets"]
                    lengths = packed_npz[f"{k}_lengths"]
                    dtype = data.dtype
                    vlen_dt = h5py.vlen_dtype(dtype)
                    ds = f.create_dataset(
                        k,
                        shape=(len(lengths),),
                        dtype=vlen_dt,
                        compression=self.h5_compression if data.size > 0 else None,
                    )
                    for i, (off, L) in enumerate(zip(offsets, lengths)):
                        ds[i] = data[off : off + L]

        self._saved = True
        return npz_path, mat_path

    # ---------------- Internals ----------------

    def _n_rows(self) -> int:
        # Length of the longest buffer
        if not self._buf:
            return 0
        return max((len(v) for v in self._buf.values()), default=0)

    def _atexit_hook(self):
        # try:
        if not self._saved and self._n_rows() > 0:
            self.save_now()

    # except Exception as e:
    #     # Avoid raising on interpreter shutdown
    #     print(f"[Logger] Save at exit failed: {e}")

    def _signal_hook(self, signum, frame):
        try:
            if not self._saved and self._n_rows() > 0:
                self.save_now()
        finally:
            # Re-raise default to terminate
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _is_object_array(self, arr: np.ndarray) -> bool:
        return isinstance(arr, np.ndarray) and arr.dtype == object

    def _to_matlab_safe_cell(self, seq):
        """
        Ensure a MATLAB-friendly cell array: a Python list whose elements are
        numeric ndarrays (1-D) or strings. Replace None with empty array or "".
        """
        out = []
        for el in seq:
            if el is None:
                out.append(np.asarray([], dtype=float))
            elif isinstance(el, str):
                out.append(el)
            elif isinstance(el, (list, tuple)):
                # Try convert to 1-D numeric
                a = np.asarray(el)
                if a.ndim > 1:
                    a = a.reshape(-1)
                out.append(a)
            elif isinstance(el, np.ndarray):
                a = el
                if a.dtype == object:
                    # Flatten and drop None
                    flat = []
                    for x in a.ravel():
                        if x is None:
                            continue
                        elif isinstance(x, (int, float, np.floating, np.integer)):
                            flat.append(float(x))
                        else:
                            # fallback: skip/ignore non-numeric
                            pass
                    out.append(np.asarray(flat, dtype=float))
                else:
                    if a.ndim > 1:
                        a = a.reshape(-1)
                    out.append(a)
            elif isinstance(el, (int, float, np.floating, np.integer)):
                out.append(np.asarray([float(el)], dtype=float))
            else:
                # Unknown type -> empty vector to avoid MATLAB error
                out.append(np.asarray([], dtype=float))
        return out

    def _mat_sanitize_dict(self, d: dict) -> dict:
        """
        Convert values to MATLAB-writable forms:
        - numeric ndarrays are OK
        - object ndarrays -> convert to cell/list with safe elements
        - Python lists -> safe cell via _to_matlab_safe_cell
        - None -> drop or convert to minimal safe form
        """
        safe = {}
        for k, v in d.items():
            if v is None:
                # Skip entirely rather than writing None
                continue
            if isinstance(v, np.ndarray):
                if self._is_object_array(v):
                    # convert to cell list
                    safe[k] = self._to_matlab_safe_cell([vv for vv in v])
                else:
                    safe[k] = v
            elif isinstance(v, list):
                safe[k] = self._to_matlab_safe_cell(v)
            elif isinstance(v, (int, float, np.floating, np.integer, str)):
                safe[k] = v  # scalars/strings are fine
            else:
                # Unknown object -> drop to avoid TypeError
                # (alternatively, turn into an empty array)
                # safe[k] = np.asarray([], dtype=float)
                continue
        return safe
