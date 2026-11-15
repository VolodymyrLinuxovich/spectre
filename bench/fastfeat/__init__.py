"""Optional C++ spectrum featurizer.

    from bench.fastfeat import featurize_coo, available
    out = featurize_coo(df["mzs"], df["intensities"])   # None if the library isn't built
    if out is not None:
        rows, bins, vals = out

Build once with `bash bench/fastfeat/build.sh` (needs a C++17 compiler). Nothing else in the
repo depends on this: callers must fall back to their Python loop when it returns None.
Output matches SparseSpectra.__init__ in bench/baseline/model.py exactly
(see bench/fastfeat/test_featurize.py).
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB_NAME = "libspectre_feat.dylib" if sys.platform == "darwin" else "libspectre_feat.so"
_lib = None
_tried = False


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if os.environ.get("SPECTRE_NO_CPP"):
        return None
    path = _HERE / _LIB_NAME
    if not path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError:
        return None
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    lib.spectre_featurize.argtypes = [i64p, f64p, f64p, ctypes.c_int64, ctypes.c_double, ctypes.c_double,
                                      i64p, i64p, f64p, ctypes.c_int]
    lib.spectre_featurize.restype = ctypes.c_int64
    _lib = lib
    return lib


def available() -> bool:
    return _load() is not None


def _flatten(arrays) -> tuple[np.ndarray, np.ndarray]:
    lens = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    flat = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays]) if len(arrays) else np.zeros(0)
    return offsets, np.ascontiguousarray(flat, dtype=np.float64)


def featurize_coo(mzs, intensities, bin_width: float = 0.1, max_mz: float = 1000.0,
                  threads: int | None = None):
    """(rows int64, bins int64, vals float64) for every spectrum, or None if the C++ library is missing."""
    lib = _load()
    if lib is None:
        return None
    mzs, intensities = list(mzs), list(intensities)
    off_m, mz = _flatten(mzs)
    off_i, it = _flatten(intensities)
    if not np.array_equal(off_m, off_i):
        raise ValueError("mzs and intensities have different peak counts")
    cap = int(off_m[-1])
    rows = np.empty(cap, dtype=np.int64)
    bins = np.empty(cap, dtype=np.int64)
    vals = np.empty(cap, dtype=np.float64)
    n = lib.spectre_featurize(off_m, mz, it, len(mzs), float(bin_width), float(max_mz),
                              rows, bins, vals, int(threads or os.cpu_count() or 1))
    return rows[:n], bins[:n], vals[:n]
