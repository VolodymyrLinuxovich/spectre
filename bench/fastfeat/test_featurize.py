"""The C++ featurizer must give exactly the Python result.

Run:  bash bench/fastfeat/build.sh && .venv/bin/python -m bench.fastfeat.test_featurize
      .venv/bin/python -m bench.fastfeat.test_featurize --bench      # adds a speed benchmark

`python_reference` is the loop from SparseSpectra.__init__ in bench/baseline/model.py, copied
without the torch parts. If torch is installed, the test also checks SparseSpectra itself.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from bench.fastfeat import available, featurize_coo

BIN = 0.1
MAX_MZ = 1000.0


def python_reference(mzs, intensities):
    rows, bins, vals = [], [], []
    for i, (mz, it) in enumerate(zip(mzs, intensities)):
        mz = np.asarray(mz, dtype=np.float64)
        it = np.asarray(it, dtype=np.float64)
        keep = (mz >= 0) & (mz < MAX_MZ)
        mz, it = mz[keep], it[keep]
        if len(mz) == 0:
            continue
        b = (mz / BIN).astype(np.int64)
        v = np.sqrt(np.maximum(it, 0) / max(it.max(), 1e-12))
        ub, inv = np.unique(b, return_inverse=True)
        agg = np.zeros(len(ub))
        np.maximum.at(agg, inv, v)
        rows.append(np.full(len(ub), i))
        bins.append(ub)
        vals.append(agg)
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    return cat(rows, np.int64), cat(bins, np.int64), cat(vals, np.float64)


def synthetic(n, seed=0):
    """Spectra shaped like MassSpecGym: lognormal peak counts, a few peaks outside 0..1000,
    duplicate bins, and occasional negative or zero intensities."""
    g = np.random.default_rng(seed)
    counts = np.clip(g.lognormal(3.0, 0.8, n).astype(int), 1, 600)
    mzs, its = [], []
    for c in counts:
        mz = g.uniform(30, 1010, c)
        dup = g.random(c) < .05                       # some peaks share one 0.1 Da bin
        if dup.sum() > 1:
            mz[dup] = np.round(mz[dup].mean(), 1) + g.uniform(0, .09, dup.sum())
        it = g.lognormal(6, 2, c)
        it[g.random(c) < .02] *= -1
        mzs.append(mz)
        its.append(it)
    return mzs, its


EDGE = [
    ([], []),                                         # empty spectrum
    ([1000.0, 1200.0, -1.0], [5.0, 6.0, 7.0]),          # nothing kept
    ([0.0, 999.9999, 1000.0], [1.0, 2.0, 3.0]),         # range edges
    ([0.3, 0.7, 2.3, 100.05, 100.0999], [1, 2, 3, 4, 5]),  # float bin boundaries (0.3 / 0.1 < 3)
    ([50.0, 50.01, 50.09, 60.0], [1.0, 9.0, 4.0, 0.0]),   # duplicate bin, max wins
    ([10.0, 20.0], [-5.0, -1.0]),                       # all negative intensities
    ([10.0, 20.0], [0.0, 0.0]),                         # all zero
    ([123.456], [1e-300]),                              # tiny intensity
]


def check(mzs, its, label, threads=None):
    ref = python_reference(mzs, its)
    got = featurize_coo(mzs, its, BIN, MAX_MZ, threads=threads)
    assert got is not None, "C++ library not built"
    for name, a, b in zip(("rows", "bins", "vals"), ref, got):
        assert a.dtype == b.dtype, f"{label}: {name} dtype {a.dtype} vs {b.dtype}"
        assert a.shape == b.shape, f"{label}: {name} shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), f"{label}: {name} differ at {np.flatnonzero(a != b)[:5]}"
    print(f"ok  {label}: {len(ref[0])} entries identical")


def check_against_model():
    try:
        import pandas as pd
        import torch
        from bench.baseline.model import SparseSpectra
    except Exception as e:  # torch not installed locally
        print(f"skip SparseSpectra check ({type(e).__name__}: {e})")
        return
    mzs, its = synthetic(2000, seed=3)
    df = pd.DataFrame({"mzs": mzs, "intensities": its, "precursor_mz": 500.0, "adduct": "[M+H]+"})
    s = SparseSpectra(df, torch.device("cpu"))
    r, b, v = featurize_coo(mzs, its, BIN, MAX_MZ)
    assert np.array_equal(s.row.numpy(), r) and np.array_equal(s.bin.numpy(), b)
    assert np.array_equal(s.val.numpy(), v.astype(np.float32))
    print("ok  SparseSpectra in bench/baseline/model.py matches")


def bench(n=194_119):
    mzs, its = synthetic(n, seed=1)
    peaks = sum(map(len, mzs))
    t0 = time.perf_counter(); python_reference(mzs, its); tp = time.perf_counter() - t0
    res = {}
    for th in (1, None):
        t0 = time.perf_counter(); featurize_coo(mzs, its, BIN, MAX_MZ, threads=th); res[th] = time.perf_counter() - t0
    print(f"\n{n:,} synthetic spectra, {peaks:,} peaks")
    print(f"  Python loop         {tp:7.2f} s")
    print(f"  C++, 1 thread       {res[1]:7.2f} s   {tp / res[1]:5.1f}x")
    print(f"  C++, all threads    {res[None]:7.2f} s   {tp / res[None]:5.1f}x")


if __name__ == "__main__":
    if not available():
        sys.exit("C++ library not built. Run: bash bench/fastfeat/build.sh")
    for k, (m, i) in enumerate(EDGE):
        check([m], [i], f"edge case {k}")
    check([m for m, _ in EDGE], [i for _, i in EDGE], "all edge cases in one batch")
    for seed in range(5):
        mzs, its = synthetic(3000, seed=seed)
        check(mzs, its, f"random batch seed {seed}", threads=1 + seed)
    check_against_model()
    if "--bench" in sys.argv:
        bench()
