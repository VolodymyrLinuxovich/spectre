"""Cached Morgan fingerprints for training molecules.

Every training run calls smiles_to_fp on every unique train SMILES (bench/baseline/model.py).
That is the slow part of featurizing. This module precomputes them once.

Build (Modal, writes only to spectre-data:/fp/, which sandboxes do not mount):
    modal run bench/fastfeat/fpcache.py::build
Verify the cache equals smiles_to_fp, and time both:
    modal run bench/fastfeat/fpcache.py::verify

Use (after integration):
    from bench.fastfeat.fpcache import lookup
    fp = lookup(list_of_smiles)        # uint8 (n, 2048), same as smiles_to_fp(list_of_smiles)

Only TRAIN molecules are cached, so nothing about val or test can leak. Lookups go by SMILES
string, so the label-shuffle canary (which permutes SMILES across rows) still gets the right
fingerprint for each molecule. SMILES missing from the cache are computed with RDKit on the spot.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

FP_BITS = 2048
CACHE_REL = "fp/train_morgan_r2_2048.parquet"
_DEFAULT_PATHS = [Path("/data") / CACHE_REL, Path("/src") / CACHE_REL]  # sandbox mount, Modal build mount

_cache: dict | None = None


def smiles_to_fp(smiles):
    """Copy of bench/baseline/model.py smiles_to_fp: Morgan r2 2048 as uint8 (n, 2048), unparseable -> zeros."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    out = np.zeros((len(smiles), FP_BITS), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out[i] = gen.GetFingerprintAsNumPy(m)
    return out


def _load(path: str | Path | None):
    global _cache
    if _cache is not None:
        return _cache
    import pandas as pd

    cands = [Path(path)] if path else [Path(p) for p in os.environ.get("SPECTRE_FP_CACHE", "").split(":") if p] + _DEFAULT_PATHS
    for p in cands:
        if p.exists():
            df = pd.read_parquet(p)
            packed = np.frombuffer(b"".join(df["fp"].tolist()), dtype=np.uint8).reshape(len(df), FP_BITS // 8)
            _cache = {"index": {s: k for k, s in enumerate(df["smiles"])}, "packed": packed, "path": str(p)}
            return _cache
    _cache = {}
    return _cache


def lookup(smiles, path: str | Path | None = None) -> np.ndarray:
    """Fingerprints for `smiles`, from the cache where possible, RDKit for the rest."""
    smiles = list(smiles)
    c = _load(path)
    if not c:
        return smiles_to_fp(smiles)
    idx = np.fromiter((c["index"].get(s, -1) for s in smiles), dtype=np.int64, count=len(smiles))
    out = np.empty((len(smiles), FP_BITS), dtype=np.uint8)
    hit = idx >= 0
    if hit.any():
        out[hit] = np.unpackbits(c["packed"][idx[hit]], axis=1)
    if (~hit).any():
        miss = [s for s, h in zip(smiles, hit) if not h]
        out[~hit] = smiles_to_fp(miss)
    return out


# ---------------------------------------------------------------- Modal: build and verify
try:
    import modal

    app = modal.App("spectre-fpcache")
    image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
        "numpy==2.1.3", "pandas==2.2.3", "pyarrow==18.1.0", "rdkit==2024.9.6")  # same versions as the sandbox
    vol = modal.Volume.from_name("spectre-data")

    @app.function(image=image, volumes={"/src": vol}, cpu=8, timeout=1800)
    def build():
        import pandas as pd

        t0 = time.time()
        uniq = sorted(set(pd.read_parquet("/src/train.parquet", columns=["smiles"])["smiles"]))
        fp = smiles_to_fp(uniq)
        packed = np.packbits(fp, axis=1)
        os.makedirs("/src/fp", exist_ok=True)
        pd.DataFrame({"smiles": uniq, "fp": [r.tobytes() for r in packed]}).to_parquet(f"/src/{CACHE_REL}")
        vol.commit()
        print(f"cached {len(uniq):,} train molecules in {time.time() - t0:.1f}s -> spectre-data:/{CACHE_REL}")

    @app.function(image=image, volumes={"/src": vol}, cpu=8, timeout=1800)
    def verify():
        import pandas as pd

        train = pd.read_parquet("/src/train.parquet", columns=["smiles"])["smiles"].to_numpy()
        uniq = list(np.unique(train))
        t0 = time.time(); ref = smiles_to_fp(uniq); t_rdkit = time.time() - t0
        global _cache
        _cache = None
        t0 = time.time(); got = lookup(uniq, f"/src/{CACHE_REL}"); t_cache = time.time() - t0
        assert got.shape == ref.shape and np.array_equal(got, ref), "cache differs from RDKit"
        # shuffled order and a molecule outside the cache must also match
        rng = np.random.default_rng(0)
        sample = list(rng.permutation(train)[:5000]) + ["CCO", "not a smiles"]
        assert np.array_equal(lookup(sample), smiles_to_fp(sample)), "shuffled lookup differs"
        print(f"ok: {len(uniq):,} molecules identical, plus shuffled sample and 2 uncached SMILES")
        print(f"RDKit {t_rdkit:.1f}s   cache (first load + lookup) {t_cache:.2f}s   {t_rdkit / t_cache:.0f}x")
except ImportError:
    pass
