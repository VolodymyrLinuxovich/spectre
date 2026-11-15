"""Make three fake cases with known answers, to test verify.cpp without Modal runs.

The rank vectors are invented, but the claimed numbers are produced by bench/referee.py's
own metrics / rr_vec / boot_ci, following Referee.judge line for line. So if verify.cpp
agrees on these, it agrees with the real Referee math.

  real_gain   candidate fixes a fixed set of spectra on every seed    -> PROMOTED
  noise_gain  candidate is the incumbent retrained with other seeds   -> NOISE
  prior_gain  the gain survives with spectra zeroed                   -> PRIOR

Usage:  .venv/bin/python verify/make_synthetic.py   (writes verify/cases_synthetic/)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))

from bench.referee import SEEDS, boot_ci, metrics, rr_vec  # noqa: E402
from export import write_case  # noqa: E402

OUT = ROOT / "verify" / "cases_synthetic"
# rr_vec computes 1/(r+1) for every rank before np.where masks out r = -1. Harmless.
warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)
N_SPECTRA = 1500


def model_ranks(rng, base, seed_noise=0.25, fixed=None, fix_rate=0.0):
    """One model's ranks: the spectrum's base difficulty, a fresh draw for a random
    `seed_noise` share of spectra (retraining noise), and rank 0 for a `fix_rate` share
    of the `fixed` spectra (a real improvement)."""
    r = base.copy()
    redraw = rng.random(len(r)) < seed_noise
    r[redraw] = random_ranks(rng, redraw.sum())
    if fixed is not None:
        r[fixed & (rng.random(len(r)) < fix_rate)] = 0
    return r


def random_ranks(rng, n, miss=0.45):
    r = np.minimum(rng.geometric(0.25, n) - 1, 19)
    r[rng.random(n) < miss] = -1
    return r


def offline_judge(runs: dict) -> dict:
    """Referee.judge's guarded branch, minus Modal: same functions, same order."""
    C = np.stack([rr_vec(runs[("cand", "val", s, "plain")]) for s in SEEDS])
    B = np.stack([rr_vec(runs[("inc", "val", s, "plain")]) for s in SEEDS])
    per_seed = (C.mean(1) - B.mean(1)).tolist()
    gain = float(C.mean() - B.mean())
    lo, hi = boot_ci(C.mean(0) - B.mean(0))
    seed_std = float(B.mean(1).std(ddof=1))
    v = dict(val_gain=gain, val_ci=[lo, hi], per_seed_gain=per_seed, noise_band=2 * seed_std)
    if lo <= 0 or gain <= 2 * seed_std or min(per_seed) <= 0:
        return {**v, "verdict": "NOISE"}

    mrr = lambda who, mode: metrics(runs[(who, "val", 0, mode)])["mrr"]  # noqa: E731
    sh = mrr("cand", "shuffle") - mrr("inc", "shuffle")
    ab = mrr("cand", "ablate") - mrr("inc", "ablate")
    v.update(shuffle_gain=sh, ablate_gain=ab)
    if sh >= 0.5 * gain:
        return {**v, "verdict": "LEAK"}
    if ab >= 0.5 * gain:
        return {**v, "verdict": "PRIOR"}

    Ct = np.stack([rr_vec(runs[("cand", "test", s, "plain")]) for s in SEEDS])
    Bt = np.stack([rr_vec(runs[("inc", "test", s, "plain")]) for s in SEEDS])
    tlo, thi = boot_ci(Ct.mean(0) - Bt.mean(0))
    v.update(test_gain=float(Ct.mean() - Bt.mean()), test_ci=[tlo, thi])
    return {**v, "verdict": "OVERFIT" if tlo <= 0 else "PROMOTED"}


def make_runs(rng, val_fix_rate, ablate_fix_rate):
    """All 16 runs a full guarded verdict needs. Shuffled-label canaries are pure guesses
    for both models; ablated canaries keep the candidate's fix if `ablate_fix_rate` > 0."""
    runs = {}
    for split in ("val", "test"):
        base = random_ranks(rng, N_SPECTRA)
        fixed = rng.random(N_SPECTRA) < 0.3
        for s in SEEDS:
            runs[("cand", split, s, "plain")] = model_ranks(rng, base, fixed=fixed, fix_rate=val_fix_rate)
            runs[("inc", split, s, "plain")] = model_ranks(rng, base)
    prior = random_ranks(rng, N_SPECTRA, miss=0.7)
    for who in ("cand", "inc"):
        runs[(who, "val", 0, "shuffle")] = random_ranks(rng, N_SPECTRA, miss=0.9)
        rate = ablate_fix_rate if who == "cand" else 0.0
        runs[(who, "val", 0, "ablate")] = model_ranks(rng, prior, fixed=fixed, fix_rate=rate)
    return runs


def drop_unused(runs, verdict):
    """Keep only the runs the Referee would actually have: NOISE stops before canaries,
    LEAK and PRIOR stop before the test fold."""
    def used(key):
        who, split, seed, mode = key
        if verdict == "NOISE":
            return split == "val" and mode == "plain"
        if verdict in ("LEAK", "PRIOR"):
            return split == "val"
        return True
    return {k: v for k, v in runs.items() if used(k)}


def main():
    OUT.mkdir(exist_ok=True)
    specs = {"real_gain": (0.5, 0.0, "PROMOTED"),
             "noise_gain": (0.0, 0.0, "NOISE"),
             "prior_gain": (0.5, 0.5, "PRIOR")}
    names = []
    for i, (name, (fix, ablate_fix, expected)) in enumerate(specs.items()):
        rng = np.random.default_rng(100 + i)
        runs = make_runs(rng, fix, ablate_fix)
        verdict = offline_judge(runs)
        print(f"{name:11s} referee says {verdict['verdict']:8s} (intended {expected})")
        event = {"code": name, "incumbent": "synthetic_inc", **verdict}
        write_case(OUT / f"{name}.txt", name, event, drop_unused(runs, verdict["verdict"]))
        names.append(name)
    (OUT / "index.txt").write_text("".join(f"{n}.txt\n" for n in names))


if __name__ == "__main__":
    main()
