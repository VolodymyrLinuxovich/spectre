"""The Referee: the only process that ever sees labels.

A claim is PROMOTED only if it survives, in order:
  1. CRASH   every sandbox run must finish.
  2. NOISE   paired gain over the incumbent on val (MRR@20, seed-averaged) must have a
             bootstrap 95% CI above zero, beat 2x the incumbent's seed-to-seed std,
             and be positive on every seed.
  3. LEAK    retrain with shuffled train labels. Anything that still beats the
             incumbent's shuffled score by >= half the claimed gain did not come from
             learning (answer smuggled through inputs or candidate structure).
  4. PRIOR   zero the spectra at inference. If >= half the gain survives, the model is
             not reading the spectrum (precursor mass / formula / candidate priors).
  5. OVERFIT hidden test fold, 3 seeds. Proposers never see test numbers, only the verdict.
Promoted code becomes the new incumbent, so the system improves from its own results.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
CACHE = RUNS / "referee_cache.json"
INCUMBENT = RUNS / "incumbent.py"
EVENTS = RUNS / "events.jsonl"
SEEDS = (0, 1, 2)
TOPK = 20

app = modal.App("spectre-referee")
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "pandas==2.2.3", "pyarrow==18.1.0", "rdkit==2024.9.6", "numpy<2.2")


@app.function(image=image, volumes={"/data": modal.Volume.from_name("spectre-data")},
              timeout=600, cpu=4)
def truth_ranks(preds: dict, split: str) -> list[int]:
    """0-based rank of the true molecule in each prediction list, -1 if absent.
    Order follows the labelled parquet so vectors are paired across runs."""
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    lab = pd.read_parquet(f"/data/{split}.parquet", columns=["id", "smiles", "inchikey14"])
    memo: dict[str, str] = {}

    def ik14(s):
        if s not in memo:
            m = Chem.MolFromSmiles(s)
            memo[s] = Chem.MolToInchiKey(m)[:14] if m is not None else ""
        return memo[s]

    out = []
    for i, smi, key in zip(lab.id, lab.smiles, lab.inchikey14):
        r = -1
        for j, c in enumerate(preds.get(str(i), [])[:TOPK]):
            if c == smi or ik14(c) == key:
                r = j
                break
        out.append(r)
    return out


# ---------------------------------------------------------------- local side

def log(_who: str, _kind: str, **payload):
    if "agent" in payload:
        payload["proposer"] = payload.pop("agent")
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"ts": time.time(), "agent": _who, "kind": _kind, **payload}, default=str) + "\n")


def sha(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def metrics(ranks) -> dict:
    r = np.asarray(ranks)
    rr = np.where(r >= 0, 1.0 / (r + 1), 0.0)
    return {"mrr": float(rr.mean()), "hit1": float((r == 0).mean()),
            "hit5": float(((r >= 0) & (r < 5)).mean()), "hit20": float((r >= 0).mean())}


def rr_vec(ranks) -> np.ndarray:
    r = np.asarray(ranks)
    return np.where(r >= 0, 1.0 / (r + 1), 0.0)


def boot_ci(diff: np.ndarray, n=2000, seed=0) -> tuple[float, float]:
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


class Referee:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self._ranks = modal.Function.from_name("spectre-referee", "truth_ranks")
        from bench.sandbox import run_candidate
        self._run = run_candidate

    def _save(self):
        import threading
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        with self._lock:
            tmp = CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(self.cache)))
            tmp.replace(CACHE)

    def run(self, src: str, split: str, seed: int, mode: str = "plain") -> dict:
        """mode: plain | ablate | shuffle. Cached by (code hash, split, seed, mode)."""
        key = f"{sha(src)}:{split}:{seed}:{mode}"
        if key in self.cache:
            return self.cache[key]
        res = self._run(src, seed=seed, split=split, topk=TOPK,
                        ablate_spectra=mode == "ablate", shuffle_labels=mode == "shuffle")
        if not res.get("ok"):
            out = {"ok": False, "error": res.get("error"),
                   "stderr_tail": (res.get("stderr_tail") or "")[-1500:]}
        else:
            ranks = self._ranks.remote(res["preds"], split)
            out = {"ok": True, "ranks": ranks, "wall_s": res.get("wall_s"),
                   "train_s": res.get("train_s")}
        self.cache[key] = out
        self._save()
        return out

    def batch(self, jobs):
        with ThreadPoolExecutor(len(jobs)) as ex:
            return list(ex.map(lambda j: self.run(*j), jobs))

    def incumbent(self) -> str:
        if not INCUMBENT.exists():
            INCUMBENT.write_text((ROOT / "bench/baseline/model.py").read_text())
        return INCUMBENT.read_text()

    def judge(self, src: str, claim: str = "", agent: str = "?", guarded: bool = True,
              exp_id: str = "", tag: str = "") -> dict:
        """Full protocol. guarded=False reproduces the naive loop (seed 0 val only)."""
        t0 = time.time()
        inc = self.incumbent()
        v = {"agent": agent, "claim": claim, "code": sha(src), "incumbent": sha(inc),
             "exp_id": exp_id, "tag": tag}

        if not guarded:
            c, b = self.batch([(src, "val", 0), (inc, "val", 0)])
            if not (c["ok"] and b["ok"]):
                v.update(verdict="CRASH", error=c.get("error") or b.get("error"))
            else:
                g = metrics(c["ranks"])["mrr"] - metrics(b["ranks"])["mrr"]
                v.update(verdict="ACCEPTED" if g > 0 else "REJECTED", val_gain=g,
                         val=metrics(c["ranks"]), base_val=metrics(b["ranks"]))
            v["wall_s"] = time.time() - t0
            log(agent, "naive_verdict", **dict(v))
            return v

        jobs = [(s, "val", sd, "plain") for s in (src, inc) for sd in SEEDS]
        jobs += [(s, "val", 0, m) for s in (src, inc) for m in ("ablate", "shuffle")]
        res = dict(zip([(j[0] == src, j[2], j[3]) for j in jobs], self.batch(jobs)))
        bad = [r for r in res.values() if not r["ok"]]
        if bad:
            v.update(verdict="CRASH", error=bad[0].get("error"), stderr_tail=bad[0].get("stderr_tail"))
            return self._done(v, t0)

        C = np.stack([rr_vec(res[(True, s, "plain")]["ranks"]) for s in SEEDS])
        B = np.stack([rr_vec(res[(False, s, "plain")]["ranks"]) for s in SEEDS])
        per_seed = (C.mean(1) - B.mean(1)).tolist()
        gain = float(C.mean() - B.mean())
        lo, hi = boot_ci(C.mean(0) - B.mean(0))
        seed_std = float(B.mean(1).std(ddof=1))
        v.update(val=metrics(res[(True, 0, "plain")]["ranks"]),
                 base_val=metrics(res[(False, 0, "plain")]["ranks"]),
                 val_gain=gain, val_ci=[lo, hi], per_seed_gain=per_seed,
                 noise_band=2 * seed_std)
        if lo <= 0 or gain <= 2 * seed_std or min(per_seed) <= 0:
            why = []
            if lo <= 0: why.append(f"95% CI [{lo:+.4f}, {hi:+.4f}] includes 0")
            if gain <= 2 * seed_std: why.append(f"gain {gain:+.4f} inside noise band ±{2*seed_std:.4f}")
            if min(per_seed) <= 0: why.append(f"not positive on every seed {[round(x,4) for x in per_seed]}")
            v.update(verdict="NOISE", evidence="; ".join(why))
            return self._done(v, t0)

        sh = metrics(res[(True, 0, "shuffle")]["ranks"])["mrr"] - metrics(res[(False, 0, "shuffle")]["ranks"])["mrr"]
        ab = metrics(res[(True, 0, "ablate")]["ranks"])["mrr"] - metrics(res[(False, 0, "ablate")]["ranks"])["mrr"]
        v.update(shuffle_gain=sh, ablate_gain=ab)
        if sh >= 0.5 * gain:
            v.update(verdict="LEAK", evidence=f"{sh/gain:.0%} of the gain survives with train labels shuffled: "
                                              "it is not learned from data")
            return self._done(v, t0)
        if ab >= 0.5 * gain:
            v.update(verdict="PRIOR", evidence=f"{ab/gain:.0%} of the gain survives with spectra zeroed: "
                                               "the model is not reading the spectrum")
            return self._done(v, t0)

        tj = [(s, "test", sd, "plain") for s in (src, inc) for sd in SEEDS]
        tr = self.batch(tj)
        if not all(r["ok"] for r in tr):
            v.update(verdict="CRASH", error="test run failed")
            return self._done(v, t0)
        Ct = np.stack([rr_vec(r["ranks"]) for r in tr[:3]])
        Bt = np.stack([rr_vec(r["ranks"]) for r in tr[3:]])
        tg = float(Ct.mean() - Bt.mean())
        tlo, thi = boot_ci(Ct.mean(0) - Bt.mean(0))
        v.update(test_gain=tg, test_ci=[tlo, thi], test=metrics(tr[0]["ranks"]),
                 base_test=metrics(tr[3]["ranks"]))
        if tlo <= 0:
            v.update(verdict="OVERFIT", evidence=f"val gain {gain:+.4f} but hidden test CI [{tlo:+.4f}, {thi:+.4f}]")
        else:
            v.update(verdict="PROMOTED", evidence=f"val {gain:+.4f}, hidden test {tg:+.4f} CI [{tlo:+.4f}, {thi:+.4f}]")
            (RUNS / f"incumbent_{sha(inc)}.py").write_text(inc)
            INCUMBENT.write_text(src)
        return self._done(v, t0)

    def _done(self, v, t0):
        v["wall_s"] = time.time() - t0
        log("referee", "verdict", **dict(v))
        return v

    @staticmethod
    def public_view(v: dict) -> dict:
        """What proposers may see: no test numbers."""
        return {k: x for k, x in v.items() if not k.startswith("test") and not k.startswith("base_test")}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    ref = Referee()
    src = Path(sys.argv[1]).read_text()
    print(json.dumps(ref.judge(src, claim=sys.argv[2] if len(sys.argv) > 2 else "", agent="cli"), indent=2))
