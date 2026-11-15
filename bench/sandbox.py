"""Run an agent-written model.py inside a locked-down Modal Sandbox.

Isolation guarantees:
  * block_network=True: no internet, no exfiltration, no pip installs.
  * Labels are never mounted. The sandbox mounts Volume `spectre-inputs`
    (read only), which holds ONLY train.parquet, {val,test}_inputs.parquet and
    cands/{val,test}.parquet. `sync_inputs` copies exactly those files from
    `spectre-data`; val.parquet / test.parquet never leave `spectre-data`.
  * Hard 15 minute timeout. On crash, the stdout/stderr tails come back.

Usage (from the referee / orchestrator):
    from bench.sandbox import run_candidate
    res = run_candidate(open("bench/baseline/model.py").read(), seed=0, split="val")
    res["ok"], res["preds"]  # preds: {id: [cand_smiles ranked, top-k]}

Canary switches used by the Referee:
    ablate_spectra=True   zero every intensity at inference (train is untouched)
    shuffle_labels=True   permute smiles/inchikey across train rows before train()

One-time setup (after d1 fills `spectre-data`):
    modal run bench/sandbox.py::sync_inputs
Smoke test:
    .venv/bin/python bench/sandbox.py bench/baseline/model.py val 0
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Literal

import modal

APP_NAME = "spectre-sandbox"
DATA_VOL = "spectre-data"
INPUTS_VOL = "spectre-inputs"
TIMEOUT_S = 15 * 60
CPU_TIMEOUT_S = 25 * 60


def _default_gpu():
    """SPECTRE_GPU env: a Modal GPU type (default A10G) or 'none' for CPU only."""
    g = os.environ.get("SPECTRE_GPU", "A10G").strip()
    return None if g.lower() in ("none", "cpu", "") else g

_lock = threading.Lock()
_sb_app = None


def _get_app():
    """Look the App up once; run_candidate is called from many threads."""
    global _sb_app
    with _lock:
        if _sb_app is None:
            _sb_app = _get_app()
        return _sb_app
ALLOWED = ["train.parquet", "val_inputs.parquet", "test_inputs.parquet",
           "cands/val.parquet", "cands/test.parquet"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1", "numpy==2.1.3", "pandas==2.2.3", "pyarrow==18.1.0",
        "scikit-learn==1.5.2", "scipy==1.14.1", "rdkit==2024.9.6",
    )
)

app = modal.App(APP_NAME, image=image)

HARNESS = r'''
import json, os, random, sys, time, traceback
import numpy as np, pandas as pd

seed = int(sys.argv[1]); split = sys.argv[2]; topk = int(sys.argv[3])
ablate = sys.argv[4] == "1"; shuffle = sys.argv[5] == "1"

random.seed(seed); np.random.seed(seed)
try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
except Exception:
    pass

D = "/data"
train_df = pd.read_parquet(f"{D}/train.parquet")
inputs_df = pd.read_parquet(f"{D}/{split}_inputs.parquet")
cands_df = pd.read_parquet(f"{D}/cands/{split}.parquet")

if shuffle:  # label-shuffle canary: break spectrum <-> structure link
    perm = np.random.default_rng(10_000 + seed).permutation(len(train_df))
    for col in ("smiles", "inchikey14"):
        if col in train_df.columns:
            train_df[col] = train_df[col].to_numpy()[perm]
if ablate:   # spectrum-ablation canary: model sees no peak information
    inputs_df["intensities"] = [np.zeros(len(x), dtype=float).tolist() for x in inputs_df["intensities"]]

sys.path.insert(0, "/work")
import model

t0 = time.time()
state = model.train(train_df, seed)
t1 = time.time()
preds = model.rank(state, inputs_df, cands_df)
t2 = time.time()

if not isinstance(preds, dict):
    raise TypeError(f"rank() must return dict[id, list[str]], got {type(preds).__name__}")
valid = {str(i): set(c) for i, c in zip(cands_df["id"], cands_df["cand_smiles"])}
out, bad = {}, 0
for i in inputs_df["id"]:
    ranked = [s for s in list(preds.get(i, preds.get(str(i), []))) if s in valid.get(str(i), ())]
    seen, clean = set(), []
    for s in ranked:
        if s not in seen:
            seen.add(s); clean.append(s)
    if not clean:
        bad += 1
    out[str(i)] = clean[:topk]

os.makedirs("/work/out", exist_ok=True)
with open("/work/out/preds.json", "w") as f:
    json.dump({"preds": out, "n_missing": bad, "train_s": round(t1 - t0, 2), "rank_s": round(t2 - t1, 2)}, f)
print(f"HARNESS_OK n={len(out)} missing={bad} train_s={t1-t0:.1f} rank_s={t2-t1:.1f}")
'''


@app.function(volumes={"/src": modal.Volume.from_name(DATA_VOL),
                       "/dst": modal.Volume.from_name(INPUTS_VOL, create_if_missing=True)},
              timeout=30 * 60)
def sync_inputs():
    """Copy only the label-free files from spectre-data into spectre-inputs."""
    import os, shutil
    os.makedirs("/dst/cands", exist_ok=True)
    for rel in ALLOWED:
        shutil.copyfile(f"/src/{rel}", f"/dst/{rel}")
        print("copied", rel, os.path.getsize(f"/dst/{rel}"))
    for rel in ("val.parquet", "test.parquet"):
        if os.path.exists(f"/dst/{rel}"):
            os.remove(f"/dst/{rel}")
    modal.Volume.from_name(INPUTS_VOL).commit()


def _tail(s: str, n: int = 4000) -> str:
    return s[-n:] if s else ""


def run_candidate(
    model_py_source: str,
    seed: int,
    split: Literal["val", "test"],
    *,
    topk: int = 20,
    ablate_spectra: bool = False,
    shuffle_labels: bool = False,
    gpu: str | None = "env",
    timeout_s: int | None = None,
) -> dict:
    """Train + rank a candidate model.py in an isolated GPU sandbox.

    Returns {ok, preds, n_missing, train_s, rank_s, wall_s, returncode,
             stdout_tail, stderr_tail, error}.
    """
    assert split in ("val", "test"), split
    if gpu == "env":
        gpu = _default_gpu()
    if timeout_s is None:
        timeout_s = TIMEOUT_S if gpu else CPU_TIMEOUT_S
    res_kw = {} if gpu else {"cpu": 16.0, "memory": 32 * 1024}
    t0 = time.time()
    res = {"ok": False, "preds": None, "returncode": None, "stdout_tail": "",
           "stderr_tail": "", "error": None, "seed": seed, "split": split, "gpu": gpu or "none",
           "ablate_spectra": ablate_spectra, "shuffle_labels": shuffle_labels}
    sb = None
    try:
        sb_app = _get_app()
        sb = modal.Sandbox.create(
            "sleep", "infinity",
            app=sb_app,
            image=image,
            gpu=gpu,
            **res_kw,
            block_network=True,
            timeout=timeout_s + 120,
            volumes={"/data": modal.Volume.from_name(INPUTS_VOL).read_only()},
            tags={"project": "spectre", "split": split, "seed": str(seed)},
        )
        sb.filesystem.make_directory("/work")
        sb.filesystem.write_text(model_py_source, "/work/model.py")
        sb.filesystem.write_text(HARNESS, "/work/harness.py")
        p = sb.exec("python", "/work/harness.py", str(seed), split, str(topk),
                    "1" if ablate_spectra else "0", "1" if shuffle_labels else "0",
                    timeout=timeout_s, workdir="/work")
        out, err = p.stdout.read(), p.stderr.read()
        rc = p.wait()
        res.update(returncode=rc, stdout_tail=_tail(out), stderr_tail=_tail(err))
        if rc == 0 and "HARNESS_OK" in out:
            payload = json.loads(sb.filesystem.read_text("/work/out/preds.json"))
            res.update(ok=True, **payload)
        else:
            res["error"] = "timeout" if rc in (-1, 124, 137) else f"exit {rc}"
    except Exception as e:  # sandbox infra failure, timeout, etc.
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        if sb is not None:
            try:
                sb.terminate()
            except Exception:
                pass
    res["wall_s"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/baseline/model.py"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    r = run_candidate(open(path).read(), seed, split)
    preds = r.pop("preds") or {}
    print(json.dumps(r, indent=1))
    print("n_preds", len(preds), "example", next(iter(preds.items()), None))
