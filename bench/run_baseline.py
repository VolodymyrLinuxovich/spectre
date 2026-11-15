"""Train + score the baseline on Modal GPU. Run: modal run bench/run_baseline.py [--seed 0]"""
import os

import modal

app = modal.App("spectre-baseline")
vol = modal.Volume.from_name("spectre-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.1.3", "pandas==2.2.3", "pyarrow==18.1.0", "rdkit==2024.9.6", "torch==2.5.1")
    .add_local_python_source("bench")
)


def hit_at_k(ranked: dict, labels: dict, ks=(1, 5, 20)):
    res = {}
    for k in ks:
        res[f"hit@{k}"] = sum(labels[i] in ranked.get(i, [])[:k] for i in labels) / len(labels)
    return res


def _run(seed: int, module: str = "bench.baseline.model", overrides: str = "{}"):
    import time
    import numpy as np
    import pandas as pd
    import importlib, json
    model = importlib.import_module(module)
    for k, v in json.loads(overrides).items():  # module constants, e.g. {"USE_NL": false}
        setattr(model, k, v)
    if hasattr(model, "N_NL"):
        model.N_NL = int(model.NL_MAX / model.BIN) if model.USE_NL else 0

    t0 = time.time()
    train_df = pd.read_parquet("/data/train.parquet")
    inputs = pd.read_parquet("/data/val_inputs.parquet")
    cands = pd.read_parquet("/data/cands/val.parquet")
    vdf = pd.read_parquet("/data/val.parquet", columns=["id", "smiles"])
    labels = dict(zip(vdf["id"], vdf["smiles"]))
    t1 = time.time()
    state = model.train(train_df, seed)
    t_train = time.time() - t1
    ranked = model.rank(state, inputs, cands)
    metrics = hit_at_k(ranked, labels)
    rng = np.random.default_rng(seed)
    rand = {c: list(rng.permutation(list(s))) for c, s in zip(cands["id"], cands["cand_smiles"])}
    rand_m = hit_at_k(rand, labels)
    expected_rand1 = float(np.mean([1 / len(s) if len(s) else 0 for s in cands["cand_smiles"]]))
    out = {"module": module, "overrides": overrides, "seed": seed, "val": metrics, "random": rand_m, "random_hit1_expected": expected_rand1,
           "train_s": round(t_train), "total_s": round(time.time() - t0)}
    print(out)
    return out


# SPECTRE_GPU=none runs on a 16-core CPU container (Modal rejects the whole app if any function asks for a GPU
# without a payment method on the account, so the GPU must not even be declared then).
GPU = os.environ.get("SPECTRE_GPU", "A10G")
GPU = None if GPU.lower() in ("", "none", "cpu") else GPU


@app.function(image=image, gpu=GPU, cpu=None if GPU else 16, volumes={"/data": vol}, memory=32768, timeout=3600)
def run(seed: int = 0, module: str = "bench.baseline.model", overrides: str = "{}"):
    import torch
    if not torch.cuda.is_available():  # env var is not set inside the container
        torch.set_num_threads(16)
    return _run(seed, module, overrides)


@app.local_entrypoint()
def main(seed: int = 0, module: str = "bench.baseline.model", variants: str = ""):
    """--variants: JSON list of override dicts; each runs in its own container in parallel."""
    import json
    if variants:
        vs = [json.dumps(v) for v in json.loads(variants)]
        for r in run.starmap([(seed, module, v) for v in vs]):
            print("RESULT", json.dumps(r))
    else:
        print("RESULT", json.dumps(run.remote(seed, module)))
