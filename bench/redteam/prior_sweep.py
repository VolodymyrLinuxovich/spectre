"""Sweep non-spectral priors on top of the baseline, emulating the Referee's plain/ablate/shuffle modes.
Run: modal run bench/redteam/prior_sweep.py   (A10G, ~4 min)

Trains the baseline once per mode (plain, shuffled labels) on seed 0, then rescored candidates with
score = cos(pred, cand) + w * prior(cand), prior z-scored within each candidate list.
Round 1 (global / same-formula similarity to train molecules) lowered MRR at every weight; see runs/prior_sweep_round1.log. Reports val MRR@20 for plain, spectra-zeroed and label-shuffled
inputs, and the gain over the baseline (w = 0) in each mode. Matching is by exact SMILES.
"""
import json

import modal

app = modal.App("spectre-prior-sweep")
vol = modal.Volume.from_name("spectre-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.1.3", "pandas==2.2.3", "pyarrow==18.1.0", "rdkit==2024.9.6", "torch==2.5.1")
    .add_local_python_source("bench")
)
WEIGHTS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]  # priors are z-scored per spectrum; cosines are ~0.1-0.6


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, memory=32768, timeout=3600)
def sweep(seed: int = 0):
    import numpy as np
    import pandas as pd
    import torch
    from bench.baseline import model as M

    train_df = pd.read_parquet("/data/train.parquet")
    inputs = pd.read_parquet("/data/val_inputs.parquet")
    cands = pd.read_parquet("/data/cands/val.parquet")
    lab = pd.read_parquet("/data/val.parquet", columns=["id", "smiles"])
    truth = dict(zip(lab.id, lab.smiles))
    abl = inputs.copy()
    abl["intensities"] = [np.zeros(len(x)).tolist() for x in abl["intensities"]]
    sh_df = train_df.copy()
    perm = np.random.default_rng(10_000 + seed).permutation(len(sh_df))  # same as sandbox harness
    for col in ("smiles", "inchikey14"):
        sh_df[col] = sh_df[col].to_numpy()[perm]

    def predict(state, df):
        dev = M._device()
        net = M._net().to(dev); net.load_state_dict(state["net"]); net.eval()
        data = M.SparseSpectra(df, dev)
        pred = torch.zeros(len(df), M.FP_BITS, device=dev)
        with torch.no_grad():
            for ids, x in data.batches(torch.arange(len(df), device=dev), 1024):
                pred[ids] = torch.sigmoid(net(x))
        return torch.nn.functional.normalize(pred, dim=1).cpu().numpy()

    import re
    from collections import Counter
    ELS = ("N", "O", "S", "P", "F", "Cl", "Br", "I", "Si")

    def elset(f):
        return tuple(e for e in ELS if re.search(e + r"(?![a-z])", f))

    # candidate pool (unlabeled): every val candidate, deduplicated
    pool_s, pool_fp = {}, []
    for smi, fpb in zip(cands.cand_smiles, cands.cand_fp):
        c = M.unpack_fp(fpb)
        for s_, row in zip(smi, c):
            if s_ not in pool_s:
                pool_s[s_] = len(pool_fp); pool_fp.append(row)
    pool_fp = np.asarray(pool_fp, dtype=np.float32)
    p_pool = (pool_fp.sum(0) + 1) / (len(pool_fp) + 2)
    freq = Counter(s_ for smi in cands.cand_smiles for s_ in smi)

    def nb_w(fp_pos):
        p = (fp_pos.sum(0) + 1) / (len(fp_pos) + 2)
        w1 = np.log(p / p_pool); w0 = np.log((1 - p) / (1 - p_pool))
        return w1 - w0, w0.sum()

    def priors_from(tdf):
        """Priors from train (formula, adduct, smiles) + the unlabeled candidate pool. No spectra."""
        t = tdf[["formula", "adduct", "smiles"]].drop_duplicates()
        us = t.smiles.unique()
        fp = M.smiles_to_fp(list(us)).astype(np.float32)
        row = {s_: i for i, s_ in enumerate(us)}
        glob = nb_w(fp)
        cond = {}
        t = t.assign(es=t.formula.map(elset))
        for key, grp in t.groupby(["es", "adduct"]):
            if len(grp) >= 50:
                cond[key] = nb_w(fp[[row[s_] for s_ in grp.smiles.unique()]])
        return glob, cond

    def scores(pred, pri, df):
        glob, cond = pri
        fmap = dict(zip(df.id, zip(df.formula, df.adduct)))
        out = []
        for k, (cid, smi, fpb) in enumerate(zip(cands.id, cands.cand_smiles, cands.cand_fp)):
            c = M.unpack_fp(fpb).astype(np.float32)
            cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
            f, a = fmap[cid]
            wg, _ = glob
            wc, _ = cond.get((elset(f), a), glob)
            ng = c @ wg; nc = c @ wc
            ng = (ng - ng.mean()) / (ng.std() + 1e-8); nc = (nc - nc.mean()) / (nc.std() + 1e-8)
            fq = np.log([freq[s_] for s_ in smi]); fq = (fq - fq.mean()) / (fq.std() + 1e-8)
            out.append((list(smi), cn @ pred[k], {"nb_global": ng, "nb_cond": nc, "cand_freq": fq}))
        return out

    def mrr(sc, kind, w):
        rr = []
        for (smi, base, pr), cid in zip(sc, cands.id):
            p = pr[kind] if kind in pr else 0
            o = np.argsort(-(base + w * p), kind="stable")[:20]
            t = truth[cid]
            r = next((j for j, i in enumerate(o) if smi[i] == t), -1)
            rr.append(1 / (r + 1) if r >= 0 else 0.0)
        return float(np.mean(rr))

    assert list(inputs.id) == list(cands.id)
    st = M.train(train_df, seed)
    st_sh = M.train(sh_df, seed)
    pri, pri_sh = priors_from(train_df), priors_from(sh_df)
    sc = {"plain": scores(predict(st, inputs), pri, inputs),
          "ablate": scores(predict(st, abl), pri, abl),
          "shuffle": scores(predict(st_sh, inputs), pri_sh, inputs)}
    base = {m: mrr(sc[m], "none", 0) for m in sc}
    rows = [{"kind": "baseline", "w": 0, **{m: round(base[m], 4) for m in sc}}]
    for kind in ("nb_global", "nb_cond", "cand_freq"):
        for w in WEIGHTS:
            r = {"kind": kind, "w": w}
            for m in sc:
                v = mrr(sc[m], kind, w)
                r[m] = round(v, 4); r[f"{m}_gain"] = round(v - base[m], 4)
            r["ablate_frac"] = round(r["ablate_gain"] / r["plain_gain"], 2) if r["plain_gain"] > 0 else None
            r["shuffle_frac"] = round(r["shuffle_gain"] / r["plain_gain"], 2) if r["plain_gain"] > 0 else None
            rows.append(r)
            print(json.dumps(r), flush=True)
    return rows


@app.local_entrypoint()
def main(seed: int = 0):
    for r in sweep.remote(seed):
        print("ROW", json.dumps(r))
