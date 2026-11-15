"""Data prep for Spectre. Run: modal run bench/prep.py

Writes to Modal Volume `spectre-data` (see PLAN.md Contract):
  /train.parquet, /val.parquet, /test.parquet
  /val_inputs.parquet, /test_inputs.parquet       (no smiles / inchikey14)
  /cands/val.parquet, /cands/test.parquet          id, cand_smiles, cand_fp
cand_fp: one bytes value per row = n_cands * 256 bytes, Morgan r2 2048 bits,
np.packbits (bitorder big). Unpack: np.unpackbits(np.frombuffer(b, np.uint8)).reshape(-1, 2048)
Unparseable candidate smiles get an all-zero fingerprint.
"""
import modal

app = modal.App("spectre-data")
vol = modal.Volume.from_name("spectre-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.1.3", "pandas==2.2.3", "pyarrow==18.1.0", "rdkit==2024.9.6", "huggingface_hub==0.26.5"
)
REPO = "roman-bushuiev/MassSpecGym"
TSV = "data/MassSpecGym.tsv"
CANDS = "data/molecules/MassSpecGym_retrieval_candidates_formula.json"
N_SUB = 4000
SEED = 0
FP_BYTES = 256
CHUNK = 20000


@app.function(image=image, cpu=1, memory=2048, timeout=1800)
def fp_chunk(smiles: list[str]) -> tuple[bytes, int]:
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = np.zeros((len(smiles), FP_BYTES), dtype=np.uint8)
    bad = 0
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad += 1
            continue
        out[i] = np.packbits(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
    return out.tobytes(), bad


@app.function(image=image, volumes={"/data": vol}, cpu=4, memory=32768, timeout=3600)
def prep():
    import json, os, zlib
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    os.makedirs("/data/raw", exist_ok=True)
    tsv = hf_hub_download(REPO, TSV, repo_type="dataset", local_dir="/data/raw")
    cj = hf_hub_download(REPO, CANDS, repo_type="dataset", local_dir="/data/raw")
    vol.commit()

    df = pd.read_csv(tsv, sep="\t")
    print("columns:", list(df.columns))
    print("fold counts:", df["fold"].value_counts().to_dict())
    print("adducts:", df["adduct"].value_counts().to_dict())
    df = pd.DataFrame({
        "id": df["identifier"],
        "mzs": df["mzs"].map(lambda s: [float(x) for x in s.split(",")]),
        "intensities": df["intensities"].map(lambda s: [float(x) for x in s.split(",")]),
        "precursor_mz": df["precursor_mz"].astype(float),
        "adduct": df["adduct"],
        "formula": df["formula"],
        "smiles": df["smiles"],
        "inchikey14": df["inchikey"].str[:14],
        "fold": df["fold"],
    })
    cols = ["id", "mzs", "intensities", "precursor_mz", "adduct", "formula", "smiles", "inchikey14"]
    train = df[df.fold == "train"][cols].reset_index(drop=True)
    subs = {}
    for f in ["val", "test"]:
        d = df[df.fold == f]
        subs[f] = d.sample(n=min(N_SUB, len(d)), random_state=SEED).sort_values("id")[cols].reset_index(drop=True)

    with open(cj) as fh:
        cand_map = json.load(fh)
    print("candidate lists:", len(cand_map))

    uniq = sorted({s for d in subs.values() for q in d.smiles for s in cand_map.get(q, [])})
    print("unique candidate smiles:", len(uniq))
    chunks = [uniq[i:i + CHUNK] for i in range(0, len(uniq), CHUNK)]
    fps, bad = [], 0
    for b, nb in fp_chunk.map(chunks):
        fps.append(np.frombuffer(b, dtype=np.uint8).reshape(-1, FP_BYTES))
        bad += nb
    fps = np.concatenate(fps)
    idx = {s: i for i, s in enumerate(uniq)}
    print("unparseable candidates:", bad)

    train.to_parquet("/data/train.parquet")
    os.makedirs("/data/cands", exist_ok=True)
    report = {"train": len(train), "unparseable_cands": bad, "unique_cands": len(uniq)}
    for f, d in subs.items():
        d.to_parquet(f"/data/{f}.parquet")
        d.drop(columns=["smiles", "inchikey14"]).to_parquet(f"/data/{f}_inputs.parquet")
        raw = [cand_map.get(q, []) for q in d.smiles]
        # true-candidate position in the ORIGINAL MassSpecGym order (leak check)
        pos = np.array([c.index(q) if q in c else -1 for q, c in zip(d.smiles, raw)])
        # shuffle each list with a per-id deterministic RNG so order carries no label signal
        cl = []
        for i, c in zip(d["id"], raw):
            perm = np.random.default_rng([SEED, zlib.crc32(i.encode())]).permutation(len(c))
            cl.append([c[j] for j in perm])
        cdf = pd.DataFrame({
            "id": d["id"],
            "cand_smiles": cl,
            "cand_fp": [fps[[idx[s] for s in c]].tobytes() for c in cl],
        })
        cdf.to_parquet(f"/data/cands/{f}.parquet")
        n = np.array([len(c) for c in cl])
        found = pos[pos >= 0]
        report[f] = {"rows": len(d), "cands_mean": float(n.mean()), "cands_min": int(n.min()),
                     "cands_max": int(n.max()), "no_cands": int((n == 0).sum()),
                     "true_in_cands": float((pos >= 0).mean()),
                     "orig_true_pos_frac_idx0": float((pos == 0).mean()),
                     "orig_true_pos_hist": {"0": int((found == 0).sum()), "1-4": int(((found >= 1) & (found < 5)).sum()),
                                            "5-19": int(((found >= 5) & (found < 20)).sum()), "20+": int((found >= 20).sum()),
                                            "missing": int((pos < 0).sum())},
                     "shuffled_true_pos_frac_idx0": float(np.mean([c[0] == q if c else False for q, c in zip(d.smiles, cl)])),
                     "random_hit1": float(np.mean([1 / x for x in n if x > 0]))}
    vol.commit()
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main():
    print(prep.remote())
