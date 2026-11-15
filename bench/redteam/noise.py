"""Baseline: binned spectrum + precursor features -> MLP -> Morgan r2 2048 fingerprint (BCE).
Candidates ranked by cosine(predicted fp probs, candidate fp).

Interface (PLAN.md Contract):
  train(train_df, seed) -> state
  rank(state, inputs_df, cands_df) -> {id: [cand_smiles ranked best first]}
"""
import time

import numpy as np
import torch
import torch.nn as nn

BIN = 0.1
MAX_MZ = 1000.0
N_BINS = int(MAX_MZ / BIN)  # 10000
ADDUCTS = ["[M+H]+", "[M+Na]+"]
N_EXTRA = 1 + len(ADDUCTS) + 1  # precursor_mz/1000, adduct one-hot (+other)
N_IN = N_BINS + N_EXTRA
FP_BITS = 2048
HIDDEN = 1056
EPOCHS = 25
BATCH = 512
LR = 1e-3
DROPOUT = 0.3


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smiles_to_fp(smiles):
    """Morgan r2 2048 bits as uint8 (n, 2048). Unparseable -> zeros."""
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


def unpack_fp(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).reshape(-1, FP_BITS)


class SparseSpectra:
    """Flat COO storage of binned spectra; dense batches are built on the device."""

    def __init__(self, df, device):
        rows, bins, vals = [], [], []
        for i, (mz, it) in enumerate(zip(df["mzs"], df["intensities"])):
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
        n = len(df)
        self.n = n
        self.row = torch.tensor(np.concatenate(rows) if rows else np.zeros(0, int), device=device)
        self.bin = torch.tensor(np.concatenate(bins) if bins else np.zeros(0, int), device=device)
        self.val = torch.tensor(np.concatenate(vals) if vals else np.zeros(0), dtype=torch.float32, device=device)
        extra = np.zeros((n, N_EXTRA), dtype=np.float32)
        extra[:, 0] = df["precursor_mz"].to_numpy(dtype=np.float32) / 1000.0
        for j, a in enumerate(df["adduct"]):
            extra[j, 1 + (ADDUCTS.index(a) if a in ADDUCTS else len(ADDUCTS))] = 1.0
        self.extra = torch.tensor(extra, device=device)
        self.device = device

    def batches(self, order, bs):
        """Yield (row_ids, dense_x) following `order` (a LongTensor permutation of row ids)."""
        pos = torch.empty(self.n, dtype=torch.long, device=self.device)
        pos[order] = torch.arange(self.n, device=self.device)
        epos = pos[self.row]
        sort = torch.argsort(epos)
        epos_s = epos[sort]
        bounds = torch.searchsorted(epos_s, torch.arange(0, self.n + bs, bs, device=self.device)).tolist()
        for k, start in enumerate(range(0, self.n, bs)):
            ids = order[start:start + bs]
            x = torch.zeros(len(ids), N_IN, device=self.device)
            sl = sort[bounds[k]:bounds[k + 1]]
            x[epos[sl] - start, self.bin[sl]] = self.val[sl]
            x[:, N_BINS:] = self.extra[ids]
            yield ids, x


def _net():
    return nn.Sequential(
        nn.Linear(N_IN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, FP_BITS),
    )


def train(train_df, seed):
    t0 = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = _device()
    uniq, inv = np.unique(train_df["smiles"].to_numpy(), return_inverse=True)
    y = torch.tensor(smiles_to_fp(list(uniq)), dtype=torch.float32, device=dev)[torch.tensor(inv, device=dev)]
    data = SparseSpectra(train_df, dev)
    print(f"[baseline] featurized {len(train_df)} spectra, {len(uniq)} molecules in {time.time() - t0:.0f}s")
    net = _net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * ((len(train_df) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    net.train()
    for ep in range(EPOCHS):
        order = torch.randperm(len(train_df), generator=g).to(dev)
        tot = 0.0
        for ids, x in data.batches(order, BATCH):
            loss = lossf(net(x), y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(ids)
        print(f"[baseline] epoch {ep + 1}/{EPOCHS} loss {tot / len(train_df):.4f} t={time.time() - t0:.0f}s")
    net.eval()
    return {"net": net.cpu().state_dict()}


@torch.no_grad()
def rank(state, inputs_df, cands_df):
    dev = _device()
    net = _net().to(dev)
    net.load_state_dict(state["net"])
    net.eval()
    data = SparseSpectra(inputs_df, dev)
    order = torch.arange(len(inputs_df), device=dev)
    pred = torch.zeros(len(inputs_df), FP_BITS, device=dev)
    for ids, x in data.batches(order, 1024):
        pred[ids] = torch.sigmoid(net(x))
    pred = nn.functional.normalize(pred, dim=1).cpu().numpy()
    row_of = {i: k for k, i in enumerate(inputs_df["id"])}
    out = {}
    for cid, smi, fpb in zip(cands_df["id"], cands_df["cand_smiles"], cands_df["cand_fp"]):
        smi = list(smi)
        if cid not in row_of or len(smi) == 0:
            out[cid] = smi
            continue
        c = unpack_fp(fpb).astype(np.float32)
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
        s = c @ pred[row_of[cid]]
        o = np.argsort(-s, kind="stable")
        out[cid] = [smi[j] for j in o]
    return out
