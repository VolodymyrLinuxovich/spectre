"""Honest candidate v2: honest.py (neutral losses + sparse first layer) scaled up on GPU.

Same inputs, same interface, same ranking rule (cosine to the predicted fingerprint, no priors).
Every spectrum-derived input is still weighted by peak intensity, so zeroed spectra remove all of it.
New knobs, set by the seed-0 sweep in runs/honest_v2_sweep.log:
  HIDDEN / DEPTH / EPOCHS / DROPOUT  network size and training length
  ENSEMBLE                            K networks with different init seeds, predictions averaged

Seed-0 val sweep on A10G (runs/honest_v2_sweep.log). Baseline MRR@20 0.1648, hit@1 0.1013; spectra zeroed 0.1044.
  config                              MRR@20   hit@1   zeroed MRR@20   train
  HIDDEN 1024 (= honest.py)           0.1731   0.1128  0.1059           69 s
  HIDDEN 2048                         0.1869   0.1250  0.1052          126 s
  HIDDEN 2048, 40 epochs              0.1807   0.1178  0.1076          191 s
  HIDDEN 2048, DEPTH 3, dropout 0.4   0.1754   0.1145  0.1045          143 s
  HIDDEN 1024, ENSEMBLE 3             0.1775   0.1153  0.1001          180 s
  HIDDEN 2048, ENSEMBLE 3             0.1885   0.1253  0.1092          351 s
  HIDDEN 2048, 40 epochs, dropout 0.4 0.1749   0.1135  0.1091          220 s
  HIDDEN 4096, dropout 0.5 (DEFAULT)  0.2034   0.1425  0.1154          401 s
"""
import time

import numpy as np
import torch
import torch.nn as nn

BIN = 0.1
MAX_MZ = 1000.0
N_BINS = int(MAX_MZ / BIN)  # 10000 m/z bins
NL_MAX = 500.0
USE_NL = True
N_NL = int(NL_MAX / BIN) if USE_NL else 0
ADDUCTS = ["[M+H]+", "[M+Na]+"]
N_EXTRA = 1 + len(ADDUCTS) + 1
FP_BITS = 2048
HIDDEN = 4096
DEPTH = 2  # hidden layers after the sparse input layer, counting it
ENSEMBLE = 1
EPOCHS = 25
BATCH = 512
LR = 1e-3
DROPOUT = 0.5
COS_W = 0.0  # cosine term tested: +0 hit@1, +0.5pt hit@5, 1.5x slower -> off


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smiles_to_fp(smiles):
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


class Spectra:
    """CSR peak features: idx in [0, N_BINS + N_NL), val = sqrt relative intensity."""

    def __init__(self, df, device):
        n_nl = int(NL_MAX / BIN) if USE_NL else 0
        idxs, vals, lens = [], [], np.zeros(len(df), dtype=np.int64)
        for i, (mz, it, pm) in enumerate(zip(df["mzs"], df["intensities"], df["precursor_mz"])):
            mz = np.asarray(mz, dtype=np.float64)
            it = np.asarray(it, dtype=np.float64)
            v = np.sqrt(np.maximum(it, 0) / max(it.max() if len(it) else 0, 1e-12))
            k = (mz >= 0) & (mz < MAX_MZ)
            b, w = [(mz[k] / BIN).astype(np.int64)], [v[k]]
            if n_nl:
                nl = float(pm) - mz
                k2 = (nl >= 0) & (nl < NL_MAX)
                b.append(N_BINS + (nl[k2] / BIN).astype(np.int64))
                w.append(v[k2])
            b, w = np.concatenate(b), np.concatenate(w)
            if len(b) == 0:
                continue
            ub, inv = np.unique(b, return_inverse=True)
            agg = np.zeros(len(ub))
            np.maximum.at(agg, inv, w)
            idxs.append(ub)
            vals.append(agg)
            lens[i] = len(ub)
        self.n = len(df)
        self.idx = torch.tensor(np.concatenate(idxs) if idxs else np.zeros(0, np.int64), device=device)
        self.val = torch.tensor(np.concatenate(vals) if vals else np.zeros(0), dtype=torch.float32, device=device)
        self.len = torch.tensor(lens, device=device)
        self.start = torch.cumsum(self.len, 0) - self.len
        extra = np.zeros((self.n, N_EXTRA), dtype=np.float32)
        extra[:, 0] = df["precursor_mz"].to_numpy(dtype=np.float32) / 1000.0
        for j, a in enumerate(df["adduct"]):
            extra[j, 1 + (ADDUCTS.index(a) if a in ADDUCTS else len(ADDUCTS))] = 1.0
        self.extra = torch.tensor(extra, device=device)

    def batch(self, ids):
        ln = self.len[ids]
        offs = torch.cumsum(ln, 0) - ln
        flat = torch.repeat_interleave(self.start[ids] - offs, ln) + torch.arange(int(ln.sum()), device=ln.device)
        return self.idx[flat], self.val[flat], offs, self.extra[ids]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(N_BINS + N_NL, HIDDEN, mode="sum")
        self.ext = nn.Linear(N_EXTRA, HIDDEN)
        layers = [nn.ReLU(), nn.Dropout(DROPOUT)]
        for _ in range(DEPTH - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT)]
        self.mlp = nn.Sequential(*layers, nn.Linear(HIDDEN, FP_BITS))
        nn.init.uniform_(self.emb.weight, -(N_BINS + N_NL) ** -0.5, (N_BINS + N_NL) ** -0.5)

    def forward(self, idx, val, offs, extra):
        return self.mlp(self.emb(idx, offs, per_sample_weights=val) + self.ext(extra))


def train(train_df, seed):
    t0 = time.time()
    np.random.seed(seed)
    dev = _device()
    uniq, inv = np.unique(train_df["smiles"].to_numpy(), return_inverse=True)
    y = torch.tensor(smiles_to_fp(list(uniq)), dtype=torch.float32, device=dev)[torch.tensor(inv, device=dev)]
    data = Spectra(train_df, dev)
    print(f"[honest_v2] featurized {len(train_df)} spectra in {time.time() - t0:.0f}s")
    nets = [_train_one(data, y, len(train_df), seed * 1000 + k, t0) for k in range(ENSEMBLE)]
    return {"nets": nets}


def _train_one(data, y, n, seed, t0):
    dev = y.device
    torch.manual_seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    nb = (n + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * nb)
    bce = nn.BCEWithLogitsLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    net.train()
    for ep in range(EPOCHS):
        order = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for k in range(nb):
            ids = order[k * BATCH:(k + 1) * BATCH]
            logits = net(*data.batch(ids))
            t = y[ids]
            loss = bce(logits, t)
            if COS_W:
                loss = loss + COS_W * (1 - nn.functional.cosine_similarity(torch.sigmoid(logits), t, dim=1, eps=1e-8)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(ids)
        print(f"[honest_v2] net seed {seed} epoch {ep + 1}/{EPOCHS} loss {tot / n:.4f} t={time.time() - t0:.0f}s")
    net.eval()
    return net.cpu().state_dict()


@torch.no_grad()
def rank(state, inputs_df, cands_df):
    dev = _device()
    data = Spectra(inputs_df, dev)
    pred = torch.zeros(len(inputs_df), FP_BITS, device=dev)
    for sd in state["nets"]:
        net = Net().to(dev)
        net.load_state_dict(sd)
        net.eval()
        for s in range(0, len(inputs_df), 1024):
            ids = torch.arange(s, min(s + 1024, len(inputs_df)), device=dev)
            pred[ids] += torch.sigmoid(net(*data.batch(ids))) / len(state["nets"])
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
        o = np.argsort(-(c @ pred[row_of[cid]]), kind="stable")
        out[cid] = [smi[j] for j in o]
    return out
