"""Memory comparison: past.dev vs common approaches, on a small synthetic dated swarm log.

Scenario (generator below, deterministic): 3 proposers each claim ideas; the referee later
writes a verdict for each (verdict supersedes the claim). Some ideas get a second, later
verdict that flips the first (e.g. NOISE at 1 seed, PROMOTED at 3 seeds). The referee also
writes hidden test scores into `referee-only`.

Questions are paraphrases a different proposer would ask before re-trying an idea.
Measured per backend:
  accuracy      top-1 hit carries the CURRENT verdict for the right experiment
  stale         top-1 is a superseded claim/verdict (the brief's "old fact with the correction")
  boundary      proposers never see referee-only text (must be 100%)
  cross_agent   reader != author of the top-1 hit (agent uses another agent's data)
  recall_ms     mean read latency; write_to_read_s for past.dev ingestion
  evidence_tok  tokens a proposer would paste into its prompt per question
  cost          past.dev credits (1 per 350 B ingested + 1 per recall) / embedding tokens

Run:  python -m tools.memory_bench --backends local,local-sup,local-embed,pastdev
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .memory import LocalMemory, PastDevMemory, ROOT

IDEAS = [
    ("cosine similarity between predicted and candidate Morgan fingerprints instead of Tanimoto", "NOISE", None),
    ("add precursor m/z and adduct one-hot as extra input features to the MLP", "PROMOTED", None),
    ("sqrt intensity transform before binning spectra at 0.1 Da", "NOISE", "PROMOTED"),
    ("rank candidates by number of heavy atoms close to the training prior", "PRIOR", None),
    ("train on val spectra whose InChIKey14 also appears in train", "LEAK", None),
    ("contrastive spectrum-molecule encoder like CLERMS / JESTR with in-batch negatives", "PROMOTED", None),
    ("formula-aware peak annotation: subformula assignment for each fragment peak", "CRASH", "NOISE"),
    ("ensemble of 5 seeds averaging fingerprint predictions", "PROMOTED", None),
    ("neutral loss features (precursor minus fragment) added to binned peaks", "NOISE", None),
    ("MIST-style formula transformer over peak subformulae", "CRASH", None),
    ("label smoothing on the fingerprint BCE loss", "NOISE", None),
    ("rerank top-20 with a learned pairwise scorer over candidate fingerprints", "PROMOTED", "LEAK"),
]
PARAPHRASE = [
    "did anyone try cosine instead of tanimoto for fingerprint matching?",
    "is it worth feeding precursor mass and adduct type into the model?",
    "square-root scaling of peak intensities, did it help?",
    "what happened with the heavy atom count prior for ranking?",
    "using validation molecules that overlap training structures",
    "contrastive learning between spectra and molecules, result?",
    "subformula annotation of fragment peaks, outcome?",
    "averaging predictions over several random seeds",
    "neutral loss features from the precursor, did they work?",
    "transformer over peak formulae like MIST",
    "smoothing the labels of the fingerprint loss",
    "pairwise reranker on the top 20 candidates",
]


def generate(n_proposers: int = 3, start: datetime | None = None, seed: int = 0):
    """Dated event list: (ts, author, exp dict). Kept small (tens of docs) per brief 7.1."""
    rnd = random.Random(seed)
    t = start or datetime(2026, 11, 15, 1, 0, tzinfo=timezone.utc)
    events, truth = [], {}
    for i, (idea, v1, v2) in enumerate(IDEAS):
        eid = f"exp-{i + 1:03d}"
        prop = f"proposer-{i % n_proposers + 1}"
        val_gain = round(rnd.uniform(0.2, 2.5), 2)
        events.append((t, prop, {"exp_id": eid, "kind": "claim", "idea": idea,
                                 "metrics": {"val_hit@1_gain_claimed": f"+{val_gain}"}}))
        t += timedelta(minutes=7)
        events.append((t, "referee", {"exp_id": eid, "kind": "verdict", "idea": idea, "verdict": v1,
                                      "metrics": {"seeds": 1 if v2 else 3},
                                      "evidence": f"paired gain vs noise band, canaries ({v1})"}))
        truth[eid] = v1
        if v2:
            t += timedelta(minutes=5)
            events.append((t, "referee", {"exp_id": eid, "kind": "verdict", "idea": idea,
                                          "verdict": v2, "metrics": {"seeds": 3},
                                          "evidence": f"re-run with 3 seeds overturned {v1} -> {v2}"}))
            truth[eid] = v2
        events.append((t + timedelta(minutes=1), "referee",
                       {"exp_id": eid, "kind": "test_score",
                        "idea": f"SECRET-TEST {eid} hidden test hit@1 = {round(rnd.uniform(5, 40), 2)}"}))
        t += timedelta(minutes=9)
    return events, truth, t


def run(mem, label: str, n_proposers: int, wait_each: bool = False):
    events, truth, t_end = generate(n_proposers)
    mem.setup(n_proposers)
    ingest_ids, w0 = [], time.time()
    for ts, author, exp in events:
        r = mem.record(exp, author, timestamp=ts.isoformat().replace("+00:00", "Z"))
        if r.get("ingestionId"):
            ingest_ids.append(r["ingestionId"])
    write_s = time.time() - w0
    readable_s = 0.0
    if ingest_ids:
        t0 = time.time()
        for iid in ingest_ids:
            mem.wait(iid)
        readable_s = time.time() - t0

    res = {"backend": label, "n_docs": len(events), "write_s": round(write_s, 2),
           "all_readable_after_s": round(readable_s, 1)}
    ok = stale = cross = leak = 0
    lat, rows = [], []
    for i, q in enumerate(PARAPHRASE):
        eid = f"exp-{i + 1:03d}"
        reader = f"proposer-{(i + 1) % n_proposers + 1}"   # a different proposer than the author
        t0 = time.time()
        hits = mem.recall(q, k=5, reader=reader)
        lat.append(1000 * (time.time() - t0))
        if any("SECRET-TEST" in h.content for h in hits):
            leak += 1
        top = next((h for h in hits if eid in h.exp_ids or eid in h.content), None)
        first = hits[0] if hits else None
        got = None
        if first is not None and (eid in first.exp_ids or eid in first.content):
            got = first.verdict
            if got != truth[eid]:   # on-topic but not the current verdict = stale fact
                stale += 1
            if reader not in first.authors:
                cross += 1
        correct = got == truth[eid]
        ok += correct
        rows.append({"q": q, "reader": reader, "truth": truth[eid], "top1_verdict": got,
                     "top1_superseded": bool(first and first.superseded_at), "correct": correct,
                     "found_in_top5": top is not None,
                     "top1_excerpt": (first.content[:160] if first else None)})
    # explicit read-boundary probe: every proposer asks for the hidden test scores
    for p in range(1, n_proposers + 1):
        hits = mem.recall("hidden test hit@1 score SECRET-TEST for each experiment", k=10,
                          reader=f"proposer-{p}")
        leak += sum("SECRET-TEST" in h.content for h in hits)
    ref_hits = mem.recall("hidden test hit@1 score SECRET-TEST for each experiment", k=10,
                          reader="referee")
    res["referee_sees_test"] = sum("SECRET-TEST" in h.content for h in ref_hits)
    n = len(PARAPHRASE)
    st = mem.stats
    res.update({
        "accuracy": round(ok / n, 3), "stale_top1": stale, "boundary_ok": leak == 0,
        "boundary_leaks": leak, "cross_agent_top1": cross,
        "recall_ms_mean": round(sum(lat) / n, 1),
        "evidence_tokens_per_q": round(st["evidence_tokens"] / max(1, st["recalls"]), 0),
        "ingest_bytes": st["ingest_bytes"],
    })
    if isinstance(mem, PastDevMemory):
        res["credits_est"] = sum(-(-len(json.dumps(e[2]).encode()) // 350) for e in events) + n
    if st.get("embed_tokens"):
        res["embed_tokens"] = st["embed_tokens"]
    return res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="local,local-sup")
    ap.add_argument("--n-proposers", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "runs" / "memory_bench.json"))
    a = ap.parse_args()
    results = []
    for b in a.backends.split(","):
        tmp = ROOT / "runs" / f"memory_bench_{b}.jsonl"
        tmp.unlink(missing_ok=True)
        mem = {"local": lambda: LocalMemory(tmp),
               "local-sup": lambda: LocalMemory(tmp, supersede=True),
               "local-embed": lambda: LocalMemory(tmp, mode="embed"),
               "local-unscoped": lambda: LocalMemory(tmp, scoped=False),
               "local-embed-sup": lambda: LocalMemory(tmp, mode="embed", supersede=True),
               "pastdev": lambda: PastDevMemory(ns="bench-")}[b]()
        res, rows = run(mem, b, a.n_proposers)
        results.append({"summary": res, "rows": rows})
        print(json.dumps(res))
    Path(a.out).write_text(json.dumps(results, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
