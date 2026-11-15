"""Reproduce the README result tables from the event log.

    python3 results/summarize.py                         # runs/events.jsonl
    python3 results/summarize.py runs/events.cloud.jsonl runs/events.seeded.jsonl
    python3 results/summarize.py --json                  # machine readable

Rules (stated so the numbers can be checked):
  * One Referee verdict per experiment: the last non-INFRA verdict for that exp_id wins.
  * INFRA: verdict "INFRA", or a "CRASH" whose error says the sandbox never ran
    (ResourceExhausted, Insufficient capacity, cancelled, not found). INFRA is reported but
    left out of every rate, because it says nothing about the code.
  * Naive loop: a claim counts as accepted when its naive_verdict is ACCEPTED (val gain > 0 on
    one run). Accepted-claim precision = accepted claims the Referee PROMOTED / accepted claims
    the Referee judged. A naive acceptance is matched to a Referee verdict by exp_id, else by code hash.
  * Guarded runs: precision = PROMOTED / judged claims (every promoted claim is accepted).
  * Model of a verdict = the `model` on the claim event with the same exp_id.
  * Compute = sum of wall_s over verdict and naive_verdict events.
Standard library only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INFRA_RE = re.compile(r"ResourceExhausted|Insufficient capacity|cancelled|not found", re.I)
ORDER = ["PROMOTED", "NOISE", "LEAK", "PRIOR", "OVERFIT", "CRASH"]


def load(paths):
    ev = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line still being written by a live run
    ev.sort(key=lambda e: e.get("ts", 0))
    return ev


def is_infra(v):
    if v.get("verdict") == "INFRA":
        return True
    return v.get("verdict") == "CRASH" and bool(INFRA_RE.search(str(v.get("error") or "")))


def summarize(ev):
    claims = {e["exp_id"]: e for e in ev if e.get("kind") == "claim" and e.get("exp_id")}
    naive = {e["exp_id"]: e for e in ev if e.get("kind") == "naive_verdict" and e.get("exp_id")}
    verdicts, infra, by_code = {}, Counter(), {}
    for e in ev:
        if e.get("kind") != "verdict" or not e.get("exp_id"):
            continue
        if is_infra(e):
            infra[e.get("tag") or "untagged"] += 1
            continue
        verdicts[e["exp_id"]] = e
        if e.get("code"):
            by_code[e["code"]] = e

    tags = defaultdict(lambda: {"claims": 0, "verdicts": Counter(), "infra": 0, "wall_s": 0.0,
                                "naive_accepted": 0, "naive_rejected": 0, "naive_rejudged": 0, "naive_real": 0})
    for c in claims.values():
        tags[c.get("tag") or "untagged"]["claims"] += 1
    for t, n in infra.items():
        tags[t]["infra"] = n
    for e in ev:
        if e.get("kind") in ("verdict", "naive_verdict") and e.get("wall_s"):
            tags[e.get("tag") or "untagged"]["wall_s"] += float(e["wall_s"])
    for v in verdicts.values():
        tags[v.get("tag") or "untagged"]["verdicts"][v["verdict"]] += 1
    for x, n in naive.items():
        t = tags[n.get("tag") or "untagged"]
        if n.get("verdict") != "ACCEPTED":
            t["naive_rejected"] += 1
            continue
        t["naive_accepted"] += 1
        ref = verdicts.get(x) or by_code.get(n.get("code"))
        if ref is not None:
            t["naive_rejudged"] += 1
            t["naive_real"] += ref["verdict"] == "PROMOTED"

    models = defaultdict(lambda: {"claims": 0, "verdicts": Counter()})
    for c in claims.values():
        models[c.get("model") or "unknown"]["claims"] += 1
    for x, v in verdicts.items():
        m = (claims.get(x) or {}).get("model") or "unknown"
        if x in claims:
            models[m]["verdicts"][v["verdict"]] += 1
    return dict(tags), dict(models)


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def mix(c: Counter):
    keys = [k for k in ORDER if c.get(k)] + sorted(k for k in c if k not in ORDER)
    return ", ".join(f"{k} {c[k]}" for k in keys) or "none yet"


def render(tags, models):
    out = ["| Run (tag) | Claims | Referee verdicts | INFRA (excluded) | Promoted | Accepted-claim precision | Compute |",
           "|---|---|---|---|---|---|---|"]
    for t in sorted(tags):
        d = tags[t]
        judged = sum(d["verdicts"].values())
        prom = d["verdicts"].get("PROMOTED", 0)
        if d["naive_accepted"] or d["naive_rejected"]:
            prec = pct(d["naive_real"], d["naive_rejudged"])
            prec += f", {d['naive_accepted']} accepted by the naive check"
            if d["naive_rejudged"] < d["naive_accepted"]:
                prec += f", {d['naive_accepted'] - d['naive_rejudged']} not re-judged yet"
        elif d["claims"] == 0:
            prec = "n/a, no proposer claims (planted or hand written files)"
        else:
            prec = pct(prom, judged)
        out.append(f"| `{t}` | {d['claims']} | {mix(d['verdicts'])} | {d['infra']} | {prom} | {prec} | "
                   f"{d['wall_s'] / 3600:.1f} h |")
    out += ["", "| Model | Claims | Verdict mix | Promoted |", "|---|---|---|---|"]
    for m in sorted(models):
        d = models[m]
        out.append(f"| {m} | {d['claims']} | {mix(d['verdicts'])} | {d['verdicts'].get('PROMOTED', 0)} |")
    return "\n".join(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = args or [ROOT / "runs" / "events.jsonl"]
    tags, models = summarize(load(paths))
    if "--json" in sys.argv:
        print(json.dumps({"tags": tags, "models": models}, indent=1, default=dict))
    else:
        print(render(tags, models))
