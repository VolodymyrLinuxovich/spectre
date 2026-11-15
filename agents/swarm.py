"""Spectre swarm: proposers search prior work (Exa), check shared memory (past.dev),
write code, and submit to the Referee. Only Referee verdicts become facts.

Run:
  .venv/bin/python -m agents.swarm --proposers 4 --rounds 3            # full system
  .venv/bin/python -m agents.swarm --naive ...                          # unguarded loop
  .venv/bin/python -m agents.swarm --proposers 1 --rounds 12 --tag single  # single agent, same budget
  .venv/bin/python -m agents.swarm --no-exa / --memory local / --memory none   # ablations
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from bench.referee import Referee, log  # noqa: E402
from tools import exa_search  # noqa: E402
from tools.memory import get_memory, summarize_for_prompt  # noqa: E402

FOCUS = [
    "spectrum featurization: peak encoding, intensity transforms, neutral losses relative to the precursor, "
    "peak-difference features, binning resolution",
    "model and training objective: loss for fingerprint prediction (focal, weighted BCE, cosine), "
    "contrastive learning between spectra and candidate fingerprints, regularization, capacity",
    "candidate scoring: how predicted fingerprints are compared to candidates (Tanimoto, weighted bits, "
    "learned reranker over candidates), using candidate structure only through the learned model",
    "data and augmentation: adduct handling, collision energy robustness, peak dropout or noise augmentation, "
    "multiple fingerprint types (counts, larger radius, MACCS)",
]

RULES = """Hard rules for model.py (the sandbox enforces them; violating them wastes the run):
- Keep the interface: train(train_df, seed) -> state ; rank(state, inputs_df, cands_df) -> {id: [cand_smiles best first]}.
- Only these packages: numpy, pandas, torch, scikit-learn, scipy, rdkit. No network. One GPU. Train+rank must finish in 10 minutes.
- inputs_df columns: id, mzs, intensities, precursor_mz, adduct, formula. train_df also has smiles, inchikey14.
- cands_df columns: id, cand_smiles (list[str]), cand_fp (bytes; np.unpackbits(np.frombuffer(b, np.uint8)).reshape(-1, 2048) = Morgan r2 2048).
- Candidate order is randomized. Labels for val/test do not exist in the sandbox.
- Return the FULL file in one ```python block."""


class LLM:
    def __init__(self):
        url = os.getenv("SPECTRE_LLM_URL")
        if url:
            self.client = OpenAI(base_url=url, api_key=os.getenv("SPECTRE_LLM_API_KEY", "x"))
            self.model = os.getenv("SPECTRE_LLM_MODEL")
            self.provider = "modal"
        else:
            self.client = OpenAI()
            self.model = os.getenv("SPECTRE_FALLBACK_MODEL", "gpt-5.5")
            self.provider = "openai"

    def chat(self, system: str, user: str, max_tokens: int = 12000) -> str:
        lim = {"max_completion_tokens": max_tokens * 2} if self.provider == "openai" else {"max_tokens": max_tokens}
        r = self.client.chat.completions.create(
            model=self.model, **lim,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        txt = r.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()


def extract_json(t: str) -> dict:
    m = re.search(r"```json\s*(.*?)```", t, re.S) or re.search(r"(\{.*\})", t, re.S)
    return json.loads(m.group(1)) if m else {}


def extract_code(t: str) -> str | None:
    blocks = re.findall(r"```python\s*(.*?)```", t, re.S)
    if not blocks:
        return None
    code = max(blocks, key=len)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    return code if {"train", "rank"} <= names else None


class Proposer:
    def __init__(self, i: int, args, llm: LLM, ref: Referee, mem, budget: "Budget"):
        self.i, self.name, self.args = i, f"proposer-{i}", args
        self.focus = FOCUS[(i - 1) % len(FOCUS)]
        self.llm, self.ref, self.mem, self.budget = llm, ref, mem, budget

    def ev(self, kind, **kw):
        log(self.name, kind, tag=self.args.tag, **kw)

    def step(self):
        inc = self.ref.incumbent()
        exp_id = f"exp-{uuid.uuid4().hex[:6]}"

        # 1. what does the swarm already know? (cross-agent read)
        known = ""
        if self.mem is not None:
            hits = self.mem.recall(f"experiments on {self.focus}", k=8, reader=self.name)
            known = summarize_for_prompt(hits)
            self.ev("memory_read", exp_id=exp_id, n=len(hits),
                    authors=sorted({a for h in hits for a in h.authors}),
                    hits=[h.to_dict() for h in hits[:5]])

        # 2. hypothesis
        sys_p = ("You are a computational mass spectrometry researcher in an autoresearch swarm. "
                 "Your claims are worthless until an independent Referee verifies them with multiple seeds, "
                 "a hidden test set, a label-shuffle canary and a spectrum-ablation canary. "
                 "Propose changes that genuinely improve learning from the spectrum.")
        h = extract_json(self.llm.chat(sys_p, f"""Your focus: {self.focus}

Current incumbent model.py:
```python
{inc}
```

What the swarm already tried (Referee verdicts are facts; claims are not):
{known or "(nothing yet)"}

Propose ONE concrete hypothesis not already tested. Reply ```json {{"hypothesis": "...", "mechanism": "why it should help", "expected_gain": "e.g. +0.005 MRR", "search_query": "query to find prior work on exactly this"}}```""", 2000))
        if not h.get("hypothesis"):
            self.ev("error", exp_id=exp_id, error="no hypothesis")
            return
        self.ev("hypothesis", exp_id=exp_id, **h)

        # 3. prior work via Exa: has it been tested? how did they do it?
        sources, cites = "", []
        if not self.args.no_exa:
            rows = exa_search.search_methods(h.get("search_query") or h["hypothesis"], n=5)
            sources = exa_search.format_for_prompt(rows)
            cites = [f"{r['title']} <{r['url']}>" for r in rows]
            self.ev("exa_search", exp_id=exp_id, query=h.get("search_query"),
                    results=[{k: r.get(k) for k in ("title", "url", "published")} for r in rows])

        # 4. decide + implement
        out = self.llm.chat(sys_p, f"""Hypothesis: {h['hypothesis']}
Mechanism: {h.get('mechanism')}

Prior work found by web search (cite by [n] where you use it):
{sources or "(search disabled)"}

Swarm memory:
{known or "(nothing yet)"}

Incumbent model.py:
```python
{inc}
```

{RULES}

First a ```json {{"decision": "implement" | "skip", "reason": "...", "uses_sources": [n,...], "change_summary": "..."}}```.
Skip if prior work or swarm memory shows it already failed or is already in the incumbent.
If implement, then the full new model.py.""")
        d = extract_json(out)
        used = [cites[k - 1] for k in d.get("uses_sources", []) if isinstance(k, int) and 0 < k <= len(cites)]
        if d.get("decision") == "skip":
            self.ev("skipped", exp_id=exp_id, reason=d.get("reason"), citations=used)
            if self.mem is not None:
                self.mem.record({"exp_id": exp_id, "kind": "note", "idea": h["hypothesis"],
                                 "evidence": "skipped before compute: " + str(d.get("reason")),
                                 "citations": used}, author=self.name)
            return
        code = extract_code(out)
        if code is None:
            self.ev("error", exp_id=exp_id, error="no valid model.py")
            return
        claim = {"exp_id": exp_id, "kind": "claim", "idea": h["hypothesis"],
                 "diff_summary": d.get("change_summary"), "citations": used}
        if self.mem is not None:
            self.mem.record(claim, author=self.name)
        self.ev("claim", exp_id=exp_id, change=d.get("change_summary"), citations=used,
                expected=h.get("expected_gain"))

        # 5. Referee (one repair attempt on crash)
        if not self.budget.take():
            return
        v = self.ref.judge(code, claim=h["hypothesis"], agent=self.name, guarded=not self.args.naive,
                               exp_id=exp_id, tag=self.args.tag)
        if v["verdict"] == "CRASH" and self.budget.take():
            fix = extract_code(self.llm.chat(sys_p, f"This model.py crashed:\n```python\n{code}\n```\nError:\n{v.get('error')}\n{v.get('stderr_tail')}\n\n{RULES}\nReturn the fixed full file."))
            if fix:
                code = fix
                v = self.ref.judge(code, claim=h["hypothesis"], agent=self.name, guarded=not self.args.naive,
                               exp_id=exp_id, tag=self.args.tag)
        pub = Referee.public_view(v)
        (ROOT / "runs" / "code").mkdir(exist_ok=True)
        (ROOT / "runs" / "code" / f"{exp_id}.py").write_text(code)
        if self.mem is not None:
            self.mem.record({"exp_id": exp_id, "kind": "verdict", "idea": h["hypothesis"],
                             "verdict": v["verdict"], "evidence": v.get("evidence") or v.get("error"),
                             "metrics": {k: round(pub[k], 4) for k in ("val_gain",) if k in pub},
                             "citations": used}, author="referee")
            if "test_gain" in v:
                self.mem.record({"exp_id": exp_id, "kind": "test_score",
                                 "idea": f"test gain {v['test_gain']:+.4f} CI {v['test_ci']}"},
                                author="referee")


class Budget:
    def __init__(self, n):
        self.n, self.lock = n, threading.Lock()

    def take(self):
        with self.lock:
            if self.n <= 0:
                return False
            self.n -= 1
            return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=None, help="max Referee evaluations")
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--no-exa", action="store_true")
    ap.add_argument("--memory", default="pastdev", choices=["pastdev", "local", "none"])
    ap.add_argument("--tag", default="swarm")
    args = ap.parse_args()

    llm, ref = LLM(), Referee()
    mem = None if args.memory == "none" else get_memory(args.memory)
    if mem is not None:
        mem.setup(args.proposers)
    budget = Budget(args.budget or args.proposers * args.rounds * 2)
    log("orchestrator", "start", tag=args.tag, llm=llm.provider, model=llm.model, **vars(args))
    ps = [Proposer(i + 1, args, llm, ref, mem, budget) for i in range(args.proposers)]

    def loop(p):
        for _ in range(args.rounds):
            try:
                p.step()
            except Exception as e:  # keep the swarm alive
                p.ev("error", error=repr(e)[:500])

    with ThreadPoolExecutor(len(ps)) as ex:
        list(ex.map(loop, ps))
    log("orchestrator", "end", tag=args.tag)


if __name__ == "__main__":
    main()
