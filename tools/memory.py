"""Shared experiment memory for the Spectre swarm.

Two backends, one interface:
  PastDevMemory  -> past.dev Memory API (identities + audiences, async ingest, ranked recall)
  LocalMemory    -> shared JSONL file + naive lexical (or OpenAI embedding) match; the
                    "common approach" baseline for the comparison.

Interface (both backends):
  setup(n_proposers)                      create identities/audiences (no-op locally)
  record(exp, author, audience=None, wait=False) -> {"id", "ingestionId"?, ...}
  recall(idea_text, k=5, reader="orchestrator") -> [Hit]
  list_all(reader=None) -> [record dicts]   (local log of everything this process/team wrote)
  delete(source_id)

Design (agreed with lead):
  identities : proposer-1..N, referee, orchestrator
  audiences  : verdicts            (all identities read; referee verdicts + proposer claims)
               proposer-<i>-scratch (only proposer-i reads)
               referee-only        (referee + orchestrator; hidden test-set scores)
  A proposer's claim and the referee's verdict about the same experiment both go to
  `verdicts`; the verdict is written later, so it supersedes the claim (past.dev does this
  via fact supersession; LocalMemory emulates it by exp_id).

Experiment dict fields (all optional except idea):
  exp_id, idea, diff_summary, verdict (PROMOTED|NOISE|LEAK|PRIOR|CRASH|CLAIM), metrics{}, evidence,
  citations[] (Exa URLs), kind ("claim"|"verdict"|"note"|"test_score")
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from ._env import ROOT, load_env

BASE = "https://api.past.dev/api/v1"
LOG = ROOT / "runs" / "memory_log.jsonl"          # what we sent to past.dev (list_all, audit)
LOCAL_DB = ROOT / "runs" / "memory_local.jsonl"   # LocalMemory store
VERDICTS = ("PROMOTED", "NOISE", "LEAK", "PRIOR", "CRASH")


def identities(n_proposers: int) -> list[str]:
    return [f"proposer-{i}" for i in range(1, n_proposers + 1)] + ["referee", "orchestrator"]


def audiences(n_proposers: int) -> dict[str, list[str]]:
    ids = identities(n_proposers)
    a = {"verdicts": ids, "referee-only": ["referee", "orchestrator"]}
    for i in range(1, n_proposers + 1):
        a[f"proposer-{i}-scratch"] = [f"proposer-{i}"]
    return a


def default_audience(exp: dict, author: str) -> str:
    kind = exp.get("kind") or ("verdict" if author == "referee" else "claim")
    if kind == "test_score":
        return "referee-only"
    if kind == "note" and author.startswith("proposer-"):
        return f"{author}-scratch"
    return "verdicts"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def render(exp: dict, author: str) -> tuple[str, str]:
    """Experiment -> (label, content). Plain sentences so past.dev's fact extraction can
    tell a claim (hypothesis) from a verdict (fact) and who wrote each."""
    eid = exp.get("exp_id") or "exp-?"
    kind = exp.get("kind") or ("verdict" if author == "referee" else "claim")
    idea = exp.get("idea", "").strip()
    v = (exp.get("verdict") or "").upper()
    lines = []
    if kind == "verdict":
        lines.append(f"Referee verdict for experiment {eid}: {v}.")
        lines.append(f"Idea tested in {eid}: {idea}")
        lines.append(f"The idea in {eid} {'WORKED (a real gain on validation, promoted)' if v == 'PROMOTED' else 'DID NOT WORK (' + v + ')'}.")
    elif kind == "test_score":
        lines.append(f"Hidden test score for experiment {eid} (referee only): {idea}")
    elif kind == "note":
        lines.append(f"Private note by {author} on {eid}: {idea}")
    else:
        lines.append(f"Hypothesis (unverified claim) by {author}, experiment {eid}: {idea}")
        lines.append(f"Status of {eid}: claimed, waiting for the referee.")
    if exp.get("diff_summary"):
        lines.append(f"Change: {exp['diff_summary']}")
    if exp.get("metrics"):
        lines.append("Metrics: " + ", ".join(f"{k}={v}" for k, v in exp["metrics"].items()))
    if exp.get("evidence"):
        lines.append(f"Evidence: {exp['evidence']}")
    if exp.get("citations"):
        lines.append("Sources found via Exa: " + " ; ".join(exp["citations"][:5]))
    lines.append(f"Written by {author}.")
    label = f"{eid} {kind} {v}".strip()
    return label, "\n".join(lines)


@dataclass
class Hit:
    rank: int
    content: str
    occurred_at: str | None
    source_ids: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    exp_ids: list[str] = field(default_factory=list)
    verdict: str | None = None          # from source metadata when present
    superseded_at: str | None = None
    kind: str | None = None             # past.dev artifact kind
    excerpts: list[str] = field(default_factory=list)
    score: float | None = None

    def to_dict(self):
        return asdict(self)


def _verdict_from_text(t: str) -> str | None:
    m = re.search(r"verdict for experiment \S+: (\w+)", t)
    return m.group(1) if m else ("CLAIM" if "unverified claim" in t else None)


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- past.dev
class PastDevMemory:
    name = "pastdev"

    def __init__(self, api_key: str | None = None, timeout: float = 30, ns: str = ""):
        """ns: prefix for identities, audiences and source ids (e.g. "bench-") so a
        benchmark or test can share the one past.dev project without the swarm reading it."""
        self.ns = ns
        load_env()
        self.key = api_key or os.environ.get("PAST_API_KEY")
        if not self.key:
            raise RuntimeError("PAST_API_KEY is not set in .env")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.key}",
                               "Content-Type": "application/json"})
        self.timeout = timeout
        self.stats = Counter()   # calls, recall_ms_total, ingest_bytes, evidence_tokens

    def _req(self, method: str, path: str, **kw):
        for attempt in range(5):
            r = self.s.request(method, BASE + path, timeout=self.timeout, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"past.dev {method} {path} -> {r.status_code}: {r.text[:400]}")
            self.stats["calls"] += 1
            return r.json() if r.content else {}
        raise RuntimeError(f"past.dev {method} {path} kept failing ({r.status_code})")

    # --- setup
    def setup(self, n_proposers: int = 3) -> dict:
        out = {}
        ns = self.ns
        for slug, members in audiences(n_proposers).items():
            out[slug] = self._req("PUT", f"/audiences/{ns}{slug}",
                                  json={"kind": "fixed", "name": ns + slug,
                                        "identities": [ns + m for m in members]})
        self._req("POST", "/identities/bulk",
                  json=[{"identity": ns + i, "traits": {"role": i.split("-")[0]}}
                        for i in identities(n_proposers)])
        return out

    def reach(self, identity: str) -> dict:
        return self._req("GET", f"/identities/{self.ns}{identity}")

    # --- write
    def record(self, exp: dict, author: str, audience: str | None = None,
               wait: bool = False, timestamp: str | None = None) -> dict:
        exp = dict(exp)
        exp.setdefault("exp_id", f"exp-{uuid.uuid4().hex[:8]}")
        kind = exp.get("kind") or ("verdict" if author == "referee" else "claim")
        audience = audience or default_audience(exp, author)
        label, content = render(exp, author)
        sid = f"{self.ns}{exp['exp_id']}-{kind}-{author}"
        meta = {"exp_id": exp["exp_id"], "kind": kind, "author": author,
                "verdict": (exp.get("verdict") or ("CLAIM" if kind == "claim" else None)),
                "audience": audience}
        body = {"id": sid, "content": content, "label": label,
                "timestamp": timestamp or _now(), "identity": self.ns + author,
                "audience": self.ns + audience, "metadata": meta}
        t0 = time.time()
        res = self._req("POST", "/ingest", json=body)
        self.stats["ingest_bytes"] += len(content.encode())
        out = {"id": sid, "ingestionId": res.get("ingestionId"), "status": res.get("status"),
               "accept_ms": round(1000 * (time.time() - t0), 1)}
        _append(LOG, {"ts": body["timestamp"], "backend": self.name, **meta,
                      "source_id": sid, "content": content, "exp": exp})
        if wait and out["ingestionId"]:
            out["readable_s"] = self.wait(out["ingestionId"])
        return out

    def wait(self, ingestion_id: str, max_s: float = 600) -> float:
        """Poll until completed. Returns seconds from now until readable."""
        t0 = time.time()
        delay = 1.0
        while time.time() - t0 < max_s:
            st = self._req("GET", f"/ingest/{ingestion_id}")
            if st.get("status") == "completed":
                return round(time.time() - t0, 2)
            if st.get("status") == "failed":
                raise RuntimeError(f"ingestion {ingestion_id} failed: {st.get('failure')}")
            time.sleep(delay)
            delay = 2.0 if time.time() - t0 < 60 else 10.0
        raise TimeoutError(f"ingestion {ingestion_id} not completed after {max_s}s")

    # --- read
    def recall(self, idea_text: str, k: int = 5, reader: str = "orchestrator",
               query_timestamp: str | None = None) -> list[Hit]:
        body = {"query": idea_text, "identity": self.ns + reader, "limit": k}
        if query_timestamp:
            body["queryTimestamp"] = query_timestamp
        t0 = time.time()
        res = self._req("POST", "/recall", json=body)
        self.stats["recall_ms_total"] += 1000 * (time.time() - t0)
        self.stats["recalls"] += 1
        self.stats["evidence_tokens"] += res.get("usedEvidenceTokens", 0) or 0
        self.last_pending = res.get("pending")
        hits = []
        for r in res.get("results", [])[:k]:
            srcs = r.get("sources") or []
            metas = [s.get("metadata") or {} for s in srcs]
            art = r.get("artifact") or {}
            verdict = next((m.get("verdict") for m in metas if m.get("kind") == "verdict"), None) \
                or next((m.get("verdict") for m in metas if m.get("verdict")), None) \
                or _verdict_from_text(r.get("content", ""))
            hits.append(Hit(
                rank=r.get("rank", len(hits) + 1), content=r.get("content", ""),
                occurred_at=r.get("occurredAt"),
                source_ids=[s.get("sourceId") for s in srcs],
                authors=sorted({m.get("author") for m in metas if m.get("author")}),
                exp_ids=sorted({m.get("exp_id") for m in metas if m.get("exp_id")}),
                verdict=verdict, superseded_at=art.get("supersededAt"), kind=art.get("kind"),
                excerpts=[e for s in srcs for e in (s.get("excerpts") or [])][:6],
            ))
        return hits

    def list_all(self, reader: str | None = None) -> list[dict]:
        """past.dev has no list route; return our write log (filtered by reader reach)."""
        return _read_log(LOG, self.name, reader)

    def delete(self, source_id: str) -> dict:
        return self._req("DELETE", f"/data-points/{source_id}")

    def usage(self) -> dict:
        return self._req("GET", "/usage")


def _read_log(path: Path, backend: str, reader: str | None) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("backend") == backend]
    if reader:
        rows = [r for r in rows if _can_read(reader, r.get("audience"))]
    return rows


def _can_read(reader: str, audience: str | None) -> bool:
    if audience in (None, "project", "verdicts"):
        return True
    if audience == "referee-only":
        return reader in ("referee", "orchestrator")
    m = re.fullmatch(r"(proposer-\d+)-scratch", audience or "")
    return bool(m) and m.group(1) == reader


# --------------------------------------------------------------------------- local baseline
_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or is are was were be by with from at as it this that "
            "experiment idea written claim verdict referee proposer status".split())


def _toks(t: str) -> list[str]:
    return [w for w in _TOK.findall(t.lower()) if w not in _STOP and len(w) > 1]


class LocalMemory:
    """Common approach: one shared JSONL scratchpad, lexical (BM25-ish) or embedding match.

    supersede=False is the plain "vector store / scratchpad" baseline the past.dev brief
    describes (returns the stale claim next to the verdict). supersede=True adds a hand-
    written latest-by-exp_id rule, i.e. what you'd have to build yourself.
    """
    name = "local"

    def __init__(self, path: Path = LOCAL_DB, mode: str = "lexical", supersede: bool = False,
                 scoped: bool = True):
        self.path = Path(path)
        self.mode = mode
        self.supersede = supersede
        self.scoped = scoped
        self.stats = Counter()
        self._emb_cache: dict[str, list[float]] = {}

    def setup(self, n_proposers: int = 3) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        return audiences(n_proposers)

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def record(self, exp: dict, author: str, audience: str | None = None,
               wait: bool = False, timestamp: str | None = None) -> dict:
        exp = dict(exp)
        exp.setdefault("exp_id", f"exp-{uuid.uuid4().hex[:8]}")
        kind = exp.get("kind") or ("verdict" if author == "referee" else "claim")
        audience = audience or default_audience(exp, author)
        label, content = render(exp, author)
        sid = f"{exp['exp_id']}-{kind}-{author}"
        rec = {"source_id": sid, "ts": timestamp or _now(), "author": author, "audience": audience,
               "exp_id": exp["exp_id"], "kind": kind,
               "verdict": exp.get("verdict") or ("CLAIM" if kind == "claim" else None),
               "content": content, "exp": exp, "backend": self.name}
        t0 = time.time()
        _append(self.path, rec)
        self.stats["ingest_bytes"] += len(content.encode())
        return {"id": sid, "accept_ms": round(1000 * (time.time() - t0), 2), "readable_s": 0.0}

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        load_env()
        need = [t for t in texts if t not in self._emb_cache]
        if need:
            r = OpenAI().embeddings.create(model="text-embedding-3-small", input=need)
            for t, d in zip(need, r.data):
                self._emb_cache[t] = d.embedding
            self.stats["embed_tokens"] += r.usage.total_tokens
        return [self._emb_cache[t] for t in texts]

    def recall(self, idea_text: str, k: int = 5, reader: str = "orchestrator",
               query_timestamp: str | None = None) -> list[Hit]:
        t0 = time.time()
        rows = self._rows()
        if self.scoped:
            rows = [r for r in rows if _can_read(reader, r["audience"])]
        if query_timestamp:
            rows = [r for r in rows if r["ts"] <= query_timestamp]
        superseded: dict[str, str] = {}
        if self.supersede:
            latest: dict[str, dict] = {}
            for r in sorted(rows, key=lambda r: r["ts"]):
                prev = latest.get(r["exp_id"])
                if prev is not None and r["kind"] == "verdict":
                    superseded[prev["source_id"]] = r["ts"]
                if r["kind"] in ("claim", "verdict"):
                    latest[r["exp_id"]] = r
        scored = []
        if rows and self.mode == "embed":
            q, *docs = self._embed([idea_text] + [r["content"] for r in rows])
            qn = math.sqrt(sum(x * x for x in q))
            for r, d in zip(rows, docs):
                s = sum(a * b for a, b in zip(q, d)) / (qn * math.sqrt(sum(x * x for x in d)) + 1e-9)
                scored.append((s, r))
        elif rows:
            docs = [_toks(r["content"]) for r in rows]
            df = Counter(w for d in docs for w in set(d))
            n = len(docs)
            avg = sum(map(len, docs)) / n
            qt = _toks(idea_text)
            ql = idea_text.lower()
            for r, d in zip(rows, docs):
                tf = Counter(d)
                s = sum(math.log(1 + (n - df[w] + .5) / (df[w] + .5)) * tf[w] * 2.2
                        / (tf[w] + 1.2 * (.25 + .75 * len(d) / avg)) for w in qt if w in tf)
                if r["exp_id"].lower() in ql:
                    s += 5
                scored.append((s, r))
        scored.sort(key=lambda x: (-x[0], x[1]["ts"]))
        if self.supersede:  # demote superseded rows below live ones
            scored.sort(key=lambda x: x[1]["source_id"] in superseded)
        hits = [Hit(rank=i + 1, content=r["content"], occurred_at=r["ts"],
                    source_ids=[r["source_id"]], authors=[r["author"]], exp_ids=[r["exp_id"]],
                    verdict=r["verdict"], superseded_at=superseded.get(r["source_id"]),
                    kind="source", excerpts=[r["content"][:300]], score=round(s, 4))
                for i, (s, r) in enumerate(scored[:k]) if s > 0]
        self.stats["recall_ms_total"] += 1000 * (time.time() - t0)
        self.stats["recalls"] += 1
        self.stats["evidence_tokens"] += sum(len(h.content) for h in hits) // 4
        return hits

    def list_all(self, reader: str | None = None) -> list[dict]:
        rows = self._rows()
        return [r for r in rows if reader is None or _can_read(reader, r["audience"])]

    def delete(self, source_id: str) -> dict:
        rows = [r for r in self._rows() if r["source_id"] != source_id]
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return {"deleted": source_id}


def get_memory(backend: str | None = None, **kw):
    """backend: 'pastdev' | 'local' (default: env SPECTRE_MEMORY or pastdev if key set)."""
    load_env()
    backend = backend or os.environ.get("SPECTRE_MEMORY") or \
        ("pastdev" if os.environ.get("PAST_API_KEY") else "local")
    return PastDevMemory(**kw) if backend == "pastdev" else LocalMemory(**kw)


def summarize_for_prompt(hits: list[Hit], max_chars: int = 2500) -> str:
    """What a proposer sees before proposing: prior ideas, who tried them, verdicts."""
    out, n = [], 0
    for h in hits:
        tag = h.verdict or "?"
        sup = f" [SUPERSEDED {h.superseded_at[:19]}]" if h.superseded_at else ""
        who = ",".join(h.authors) or "?"
        line = f"- ({tag}{sup}; by {who}; {', '.join(h.exp_ids) or ''}) {' '.join(h.content.split())[:400]}"
        if n + len(line) > max_chars:
            break
        out.append(line)
        n += len(line)
    return "\n".join(out) or "(no prior experiments found)"
