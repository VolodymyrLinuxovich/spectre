"""Exa search for proposer agents: find methods the model doesn't know, with sources.

search_methods(query, n=5) -> [{title, url, published, highlights, score, query}]
fetch_contents(url) -> {title, url, published, text, highlights}

Every call (hit or miss) is appended to runs/exa_cache.jsonl; identical calls are served
from the cache so reruns are free and the demo can replay sources offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from ._env import ROOT, load_env

CACHE = ROOT / "runs" / "exa_cache.jsonl"

# Bias toward our domain without forcing it: Exa neural search reads the whole query,
# so a short domain suffix steers ranking toward MS/MS + molecular retrieval papers.
DOMAIN_HINT = "tandem mass spectrometry MS/MS metabolomics molecular structure retrieval"
PAPER_DOMAINS = [
    "arxiv.org", "biorxiv.org", "chemrxiv.org", "pubs.acs.org", "nature.com",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "openreview.net", "proceedings.neurips.cc",
    "academic.oup.com", "sciencedirect.com", "link.springer.com", "jcheminf.biomedcentral.com",
    "pubs.rsc.org", "wiley.com", "mdpi.com",
]

_client = None
_mem: dict[str, dict] | None = None


def _exa():
    global _client
    if _client is None:
        load_env()
        key = os.environ.get("EXA_API_KEY")
        if not key:
            raise RuntimeError("EXA_API_KEY is not set in .env")
        from exa_py import Exa
        _client = Exa(key)
    return _client


def _key(kind: str, payload: dict) -> str:
    return hashlib.sha256(json.dumps([kind, payload], sort_keys=True).encode()).hexdigest()[:24]


def _load_cache() -> dict[str, dict]:
    global _mem
    if _mem is None:
        _mem = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    _mem[rec["key"]] = rec
                except Exception:
                    pass
    return _mem


def _put(key: str, kind: str, payload: dict, result, latency_s: float) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    rec = {"key": key, "kind": kind, "ts": time.time(), "payload": payload,
           "latency_s": round(latency_s, 3), "result": result}
    with CACHE.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    _load_cache()[key] = rec


def _row(r, query: str | None = None) -> dict:
    return {
        "title": getattr(r, "title", None),
        "url": r.url,
        "published": getattr(r, "published_date", None),
        "highlights": list(getattr(r, "highlights", None) or []),
        "score": getattr(r, "score", None),
        **({"query": query} if query else {}),
    }


def search_methods(query: str, n: int = 5, *, domain_bias: bool = True,
                   papers_only: bool = True, since: str | None = None,
                   use_cache: bool = True) -> list[dict]:
    """Neural search over research papers, with text highlights for each hit.

    since: ISO date (e.g. "2025-01-01") to keep only recent work.
    """
    q = f"{query} ({DOMAIN_HINT})" if domain_bias else query
    payload = {"q": q, "n": n, "papers_only": papers_only, "since": since}
    key = _key("search", payload)
    if use_cache and key in _load_cache():
        return _load_cache()[key]["result"]

    kw = dict(
        num_results=n,
        type="neural",
        contents={"highlights": {"num_sentences": 3, "query": query}},
    )
    if papers_only:
        kw["category"] = "research paper"
    if since:
        kw["start_published_date"] = since
    t0 = time.time()
    resp = _exa().search(q, **kw)
    rows = [_row(r, query) for r in resp.results]
    _put(key, "search", payload, rows, time.time() - t0)
    return rows


def fetch_contents(url: str, max_chars: int = 8000, use_cache: bool = True) -> dict:
    """Full text (truncated) + highlights for one URL."""
    payload = {"url": url, "max_chars": max_chars}
    key = _key("contents", payload)
    if use_cache and key in _load_cache():
        return _load_cache()[key]["result"]
    t0 = time.time()
    resp = _exa().get_contents([url], text={"max_characters": max_chars},
                               highlights={"num_sentences": 3})
    if not resp.results:
        out = {"url": url, "title": None, "published": None, "text": "", "highlights": []}
    else:
        r = resp.results[0]
        out = {**_row(r), "text": getattr(r, "text", "") or ""}
    _put(key, "contents", payload, out, time.time() - t0)
    return out


def format_for_prompt(rows: list[dict], max_hl: int = 2) -> str:
    """Compact citation block for an LLM prompt: [n] title (date) url + highlights."""
    out = []
    for i, r in enumerate(rows, 1):
        date = (r.get("published") or "")[:10]
        out.append(f"[{i}] {r.get('title') or '(untitled)'} ({date}) {r['url']}")
        for h in (r.get("highlights") or [])[:max_hl]:
            out.append(f"    - {' '.join(h.split())[:300]}")
    return "\n".join(out)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "contrastive learning spectra to molecular fingerprints candidate ranking"
    rows = search_methods(q, n=5)
    print(format_for_prompt(rows))
    if rows:
        c = fetch_contents(rows[0]["url"], max_chars=1500)
        print("\n--- contents of [1]:", c["url"], len(c["text"]), "chars")
        print(c["text"][:400])
