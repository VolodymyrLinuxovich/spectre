"""Live smoke tests. Run: .venv/bin/python -m tools.test_live [exa] [past]

exa : one cached-bypassing search + one contents fetch, prints sources.
past: setup identities/audiences, proposer-1 writes a claim, referee writes a verdict and a
      referee-only test score, then proposer-2 recalls (cross-agent read) and proposer-1
      probes the referee-only boundary. Prints timings.
"""
import sys
import time
import uuid

from .memory import PastDevMemory, summarize_for_prompt


def test_exa():
    from .exa_search import search_methods, fetch_contents, format_for_prompt
    t0 = time.time()
    rows = search_methods("learned spectrum embedding for candidate molecule retrieval", n=5,
                          use_cache=False)
    print(f"exa search: {len(rows)} hits in {time.time() - t0:.1f}s")
    print(format_for_prompt(rows))
    assert rows and rows[0]["url"], "no Exa results"
    assert any(r["highlights"] for r in rows), "no highlights returned"
    c = fetch_contents(rows[0]["url"], max_chars=2000, use_cache=False)
    print(f"contents: {len(c['text'])} chars from {c['url']}")
    assert c["text"], "empty contents"
    print("EXA OK")


def test_past():
    m = PastDevMemory(ns="test-")  # isolated from the swarm
    m.setup(n_proposers=3)
    eid = f"exp-live-{uuid.uuid4().hex[:6]}"
    idea = "add precursor m/z and adduct one-hot as extra inputs to the fingerprint MLP"
    a = m.record({"exp_id": eid, "kind": "claim", "idea": idea,
                  "metrics": {"val_hit@1_gain_claimed": "+1.4"}}, author="proposer-1")
    b = m.record({"exp_id": eid, "kind": "verdict", "idea": idea, "verdict": "NOISE",
                  "metrics": {"paired_gain": 0.3, "noise_band_2sd": 0.9},
                  "evidence": "gain inside 2 sigma seed band"}, author="referee")
    c = m.record({"exp_id": eid, "kind": "test_score",
                  "idea": f"SECRET-TEST {eid} hidden test hit@1 = 17.2"}, author="referee")
    print("accepted in ms:", a["accept_ms"], b["accept_ms"], c["accept_ms"])
    t0 = time.time()
    for r in (a, b, c):
        m.wait(r["ingestionId"])
    print(f"write->readable: {time.time() - t0:.1f}s for 3 sends")

    hits = m.recall("did adding precursor mass and adduct inputs help?", k=5, reader="proposer-2")
    print("proposer-2 recall:\n" + summarize_for_prompt(hits))
    assert hits, "no recall hits"
    assert any("proposer-1" in h.authors or "referee" in h.authors for h in hits)

    leak = m.recall(f"hidden test hit@1 score for {eid}", k=10, reader="proposer-1")
    assert not any("SECRET-TEST" in h.content for h in leak), "BOUNDARY LEAK"
    ref = m.recall(f"hidden test hit@1 score for {eid}", k=10, reader="referee")
    print("referee sees test score:", any("SECRET-TEST" in h.content for h in ref))
    print("stats:", dict(m.stats))
    print("PAST OK")


if __name__ == "__main__":
    which = sys.argv[1:] or ["exa", "past"]
    if "exa" in which:
        test_exa()
    if "past" in which:
        test_past()
