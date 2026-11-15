# past.dev challenge

## Agents

All agents write to one past.dev project. Each agent reads with its own identity. The audiences decide what each one can see. Setup is `PastDevMemory.setup(n)` in `tools/memory.py:168`.

| Agent | Identity | Writes | Reads | Audiences it reaches |
|---|---|---|---|---|
| Proposer i (1 to 4) | `proposer-i` | Claims about its experiment to `verdicts`. Private notes on skipped ideas to `proposer-i-scratch`. | Before each hypothesis it recalls "experiments on <its focus>". It sees claims and verdicts from every agent and only its own notes. | `verdicts`, `proposer-i-scratch` |
| Referee | `referee` | A verdict for each experiment (PROMOTED, NOISE, LEAK, PRIOR or CRASH) with evidence to `verdicts`. Hidden test scores to `referee-only`. | Can read everything, including test scores. | `verdicts`, `referee-only` |
| Orchestrator | `orchestrator` | Nothing yet. | Reads for the dashboard and audits. | `verdicts`, `referee-only` |

The proposers never hold test labels. The `referee-only` audience keeps them from ever seeing test scores, even through memory. This is the same rule the Referee enforces for compute. It is also the read boundary a judge can test. Ask as any `proposer-i` for a test score and nothing comes back.

The cross-agent read happens in `agents/swarm.py:110`. Proposer B reads the Referee's verdict on an idea that proposer A tried. It then skips that idea or builds on it. The decision is at `agents/swarm.py:168`.

Routing lives in `default_audience` at `tools/memory.py:63`. Each record is rendered as plain sentences that name the author, the experiment id and whether it is a claim or a verdict (`tools/memory.py:76`). Author, kind, verdict and exp id also go in `metadata`. Recall results show who wrote each fact.

## Layers

| Layer | What it holds | Where it lives | Why |
|---|---|---|---|
| Working | The current hypothesis, the incumbent `model.py`, Exa results for this step | The proposer's prompt | It is only needed for one step. |
| Episodic | One data point per event: a claim, a verdict, a skip note, a test score. Each has a timestamp and an author. | past.dev data points | Agents disagree here. A proposer claims a gain and the Referee later rules on it. past.dev keeps both with dates. |
| Semantic | Facts past.dev extracts, such as "the referee classified idea X as noise" | past.dev memories | We do not consolidate by hand. past.dev does it at write time, so a proposer reads a short fact instead of the raw log. |
| Procedural | How to write a valid `model.py`, the sandbox rules | Prompt text in `agents/swarm.py:46` | It is fixed for the run and does not change between agents. |

Other stores, and what each holds:
* `runs/events.jsonl` is an append-only event log. The dashboard reads it. It is for people, not for agents.
* `runs/exa_cache.jsonl` caches every Exa call so runs can be replayed.
* `runs/memory_log.jsonl` is our copy of every data point we sent to past.dev. past.dev has no route to list data points.

Ownership design. Every agent writes to the shared store and scoping controls the reads. We chose this over private stores with published conclusions. A proposer's claim must stay visible next to the verdict that overturns it. That contrast is what the Referee exists to show.

Supersession. A claim and its verdict share an experiment id and land in the same audience. The verdict is written later, so it supersedes the claim.

### Expected result, written before the run (2026-11-15, 06:45 PT)

1. On our memory benchmark, past.dev will return the current verdict more often than a plain embedding store. The embedding store will often return the original claim, because the claim text is closest to the question.
2. past.dev will roughly match a local store with a hand-written "latest verdict wins" rule. It will not clearly beat it on this small log.
3. Writes will be slower than a local file. Ingest is asynchronous, so a write becomes readable seconds later, not instantly. Recall will take hundreds of ms instead of under 1 ms.
4. The `referee-only` boundary will hold for every proposer.
5. In the swarm, proposers with memory will repeat fewer ideas that already got a NOISE, LEAK or PRIOR verdict than proposers without memory.

## Results

### Live API test (`tools/test_live.py`, 2026-11-15)

* Cross-agent read works. `proposer-2` asked "did adding precursor mass and adduct inputs help?" It got `proposer-1`'s claim and the Referee's NOISE verdict for the same experiment. The verdict ranked first. The claim came back marked as an unverified claim.
* past.dev also returned its own extracted facts. One read "the referee classified the result as noise". Authors stayed attached through metadata.
* The read boundary holds. `proposer-1` asked for the hidden test score and got nothing. `referee` asked the same question and got it.
* Latency. Each write was accepted in 150 to 190 ms. All 3 writes were readable 29 s later. One recall took about 440 ms.

### Memory benchmark (`tools/memory_bench.py`)

The log is synthetic and dated, and its generator is in the same file. It has 39 data points. There are 12 ideas, each with a proposer claim and a Referee verdict. 4 ideas get a second verdict that overturns the first, for example NOISE at 1 seed and then PROMOTED at 3 seeds. There are also 12 hidden test scores. For each idea, a different proposer asks a paraphrased question. We score whether the top result carries the current verdict.

| Store | Current verdict at top 1 (of 12) | Stale fact at top 1 | Test score leaks to proposers | Recall latency | Evidence tokens per question |
|---|---|---|---|---|---|
| Shared file, keyword match | 0.25 | 6 | 0 | under 1 ms | about 370 |
| Shared file, keyword match, hand-written supersede rule | 0.58 | 0 | 0 | under 1 ms | about 190 |
| Embedding store (OpenAI text-embedding-3-small) | 0.00 | 12 | 0 | about 340 ms | about 390 |
| Shared file with no read scoping | 0.25 | 6 | 30 | under 1 ms | about 310 |
| past.dev | 0.58 | 5 | 0 | about 430 ms | about 460 |

The 39 past.dev writes were all accepted in 7.3 s. They were all readable 266 s later. The run cost about 51 credits.

What past.dev did. The right experiment was in the top 5 for all 12 questions. 7 of 12 had the current verdict on top, which ties the hand-written rule. All 12 top results came from an agent other than the reader. The 5 misses all had the proposer's claim on top, above the verdict. None of those claims carried `supersededAt`. 2 of the 4 overturned verdicts came back current. On this log, past.dev did not mark a claim as superseded by the later verdict.

Against our expected result. Points 1, 3 and 4 held. Point 2 held. past.dev tied the hand-written rule and did not beat it. We expected writes to be readable within seconds, but they took minutes when 39 were sent at once. Sends in one project are processed one after another, as the docs say.

The embedding store always returned the claim and never the verdict. This is the "old fact next to the correction" failure from the challenge brief. Without scoping, the file store leaked test scores to every proposer.

### Swarm run

TODO(lead): trace of one read where a proposer uses another agent's verdict. Show `authors`, `superseded_at` and the excerpt. Also give repeated-idea counts with and without memory (`--memory none`).

## Caveats

* Ingest is asynchronous. A single write became readable about 30 s after it was accepted. A burst of 39 writes took 266 s to all become readable. Within one round, other proposers often do not see a verdict that was just written. Our swarm steps are slower than that, so the effect is small. A faster loop would read stale memory.
* `setup(n)` replaces each audience. Running it with a smaller `n` removes proposers from `verdicts`. Always call it with the real proposer count.
* The stats counter in `PastDevMemory` is not thread safe. Proposers run in threads, so latency and token totals can be slightly off. Results are not affected.
* The benchmark is small and synthetic, with 39 data points. It shows behavior. It is not a measure of accuracy at scale.
* The benchmark questions were written by us. Keyword search scores low partly because the questions are paraphrases.
* Supersession in past.dev is decided by its fact extraction, not by our experiment id. In the benchmark it never set `supersededAt` on a claim, and 5 of 12 questions got the stale claim on top. A proposer must still read the verdict tag rather than trust rank 1. `summarize_for_prompt` puts the tag first for that reason.
* The benchmark and the live test use namespaced identities and audiences (`bench-`, `test-`) in the same project as the swarm. The swarm cannot read them. They do share the project's credits and ingest queue.
* We deleted the 3 data points from the first live test so the swarm would not treat them as real experiments.
* past.dev allows one project and 8,000 recall tokens on this plan. We did not hit either limit.

# Exa

Exa runs in the main loop of every proposer, once per step. The swarm is a set of proposers. Each one proposes a change to the model, searches prior work, decides whether to spend compute, and sends the change to the Referee.

Where Exa is called:
* `agents/swarm.py:103` is `Proposer.step`, the main loop.
* `agents/swarm.py:139` to `141`. After the proposer writes a hypothesis and a search query, it calls `exa_search.search_methods(query, n=5)`. The results are formatted as a numbered citation list.
* `agents/swarm.py:163`. The proposer gets the sources and the swarm memory. It must answer `implement` or `skip` and list the sources it used by number.
* `agents/swarm.py:167` to `173`. The cited sources are kept. If prior work shows the idea already failed, or the incumbent already does it, the proposer skips it. It spends no compute and writes a note with the citations.
* `tools/exa_search.py:86` is `search_methods`. It uses neural search with category `research paper`, a short MS/MS metabolomics hint added to the query, and query-specific highlights.
* `tools/exa_search.py:115` is `fetch_contents`. It returns the full text of one paper.
* `tools/exa_search.py:133` is `format_for_prompt`. It builds the `[n] title (date) url` list with highlights that goes into the prompt.
* Every call is cached in `runs/exa_cache.jsonl` with its latency (`tools/exa_search.py:20`).

What Exa adds. The model knows older MS/MS methods. Exa finds recent ones, such as JESTR, CSU-MS2, DreaMS and 2026 work, and lets a proposer check whether an idea was already tested before it runs. Without search, the proposer decides from model knowledge alone. The `--no-exa` ablation turns search off with everything else the same.

How sources are shown:
* Dashboard (`ui/index.html:462`). Each experiment card lists the Exa results with title, domain and year. Results the proposer cited are tagged "cited". The query is shown under the list.
* past.dev. Claims, skip notes and verdicts carry a `citations` field (`agents/swarm.py:173`, `180`, `204`). `render` in `tools/memory.py:76` writes it as "Sources found via Exa". Another proposer recalling that experiment sees which papers it was based on.
* Event log. `exa_search` events in `runs/events.jsonl` keep the query and every result. Claim and skip events keep the citations used.

TODO(lead): Exa ablation numbers, with and without Exa. Include skips before compute, verdict mix, and the promoted rate.
