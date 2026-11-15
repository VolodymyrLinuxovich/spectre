# Sponsor challenges: exact asks

Sources (read 2026-11-15):
- Event page, Resources tab: https://iterate.inc/ai-research
- Exa brief PDF: https://drive.google.com/file/d/18wjMQsxnerTGfqOcsZMRkMvRNBkQ6Vyf/view
- past.dev brief PDF (10 pages): https://drive.google.com/file/d/1uM3EgLx18NqHrTaSAKzIReK9YKTSludC/view

## Deadlines (event page)
- 12:00 Sun: team list closes on the platform.
- 13:30: judging schedule on Discord.
- **14:00 "Hacking ends, submissions due"** (schedule). The "What to submit" box says **by Sunday 14:30**. Treat 14:00 as hard.
- 14:45 Round 1 (3:00 pitch + 2:00 Q&A per team, top 2 per track to finals). 16:30 finals (3 min, live demo, no Q&A). 18:00 awards.
- Submit: 1 track + challenges entered, GitHub repo link (public), **2-minute demo video** (YouTube/Drive/Loom), short description.
- Rules: built entirely during the event, **no prior commits to the repo**, credit open-source/APIs/models, all members on site.

## Overall judging (5 x 20%)
Research rigour (clear question, baselines/ablations, claims backed) · Novelty · Results (measurable, vs baseline, live demo) · Autonomy & self-improvement (extra credit if it improves from its own results) · Interpretability & reproducibility (logs, traces, sources, failures reported honestly).

## Track 3: AI Scientist Swarms (the track PLAN.md picks)
Question as written: "can a team of agents reproduce a full computational biology study from raw data better than the best single frontier model?" Build proposer / experimenter / critic agents that check each other before anything is reported. Record to beat: **BixBench3** (Edison Scientific), best 0.523. **Judged on: score vs the record and a single-agent baseline on the same compute budget, rigour, reproducibility. Report what the swarm rejected as well as what it found.**
Note: our benchmark is MassSpecGym, not BixBench3. The "single-agent baseline, same compute" comparison and "report what was rejected" map well onto the Referee. Lead should decide whether to pitch under Track 3 or Track 5 (Open, same criteria).

## Exa: "Research agents that search"
**Prompt:** build a research agent, AI scientist or self-improving system that uses Exa to find and work with information from the web. Strongest projects do something they couldn't from model knowledge alone: pull in data published last week, check a claim against real sources, or discover a method nobody told the agent about.

**Requirements**
1. At least one Exa API call (search, contents, find similar, answer, research or Websets) runs in the project's **main loop**, not only a side feature.
2. Submit a repo link and **point to where Exa is called** (file:line in README).
3. **Show your sources**: the demo makes clear what the system found through Exa and how it used it.
4. Built during the hack.

**Idea matching us (from the brief):** "AI scientists: an agent that proposes a hypothesis, then searches prior work to see whether it's already been tested before spending compute on it."

| Criterion | What they look for | Weight |
|---|---|---|
| Exa is central | Project would break or get clearly worse without search | 30% |
| Research value | Solves a real research problem or speeds up real research work | 25% |
| Creativity | An idea or use of search they haven't seen | 25% |
| Checkable output | Results trace back to real sources, works in a live demo | 20% |

Exa judges review every entry Sunday afternoon. Prizes $5K/$3K/$1K credits + merch.
**Proof to prepare:** an ablation "proposer with Exa vs without" (e.g. ideas grounded in a cited paper, promoted rate), and each proposal in events.jsonl carrying the Exa URLs it used, shown on the dashboard.

API docs: https://docs.exa.ai (search: `exa.search(query, type="auto"|"neural"|"fast", category="research paper", num_results, contents={"highlights":..., "text":...})`; `exa.get_contents(urls, ...)`; also find_similar, answer, research). Credits: dashboard.exa.ai code EXAITERATE50 ($50 per hacker).

## past.dev: "Multi-Agent Memory Challenge"
**Prompt:** add a shared memory layer to the main project: agents write facts and data points to past.dev and use the ones written by other agents.

**Requirements**
1. More than one agent **writes** to the memory layer.
2. More than one agent **reads** from it.
3. At least one agent reads data written by a **different** agent and uses it in its tasks.
4. May combine with markdown/RAG; README must say which technique holds which data.
5. Do not have to prove memory improves results; if it didn't, explain why in README.

**Important correction to PLAN.md:** the brief says "You do not have to measure your system. There are no metrics, no leaderboard." The "prove better accuracy, latency and cost vs common approaches" line is only on the Luma blurb. A baseline comparison is a nice extra (fits "Evidence"), not the scoring axis.

**Judging (1 to 5 each, highest total wins, tie broken by Evidence)**
| Criterion | Requirement |
|---|---|
| A real swarm | >1 agent, real writes and reads between agents. **Read boundary must stay correct when the judge tests it.** One agent with memory calls is not a swarm. |
| Memory design with reasons | Explain why each layer (working/episodic/semantic/procedural) and not another; say the result you expected before the run. |
| Evidence | A trace where one agent uses another's data, a read boundary holding in the demo, a failure with explanation. |
| Honesty | Caveats in README. A gap the judge finds that isn't listed lowers the score. |

**Required proofs / submission**
1. Select the past.dev challenge in the Iterate submission, repo link openable by the judge.
2. README section titled exactly **"past.dev challenge"** with four headings: **Agents** (per agent: what it writes, what it reads, its identity, its audiences), **Layers** (division + reason), **Results** (what you changed, what you observed), **Caveats**.
3. Round 1 demo: **show one read in which an agent uses data from a different agent.**
4. Demo should use `artifact.supersededAt` and `sources[].excerpts` to show which agent wrote a fact and what replaced it.
5. Use identities (one per agent) + audiences (one per read boundary); **always send identity on recall** (without it you get the whole project). Show isolation.
6. Optional but recommended: delete a source and show the answer changes.
7. Dataset: dated, small (tens of docs/day), synthetic or public only. Hypotheses vs facts, rules, deadlines "give the clearest results".

Design topics they suggest: consolidation (when an episode becomes a fact), ownership (private + publish conclusions vs all shared + scoped reads), deletion (agent B finds agent A's error).
Fit for Spectre: Referee is the authority that writes verdicts; proposers write hypotheses; a later PROMOTED/NOISE verdict supersedes the proposer's claim. Audience "verdicts" readable by all proposers; per-proposer private scratch audience; test-set scores in a "referee-only" audience that proposers can't read (read boundary the judge can test, and it matches our "no test labels" thesis).

Limits: 1 project, 500 req/min, 8,000 tokens per recall. Ingest is async: poll until `completed` before recall (can take seconds to minutes).
Credits: sign up at past.dev, select 500,000 credits, code SFAIHACK500, create project key. Judge Mehdi Djabri (CEO), mentor on site Lazare Masset.

API (from https://past.dev/openapi.json and https://past.dev/docs/memory-api/setup):
- Base `https://api.past.dev/api/v1`, `Authorization: Bearer past_sk_...`
- `POST /ingest` {content, timestamp, id, label, metadata, identity (author), audience} -> 202 {ingestionId, sourceId, status}
- `POST /ingest/batch` {idempotencyKey, items[<=1000]}; `GET /ingest/{ingestionId}` poll status processing|completed|failed
- `POST /recall` {query, identity (reader, required), limit, queryTimestamp, occurredFrom/To, level} -> {asOf, usedEvidenceTokens, results[{rank, occurredAt, content, artifact{kind, supersededAt}, sources[{sourceId, label, metadata, excerpts}]}]}
- `PUT /audiences/{slug}` {kind: fixed|rule, name, identities[]}; `POST /identities/bulk`; `GET /identities/{id}` lists reach
- `DELETE /data-points/{sourceId}`; `GET /usage` (credits)
- Pricing: 1 credit per 350 bytes ingested; recall 1 credit on pay-as-you-go.

## Modal: "Best use of Modal"
Best use of Modal for training, inference, or running large numbers of experiments and agent sandboxes in parallel. Open to any team using Modal. $5,000 credits. No further brief published. Credits form: https://modal.fillout.com/t/qMXCmRGseUus?code=ASJ-QQN-PQX

## Track 1: Build Your Own Autoresearch Framework (verbatim from event page)
**The question:** can your system beat a best-known result in mathematics on its own?
**What to build:** a system that runs the research loop by itself: it picks what to try, runs it, reads the result and decides what to try next, until it beats a record. Karpathy's autoresearch (March 2026) showed the minimal version: an agent edits a single training script, runs 5-minute experiments and keeps only the changes that lower val_bpb. Take it further, for example with better search over ideas, memory of what failed, multi-objective evaluation, or a new domain.
**Records to beat:** any problem on EinsteinArena (public verifier + live leaderboard), any constant in Terence Tao's table of optimization constants (https://teorth.github.io/optimizationproblems), or any of the 67 problems in the AlphaEvolve repository of problems (https://github.com/google-deepmind/alphaevolve_repository_of_problems). Improving a known bound is welcome.
**Judged on:** how far past the record you get, how new your loop is, and whether we can follow each decision (show the log).

## Track 5: Open Track (verbatim)
Your idea doesn't fit tracks 1-4? Build whatever you want. Same rules: built during the hackathon, public repo, demo, same judging criteria. Sponsor challenges (Exa, past.dev, Modal) are open to this track too.
(No record to beat; judged only on the 5 overall criteria above: rigour, novelty, results, autonomy, interpretability/reproducibility.)
