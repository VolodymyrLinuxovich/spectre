<div align="center">

# Spectre

**An autoresearch swarm that can't fool itself.** Proposer agents try to improve a model that identifies molecules
from mass spectra. **Exa** finds prior work, **past.dev** holds what the swarm already tried, and a **Referee** decides
what is real. The one rule: **a claim becomes a fact only when the Referee promotes it**, and the Referee is the only
process that ever sees labels.

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Modal](https://img.shields.io/badge/Modal-7FEE64?style=for-the-badge&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-0A5C8A?style=for-the-badge&logoColor=white)
![Exa](https://img.shields.io/badge/Exa-1F40ED?style=for-the-badge&logoColor=white)
![past.dev](https://img.shields.io/badge/past.dev-111111?style=for-the-badge&logoColor=white)

</div>

---

SF AI Research Hackathon, November 14 and 15, 2026. Sponsor challenges entered: Exa, past.dev, Modal.

| | |
|---|---|
| **The task** | [MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) retrieval. Given an MS/MS spectrum, rank candidate molecules with the same formula. Scored by MRR@20 and hit@1, 5, 20 |
| **The trap** | In the raw candidate lists the true molecule is first for **4,000 of 4,000** val spectra and 4,000 of 4,000 test spectra. An agent that "discovers" this gets a perfect score |
| **Try it** | `python3 ui/server.py`, then open http://localhost:8765 to watch the swarm. `SPECTRE_SAMPLE=1` shows sample data |
| **Pitch** | [PITCH.md](PITCH.md) |
| **Sponsor briefs** | [tools/SPONSORS.md](tools/SPONSORS.md) |

---

## The Rule

A proposer's claim is worth nothing until the Referee promotes it.

- **Proposers never see labels.** Their code runs in a Modal Sandbox with no network. Only training data and unlabeled
  val and test inputs are mounted. `val.parquet` and `test.parquet` never leave the Referee's volume.
- **Every claim is checked in order**, and the first failed check is the verdict:
  1. **CRASH.** Every sandbox run must finish.
  2. **NOISE.** The gain over the incumbent is measured on 3 seeds. It must beat twice the incumbent's seed spread,
     its bootstrap 95% CI must stay above zero, and it must be positive on every seed.
  3. **LEAK.** Retrain with the training labels shuffled. If half the gain survives, it was not learned.
  4. **PRIOR.** Blank every spectrum at inference. If half the gain survives, the model is not reading the spectrum.
  5. **OVERFIT.** Only now is the hidden test fold scored, again on 3 seeds. Its CI must stay above zero.
- **Only PROMOTED code becomes the new incumbent**, so the system improves from its own verified results.
- **Proposers see verdicts, never test numbers.** `Referee.public_view` strips them, and past.dev keeps test scores in
  an audience no proposer can read.
- **Candidate lists are reshuffled per spectrum** with a fixed seed, so their order carries no label signal.

### The math behind each verdict

Spectrum $i$ has $n_i$ candidate molecules. A model returns a ranking, and $r_i$ is the 0-based position of the true
molecule in its top 20. The score is truncated mean reciprocal rank:

$$
\mathrm{RR}_i = \begin{cases} \dfrac{1}{r_i + 1} & r_i < 20 \\[4pt] 0 & \text{otherwise} \end{cases}
\qquad
\mathrm{MRR@20} = \frac{1}{N}\sum_{i=1}^{N} \mathrm{RR}_i
$$

Guessing at random gives $\mathbb{E}[\text{hit@1}] = \frac{1}{N}\sum_i \frac{1}{n_i} = 0.0377$ on our val split.

**Paired gain.** The candidate (c) and the incumbent (b) are run on the same spectra with the same seeds
$s \in \{0, 1, 2\}$. For each spectrum we average over seeds, then take the difference:

$$
d_i = \frac{1}{S}\sum_{s=1}^{S}\left(\mathrm{RR}^{c}_{i,s} - \mathrm{RR}^{b}_{i,s}\right)
\qquad
\Delta = \frac{1}{N}\sum_{i=1}^{N} d_i
$$

**NOISE** unless all three hold:

$$
\underbrace{\Delta > 2\,\sigma_{\text{seed}}}_{\text{beats retraining noise}}
\qquad
\underbrace{\mathrm{CI}^{95\%}_{\text{lo}}(\Delta) > 0}_{\text{beats spectrum sampling noise}}
\qquad
\underbrace{\Delta_s > 0 \;\; \forall s}_{\text{holds on every seed}}
$$

where $\sigma_{\text{seed}}$ is the standard deviation of the incumbent's MRR@20 across seeds, and the CI comes from a
paired bootstrap over spectra ($B = 2000$ resamples of $\{d_i\}$ with replacement, 2.5th and 97.5th percentiles).
The two noise sources are different. Seed noise is what you get by retraining the same code. Sampling noise is what
you get from which 4,000 spectra happen to be in val. A real gain has to beat both.

**LEAK** and **PRIOR** use canaries. Each one removes the thing a real gain must come from, then asks how much gain is
left:

$$
\Delta_{\text{shuffle}} = \mathrm{MRR}^{c}_{\text{shuffled labels}} - \mathrm{MRR}^{b}_{\text{shuffled labels}}
\qquad
\Delta_{\text{ablate}} = \mathrm{MRR}^{c}_{\text{spectra} = 0} - \mathrm{MRR}^{b}_{\text{spectra} = 0}
$$

$$
\text{LEAK if } \Delta_{\text{shuffle}} \ge \tfrac{1}{2}\Delta
\qquad
\text{PRIOR if } \Delta_{\text{ablate}} \ge \tfrac{1}{2}\Delta
$$

With shuffled labels nothing can be learned, so any surviving gain was smuggled in. With blank spectra the model can
only use precursor mass, adduct and candidate structure, so any surviving gain does not come from reading the spectrum.

**OVERFIT** unless the same paired bootstrap on the hidden test fold gives $\mathrm{CI}^{95\%}_{\text{lo}}(\Delta_{\text{test}}) > 0$.
Proposers never see $\Delta_{\text{test}}$, only the verdict, so they cannot tune against it.

---

## Architecture at a Glance

```mermaid
flowchart TD
    P["Proposer agents 1 to 4<br/>Qwen3 Coder served on Modal"]
    M[("past.dev<br/>claims, verdicts, private notes")]
    E["Exa<br/>research paper search"]
    D{"implement or skip?"}
    S["Modal Sandbox<br/>no network, no labels"]
    R["Referee<br/>sole holder of labels"]
    INC["Incumbent model.py"]
    LOG["runs/events.jsonl"]
    UI["Live dashboard<br/>ui/"]

    P -->|"what did the swarm try?"| M
    P -->|"has this been done?"| E
    M --> D
    E --> D
    D -.->|"skip, no compute spent"| M
    D -->|"new model.py"| R
    R -->|"runs seeds and canaries"| S
    S -->|"ranked candidates"| R
    R -->|"verdict with evidence"| M
    R -->|PROMOTED| INC
    INC --> P
    R --> LOG
    P --> LOG
    LOG --> UI
```

### What happens to one claim

```mermaid
sequenceDiagram
    participant P as Proposer
    participant X as Exa
    participant M as past.dev
    participant R as Referee
    participant S as Sandbox

    P->>M: recall "experiments on my focus" as proposer-i
    M-->>P: verdicts from the Referee, claims from other proposers
    P->>P: hypothesis and search query
    P->>X: search_methods(query)
    X-->>P: 5 papers with highlights
    P->>P: implement or skip, cite sources by number
    P->>M: claim, with citations
    P->>R: judge(model.py)
    R->>S: 3 seeds, plus label shuffle and spectrum ablation
    S-->>R: ranked candidates on val
    R->>R: noise band and CI, LEAK, PRIOR
    R->>S: 3 seeds on hidden test, only if all passed
    R-->>M: verdict for everyone, test score for referee only
    R-->>P: verdict without test numbers
```

---

## See It Run

<div align="center">

TODO: `docs/demo.gif` of the dashboard during a swarm run.

</div>

---

## Who Does Each Step

| Step | Who does it | Why |
|------|-------------|-----|
| Propose a change | **Proposer agent**, an open weight coder model on Modal | Each proposer has its own focus, so the swarm covers more ideas |
| Check what the swarm already tried | **past.dev** recall as that proposer's identity | Verdicts from other agents stop repeats before any compute is spent |
| Check prior work | **Exa** neural search over research papers | Finds methods the model doesn't know and whether an idea was already tested |
| Decide to implement or skip | **Proposer**, with sources cited by number | A skip costs nothing and is written to memory with its reason |
| Run the code | **Modal Sandbox**, no network, 16 CPU or GPU, hard timeout | Agent code can't reach labels or the internet |
| Decide if the gain is real | **Referee**, plain code | It holds the labels. Nothing to gain from flexible judgment here |
| Show what happened | **Dashboard** in `ui/`, `runs/events.jsonl` | Every hypothesis, search, memory read, claim and verdict is one event |

---

## Results

### Data checks

From `runs/prep.log`. 4,000 val and 4,000 test spectra, fixed with seed 0.

| | Val | Test |
|---|---|---|
| True molecule first in raw candidate lists | 4,000 of 4,000 | 4,000 of 4,000 |
| True molecule first after our per spectrum shuffle | 4.2% | 3.1% |
| Random hit@1 | 3.8% | 3.0% |
| Mean candidates per spectrum | 160 | 166 |

### Baseline and a hand written candidate

Seed 0 on val, 16 CPU sandbox. From `bench/candidates/honest.py` and `runs/honest_sweep.log`.

| Model | hit@1 | hit@5 | hit@20 | Train time |
|-------|-------|-------|--------|------------|
| Random ranking | 0.0395 | 0.1175 | 0.2935 | |
| Baseline MLP, binned spectrum to fingerprint | 0.1003 | 0.2298 | 0.4195 | 576 s |
| Sparse first layer | 0.1070 | 0.2295 | 0.4250 | 520 s |
| Sparse first layer and neutral losses | 0.1145 | 0.2378 | 0.4285 | 456 s |

These are single seed numbers. The Referee decides whether any of them is a real gain.

### Planted fake gains

`bench/redteam/` holds two fakes built from the baseline. `noise.py` only changes the hidden width, so any gain is seed
noise. `prior.py` boosts candidates that look like training molecules without reading the spectrum.

| Fake | Expected verdict | Referee verdict |
|------|------------------|-----------------|
| `noise.py` | NOISE | [VERDICT] |
| `prior.py` | PRIOR | [VERDICT] |
| `honest.py` | PROMOTED | [VERDICT] |

### Swarm

| Run | Command | Claims | Promoted | Accepted claim precision |
|-----|---------|--------|----------|--------------------------|
| Naive loop, no Referee | `--naive` | [NUMBER] | [NUMBER] | [NUMBER] |
| Swarm, 4 proposers | default | [NUMBER] | [NUMBER] | [NUMBER] |
| Single agent, same budget | `--proposers 1 --rounds 12` | [NUMBER] | [NUMBER] | [NUMBER] |
| Swarm without Exa | `--no-exa` | [NUMBER] | [NUMBER] | [NUMBER] |
| Swarm without memory | `--memory none` | [NUMBER] | [NUMBER] | [NUMBER] |

### Shared memory benchmark

`tools/memory_bench.py` runs a small synthetic, dated swarm log against five stores. It has 39 data points: 12 ideas,
each with a proposer claim and a Referee verdict, 4 later verdicts that overturn earlier ones, and 12 hidden test
scores. A different proposer asks about each idea in its own words.

| Store | Current verdict ranked first (of 12) | Stale fact ranked first | Test score leaks | Recall latency |
|---|---|---|---|---|
| Shared file, keyword match | 3 | 6 | 0 | under 1 ms |
| Shared file, keyword match, hand written "latest verdict wins" | 7 | 0 | 0 | under 1 ms |
| Embedding store | 0 | 12 | 0 | about 520 ms |
| Shared file without read scoping | 3 | 6 | 30 | under 1 ms |
| past.dev | 7 | 5 | 0 | about 430 ms |

---

## Run It

Needs Python 3.11, a Modal account, and keys in `.env`: `EXA_API_KEY`, `PAST_API_KEY`, `SPECTRE_LLM_URL`,
`SPECTRE_LLM_API_KEY`.

```bash
python -m venv .venv && .venv/bin/pip install modal exa-py requests openai numpy pandas pyarrow python-dotenv

modal run bench/prep.py                       # MassSpecGym to Modal Volume spectre-data
modal run bench/sandbox.py::sync_inputs       # copy only unlabeled inputs to spectre-inputs
modal deploy llm/serve.py                     # open weight coder model, OpenAI compatible

.venv/bin/python -m agents.swarm --proposers 4 --rounds 3      # full system
.venv/bin/python -m agents.swarm --naive                       # unguarded loop
.venv/bin/python -m agents.swarm --proposers 1 --rounds 12 --tag single
.venv/bin/python -m agents.swarm --no-exa                      # or --memory local, --memory none

python3 ui/server.py                          # dashboard on http://localhost:8765
```

Judge files by hand and run the sponsor tool checks:

```bash
.venv/bin/python -m bench.judge_many seeded bench/redteam/noise.py:"wider layer" bench/redteam/prior.py:"prior boost"
.venv/bin/python -m tools.test_live           # live Exa search and past.dev cross agent read
.venv/bin/python -m tools.memory_bench --backends local,local-sup,local-embed,local-unscoped,pastdev
```

---

## Code Map

| Path | What |
|------|------|
| `bench/prep.py` | Downloads MassSpecGym, builds train, val and test, candidate fingerprints, reshuffles candidate order |
| `bench/sandbox.py` | Runs agent code in a Modal Sandbox: no network, no labels, timeouts, label shuffle and spectrum ablation switches |
| `bench/referee.py` | **The Referee.** Seeds, bootstrap CI, noise band, LEAK and PRIOR canaries, hidden test, promotion |
| `bench/baseline/model.py` | Baseline: binned spectrum to Morgan fingerprint MLP, cosine ranking |
| `bench/candidates/honest.py` | Hand written candidate with neutral losses and a sparse first layer |
| `bench/redteam/` | Planted fake gains the Referee must reject |
| `bench/judge_many.py` | Judge several files against the incumbent in one process |
| `agents/swarm.py` | Proposers: memory read, hypothesis, Exa search, implement or skip, claim, Referee |
| `llm/` | vLLM server on Modal and a smoke test |
| `tools/exa_search.py` | Exa search and contents, cached in `runs/exa_cache.jsonl` |
| `tools/memory.py` | past.dev memory with identities and audiences, plus a local baseline with the same interface |
| `tools/memory_bench.py` | Memory benchmark and its synthetic data generator |
| `tools/test_live.py` | Live checks for Exa and past.dev |
| `ui/` | Dashboard server and page |

---

## past.dev challenge

### Agents

All agents write to one past.dev project. Each reads with its own identity, and audiences decide what it sees.

| Agent | Identity | Writes | Reads | Audiences |
|---|---|---|---|---|
| Proposer i (1 to 4) | `proposer-i` | Claims to `verdicts`. Notes on skipped ideas to its own scratch | Before each hypothesis, "experiments on my focus" | `verdicts`, `proposer-i-scratch` |
| Referee | `referee` | Verdicts with evidence to `verdicts`. Hidden test scores to `referee-only` | Everything | `verdicts`, `referee-only` |
| Orchestrator | `orchestrator` | Nothing yet | Dashboard and audits | `verdicts`, `referee-only` |

The cross agent read is at `agents/swarm.py:110`. A proposer reads the Referee's verdict on an idea another proposer
tried, then skips or builds on it at `agents/swarm.py:168`. The read boundary a judge can test: ask as any
`proposer-i` for a test score and nothing comes back.

### Layers

| Layer | What it holds | Where | Why |
|---|---|---|---|
| Working | Hypothesis, incumbent code, this step's Exa results | The prompt | Needed for one step only |
| Episodic | One data point per claim, verdict, skip note or test score, with time and author | past.dev data points | Proposers and the Referee disagree here, and both sides must stay visible with dates |
| Semantic | Facts past.dev extracts, such as "the referee classified idea X as noise" | past.dev memories | A proposer reads a short fact instead of the raw log |
| Procedural | Sandbox rules and how to write a valid `model.py` | Prompt text in `agents/swarm.py` | Fixed for the run |

`runs/events.jsonl` is the event log for people and the dashboard. `runs/exa_cache.jsonl` caches Exa calls.
Every agent writes to the shared store and scoping controls reads. We chose this so a claim stays visible next to the
verdict that overturns it.

Expected result, written before the run on 2026-11-15 at 06:45 PT: past.dev returns the current verdict more often
than an embedding store, roughly ties a hand written supersede rule, is slower to write and read than a local file,
keeps the test score boundary, and lowers repeated failed ideas in the swarm.

### Results

- **Live test.** `proposer-2` asked about `proposer-1`'s idea and got the claim and the Referee's NOISE verdict, verdict
  first. `proposer-1` asked for the hidden test score and got nothing. `referee` got it.
- **Latency.** Writes were accepted in 150 to 190 ms. A single write became readable after about 30 s. A burst of 39
  writes took 266 s until all were readable. Recall took about 430 ms.
- **Benchmark.** See the table above. past.dev tied the hand written rule at 7 of 12 and beat the embedding store,
  which always returned the stale claim. The right experiment was in the top 5 for 12 of 12 questions. In the 5
  misses the claim ranked above the verdict and `supersededAt` was never set. The run cost about 51 credits.
- **Swarm.** [TRACE: one read where a proposer uses another agent's verdict, with authors and excerpt. Repeated
  failed ideas with memory: [NUMBER], without: [NUMBER].]

### Caveats

- Writes take about 30 s to become readable, and minutes in a burst. Proposers in the next round can read stale memory.
- past.dev never marked a claim as superseded in our benchmark. Proposers must read the verdict tag, not trust rank 1.
- `setup(n)` replaces each audience. Calling it with a smaller `n` removes proposers from `verdicts`.
- The stats counter in `PastDevMemory` is not thread safe. Totals can be slightly off, results are not.
- The benchmark is small, synthetic and its questions were written by us.
- The benchmark and live test use `bench-` and `test-` identities in the same project. The swarm can't read them but
  they share its credits and ingest queue.

---

## Exa

Exa runs in the main loop of every proposer, once per step.

- `agents/swarm.py:103` `Proposer.step` is the loop.
- `agents/swarm.py:139` to `141` call `exa_search.search_methods` with the proposer's own query and format a numbered
  source list.
- `agents/swarm.py:163` to `173` make the proposer decide to implement or skip, citing sources by number. A skip
  spends no compute and is written to memory with its citations.
- `tools/exa_search.py:86` `search_methods` uses neural search, category research paper, and query specific
  highlights. `:115` `fetch_contents` returns full text. `:133` formats the citation list.

Sources show on the dashboard next to each experiment, with cited ones tagged (`ui/index.html:462`). They are saved in
every past.dev claim, skip note and verdict as "Sources found via Exa", so another proposer sees which papers an idea
came from. In the live test Exa returned JESTR, CSU-MS2, DreaMS and a 2026 review for a spectrum embedding query.

Exa ablation: [NUMBER] skips before compute with Exa, [NUMBER] without. Promoted rate [NUMBER] vs [NUMBER].

---

## Limits

- **One benchmark.** Everything is measured on MassSpecGym retrieval. The Referee is benchmark agnostic in design but
  has only been run here.
- **The canaries are necessary, not sufficient.** A leak that survives label shuffling, such as one through candidate
  structure, would pass LEAK. Reshuffling candidates removes the one we know about.
- **Small real gains get rejected.** A gain smaller than the noise band is called NOISE. We accept that on purpose.
- **Val is reused across rounds.** The hidden test catches overfitting, but only for claims that reach it.
- **Compute bound.** A full verdict takes up to 16 sandbox runs for the claim and the incumbent (incumbent runs are
  cached). Val and test are fixed 4,000 spectrum subsets, not the full folds.
- **GPU access.** Our Modal account could not start A10G functions without a payment method, so runs used 16 CPU
  sandboxes.
