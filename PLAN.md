# Spectre: an autoresearch swarm that can't fool itself

SF AI Research Hackathon, Day 2. **Hard deadline 12:00 PT.** Track: AI Scientist Swarms.
Sponsor challenges: Exa, past.dev, Modal (all three).

## Problem
MS/MS spectrum -> molecular structure. Most spectra in a metabolomics run never get a structure.
Benchmark: **MassSpecGym** retrieval (HF `roman-bushuiev/MassSpecGym`), MCES-disjoint folds,
formula candidates (`data/molecules/MassSpecGym_retrieval_candidates_formula.json`), metric hit@1/5/20.

Autoresearch agents on this task produce fake gains (learned the hard way on a 5k-team Kaggle comp):
run-to-run noise bigger than most "improvements", leaked validation (memorized structures),
priors that score without reading the spectrum, inflated reported scores.

**Thesis:** the bottleneck for AI scientists is not ideas, it's an evaluator they can't game.
We build the Referee, then let a swarm loose under it.

## Architecture
```
Proposer agents (open-weight LLM served on Modal, OpenAI-compatible)
   | Exa: find methods they don't know      | past.dev: shared memory of tried ideas + verdicts
   v
patch to bench/baseline/model.py  ->  Modal Sandbox (GPU, no network, NO test labels mounted)
   -> writes val + test predictions (ranked candidate lists)
   v
Referee (Modal function, sole holder of labels)
   1. multi-seed: paired gain vs baseline must exceed noise band (2 sigma of seed std)
   2. spectrum-ablation canary: zero the spectra at inference; if hit@1 stays well above
      the prior-only baseline, the gain is not from the spectrum
   3. label-shuffle canary: retrain with shuffled train labels; must collapse to chance
   4. hidden holdout: agents only ever see val; test is scored only on promotion
   verdict: PROMOTED | NOISE | LEAK | PRIOR | CRASH  (+ evidence)
```

## Contract (do not change without telling everyone)
- Modal app names: `spectre-data` (prep), `spectre-referee`, `spectre-llm`, `spectre-sandbox`.
- Modal Volume `spectre-data`:
  - `/train.parquet` : id, mzs(list[float]), intensities(list[float]), precursor_mz, adduct, formula, smiles, inchikey14
  - `/val.parquet`, `/test.parquet` : same columns, used ONLY by referee
  - `/val_inputs.parquet`, `/test_inputs.parquet` : no smiles/inchikey (what sandboxes see)
  - `/cands/{val,test}.parquet` : id, cand_smiles(list[str]), cand_fp (packed Morgan r2 2048 per cand)
  - Subsets for speed: val = fixed 4000 spectra, test = fixed 4000 spectra (seed 0).
- Candidate model interface (`bench/baseline/model.py`):
  `train(train_df, seed) -> state` ; `rank(state, inputs_df, cands_df) -> dict[id, list[cand_smiles ranked]]`
- Referee API: `score(patch_dir_or_code: str, seeds=[0,1,2]) -> Verdict JSON` (schema in bench/verdict.py)
- LLM: Modal vLLM endpoint, OpenAI-compatible; base_url in `.env` as `SPECTRE_LLM_URL`. Fallback OpenAI.
- Memory: past.dev; one record per experiment {idea, diff_summary, verdict, metrics, evidence}.
- Logs: every event appended to `runs/events.jsonl` (ts, agent, kind, payload) -> dashboard reads it.

## Owners
- d1: data prep + candidate fingerprints + baseline model on Modal (critical path)
- d2: Modal vLLM OpenAI-compatible endpoint for an open-weight model + Sandbox runner
- f2: Exa tool + past.dev memory tool + read both sponsor challenge briefs, report exact asks
- 49 (lead): Referee, swarm orchestrator, integration, demo, pitch

## Demo (2 min)
1. Problem + my Kaggle story (rank 30/5000, the traps).
2. Unguarded loop: agent "improves" val, Referee shows NOISE / PRIOR / LEAK with evidence.
3. Guarded swarm: live dashboard, Exa citations, past.dev memory preventing repeats, a real PROMOTED gain on hidden test.
4. Numbers: accepted-claim precision unguarded vs guarded; memory ablation (repeats, $ , time).
