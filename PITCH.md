# Spectre pitch

Facts checked against the repo on 2026-11-15:
* Index 0 trap. In `runs/prep.log`, `orig_true_pos_frac_idx0` is 1.0 for val and for test. That means 4000 of 4000 spectra. After our per-id shuffle, the true molecule is first 4.2% of the time on val and 3.1% on test. Random hit@1 is 3.8% on val and 3.0% on test.
* The Referee's verdicts are in `bench/referee.py:1`: CRASH, NOISE, LEAK, PRIOR, OVERFIT, PROMOTED. The metric is MRR@20 on val, averaged over seeds.
* Kaggle. Enveda CASMI26, rank 30 of about 5,000 teams.

Fill the `[NUMBER]` slots from `runs/events.jsonl` before recording.

## (a) Round 1 pitch, 3:00

Target is about 420 spoken words. Times are cumulative.

**0:00 Hook (about 25 s)**
> MassSpecGym is a public benchmark. You get a mass spectrum, you rank candidate molecules, and you try to put the right one first.
> In the raw candidate lists, the right molecule is already first. Every time. 4,000 out of 4,000.
> An AI scientist that "discovers" this gets a perfect score. It will tell you it solved the problem.

**0:25 Why I care (about 25 s)**
> I'm ranked around 30th of 5,000 teams in a Kaggle competition on this exact kind of data.
> I spent weeks chasing improvements. Most of them were not real. Some were seed noise. Some came from validation data leaking into training. Some came from a model that guessed popular molecules without reading the spectrum.
> Autoresearch agents fall into the same traps, only faster.

**0:50 Thesis (about 15 s)**
> AI scientists do not lack ideas. They lack a judge they can't fool.
> So we built the judge first. Then we let a swarm of agents loose under it.

**1:05 The Referee (about 35 s)**
> The Referee is the only process that holds labels. Agents send code. They never see the test set.
> Every claim goes through four checks.
> One. Is the gain bigger than noise? We run several seeds. The gain must clear twice the seed spread, and its bootstrap confidence interval must stay above zero.
> Two. Label shuffle. We retrain with the answers scrambled. If the gain survives, it was leaked, not learned.
> Three. Spectrum ablation. We blank the spectra. If the gain survives, the model isn't reading the data.
> Four. Hidden test. Only claims that pass all three get scored there.
> Each verdict comes with its evidence.

**1:40 Live (about 45 s)**, on the dashboard
> These are four proposer agents. Each one has an idea, searches the literature with Exa, and checks shared memory in past.dev.
> Here proposer [NUMBER] wants to try [IDEA]. Exa found [PAPER]. It cites it right here.
> Here proposer [NUMBER] read the Referee's verdict on another agent's idea. It skipped it, and spent no compute.
> And here is a rejection. The agent claimed [NUMBER] points. The Referee says [VERDICT], because [EVIDENCE].

**2:25 Results (about 25 s)**
> Without the Referee, a plain loop accepted [NUMBER] claims. [NUMBER] of them were real. That is [NUMBER] percent precision.
> With the Referee, precision was [NUMBER] percent.
> On the same compute budget, the swarm promoted [NUMBER] real gains and a single agent promoted [NUMBER].
> With Exa turned off, [NUMBER].

**2:50 Close (about 10 s)**
> The bottleneck for AI scientists is an evaluator they can't game. Spectre is that evaluator, and a swarm that gets better under it.

## (b) Demo video, 2:00 shot list

| Time | Shot | On screen | Voice-over |
|---|---|---|---|
| 0:00 to 0:12 | Terminal | `runs/prep.log` with `orig_true_pos_frac_idx0: 1.0` highlighted, then a naive agent's val score of [NUMBER] | "The right answer is always first. A naive agent finds that and calls it a breakthrough." |
| 0:12 to 0:25 | Diagram or code | `bench/referee.py` docstring, the 5 checks | "Our Referee holds the labels. Every claim must survive noise, leak, prior and hidden test checks." |
| 0:25 to 0:40 | Terminal | `python -m agents.swarm --proposers 4` starting, then `python3 ui/server.py` | "Four proposers start. They share memory on past.dev and search papers with Exa." |
| 0:40 to 1:00 | Dashboard, one card | Hypothesis, Exa sources with the "cited" tag, claim | "Proposer [NUMBER] proposes [IDEA]. Exa finds [PAPER]. The claim cites it." |
| 1:00 to 1:15 | Dashboard, memory row | The memory read showing authors = referee, and the skip reason | "Another proposer reads that verdict from shared memory and skips a repeat." |
| 1:15 to 1:30 | Dashboard, rejection | Verdict NOISE, LEAK or PRIOR with its evidence string | "The Referee rejects a claimed gain of [NUMBER]. Here is why." |
| 1:30 to 1:40 | Terminal | `tools/test_live.py` boundary check: a proposer asks for the test score and gets nothing | "Proposers can't read test scores, even through memory." |
| 1:40 to 1:55 | Results table | Naive vs guarded precision, swarm vs single agent, Exa on vs off | "[NUMBER] percent precision unguarded. [NUMBER] percent with the Referee." |
| 1:55 to 2:00 | Title card | Spectre, repo link | "AI scientists need a judge they can't fool." |

Recording notes. Run the swarm before recording and replay events if live is slow. past.dev writes take about 30 s to become readable, and a burst can take minutes. Keep the dashboard zoomed so the citations are readable.

## Likely judge questions

**Why MassSpecGym and not BixBench?**
> MassSpecGym is a real open problem with a public benchmark and a known leak. It shows the failure modes we care about. Those are noise, leakage and priors. We know this domain from the Kaggle competition, so we can tell a real gain from a fake one. The Referee itself is benchmark agnostic. It needs a model interface and labels it keeps private.

**Are the canaries sufficient?**
> No. They catch the failures we have seen most: seed noise, labels leaking through inputs, and priors that ignore the spectrum. They are necessary checks, not proof. The hidden test is the last line. Promotions are also capped by compute, so the Referee can be slow and careful.

**What can still fool the Referee?**
> A leak that also survives label shuffling. For example, structure leaking through the candidate lists themselves. That is why we reshuffle candidates per spectrum. Also overfitting to val over many rounds. The hidden test catches that, but only for promoted claims. And a real gain smaller than our noise band gets rejected. That is a false negative we accept on purpose.

**How do you know the rejections are right?**
> We red-teamed it. `bench/redteam` holds planted fake improvements: pure noise, a prior-only model, and a leak. The Referee labels them [VERDICTS]. Each verdict carries the numbers behind it, so a person can check it.

**What does the swarm add over one agent?**
> On the same Referee budget, the swarm found [NUMBER] promoted gains and the single agent found [NUMBER]. Agents read each other's verdicts in past.dev, so they don't spend compute on ideas that already failed. [NUMBER] experiments were skipped that way.
