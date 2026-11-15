"""Export every Referee verdict as a plain text case file that verify.cpp can check.

The Referee stores raw per-spectrum ranks in runs/referee_cache.json and the verdicts it
reached in runs/events.jsonl. This script joins the two: for each verdict it writes the
claimed numbers plus every rank vector the verdict was computed from. Nothing is
recomputed here. The point is that the C++ verifier redoes all the math on its own.

Case file format (whitespace separated, one record per line):

    CASE <name>
    CLAIM <field> <value>          verdict, val_gain, val_ci_lo, ... (only fields present)
    RUN <cand|inc> <split> <seed> <mode> <N>
    <N ints: 0-based rank of the true molecule, -1 = not in top 20>
    END

Usage:  python verify/export.py            (writes verify/cases/*.txt + index.txt)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / "referee_cache.json"
EVENTS = ROOT / "runs" / "events.jsonl"
OUT = Path(__file__).resolve().parent / "cases"
SEEDS = (0, 1, 2)

# Which runs each kind of verdict was computed from: (who, split, seed, mode).
GUARDED_RUNS = (
    [(who, "val", s, "plain") for who in ("cand", "inc") for s in SEEDS]
    + [(who, "val", 0, m) for who in ("cand", "inc") for m in ("shuffle", "ablate")]
    + [(who, "test", s, "plain") for who in ("cand", "inc") for s in SEEDS]
)
NAIVE_RUNS = [("cand", "val", 0, "plain"), ("inc", "val", 0, "plain")]


def claimed_fields(event: dict) -> list[tuple[str, object]]:
    """Flatten the numbers the Referee claimed into (name, value) pairs."""
    fields: list[tuple[str, object]] = [("verdict", event["verdict"])]
    for name in ("val_gain", "noise_band", "shuffle_gain", "ablate_gain", "test_gain"):
        if event.get(name) is not None:
            fields.append((name, event[name]))
    for name in ("val_ci", "test_ci"):
        if event.get(name) is not None:
            lo, hi = event[name]
            fields += [(f"{name}_lo", lo), (f"{name}_hi", hi)]
    for i, g in enumerate(event.get("per_seed_gain") or []):
        fields.append((f"per_seed_gain_{i}", g))
    return fields


def format_value(value) -> str:
    # repr keeps every bit of a float, so the verifier can compare at 1e-9.
    return repr(float(value)) if isinstance(value, (int, float)) else str(value)


def write_case(path: Path, name: str, event: dict, runs: dict) -> None:
    """runs maps (who, split, seed, mode) -> list of ranks."""
    lines = [f"CASE {name}"]
    lines += [f"CLAIM {k} {format_value(v)}" for k, v in claimed_fields(event)]
    for (who, split, seed, mode), ranks in runs.items():
        lines.append(f"RUN {who} {split} {seed} {mode} {len(ranks)}")
        lines.append(" ".join(str(int(r)) for r in ranks))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def collect_runs(event: dict, cache: dict, wanted) -> dict:
    """Pull the cached rank vectors this verdict used. Missing ones are skipped,
    which is what happens for early stops (NOISE never reaches the test fold)."""
    runs = {}
    for who, split, seed, mode in wanted:
        sha = event["code"] if who == "cand" else event["incumbent"]
        entry = cache.get(f"{sha}:{split}:{seed}:{mode}")
        if entry and entry.get("ok"):
            runs[(who, split, seed, mode)] = entry["ranks"]
    return runs


def main() -> int:
    if not CACHE.exists() or not EVENTS.exists():
        print(f"need {CACHE} and {EVENTS}; run the Referee first", file=sys.stderr)
        return 1
    cache = json.loads(CACHE.read_text())
    OUT.mkdir(exist_ok=True)
    index, skipped = [], 0
    for line in EVENTS.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("kind") not in ("verdict", "naive_verdict"):
            continue
        if event.get("verdict") == "CRASH":
            skipped += 1  # a crash has no ranks to check
            continue
        wanted = NAIVE_RUNS if event["kind"] == "naive_verdict" else GUARDED_RUNS
        runs = collect_runs(event, cache, wanted)
        name = event.get("exp_id") or event["code"]
        if name in index:
            name = f"{name}_{len(index)}"
        write_case(OUT / f"{name}.txt", name, event, runs)
        index.append(name)
    (OUT / "index.txt").write_text("".join(f"{n}.txt\n" for n in index))
    print(f"wrote {len(index)} cases to {OUT} (skipped {skipped} CRASH verdicts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
