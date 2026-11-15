"""Run the whole experiment plan on Modal, so it survives the laptop going offline.

One container runs every track as threads in one process. That matters: the Referee cache,
the event log and past.dev writes are shared, and one writer per file keeps them consistent.
State lives on the Volume `spectre-runs`, which is mounted where the code expects `runs/`.

  modal run --detach agents/cloud.py --plan main
  modal volume get spectre-runs events.jsonl runs/events.jsonl --force   # pull for the dashboard
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
CODE = "/root/spectre"
runs_vol = modal.Volume.from_name("spectre-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("modal==1.6.4", "openai", "anthropic", "exa-py", "python-dotenv",
                    "numpy", "pandas", "requests")
    .add_local_dir(LOCAL, CODE, ignore=["runs", ".venv", ".git", "ui/dist", "**/__pycache__",
                                        "docs", "verify", "*.pdf", ".env"])
)
app = modal.App("spectre-orchestrator", image=image)

# Each plan is a list of swarm argument lists; all run concurrently in one process.
PLANS = {
    "main": [
        "--tag swarm --proposers 4 --rounds 4 --memory pastdev --models gpt-5.5,qwen",
        "--tag naive --naive --proposers 4 --rounds 4 --memory none --models gpt-5.5,qwen",
        "--tag single --proposers 1 --rounds 16 --budget 32 --memory local --models gpt-5.5",
    ],
    "noexa": ["--tag noexa --no-exa --proposers 4 --rounds 3 --memory local --models gpt-5.5,qwen"],
}


def _committer(stop: threading.Event):
    """Volumes only persist on commit; commit every 20 s so the dashboard can follow along."""
    while not stop.wait(20):
        try:
            runs_vol.commit()
        except Exception as e:  # never kill the run over a sync hiccup
            print("commit failed:", e, flush=True)


@app.function(volumes={"/vol": runs_vol}, timeout=6 * 3600,
              secrets=[modal.Secret.from_dotenv(LOCAL)], cpu=4, memory=8192)
def run_plan(plan: str):
    os.makedirs("/vol/runs", exist_ok=True)
    runs = Path(CODE) / "runs"
    if not runs.exists():
        runs.symlink_to("/vol/runs")
    os.chdir(CODE)
    sys.path.insert(0, CODE)

    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()

    import shlex
    from agents import swarm

    def track(argline: str):
        sys_argv = ["swarm"] + shlex.split(argline)
        print("start", argline, flush=True)
        try:
            swarm.main(sys_argv[1:])
        except Exception as e:
            print("track failed", argline, repr(e), flush=True)

    threads = [threading.Thread(target=track, args=(a,)) for a in PLANS[plan]]
    for t in threads:
        t.start()
        time.sleep(5)
    for t in threads:
        t.join()
    stop.set()
    runs_vol.commit()


@app.local_entrypoint()
def main(plan: str = "main"):
    run_plan.remote(plan)


@app.function(volumes={"/vol": runs_vol}, timeout=3 * 3600,
              secrets=[modal.Secret.from_dotenv(LOCAL)], cpu=2, memory=4096)
def run_seeded(items: list[str], state: str = "seeded"):
    """Judge hand-written candidates in their own state dir (/vol/seeded), so this container
    never writes the files the swarm container owns."""
    os.makedirs(f"/vol/{state}", exist_ok=True)
    os.environ["SPECTRE_RUNS"] = f"/vol/{state}"
    os.environ.setdefault("SPECTRE_GPU_SLOTS", "3")
    os.environ["SPECTRE_GPU_WAIT"] = "5400"  # big models must not fall back to CPU
    os.chdir(CODE)
    sys.path.insert(0, CODE)
    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()
    sys.argv = ["judge_many", "seeded"] + items
    import runpy
    runpy.run_module("bench.judge_many", run_name="__main__")
    stop.set()
    runs_vol.commit()


@app.local_entrypoint()
def seeded(items: str, state: str = "seeded"):
    """items: 'file:claim;;file:claim' (paths relative to the repo). state: its own dir on the volume."""
    run_seeded.remote(items.split(";;"), state)
