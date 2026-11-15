#!/usr/bin/env bash
# Keep runs/events.jsonl = cloud events (Modal volume) + local seeded events (runs_local/), every 20 s.
cd "$(dirname "$0")/.."
while true; do
  if .venv/bin/modal volume get --force spectre-runs runs/events.jsonl runs/events.cloud.jsonl >/dev/null 2>&1; then
    .venv/bin/modal volume get --force spectre-runs seeded/events.jsonl runs/events.seeded.jsonl >/dev/null 2>&1
    .venv/bin/modal volume get --force spectre-runs seeded2/events.jsonl runs/events.seeded2.jsonl >/dev/null 2>&1
    cat runs/events.cloud.jsonl runs/events.seeded.jsonl runs/events.seeded2.jsonl runs_local/events.jsonl 2>/dev/null > runs/events.tmp && mv runs/events.tmp runs/events.jsonl
  fi
  sleep 20
done
