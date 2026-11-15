#!/usr/bin/env bash
# Primary LLM: Modal Dedicated Endpoint (https://modal.com/docs/guide/endpoints).
# Qwen3.6 35B A3B FP8: MoE with 3B active params, strong at code + tool calling, fast on one GPU.
# min-containers 1 + long scaledown so the demo never waits on a cold start.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/modal endpoint create \
  --name spectre-llm \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --min-containers 1 \
  --max-containers 3 \
  --scaledown-window 3600
