"""Smoke test the LLM endpoint: one chat completion + one tool call.

Reads SPECTRE_LLM_URL, SPECTRE_LLM_MODEL, SPECTRE_LLM_API_KEY from env or .env.
Usage: .venv/bin/python llm/smoke.py [base_url model]
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


def load_env():
    p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    base = sys.argv[1] if len(sys.argv) > 1 else os.environ["SPECTRE_LLM_URL"]
    model = sys.argv[2] if len(sys.argv) > 2 else os.environ["SPECTRE_LLM_MODEL"]
    key = os.environ.get("SPECTRE_LLM_API_KEY") or "none"
    client = OpenAI(base_url=base, api_key=key, timeout=600)

    t = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Write a one-line Python lambda that squares x. Reply with code only."}],
        max_tokens=200,
    )
    print(f"chat ok in {time.time()-t:.1f}s:", r.choices[0].message.content.strip()[:200])

    tools = [{
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Submit a new model.py for evaluation by the referee.",
            "parameters": {
                "type": "object",
                "properties": {
                    "idea": {"type": "string", "description": "one-sentence hypothesis"},
                    "code": {"type": "string", "description": "full model.py source"},
                },
                "required": ["idea", "code"],
            },
        },
    }]
    t = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Submit a patch whose idea is 'cosine similarity baseline' and code is 'def train(df, seed):\\n    return None'."}],
        tools=tools,
        tool_choice="auto",
        max_tokens=800,
    )
    calls = r.choices[0].message.tool_calls or []
    assert calls, f"no tool call: {r.choices[0].message.content!r}"
    args = json.loads(calls[0].function.arguments)
    print(f"tool call ok in {time.time()-t:.1f}s:", calls[0].function.name, sorted(args))


if __name__ == "__main__":
    main()
