#!/usr/bin/env bash
# Publish a read-only snapshot of the live dashboard to Vercel.
# Copies the page, injects the static flag, and bundles runs/events.jsonl as events.json.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=ui/dist
rm -rf "$OUT" && mkdir -p "$OUT"
sed 's#<head>#<head><script>window.SPECTRE_STATIC=1</script>#' ui/index.html > "$OUT/index.html"
python3 - <<'PY'
import json, time
evs = [json.loads(l) for l in open("runs/events.jsonl") if l.strip()]
json.dump({"source": f"snapshot {time.strftime('%H:%M PT')}", "sample": False, "events": evs},
          open("ui/dist/events.json", "w"))
print(len(evs), "events")
PY
cd "$OUT" && vercel deploy --prod --yes --name spectre-dashboard 2>&1 | tail -3
