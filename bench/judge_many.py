"""Judge several candidate files against the current incumbent, in one process (shared cache).
usage: python -m bench.judge_many tag file1:claim1 file2:claim2 ..."""
import sys, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from bench.referee import Referee, SEEDS

tag, items = sys.argv[1], [a.split(":", 1) for a in sys.argv[2:]]
ref = Referee()
inc = ref.incumbent()
# warm incumbent runs once so parallel judges share them
ref.batch([(inc, "val", s, "plain") for s in SEEDS] + [(inc, "val", 0, m) for m in ("ablate", "shuffle")])
def j(it):
    f, claim = it
    return ref.judge(Path(f).read_text(), claim=claim, agent="human-seeded", exp_id=Path(f).stem, tag=tag)
with ThreadPoolExecutor(len(items)) as ex:
    for v in ex.map(j, items):
        print(json.dumps({k: v.get(k) for k in ("exp_id", "verdict", "val_gain", "val_ci", "per_seed_gain",
                                               "noise_band", "shuffle_gain", "ablate_gain", "test_gain", "evidence")}))
