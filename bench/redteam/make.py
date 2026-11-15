"""Seeded fake gains: each is a real trap from Kaggle-style research. The Referee must catch them.
noise.py  : cosmetic width change, 'gain' is seed noise          -> expect NOISE
prior.py  : boosts candidates that look like training molecules  -> expect PRIOR (gain survives zeroed spectra)
"""
from pathlib import Path
B = Path(__file__).resolve().parent.parent / "baseline" / "model.py"
src = B.read_text()
out = Path(__file__).resolve().parent

def sub(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b)

(out / "noise.py").write_text(sub(src, "HIDDEN = 1024", "HIDDEN = 1056"))

p = sub(src, '    return {"net": net.cpu().state_dict()}',
        '    prior = y.mean(0)\n    prior = (prior / prior.norm()).cpu().numpy()\n'
        '    return {"net": net.cpu().state_dict(), "prior": prior}')
p = sub(p, "        s = c @ pred[row_of[cid]]",
        "        s = c @ pred[row_of[cid]] + 1.0 * (c @ state[\"prior\"])")
(out / "prior.py").write_text(p)
print("ok")
