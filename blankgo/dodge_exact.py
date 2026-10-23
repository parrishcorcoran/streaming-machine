"""The dodge reading of work.py with its chance level worked out exactly instead of drawn: how hard the machine's own
light meets the blank, against the average over EVERY order of the light's lanes.
    python3 dodge_exact.py width=64 layers=2 recipe=A

HIS WORDS (2026-10-06): "because the embedding table is the opposite as the ENTIRE all neuron rows, a tiny change in
embedding is a bigger change on other side, so if it's not balanced perfectly then the blank is stuck inside the
embedding table."
SESSION'S RESTATEMENT (the session's words): if the blank stays and the light is built to steer around it, the blank's
part of every pile is smaller than the same light would give with its lanes in any other order.

work.py compared |x16 . s| with 5 drawn lane orders; the draws scatter by up to +-0.12 at the down plates, and
workcheck.py's own 3 draws moved the ratio by up to 0.15. Here, for every position and neuron, the average of
(x_order . s_j)^2 over ALL n! orders of the n lanes is exact:
    E = (sum s^2)(sum x^2)/n + ((sum s)^2 - sum s^2)((sum x)^2 - sum x^2)/(n(n-1))
(each lane meets each blank value with chance 1/n; two different lanes meet two different values with chance
1/(n(n-1))). ratio = sqrt( sum |x . s|^2 / sum E ), on the machine's own light (bmachine.py, finished plates), every
plate, every snapshot, the 128 held-out windows. Under 1: the light steers around the blank. Over 1: it leans into it.
The same exact chance level for the built part (x . (w - s)) and the finished plate (x . w), for comparison.

PREDICTIONS (the session's, written before any run of this script; work.py's drawn ratios were already seen: about 1 at
qkv, fc and head; 1.03-1.5 at the out and down plates):
  D1. Step 0: ratio 1 +- 0.02 at every plate (the blank light and the blank plate are separate draws).
  D2. From step 100 on, the out (proj) and down (mproj) plates: ratio 1.05-1.6 (the light leans into the blank) in every
      build; qkv, fc, head within 0.95-1.08.
  D3. The built part: ratio well above 1 at every plate from step 100 on (2-10: the built plate is matched to the light,
      which is what building does; given as the scale for the blank's number).
"""
import os, sys, json, math, numpy as np
from model import HERE, tag, data, held_windows
from snapio import build_log
from bmachine import Machine, machine_file

np.seterr(all="ignore")
A = dict(a.split("=") for a in sys.argv[1:])
d, L, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
name = f"{tag(d, L)}_{RECIPE}"
_, held = data(); WIN = held_windows(held)
PLATES = [f"{k}.{l}" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
res = dict(shape=tag(d, L), recipe=RECIPE, snaps={})
print(f"dodge_exact {name}", flush=True)
for t in sorted(int(s) for s in build_log(d, L, RECIPE)["snaps"]):
    M = machine_file(d, L, RECIPE, t); m = Machine(d, L, RECIPE, t, "finished", capture=True, M=M)
    Q = {}
    for pl in PLATES:
        w, s = M[pl + ".w"].astype(np.float64), M[pl + ".s"].astype(np.float64); cs2 = (2.0 ** M[pl + ".e"].astype(np.float64)) ** 2
        Q[pl] = {lab: (a, (a * a).sum(0), a.sum(0) ** 2) for lab, a in (("blank", s), ("built", w - s), ("finished", w))}
        Q[pl]["cs2"] = cs2
    acc = {pl: {lab: [0.0, 0.0] for lab in ("blank", "built", "finished")} for pl in PLATES}
    for win in WIN:
        m.cap = {}; m.forward(win[:-1], 128)
        for pl in PLATES:
            x16, ex = m.cap[pl][0]; x = x16.astype(np.float64); n = x.shape[1]; rs2 = (2.0 ** ex.astype(np.float64)) ** 2
            sx2 = (x * x).sum(1); sx = x.sum(1); cs2 = Q[pl]["cs2"]
            for lab in ("blank", "built", "finished"):
                a, sa2, sa_sq = Q[pl][lab]
                real = (x @ a) ** 2
                assert np.isfinite(real).all()
                E = np.outer(sx2, sa2) / n + np.outer(sx * sx - sx2, sa_sq - sa2) / (n * (n - 1))
                acc[pl][lab][0] += float((real * rs2[:, None] * cs2[None, :]).sum())
                acc[pl][lab][1] += float((E * rs2[:, None] * cs2[None, :]).sum())
    out = {pl: {lab: (math.sqrt(v[0] / v[1]) if v[1] > 0 else None) for lab, v in acc[pl].items()} for pl in PLATES}
    res["snaps"][t] = out
    print(f"  step {t:5d}  blank: " + " ".join(f"{pl} {out[pl]['blank']:.3f}" for pl in PLATES)
          + "   | built: " + " ".join(f"{pl} {out[pl]['built']:.2f}" for pl in PLATES if out[pl]["built"]), flush=True)
    json.dump(res, open(os.path.join(HERE, "evals", f"dodge_exact_{name}.json"), "w"), indent=1)
print("done")
