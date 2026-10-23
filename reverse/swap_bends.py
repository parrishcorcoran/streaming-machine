"""THE NEXT WAY, SECOND STEP: WHERE IN THE BENDS DOES THE TEXT SIT? swap_kinds.py found: with B's bend shares on A's words
the answer leaves the top 5 in 30/30 texts; with B's water levels and B's arrival shares but A's own bends it stays in
the top 5 in 22/30. So the text is written in the bends. Here B's bend shares are held at ONE layer only (A's own light
sets every other dimming), then at every layer but one, then at the earlier positions only (positions 0..3, A's own
bends at the last position: in the streaming machine the earlier columns are already standing) and at the last
position only. A = each of the 30 refs texts (5 tokens), B = the next text. Where does A's answer fall among 50,257?
Control: A's 2nd..5th words; coincidence: A's answer in B's own top 5 (1/30, median rank 942, swap_kinds.log).

PREDICTION (Claude's, written before running):
  - B's bends at one layer only: the answer stays in the top 5 in most texts (>= 20/30) at every layer; the text is
    spread over the layers' bends.
  - B's bends at every layer but one: the answer leaves the top 5 in most texts whichever layer is kept.
  - B's bends at the earlier positions only: the answer stays in the top 5 in about half the texts (~15/30); at the
    last position only: it leaves in most.
    python3 reverse/swap_bends.py       -> reverse/swap_bends.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

def rank_of(V, i): return int((V > V[i]).sum())

def own_bends(rec): return [Ly["bend"]["share"] for Ly in rec["layers"]]

def held_bends(shares): return dict(layers=[dict(bend=dict(share=s)) for s in shares])

t0 = time.time(); NT = len(TEXTS); recs = [record(ids) for ids in TEXTS]
CONF = [f"B at layer {l} only" for l in range(12)] + [f"B at all but layer {l}" for l in range(12)] + \
       ["B at positions 0-3 only", "B at the last position only"]
R = {c: dict(ans=[], ctl=[]) for c in CONF}
for i in range(NT):
    A, B = recs[i], recs[(i + 1) % NT]; SA = A["top"]["scores"]; wA = race(SA)
    top5 = [int(x) for x in np.argsort(-SA, kind="stable")[:5]]
    a, b = own_bends(A), own_bends(B)
    sets = {}
    for l in range(12):
        sets[f"B at layer {l} only"] = [b[k] if k == l else a[k] for k in range(12)]
        sets[f"B at all but layer {l}"] = [a[k] if k == l else b[k] for k in range(12)]
    sets["B at positions 0-3 only"] = [np.concatenate([b[k][:4], a[k][4:]]) for k in range(12)]
    sets["B at the last position only"] = [np.concatenate([a[k][:4], b[k][4:]]) for k in range(12)]
    same = (record(TEXTS[i], held=held_bends(a))["top"]["scores"] == SA).all()          # own bends held: the machine itself
    assert same
    for c in CONF:
        fz = record(TEXTS[i], held=held_bends(sets[c]))["top"]["scores"]
        R[c]["ans"].append(rank_of(fz, wA)); R[c]["ctl"] += [rank_of(fz, w) for w in top5[1:]]
    print(f"  text {i:2d} {word(wA)!r:12s} B at one layer: " + " ".join(str(R[f'B at layer {l} only']['ans'][-1]) for l in range(12))
          + " | all but one: " + " ".join(str(R[f'B at all but layer {l}']['ans'][-1]) for l in range(12))
          + f" | pos 0-3: {R['B at positions 0-3 only']['ans'][-1]}, last: {R['B at the last position only']['ans'][-1]}  ({time.time() - t0:.0f}s)", flush=True)
pc = lambda x: f"{100 * x:.1f}%"
print(f"\n(A's own bends held exactly reproduce the machine: checked on all {NT})")
print("A's words; bend shares held from B where said, every other dimming set by A's own light: A's answer top 5 / first / median rank | A's 2nd..5th in top 5")
for c in CONF:
    a_ = np.array(R[c]["ans"]); k = np.array(R[c]["ctl"])
    print(f"  {c:30s} {int((a_ < 5).sum()):2d}/{NT}  {int((a_ == 0).sum()):2d}/{NT}  {np.median(a_):7.0f}   | {pc((k < 5).mean())}")
print(f"({time.time() - t0:.0f}s)")
