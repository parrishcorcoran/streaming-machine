"""Where the blank's cost sits: the blank taken out of ONE place at a time, against that place's blank scrambled.
    python3 where.py width=64 layers=2 recipe=A [steps=300,1000,3000]

HIS WORDS (2026-10-06): "because the embedding table is the opposite as the ENTIRE all neuron rows, a tiny change in
embedding is a bigger change on other side, so if it's not balanced perfectly then the blank is stuck inside the
embedding table."
SESSION'S RESTATEMENT (the session's words, not his): the input word table faces every plate after it, so a small
change in a word's row moves every pile it meets; if the building does not balance the blank there exactly, the blank
stays in the word table. eval.py took the blank out of every place at once. Here it is taken out of one place at a
time, everything else left as built (blank + pushes):
  wte    the input word table          wpe    the position rows
  plates every qkv, out, up, down plate     head   the output word table
For each place: held-out loss with that place's blank taken out (P alone there), and with that place's blank
scrambled within each neuron (snapio.scramble, draws 1-3, the same draws as eval.py and machine.py).

PREDICTIONS (the session's, written before any run of this script):
  W1. A at step 3000: taking the blank out of the input word table alone costs more than out of the plates alone
      (A's pushes there are 0.24-0.37 of the blank by size, so most of each word's row is blank).
  W2. B and C at step 3000: the output word table alone costs the least of the four places (pushes there are 6.7-7.3
      times the blank).
  W3. B and C at step 3000: in every place, scrambling that place's blank costs at least as much as taking it out (as
      for all places at once in eval.py).
  No prediction on whether the four single-place costs add up to the all-places cost.
"""
import os, sys, json, numpy as np
from model import HERE, layout, drawn, tag, data, held_windows, to_float, held_loss
from snapio import snapshots, scramble

A = dict(a.split("=") for a in sys.argv[1:])
d, L, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
STEPS = [int(s) for s in A.get("steps", "300,1000,3000").split(",")]
names = [n for n, _, _ in layout(d, L)]; dr = drawn(d, L)
PLACES = {"wte": ["wte"], "wpe": ["wpe"], "plates": [n for n in dr if n.endswith(".w")], "head": ["head"]}
assert sorted(sum(PLACES.values(), [])) == sorted(dr)
_, held = data(); win = held_windows(held)
ev = json.load(open(os.path.join(HERE, "evals", f"eval_{tag(d, L)}_{RECIPE}.json")))
res = dict(width=d, layers=L, recipe=RECIPE, snaps={})
for t, Bt, P, W in snapshots(d, L, RECIPE, only=set(STEPS)):
    fin = ev["snaps"][str(t)]["finished"]["held_loss"]; out = dict(finished=fin, all_out=ev["snaps"][str(t)]["pushes"]["held_loss"])
    for place, ks in PLACES.items():
        Wo = dict(W); Wo.update({n: P[n] for n in ks})
        rec = dict(out=held_loss(to_float(Wo), win, d, L)[0])
        for k in (1, 2, 3):
            Ws = dict(W); Ws.update({n: P[n] + scramble(n, Bt[n], k) for n in ks})
            rec[f"scr{k}"] = held_loss(to_float(Ws), win, d, L)[0]
        out[place] = rec
    res["snaps"][t] = out
    print(f"  step {t:5d}  finished {fin:.3f}  all places out +{out['all_out'] - fin:.3f}  |  " + "  ".join(
        f"{p} out +{out[p]['out'] - fin:.3f} scr +{min(out[p][f'scr{k}'] for k in (1, 2, 3)) - fin:.3f}..+{max(out[p][f'scr{k}'] for k in (1, 2, 3)) - fin:.3f}"
        for p in PLACES), flush=True)
    json.dump(res, open(os.path.join(HERE, "evals", f"where_{tag(d, L)}_{RECIPE}.json"), "w"), indent=1)
