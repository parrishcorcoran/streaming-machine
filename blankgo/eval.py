"""The float side of every snapshot: held-out loss and the float model's own words, for the finished weights and with
the blank taken out, against scrambled blanks.
    python3 eval.py width=64 layers=2 recipe=A

For each snapshot (whole numbers at 2^-30, read back exactly by snapio.snapshots), five sets of weights, each run in
float64:
  finished   W = Bt + P
  pushes     P alone: the drawn blank (as it stands at that step) taken out of every plate, both word tables and the
             position rows. Gains, offsets and biases are left as they are (their start, 1 or 0, was never drawn).
  scr1-3     P + the blank scrambled within each neuron (snapio.scramble, three draws): the same blank values in the
             same neurons, meeting the wrong input lines. Control: a blank of exactly the same values that the pushes
             never met.
Measured on each: held-out loss (128 fixed windows of 128 words of the held-out part), and at the 30 held-out texts
(8 words lead, then 16 positions) the top word, the top 5, and the lead of word 1 over word 2. Written to
evals/eval_<shape>_<recipe>.json; the words are the references the whole-number machine is checked against.

PREDICTIONS (the session's, written before any run of this script):
  E1. Step 0: P is 0 in every drawn part, so 'pushes' scores every word 0: loss = ln 8193 = 9.0110 exactly (fixed by
      the definition; a check that the code removes what it says). The scrambled blanks at step 0 give 9.02-9.05,
      the blank's own level.
  E2. A and B (shave off), every snapshot from step 10 on: loss(pushes) > loss(finished). Taking the blank out costs
      the answer, because every push was worked out with the blank in place.
  E3. Every snapshot from step 10 on: loss(scr) > loss(finished); the three draws within 0.05 nats of each other.
      If the pushes did not depend on this particular blank, the scrambled blank would cost nothing; the session
      expects it to cost about as much as taking the blank out.
  E4. At step 3000 the cost of taking the blank out, loss(pushes) - loss(finished), is largest in A and smallest in
      C (A > B > C), because the scaled pushes in B and C grow several times larger than the blank (0.02) while A's
      plain pushes in the plates stay near the blank's size.
  E5. Float top word, pushes vs finished, at the 480 positions, step 3000: A 10-40%, B and C 30-70%.
"""
import os, sys, json, numpy as np, torch
from model import HERE, V, layout, drawn, tag, data, held_windows, word_texts, to_float, forward, held_loss
from snapio import snapshots, scramble

A = dict(a.split("=") for a in sys.argv[1:])
d, L, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
NS = int(A.get("scrambles", 3))
names = [n for n, _, _ in layout(d, L)]; dr = set(drawn(d, L))
_, held = data(); win = held_windows(held); texts = word_texts(held)
OUT = os.path.join(HERE, "evals"); os.makedirs(OUT, exist_ok=True)

def words(p):
    with torch.no_grad():
        lg = forward(p, torch.from_numpy(texts), d, L)[:, 8:, :]                 # (30, 16, V): after each of words 9..24
    top = torch.topk(lg, 5, -1)
    return dict(top1=top.indices[..., 0].tolist(), top5=top.indices.tolist(),
                margin=(top.values[..., 0] - top.values[..., 1]).tolist())

res = dict(width=d, layers=L, recipe=RECIPE, texts=texts.tolist(), snaps={})
for t, Bt, P, W in snapshots(d, L, RECIPE):
    sets = {"finished": W, "pushes": {n: (P[n] if n in dr else W[n]) for n in names}}
    for k in range(1, NS + 1):
        sets[f"scr{k}"] = {n: (P[n] + scramble(n, Bt[n], k) if n in dr else W[n]) for n in names}
    out = {}
    for s, Wi in sets.items():
        p = to_float(Wi); lo, se = held_loss(p, win, d, L); w = words(p)
        out[s] = dict(held_loss=lo, held_se=se, **w)
    f1 = np.array(out["finished"]["top1"])
    for s in out:
        out[s]["agree_finished_top1"] = int((np.array(out[s]["top1"]) == f1).sum())
        out[s]["finished_in_top5"] = int(sum(f1[i, j] in out[s]["top5"][i][j] for i in range(f1.shape[0]) for j in range(f1.shape[1])))
    out["control_pairing"] = int((np.roll(f1, 1, 0) == f1).sum())        # each text's words against the previous text's
    res["snaps"][t] = out
    print(f"  step {t:5d}  loss  finished {out['finished']['held_loss']:.4f}  pushes {out['pushes']['held_loss']:.4f}  "
          + "  ".join(f"{s} {out[s]['held_loss']:.4f}" for s in out if s.startswith("scr"))
          + f"   top word as finished: pushes {out['pushes']['agree_finished_top1']}/480  "
          + " ".join(f"{s} {out[s]['agree_finished_top1']}" for s in out if s.startswith("scr")), flush=True)
    json.dump(res, open(os.path.join(OUT, f"eval_{tag(d, L)}_{RECIPE}.json"), "w"))
print("done")
