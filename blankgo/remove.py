"""Remove it: the whole-number machine run with the blank taken out exactly, with the blank alone, and finished; and with
the blank taken out together with the counter work.py and inplace.py found.
    python3 remove.py width=64 layers=2 recipe=A

HIS WORDS (2026-10-06): "Maybe try a couple small model, train a little bit and then remove blank slates or see wheee
they go. It's an exact thing."

Machines (bmachine.py, machine.py's arithmetic part for part), every snapshot, all on the finished plate's grid:
  (iii) finished   .w
  (i)   pushes     .w - .s on every drawn part (the blank removed exactly, digit for digit)
  (ii)  blank      .s on every drawn part; gains, offsets and biases finished (their start was never drawn), so (i)
                   and (ii) split the drawn parts exactly between them
  (iv)  the blank removed together with its counter. A counter is a part of what was built that runs opposite to the
        blank. Two readings of it, one number per part, no search:
          iv_light  from work.py, on the real light: in each plate where gamma < 1 (the built part cancels 1 - gamma of
                    the blank's pile), the counter is -(1 - gamma) x blank; removing the blank and the counter leaves
                    .w - gamma x .s there. Plates with gamma >= 1 have no counter: .w - .s. The word tables and position
                    rows (no plate pile of their own): .w - .s.
          iv_place  from inplace.py, in place: in each drawn part where s < 1 the counter is -(1 - s) x blank, so
                    .w - s x .s there; .w - .s elsewhere.
        (rint to whole numbers on the same grid.) gamma was measured on the 128 held-out loss windows, so the held-out
        loss of iv_light is read on the same text it was set from; the words are read on the 30 separate texts.
Readings against (iii):
  top-1   the machine's word (race) at the 30 held-out texts' 480 positions (8 words lead, then 16), equal to (iii)'s
  top-5   (iii)'s word inside the variant's top 5
  greedy  from each text's 8-word lead, 16 words each machine chooses itself (its word fed back); positions equal to
          (iii)'s own greedy words, and the words before the first difference. Pairing control: (iii)'s greedy words
          against the previous text's.
  loss    held-out loss from the machine's whole-number head piles (log-sum-exp in float64 as a reading), the first 64 of
          the 128 windows (8,192 words)
Identity: (iii) and (i) head piles at the 30 texts equal machine.py's, sha256 for sha256.
Changes after the first trial run (w64 L2 A, step 3000 only, 58 s), made before any full run: the loss is read on the
first 64 windows instead of all 128 (time); workcheck.py reads the float loss of (i) and (ii) on the same 64 windows for
R1; a pairing control for the words before the first greedy difference was added (the finished machine's greedy words
matched the previous text's at 390 of 480 positions: the greedy runs fall into the same repeated words).

PREDICTIONS (the session's, written before any run of this script; work.py's first trial (w128 L4 B step 3000, 8
windows) showed gamma ABOVE 1 in 15 of 16 trunk plates (1.2-6.7, the built part running along the blank) and below 1 at
fc.3 (0.67) and the head (0.41)):
  R1. (i) gives eval.py's float numbers: top-1 as finished within 5 of the float's count, machine loss within 0.01 nats
      of the float loss, every snapshot.
  R2. (ii) blank alone, from step 300 on: top-1 as finished at most 30 of 480 except where (iii) says one word
      everywhere (A's one-word stage); loss 8.9-10 nats (worse than the blank's own 9.02-9.04 where the finished water
      levels' gains are larger than 1).
  R3. greedy at step 3000: (i) matches (iii)'s greedy words at 15-50% of 480 positions in B and C, the first difference
      within 1-4 words on average; (ii) at most 5%.
  R4. iv_light and iv_place at step 3000 cost less than (i) (by size alone: less of the blank is removed), 30-80% of (i)'s
      loss cost; top-1 as finished between (i)'s and 480.
"""
import os, sys, json, math, time, numpy as np
from model import HERE, tag, data, held_windows, drawn
from snapio import build_log
from bmachine import Machine, machine_file, pile_sha, ES

np.seterr(all="ignore")
A = dict(a.split("=") for a in sys.argv[1:])
d, L, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
name = f"{tag(d, L)}_{RECIPE}"
NWL = int(A.get("windows", 64))
_, held = data(); WIN = held_windows(held)[:NWL]
EV = json.load(open(os.path.join(HERE, "evals", f"eval_{name}.json")))
WJ = json.load(open(os.path.join(HERE, "words", f"words_{name}.json")))
WK = json.load(open(os.path.join(HERE, "evals", f"work_{name}.json")))
IP = json.load(open(os.path.join(HERE, "evals", f"inplace_{name}.json")))
TEXTS = np.array(EV["texts"]); LEAD, NREAD = 8, 16
STEPS = sorted(int(s) for s in build_log(d, L, RECIPE)["snaps"])
if "steps" in A: STEPS = [int(x) for x in A["steps"].split(",")]
KEYPART = lambda key: key + ".w" if key not in ("wte", "wpe", "head") else key       # machine key -> drawn part name

def make_iv(coef):                                    # coef: machine key -> fraction of the blank to remove (1 = all)
    def f(key, w, s):
        c = coef.get(key, 1.0)
        return w - s if c == 1.0 else w - np.rint(c * s).astype(np.int64)
    return f

def run(m, texts):
    tops, top5, shas = [], [], []
    for ids in texts:
        g, lg = m.forward(ids, NREAD); shas.append(pile_sha(lg)); tops.append(g)        # g: the race's winners
        top5.append([np.argsort(-row, kind="stable")[:5].tolist() for row in lg])
    return np.array(tops), top5, shas

def greedy(m, texts, n=16):
    out = []
    for ids in texts:
        seq = list(ids[:LEAD])
        for _ in range(n):
            g, _ = m.forward(np.array(seq), 1); seq.append(g[0])
        out.append(seq[LEAD:])
    return np.array(out)

def loss(m):
    tot = 0.0
    for win in WIN:
        _, lg = m.forward(win[:-1], 128); z = lg.astype(np.float64) / 2.0 ** ES
        mx = z.max(1); tot += float((mx + np.log(np.exp(z - mx[:, None]).sum(1)) - z[np.arange(128), win[1:]]).sum())
    return tot / (len(WIN) * 128)

res = dict(shape=tag(d, L), recipe=RECIPE, loss_windows=NWL, snaps={})
print(f"remove {name}", flush=True)
for t in STEPS:
    t0 = time.time(); M = machine_file(d, L, RECIPE, t)
    variants = {"finished": "finished", "pushes": "pushes", "blank": "blank"}
    if t > 0:
        pl = WK["snaps"][str(t)]["plates"]
        cl = {k: min(max(o["gamma"], 0.0), 1.0) for k, o in pl.items() if "gamma" in o and o["gamma"] < 1}
        cp = {k: min(max(IP["snaps"][str(t)]["in_place"][KEYPART(k)]["s"], 0.0), 1.0)
              for k in [f"{n}.{l}" for l in range(L) for n in ("qkv", "proj", "fc", "mproj")] + ["wte", "wpe", "head"]
              if IP["snaps"][str(t)]["in_place"][KEYPART(k)]["s"] < 1}
        variants["iv_light"] = make_iv(cl); variants["iv_place"] = make_iv(cp)
        coefs = dict(iv_light=cl, iv_place=cp)
    else:
        coefs = {}
    rec = dict(coefs=coefs)
    fin = None
    for vn, v in variants.items():
        m = Machine(d, L, RECIPE, t, v, M=M)
        tops, top5, shas = run(m, TEXTS); gr = greedy(m, TEXTS); lo = loss(m)
        r = dict(loss=lo, top1=tops.tolist(), greedy=gr.tolist())
        if vn in ("finished", "pushes"):
            r["identity_piles_same_as_machine_py"] = sum(a == b for a, b in zip(shas, WJ["snaps"][str(t)][vn]["piles_sha256"]))
        if vn == "finished":
            fin = dict(tops=tops, greedy=gr, loss=lo)
            r["greedy_pairing_control"] = int((gr == np.roll(gr, 1, 0)).sum())
            r["top1_pairing_control"] = int((tops == np.roll(tops, 1, 0)).sum())
            pr = np.roll(gr, 1, 0)
            r["greedy_first_difference_pairing_control"] = float(np.mean([int(np.argmax(a != b)) if (a != b).any() else len(a) for a, b in zip(gr, pr)]))
        r["top1_as_finished"] = int((tops == fin["tops"]).sum())
        r["finished_in_top5"] = int(sum(fin["tops"][i, j] in top5[i][j] for i in range(len(TEXTS)) for j in range(NREAD)))
        r["greedy_as_finished"] = int((gr == fin["greedy"]).sum())
        first = [int(np.argmax(row != frow)) if (row != frow).any() else len(row) for row, frow in zip(gr, fin["greedy"])]
        r["greedy_words_before_first_difference_mean"] = float(np.mean(first))
        r["loss_cost"] = lo - fin["loss"]
        r["float_loss_128_windows"] = EV["snaps"][str(t)][vn]["held_loss"] if vn in EV["snaps"][str(t)] else None
        rec[vn] = r
        print(f"  step {t:5d} {vn:9s} loss {lo:.4f} (+{lo - fin['loss']:.4f})" + (f" (float, 128 windows: {r['float_loss_128_windows']:.4f})" if r["float_loss_128_windows"] is not None else "")
              + f"  top1 as finished {r['top1_as_finished']}/480  finished in top5 {r['finished_in_top5']}  greedy {r['greedy_as_finished']}/480"
              f" (first difference after {r['greedy_words_before_first_difference_mean']:.1f})"
              + (f"  identity {r['identity_piles_same_as_machine_py']}/30" if "identity_piles_same_as_machine_py" in r else "")
              + (f"  pairing control top1 {r['top1_pairing_control']} greedy {r['greedy_pairing_control']} first difference {r['greedy_first_difference_pairing_control']:.1f}" if vn == "finished" else ""), flush=True)
    rec["seconds"] = time.time() - t0
    res["snaps"][t] = rec
    json.dump(res, open(os.path.join(HERE, "evals", f"remove_{name}.json"), "w"))
print("done")
