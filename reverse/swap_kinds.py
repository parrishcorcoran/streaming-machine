"""THE NEXT WAY, FIRST STEP: WHICH DIMMINGS CARRY THE TEXT? The reverse showed that within one text every word's reversed
pattern shares its sign on ~95% of entries, and between texts the same word's pattern agrees only at coin toss. So the
text lives in the dimmings. Three kinds of dimming: the water levels (one size per position per layer, and the top's),
the bends (one share per neuron per position per layer), the arrivals (the shares s_pj between positions).

For each text A (30 refs texts, 5 tokens) and B = the next text: run A's own words through the machine (its own
arithmetic) with ONE kind of dimming held from B and the other two set by A's own light; then with two kinds from B
and one kind A's own; then all three from B. Where does A's answer fall among all 50,257 words? Controls: A's 2nd..5th
words (frequency-matched); A's answer in B's own scores (coincidence); B's answer.
Also, the own-part look of (a) with its control: (R_answer - R_random) and (R_2nd - R_random) at the last position met
with every word row: where the input word and the word itself fall.

PREDICTION (Claude's, written before running; his prediction was not asked for this extra step):
  - B's water levels only: A's answer stays in the top 5 in most texts (>= 20/30) -- the light's size at each position
    and layer is much alike between texts.
  - B's bends only: the answer in the top 5 in few texts (<= 10/30).
  - B's arrival shares only: the answer in the top 5 in few texts (<= 10/30).
  - own part: the input word sits near the top for the 2nd word's own part as well (it is not the answer's alone).
    python3 reverse/swap_kinds.py       -> reverse/swap_kinds.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

def held_kinds(rec, kinds):
    L = []
    for Ly in rec["layers"]:
        d = {}
        if "water" in kinds: d["qkv"] = dict(LS=Ly["qkv"]["LS"]); d["fc"] = dict(LS=Ly["fc"]["LS"])
        if "bend" in kinds: d["bend"] = dict(share=Ly["bend"]["share"])
        if "arrival" in kinds: d["arr"] = dict(A=Ly["arr"]["A"], L=Ly["arr"]["L"])
        L.append(d)
    out = dict(layers=L)
    if "water" in kinds: out["top"] = dict(LS=rec["top"]["LS"])
    return out

def rank_of(V, i): return int((V > V[i]).sum())
def rows_met(Nl):
    two = np.stack([(Nl & 0xFFFF), (Nl >> 16)], 1).astype(np.float64); out = np.empty(WTE.shape[0], np.int64)
    for r0 in range(0, WTE.shape[0], 8192):
        ab = WTE[r0:r0 + 8192].astype(np.float64) @ two; assert np.abs(ab).max() < 2.0 ** 53
        out[r0:r0 + 8192] = (ab[:, 1].astype(np.int64) << 16) + ab[:, 0].astype(np.int64)
    return out

CONF = [("water",), ("bend",), ("arrival",), ("bend", "arrival"), ("water", "arrival"), ("water", "bend"), ("water", "bend", "arrival")]
t0 = time.time(); NT = len(TEXTS); recs = [record(ids) for ids in TEXTS]
for r, ids in zip(recs, TEXTS): assert race(r["top"]["scores"]) == forward(ids)
R = {c: dict(ans=[], wB=[], ctl=[], coinc=[]) for c in CONF}
for i in range(NT):
    A, B = recs[i], recs[(i + 1) % NT]; SA, SB = A["top"]["scores"], B["top"]["scores"]
    wA, wB = race(SA), race(SB); top5 = [int(x) for x in np.argsort(-SA, kind="stable")[:5]]
    for c in CONF:
        fz = record(TEXTS[i], held=held_kinds(B, c))["top"]["scores"]
        R[c]["ans"].append(rank_of(fz, wA)); R[c]["wB"].append(rank_of(fz, wB))
        R[c]["ctl"] += [rank_of(fz, w) for w in top5[1:]]; R[c]["coinc"].append(rank_of(SB, wA))
    print(f"  text {i:2d} answer {word(wA)!r:12s} ranks with B's " + ", ".join(f"{'+'.join(c)}: {R[c]['ans'][-1]}" for c in CONF)
          + f"  ({time.time() - t0:.0f}s)", flush=True)
pc = lambda x: f"{100 * x:.1f}%"
print("\nA's words, some dimmings held from B (the rest set by A's own light):  A's answer top 5 / first / median rank"
      " | A's 2nd..5th words top 5 | B's answer top 5, median rank")
for c in CONF:
    a = np.array(R[c]["ans"]); b = np.array(R[c]["wB"]); k = np.array(R[c]["ctl"])
    print(f"  from B: {'+'.join(c):24s} {int((a < 5).sum()):2d}/{NT}  {int((a == 0).sum()):2d}/{NT}  {np.median(a):7.0f}   | {pc((k < 5).mean()):>6s}"
          f" | {int((b < 5).sum()):2d}/{NT}, {np.median(b):.0f}")
co = np.array(R[CONF[0]]["coinc"])
print(f"  coincidence: A's answer in B's own top 5 {int((co < 5).sum())}/{NT}, median rank {np.median(co):.0f};  A's own dimmings: 30/30 first (the machine)")

print("\nOWN PART at the last position, with its control (16 places): where the input word and the word itself fall among 50,257 rows")
rng = np.random.default_rng(77); own = dict(in_ans=[], self_ans=[], in_2nd=[], self_2nd=[], in_r2=[], self_r2=[])
for i in range(NT):
    rec = recs[i]; S = rec["top"]["scores"]; P = rec["P"]; t = P - 1; win = TEXTS[i][-1]
    top2 = [int(x) for x in np.argsort(-S, kind="stable")[:2]]
    r1, r2 = [int(x) for x in rng.choice(50257, 2, replace=False)]
    (N, e), _ = reverse16(rec, top2 + [r1, r2])
    for k, name in [(0, "ans"), (1, "2nd"), (3, "r2")]:
        Do, eo = add16((N[k, t][None], e[k, t][None]), (-N[2, t][None], e[2, t][None]))
        V = rows_met(Do[0]); w = (top2 + [r1, r2])[k]
        own["in_" + name].append(rank_of(V, win)); own["self_" + name].append(rank_of(V, w))
for name, lab in [("ans", "answer - random"), ("2nd", "2nd word - random"), ("r2", "another random word - random")]:
    print(f"  {lab:30s} input word median rank {np.median(own['in_' + name]):6.0f} (top 100 in {sum(1 for x in own['in_' + name] if x < 100)}/{NT})"
          f" | the word itself median rank {np.median(own['self_' + name]):6.0f}")
print(f"({time.time() - t0:.0f}s)")
