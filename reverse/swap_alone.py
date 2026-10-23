"""THE NEXT WAY, FOURTH STEP: IS THE NEWEST COLUMN'S BEND PATTERN THE TOKEN'S OWN? The text is written in the newest
column's bends (swap_bends.log). If those bends were the token's own -- the shares its light gets alone at that
position (its arrival over only itself, everything else as the machine) -- they could be stored once per word and
position. Here, for each of the 30 refs texts (5 tokens): run the last token alone at position 4 (the arrival at
position 4 sees only position 4, every other position as the machine), take every layer's bend shares at position 4,
and put them into the machine's own run at position 4 (all other dimmings set by the text's own light). Does the answer
stay? Also: share of the newest column's bend shares exactly equal alone vs in the text.

PREDICTION (Claude's, written before running): the answer leaves the top 5 in most texts (<= 5/30); exactly-equal
shares well under 5% (alone and in context were two different pages in thinpage/).
    python3 reverse/swap_alone.py       -> reverse/swap_alone.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

def rank_of(V, i): return int((V > V[i]).sum())
t0 = time.time(); NT = len(TEXTS); ans, ctl, eq = [], [], []
L24 = int(log2fix([1 << 24])[0])
for i in range(NT):
    ids = TEXTS[i]; A = record(ids); SA = A["top"]["scores"]; wA = race(SA); P = A["P"]; t = P - 1
    top5 = [int(x) for x in np.argsort(-SA, kind="stable")[:5]]
    arr = []
    for Ly in A["layers"]:
        Aa, La = Ly["arr"]["A"].copy(), Ly["arr"]["L"].copy()
        Aa[:, t, :] = 0; Aa[:, t, t] = 1 << 24; La[:, t] = L24          # position t's arrival over only itself
        arr.append(dict(arr=dict(A=Aa, L=La)))
    alone = record(ids, held=dict(layers=arr))
    assert all((alone["layers"][l]["bend"]["share"][:t] == A["layers"][l]["bend"]["share"][:t]).all() for l in range(12))
    sh = []
    for l in range(12):
        s = A["layers"][l]["bend"]["share"].copy(); al = alone["layers"][l]["bend"]["share"][t]
        eq.append(float((s[t] == al).mean())); s[t] = al; sh.append(s)
    fz = record(ids, held=dict(layers=[dict(bend=dict(share=s)) for s in sh]))["top"]["scores"]
    ans.append(rank_of(fz, wA)); ctl += [rank_of(fz, w) for w in top5[1:]]
    print(f"  text {i:2d} {word(wA)!r:12s} rank with the token-alone bends in the newest column: {ans[-1]}", flush=True)
a = np.array(ans); k = np.array(ctl)
print(f"\nnewest column's bends from the token alone: A's answer top 5 {int((a < 5).sum())}/{NT}, first {int((a == 0).sum())}/{NT},"
      f" median rank {np.median(a):.0f};  A's 2nd..5th in top 5 {100 * (k < 5).mean():.1f}%")
print(f"  newest column's bend shares exactly equal alone vs in the text: {100 * np.mean(eq):.2f}%   ({time.time() - t0:.0f}s)")
