"""THE NEXT WAY, THIRD STEP: DOES THE NEWEST COLUMN'S BEND PATTERN STAND FROM THE COLUMN BEFORE? swap_bends.py found the
text written mostly in the last position's bends (B's bends there alone: answer in the top 5 1/30; B's bends at the
earlier positions with A's own at the last: 18/30). In the streaming machine the earlier columns already stand. Here,
at the last position only, every layer's bend shares are copied from the position before it in the SAME text (A's own
light sets every other dimming). Does A's answer stay? Also counted: the share of the last column's bend shares
exactly equal to the column before (same text) and to the last column of another text (B, control).
A = each of the 30 refs texts (5 tokens), B = the next text.

PREDICTION (Claude's, written before running): the answer leaves the top 5 in most texts (<= 8/30): each new token
rewrites its column's bends; exactly-equal shares are a few percent, about the same for the column before and for
another text's column.
    python3 reverse/swap_prev.py        -> reverse/swap_prev.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *

def rank_of(V, i): return int((V > V[i]).sum())
t0 = time.time(); NT = len(TEXTS); recs = [record(ids) for ids in TEXTS]
ans, ctl, eq_prev, eq_B, near_prev, near_B = [], [], [], [], [], []
for i in range(NT):
    A, B = recs[i], recs[(i + 1) % NT]; SA = A["top"]["scores"]; wA = race(SA)
    top5 = [int(x) for x in np.argsort(-SA, kind="stable")[:5]]
    sh = []
    for Ly, LyB in zip(A["layers"], B["layers"]):
        s = Ly["bend"]["share"].copy(); last, prev, other = s[4].copy(), s[3].copy(), LyB["bend"]["share"][4]
        s[4] = prev; sh.append(s)
        eq_prev.append(float((last == prev).mean())); eq_B.append(float((last == other).mean()))
        near_prev.append(float((np.abs(last - prev) <= 655).mean())); near_B.append(float((np.abs(last - other) <= 655).mean()))
    fz = record(TEXTS[i], held=dict(layers=[dict(bend=dict(share=s)) for s in sh]))["top"]["scores"]
    ans.append(rank_of(fz, wA)); ctl += [rank_of(fz, w) for w in top5[1:]]
    print(f"  text {i:2d} {word(wA)!r:12s} rank with the column before's bends at the last position: {ans[-1]}", flush=True)
a = np.array(ans); k = np.array(ctl)
print(f"\nlast column's bends copied from the column before (same text): A's answer top 5 {int((a < 5).sum())}/{NT}, first {int((a == 0).sum())}/{NT},"
      f" median rank {np.median(a):.0f};  A's 2nd..5th in top 5 {100 * (k < 5).mean():.1f}%")
print(f"  (for comparison, swap_bends.log: another text's last column there 1/{NT}; the coincidence control 1/{NT})")
print(f"last column's bend shares exactly equal to: the column before {100 * np.mean(eq_prev):.1f}%, another text's last column {100 * np.mean(eq_B):.1f}%")
print(f"   within 1% of full open (655 of 65,536): the column before {100 * np.mean(near_prev):.1f}%, another text's last column {100 * np.mean(near_B):.1f}%")
print(f"({time.time() - t0:.0f}s)")
