"""THE NEWEST FILM, A LOOK AT THE DARK GRAINS (reads only; nothing in the machine changes). film_develop_more.log: when
the 31 smallest-|part| grains per layer take the token-alone column's shares (the other 3,041 developed from the light),
the top-5 set is kept 6/30; when 31 random grains do, 21/30. A grain's part = light x bend output, so a grain whose
bend is shut has a part near 0 however much pile stands behind it. Given the token-alone column's share, that pile
goes through.

For each text and layer (newest column, the machine's own run; H = the pile arriving at the bend, share = its dimming,
share_al = the token-alone column's share): the 31 smallest-|part| grains, 31 random grains, the 31 largest-|part|
grains:
  shut      share below 1% of full open (655 of 65,536)
  |dB|      |rshift(H x share_al, 16) - bend output|: how far the bend output moves if the pile stood and only the
            share changed (units)
  held      sum over the grains of (lead's light at the bend output) x dB, all 12 layers' grains together, every other
            dimming held: the change the exact relation gives (whole numbers; shown as a multiple of the lead)

PREDICTION (Claude's, written before running): over half the smallest-|part| grains are shut, against about a fifth of
random grains; their |dB| is several times the random grains'; their held change is larger than the random grains'
and comparable to the lead or more.
    python3 reverse/film_dark.py        -> reverse/film_dark.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from machine import *
from fractions import Fraction

t0 = time.time()
D = np.load("/Users/abundancemachine/streaming-machine/reverse/film_split.npz")
GN, Ge, BO, SH, SHA, LEAD = D["GboN"], D["Gboe"], D["Bout"], D["share"], D["share_al"], D["lead"]
NT = GN.shape[0]; rng = np.random.default_rng(20261007)
SETS = ["smallest |part|", "random", "largest |part|"]
shut = {s: [] for s in SETS}; dB = {s: [] for s in SETS}; held = {s: [] for s in SETS}; shut_all = []
for i in range(NT):
    rec = record(TEXTS[i]); t = rec["t"]; tot = {s: Fraction(0) for s in SETS}
    for l in range(12):
        H = rec["layers"][l]["fc"]["out"][t]; s_ = rec["layers"][l]["bend"]["share"][t]
        assert (s_ == SH[i, l]).all() and (rshift(H * s_, 16) == BO[i, l]).all()
        moved = rshift(H * SHA[i, l], 16) - BO[i, l]
        order = sorted(range(3072), key=lambda n: -abs(int(GN[i, l, n]) * int(BO[i, l, n])))
        pick = {"smallest |part|": order[-31:], "random": list(rng.choice(3072, 31, replace=False)), "largest |part|": order[:31]}
        shut_all.append(float((s_ < 655).mean()))
        for s in SETS:
            ix = np.array(pick[s])
            shut[s].append(float((s_[ix] < 655).mean())); dB[s].append(float(np.abs(moved[ix]).mean()))
            tot[s] += Fraction(sum(int(GN[i, l, n]) * int(moved[n]) for n in ix)) * Fraction(2) ** int(Ge[i, l])
    for s in SETS: held[s].append(float(tot[s]) / int(LEAD[i]))
    print(f"  text {i:2d} held change / lead: " + ", ".join(f"{s} {held[s][-1]:+.2f}" for s in SETS) + f"  ({time.time() - t0:.0f}s)", flush=True)

print(f"\nnewest column, all grains: shut (share < 1% open) {np.mean(shut_all):.1%}")
print("   31 grains per layer      | shut   | mean |dB| (units) | held change of the lead, all 12 layers' grains: median |.| / lead, [10th, 90th] of |.|")
for s in SETS:
    h = np.abs(np.array(held[s]))
    print(f"   {s:24s} | {np.mean(shut[s]):6.1%} | {np.mean(dB[s]):17.0f} | {np.median(h):6.2f}  [{np.percentile(h, 10):.2f}, {np.percentile(h, 90):.2f}]")
print(f"({time.time() - t0:.0f}s)")
