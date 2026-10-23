"""Report of fringes/fringe_hists.npz (made by fringe_map.py). Whole-number counts throughout; shares printed as
percent. Place 15 is the top place (the sign), place 0 the bottom. Percentiles are nearest-rank: the smallest |tip| v
with at least that fraction of meetings at or below v.
    python3 fringes/fringe_report.py
"""
import json, numpy as np
Z = np.load("/Users/abundancemachine/streaming-machine/fringes/fringe_hists.npz")
POS = json.loads(str(Z["positions"]))
BANKS = [f"{k}.{l}" for l in range(12) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]

def rank(h, num, den):                                   # nearest-rank percentile num/den of a histogram over |tip| (last axis)
    c = h.cumsum(-1); C = c[..., -1:]
    return (c * den < C * num).sum(-1)
def pct(a, b, digits=2): return f"{100 * a / b:.{digits}f}%"

print(f"piles from the counts not equal to the machine's (all banks, all texts): {int(Z['bad'])}")
print("bank      N  pos   meetings     |tip| real: med p90 p99  fringe   | control: med p90 p99  fringe  | above-chance pairs  "
      "| pile magnitude carried by them: real (control)")
SUM = {}
for k in BANKS:
    R = Z[k]; Cn = Z[k + "__control"]; n = R.shape[0]; half = R.shape[-1] - 1; N = 2 * half
    v = np.arange(half + 1)
    hr, hc = R.sum((0, 1)), Cn.sum((0, 1)); C = int(hr.sum()); assert C == int(hc.sum())
    fr, fc_ = 2 * int((v * hr).sum()), 2 * int((v * hc).sum())           # sum of 2|tip| = fringe pixels
    mr, mc = rank(R, 1, 2), rank(Cn, 1, 2)                               # per place pair medians (n, n)
    diff = mr - mc; above = diff >= 1; below = diff <= -1
    w = [[1 << (a + b) for b in range(n)] for a in range(n)]
    sr = [[w[a][b] * int((v * R[a, b]).sum()) for b in range(n)] for a in range(n)]
    sc = [[w[a][b] * int((v * Cn[a, b]).sum()) for b in range(n)] for a in range(n)]
    totr = sum(map(sum, sr)); totc = sum(map(sum, sc))
    Ar = sum(sr[a][b] for a in range(n) for b in range(n) if above[a, b]); Ac = sum(sc[a][b] for a in range(n) for b in range(n) if above[a, b])
    print(f"{k:8s} {N:5d} {POS[k]:4d} {C:12,d}   {rank(hr,1,2):4d} {rank(hr,9,10):3d} {rank(hr,99,100):3d}  {pct(fr, C * N)}  "
          f"|   {rank(hc,1,2):4d} {rank(hc,9,10):3d} {rank(hc,99,100):3d}  {pct(fc_, C * N)}  |  {int(above.sum()):3d} above, {int(below.sum())} below"
          f"  |  {pct(Ar, totr, 1)} ({pct(Ac, totc, 1)})")
    SUM[k] = dict(mr=mr, mc=mc, above=above, below=below, Ar=Ar, totr=totr, Ac=Ac, totc=totc, fr=fr, fc=fc_, C=C, N=N)

print("\nwhere the above-chance pairs sit (real median |tip| / control median |tip|), light place : plate places")
for k in BANKS:
    s = SUM[k]; mr, mc, above, below = s["mr"], s["mc"], s["above"], s["below"]; n = mr.shape[0]
    ctl = np.sort(mc.ravel()); typ = f"control typical {int(ctl[len(ctl) // 2])} (range {int(ctl[0])}-{int(ctl[-1])})"
    parts = []
    for a in range(n - 1, -1, -1):
        bs = [b for b in range(n - 1, -1, -1) if above[a, b]]
        if bs: parts.append(f"L{a}: " + " ".join(f"P{b}={mr[a, b]}/{mc[a, b]}" for b in bs))
    lows = [f"L{a}xP{b}={mr[a, b]}/{mc[a, b]}" for a in range(n) for b in range(n) if below[a, b]]
    la = [a for a in range(n) if above[a].any()]; pb = [b for b in range(n) if above[:, b].any()]
    box = f"light places {min(la)}-{max(la)}, plate places {min(pb)}-{max(pb)}" if la else "none"
    corner = bool((above[:14, :].sum() == 0) and (above[:, :12].sum() == 0))
    print(f"{k:8s} {typ}; box: {box}; inside light top 2 x plate top 4: {corner}")
    for p in parts: print(f"           {p}")
    if lows: print(f"           below control: {' '.join(lows)}")

import os
P2 = "/Users/abundancemachine/streaming-machine/fringes/fringe_hists_control2.npz"
if os.path.exists(P2):
    Y = np.load(P2)
    print("\nSECOND CONTROL (another random order of each position's lines). Per bank: place pairs whose median |tip| differs by"
          " >= 1 between the two controls; pairs above BOTH controls by >= 1; below both by >= 1; fringe share of control 2;"
          " and the ratio of mean |tip| real / control (control = both controls together) at the top pair and its range over all pairs")
    for k in BANKS:
        R = Z[k]; C1 = Z[k + "__control"]; C2 = Y[k + "__control"]; n = R.shape[0]; half = R.shape[-1] - 1; N = 2 * half
        v = np.arange(half + 1)
        m1, m2, mr = rank(C1, 1, 2), rank(C2, 1, 2), rank(R, 1, 2)
        flick = int((np.abs(m1 - m2) >= 1).sum())
        up = (mr - m1 >= 1) & (mr - m2 >= 1); dn = (mr - m1 <= -1) & (mr - m2 <= -1)
        h2 = C2.sum((0, 1)); f2 = 2 * int((v * h2).sum())
        sr = (R * v).sum(-1); sc = (C1 * v).sum(-1) + (C2 * v).sum(-1)       # sums of |tip| per pair; control sum is over 2 controls
        ratio = 2 * sr / sc
        la = [x for x in range(n) if up[x].any()]; pb = [x for x in range(n) if up[:, x].any()]
        box = f"light {min(la)}-{max(la)} x plate {min(pb)}-{max(pb)}" if la else "none"
        dla = [x for x in range(n) if dn[x].any()]; dpb = [x for x in range(n) if dn[:, x].any()]
        dbox = f"light {min(dla)}-{max(dla)} x plate {min(dpb)}-{max(dpb)}" if dla else "none"
        print(f"{k:8s} controls differ at {flick:3d} pairs | above both: {int(up.sum()):3d} ({box}) | below both: {int(dn.sum()):3d} ({dbox})"
              f" | control-2 fringe {pct(f2, int(h2.sum()) * N)} | mean|tip| real/control: top pair L15xP15 {ratio[15, 15]:.3f}, over all pairs {ratio.min():.3f}-{ratio.max():.3f}")
