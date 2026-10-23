"""Rank each neuron's weights once (the plate, fixed). Pile = sum_k gap_k x (light above rank k): the gaps are the
tiny changes between neighbours in rank, the shared photons slipped. Exact (abelian). Counted: photon meetings
(1s x 1s) the plain way against the ranked way, on every bank, with the 30 pieces' real light. Base 2.
    python3 rank_gaps.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def pc(A):
    A = np.abs(A.astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += POP[A & 0xFFFF]; A >>= 16
    return n
lights = {}
def grab(name, x16): lights.setdefault(name, []).append(x16[-1].copy())
for ids in inputs:
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        x16, ex, _ = gauge_log(R); grab(f"qkv.{l}", x16); A = attention(gauged_bank(R, f"qkv.{l}"))
        grab(f"proj.{l}", to16(A, -ES)[0]); R = R + plain_bank(A, f"proj.{l}")
        x16, ex, _ = gauge_log(R); grab(f"fc.{l}", x16); H = edge(gauged_bank(R, f"fc.{l}"))
        grab(f"mproj.{l}", to16(H, -ES)[0]); R = R + plain_bank(H, f"mproj.{l}")
tot = {}
for name, xs in lights.items():
    w = M[name + ".w"].astype(np.int64)                     # (nin, nout): each neuron a column
    order = np.argsort(w, axis=0, kind="stable"); ws = np.take_along_axis(w, order, 0)
    gaps = np.diff(ws, axis=0, prepend=0)                   # first: the lowest weight itself; then the tiny changes
    pw, pg = pc(w), pc(gaps)
    t = tot.setdefault(name.split(".")[0], np.zeros(4, np.int64))
    for x in xs[:6]:                                        # six pieces' light a bank (enough to count)
        X = x.astype(np.int64)[order]                       # the light in each neuron's rank order
        S = np.cumsum(X[::-1], axis=0)[::-1]                # light above each rank
        if name == "fc.0" and x is xs[0]: assert np.array_equal((gaps * S).sum(0), (w * x[:, None]).sum(0)), "not exact"
        t += [(pw * pc(x)[:, None]).sum(), (pg * pc(S)).sum(), pw.sum(), pg.sum()]
print(f"  {'bank':6s} | photons in weights -> in gaps | photon meetings plain -> ranked")
for k, (mp, mr, w1, g1) in tot.items():
    print(f"  {k:6s} | {w1:>12,} -> {g1:>12,} ({100 * g1 / w1:4.0f}%) | {mp:>14,} -> {mr:>14,} ({100 * mr / mp:4.0f}%)")
