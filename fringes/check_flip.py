"""CHECKER (own code). The flip on the whole machine, up/down form, every meeting (48 banks, q.k, shares x v, top read).
Disagreement counts come from 0/1 place rows met by an overlap product (whole numbers below 2^24, exact):
    d_ab = (ones in light row a) + (ones in plate row b) - 2 (lines where both are 1)
Machine A: light and plate both flipped (v -> -v-1), pile from the flipped counts with the machine's own stacks.
Machine B: only the plate flipped, pile = 1/4 [ -S' - ts - ws + N ].
Each machine runs forward on its own piles; every pile is checked against base2's whole-number product.
    python3 fringes/check_flip.py [texts=30]
"""
import os, sys, json, time, resource, numpy as np
from collections import defaultdict
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")].replace('M[k + ".w"].astype(np.float64)', 'M[k + ".w"]')
G = {"__name__": "b2"}; _argv = sys.argv; sys.argv = ["x"]; exec(src, G); sys.argv = _argv
M, W = G["M"], G["W"]
for k in list(W): W[k] = (M[k + ".w"], W[k][1], W[k][2])          # int16 plates
WTE, WPE = M["wte"], M["wpe"]
NAME = {id(W[k][0]): k for k in W}
refs = json.load(open("refs_many.json")); texts = [(r["ids"] + r["ref"])[:5] for r in refs][:NT]
OUT = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"

def exact(x, w):
    p = x.astype(np.float64) @ np.asarray(w).astype(np.float64); assert np.abs(p).max() < 2 ** 52
    return p.astype(np.int64)

def kind(x, w):
    nm = NAME.get(id(w))
    if nm is not None: return nm.split(".")[0]
    return "qk" if w.shape[0] == 64 else "sv"

def xplanes(u, n):                                   # (n*P, N) float32 0/1, row a*P + p
    return ((u[None] >> np.arange(n, dtype=np.int64)[:, None, None]) & 1).astype(np.float32).reshape(n * u.shape[0], -1)

def dis(Xp, cx, Uw, b, n, P):                        # disagreements of every light place a with plate place b: (n, P, K)
    Wb = ((Uw >> b) & 1).astype(np.float32)
    O = (Xp @ Wb).reshape(n, P, -1)
    return (cx[:, :, None] + Wb.sum(0, dtype=np.float64)[None, None, :] - 2 * O.astype(np.float64)).astype(np.int64)

ST = defaultdict(int); NS = defaultdict(set); LIGHT = {}
def make(mode):
    def prod(x, w):
        kd = kind(x, w); x = np.asarray(x).astype(np.int64); wi = np.asarray(w).astype(np.int64 if np.asarray(w).size < 10 ** 7 else np.int32)
        P, N = x.shape; K = wi.shape[1]
        if mode == "plain":
            nm = NAME.get(id(w))
            if nm is not None: LIGHT[nm] = x.copy()
            return exact(x, wi)
        n = max(int(np.abs(x).max()).bit_length(), int(np.abs(wi).max()).bit_length()) + 1; off = 1 << (n - 1)
        assert n <= 28; NS[kd].add(n)
        if K > 8192:                                 # the top read: same n for the whole plate, taken in column pieces
            return np.concatenate([core(x, wi[:, c:c + 8192], kd, n, off) for c in range(0, K, 8192)], 1)
        return core(x, wi, kd, n, off)
    def core(x, wi, kd, n, off):
        P, N = x.shape; K = wi.shape[1]
        p0 = exact(x, wi); ts = 2 * x.sum(1) + N; ws = 2 * wi.sum(0, dtype=np.int64) + N
        xf, wf = -x - 1, -wi - 1
        dt = np.int32 if n <= 30 else np.int64
        Xp = xplanes(x + off, n); cx = Xp.reshape(n, P, N).sum(-1, dtype=np.float64)
        Uw = (wi + off).astype(dt); Uf = (wf + off).astype(dt)
        assert Uw.min() >= 0 and Uw.max() < 2 ** n and Uf.min() >= 0 and Uf.max() < 2 ** n
        if mode == "A": Xq = xplanes(xf + off, n); cxq = Xq.reshape(n, P, N).sum(-1, dtype=np.float64)
        else: Xq, cxq = Xp, cx
        pw = (np.int64(1) << np.arange(n, dtype=np.int64))[:, None, None]
        S = np.zeros((P, K), np.int64); Sq = np.zeros((P, K), np.int64); bad = 0
        for b in range(n):
            d = dis(Xp, cx, Uw, b, n, P)
            dq = dis(Xq, cxq, Uf, b, n, P)
            bad += int((dq != d).sum()) if mode == "A" else int((dq != N - d).sum())
            S += ((N - 2 * d) * pw).sum(0) << b
            Sq += ((N - 2 * dq) * pw).sum(0) << b
        ST[(mode, kd, "meet")] += n * n * P * K; ST[(mode, kd, "piles")] += P * K
        ST[(mode, kd, "count_bad")] += bad
        base = S - ts[:, None] - ws[None, :] + N
        ST[(mode, kd, "orig_pile_bad")] += int(((base % 4) != 0).sum() + (base // 4 != p0).sum())
        if mode == "A":
            ST[(mode, kd, "S_changed")] += int((Sq != S).sum())
            v = Sq - ts[:, None] - ws[None, :] + N
            pile = v // 4; ST[(mode, kd, "pile_bad")] += int(((v % 4) != 0).sum() + (pile != p0).sum())
            tsf, wsf = 2 * xf.sum(1) + N, 2 * wf.sum(0, dtype=np.int64) + N
            ST[(mode, kd, "stack_not_negated")] += int((tsf != -ts).sum() + (wsf != -ws).sum())
            q = (Sq - tsf[:, None] - wsf[None, :] + N) // 4
            ST[(mode, kd, "flipstack_ne_(x+1)(w+1)")] += int((q != exact(xf, wf)).sum())
            ST[(mode, kd, "flipstack_ne_pile+(ts+ws)/2")] += int((q != p0 + (ts[:, None] + ws[None, :]) // 2).sum())
            ST[(mode, kd, "flipstack_eq_pile")] += int((q == p0).sum())
            ST[(mode, kd, "ts+ws==0")] += int(((ts[:, None] + ws[None, :]) == 0).sum())
            return pile
        ST[(mode, kd, "S_not_negated")] += int((Sq != -S).sum())
        v = -Sq - ts[:, None] - ws[None, :] + N
        pile = v // 4; ST[(mode, kd, "pile_bad")] += int(((v % 4) != 0).sum() + (pile != p0).sum())
        pp = (Sq - ts[:, None] + ws[None, :] + N) // 4                     # the flipped plate's own pile (its stack is -ws)
        ST[(mode, kd, "p'_ne_x(-w-1)")] += int((pp != exact(x, wf)).sum())
        ST[(mode, kd, "pile_ne_-p'-sumx")] += int((pile != -pp - x.sum(1)[:, None]).sum())
        return pile
    return prod

def forward(prod):
    G["product"] = prod; T = len(texts)
    R = np.concatenate([WTE[ids] + WPE[:5] for ids in texts])
    for l in range(12):
        Q = G["gauged_bank"](R, f"qkv.{l}")
        A = np.concatenate([G["attention"](Q[t * 5:(t + 1) * 5]) for t in range(T)])
        R = R + G["plain_bank"](A, f"proj.{l}")
        R = R + G["plain_bank"](G["edge"](G["gauged_bank"](R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = G["gauge_log"](R[4::5])
    return G["power"](G["product"](x16, w), (we[None, :] + ex[:, None] + G["ES"]) * 65536 - LS[:, None]) + b

t0 = time.time()
S0 = forward(make("plain"))
np.savez(f"{OUT}/lights.npz", **LIGHT)
G["product"] = make("plain")                         # base2's own per-text forward, for the word
same_word = sum(G["race"](S0[t]) == G["forward"](ids) for t, ids in enumerate(texts))
print(f"plain batched machine: word equals base2.forward on {same_word}/{len(texts)} texts ({time.time() - t0:.0f}s)", flush=True)
res = {}
for mode in ("A", "B"):
    t1 = time.time(); Sm = forward(make(mode))
    eq = sum(np.array_equal(Sm[t], S0[t]) for t in range(len(texts)))
    wd = sum(G["race"](Sm[t]) == G["race"](S0[t]) for t in range(len(texts)))
    print(f"machine {mode}: final scores array_equal {eq}/{len(texts)}, same word {wd}/{len(texts)}  ({time.time() - t1:.0f}s)", flush=True)
    res[mode] = (eq, wd)
for k in sorted(ST): print(f"  {k[0]} {k[1]:6s} {k[2]:32s} {ST[k]:,}")
for m in ("A", "B"):
    print(f"  {m}: meetings {sum(v for k, v in ST.items() if k[0] == m and k[2] == 'meet'):,}  piles {sum(v for k, v in ST.items() if k[0] == m and k[2] == 'piles'):,}")
print("  places n by kind:", {k: sorted(v) for k, v in NS.items()})
print(f"  peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB, total {time.time() - t0:.0f}s")
