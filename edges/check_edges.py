"""Independent rebuild of edges.py's core numbers at depths 0, 3, 7, 11, by another route:
  - the forward is reverse/machine.py's record() (its own copy of the machine's steps), not edges/forward.py;
  - the fixed part is formed as ONE light vector (position row + stream biases, and from depth 3 the whole light's
    value on line 447 in place of theirs there), met with the plate in plain int64 (numpy integer products, no float);
  - "taken away" is formed on the light, not the pile: (x16 x 2^sh - the write) met with the plate;
  - every side and table entry decided in Python whole numbers only (no float bound).
Compared with edges.json: bright %, the fixed part on its true side %, within one table step %, and the edges that
leave their side when the newest write, depth k-1's two writes, or the word row is taken away.
    python3 edges/check_edges.py
"""
import sys, os, json, time, numpy as np
np.seterr(all="ignore")
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
from machine import record, W, M, EXP2, ES, PHI, gauge_log, product
os.chdir("/Users/abundancemachine/streaming-machine")
refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
DEPTHS = (0, 3, 7, 11); STONE = 447
J = json.load(open("edges/edges.json"))["report"]
t0 = time.time()
RECS = []
for t, s in enumerate(TEXTS):
    rec = record(s, check=False)
    RECS.append(dict(Rmid=[rec["layers"][l]["Rmid"] for l in range(12)], H=[rec["layers"][l]["fc"]["out"] for l in range(12)],
                     A=[rec["layers"][l]["proj"]["out"] for l in range(12)], Mw=[rec["layers"][l]["mproj"]["out"] for l in range(12)]))
    del rec
print(f"forward of 30 texts by record(): {time.time() - t0:.0f}s", flush=True)
WTE, WPE = M["wte"], M["wpe"]

def py(f): return np.frompyfunc(f, 2, 1)
mul_pow = py(lambda a, ee: a << ee if ee >= 0 else a)           # helpers on object arrays
def side_int(N, E, e, B):
    """N x E x 2^e + B > 0, Python whole numbers: multiply through by 2^max(0, -e)"""
    N, E, e, B = (np.asarray(x, dtype=object) for x in np.broadcast_arrays(N, E, e, B))
    out = np.empty(N.shape, bool)
    for i, (n_, E_, e_, b_) in enumerate(zip(N.ravel(), E.ravel(), e.ravel(), B.ravel())):
        out.flat[i] = ((n_ * E_ << e_) + b_ if e_ >= 0 else n_ * E_ + (b_ << -e_)) > 0
    return out
def entry_int(N, E, e, B):
    N, E, e, B = (np.asarray(x, dtype=object) for x in np.broadcast_arrays(N, E, e, B))
    out = np.empty(N.shape, np.int64)
    for i, (n_, E_, e_, b_) in enumerate(zip(N.ravel(), E.ravel(), e.ravel(), B.ravel())):
        x = n_ * E_; h = (x << e_ if e_ >= 0 else (x + (1 << (-e_ - 1))) >> (-e_)) + b_
        out.flat[i] = min(max(((h + 8) >> 4) + 32768, -1), 65536)
    return out

res = {}
for k in DEPTHS:
    w, we, b = W[f"fc.{k}"]; wi = w.astype(np.int64)
    sb = sum(W[f"proj.{l}"][2] for l in range(k + 1)) + sum((W[f"mproj.{l}"][2] for l in range(k)), np.zeros(768, np.int64))
    cnt = dict(n=0, bright=0, fixed=0, within=0, newest=0, pair=0, word=0)
    for t, s in enumerate(TEXTS):
        P = len(s); r = RECS[t]; R = r["Rmid"][k]; H = r["H"][k]
        assert np.array_equal(R, WTE[s] + WPE[:P] + sum(r["A"][l] for l in range(k + 1)) + sum((r["Mw"][l] for l in range(k)), np.zeros((P, 768), np.int64)))
        x16, ex, LS = gauge_log(R); sh = ex + ES
        F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
        I, E = F >> 16, EXP2[F & 0xFFFF].astype(np.int64); e = I - 16 - sh[:, None]
        Lfix = WPE[:P] + sb[None, :]
        if k >= 3: Lfix = Lfix.copy(); Lfix[:, STONE] = R[:, STONE]
        Nfix = Lfix @ wi                                            # int64, exact
        sl = slice(1, P)
        side = H[sl] > 0
        fs = side_int(Nfix[sl], E[sl], e[sl], b[None, :])
        ce = entry_int(Nfix[sl], E[sl], e[sl], b[None, :]); ct = np.clip(((H[sl] + 8) >> 4) + 32768, -1, 65536)
        X = x16 << sh[:, None]
        def gone(Sw):                                               # H with the light Sw taken away, pile-side exact
            Nw = (X - Sw) @ wi; Nt = X @ wi
            return side_int((Nw - Nt)[sl], E[sl], e[sl], H[sl])
        An = r["A"][k] - W[f"proj.{k}"][2]
        cnt["newest"] += int((gone(An) != side).sum())
        if k >= 1:
            pair = (r["A"][k - 1] - W[f"proj.{k - 1}"][2]) + (r["Mw"][k - 1] - W[f"mproj.{k - 1}"][2])
            cnt["pair"] += int((gone(pair) != side).sum())
        cnt["word"] += int((gone(WTE[s]) != side).sum())
        cnt["n"] += side.size; cnt["bright"] += int(side.sum()); cnt["fixed"] += int((fs == side).sum())
        cnt["within"] += int((np.abs(ce - ct) <= 1).sum())
    n = cnt["n"]; mine = dict(bright=100 * cnt["bright"] / n, fixed=100 * cnt["fixed"] / n, within=100 * cnt["within"] / n,
                             newest=100 * cnt["newest"] / n, pair=100 * cnt["pair"] / n if k >= 1 else None, word=100 * cnt["word"] / n)
    rp = J[str(k)]["real"]
    theirs = dict(bright=rp["bright"], fixed=rp["fixed"], within=rp["within"], newest=rp["flips"]["newest"][0],
                  pair=rp["flips"]["pair k-1"][0] if k >= 1 else None, word=rp["flips"]["word"][0])
    same = all((mine[q] is None and theirs[q] is None) or abs(mine[q] - theirs[q]) < 1e-9 for q in mine)
    print(f"depth {k:2d}: " + "  ".join(f"{q} {mine[q]:.4f}|{theirs[q]:.4f}" for q in mine if mine[q] is not None)
          + f"   counts {cnt}   {'SAME' if same else 'DIFFERENT'}   ({time.time() - t0:.0f}s)", flush=True)
    res[k] = same
print(f"rebuilt the same at {sum(res.values())}/{len(res)} depths")
