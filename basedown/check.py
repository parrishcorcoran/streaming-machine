"""BASEDOWN/CHECK: a separate rebuild of lineup.py's and combine.py's counts at four banks, in numpy only (no walk.c,
no gwalk.c, no lineup()/group_lineup() code): the gaps by sorting each side on its own, the walk, the cancel share,
the landings (every product w x x, token by token), the piles by prefix sums of w x gap scattered into the tokens;
and combine.py's Q' landed, rebuilt group by group from the direct group parts in each group's own order (np.lexsort
on the interleaved places, written again here). Compared with basedown/out/lineup.json and combine.json.
    python3 basedown/check.py -> basedown/out/check.log
"""
import json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/basedown/out"
LOG = open(f"{OUT}/check.log", "w"); T0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
import sys; sys.path.insert(0, f"{HERE}/groups2"); _a = sys.argv; sys.argv = [sys.argv[0]]
import uh_rebuild as R; import uh_perplace as PP
sys.argv = _a
LJ = {r["bank"]: r for r in json.load(open(f"{OUT}/lineup.json"))["rec"]}
CJ = {r["bank"]: r for r in json.load(open(f"{OUT}/combine.json"))["rec"]}
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); NPOS = REF["npos"].astype(int); P = int(NPOS.sum())
st = np.concatenate([[0], np.cumsum(NPOS)]); SOUPS = {"a": [np.arange(st[i], st[i + 1]) for i in range(30)], "b": [np.arange(P)]}
Wz = np.load(f"{HERE}/base2_gpt2.npz"); bad = 0
for name in ("qkv.3", "proj.7", "fc.10", "mproj.1"):
    kind, l = name.split("."); l = int(l)
    x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[l], np.int64)
    w = np.asarray(Wz[f"{name}.w"], np.int64); N, M = w.shape; nrow = naf(w).sum(1); ref = np.asarray(REF[f"{kind}_pile_{l}"])
    land = sum(int(ones(x[t][:, None] * w).sum()) for t in range(P))
    for sm in ("a", "b"):
        walk = g1 = v1 = 0; pile = np.zeros((P, M), np.int64)
        for rows in SOUPS[sm]:
            for j in range(N):
                v = x[rows, j]
                for side in (v > 0, v < 0):
                    t = rows[side]; vv = v[side]
                    if not len(t): continue
                    o = np.argsort(np.abs(vv), kind="stable"); t = t[o]; vv = vv[o]
                    gap = np.diff(np.concatenate([[0], vv]))
                    walk += int(naf(gap).sum()) * int(nrow[j]); g1 += int(ones(gap).sum()); v1 += int(ones(vv).sum())
                    if sm == "b": pile[t] += np.cumsum(gap[:, None] * w[j][None, :], 0)
        r = LJ[name][sm]; ok = (walk == r["walk"], g1 == r["g1"], v1 == r["v1"], land == r["land"])
        msg = f"  {name} ({sm}) walk {walk} vs {r['walk']}, gaps' 1s {g1} vs {r['g1']}, values' 1s {v1} vs {r['v1']}, landings {land} vs {r['land']}"
        if sm == "b": pd = int((pile != ref).any(1).sum()); ok += (pd == 0,); msg += f"; piles by prefix sums differing from plain_refs {pd}"
        log(msg + ("  SAME" if all(ok) else "  DIFFERENT")); bad += not all(ok)
    # combine.py's Q' landed, rebuilt
    up, _ = PP.rule(w, [(kind, 0, M)], 12); kc = R.prepare(w, up, [(kind, 0, M)])["kcorr"]
    gl, go = R.make_groups(N, 12, [1, R.SLABS.index(kind), l, 12])
    for sm in ("a", "b"):
        q = 0
        for k in range(len(go) - 1):
            L = np.sort(gl[go[k]:go[k + 1]]); part = 2 * (x[:, L] @ w[L]) - kc[None, :] * x[:, L].sum(1)[:, None]   # Q' of each token, direct
            q += int(ones(part).sum())        # Q' at a token does not depend on the order it was walked in: the order only sets the steps
        r = CJ[name][sm]; ok = q == r["qland"]; bad += not ok
        log(f"  {name} ({sm}) Q' landed {q} vs combine.json {r['qland']}  {'SAME' if ok else 'DIFFERENT'}")
log(f"CHECK {'PASSED' if bad == 0 else 'FAILED'} ({time.time() - T0:.0f}s)")
