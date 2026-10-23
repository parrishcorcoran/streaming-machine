"""What is left to store when the blank's remainder is removed: W - round(v x W0) against W, per plate kind. 2026-10-06.

His words: "If the blank slate were saved somewhere you would be able to remove EXACTLY each blank weight exactly"
SESSION'S READING (the session's words, not his): what is left of each blank weight inside the finished weight is c x it.
Prediction in PREDICTION.txt (written before this first ran).

On the shared 16-place grid of ../base2_pythia-<size>.npz, for every plate kind (qkv, proj, fc, mproj, head, embed):
  W     the finished plate
  Dk    W - round(c x W0), c from the 241 never-built rows (../foundation_check.json c_med)
  Dp    W - round(P x W0), P the settings' product
  Dk~1-3, Dp~1-3   the same with W0 scrambled within each plate (the same 3 draws as remove_exact.py)
Measures, as ../foundation_share.py: lit places a weight = nonzero digits of the single-wave signed-digit writing
(popcount(|x| ^ 3|x|)); bits a weight at the floor = entropy of a bell curve with each neuron's own spread, rounded to
whole numbers on its grid (neurons: plates' output columns, the head's word columns, the input table's word rows).
The input table is also split: the 241 never-built rows, and the other 50,063.

    python3 store_exact.py size=14m
"""
import os, sys, json, math, time, zlib, resource
import numpy as np

A = dict(a.split("=") for a in sys.argv[1:])
SIZE = A.get("size", "14m")
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
M = np.load(os.path.join(UP, f"base2_pythia-{SIZE}.npz")); L = int(M["L"])
CHK = json.load(open(os.path.join(UP, "foundation_check.json")))[f"{SIZE}@143000"]
C_NB, PROD = CHK["embed"]["c_med"], CHK["product"]; NB = np.array(CHK["embed"]["ids"])
def to30(s): return int(round(s * 2 ** 30))
def times(v30, q): return (v30 * q + (1 << 29)) >> 30
V30 = {"k": to30(C_NB), "p": to30(PROD)}
DRAWS = {"1": 20261006, "2": 2, "3": 3}
t0 = time.time()
LOGF = open(os.path.join(HERE, f"store_exact_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30

def rshift(X, s):
    s = np.asarray(s, dtype=np.int64)
    up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)

def scrambled_start(name, seed):                                      # the same scramble as remove_exact.py
    S = M[name + ".s"]; e = M[name + ".e"].astype(np.int16); assert -128 <= int(e.min()) and int(e.max()) < 128
    shape, size = S.shape, S.size
    E = np.ascontiguousarray(np.broadcast_to(e[:, None] if name == "embed" else e[None, :], shape)).ravel()
    pk = S.ravel().astype(np.int32); del S; pk <<= 8; pk |= (E + 128).astype(np.int32)
    np.random.default_rng(seed + zlib.crc32(name.encode())).shuffle(pk)
    out = np.empty(size, np.int32)
    for a in range(0, size, 1 << 22):
        b = min(size, a + (1 << 22)); src = (pk[a:b] >> 8).astype(np.int64)
        sh = ((pk[a:b] & 255) - 128).astype(np.int64) - E[a:b].astype(np.int64)
        out[a:b] = rshift(src, -sh)
    return out.reshape(shape)

def naf(x): a = np.abs(x); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def hbits(sig):
    sig = np.asarray(sig, np.float64); out = np.zeros_like(sig)
    big = sig >= 2; out[big] = np.log2(sig[big] * math.sqrt(2 * math.pi * math.e))
    for i in np.flatnonzero((sig > 0) & ~big):
        s = sig[i]; k = np.arange(-int(14 * s) - 3, int(14 * s) + 4)
        p = np.array([0.5 * (math.erf((kk + .5) / (s * math.sqrt(2))) - math.erf((kk - .5) / (s * math.sqrt(2)))) for kk in k])
        p = p[p > 0]; out[i] = float(-(p * np.log2(p)).sum())
    return out

def neurons(X, name): return X if name == "embed" else X.T             # neurons as rows

FORMS = ["W", "Dk", "Dp"] + [f"{f}~{dr}" for f in ("Dk", "Dp") for dr in DRAWS]
def stats(name):
    """per neuron (rows): sum, sum of squares, lit places, for every form"""
    W = neurons(M[name + ".w"], name)
    S0 = neurons(M[name + ".s"], name)
    SS = {dr: neurons(scrambled_start(name, sd), name) for dr, sd in DRAWS.items()}
    R, C = W.shape; out = {f: [np.zeros(R, np.int64) for _ in range(3)] for f in FORMS}
    B = max(1, (1 << 21) // C)
    for a in range(0, R, B):
        b = min(R, a + B); w = np.ascontiguousarray(W[a:b]).astype(np.int64); w0 = np.ascontiguousarray(S0[a:b]).astype(np.int64)
        xs = {"W": w, "Dk": w - times(V30["k"], w0), "Dp": w - times(V30["p"], w0)}
        for dr in DRAWS:
            ws = np.ascontiguousarray(SS[dr][a:b]).astype(np.int64)
            xs[f"Dk~{dr}"] = w - times(V30["k"], ws); xs[f"Dp~{dr}"] = w - times(V30["p"], ws)
        for f, x in xs.items():
            out[f][0][a:b] = x.sum(1); out[f][1][a:b] = (x * x).sum(1); out[f][2][a:b] = naf(x).sum(1)
    return out, C

def summ(st, C, rows=None):
    """lit places a weight and bits a weight at the floor, over the chosen neurons"""
    res = {}
    for f, (s, q, lit) in st.items():
        if rows is not None: s, q, lit = s[rows], q[rows], lit[rows]
        sig = np.sqrt(np.maximum(q / C - (s / C) ** 2, 0.0))
        res[f] = dict(lit=float(lit.sum() / (len(s) * C)), bits=float(hbits(sig).mean()), values=int(len(s) * C))
    return res

log(f"store_exact.py  size {SIZE}:  c {C_NB:.6e} ({V30['k']} x 2^-30),  settings' product {PROD:.6e} ({V30['p']} x 2^-30)")
names = [f"{nm}.{l}" for l in range(L) for nm in ("qkv", "proj", "fc", "mproj")] + ["head", "embed"]
KIND = {}; RES = dict(size=SIZE, c=C_NB, product=PROD, kinds={}, embed_split={})
for name in names:
    st, C = stats(name); k = name.split(".")[0]
    if name == "embed":
        others = np.setdiff1d(np.arange(st["W"][0].size), NB)
        RES["embed_split"] = dict(never_built=summ(st, C, NB), other_rows=summ(st, C, others))
    one = summ(st, C)
    if k not in KIND: KIND[k] = {f: dict(lit=0.0, bits=0.0, values=0) for f in FORMS}
    for f in FORMS:
        v = one[f]["values"]; KIND[k][f]["lit"] += one[f]["lit"] * v; KIND[k][f]["bits"] += one[f]["bits"] * v; KIND[k][f]["values"] += v
    del st
tot = {f: dict(lit=0.0, bits=0.0, values=0) for f in FORMS}
for k, d_ in KIND.items():
    for f in FORMS:
        v = d_[f]["values"]; tot[f]["lit"] += d_[f]["lit"]; tot[f]["bits"] += d_[f]["bits"]; tot[f]["values"] += v
        d_[f] = dict(lit=d_[f]["lit"] / v, bits=d_[f]["bits"] / v, values=v)
KIND["all"] = {f: dict(lit=tot[f]["lit"] / tot[f]["values"], bits=tot[f]["bits"] / tot[f]["values"], values=tot[f]["values"]) for f in FORMS}
RES["kinds"] = KIND

def line(lab, d_):
    W = d_["W"]; rng = lambda f: [d_[f"{f}~{dr}"] for dr in DRAWS]
    lk = rng("Dk"); lp = rng("Dp")
    return (f"  {lab:12s} {W['values']:>11,} values | lit/wt W {W['lit']:.4f}  Dk {d_['Dk']['lit']:.4f} ({d_['Dk']['lit'] / W['lit'] - 1:+.2%})  "
            f"Dp {d_['Dp']['lit']:.4f} ({d_['Dp']['lit'] / W['lit'] - 1:+.2%})  scrambled Dk {min(x['lit'] for x in lk) / W['lit'] - 1:+.2%}..{max(x['lit'] for x in lk) / W['lit'] - 1:+.2%}  "
            f"Dp {min(x['lit'] for x in lp) / W['lit'] - 1:+.2%}..{max(x['lit'] for x in lp) / W['lit'] - 1:+.2%} | bits/wt W {W['bits']:.4f}  "
            f"Dk {d_['Dk']['bits'] - W['bits']:+.5f}  Dp {d_['Dp']['bits'] - W['bits']:+.5f}  scrambled Dk {min(x['bits'] for x in lk) - W['bits']:+.5f}..{max(x['bits'] for x in lk) - W['bits']:+.5f}  "
            f"Dp {min(x['bits'] for x in lp) - W['bits']:+.5f}..{max(x['bits'] for x in lp) - W['bits']:+.5f}")
log("  per kind: lit places a weight (change against W) | bits a weight at the floor (change against W, minus = saved)")
for k in ["qkv", "proj", "fc", "mproj", "head", "embed", "all"]: log(line(k, KIND[k]))
log(line("embed: 241", RES["embed_split"]["never_built"]))
log(line("embed: rest", RES["embed_split"]["other_rows"]))
json.dump(RES, open(os.path.join(HERE, f"store_exact_{SIZE}.json"), "w"), indent=1)
log(f"done {time.time() - t0:.0f}s  peak memory {mem():.2f} GB")
