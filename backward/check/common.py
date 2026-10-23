"""INDEPENDENT CHECKER of the backward run (backward/check/, the checker's own files).

The machine is base2.py's own functions (exec'd from its source, as judge5.py does). Everything else here is written
from scratch for the check: the forward record, every interval step, the prover of the top 5, the diagonal tables.
Nothing from backward/core.py, top.py, stops.py, req.py or diag.py is imported or copied.
Whole numbers only; float64/float32 are used only as exact adders of whole-number products, with the size of every
sum asserted below the format's exact range.
"""
import os, sys, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; CK = f"{HERE}/backward/check"

_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
for a_, b_ in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"),
               ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('M = np.load("base2_gpt2.npz")', f'M = np.load("{HERE}/base2_gpt2.npz")'),
               ('W = {k: (M[k + ".w"].astype(np.float64), M[k + ".e"], M[k + ".b"]) for k in [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]}', 'W = {}')]:
    assert _src.count(a_) == 1, a_
    _src = _src.replace(a_, b_)
G = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, G); sys.argv = _argv
M, W = G["M"], G["W"]
rshift, to16, product, race, log2fix, power = G["rshift"], G["to16"], G["product"], G["race"], G["log2fix"], G["power"]
gauge_log, gauged_bank, plain_bank, edge, attention = G["gauge_log"], G["gauged_bank"], G["plain_bank"], G["edge"], G["attention"]
ES = G["ES"]; PHI = np.asarray(G["PHI"]).astype(np.int64); ARR = np.asarray(G["ARR"]).astype(np.int64)
LOG2 = np.asarray(G["LOG2"]).astype(np.int64); EXP2 = np.asarray(G["EXP2"]).astype(np.int64)
LOG768, EPS40 = int(G["LOG768"]), int(G["EPS40"])
# the tables' shapes the interval steps rely on (checked, not assumed)
assert (np.diff(LOG2) >= 0).all() and LOG2[-1] < 65536 and LOG2[0] >= 0
assert (np.diff(EXP2) >= 0).all() and EXP2[0] == 65536 and EXP2[-1] < 2 * 65536
assert (np.diff(PHI) >= 0).all() and PHI[0] >= 0 and PHI[-1] <= 65536

WTE = M["wte"]; WPE = M["wpe"]
HW = M["head.w"]; HE = M["head.e"].astype(np.int64); HB = M["head.b"].astype(np.int64)
assert (HE == HE[0]).all(); HE0 = int(HE[0]); NW = HW.shape[1]

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6

def load_layer(l):
    W.clear()
    for nm in ("qkv", "proj", "fc", "mproj"):
        k = f"{nm}.{l}"; W[k] = (M[k + ".w"].astype(np.float64), M[k + ".e"], M[k + ".b"])

def head_scores(x16, ex, LS):
    out = np.empty((len(x16), NW), np.int64); xf = x16.astype(np.float64)
    F = (HE0 + ex[:, None] + ES) * 65536 - LS[:, None]
    for j in range(0, NW, 8192):
        p = xf @ HW[:, j:j + 8192].astype(np.float64); assert np.abs(p).max() < 2 ** 52
        out[:, j:j + 8192] = power(p.astype(np.int64), F) + HB[j:j + 8192]
    return out

def top5_of(S):
    """the five largest, in order (judge5's top5)"""
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; o = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, o, 1)

def forward_keep(ids, keep=11):
    """base2.py's forward over every position of one text, keeping layer `keep`'s values and the final light"""
    P = len(ids); R = WTE[ids] + WPE[:P]; rec = {}
    for l in range(12):
        load_layer(l)
        if l == keep: rec["Rin"] = R.copy()
        QKV = gauged_bank(R, f"qkv.{l}"); ATT = attention(QKV); A = plain_bank(ATT, f"proj.{l}"); R = R + A
        if l == keep: rec.update(QKV=QKV, ATT=ATT, A=A, Rmid=R.copy())
        H = gauged_bank(R, f"fc.{l}"); E = edge(H); Mw = plain_bank(E, f"mproj.{l}"); R = R + Mw
        if l == keep: rec.update(H=H, E=E, Mw=Mw, Rout=R.copy())
    W.clear()
    x16, ex, LS = gauge_log(R)
    S = head_scores(x16, ex, LS)
    rec.update(x16=x16, ex=ex, LS=LS, top5=top5_of(S), S=S)
    return rec

# ------------------------------------------------------------------ whole-number interval steps (the checker's own)
def nbits(v):
    """bit length of |v|, elementwise, whole numbers only"""
    v = np.abs(np.asarray(v, np.int64)); n = np.zeros(v.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        big = v >= (np.int64(1) << s); n = n + np.where(big, s, 0); v = np.where(big, v >> s, v)
    return n + (v > 0)

def lvl_range(lo, hi):
    """every water level (to16's shift) a box can produce, per row: [s_min, s_max]. The level is set by the largest
    |line|; over the box that largest |line| lies between max_i min|x_i| and max_i max|x_i|, and the level is monotone in it."""
    big = np.maximum(np.abs(lo), np.abs(hi)).max(1)
    small = np.where((lo <= 0) & (hi >= 0), 0, np.minimum(np.abs(lo), np.abs(hi))).max(1)
    return np.maximum(nbits(small) - 15, 0), np.maximum(nbits(big) - 15, 0)

def eps_of(ex):
    e = 40 + 2 * np.asarray(ex, np.int64)
    return np.array([EPS40 >> int(v) if v >= 0 else EPS40 << int(-v) for v in np.atleast_1d(e)], np.int64)

def log2fix_arr(V):
    V = np.asarray(V, np.int64); assert (V > 0).all()
    t = nbits(V) - 1
    m = np.where(t >= 16, V >> np.maximum(t - 16, 0), V << np.maximum(16 - t, 0)) & 0xFFFF
    return t * 65536 + LOG2[m]

def gauge_range(xlo, xhi, ex):
    """the gauge's log LS over every light in the box (water level ex fixed): V = 768 S2 - S1^2 + eps, log2fix and the
    halving are monotone in V, so bound V. S2 over the box: per line the smallest and largest square; S1 likewise."""
    sq_hi = np.maximum(xlo * xlo, xhi * xhi)
    sq_lo = np.where((xlo <= 0) & (xhi >= 0), 0, np.minimum(xlo * xlo, xhi * xhi))
    s1lo, s1hi = xlo.sum(1), xhi.sum(1)
    s1sq_hi = np.maximum(s1lo * s1lo, s1hi * s1hi)
    s1sq_lo = np.where((s1lo <= 0) & (s1hi >= 0), 0, np.minimum(s1lo * s1lo, s1hi * s1hi))
    e = eps_of(ex)
    Vlo = np.maximum(768 * sq_lo.sum(1) - s1sq_hi, 0) + e       # the machine's V is never below 0 + eps (Cauchy)
    Vhi = 768 * sq_hi.sum(1) - s1sq_lo + e
    return (log2fix_arr(Vlo) >> 1) - LOG768 + ex * 65536, (log2fix_arr(Vhi) >> 1) - LOG768 + ex * 65536

def gauge_exact(x16, ex):
    """the gauge's log of an exact light (base2's gauge_log, from the light and its level)"""
    S1 = x16.sum(1); S2 = (x16 * x16).sum(1); V = 768 * S2 - S1 * S1 + eps_of(ex)
    return (log2fix_arr(V) >> 1) - LOG768 + ex * 65536

def dot_box(lo, hi, w16):
    """[lo, hi] (rows x n) met with a plate w16 (n x m whole numbers): each pile's exact smallest and largest value.
    smallest = lo . w+ + hi . w-, largest = hi . w+ + lo . w-"""
    w16 = np.asarray(w16).astype(np.int64)                        # (int16's -32768 has no int16 absolute value)
    wp = np.maximum(w16, 0).astype(np.float64); wn = np.minimum(w16, 0).astype(np.float64)
    lof, hif = lo.astype(np.float64), hi.astype(np.float64)
    bound = np.maximum(np.abs(lo), np.abs(hi)).astype(np.float64) @ np.abs(w16).astype(np.float64)
    assert bound.max(initial=0) < 2 ** 52
    return (lof @ wp + hif @ wn).astype(np.int64), (hif @ wp + lof @ wn).astype(np.int64)

def power_py(X, F):
    """base2's power with Python whole numbers (no int64 limit): rshift(X * EXP2[f], 16 - I)"""
    X = np.asarray(X, np.int64); F = np.broadcast_to(np.asarray(F, np.int64), X.shape)
    small = (np.abs(X) < (1 << 44))
    out = power(np.where(small, X, 0), F)
    for ix in zip(*np.nonzero(~small)):
        x = int(X[ix]); f = int(F[ix]); I, fr = f >> 16, f & 0xFFFF; v = x * int(EXP2[fr]); s = 16 - I
        out[ix] = (v + (1 << (s - 1))) >> s if s > 0 else v << (-s)
    return out

def power_box(plo, phi, Flo, Fhi):
    """power over pile in [plo, phi] and F in [Flo, Fhi]: monotone in the pile for each F, monotone in F for each pile
    (rising for a pile >= 0, falling for a pile < 0), so the extremes sit at the four corners"""
    c = [power_py(plo, Flo), power_py(plo, Fhi), power_py(phi, Flo), power_py(phi, Fhi)]
    return np.minimum(np.minimum(c[0], c[1]), np.minimum(c[2], c[3])), np.maximum(np.maximum(c[0], c[1]), np.maximum(c[2], c[3]))

# the bend on an interval: open side (H >= 0) rises; the shut side read from an exact range table of every value
_NEGMAX = 1 << 20
_neg = edge(-np.arange(0, _NEGMAX + 1, dtype=np.int64))         # _neg[k] = edge(-k)
_ZERO_FROM = int(np.flatnonzero(_neg != 0).max()) + 1            # edge(-k) = 0 for every k >= _ZERO_FROM (checked on 2^20)
assert (_neg[_ZERO_FROM:] == 0).all() and (edge(-np.array([2 ** 21, 2 ** 25, 2 ** 31, 2 ** 40], np.int64)) == 0).all()
_pos = edge(np.arange(0, 1 << 20, dtype=np.int64)); assert (np.diff(_pos) >= 0).all()
_TBL = _neg[:_ZERO_FROM + 1]
_SMIN, _SMAX = [_TBL], [_TBL]
while (1 << len(_SMIN)) <= len(_TBL):
    h = 1 << (len(_SMIN) - 1)
    _SMIN.append(np.minimum(_SMIN[-1][:-h], _SMIN[-1][h:])); _SMAX.append(np.maximum(_SMAX[-1][:-h], _SMAX[-1][h:]))
def _range(T, i, j, f):
    n = j - i + 1; k = nbits(n) - 1; out = np.empty(i.shape, np.int64)
    for kk in np.unique(k):
        s = k == kk; out[s] = f(T[kk][i[s]], T[kk][j[s] - (1 << kk) + 1])
    return out
def edge_box(lo, hi):
    """the bend's exact smallest and largest output over every whole number in [lo, hi]"""
    shp = lo.shape; lo = lo.ravel().astype(np.int64); hi = hi.ravel().astype(np.int64)
    olo = np.full(lo.shape, np.iinfo(np.int64).max); ohi = np.full(lo.shape, np.iinfo(np.int64).min)
    m = hi >= 0                                                   # the open part [max(lo,0), hi]
    if m.any():
        olo[m] = np.minimum(olo[m], edge(np.maximum(lo[m], 0))); ohi[m] = np.maximum(ohi[m], edge(hi[m]))
    m = lo <= -1                                                  # the shut part [lo, min(hi,-1)]: k = -H in [kmin, kmax]
    if m.any():
        kmin = -np.minimum(hi[m], -1); kmax = -lo[m]
        far = kmax > _ZERO_FROM                                   # reaches the all-zero tail
        kmx = np.minimum(kmax, _ZERO_FROM); kmn = np.minimum(kmin, _ZERO_FROM)
        a = _range(_SMIN, kmn, kmx, np.minimum); b = _range(_SMAX, kmn, kmx, np.maximum)
        a = np.where(far, np.minimum(a, 0), a); b = np.where(far, np.maximum(b, 0), b)
        olo[m] = np.minimum(olo[m], a); ohi[m] = np.maximum(ohi[m], b)
    return olo.reshape(shp), ohi.reshape(shp)

# ------------------------------------------------------------------ the prover (the checker's own)
def prove_order(branches, t5, slab=8192):
    """Is t5 (the machine's five words, in order) the top 5, set and order, strictly, for EVERY light in every branch?
    branches: list of dicts with xlo, xhi (rows x 768 light boxes), ex (rows), LSlo, LShi (rows), valid (rows).
    Requirements: score[t0] > score[t1] > ... > score[t4] > score[j] for every other word j.
    Each requirement (a, b) in a branch: first by the two words' own score intervals; else by the pair bound
        score_a - score_b >= floor(Dlo * EXP2[f] / 2^s) + bias_a - bias_b  (the smaller of the gauge's two ends),
    with Dlo the exact smallest pile difference over the box (rounding: floor(u+1/2) - floor(v+1/2) >= floor(u-v)).
    Returns per row: proved (bool), and the number of requirements that needed the pair bound."""
    R = len(t5); ok = np.ones(R, bool); npair = np.zeros(R, np.int64)
    for b in branches:
        v = b["valid"]; xlo, xhi = b["xlo"], b["xhi"]
        base = (HE0 + b["ex"] + ES) * 65536; Flo = base - b["LShi"]; Fhi = base - b["LSlo"]
        Slo = np.empty((R, NW), np.int64); Shi = np.empty((R, NW), np.int64)
        for j in range(0, NW, slab):
            plo, phi = dot_box(xlo, xhi, HW[:, j:j + slab])
            a_, b_ = power_box(plo, phi, Flo[:, None], Fhi[:, None]); Slo[:, j:j + slab] = a_ + HB[j:j + slab]; Shi[:, j:j + slab] = b_ + HB[j:j + slab]
        for r in range(R):
            if not v[r] or not ok[r]: continue
            t = t5[r]; need = [(t[k], t[k + 1]) for k in range(4) if not Slo[r, t[k]] > Shi[r, t[k + 1]]]
            rest = np.ones(NW, bool); rest[t] = False
            ch = np.flatnonzero(rest & (Shi[r] >= Slo[r, t[4]]))
            pairs = need + [(t[4], int(j)) for j in ch]
            npair[r] += len(pairs)
            if not pairs: continue
            A = np.array([p[0] for p in pairs]); Bw = np.array([p[1] for p in pairs])
            for q in range(0, len(A), 4096):
                aa, bb = A[q:q + 4096], Bw[q:q + 4096]
                d = HW[:, aa].astype(np.int64) - HW[:, bb].astype(np.int64)        # (768, pairs)
                Dlo = (xlo[r].astype(np.float64) @ np.maximum(d, 0).astype(np.float64) + xhi[r].astype(np.float64) @ np.minimum(d, 0).astype(np.float64))
                assert np.abs(Dlo).max() < 2 ** 50; Dlo = Dlo.astype(np.int64)
                lbs = []
                for F in (int(Flo[r]), int(Fhi[r])):
                    I, f = F >> 16, F & 0xFFFF; s = 16 - I; e = int(EXP2[f]); assert 1 <= s <= 62
                    assert np.abs(Dlo).max() * e < 2 ** 62
                    lbs.append(((Dlo * e) >> s) + HB[aa] - HB[bb])
                if not (np.minimum(lbs[0], lbs[1]) >= 1).all(): ok[r] = False; break
    anyv = np.zeros(R, bool)
    for b in branches: anyv |= b["valid"]
    return ok & anyv, npair

def final_branches(Rlo, Rhi):
    """a box on the final stream -> one branch per water level it allows: the light box and the gauge's range"""
    s0, s1 = lvl_range(Rlo, Rhi); out = []
    for o in range(int((s1 - s0).max()) + 1):
        sh = s0 + o; valid = sh <= s1; ex = sh - ES
        xlo = rshift(Rlo, sh[:, None]); xhi = rshift(Rhi, sh[:, None])
        LSlo, LShi = gauge_range(xlo, xhi, ex)
        out.append(dict(xlo=xlo, xhi=xhi, ex=ex, LSlo=LSlo, LShi=LShi, valid=valid))
    return out, s1 - s0
