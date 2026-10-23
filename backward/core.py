"""BACKWARD (backward/): the base-2 GPT-2 (base2.py's own steps, unchanged) recorded at every point of every layer,
and the same machine carried on BOXES: every value a whole-number interval [lo, hi] that provably holds it.

HIS WORDS (2026-10-08), the only ones quoted as his: "Start backward I think that's your problem. Start on the next
token and work backwards".

What this file gives the backward scripts (the session's own tools, labelled as such):
  record(ids)            base2.py's forward on one text, every value kept at every point of every layer:
                         Rin (stream into the layer), QKV (the gauged q|k|v), ATT (the arrival's output), A (the
                         arrival write), Rmid, H (the gauged fc value at the knife edge), E (the bend's output),
                         Mw (the gate write), Rout; the lights and piles of the four banks; the top light and scores.
  push(rec, rows_p, l, point, lo, hi)
                         the box [lo, hi] placed at one point of layer l for the given positions (every other value
                         of that position exact, every earlier position's keys and values exact), carried up through
                         the rest of the machine by exact whole-number interval steps, to the light at the top.
  certify(branches)      the top read on boxes: is the top 5 (set and order) the same for EVERY value the boxes allow?
                         Proved by bounds only (per word, then word against word with the shared gauge).

Every interval step is exact (the true value always inside), built from base2.py's own monotone pieces:
  rshift is monotone; to16's water level is set by the largest line, so a box gives a range of levels and each level
  is carried as a branch; the gauge's V = 768 S2 - S1^2 is bounded from the box's sums (and is never below the eps
  term, since 768 S2 >= S1^2); log2fix, the power table and the share table are monotone (checked below); the bend is
  monotone on the open side and is read from an exact range-min/max table of every value on the shut side; the
  arrival's clock start is the race's winner (a box on every score gives a box on the start), the arrival table is
  monotone in the distance, and the head's gauge is the arrivals' total, monotone.
A product of a box with a plate is exact: 2 x pile_lo = (lo + hi) . w - (hi - lo) . |w|, every sum below 2^52.
Whole numbers only. Float64 is used only as base2.py uses it, as an exact adder of whole-number products.
"""
import os, sys, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")     # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; BK = f"{HERE}/backward"; OUT = f"{BK}/out"

_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
for _a, _b in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"),
               ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('M = np.load("base2_gpt2.npz")', f'M = np.load("{HERE}/base2_gpt2.npz")'),
               ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')]:
    assert _src.count(_a) == 1, _a; _src = _src.replace(_a, _b)
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
W = B["W"]; assert W == {}
ES, SC = B["ES"], B["SC"]
PHI, ARR, LOG2, EXP2 = (np.asarray(B[k]).astype(np.int64) for k in ("PHI", "ARR", "LOG2", "EXP2"))
LOG768, EPS40 = int(B["LOG768"]), int(B["EPS40"])
rshift, to16, product, race, log2fix, power = B["rshift"], B["to16"], B["product"], B["race"], B["log2fix"], B["power"]
gauge_log, gauged_bank, plain_bank, edge, attention = B["gauge_log"], B["gauged_bank"], B["plain_bank"], B["edge"], B["attention"]
assert (np.diff(PHI) >= 0).all() and (np.diff(ARR) <= 0).all() and (np.diff(LOG2) >= 0).all() and (np.diff(EXP2) >= 0).all()
assert EXP2[0] == 65536 and EXP2[-1] < 131072 and LOG2[-1] < 65536    # the power and log tables join up monotone across places

_MZ = np.load(f"{HERE}/base2_gpt2.npz")
_w = _MZ["wte"]; assert np.abs(_w).max() < 2 ** 31
WTE, WPE = _w.astype(np.int32), _MZ["wpe"]; del _w          # int32 holds every entry (checked)
HEAD_W16 = _MZ["head.w"]; HEAD_E = _MZ["head.e"].astype(np.int64); HEAD_B = _MZ["head.b"].astype(np.int64)
assert (HEAD_E == HEAD_E[0]).all(); HE0 = int(HEAD_E[0])
NW = HEAD_W16.shape[1]

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)

_P16 = {}
def plate(name):
    """base2.py's own (w as float64, we, b); the int16 plates are kept, the float64 copy is made when used (memory)"""
    if name not in _P16:
        _P16[name] = (_MZ[name + ".w"], _MZ[name + ".e"].astype(np.int64), _MZ[name + ".b"].astype(np.int64))
    w16, we, b = _P16[name]
    return w16.astype(np.float64), we, b
def head_product(x16):
    """x16 met with the whole top table, exact, a slab of words at a time (memory)"""
    out = np.empty((len(x16), NW), np.int64); xf = np.asarray(x16).astype(np.float64)
    for j0 in range(0, NW, 8192):
        p = xf @ HEAD_W16[:, j0:j0 + 8192].astype(np.float64); assert np.abs(p).max(initial=0) < 2 ** 52; out[:, j0:j0 + 8192] = p
    return out
def head_box(lo, hi):
    """a box of lights met with the whole top table: the pile's exact interval, a slab of words at a time"""
    plo = np.empty((len(lo), NW), np.int64); phi = np.empty((len(lo), NW), np.int64)
    c = (lo + hi).astype(np.float64); r = (hi - lo).astype(np.float64)
    for j0 in range(0, NW, 8192):
        w = HEAD_W16[:, j0:j0 + 8192].astype(np.float64); c2 = c @ w; r2 = r @ np.abs(w)
        assert np.abs(c2).max(initial=0) < 2 ** 52 and np.abs(r2).max(initial=0) < 2 ** 52
        c2 = c2.astype(np.int64); r2 = r2.astype(np.int64); assert not ((c2 - r2) & 1).any()
        plo[:, j0:j0 + 8192] = (c2 - r2) >> 1; phi[:, j0:j0 + 8192] = (c2 + r2) >> 1
    return plo, phi

def top5(S):                                                 # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

# ---------------------------------------------------------------- the exact machine, recorded
LAYER_KEYS = ["Rin", "x16q", "exq", "LSq", "qkvpile", "QKV", "ATT", "x16a", "exa", "projpile", "A", "Rmid",
              "x16f", "exf", "LSf", "fcpile", "H", "E", "x16e", "exe", "mprojpile", "Mw", "Rout"]
KEEP_SMALL = ["Rin", "QKV", "ATT", "A", "Rmid", "H", "E", "Mw", "Rout", "exq", "LSq", "exa", "exf", "LSf", "exe"]
def record(ids, keep_from=0, keys=None, keep_only=None):
    """base2.py's forward on one text (every position), each value kept. Returns dict: L[l][key] (P x n), top dict.
    keep_from: layers below it are run but not kept (None in their place); keys: which values to keep (memory)."""
    P = len(ids); R = WTE[ids].astype(np.int64) + WPE[:P]; L = []
    for l in range(12):
        d = {"Rin": R}
        for k_ in [k_ for k_ in W if k_ != "head"]: del W[k_]
        for nm in ("qkv", "proj", "fc", "mproj"): W[f"{nm}.{l}"] = plate(f"{nm}.{l}")
        q = f"qkv.{l}"
        d["x16q"], d["exq"], d["LSq"] = gauge_log(R); d["qkvpile"] = product(d["x16q"], W[q][0])
        d["QKV"] = gauged_bank(R, q)
        d["ATT"] = attention(d["QKV"])
        p_ = f"proj.{l}"
        d["x16a"], d["exa"] = to16(d["ATT"], -ES); d["projpile"] = product(d["x16a"], W[p_][0])
        d["A"] = plain_bank(d["ATT"], p_)
        R = R + d["A"]; d["Rmid"] = R
        f = f"fc.{l}"
        d["x16f"], d["exf"], d["LSf"] = gauge_log(R); d["fcpile"] = product(d["x16f"], W[f][0])
        d["H"] = gauged_bank(R, f); d["E"] = edge(d["H"])
        m = f"mproj.{l}"
        d["x16e"], d["exe"] = to16(d["E"], -ES); d["mprojpile"] = product(d["x16e"], W[m][0])
        d["Mw"] = plain_bank(d["E"], m)
        R = R + d["Mw"]; d["Rout"] = R
        if l < keep_from or (keep_only is not None and l != keep_only): d = None
        elif keys is not None: d = {k_: d[k_] for k_ in keys}
        L.append(d)
    for k_ in list(W): del W[k_]
    x16, ex, LS = gauge_log(R)
    pile = head_product(x16)
    S = power(pile, (HE0 + ex[:, None] + ES) * 65536 - LS[:, None]) + HEAD_B
    top = dict(x16=x16, ex=ex, LS=LS, S=None if keys is not None else S, top5=top5(S), win=np.array([race(s) for s in S]))
    return dict(L=L, top=top, P=P, ids=list(ids))

def kv_cache(rec, l):
    """the keys (to16 per head, with their levels) and values of every position at layer l, exact"""
    if "kv" not in rec: rec["kv"] = {}
    if l not in rec["kv"]:
        Q = rec["L"][l]["QKV"]; P = len(Q)
        k16 = np.zeros((P, 12, 64), np.int64); ek = np.zeros((P, 12), np.int64)
        for h in range(12):
            k16[:, h], ek[:, h] = to16(Q[:, 768 + h * 64:768 + (h + 1) * 64], -ES)
        rec["kv"][l] = (k16, ek, Q[:, 1536:].reshape(P, 12, 64).copy())
    return rec["kv"][l]

# ---------------------------------------------------------------- whole-number helpers on arrays
def bitlen(v):
    v = np.abs(np.asarray(v, np.int64)).copy(); n = np.zeros(v.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        m = v >= (np.int64(1) << s); n += np.where(m, s, 0); v = np.where(m, v >> s, v)
    return n + (v > 0)

def log2fix_v(V):
    V = np.asarray(V, np.int64); assert (V > 0).all()
    t = bitlen(V) - 1
    m = np.where(t >= 16, np.right_shift(V, np.maximum(t - 16, 0)), np.left_shift(V, np.maximum(16 - t, 0))) & 0xFFFF
    return t * 65536 + LOG2[m]

def eps_term(ex):
    e = 40 + 2 * np.asarray(ex, np.int64)
    return np.where(e >= 0, np.right_shift(np.int64(EPS40), np.maximum(e, 0)), np.left_shift(np.int64(EPS40), np.maximum(-e, 0)))

def box_product(lo, hi, wf, wa=None):
    """[lo, hi] (rows x n) met with a plate (n x m, float64 whole numbers): the pile's exact interval"""
    c2 = (lo + hi).astype(np.float64) @ wf
    r2 = (hi - lo).astype(np.float64) @ (np.abs(wf) if wa is None else wa)
    assert np.abs(c2).max(initial=0) < 2 ** 52 and np.abs(r2).max(initial=0) < 2 ** 52
    c2 = c2.astype(np.int64); r2 = r2.astype(np.int64)
    assert not ((c2 - r2) & 1).any()
    return (c2 - r2) >> 1, (c2 + r2) >> 1

def power_safe(X, F):
    """base2's power(X, F) = rshift(X x EXP2[f], 16 - I), the same whole number; values too large for int64's product
    (only ever inside a wide box) are done with Python's whole numbers"""
    X = np.asarray(X, np.int64); F = np.broadcast_to(np.asarray(F, np.int64), X.shape)
    big = np.abs(X) >= (np.int64(1) << 45)
    if not big.any(): return power(X, F)
    out = power(np.where(big, 0, X), F)
    for ix in zip(*np.nonzero(big)):
        x = int(X[ix]); f = int(F[ix]); I, fr = f >> 16, f & 0xFFFF; v = x * int(EXP2[fr]); sh = 16 - I
        r = (v + (1 << (sh - 1))) >> sh if sh > 0 else v << (-sh)
        assert abs(r) < (1 << 62); out[ix] = r
    return out

def pow_box(plo, phi, Flo, Fhi):
    """power(pile, F) over pile in [plo, phi], F in [Flo, Fhi]: monotone in each, so the corners"""
    a, b = power_safe(plo, Flo), power_safe(plo, Fhi); c, d = power_safe(phi, Flo), power_safe(phi, Fhi)
    return np.minimum(a, b), np.maximum(c, d)

# ---------------------------------------------------------------- boxes through to16 and the gauge
MAXSPAN = 2                                                 # a box allowing more than 3 water levels at any reader
WIDE = None                                                 # is called blown (it spans a factor of 8 in its largest line)
def levels(lo, hi):
    """the range of water levels (to16's shift) the box allows, per row"""
    global WIDE
    mhi = np.maximum(np.abs(lo), np.abs(hi)).max(1)
    mlo = np.where((lo <= 0) & (hi >= 0), 0, np.minimum(np.abs(lo), np.abs(hi))).max(1)
    s0, s1 = np.maximum(bitlen(mlo) - 15, 0), np.maximum(bitlen(mhi) - 15, 0)
    wide = s1 - s0 > MAXSPAN
    if WIDE is not None and len(WIDE) == len(wide): WIDE |= wide
    return s0, np.where(wide, s0, s1)

def ls_box(xlo, xhi, ex):
    """the gauge's log (LS) over every light in the box at water level ex"""
    S1lo, S1hi = xlo.sum(1), xhi.sum(1)
    sqhi = np.maximum(xlo * xlo, xhi * xhi); sqlo = np.where((xlo <= 0) & (xhi >= 0), 0, np.minimum(xlo * xlo, xhi * xhi))
    S2lo, S2hi = sqlo.sum(1), sqhi.sum(1)
    s1max = np.maximum(S1lo * S1lo, S1hi * S1hi); s1min = np.where((S1lo <= 0) & (S1hi >= 0), 0, np.minimum(S1lo * S1lo, S1hi * S1hi))
    eps = eps_term(ex)
    Vlo = np.maximum(768 * S2lo - s1max, 0) + eps; Vhi = 768 * S2hi - s1min + eps   # 768 S2 - S1^2 is never below 0
    Vlo = np.maximum(Vlo, 1); Vhi = np.maximum(Vhi, 1)        # the machine's own V is a whole number above 0
    return (log2fix_v(Vlo) >> 1) - LOG768 + ex * 65536, (log2fix_v(Vhi) >> 1) - LOG768 + ex * 65536

def at_level(lo, hi, sh, s1):
    """the branch where the water level is sh: every line is below 2^(sh+15) there (the level is set by the largest
    line), so the box is clipped to that, and the branch is impossible if a line cannot be. Returns x16 box, valid."""
    cap = (np.int64(1) << (sh + 15)) - 1
    lo2 = np.maximum(lo, -cap[:, None]); hi2 = np.minimum(hi, cap[:, None])
    valid = (sh <= s1) & (lo2 <= hi2).all(1)
    lo2 = np.where(valid[:, None], lo2, 0); hi2 = np.where(valid[:, None], hi2, 0)
    return rshift(lo2, sh[:, None]), rshift(hi2, sh[:, None]), valid

def union(acc, lo, hi, valid):
    if acc is None:
        return np.where(valid[:, None], lo, np.iinfo(np.int64).max), np.where(valid[:, None], hi, np.iinfo(np.int64).min)
    return (np.where(valid[:, None], np.minimum(acc[0], lo), acc[0]), np.where(valid[:, None], np.maximum(acc[1], hi), acc[1]))

def i_gauged(Rlo, Rhi, name):
    """gauged_bank on a box: every water level the box allows, the gauge's range, the pile's interval, the corners"""
    wf, we, b = plate(name); wa = np.abs(wf)
    s0, s1 = levels(Rlo, Rhi); acc = None
    for o in range(int((s1 - s0).max()) + 1):
        sh = s0 + o; xlo, xhi, valid = at_level(Rlo, Rhi, sh, s1); ex = sh - ES
        LSlo, LShi = ls_box(xlo, xhi, ex)
        plo, phi = box_product(xlo, xhi, wf, wa)
        base = (we[None, :] + ex[:, None] + ES) * 65536
        olo, ohi = pow_box(plo, phi, base - LShi[:, None], base - LSlo[:, None])
        acc = union(acc, olo + b, ohi + b, valid)
    return acc

def i_plain(Xlo, Xhi, name):
    wf, we, b = plate(name); wa = np.abs(wf)
    s0, s1 = levels(Xlo, Xhi); acc = None
    for o in range(int((s1 - s0).max()) + 1):
        sh = s0 + o; xlo, xhi, valid = at_level(Xlo, Xhi, sh, s1); ex = sh - ES
        plo, phi = box_product(xlo, xhi, wf, wa)
        s = -(we[None, :] + ex[:, None] + ES)
        acc = union(acc, rshift(plo, s) + b, rshift(phi, s) + b, valid)
    return acc

# ---------------------------------------------------------------- the bend on a box
KNEG = 524296                                               # below -KNEG the share table reads 0: the bend gives 0
_EN = edge(-np.arange(0, KNEG + 1, dtype=np.int64))         # _EN[k] = edge(-k)
assert edge(np.array([-KNEG - 1, -KNEG - 50, -10 ** 9], np.int64)).tolist() == [0, 0, 0]
def _sparse(a, f):
    T = [a]; k = 1
    while (1 << k) <= len(a):
        prev = T[-1]; T.append(f(prev[:-(1 << (k - 1))], prev[(1 << (k - 1)):])); k += 1
    return T
_STMIN, _STMAX = _sparse(_EN, np.minimum), _sparse(_EN, np.maximum)
def _rq(T, i, j):                                            # min or max of _EN[i..j], i <= j
    n = j - i + 1; k = bitlen(n) - 1
    out = np.empty(i.shape, np.int64)
    for kk in np.unique(k):
        m = k == kk; t = T[kk]; out[m] = (np.minimum if T is _STMIN else np.maximum)(t[i[m]], t[j[m] - (1 << kk) + 1])
    return out
def i_edge(lo, hi):
    lo = np.asarray(lo, np.int64); hi = np.asarray(hi, np.int64); sh = lo.shape; lo = lo.ravel(); hi = hi.ravel()
    BIG = np.iinfo(np.int64).max; olo = np.full(lo.shape, BIG); ohi = np.full(lo.shape, -BIG)
    m = hi >= 0                                              # the open side: monotone
    if m.any():
        olo[m] = np.minimum(olo[m], edge(np.maximum(lo[m], 0))); ohi[m] = np.maximum(ohi[m], edge(hi[m]))
    m = lo <= -KNEG - 1                                      # far shut: 0
    olo[m] = np.minimum(olo[m], 0); ohi[m] = np.maximum(ohi[m], 0)
    a = np.maximum(lo, -KNEG); bb = np.minimum(hi, -1); m = a <= bb   # the shut side's table part
    if m.any():
        i, j = -bb[m], -a[m]
        olo[m] = np.minimum(olo[m], _rq(_STMIN, i, j)); ohi[m] = np.maximum(ohi[m], _rq(_STMAX, i, j))
    return olo.reshape(sh), ohi.reshape(sh)

# ---------------------------------------------------------------- the arrival on a box (one head at a time)
def arr_of(d):
    return np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)

def i_attention(rec, l, rows_p, Qlo, Qhi):
    """the arrival for positions rows_p, each with its own q|k|v box; every earlier position's key and value exact"""
    k16, ek, V = kv_cache(rec, l); R = len(rows_p); T = int(rows_p.max()) + 1
    tt = np.arange(T); past = tt[None, :] < rows_p[:, None]                  # earlier positions, per row
    out_lo = np.zeros((R, 768), np.int64); out_hi = np.zeros((R, 768), np.int64)
    for h in range(12):
        sl = slice(h * 64, (h + 1) * 64)
        qlo, qhi = Qlo[:, sl], Qhi[:, sl]; klo, khi = Qlo[:, 768 + h * 64:768 + (h + 1) * 64], Qhi[:, 768 + h * 64:768 + (h + 1) * 64]
        vlo, vhi = Qlo[:, 1536 + h * 64:1536 + (h + 1) * 64], Qhi[:, 1536 + h * 64:1536 + (h + 1) * 64]
        sq0, sq1 = levels(qlo, qhi); sk0, sk1 = levels(klo, khi); acc = None
        Kp = k16[:T, h].astype(np.float64); Vp = V[:T, h]                    # (T, 64)
        for oq in range(int((sq1 - sq0).max()) + 1):
            for ok in range(int((sk1 - sk0).max()) + 1):
                shq = sq0 + oq; shk = sk0 + ok
                q16lo, q16hi, vq = at_level(qlo, qhi, shq, sq1); eq = shq - ES
                k6lo, k6hi, vk = at_level(klo, khi, shk, sk1); eks = shk - ES; valid = vq & vk
                # scores against earlier positions (their keys exact)
                c2 = (q16lo + q16hi).astype(np.float64) @ Kp.T; r2 = (q16hi - q16lo).astype(np.float64) @ np.abs(Kp).T
                c2 = c2.astype(np.int64); r2 = r2.astype(np.int64)
                plo, phi = (c2 - r2) >> 1, (c2 + r2) >> 1                         # (R, T)
                sft = -(eq[:, None] + ek[None, :T, h] - 3 + SC)
                slo, shi = rshift(plo, sft), rshift(phi, sft)
                # the score against itself: both boxes
                cr = np.stack([q16lo * k6lo, q16lo * k6hi, q16hi * k6lo, q16hi * k6hi])
                pslo, pshi = cr.min(0).sum(1), cr.max(0).sum(1)
                sfs = -(eq + eks - 3 + SC)
                sslo, sshi = rshift(pslo, sfs), rshift(pshi, sfs)
                me = tt[None, :] == rows_p[:, None]
                slo = np.where(me, sslo[:, None], slo); shi = np.where(me, sshi[:, None], shi)
                inn = past | me
                NEG = -(1 << 62)
                mxlo = np.where(inn, slo, NEG).max(1); mxhi = np.where(inn, shi, NEG).max(1)
                dlo = np.maximum(mxlo[:, None] - shi, 0); dhi = np.maximum(mxhi[:, None] - slo, 0)
                alo = np.where(inn, arr_of(dhi), 0); ahi = np.where(inn, arr_of(dlo), 0)
                # num = sum a_t v_t: earlier values exact, its own value a box
                Vpos = (Vp >= 0)[None]                                           # (1, T, 64)
                nlo = np.where(Vpos, alo[:, :, None] * Vp[None], ahi[:, :, None] * Vp[None])
                nhi = np.where(Vpos, ahi[:, :, None] * Vp[None], alo[:, :, None] * Vp[None])
                nlo = np.where(past[:, :, None], nlo, 0).sum(1); nhi = np.where(past[:, :, None], nhi, 0).sum(1)
                ri = np.arange(R); aslo, ashi = alo[ri, rows_p], ahi[ri, rows_p]
                cs = np.stack([aslo[:, None] * vlo, aslo[:, None] * vhi, ashi[:, None] * vlo, ashi[:, None] * vhi])
                nlo = nlo + cs.min(0); nhi = nhi + cs.max(0)
                Slo, Shi = np.maximum(alo.sum(1), ARR[0]), ahi.sum(1)          # the race's winner always arrives with ARR[0]
                Llo, Lhi = log2fix_v(Slo), log2fix_v(Shi)
                olo, ohi = pow_box(nlo, nhi, -Lhi[:, None], -Llo[:, None])
                acc = union(acc, olo, ohi, valid)
        out_lo[:, h * 64:(h + 1) * 64], out_hi[:, h * 64:(h + 1) * 64] = acc
    return out_lo, out_hi

# ---------------------------------------------------------------- carrying a box up to the top
POINTS = ["Rin", "QKV", "ATT", "A", "Rmid", "H", "E", "Mw", "Rout"]
PILE_POINT = {"qkvpile": "QKV", "projpile": "A", "fcpile": "H", "mprojpile": "Mw"}
def pile_out(rec, l, point, rows_p, plo, phi):
    """a box on a bank's piles turned into the box on that bank's output, by the bank's own last steps"""
    d = rec["L"][l]
    if point == "qkvpile":
        wf, we, b = plate(f"qkv.{l}"); F = (we[None, :] + d["exq"][rows_p, None] + ES) * 65536 - d["LSq"][rows_p, None]
        return power(plo, F) + b, power(phi, F) + b
    if point == "fcpile":
        wf, we, b = plate(f"fc.{l}"); F = (we[None, :] + d["exf"][rows_p, None] + ES) * 65536 - d["LSf"][rows_p, None]
        return power(plo, F) + b, power(phi, F) + b
    if point == "projpile":
        wf, we, b = plate(f"proj.{l}"); s = -(we[None, :] + d["exa"][rows_p, None] + ES)
        return rshift(plo, s) + b, rshift(phi, s) + b
    if point == "mprojpile":
        wf, we, b = plate(f"mproj.{l}"); s = -(we[None, :] + d["exe"][rows_p, None] + ES)
        return rshift(plo, s) + b, rshift(phi, s) + b
    raise KeyError(point)

LIM = np.int64(1) << 32                                     # a box wider than this anywhere is called blown: it
                                                            # proves nothing at the top (every true value is below 2^31)
def push(rec, rows_p, l, point, lo, hi, check=False):
    """the box at (layer l, point) for positions rows_p, carried to the stream after layer 11.
    Returns (Rlo, Rhi, blown). A blown row's box went past 2^32 somewhere: it is not carried further and proves nothing.
    check=True asserts the true value lies inside every box on the way."""
    rows_p = np.asarray(rows_p, np.int64); blown = np.zeros(len(rows_p), bool)
    if point in PILE_POINT:
        lo, hi = pile_out(rec, l, point, rows_p, lo, hi); point = PILE_POINT[point]
    def chk(lo_, hi_, l_, key):
        if check:
            t = rec["L"][l_][key][rows_p]; ok = ((lo_ <= t) & (t <= hi_)).all(1) | blown
            assert ok.all(), f"truth outside the box at {key} {l_}"
    global WIDE
    WIDE = np.zeros(len(rows_p), bool)
    def cap(box):
        nonlocal blown
        b = (np.abs(box[0]) > LIM).any(1) | (np.abs(box[1]) > LIM).any(1) | WIDE; blown = blown | b
        return (np.where(blown[:, None], 0, box[0]), np.where(blown[:, None], 0, box[1]))
    i = POINTS.index(point)
    if point == "Rout": l, i = l + 1, 0                       # the stream after layer l is the stream into layer l+1
    if l == 12: return lo, hi, blown
    d = rec["L"][l]; cur = cap((lo, hi))
    if i <= POINTS.index("A"): S = (d["Rin"][rows_p], d["Rin"][rows_p])
    else: S = (d["Rmid"][rows_p], d["Rmid"][rows_p])
    while True:
        name = POINTS[i]
        if name == "Rin":
            chk(*cur, l, "Rin"); S = cur; cur = cap(i_gauged(cur[0], cur[1], f"qkv.{l}")); i = 1; continue
        if name == "QKV":
            chk(*cur, l, "QKV"); cur = cap(i_attention(rec, l, rows_p, cur[0], cur[1])); i = 2; continue
        if name == "ATT":
            chk(*cur, l, "ATT"); cur = cap(i_plain(cur[0], cur[1], f"proj.{l}")); i = 3; continue
        if name == "A":
            chk(*cur, l, "A"); cur = cap((S[0] + cur[0], S[1] + cur[1])); i = 4; continue
        if name == "Rmid":
            chk(*cur, l, "Rmid"); S = cur; cur = cap(i_gauged(cur[0], cur[1], f"fc.{l}")); i = 5; continue
        if name == "H":
            chk(*cur, l, "H"); cur = cap(i_edge(cur[0], cur[1])); i = 6; continue
        if name == "E":
            chk(*cur, l, "E"); cur = cap(i_plain(cur[0], cur[1], f"mproj.{l}")); i = 7; continue
        if name == "Mw":
            chk(*cur, l, "Mw"); cur = cap((S[0] + cur[0], S[1] + cur[1])); i = 8; continue
        if name == "Rout":
            chk(*cur, l, "Rout"); l += 1; i = 0
            if l == 12: return cur[0], cur[1], blown
            continue

# ---------------------------------------------------------------- the top read on boxes
def top_branches(Rlo, Rhi):
    """the final stream's box -> the top light's boxes, one branch per water level the box allows (a row whose box
    allows more than 3 levels gets no valid branch, and so is never proved)"""
    global WIDE
    WIDE = np.zeros(len(Rlo), bool); s0, s1 = levels(Rlo, Rhi); wide = WIDE.copy(); WIDE = None; br = []
    for o in range(int((s1 - s0).max()) + 1):
        sh = s0 + o; xlo, xhi, valid = at_level(Rlo, Rhi, sh, s1); ex = sh - ES; valid = valid & ~wide
        LSlo, LShi = ls_box(xlo, xhi, ex)
        br.append(dict(valid=valid, xlo=xlo, xhi=xhi, ex=ex, LSlo=LSlo, LShi=LShi))
    return br

def _g_parts(F):
    F = np.asarray(F, np.int64); I = F >> 16; e = EXP2[F & 0xFFFF]; s = 16 - I
    assert (s >= 1).all() and (s <= 62).all()
    return e, s

def certify(br, want_top5=None):
    """Is the top 5 (set and order) the same for every light and gauge the branches allow? Proved by bounds:
    each word's score interval (the pile's interval through the gauge's range, corners), then for every pair the
    certificate still needs, the difference of the two piles bounded directly (one shared gauge for both words):
        score_a - score_b >= floor(D_lo x g) + (bias_a - bias_b),  g = EXP2[f] / 2^s at the gauge end that makes it least.
    Returns per row: cert5 (bool), certwin (bool), cand (the five words, in order) and the count of pairs bounded."""
    R = len(br[0]["xlo"])
    BIG = np.iinfo(np.int64).max; Slo = np.full((R, NW), BIG); Shi = np.full((R, NW), -BIG)
    for b in br:
        plo, phi = head_box(b["xlo"], b["xhi"])
        base = (HE0 + b["ex"] + ES) * 65536
        lo_, hi_ = pow_box(plo, phi, (base - b["LShi"])[:, None], (base - b["LSlo"])[:, None])
        v = b["valid"][:, None]
        Slo = np.where(v, np.minimum(Slo, lo_ + HEAD_B), Slo); Shi = np.where(v, np.maximum(Shi, hi_ + HEAD_B), Shi)
        del plo, phi, lo_, hi_
    mid = (Slo >> 1) + (Shi >> 1)                                            # one point of every box (any point serves)
    idx = np.argpartition(-mid, 5, axis=1)[:, :5]
    o = np.lexsort((idx, -np.take_along_axis(mid, idx, 1)), axis=1); cand = np.take_along_axis(idx, o, 1)
    cert5 = np.ones(R, bool); certwin = np.ones(R, bool); npairs = np.zeros(R, np.int64)
    todo = []                                                                # per row: arrays (a, b, kind) still needing a direct bound
    for r in range(R):
        c = cand[r]; lo_c = Slo[r, c]; hi_c = Shi[r, c]; L = []
        for k in range(4):
            if not lo_c[k] > hi_c[k + 1]: L.append((c[k], c[k + 1], 0))
        if not lo_c[0] > hi_c[1]: L.append((c[0], c[1], 1))
        for k in (2, 3, 4):
            if not lo_c[0] > hi_c[k]: L.append((c[0], c[k], 1))
        rest = np.ones(NW, bool); rest[c] = False
        w5 = np.flatnonzero(rest & (Shi[r] >= lo_c[4])); w5 = w5[np.argsort(-Shi[r, w5], kind="stable")]   # hardest first
        w1 = np.flatnonzero(rest & (Shi[r] >= lo_c[0])); w1 = w1[np.argsort(-Shi[r, w1], kind="stable")]
        L = np.array(L, np.int64).reshape(-1, 3)
        A = np.concatenate([L[:, 0], np.full(len(w5), c[4]), np.full(len(w1), c[0])])
        Bw = np.concatenate([L[:, 1], w5, w1]); K = np.concatenate([L[:, 2], np.zeros(len(w5), np.int64), np.ones(len(w1), np.int64)])
        todo.append((A, Bw, K))
    del Slo, Shi, mid
    pos = [0] * R; CH = 32; PCH = 2048
    while True:
        parts = []
        for r in range(R):
            A, Bw, K = todo[r]
            if pos[r] >= len(A) or (not cert5[r] and not certwin[r]): continue
            sl = slice(pos[r], pos[r] + CH); pos[r] += CH
            keep = np.where(K[sl] == 0, cert5[r], certwin[r])
            if keep.any(): parts.append(np.stack([np.full(int(keep.sum()), r), A[sl][keep], Bw[sl][keep], K[sl][keep]], 1))
        batch = np.concatenate(parts) if parts else []
        if len(batch) == 0: break
        for q0 in range(0, len(batch), PCH):
            P = batch[q0:q0 + PCH]; rr, aa, bb, kind = P.T; ok = np.ones(len(P), bool)
            dw = (HEAD_W16[:, aa].astype(np.float64) - HEAD_W16[:, bb]).T       # (pairs, 768)
            adw = np.abs(dw)
            for b in br:
                v = b["valid"][rr]
                c2 = np.einsum("ij,ij->i", (b["xlo"][rr] + b["xhi"][rr]).astype(np.float64), dw)
                r2 = np.einsum("ij,ij->i", (b["xhi"][rr] - b["xlo"][rr]).astype(np.float64), adw)
                assert np.abs(c2).max() < 2 ** 52 and np.abs(r2).max() < 2 ** 52
                Dlo = (c2.astype(np.int64) - r2.astype(np.int64)) >> 1
                base = (HE0 + b["ex"][rr] + ES) * 65536
                F = np.where(Dlo >= 0, base - b["LShi"][rr], base - b["LSlo"][rr])
                e, s = _g_parts(F)
                assert np.abs(Dlo).max() < 2 ** 45
                lb = ((Dlo * e) >> s) + HEAD_B[aa] - HEAD_B[bb]                    # floor(D_lo e / 2^s) + bias difference
                ok &= ~v | (lb >= 1)
            del dw, adw
            np.add.at(npairs, rr, 1)
            for r, k in zip(rr[~ok], kind[~ok]):
                if k == 0: cert5[r] = False
                else: certwin[r] = False
    certwin |= cert5                                                         # the order proved: its first word is proved
    anyvalid = np.zeros(R, bool)
    for b in br: anyvalid |= b["valid"]
    cert5 &= anyvalid; certwin &= anyvalid                                   # no allowed level: nothing is proved
    return dict(cert5=cert5, certwin=certwin, cand=cand, npairs=npairs)

def zero_box_check(rec, rows_p):
    """every point of every layer, a box of width 0 carried to the top must give the machine's own stream"""
    bad = 0
    for l in range(12):
        for pt in POINTS[:-1]:
            v = rec["L"][l][pt][rows_p]
            lo, hi, bl = push(rec, rows_p, l, pt, v, v, check=True)
            bad += int((lo != rec["L"][11]["Rout"][rows_p]).sum() + (hi != rec["L"][11]["Rout"][rows_p]).sum() + bl.sum())
    return bad
