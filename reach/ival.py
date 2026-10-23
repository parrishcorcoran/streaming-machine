"""reach/ival.py: EXACT INTERVALS CARRIED THROUGH THE BASE-2 MACHINE (the library for (2), the answer-exact margin).

Every value is carried as three whole numbers: c, the unchanged machine's own value, and lo <= c <= hi, a range that
holds every value the machine can take there when a plate's low digits are removed (the removal's largest possible
move on each pile, carried forward). Each step of the machine is met by its exact range form:
  - a bank's pile: center x the plate exactly, plus the light's range met by the plate's ups and downs separately
    (lo = e_lo . w+ + e_hi . w-, hi = e_hi . w+ + e_lo . w-); on the cut plate also +-(largest |light|) . |removed part|;
  - the water level (the gauge): the light's own power of two can only take the values its range allows (each one
    carried, the results joined); the size 768 S2 - S1^2 is moved by the light's range e exactly as
    2 sum (768 x_i - S1) e_i + (768 sum e_i^2 - (sum e_i)^2), the first term bounded term by term, the second between
    0 and 768 sum e_i^2; the log table and the power table are monotone, so the range's ends give the gain's ends;
  - every shift and rounding is monotone, so the ends go to the ends;
  - the knife edge (the bend): the share table is monotone in where the value arrives, so the share's range is the
    shares at the two ends, and value x share is bounded by its corners (the table's steps are its slope bound);
  - the arrival: the scores' range from q and k's ranges (center . center exact, plus center . range and range .
    center bounded term by term, plus |range| . |range|); the clock's start is the largest score, so it lies between
    the largest low end and the largest high end; each distance between 0 and (top high - own low); the arrival table
    is monotone (falling); the total is at least the winner's own arrival; the values met by the arrivals' range
    term by term (arrivals are never negative);
  - the top read: the head plate met by the top light's range the same way, the gauge as above.
A row (position) is BLOWN when a range grows past what whole numbers of 64 bits can carry through the next step
(2^40 on the stream, the size's low end at or below 0, a product past 2^62): it is then treated as unbounded, which
can only make the proof fail, never pass.
Self-checks: with nothing removed, every range closes to the machine's own value at every row; every range contains
the value the machine actually takes when the digits are removed (checked at every row of the control run).
"""
import numpy as np
import fastm as F
ES, SC = F.ES, F.SC; ARR, PHI, EXP2 = F.ARR, F.PHI, F.EXP2
LOG768, EPS40 = int(F.M["log2_768"]), int(F.M["eps40"])
CAP = 1 << 40; NARR = len(ARR); FORCE = False; MAXOPT = 4      # FORCE: carry the range form even where nothing moved (self-check)

def bitlen(v):                                                 # exact bit length of non-negative int64
    v = np.asarray(v, np.int64); t = np.frexp(v.astype(np.float64))[1].astype(np.int64)
    t = np.where((t > 0) & (np.right_shift(v, np.maximum(t - 1, 0)) == 0), t - 1, t)
    t = np.where(np.right_shift(v, np.minimum(t, 63)) != 0, t + 1, t)
    return t
def log2fix(V):                                                # base2's log2fix, any positive int64
    V = np.asarray(V, np.int64); assert (V > 0).all(); t = bitlen(V) - 1
    m = np.where(t >= 16, np.right_shift(V, np.maximum(t - 16, 0)), np.left_shift(V, np.maximum(16 - t, 0))) & 0xFFFF
    return t * 65536 + F.LOG2[m]

def power(X, Fv):
    """base2's power(X, F) = rshift(X x EXP2[f], 16 - I), exactly, without passing 64 bits. Returns (value, bad):
    bad where the value cannot be carried (it would pass 2^62)."""
    X = np.asarray(X, np.int64); Fv = np.broadcast_to(np.asarray(Fv, np.int64), X.shape)
    I = np.right_shift(Fv, 16); E = EXP2[np.bitwise_and(Fv, 0xFFFF)]; u = 16 - I
    ax = np.abs(X); small = ax < (1 << 45)
    Y = np.where(small, X, 0) * E                              # |Y| < 2^62
    ok_left = (u > 0) | (bitlen(np.abs(Y)) + np.maximum(-u, 0) < 61)
    us = np.clip(u, 1, 62)
    direct = np.where(u > 0, np.right_shift(Y + np.left_shift(np.int64(1), us - 1), us), np.left_shift(Y, np.clip(-u, 0, 60)))
    big = ~small & (u >= 21) & (u <= 60)
    Xh = np.right_shift(X, 20); Xl = np.bitwise_and(X, (1 << 20) - 1)
    Yh = np.where(big, Xh, 0) * E
    ub = np.clip(u, 21, 60)
    split = np.right_shift(Yh + np.right_shift(np.where(big, Xl, 0) * E, 20) + np.left_shift(np.int64(1), ub - 21), ub - 20)
    val = np.where(small, direct, split)
    bad = (small & ~ok_left) | (~small & ~big) | (np.abs(val) > (1 << 62))
    return np.where(bad, 0, val), bad

def rshift(X, s): return F.rshift(np.asarray(X, np.int64), s)

class IV:
    """c, lo, hi: (rows, n) whole numbers; bl: (rows,) blown rows."""
    def __init__(self, c, lo, hi, bl=None):
        self.c, self.lo, self.hi = c, lo, hi
        self.bl = np.zeros(len(c), bool) if bl is None else bl
        self.fix()
    def fix(self):
        self.bl = self.bl.copy()
        for r in range(0, len(self.c), 256):                  # a piece of rows at a time
            s = slice(r, r + 256)
            self.bl[s] |= (np.maximum(np.abs(self.lo[s]), np.abs(self.hi[s])) > CAP).any(1)
            b = self.bl[s]
            if b.any(): self.lo[s][b] = self.c[s][b]; self.hi[s][b] = self.c[s][b]
            assert not (~b[:, None] & ((self.lo[s] > self.c[s]) | (self.hi[s] < self.c[s]))).any(), "a range lost its own center"
    def rows(self, sl): return IV(self.c[sl], self.lo[sl], self.hi[sl], self.bl[sl])
    def cols(self, sl): return IV(self.c[:, sl], self.lo[:, sl], self.hi[:, sl], self.bl)
    def width(self): return np.where(self.bl[:, None], -1, self.hi - self.lo)
def exact(c): c = np.asarray(c, np.int64); return IV(c, c.copy(), c.copy())
def keep_bl(c, bl):                                            # nothing moved where the range is known; blown rows stay blown
    c = np.asarray(c, np.int64); return IV(c, c.copy(), c.copy(), np.asarray(bl).copy())
def add(a, b): return IV(a.c + b.c, a.lo + b.lo, a.hi + b.hi, a.bl | b.bl)

def to16_opts(X):
    """the light's own power of two: every value its range allows. Yields (sh, x16 range, ex). A row whose power of
    two could take more than MAXOPT values is blown (its range is wider than the light's own scale)."""
    amax = np.maximum(np.abs(X.lo), np.abs(X.hi)).max(1)
    amin = np.where((X.lo <= 0) & (X.hi >= 0), 0, np.minimum(np.abs(X.lo), np.abs(X.hi))).max(1)
    sl, shi = np.maximum(bitlen(amin) - 15, 0), np.maximum(bitlen(amax) - 15, 0)
    shc = np.maximum(bitlen(np.abs(X.c).max(1)) - 15, 0)
    wide = (shi - sl) >= MAXOPT; bl = X.bl | wide
    sl = np.where(bl, shc, sl); shi = np.where(bl, shc, shi)
    lo_ = np.where(bl[:, None], X.c, X.lo); hi_ = np.where(bl[:, None], X.c, X.hi)
    for k in range(int((shi - sl).max(initial=0)) + 1):
        sh = np.minimum(sl + k, shi)
        yield sh, IV(rshift(X.c, sh[:, None]), rshift(lo_, sh[:, None]), rshift(hi_, sh[:, None]), bl.copy()), -ES + sh

def pile_range(X16, w, wp, wn, cut=None):
    """the pile's range: (center pile, lo, hi). w, wp, wn float64 (the plate, its ups, its downs).
    cut: |removed part| float64: the cut plate's own largest move, +-(largest |light|) . |removed|."""
    xc = X16.c; Pc = F.product(xc, w)
    elo = (X16.lo - xc).astype(np.float64); ehi = (X16.hi - xc).astype(np.float64)
    lo = Pc + (elo @ wp + ehi @ wn).astype(np.int64); hi = Pc + (ehi @ wp + elo @ wn).astype(np.int64)
    if cut is not None:
        D = (np.maximum(np.abs(X16.lo), np.abs(X16.hi)).astype(np.float64) @ cut).astype(np.int64)
        lo = lo - D; hi = hi + D
    return Pc, lo, hi

def size_range(X16, ex):
    """the gauge's LS range from the light's range (base2's gauge_log: V = 768 S2 - S1^2 + eps, LS = log2(V)/2 - log2 768 + ex)."""
    xc = X16.c; S1 = xc.sum(1); Vc = 768 * (xc * xc).sum(1) - S1 * S1
    elo = X16.lo - xc; ehi = X16.hi - xc; cc = 2 * (768 * xc - S1[:, None])
    t1lo = np.minimum(cc * elo, cc * ehi).sum(1); t1hi = np.maximum(cc * elo, cc * ehi).sum(1)
    t2 = 768 * np.maximum(elo * elo, ehi * ehi).sum(1)
    sh = 40 + 2 * np.asarray(ex, np.int64)
    eps = np.where(sh >= 0, np.right_shift(np.int64(EPS40), np.clip(sh, 0, 63)), np.left_shift(np.int64(EPS40), np.clip(-sh, 0, 63)))
    vlo = np.maximum(Vc + t1lo, 0) + eps; vhi = Vc + t1hi + t2 + eps
    blown = vlo <= 0
    vlo = np.where(blown, 1, vlo); vhi = np.maximum(vhi, vlo)
    L = lambda V: np.right_shift(log2fix(V), 1) - LOG768 + np.asarray(ex, np.int64) * 65536
    return L(vlo), L(vhi), blown

def power_range(Plo, Phi, Flo, Fhi):
    a, b1 = power(Plo, Flo); b, b2 = power(Plo, Fhi); c, b3 = power(Phi, Flo); d, b4 = power(Phi, Fhi)
    return np.minimum(a, b), np.maximum(c, d), (b1 | b2 | b3 | b4).any(1)

class Plate:
    def __init__(self, name, w16, removed=None):
        self.name = name; w = np.asarray(w16).astype(np.float64); self.w = w
        self.wp = np.maximum(w, 0); self.wn = np.minimum(w, 0); self.we = F.E_[name]; self.b = F.B_[name]
        self.cut = None if removed is None else np.abs(np.asarray(removed)).astype(np.float64)

def gauged(R, pl, cols=None):
    """a bank after a gauge (qkv, fc, the top read): the machine's own value at the center, the range around it."""
    x16c, exc, LSc = F.gauge_log(R.c)
    w = pl.w if cols is None else pl.w[:, cols]; we = pl.we if cols is None else pl.we[cols]; b = pl.b if cols is None else pl.b[cols]
    keptc = F.power(F.product(x16c, w), (we[None, :] + exc[:, None] + ES) * 65536 - LSc[:, None]) + b
    if not FORCE and not (R.lo != R.hi).any() and pl.cut is None: return keep_bl(keptc, R.bl)
    wp = pl.wp if cols is None else pl.wp[:, cols]; wn = pl.wn if cols is None else pl.wn[:, cols]
    cut = pl.cut if (pl.cut is None or cols is None) else pl.cut[:, cols]
    lo = hi = None; bl = R.bl.copy()
    for sh, X16, ex in to16_opts(R):
        bl |= X16.bl
        LSlo, LShi, b1 = size_range(X16, ex)
        _, Plo, Phi = pile_range(X16, w, wp, wn, cut)
        base = (we[None, :] + ex[:, None] + ES) * 65536
        olo, ohi, b2 = power_range(Plo, Phi, base - LShi[:, None], base - LSlo[:, None])
        bl |= b1 | b2
        lo = olo if lo is None else np.minimum(lo, olo); hi = ohi if hi is None else np.maximum(hi, ohi)
    return IV(keptc, lo + b, hi + b, bl)

def plain(X, pl):
    """a bank with no gauge before it (proj, mproj): rshift(pile, -(we + ex + ES)) + bias."""
    x16c, exc = F.to16(X.c, -ES)
    keptc = F.rshift(F.product(x16c, pl.w), -(pl.we[None, :] + exc[:, None] + ES)) + pl.b
    if not FORCE and not (X.lo != X.hi).any() and pl.cut is None: return keep_bl(keptc, X.bl)
    lo = hi = None; bl = X.bl.copy()
    for sh, X16, ex in to16_opts(X):
        bl |= X16.bl
        _, Plo, Phi = pile_range(X16, pl.w, pl.wp, pl.wn, pl.cut)
        s = -(pl.we[None, :] + ex[:, None] + ES)
        olo, ohi = F.rshift(Plo, s), F.rshift(Phi, s)
        lo = olo if lo is None else np.minimum(lo, olo); hi = ohi if hi is None else np.maximum(hi, ohi)
    return IV(keptc, lo + pl.b, hi + pl.b, bl)

def share(H):
    idx = F.rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
def bend(Hv):
    """the knife edge: out = rshift(H x share(H), 16); share monotone in H and never negative."""
    c = F.edge(Hv.c)
    if not FORCE and not (Hv.lo != Hv.hi).any(): return keep_bl(c, Hv.bl)
    bl = Hv.bl | ((np.abs(Hv.lo) > (1 << 45)) | (np.abs(Hv.hi) > (1 << 45))).any(1)
    Hl = np.where(bl[:, None], Hv.c, Hv.lo); Hh = np.where(bl[:, None], Hv.c, Hv.hi)
    sl, sh = share(Hl), share(Hh)
    plo = np.where(Hl >= 0, Hl * sl, Hl * sh); phi = np.where(Hh >= 0, Hh * sh, Hh * sl)
    return IV(c, F.rshift(plo, 16), F.rshift(phi, 16), bl)

def arrv(d): return np.where(d < NARR, ARR[np.clip(d, 0, NARR - 1)], 0)
def attention_text(Q):
    """the arrival for one text: Q is the text's qkv range (positions x 2304)."""
    P = Q.c.shape[0]; c = F.attention(Q.c)
    if not FORCE and not (Q.lo != Q.hi).any(): return keep_bl(c, np.logical_or.accumulate(Q.bl))
    low = np.tril(np.ones((P, P), bool)); lo = np.zeros((P, 768), np.int64); hi = np.zeros((P, 768), np.int64)
    bl = np.logical_or.accumulate(Q.bl)                        # a blown row blows every later row of its text
    for h in range(12):
        q, k, v = Q.cols(slice(h*64, (h+1)*64)), Q.cols(slice(768+h*64, 768+(h+1)*64)), Q.cols(slice(1536+h*64, 1536+(h+1)*64))
        if not FORCE and not ((q.lo != q.hi).any() or (k.lo != k.hi).any() or (v.lo != v.hi).any()):
            lo[:, h*64:(h+1)*64] = c[:, h*64:(h+1)*64]; hi[:, h*64:(h+1)*64] = c[:, h*64:(h+1)*64]; continue
        slo = shi = None
        for _, q16, eq in to16_opts(q):
            bl = bl | np.logical_or.accumulate(q16.bl)
            for _, k16, ek in to16_opts(k):
                bl = bl | np.logical_or.accumulate(k16.bl)
                qc, kc = q16.c.astype(np.float64), k16.c.astype(np.float64)
                dc = F.product(q16.c, kc.T)
                eql, eqh = (q16.lo - q16.c).astype(np.float64), (q16.hi - q16.c).astype(np.float64)
                ekl, ekh = (k16.lo - k16.c).astype(np.float64), (k16.hi - k16.c).astype(np.float64)
                qp, qn, kp, kn = np.maximum(qc, 0), np.minimum(qc, 0), np.maximum(kc, 0), np.minimum(kc, 0)
                t1l = qp @ ekl.T + qn @ ekh.T; t1h = qp @ ekh.T + qn @ ekl.T
                t2l = eql @ kp.T + eqh @ kn.T; t2h = eqh @ kp.T + eql @ kn.T
                t3 = np.maximum(np.abs(eql), np.abs(eqh)) @ np.maximum(np.abs(ekl), np.abs(ekh)).T
                dl = dc + (t1l + t2l - t3).astype(np.int64); dh = dc + (t1h + t2h + t3).astype(np.int64)
                s = -(eq[:, None] + ek[None, :] - 3 + SC)
                l_, h_ = F.rshift(dl, s), F.rshift(dh, s)
                slo = l_ if slo is None else np.minimum(slo, l_); shi = h_ if shi is None else np.maximum(shi, h_)
        NEG = np.iinfo(np.int64).min // 4
        mlo = np.where(low, slo, NEG).max(1); mhi = np.where(low, shi, NEG).max(1)
        dlo = np.where(low, np.maximum(mlo[:, None] - shi, 0), NARR); dhi = np.where(low, mhi[:, None] - slo, NARR)
        alo, ahi = arrv(dhi), arrv(dlo)                       # the arrival table falls with distance
        Alo = np.maximum(alo.sum(1), ARR[0]); Ahi = ahi.sum(1)
        vbig = (np.maximum(np.abs(v.lo), np.abs(v.hi)) > (1 << 33)).any(1)
        bl = bl | np.logical_or.accumulate(vbig)
        vl = np.where(bl[:, None], v.c, v.lo); vh = np.where(bl[:, None], v.c, v.hi)
        nlo = alo @ np.maximum(vl, 0) + ahi @ np.minimum(vl, 0); nhi = ahi @ np.maximum(vh, 0) + alo @ np.minimum(vh, 0)
        Flo = -log2fix(Ahi)[:, None]; Fhi = -log2fix(Alo)[:, None]
        ol, oh, b3 = power_range(nlo, nhi, Flo, Fhi)
        bl = bl | np.logical_or.accumulate(b3)
        lo[:, h*64:(h+1)*64] = ol; hi[:, h*64:(h+1)*64] = oh
    return IV(c, lo, hi, bl)
def attention(Q, off):
    return concat([attention_text(Q.rows(slice(off[t], off[t + 1]))) for t in range(len(off) - 1)])

def concat(parts): return IV(np.concatenate([p.c for p in parts]), np.concatenate([p.lo for p in parts]),
                             np.concatenate([p.hi for p in parts]), np.concatenate([p.bl for p in parts]))
def by_rows(fn, X, *a, rows=1024):                             # a row-by-row step, a piece of rows at a time
    return concat([fn(X.rows(slice(r, r + rows)), *a) for r in range(0, len(X.c), rows)])
def fill_rows(fn, X, *a, rows=512):                            # the same, into arrays made once (no joining copy)
    n = len(X.c); out = None
    for r in range(0, n, rows):
        part = fn(X.rows(slice(r, r + rows)), *a)
        if out is None:
            m = part.c.shape[1]; out = IV.__new__(IV)
            out.c = np.empty((n, m), np.int64); out.lo = np.empty((n, m), np.int64); out.hi = np.empty((n, m), np.int64); out.bl = np.zeros(n, bool)
        out.c[r:r + rows] = part.c; out.lo[r:r + rows] = part.lo; out.hi[r:r + rows] = part.hi; out.bl[r:r + rows] = part.bl
    return out
