"""groups2/gbank.py -- THE GROUP BANK of the whole machines base2_groups.py (GPT-2) and qwen_groups.py (Qwen3-0.6B).

One plate, one light. The plate is laid once (plate only, before any light is seen):
  - THE PER-PLACE RULE of the confirm step, unchanged (groups2/uh_perplace.py's `rule`): for each plate and place b,
    L_b = the plate's lit signed digits at b, U_b = G x M (G = ceil(N / g) groups, M neurons); place b is read in
    signed digits when L_b < U_b, otherwise up/hole. A caller may instead give the up/hole places itself (up = 0
    everywhere: the plate read in signed digits alone).
  - the digits (uh_rebuild.expand): bottom place to top, a binary digit at an up/hole place, a signed digit elsewhere,
    offset C = 2^(top up/hole place); 2 x.w = sum over up/hole places of 2^b (ready pile) + 2 (signed part) + kcorr X,
    kcorr = K - 2C;
  - the random groups (uh_rebuild.make_groups: numpy's generator keyed [seed, slab, layer, g], a random line order cut
    into runs of g; seed 1 = the confirmed runs' groups), the patterns, and which half-table entries the plate reads.
Per light (groups2/bgm.c): every group's ready piles once a position; each neuron at each up/hole place adds one ready
pile, each lit signed digit meets its own light value; the pile is the doubled pile halved (a shift).

THE WORK COUNTED (photon meetings and photons added, the unit of every earlier result), per position:
  light      the light written in signed digits: its 1s (as base2_fringes2 counts it at every bank)
  table      the half table by the flip: each Gray step one light value's signed digits, by the walk's relation (the
             group total: lines 1..g-1 once; line t turns 2^(g-2-t) times), only where the plate has an up/hole place
  pwrite     every half-table entry the plate reads (itself or its flip) written in signed digits once: its 1s. NOT IN
             THE CONFIRMED COUNTS (uh_rebuild, uh_perplace, topg): reported as its own line
  reads      every up/hole read: the ready pile's signed digits
  sdreads    every lit signed digit: its light value's signed digits
  xsum       X, the light summed: N - 1 adds of light values (only where the plate has an up/hole place)
  xoff       kcorr X taken from each neuron's doubled pile: its signed digits (and kcorr's extra digits x X's)
The halving is a shift. Whole numbers only.
"""
import os, sys, ctypes, subprocess, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; G2 = f"{HERE}/groups2"
if G2 not in sys.path: sys.path.insert(0, G2)
_argv = sys.argv; sys.argv = [sys.argv[0]]
import uh_rebuild as R
import uh_perplace as PP
sys.argv = _argv
NPL = 18

def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def ones1(v): return bin(abs(int(v))).count("1")
def sd1(v): a = abs(int(v)); return bin(a ^ (3 * a)).count("1")

_LIB = None
def lib():
    global _LIB
    if _LIB is None:
        src, so = f"{G2}/bgm.c", f"{G2}/bgm.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _LIB = ctypes.CDLL(so); V, I = ctypes.c_void_p, ctypes.c_int64
        _LIB.bank_groups.argtypes = [I] * 7 + [V] * 13; _LIB.bank_groups.restype = None
        _LIB.plain_product.argtypes = [I, I, I, V, V, V]; _LIB.plain_product.restype = None
    return _LIB

def plain_product(x, w16):
    """CHECK PATH: the direct product, lines met one by one, int64"""
    x = np.ascontiguousarray(x, np.int32); w16 = np.ascontiguousarray(w16, np.int16); P, N = x.shape; M = w16.shape[1]
    out = np.empty((P, M), np.int64); lib().plain_product(P, N, M, x.ctypes.data, w16.ctypes.data, out.ctypes.data); return out

def half_coeffs(glines, goff, N):
    """the half table's adds per line, by the Gray walk's relation (as uh_rebuild and topg count it)"""
    ch = np.zeros(N, np.int64)
    for k in range(len(goff) - 1):
        L = glines[goff[k]:goff[k + 1]]; gk = len(L)
        for t in range(gk):
            ch[L[t]] = (1 if t >= 1 else 0) + ((1 << (gk - 2 - t)) if t <= gk - 2 else 0)
    return ch

def bound_per_line(upmask):
    """B(b): the most the places below b can add to the doubled pile, per unit of light (|x| summed over the lines):
    up/hole places 2^q each (|ready pile| <= the group's |x| summed); each signed run [a, c) 2 x 2^a x t(len), t_n =
    (2^(n+1) - 1) // 3 the largest a run of non-adjacent digits can hold (topg.py's bound for its per-place form)"""
    tn = lambda n: ((1 << (n + 1)) - 1) // 3 if n > 0 else 0
    runs, q = [], 0
    while q < NPL:
        if q < 16 and (upmask >> q) & 1: q += 1; continue
        a = q
        while q < NPL and not (q < 16 and (upmask >> q) & 1): q += 1
        runs.append((a, q))
    out = []
    for b in range(NPL + 1):
        v = sum(1 << q for q in range(min(b, 16)) if (upmask >> q) & 1)
        for a, c in runs:
            if a < b: v += 2 * (1 << a) * tn(min(c, b) - a)
        out.append(v)
    return out

class GPlate:
    """a plate laid for the group machine. w (N, M) whole numbers; segs [(name, a, b)] the plates of a slab (each its own
    rule); key the grouping key; reorder (ORDER-FREE) the same groups in a random order, each group's lines in a random
    order; up (optional) the up/hole places of every neuron given by the caller."""
    def __init__(self, w, segs, g, key, reorder=None, up=None):
        w = np.asarray(w, np.int64); N, M = w.shape; self.N, self.M, self.g, self.segs = N, M, g, segs
        if up is None: up, self.chosen = PP.rule(w, segs, g)
        else: up = np.asarray(up, np.int64); self.chosen = None
        self.up = up; self.up32 = np.ascontiguousarray(up, np.uint32); prep = R.prepare(w, up, segs)
        self.glines, self.goff = R.make_groups(N, g, key, reorder)
        sizes = np.diff(self.goff); self.G = len(sizes); self.gmax = int(sizes.max())
        self.anyup = bool(up.any())
        self.pat = np.ascontiguousarray(R.patterns(prep["E"], self.glines, self.goff))
        TH = 1 << (self.gmax - 1); self.readh = np.zeros((self.G, TH), np.uint8)
        UPM16 = ((up[:, None] >> np.arange(16)[None, :]) & 1).astype(bool)                   # (M, 16)
        if self.anyup:
            for k in range(self.G):
                gk = int(sizes[k]); top = 1 << (gk - 1); full = (1 << gk) - 1
                p = self.pat[k][UPM16].astype(np.int64)
                key_ = np.where(p & top, p, p ^ full) - top
                self.readh[k, np.unique(key_)] = 1
        self.ch = half_coeffs(self.glines, self.goff, N)
        self.UPM = np.zeros((M, NPL), bool); self.UPM[:, :16] = UPM16
        self.kcorr = np.ascontiguousarray(prep["kcorr"], np.int64)
        self.toff, self.tline, self.tplace, self.tsign = prep["toff"], prep["tline"], prep["tplace"], prep["tsign"]
        self.wrow = ones(w).sum(1); self.nrow = naf(w).sum(1)
        self.seg_wrow = [ones(w[:, a:b]).sum(1) for _, a, b in segs]; self.seg_nrow = [naf(w[:, a:b]).sum(1) for _, a, b in segs]
        lit = np.zeros(NPL, bool); lit[:16] |= UPM16.any(0)
        if len(self.tplace): lit[np.unique(self.tplace)] = True
        self.fmt = lit; self.read_order = [b for b in range(NPL - 1, -1, -1) if lit[b]]
        self.upmask_same = bool((up == up[0]).all()); self.upmask = int(up[0])
        self.bnd = bound_per_line(self.upmask) if self.upmask_same else None
        self.w16 = w.astype(np.int16); assert np.array_equal(self.w16.astype(np.int64), w)
        self.signed_places = {name: c["signed_places"] for name, c in (self.chosen or {}).items()}
        del prep

    def kernel(self, x):
        x = np.ascontiguousarray(x, np.int32); P, N = x.shape; assert N == self.N
        A = np.empty((P, self.M, NPL), np.int64); H = np.empty((P, self.M, NPL), np.int32); wr = np.empty(P, np.int64)
        lib().bank_groups(P, N, self.M, self.G, self.gmax, NPL, int(self.anyup), x.ctypes.data, self.glines.ctypes.data,
                          self.goff.ctypes.data, self.up32.ctypes.data, self.pat.ctypes.data,
                          self.readh.ctypes.data, self.toff.ctypes.data, self.tline.ctypes.data, self.tplace.ctypes.data,
                          self.tsign.ctypes.data, A.ctypes.data, H.ctypes.data, wr.ctypes.data)
        return A, H, wr

    def counts(self, x, H, wr, X):
        """the work of one light at this plate, per position (and the reads per segment)"""
        x = np.asarray(x, np.int64); sx = naf(x); P = len(x)
        c = dict(light=ones(x).sum(1), pwrite=wr.copy())
        c["table"] = sx @ self.ch if self.anyup else np.zeros(P, np.int64)
        c["xsum"] = sx[:, 1:].sum(1) if self.anyup else np.zeros(P, np.int64)
        xo = np.zeros(P, np.int64)
        for kv in np.unique(self.kcorr):
            if kv == 0: continue
            nj = int((self.kcorr == kv).sum()); xo += nj * naf(int(kv) * X)
            dk = sd1(kv)
            if dk > 1: xo += nj * (dk - 1) * naf(X)
        c["xoff"] = xo
        Hu = np.where(self.UPM[None], H, 0); Hs = H - Hu
        for si, (name, a, b) in enumerate(self.segs):
            c[f"reads|{name}"] = Hu[:, a:b].sum((1, 2), dtype=np.int64); c[f"sdreads|{name}"] = Hs[:, a:b].sum((1, 2), dtype=np.int64)
        return c

    def read(self, x, chunk=96, place_order=None, keep_parts=False):
        """the piles of light x (P, N) and the counts; place_order (ORDER-FREE): the place parts summed in that order.
        keep_parts: also return A and H (P, M, 18) for a place-by-place reader."""
        x = np.asarray(x, np.int64); P = len(x); pile = np.empty((P, self.M), np.int64); C = {}; parts = []
        assert np.abs(x).max(initial=0) < 1 << 24                               # int32 light; every sum below 2^62
        for c0 in range(0, P, chunk):
            xs = x[c0:c0 + chunk]; A, H, wr = self.kernel(xs); X = xs.sum(1)
            D = np.zeros((len(xs), self.M), np.int64)
            for b in (range(NPL) if place_order is None else place_order): D += A[:, :, b] << b
            D += self.kcorr[None, :] * X[:, None]
            assert not (D & 1).any(), "the halving met an odd doubled pile"
            pile[c0:c0 + chunk] = D >> 1
            for k, v in self.counts(xs, H, wr, X).items(): C.setdefault(k, []).append(v)
            if keep_parts: parts.append((A, H))
            else: del A, H
        C = {k: np.concatenate(v) for k, v in C.items()}
        return (pile, C, parts) if keep_parts else (pile, C)

def total(c, prefix=None):
    """the group work of a counts dict (every item but the light's own writing is listed; all are photons)"""
    return sum(v for k, v in c.items())
