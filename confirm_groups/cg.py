"""confirm_groups/cg.py -- THE CHECKER'S OWN GROUP PLATE AND GROUP TOP READ (independent rebuild of base2_groups.py,
GPT-2). Nothing here is imported or copied from base2_groups.py, qwen_groups.py, groups/ or groups2/; the rules were
taken from their docstrings only (group size, how places are chosen, the grouping draw's seed and keys, the top read's
set-aside rule and constants, the counting conventions). Whole numbers only.

THE RULES AS THIS CHECKER READS THEM
  grouping   numpy's generator keyed [seed, slab, layer, g] (slab: qkv 0, proj 1, fc 2, mproj 3), a random order of the
             N lines, cut into runs of g (g = 12 at the banks). Top table: key [seed, 99, 768, 16], runs of 16.
  per place  L_b = how many weights of the plate have a lit signed digit at place b (the plate's own signed digits,
             digit by digit: where the running value is odd the digit is 2 - (value mod 4); take it away; halve);
             U_b = G x M (G = ceil(N / g) groups, M neurons). Place b (0..15) is read in signed digits when L_b < U_b,
             up/hole otherwise. Fixed by the plate alone.
  digits     each weight written once from the bottom place up, starting from w + C, C = 2^(top up/hole place): a binary
             digit at an up/hole place, a signed digit (the rule above) elsewhere, places 16 and 17 signed. Then
             2 x.w = sum over up/hole places of 2^b V(pattern) + 2 (signed part) + (K - 2C) X, K = sum of 2^b over the
             up/hole places, X the light summed, V(pattern) = light on the up lines - light on the holes.
  per position, a bank's work (photons):
    light   the light written in signed digits: its 1s
    table   the half table by the flip: in each group the total costs lines 1..g-1 once, and line t turns 2^(g-2-t)
            times (each turn: that light value's signed digits); only where the plate has an up/hole place
    pwrite  every pattern pair {p, complement} the plate reads at all, written once: the pile's 1s
    reads   every up/hole read: the ready pile's signed digits
    sdreads every lit signed digit: its light value's signed digits
    xsum    X: N - 1 adds of light values (the first line is the start): the other lines' signed digits
    xoff    (K - 2C) X taken from each neuron's doubled pile: (K - 2C)'s signed digits x X's
"""
import ctypes, os, numpy as np
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = ctypes.CDLL(f"{HERE}/gk.dylib")
I64, I32, U16, U32, I8 = (ctypes.POINTER(t) for t in (ctypes.c_int64, ctypes.c_int32, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_int8))
def ptr(a, t): return a.ctypes.data_as(t)
NTH = 8
SLAB = {"qkv": 0, "proj": 1, "fc": 2, "mproj": 3}

# ---------- photons ----------
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)
def sd(A):                                                    # signed digits of each whole number
    a = np.abs(np.asarray(A, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def sd_slow(v):                                               # digit by digit (the check of sd)
    v = int(v); n = 0
    while v:
        if v & 1: d = 2 - (v % 4); v -= d; n += 1
        v //= 2
    return n
_t = np.concatenate([np.arange(-70000, 70000), np.random.default_rng(11).integers(-(1 << 40), 1 << 40, 20000)])
assert np.array_equal(sd(_t), np.array([sd_slow(v) for v in _t])), "signed-digit count"
LIB.sd_count.restype = ctypes.c_int64; LIB.sd_count.argtypes = [ctypes.c_int64]
assert all(LIB.sd_count(int(v)) == sd_slow(v) for v in _t[::97]), "the C kernel's signed-digit count"
del _t

def digits_lit(w):                                            # the plate's own signed digits: lit count at places 0..17
    r = np.asarray(w, np.int64).copy(); lit = np.zeros(18, np.int64)
    for b in range(18):
        odd = (r & 1).astype(bool); d = np.where(odd, 2 - (r & 3), 0)
        lit[b] = int(odd.sum()); r = (r - d) >> 1
    assert not r.any()
    return lit

def expand(w, up):
    """w + C written bottom up: binary digits at the up/hole places, signed digits elsewhere (places 0..17).
    Returns {b: uint8 (N, M)} binary digits, {b: int8 (N, M)} signed digits (only places with a lit digit), C, K."""
    r = np.asarray(w, np.int64).copy(); C = (1 << max(up)) if up else 0; r += C
    E, S = {}, {}
    for b in range(18):
        if b in up:
            d = r & 1; E[b] = d.astype(np.uint8)
        else:
            odd = (r & 1).astype(bool); d = np.where(odd, 2 - (r & 3), 0)
            if odd.any(): S[b] = d.astype(np.int8)
        r = (r - d) >> 1
    assert not r.any(), "a weight not written by place 17"
    K = sum(1 << b for b in up)
    back = -C + sum(E[b].astype(np.int64) << b for b in up) + sum(S[b].astype(np.int64) << b for b in S) if (up or S) else 0
    assert np.array_equal(back, np.asarray(w, np.int64)), "the digits do not give the weight back"
    return E, S, C, K

def grouping(N, g, key, reorder=None):
    """lines (G, g): numpy's generator with the key, a random line order cut into runs of g. reorder (ORDER-FREE): this
    checker's own random order of the groups and of the lines inside each group."""
    order = np.random.default_rng(key).permutation(N); assert N % g == 0
    gl = order.reshape(N // g, g)
    if reorder is not None:
        r = np.random.default_rng(reorder)
        gl = gl[r.permutation(len(gl))]; gl = np.stack([row[r.permutation(g)] for row in gl])
    return np.ascontiguousarray(gl.astype(np.int32))

def patterns(E, up_order, gl):
    """(G, M, nU) uint16: bit t of the pattern = the binary digit of the group's line t at that place."""
    G, g = gl.shape; M = next(iter(E.values())).shape[1]
    pat = np.zeros((G, M, len(up_order)), np.uint16)
    for u, b in enumerate(up_order):
        Eb = E[b]; acc = np.zeros((G, M), np.uint16)
        for t in range(g): acc |= Eb[gl[:, t]].astype(np.uint16) << np.uint16(t)
        pat[:, :, u] = acc
    return pat

def half_table_coef(g):                                      # how many times line t of a group is added in the half table
    c = np.zeros(g, np.int64); c[1:] += 1                     # the group total: lines 1..g-1 once
    c[:g - 1] += 1 << (g - 2 - np.arange(g - 1))              # the Gray walk over lines 0..g-2: line t turns 2^(g-2-t)
    return c

class Bank:
    """a bank's plate laid for the group machine (plate only, before any light). w (N, M) whole numbers."""
    def __init__(self, w, g, key, reorder=None, place_rng=None, up=None):
        w = np.asarray(w, np.int64); N, M = w.shape; self.N, self.M, self.g = N, M, g
        G = -(-N // g); self.G = G
        self.lit = digits_lit(w)
        self.up = [b for b in range(16) if self.lit[b] >= G * M] if up is None else list(up)
        E, S, C, K = expand(w, self.up)
        self.C, self.K, self.kcorr = C, K, K - 2 * C
        self.gl = grouping(N, g, key, reorder)
        self.gorder = np.arange(G, dtype=np.int32)
        uo = list(self.up) if place_rng is None else [self.up[i] for i in place_rng.permutation(len(self.up))]
        self.up_order = uo; self.ub = np.array(uo, np.int32)
        self.pat = patterns(E, uo, self.gl) if self.up else np.zeros((G, M, 0), np.uint16)
        n = 1 << g
        self.cnt = np.stack([np.bincount(self.pat[k].ravel(), minlength=n) for k in range(G)]).astype(np.uint32) if self.up \
            else np.zeros((G, n), np.uint32)
        self.sd_places = sorted(S)
        self.S = {b: S[b] for b in self.sd_places}
        self.nsd_row = sum((S[b] != 0).sum(1).astype(np.int64) for b in self.sd_places) if S else np.zeros(N, np.int64)
        self.coef = half_table_coef(g)
        self.wrow = ones(w).sum(1)                            # plain 1s per line (the plain count)

    def read(self, x, counts=True):
        """piles (P, M) of light x (P, N) and the work per position."""
        x = np.ascontiguousarray(np.asarray(x, np.int64)); P, N = x.shape; assert N == self.N
        assert np.abs(x).max(initial=0) < 1 << 20
        pile2 = np.zeros((P, self.M), np.int64); reads = np.zeros(P, np.int64); pwrite = np.zeros(P, np.int64)
        if self.up:
            LIB.bank_uh(P, N, self.M, self.g, self.G, ptr(self.gl, I32), ptr(self.gorder, I32), len(self.up), ptr(self.ub, I32),
                        ptr(np.ascontiguousarray(self.pat), U16), ptr(self.cnt, U32), ptr(x, I64), ptr(pile2, I64),
                        ptr(reads, I64), ptr(pwrite, I64), NTH)
        xf = x.astype(np.float64)
        for b in self.sd_places:                              # each lit signed digit meets its own light value
            part = xf @ self.S[b].astype(np.float64); assert np.abs(part).max(initial=0) < 2 ** 52
            pile2 += 2 * (part.astype(np.int64) << b)
        X = x.sum(1)
        if self.up: pile2 += self.kcorr * X[:, None]
        assert not (pile2 & 1).any(), "a doubled pile is odd"
        pile = pile2 >> 1
        if not counts: return pile, None
        nx = sd(x)
        c = {"light": ones(x).sum(1), "reads": reads, "pwrite": pwrite, "sdreads": nx @ self.nsd_row}
        if self.up:
            c["table"] = (nx[:, self.gl] * self.coef[None, None, :]).sum((1, 2))
            c["xsum"] = nx[:, 1:].sum(1)
            c["xoff"] = self.M * sd(np.array([self.kcorr]))[0] * sd(X)
        else:
            z = np.zeros(P, np.int64); c.update(table=z, xsum=z, xoff=z)
        return pile, c

ITEMS = ["reads", "sdreads", "pwrite", "table", "xoff", "light", "xsum"]

# ---------- the top table ----------
def t_run(n): return ((1 << (n + 1)) - 1) // 3               # the largest value n non-adjacent signed digits can hold

class Top:
    """the top table in the per-place form at g = 16, read from the top place down with the band read's set-aside.
    w (768, NW) the word table (whole numbers, one power of two), bias (NW,), head_e (one power of two)."""
    def __init__(self, w, bias, head_e, g=16, seed=1, reorder=None, word_perm=None, log=print):
        w = np.asarray(w); N, NW = w.shape; self.N, self.NW, self.g = N, NW, g
        self.wperm = np.arange(NW) if word_perm is None else np.asarray(word_perm)
        if word_perm is not None: w = w[:, word_perm]; bias = np.asarray(bias)[word_perm]
        G = N // g; self.G = G; CH = 4096
        self.lit = sum(digits_lit(w[:, c:c + CH]) for c in range(0, NW, CH))
        self.up = [b for b in range(16) if self.lit[b] >= G * NW]
        self.gl = grouping(N, g, [seed, 99, N, g], reorder)
        self.up_order = sorted(self.up, reverse=True)
        self.pat = np.zeros((G, NW, len(self.up_order)), np.uint16); pieces = {}
        for c in range(0, NW, CH):                            # the table written place by place, a chunk of words at a time
            E, S, C, K = expand(w[:, c:c + CH], self.up)
            self.pat[:, c:c + CH, :] = patterns(E, self.up_order, self.gl); del E
            for b, Sb in S.items():
                ST = np.ascontiguousarray(Sb.T); wd, ln = np.nonzero(ST)          # by word, then line
                pieces.setdefault(b, []).append((wd + c, ln.astype(np.int32), ST[wd, ln].astype(np.int8)))
            del S
        self.kcorr = K - 2 * C
        n = 1 << g
        self.cnt = np.stack([np.bincount(self.pat[k].ravel(), minlength=n) for k in range(G)]).astype(np.uint32)
        self.sd_places = sorted(pieces, reverse=True)
        ws, ls, ss = [], [], []; off = 0
        for b in self.sd_places:                              # CSR by word of each signed place's lit digits
            wd = np.concatenate([p[0] for p in pieces[b]]); ln = np.concatenate([p[1] for p in pieces[b]])
            sg = np.concatenate([p[2] for p in pieces[b]]); assert (np.diff(wd) >= 0).all()
            ws.append(np.searchsorted(wd, np.arange(NW + 1)) + off); ls.append(ln); ss.append(sg); off += len(wd)
        self.wstart = np.ascontiguousarray(np.concatenate(ws).astype(np.int64)) if ws else np.zeros(0, np.int64)
        self.sl = np.ascontiguousarray(np.concatenate(ls)) if ls else np.zeros(0, np.int32)
        self.ss = np.ascontiguousarray(np.concatenate(ss)) if ss else np.zeros(0, np.int8)
        del pieces
        self.places = sorted(set(self.up) | set(self.sd_places), reverse=True)   # read from the top place down
        self.coef = half_table_coef(g)
        self.bias = np.asarray(bias, np.int64); self.BIAS_ONES = ones(self.bias); self.BMAX = int(np.abs(self.bias).max())
        assert (np.asarray(head_e) == np.asarray(head_e)[0]).all(); self.he = int(np.asarray(head_e)[0])
        self.wrow = sum(ones(w[:, c:c + CH]).sum(1) for c in range(0, NW, CH))
        # B(b): the most the places below b can add, per unit of light, at the doubled scale
        self.B = []
        for i, b in enumerate(self.places):
            B = sum(1 << q for q in self.up if q < b)
            q = 0
            while q < b:                                      # each signed run below b
                if q in self.up: q += 1; continue
                a = q
                while q < b and q not in self.up: q += 1
                B += 2 * (1 << a) * t_run(q - a)
            self.B.append(B)
        log(f"  top table laid: up/hole places {sorted(self.up)}, signed places {sorted(self.sd_places)}, kcorr {self.kcorr}, "
            f"read order {self.places}")

    def ready(self, X):
        """per position the reads of every word at every place (C kernels): returns contrib, cost per place, pwrite."""
        X = np.ascontiguousarray(np.asarray(X, np.int64)); P = len(X); NW, nU, nS = self.NW, len(self.up_order), len(self.sd_places)
        cu = np.empty((P, nU, NW), np.int64); ku = np.empty((P, nU, NW), np.int32); pw = np.zeros(P, np.int64)
        LIB.top_uh(P, self.N, NW, self.g, self.G, ptr(self.gl, I32), nU, ptr(self.pat, U16), ptr(self.cnt, U32), ptr(X, I64),
                   ptr(cu, I64), ptr(ku, I32), ptr(pw, I64), NTH)
        cs = np.empty((P, nS, NW), np.int64); ks = np.empty((P, nS, NW), np.int32)
        if nS:
            LIB.top_sd(P, self.N, NW, nS, ptr(self.wstart, I64), ptr(self.sl, I32), ptr(self.ss, I8), ptr(X, I64),
                       ptr(cs, I64), ptr(ks, I32), NTH)
        return cu, ku, cs, ks, pw

    def read(self, X, EX, LS, J, PILE=None, bad=None):
        """the top read at every position (X the light at the top, EX its power, LS the gauge's log). J: base2's own
        steps (power, race, EXP2, ES). PILE (CHECK ONLY): base2's own product of the light and the table.
        Returns per-position top 5, race winner, survivors after each place, and the work items."""
        X = np.asarray(X, np.int64); P = len(X); NW = self.NW; bad = {} if bad is None else bad
        if PILE is not None: PILE = np.asarray(PILE)[:, self.wperm]   # CHECK PATH in this table's word order (order-free run)
        def nb(k, v): bad[k] = bad.get(k, 0) + int(v)
        out = {k: np.zeros(P, np.int64) for k in ("lwrite", "table", "pwrite", "light", "reads", "bound", "end", "stand", "cmp")}
        out["surv"] = np.zeros((P, len(self.places)), np.int64); out["top5"] = np.zeros((P, 5), np.int64)
        out["win"] = np.zeros(P, np.int64); out["mu"] = np.zeros(P, np.int64)
        BATCH = 24
        for c0 in range(0, P, BATCH):
            Xb = X[c0:c0 + BATCH]; Pb = len(Xb)
            cu, ku, cs, ks, pw = self.ready(Xb)
            for i in range(Pb):
                q = c0 + i; x = Xb[i]; nx = sd(x)
                F = (self.he + int(EX[q]) + J.ES) * 65536 - int(LS[q]); I_ = F >> 16; e = int(J.EXP2[F & 0xFFFF]); s = 16 - I_
                F2 = -F; I2 = F2 >> 16; e2 = int(J.EXP2[F2 & 0xFFFF]); s2 = 16 - I2
                beta = J.power(self.bias, F2)                 # each word's standing amount at pile scale
                twos = Fraction(2) ** s; twot = Fraction(2) ** (s + s2)
                gm = Fraction(1, 2) + self.BMAX * twos * abs(e * e2 - twot) / (e * twot)
                gamma = -(-gm.numerator // gm.denominator); delta = max(1, -(-twos.numerator // (e * twos.denominator)))
                mu = 2 * gamma + delta; out["mu"][q] = mu
                Xs = int(x.sum()); S1 = int(np.abs(x).sum())
                out["lwrite"][q] = int(ones(x).sum())
                out["table"][q] = int((nx[self.gl] * self.coef[None, :]).sum())
                out["pwrite"][q] = int(pw[i])
                out["light"][q] = 2 * int(nx[1:].sum())       # X and S1: N - 1 adds of light values each
                out["stand"][q] = int(self.BIAS_ONES.sum()) * int(ones(e2)) + int(ones(2 * beta).sum())
                V = 2 * beta; alive = np.arange(NW); reads = 0; bnd = 0; cmp_ = 0
                part = np.zeros(NW, np.int64) if PILE is not None else None
                if PILE is not None:
                    S_all = J.power(PILE[q], F) + self.bias; s5 = np.partition(S_all, NW - 5)[NW - 5]
                for j, b in enumerate(self.places):
                    if b in self.up:
                        u = self.up_order.index(b); add = cu[i, u] << b; cost = ku[i, u]
                    else:
                        sidx = self.sd_places.index(b); add = (2 * cs[i, sidx]) << b; cost = ks[i, sidx]
                    V = V + add[alive]; reads += int(cost[alive].sum(dtype=np.int64))
                    if part is not None: part += add
                    if j == len(self.places) - 1: break
                    R2 = self.B[j] * S1; thr = 2 * R2 + 2 * mu
                    if part is not None:                      # CHECK: the bound holds for every word
                        nb("bound broken (any word)", (np.abs(2 * PILE[q] - self.kcorr * Xs - part) > R2).sum())
                    na = len(alive); v5 = int(np.partition(V, na - 5)[na - 5]); keep = V > v5 - thr
                    Bj = self.B[j]
                    bnd += int(ones(Bj)) * int(ones(S1)) + int(ones(2 * mu)) + int(ones(thr))   # R2 = B x S1 a product (1s x 1s)
                    cmp_ += 2 * na
                    if PILE is not None:                      # CHECK: every word set aside is below the true 5th-best
                        nb("a word set aside that is not below the 5th-best", (S_all[alive[~keep]] >= s5).sum())
                    alive = alive[keep]; V = V[keep]; out["surv"][q, j] = len(alive)
                out["surv"][q, len(self.places) - 1] = len(alive)
                pile2 = V - 2 * beta[alive] + self.kcorr * Xs; assert not (pile2 & 1).any()
                pile = pile2 >> 1
                sc = J.power(pile, F) + self.bias[alive]
                if PILE is not None:
                    nb("words read to the end vs the direct product", (pile != PILE[q, alive]).sum())
                    nb("every word's doubled pile from its places vs the direct product", (part + self.kcorr * Xs != 2 * PILE[q]).sum())
                out["end"][q] = int(len(alive) * sd(np.array([Xs]))[0] + ones(2 * beta[alive]).sum() + ones(pile).sum() * int(ones(e))
                                    + self.BIAS_ONES[alive].sum())
                out["reads"][q] = reads; out["bound"][q] = bnd; out["cmp"][q] = cmp_
                Sp = np.full(NW, -(1 << 62), np.int64); Sp[alive] = sc
                idx = np.argpartition(-Sp, 5)[:5]; o = np.argsort(-Sp[idx]); out["top5"][q] = self.wperm[idx[o]]
                out["win"][q] = self.wperm[J.race(Sp)]
            del cu, ku, cs, ks
        out["bad"] = bad
        return out
    TOP_ITEMS = ["table", "reads", "light", "bound", "end", "stand", "pwrite", "lwrite"]
