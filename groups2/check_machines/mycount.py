"""CHECKER'S OWN COUNTER of the group form (written here; it imports nothing from gbank, bgm.c, uh_rebuild, uh_perplace,
topg or gtop). One light x (P, N), one slab w (N, M) with its plates as segments [(name, a, b)], group size g, the
grouping key. For every plate: the per-place rule from the plate alone (L_b = the plate's lit signed digits at place b,
U = ceil(N / g) x the plate's neurons; signed digits where L_b < U), or a given mask (0xFFFF: every place up/hole;
0: every place signed). Each weight written bottom place to top with the offset C = 2^(top up/hole place): a binary
digit at an up/hole place, a signed digit (2 - (r mod 4) where r is odd) elsewhere, places 0..17.
The ready piles: every pattern's pile made as a matrix of +1/-1 against the group's lines (not by a walk, not by the
lowest-bit rule). The pile: sum over up/hole places of 2^b x (ready pile) + 2 x (signed part, as x @ the signed-digit
plate) + kcorr x X, halved; checked against the reference pile by the caller.
The work, per position (photons):
  light    the light written: its 1s
  table    the half table: the group total (lines 1..gk-1 once) + the Gray steps over lines 0..gk-2, the steps counted
           here by enumerating the reflected Gray code's turning lines; each step a light value's signed digits
  pwrite   every half-table entry read by the slab (the pattern with the group's last line up, or its flip): its 1s
  reads    every up/hole read: the ready pile's signed digits (per plate)
  sdreads  every lit signed digit: its light value's signed digits (per plate)
  xsum     X: N - 1 adds of light values (lines 1..N-1)
  xoff     kcorr X taken from each neuron: kcorr X's signed digits (+ (signed digits of kcorr - 1) x X's, if more)
  plain    the light's 1s x the plate's 1s per line (per plate)
"""
import numpy as np
NPL = 18
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def sdc(A):                                                      # signed digits, counted digit by digit (no a ^ 3a)
    a = np.abs(np.asarray(A, np.int64)).copy(); c = np.zeros(a.shape, np.int64)
    while a.any():
        odd = (a & 1) == 1; c += odd; a = np.where(odd, np.where((a & 3) == 3, a + 1, a - 1), a) >> 1
    return c
def lit_places(w):                                               # the plate's own signed digits: lit places as a bit mask
    a = np.abs(np.asarray(w, np.int64)); return (a ^ (3 * a)) >> 1
def rule(wseg, N, g):
    G = -(-N // g); U = G * wseg.shape[1]; lit = lit_places(wseg)
    Lb = [int(((lit >> b) & 1).sum()) for b in range(NPL)]
    return sum(1 << b for b in range(16) if not Lb[b] < U), Lb
def groups(N, g, key):
    o = np.random.default_rng(key).permutation(N); return [o[s:s + g] for s in range(0, N, g)]
def gray_turns(n):                                               # turns of each of n lines in the reflected Gray code walk
    t = np.zeros(n, np.int64); cur = 0
    for s in range(1, 1 << n):
        nxt = s ^ (s >> 1); t[(cur ^ nxt).bit_length() - 1] += 1; cur = nxt
    return t
_GT = {}
def expand(w, up):
    """w (N, M), up (M,) masks -> E (N, M) binary digits as bits, Wsd (N, M) the signed part's value, lit signed digit
    counts per (line, neuron), kcorr (M,)"""
    w = np.asarray(w, np.int64); N, M = w.shape; up = np.asarray(up, np.int64)
    hi = np.array([int(u).bit_length() - 1 for u in up]); C = np.where(up > 0, np.left_shift(np.int64(1), np.maximum(hi, 0)), 0)
    r = w + C[None, :]; E = np.zeros((N, M), np.int64); Wsd = np.zeros((N, M), np.int64); nlit = np.zeros((N, M), np.int64)
    for b in range(NPL):
        isup = (((up >> b) & 1) == 1)[None, :] if b < 16 else np.zeros((1, M), bool)
        odd = (r & 1) == 1
        e = (odd & isup).astype(np.int64); d = np.where(odd & ~isup, 2 - (r & 3), 0)
        E |= e << b; Wsd += d << b; nlit += d != 0; r = (r - e - d) >> 1
    assert not r.any()
    assert np.array_equal(E - C[None, :] + Wsd, w), "digits do not give the plate back"
    return E, Wsd, nlit, up - 2 * C

def slab_work(x, w, segs, g, key, masks=None, want_pile=True):
    """returns (pile or None, counts dict of per-position arrays, masks per plate)"""
    x = np.asarray(x, np.int64); P, N = x.shape; w = np.asarray(w, np.int64); M = w.shape[1]
    up = np.zeros(M, np.int64); chosen = {}
    for name, a, b in segs:
        m = masks[name] if masks is not None else rule(w[:, a:b], N, g)[0]
        up[a:b] = m; chosen[name] = m
    E, Wsd, nlit, kcorr = expand(w, up)
    X = x.sum(1); sx = sdc(x); anyup = bool(up.any())
    c = dict(light=ones(x).sum(1), table=np.zeros(P, np.int64), pwrite=np.zeros(P, np.int64),
             xsum=sx[:, 1:].sum(1) if anyup else np.zeros(P, np.int64), xoff=np.zeros(P, np.int64))
    for kv in np.unique(kcorr):
        if kv == 0: continue
        nj = int((kcorr == kv).sum()); c["xoff"] += nj * sdc(kv * X); dk = int(sdc(np.array([kv]))[0])
        if dk > 1: c["xoff"] += nj * (dk - 1) * sdc(X)
    o1 = ones(x)
    for name, a, b in segs:
        c[f"plain|{name}"] = o1 @ ones(w[:, a:b]).sum(1); c[f"sdreads|{name}"] = sx @ nlit[:, a:b].sum(1)
        c[f"reads|{name}"] = np.zeros(P, np.int64)
    D = 2 * (x @ Wsd) + kcorr[None, :] * X[:, None] if want_pile else None
    seg_of = np.zeros(M, np.int64)
    for si, (name, a, b) in enumerate(segs): seg_of[a:b] = si
    if anyup:
        for L in groups(N, g, key):
            gk = len(L); n = 1 << gk; top = n >> 1; full = n - 1
            if gk not in _GT: _GT[gk] = gray_turns(gk - 1)
            c["table"] += sx[:, L[1:]].sum(1) + sx[:, L[:-1]] @ _GT[gk]
            SG = np.where(((np.arange(n)[:, None] >> np.arange(gk)[None, :]) & 1) == 1, 1, -1)    # (n, gk)
            V = x[:, L] @ SG.T                                                                 # (P, n): every pattern's pile
            sV = sdc(V); Eg = E[L]                                                             # (gk, M)
            used = np.zeros(n, bool)
            for bpl in range(16):
                sel = np.flatnonzero((up >> bpl) & 1)
                if len(sel) == 0: continue
                pat = (((Eg[:, sel] >> bpl) & 1) << np.arange(gk)[:, None]).sum(0)          # (len(sel),)
                used[pat] = True
                for si, (name, a, b) in enumerate(segs):
                    ms = seg_of[sel] == si
                    if ms.any(): c[f"reads|{name}"] += sV @ np.bincount(pat[ms], minlength=n)
                if want_pile: D[:, sel] += V[:, pat] << bpl
            reps = np.unique(np.where(np.flatnonzero(used) & top, np.flatnonzero(used), np.flatnonzero(used) ^ full))
            c["pwrite"] += ones(V[:, reps]).sum(1)
    pile = None
    if want_pile:
        assert not (D & 1).any(); pile = D >> 1
    return pile, c, chosen
