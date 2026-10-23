"""DEVELOP family, shared pieces. A camera develops only the strongest photons: here each stored value keeps only its
top k photons in the single-wave writing (every lit place up +1 or a hole -1, no two lit places side by side), and
what lies below the last kept place is rounded -- either to the nearest (deterministic) or by chance (dithered, exact
on average, the way a phone stacks tremor-shifted frames). Or every value of a word is cut at one place fixed by the
word's loudest value (one exposure per word). The developed tables are judged ONLY by words(...) out of 30.

Everything in developing, coding and decoding is whole numbers: shifts, adds, subtracts, ANDs, ORs, counts, table reads.
Stored sizes are real bytes from real coders run here: Python's lzma on the byte streams, and the adaptive binary range
coder in develop_rc.c (whole numbers only). Every side number (stream lengths, per-word places, scales) is counted.
Every stored form is decoded back and checked equal (np.array_equal) to the developed table that was judged.
"""
import os, ctypes, lzma, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "develop_results.json")
NV = 50257 * 768

# ---------------------------------------------------------------- whole-number helpers
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)

def popcount(x):                                       # photons in a non-negative whole number (table reads)
    x = np.asarray(x, dtype=np.int64); n = np.zeros(x.shape, np.int64)
    while x.any(): n += _POP[x & 0xFFFF]; x = x >> 16
    return n

def bitlen(x):                                         # place of the top 1, plus one (0 for 0): compares and shifts
    x = np.asarray(x, dtype=np.int64).copy(); b = np.zeros(x.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        m = (x >> s) > 0; b += m * s; x = np.where(m, x >> s, x)
    return b + (x > 0)

def naf_split(a):
    """a >= 0 -> (up, hole) masks: a = up - hole, no two lit places side by side (shift, add, XOR, AND)"""
    a = np.asarray(a, dtype=np.int64); ah = a >> 1; a3 = a + ah; c = ah ^ a3
    return a3 & c, ah & c

def naf_photons(V):
    p, m = naf_split(np.abs(np.asarray(V, dtype=np.int64))); return popcount(p | m)

def zigzag(v): v = np.asarray(v, dtype=np.int64); return ((v << 1) ^ (v >> 63)).astype(np.int64)
def unzigzag(z): z = np.asarray(z, dtype=np.int64); return (z >> 1) ^ (-(z & 1))

# ---------------------------------------------------------------- developing
CH = 4096                                              # rows at a time, to stay small in memory

def develop_topk(A, k, seed=None, shared=False):
    """each value keeps its top k photons (single-wave writing); below the last kept place q:
    seed None -> nearest (the NAF tail is never more than q/3, so this IS the top k photons exactly)
    seed s    -> by chance: add a random whole number in [0, q) then cut (floor), so the mean is exact."""
    A = np.asarray(A); out = np.empty(A.shape, np.int32)
    rng = None if seed is None else np.random.default_rng(seed)
    if shared: R16 = rng.integers(0, 1 << 16, size=A.shape[1], dtype=np.int64)
    for r in range(0, A.shape[0], CH):
        v = A[r:r + CH].astype(np.int64); a = np.abs(v); p, m = naf_split(a); lit = p | m; cnt = popcount(lit)
        for _ in range(max(int(cnt.max()) - k, 0)):    # put out the faintest photon while more than k are lit
            sel = cnt > k; lit = np.where(sel, lit & (lit - 1), lit); cnt = cnt - sel
        q = lit & (-lit)                               # the last kept place, as a power of two (0 when a == 0)
        keep = ~(q - 1)
        if rng is None:
            a2 = (a + (q >> 1)) & keep
            assert np.array_equal(a2, (p & keep) - (m & keep))
        else:
            if shared:                                 # one tremor for the whole table: line i's offset is the same
                u = (R16[None, :] * q) >> 16           # fraction of the step for every word
            else:
                u = rng.integers(0, 1 << 62, size=a.shape, dtype=np.int64) & (q - 1)
            a2 = (a + u) & keep
        out[r:r + CH] = np.where(v < 0, -a2, a2)
    return out

def develop_cut(A, b, seed=None, shared=False):
    """one place cut per word: the word's loudest value reaches place top-1; every value keeps places >= top-b.
    Rounded to the nearest (seed None) or by chance (dither, exact on average). Returns (developed, cut place per word)."""
    A = np.asarray(A); out = np.empty(A.shape, np.int32); P = np.empty(A.shape[0], np.int64)
    rng = None if seed is None else np.random.default_rng(seed)
    if shared: R16 = rng.integers(0, 1 << 16, size=A.shape[1], dtype=np.int64)
    for r in range(0, A.shape[0], CH):
        v = A[r:r + CH].astype(np.int64); a = np.abs(v)
        top = bitlen(np.bitwise_or.reduce(a, axis=1)); pc = np.maximum(top - b, 0); P[r:r + CH] = pc
        q = (np.int64(1) << pc)[:, None]; keep = ~(q - 1)
        if rng is None: u = q >> 1
        elif shared: u = (R16[None, :] * q) >> 16
        else: u = rng.integers(0, 1 << 62, size=a.shape, dtype=np.int64) & (q - 1)
        a2 = (a + u) & keep
        out[r:r + CH] = np.where(v < 0, -a2, a2)
    return out, P

# ---------------------------------------------------------------- coders
_rc = ctypes.CDLL(os.path.join(HERE, "develop_rc.dylib"))
_rc.rc_encode.restype = ctypes.c_int64; _rc.rc_decode.restype = ctypes.c_int64
_P = ctypes.c_void_p

def rc_encode(sym, ctx, nbits, nctx):
    assert nbits <= 16 and nctx <= 65536
    sym = np.ascontiguousarray(sym, dtype=np.uint16) if len(sym) == 0 or int(np.max(sym)) < 65536 else None
    ctx = np.ascontiguousarray(ctx, dtype=np.uint16)
    assert len(sym) == len(ctx) and (len(sym) == 0 or (int(sym.max()) < (1 << nbits) and int(ctx.max()) < nctx))
    cap = len(sym) * nbits // 8 + len(sym) // 16 + 65536
    while True:
        buf = np.empty(cap, np.uint8)
        n = _rc.rc_encode(sym.ctypes.data_as(_P), ctx.ctypes.data_as(_P), ctypes.c_int64(len(sym)), ctypes.c_int(nbits),
                          ctypes.c_int64(nctx), buf.ctypes.data_as(_P), ctypes.c_int64(cap))
        if n <= cap: return buf[:n].tobytes()
        cap = n + 65536

def rc_decode(data, ctx, nbits, nctx):
    ctx = np.ascontiguousarray(ctx, dtype=np.uint16); buf = np.frombuffer(data, np.uint8).copy()
    sym = np.empty(len(ctx), np.uint16)
    _rc.rc_decode(buf.ctypes.data_as(_P), ctypes.c_int64(len(buf)), ctx.ctypes.data_as(_P), ctypes.c_int64(len(ctx)),
                  ctypes.c_int(nbits), ctypes.c_int64(nctx), sym.ctypes.data_as(_P))
    return sym

def xz(b): return lzma.compress(b, preset=6)
def unxz(b): return lzma.decompress(b)

def nb(x): return max(int(x).bit_length(), 1)

# ---- photons coder: the brightest photon of every value first, then the next brightest, ... (develop in rounds)
LINES = 768

def _photon_chunk(V, kmax):
    """V (rows, 768) whole numbers, one slice. Returns per round (symbols uint8, contexts uint8)."""
    V = np.asarray(V, dtype=np.int64); a = np.abs(V).ravel(); s = (V.ravel() < 0).astype(np.int64)
    p, m = naf_split(a); lit = p | m; del p
    t = bitlen(lit)
    line = np.tile(np.arange(LINES, dtype=np.int64), V.shape[0])
    sym1 = np.where(t > 0, 2 * t - 1 + s, 0)            # 0 dark; else 2t-1 (up) or 2t (down); t = top place + 1
    out = [(sym1.astype(np.uint8), None)]                # round 1's context is the line: the reader knows it
    prev = t - 1; alive = t > 0; lit = np.where(alive, lit ^ (np.int64(1) << np.maximum(prev, 0)), 0)
    for j in range(2, kmax + 1):
        room = alive & (prev >= 2)                      # a further photon can only sit 2+ places lower
        l = lit[room]; pr = prev[room]; tt = bitlen(l); has = tt > 0
        hb = np.where(has, (m[room] >> np.maximum(tt - 1, 0)) & 1, 0)
        sym = np.where(has, 1 + 2 * (pr - (tt - 1) - 2) + hb, 0)
        out.append((sym.astype(np.uint8), pr.astype(np.uint8)))
        prev[room] = np.where(has, tt - 1, -1); alive[~room] = False; alive[room] = has
        lit[room] = np.where(has, l ^ (np.int64(1) << np.maximum(tt - 1, 0)), 0)
    assert not (alive & (lit != 0)).any(), "more photons than kmax"
    return out

def photon_streams(V, kmax):
    """round 1: per value its top place (0 = dark) and sign, context = its line; round j: the gap down to the j-th
    photon (single waves are never side by side, so the gap is 2+) and whether it is up or a hole, or 'stop';
    context = the place of the photon above it. Values are taken word after word, line after line (fixed order)."""
    parts = [_photon_chunk(V[r:r + CH], kmax) for r in range(0, V.shape[0], CH)]
    st = []
    for j in range(kmax):
        sym = np.concatenate([p[j][0] for p in parts])
        ctx = np.tile(np.arange(LINES, dtype=np.uint16), V.shape[0]) if j == 0 else np.concatenate([p[j][1] for p in parts])
        for p in parts: p[j] = None
        st.append((sym, ctx, nb(max(int(sym.max()) if len(sym) else 1, 1)), LINES if j == 0 else 64))
    return st

def _photon_rebuild(shape, kmax, get_round):
    """rebuild the values round by round; get_round(j, ctx) returns that round's symbols. Each round is read whole
    (its contexts come from the round before), then laid onto the values a slice at a time."""
    n = shape[0] * shape[1]; step = CH * LINES
    sym1 = get_round(0, np.tile(np.arange(LINES, dtype=np.uint16), shape[0]))
    a = np.zeros(n, np.int32); prev = np.empty(n, np.int8); alive = np.empty(n, bool); neg = np.empty(n, bool)
    for r in range(0, n, step):
        s1 = np.asarray(sym1[r:r + step]).astype(np.int32); t = (s1 + 1) >> 1
        neg[r:r + step] = (t > 0) & ((s1 & 1) == 0); alive[r:r + step] = t > 0; prev[r:r + step] = t - 1
        a[r:r + step] = np.where(t > 0, np.int32(1) << np.maximum(t - 1, 0), 0)
    del sym1
    for j in range(1, kmax):
        room = alive & (prev >= 2)
        sym = get_round(j, prev[room].astype(np.uint16))
        o = 0
        for r in range(0, n, step):
            rm = room[r:r + step]; c = int(rm.sum())
            sj = np.asarray(sym[o:o + c]).astype(np.int32); o += c
            pr = prev[r:r + step][rm].astype(np.int32)
            has = sj > 0; place = pr - (((sj - 1) >> 1) + 2); hb = ((sj - 1) & 1) == 1
            bit = np.where(has, np.int32(1) << np.maximum(place, 0), 0).astype(np.int32); bit[hb & has] *= -1
            av = a[r:r + step]; av[rm] += bit
            pv = prev[r:r + step]; pv[rm] = np.where(has, place, -1)
            al = alive[r:r + step]; al &= rm; al[rm] = has
        del sym, room
    np.negative(a, out=a, where=neg)
    return a.reshape(shape)

def same(B, V):                                         # np.array_equal, a slice at a time (no full-size copies)
    if B.shape != V.shape: return False
    return all(np.array_equal(B[r:r + CH].astype(np.int64), V[r:r + CH].astype(np.int64)) for r in range(0, V.shape[0], CH))

def store_photons(V, kmax):
    """real stored bytes of V as photon rounds: (a) the range coder with contexts, (b) lzma, one byte per symbol.
    Both decoded and checked. Side info: per round byte count (4 bytes) and symbol width (1 byte); kmax, shape (12)."""
    st = photon_streams(V, kmax)
    rc, lz, widths, nsym = [], [], [], []
    while st:                                           # code a round, then let it go
        sym, ctx, w, nctx = st.pop(0)
        rc.append(rc_encode(sym, ctx, w, nctx)); lz.append(xz(sym.tobytes())); widths.append(w); nsym.append(len(sym))
        del sym, ctx
    back = _photon_rebuild(V.shape, kmax, lambda j, ctx: rc_decode(rc[j], ctx, widths[j], LINES if j == 0 else 64))
    ok_rc = same(back, V); del back
    back = _photon_rebuild(V.shape, kmax, lambda j, ctx: np.frombuffer(unxz(lz[j]), np.uint8))
    ok_lz = same(back, V); del back
    side = 12 + 5 * kmax
    return dict(rc_bytes=sum(len(b) for b in rc) + side, lzma_bytes=sum(len(b) for b in lz) + side,
                rc_rounds=[len(b) for b in rc], lzma_rounds=[len(b) for b in lz], nsym=nsym, side=side,
                roundtrip_ok=ok_rc and ok_lz)

# ---- one cut per word: the whole numbers above the cut, the cut places as side info
def store_cut(V, P):
    """V developed with one cut place per word P. Stores Q = V >> P (exact: V is a multiple of 2^P) zigzagged,
    range-coded with the line as context, and the 50,257 cut places (range-coded, one context). Checked by decoding."""
    z = np.empty(V.size, np.uint16); w = 1
    for r in range(0, V.shape[0], CH):
        v = V[r:r + CH].astype(np.int64); q = v >> P[r:r + CH, None]
        assert np.array_equal(q << P[r:r + CH, None], v)
        zz = zigzag(q).ravel(); assert int(zz.max()) < 65536; z[r * LINES:r * LINES + zz.size] = zz
    w = nb(int(z.max())); line = np.tile(np.arange(LINES, dtype=np.uint16), V.shape[0]); zero = np.zeros(len(P), np.uint16)
    bq = rc_encode(z, line, w, LINES); bp = rc_encode(P, zero, 5, 1)
    zb = rc_decode(bq, line, w, LINES); Pb = rc_decode(bp, zero, 5, 1).astype(np.int64); del line
    ok = True
    for r in range(0, V.shape[0], CH):
        back = unzigzag(zb[r * LINES:(r + CH) * LINES].astype(np.int64)).reshape(-1, LINES) << Pb[r:r + CH, None]
        ok &= bool(np.array_equal(back, V[r:r + CH].astype(np.int64)))
    side = 12 + 2 * 5
    lz = len(xz((z.astype(np.uint8) if w <= 8 else z).tobytes())) + len(xz(P.astype(np.uint8).tobytes()))
    return dict(bytes=len(bq) + len(bp) + side, q_bytes=len(bq), place_bytes=len(bp), side=side, width=w,
                lzma_bytes=lz + side, roundtrip_ok=ok)

# ---- plain whole numbers, byte planes, lzma (the generic reference coder)
def store_planes_lzma(V):
    z = zigzag(np.asarray(V, dtype=np.int64)).ravel(); top = int(z.max()); nbp = max(1, (top.bit_length() + 7) // 8)
    blobs = [xz(((z >> (8 * b)) & 255).astype(np.uint8).tobytes()) for b in range(nbp)]
    back = np.zeros(len(z), np.int64)
    for b, bl in enumerate(blobs): back |= np.frombuffer(unxz(bl), np.uint8).astype(np.int64) << (8 * b)
    ok = bool(np.array_equal(unzigzag(back).reshape(np.asarray(V).shape), np.asarray(V, dtype=np.int64)))
    return dict(bytes=sum(len(b) for b in blobs) + 12 + 4 * nbp, roundtrip_ok=ok)

# ---------------------------------------------------------------- results
def record(row):
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    rows = [r for r in rows if r["name"] != row["name"]] + [row]
    json.dump(rows, open(OUT, "w"), indent=1)
    print("RESULT", json.dumps(row), flush=True)


# ================================================================ the extremes (a word in very few bits)
def word_mean_mag(A):
    """each word's mean loudness, whole numbers: sum of |v| times 21845 (= 2^24 / 768, rounded down), shifted by 24"""
    A = np.asarray(A); s = np.concatenate([np.abs(A[r:r + CH].astype(np.int64)).sum(1) for r in range(0, A.shape[0], CH)])
    return (s * 21845 + (1 << 23)) >> 24

def develop_sign(A, seed_shuffle=None):
    """only the sign of each value, every value at the word's mean loudness. seed_shuffle: CONTROL, each word's signs
    put on random lines (same count of up and down, the distribution kept, the placement lost)."""
    A = np.asarray(A); c = word_mean_mag(A); s = A < 0
    if seed_shuffle is not None:
        s = develop_shuffle_words(s, seed_shuffle)
    return np.where(s, -c[:, None], c[:, None]).astype(np.int32), c, s

def develop_sign_dither(A, seed):
    """one bit per value by chance: +L or -L with L the word's loudest |v|; chance of +L is (L + v) / 2L, so each value
    is exact on average (a 1-bit halftone)."""
    A = np.asarray(A); L = np.abs(A.astype(np.int64)).max(1); out = np.empty(A.shape, np.int32)
    rng = np.random.default_rng(seed)
    for r in range(0, A.shape[0], CH):
        v = A[r:r + CH].astype(np.int64); l = L[r:r + CH, None]
        u = rng.integers(0, np.broadcast_to(2 * l, v.shape), dtype=np.int64)   # a whole number in [0, 2L)
        out[r:r + CH] = np.where(u < l + v, l, -l)
    return out, L, out < 0

def develop_loud(A, n, seed_random_lines=None):
    """only the n loudest values of each word survive, each as its single top photon (+-2^place); the rest are dark.
    seed_random_lines: CONTROL, n lines picked at random per word instead of the loudest (each keeps its own top photon)."""
    A = np.asarray(A); out = np.zeros(A.shape, np.int32)
    rng = None if seed_random_lines is None else np.random.default_rng(seed_random_lines)
    t1 = develop_topk(A, 1)
    for r in range(0, A.shape[0], CH):
        a = np.abs(A[r:r + CH].astype(np.int64))
        key = (-a) if rng is None else rng.integers(0, 1 << 62, size=a.shape, dtype=np.int64)
        idx = np.argsort(key, axis=1, kind="stable")[:, :n]
        np.put_along_axis(out[r:r + CH], idx, np.take_along_axis(t1[r:r + CH], idx, axis=1), axis=1)
    return out

def develop_shuffle_words(A, seed):
    """CONTROL: each word's own 768 values, exactly, put on random lines (each word keeps its exact histogram)."""
    A = np.asarray(A); rng = np.random.default_rng(seed); out = np.empty(A.shape, A.dtype)
    for r in range(0, A.shape[0], CH):
        perm = np.argsort(rng.integers(0, 1 << 62, size=A[r:r + CH].shape, dtype=np.int64), axis=1, kind="stable")
        out[r:r + CH] = np.take_along_axis(A[r:r + CH], perm, axis=1)
    return out

def store_sign(S, c):
    """signs (one bit a value, range coder, line as context) + one loudness per word (range coder, its low 8 places and
    its high places as two streams)."""
    S = np.asarray(S).astype(np.uint16).ravel(); c = np.asarray(c, dtype=np.int64)
    line = np.tile(np.arange(LINES, dtype=np.uint16), len(c)); zero = np.zeros(len(c), np.uint16)
    lo, hi = c & 255, c >> 8; wh = nb(int(hi.max()))
    bs = rc_encode(S, line, 1, LINES); bl = rc_encode(lo, zero, 8, 1); bh = rc_encode(hi, zero, wh, 1)
    ok = (np.array_equal(rc_decode(bs, line, 1, LINES), S)
          and np.array_equal(rc_decode(bl, zero, 8, 1).astype(np.int64) + (rc_decode(bh, zero, wh, 1).astype(np.int64) << 8), c))
    lz = len(xz(np.packbits(S.astype(np.uint8)).tobytes())) + len(xz(c.astype(np.uint32).tobytes()))
    side = 12 + 15
    return dict(rc_bytes=len(bs) + len(bl) + len(bh) + side, lzma_bytes=lz + side, sign_bytes=len(bs),
                loud_bytes=len(bl) + len(bh), side=side, roundtrip_ok=bool(ok))

# ================================================================ versions: build, judge, store
import hashlib

# ================================================================ one negative, two prints (T printed from E)
GAIN = None
def gains():
    global GAIN
    if GAIN is None: GAIN = np.load(os.path.join(HERE, "develop_gain24.npy"))   # written once by develop_gain.py
    return GAIN

def print_top(E2):
    """T printed from the entry table through the lens, whole numbers: s = G x E; c = 768 s - sum_lines s;
    y = (c + 2^23) >> 24; T = (y x 1431655766 + 2^43) >> 44   (= C diag(gain) E at 2^-12, as the converter wrote T)"""
    G = gains(); E2 = np.asarray(E2); out = np.empty(E2.shape, np.int32)
    for r in range(0, E2.shape[0], CH):
        sv = E2[r:r + CH].astype(np.int64) * G[None, :]
        c = 768 * sv - sv.sum(1, keepdims=True); y = (c + (1 << 23)) >> 24
        assert int(np.abs(y).max()) < (1 << 32)
        out[r:r + CH] = (y * 1431655766 + (1 << 43)) >> 44
    return out

def lens_places():                                     # each line's gain place (top 1 of the whole-number gain)
    return bitlen(gains()) - 1

def develop_lines(A, c, cap=None):
    """exposure matched to the lens: line i keeps the places >= e_i = c - (its gain's place), clipped at 0 -- a line
    the top reads through a strong gain is exposed finely, a line with a faint gain coarsely. Rounded to the nearest."""
    e = np.maximum(c - lens_places(), 0)
    if cap is not None: e = np.minimum(e, cap)         # no line exposed coarser than 2^cap (the entry reads every line)
    q = (np.int64(1) << e)[None, :]; keep = ~(q - 1)
    A = np.asarray(A); out = np.empty(A.shape, np.int32)
    for r in range(0, A.shape[0], CH):
        v = A[r:r + CH].astype(np.int64); a = np.abs(v); a2 = (a + (q >> 1)) & keep
        out[r:r + CH] = np.where(v < 0, -a2, a2)
    return out, e

def store_lines(V, e):
    """V developed with one cut place per LINE e (768): Q = V >> e zigzagged, range coder with the line as context;
    the 768 places as side info. Checked by decoding."""
    z = np.empty(V.size, np.uint16)
    for r in range(0, V.shape[0], CH):
        v = V[r:r + CH].astype(np.int64); q = v >> e[None, :]
        assert np.array_equal(q << e[None, :], v)
        zz = zigzag(q).ravel(); assert int(zz.max()) < 65536; z[r * LINES:r * LINES + zz.size] = zz
    w = nb(int(z.max())); line = np.tile(np.arange(LINES, dtype=np.uint16), V.shape[0])
    bq = rc_encode(z, line, w, LINES); zb = rc_decode(bq, line, w, LINES); del line
    ok = True
    for r in range(0, V.shape[0], CH):
        back = unzigzag(zb[r * LINES:(r + CH) * LINES].astype(np.int64)).reshape(-1, LINES) << e[None, :]
        ok &= bool(np.array_equal(back, V[r:r + CH].astype(np.int64)))
    side = 12 + 768 + 5
    lz = len(xz((z.astype(np.uint8) if w <= 8 else z).tobytes()))
    return dict(rc_bytes=len(bq) + side, lzma_bytes=lz + side, width=w, side=side, roundtrip_ok=ok)

def build(spec, E, T):
    """spec: {'E': sub or None, 'T': sub or None}; sub: {'how': ..., params}. Returns {'E': (D, aux), 'T': (D, aux)}"""
    out = {}
    for tab, A in (("E", E), ("T", T)):
        sub = spec.get(tab)
        if sub is None: continue
        if sub["how"] == "print":                       # T printed from the (developed) E through the lens
            src = out["E"][0] if "E" in out else E
            out[tab] = (print_top(src), None)
        else: out[tab] = build_one(sub, A)
    return out

def mean_word(A):
    """the white plate: each line's average over all 50,257 words, whole numbers (sum, times 2^40/50257 rounded, shift
    by 40). Stored once: 768 numbers."""
    A = np.asarray(A); R = ((1 << 40) + 50257 // 2) // 50257
    s = np.zeros(A.shape[1], np.int64)
    for r in range(0, A.shape[0], CH): s += A[r:r + CH].astype(np.int64).sum(0)
    return (s * R + (1 << 39)) >> 40

def build_one(sub, A):
    how = sub["how"]
    if how == "topk": return develop_topk(A, sub["k"], sub.get("seed"), sub.get("shared", False)), None
    if how == "cut": return develop_cut(A, sub["b"], sub.get("seed"), sub.get("shared", False))
    if how == "sign": D, c, s = develop_sign(A, sub.get("shuffle")); return D, (c, s)
    if how == "signdither": D, c, s = develop_sign_dither(A, sub["seed"]); return D, (c, s)
    if how == "loud": return develop_loud(A, sub["n"], sub.get("random_lines")), None
    if how == "wordshuffle": return develop_shuffle_words(A, sub["seed"]), None
    if how == "same": return np.asarray(A), None
    if how == "lines": return develop_lines(A, sub["c"], sub.get("cap"))
    if how == "dark":                                   # develop only each word's departure from the white plate
        A = np.asarray(A); m = mean_word(A); dep = np.empty(A.shape, np.int32)
        for r in range(0, A.shape[0], CH): dep[r:r + CH] = A[r:r + CH].astype(np.int64) - m[None, :]
        Dd, aux = build_one(sub["inner"], dep); del dep
        D = np.empty(A.shape, np.int32)
        for r in range(0, A.shape[0], CH): D[r:r + CH] = Dd[r:r + CH].astype(np.int64) + m[None, :]
        return D, (m, Dd, aux)
    raise ValueError(how)

def sha(D):
    h = hashlib.sha256()
    for r in range(0, D.shape[0], CH): h.update(np.ascontiguousarray(D[r:r + CH], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]

def store(sub, D, aux):
    how = sub["how"]
    if how == "topk": return store_photons(D, sub["k"])
    if how == "loud": return store_photons(D, 1)
    if how == "cut":
        r = store_cut(D, aux); r["rc_bytes"] = r.pop("bytes"); return r
    if how in ("sign", "signdither"): c, s = aux; return store_sign(s, c)
    if how in ("wordshuffle", "same"): return store_photons(D, 12)
    if how == "lines": return store_lines(D, aux)
    if how == "print":                                  # nothing of T is stored: only the lens, 768 whole-number gains
        gb = len(xz(gains().astype(np.int32).tobytes())); return dict(rc_bytes=gb, lzma_bytes=gb, roundtrip_ok=True,
                                                                     note="printed from the stored E through the 768 gains")
    if how == "dark":                                   # the departure as its own kind, plus the white plate (768 x 4 bytes)
        m, Dd, inner_aux = aux
        r = store(sub["inner"], Dd, inner_aux)
        plate = len(xz(m.astype(np.int32).tobytes()))
        for key in ("rc_bytes", "lzma_bytes"):
            if key in r: r[key] += plate
        r["plate_bytes"] = plate; return r
    raise ValueError(how)

def load_tables():
    M = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
    return np.asarray(M["wte"], dtype=np.int64), np.ascontiguousarray(M["head.w"].T)

def run(specs, mode):
    """mode 'words': load the machine (table_bits), build each developed table, judge it by words(...) /30.
    mode 'store': without the machine, build the same tables again (same seeds; checked by hash), code them for real,
    decode and check, and record bytes. Both modes add to develop_results.json under the spec's name."""
    import sys, time
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    byname = {r["name"]: r for r in rows}
    if mode == "words":
        sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
        import table_bits as tb
        tb.words()                                      # the machine's own answers first (and a check: 30)
        tb.E = None; tb._WTE0 = tb._WTE0.astype(np.int32)  # same whole numbers in 32 bits: the machine reads 5 rows of it
        import gc; gc.collect(); E, T = tb._WTE0, tb.T
    else:
        E, T = load_tables()
    import gc
    if len(sys.argv) > 2: specs = [specs[int(i)] for i in sys.argv[2].split(",")]   # one fresh process per spec
    for spec in specs:
        name = spec["name"]; t0 = time.time()
        row = dict(byname.get(name, {})); row.update(name=name, spec={k: v for k, v in spec.items() if k != "name"})
        built = build(spec, E, T)
        hashes = {tab: sha(D) for tab, (D, _) in built.items()}
        if mode == "words":
            row["hash"] = hashes
            for tab, (D, aux) in built.items():
                A = E if tab == "E" else T
                row[f"photons_{tab}"] = round(float(naf_photons(D[::7]).mean()), 3)
                row[f"photons_{tab}_before"] = round(float(naf_photons(A[::7]).mean()), 3)
                if spec[tab]["how"] == "dark": row[f"photons_{tab}_departure_stored"] = round(float(naf_photons(aux[1][::7]).mean()), 3)
                row[f"changed_share_{tab}"] = round(float((D[::7] != A[::7]).mean()), 4)
                row[f"over_int16_{tab}"] = int((np.abs(D.astype(np.int64)) > 32767).sum()) if tab == "T" else None
            Ds = {tab: D for tab, (D, _) in built.items()}; del built; gc.collect()
            row["words"] = int(tb.words(entry=Ds.get("E"), top=Ds.get("T")))
            del Ds
        else:
            assert row.get("hash") in (None, hashes), f"{name}: tables differ from the ones judged"
            for tab, (D, aux) in built.items():
                twin = next((x for x in byname.values() if x["name"] != name and x.get("hash", {}).get(tab) == hashes[tab]
                             and f"store_{tab}" in x), None)
                if twin is not None:                    # the very same table (same hash) was already coded and checked
                    r = dict(twin[f"store_{tab}"]); r["same_table_as"] = twin["name"]; row[f"store_{tab}"] = r; continue
                r = store(spec[tab], D, aux)
                r["bits_per_value"] = round(8 * min(r.get("rc_bytes", 1 << 62), r.get("lzma_bytes", 1 << 62)) / D.size, 4)
                r["bits_per_word"] = round(8 * min(r.get("rc_bytes", 1 << 62), r.get("lzma_bytes", 1 << 62)) / D.shape[0], 1)
                r["MB"] = round(min(r.get("rc_bytes", 1 << 62), r.get("lzma_bytes", 1 << 62)) / 1e6, 3)
                row[f"store_{tab}"] = r
            del built
        row[f"secs_{mode}"] = round(time.time() - t0, 1)
        record(row); byname[name] = row
