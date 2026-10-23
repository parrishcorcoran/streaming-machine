"""TABLEDIFFUSE family -- error diffusion (halftoning) on the word tables.

His picture: a press with a few ink levels prints grey by carrying each dot's leftover to the next dot; the eye blurs
neighbouring dots, so the leftovers cancel and the grey comes out right. Here: the entry table E is developed with
round one's lens-matched exposure (line i keeps the places >= e_i = min(max(c - place(gain_i), 0), 12), rounded to
nearest) and each value's leftover grain is carried on -- to the next line of the same word (natural or random line
order), to the same line of the next word, or shaped twice (second order), or spread 2-D (Floyd-Steinberg). A token's
light enters as its word's row and the top read is a pile over the word's 768 lines, so grain that cancels along a
word's lines should matter less. T is printed from the developed E through the lens (develop_lib.print_top).

Everything that develops, codes and decodes is whole numbers (tablediffuse_ed.c: adds, shifts, ANDs, compares; the
range coder of develop_lib). The developed E is always on round one's grid (a multiple of 2^e_i on line i), so the very
same stored form and decoder as round one apply: the whole numbers above each line's cut, range coded with the line as
context, + the 768 cut places + the lens (768 gains, 3,072 bytes). Every stored form is decoded back and checked equal.
Judged ONLY by words(entry=D, top=print_top(D)) out of 30. Results: tablediffuse_results.json.
"""
import os, sys, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
import develop_lib as dl
OUT = os.path.join(HERE, "tablediffuse_results.json")
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
LENS_BYTES = 768 * 4                                   # the 768 gains, 4 bytes each (as check_develop_v9_store counts)

_ed = ctypes.CDLL(os.path.join(HERE, "tablediffuse_ed.dylib")); _ed.ed_develop.restype = ctypes.c_int64
_P = ctypes.c_void_p
MODES = {"nearest": 0, "lines": 1, "lines2": 2, "words": 3, "fs": 4, "leak": 5, "spread": 6}
UNITS = {"entry": 0, "step": 1, "top": 2}

def exposure(c, cap=12):
    p = dl.lens_places().astype(np.int64); e = np.minimum(np.maximum(c - p, 0), cap)
    return e.astype(np.int32), p.astype(np.int32)

def develop(E, c, how="nearest", unit="step", leak=0, line_seed=None, word_seed=None, cap=12):
    """E (50257, 768) int32 at 2^16 -> (developed int32, e). line_seed / word_seed: a fixed-seed random order of the
    lines (same for every word) / of the words; None = natural order."""
    E = np.ascontiguousarray(E, dtype=np.int32); n = E.shape[0]
    e, p = exposure(c, cap)
    lord = np.arange(768, dtype=np.int32) if line_seed is None else \
        np.random.default_rng(line_seed).permutation(768).astype(np.int32)
    word = np.arange(n, dtype=np.int64) if word_seed is None else np.random.default_rng(word_seed).permutation(n).astype(np.int64)
    out = np.empty(E.shape, np.int32)
    r = _ed.ed_develop(E.ctypes.data_as(_P), ctypes.c_int64(n), e.ctypes.data_as(_P), p.ctypes.data_as(_P),
                       ctypes.c_int(MODES[how]), ctypes.c_int(UNITS[unit]), ctypes.c_int(leak), lord.ctypes.data_as(_P),
                       word.ctypes.data_as(_P), out.ctypes.data_as(_P))
    assert r == 0
    q = (np.int64(1) << e.astype(np.int64))[None, :]
    for r0 in range(0, n, dl.CH):                      # every developed value sits on its line's grid
        assert not (out[r0:r0 + dl.CH].astype(np.int64) & (q - 1)).any()
    return out, e.astype(np.int64)

def grain_steps2(D, E, e):
    """mean square grain in units of each line's own step (a description of what the carry did, not a judge)"""
    s = 0.0
    for r in range(0, E.shape[0], dl.CH):
        d = (D[r:r + dl.CH].astype(np.int64) - E[r:r + dl.CH].astype(np.int64)) / (2.0 ** e[None, :])
        s += float((d * d).sum())
    return s / E.size

_tb = None
def machine():
    """load the machine once; keep E as int32 (the same whole numbers) to stay small"""
    global _tb
    if _tb is None:
        import table_bits as tb
        assert tb.words() == 30
        tb.E = None; tb.T = None; tb._WTE0 = tb._WTE0.astype(np.int32)   # keep only what words() reads
        import gc; gc.collect(); _tb = tb
    return _tb

def judge(D, parts=False):
    """words(entry=D, top=print_top(D)) out of 30. The printed top is put in the machine the way the harness puts a
    plate (table_bits.W[name] set, restored after) -- the same thing words(top=T) does, without its int64 detour, to
    stay under the memory budget."""
    tb = machine(); T = dl.print_top(D)
    head = tb.W["head"]; Tf = T.T.astype(np.float64); del T
    try:
        tb.W["head"] = (Tf, head[1], head[2])
        res = dict(words=int(tb.words(entry=D)))
        if parts: res["words_top_only"] = int(tb.words())
    finally:
        tb.W["head"] = head
    del Tf
    if parts: res["words_entry_only"] = int(tb.words(entry=D))
    return res

def store(D, e):
    """round one's stored form (develop_lib.store_lines without its lzma side-count): Q = D >> e_i zigzagged, range
    coded with the line as context; side: 12 + 768 + 5 bytes (shape, the 768 cut places, width) + the lens 3,072 bytes.
    Decoded back and checked equal."""
    z = np.empty(D.size, np.uint16)
    for r in range(0, D.shape[0], dl.CH):
        v = D[r:r + dl.CH].astype(np.int64); q = v >> e[None, :]
        assert np.array_equal(q << e[None, :], v)
        zz = dl.zigzag(q).ravel(); assert int(zz.max()) < 65536; z[r * 768:r * 768 + zz.size] = zz
    w = dl.nb(int(z.max())); line = np.tile(np.arange(768, dtype=np.uint16), D.shape[0])
    bq = dl.rc_encode(z, line, w, 768); zb = dl.rc_decode(bq, line, w, 768); del line, z
    ok = True
    for r in range(0, D.shape[0], dl.CH):
        back = dl.unzigzag(zb[r * 768:(r + dl.CH) * 768].astype(np.int64)).reshape(-1, 768) << e[None, :]
        ok &= bool(np.array_equal(back, D[r:r + dl.CH].astype(np.int64)))
    tot = len(bq) + 12 + 768 + 5 + LENS_BYTES
    return dict(stored_bytes=tot, MB=round(tot / 1e6, 3), bits_per_value=round(8 * tot / D.size, 4),
                roundtrip_ok=ok, coder_bytes=len(bq), width=w)

def record(row):
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    rows = [x for x in rows if x["name"] != row["name"]] + [row]
    json.dump(rows, open(OUT, "w"), indent=1)
    print("RESULT", json.dumps(row), flush=True)

def run(specs, parts=False, do_store=True, coder="rc"):
    tb = machine(); E = tb._WTE0
    for s in specs:
        t0 = time.time(); s = dict(s); name = s.pop("name"); pred = s.pop("predicted", None)
        D, e = develop(E, **s)
        row = dict(name=name, spec=s, predicted=pred)
        if do_store: row.update(store(D, e) if coder == "rc" else store2(D, e, 255))   # coded and decoded first
        row.update(judge(D, parts))
        row["grain_steps2"] = round(grain_steps2(D, E, e), 4)
        row["photons_per_value"] = round(float(dl.naf_photons(D[::7]).mean()), 3)
        row["secs"] = round(time.time() - t0, 1)
        record(row); del D

# ---------------------------------------------------------------- the screen phase (control for the judge's luck)
def develop_phase(E, c, seed, cap=12, how="nearest"):
    """nearest on a SHIFTED grid: line i's grid is moved by a phase o_i in [0, 2^e_i) (16-bit whole number r_i from a
    fixed seed, o_i = r_i x 2^e_i >> 16), D = Q(v - o_i) + o_i. Same grain power as nearest (a uniform 1/12 step^2),
    another realization of it -- a halftone screen laid at another phase. Stored: Q(v - o) on round one's grid + 768 phases."""
    E = np.ascontiguousarray(E, dtype=np.int32); e, p = exposure(c, cap); e = e.astype(np.int64)
    r16 = np.random.default_rng(seed).integers(0, 1 << 16, size=768, dtype=np.int64); o = (r16 << e) >> 16
    base, _ = develop(E - o[None, :].astype(np.int32), c, how, cap=cap)
    return base, e, o

def run_phase(cs, seeds, do_store=True, how="nearest", label="v0 screen phase"):
    tb = machine(); E = tb._WTE0
    for c in cs:
        for sd in seeds:
            t0 = time.time(); base, e, o = develop_phase(E, c, sd, how=how)
            row = dict(name=f"{label} c={c} seed={sd}", spec=dict(c=c, how=how + " on a shifted grid", seed=sd))
            if do_store:
                s = store(base, e); s["stored_bytes"] += 768 * 2; s["MB"] = round(s["stored_bytes"] / 1e6, 3)
                s["bits_per_value"] = round(8 * s["stored_bytes"] / base.size, 4); row.update(s)
            base += o[None, :].astype(np.int32); D = base   # in place: the decoded values (grid + phase)
            row.update(judge(D)); row["grain_steps2"] = round(grain_steps2(D, E, e), 4)
            row["secs"] = round(time.time() - t0, 1); record(row); del D, base

# ---------------------------------------------------------------- the counting coder (tablediffuse_rc.c)
_rc2 = ctypes.CDLL(os.path.join(HERE, "tablediffuse_rc.dylib"))
_rc2.rc2_encode.restype = ctypes.c_int64; _rc2.rc2_decode.restype = ctypes.c_int64

def rc2_encode(sym, ctx, nbits, nctx, nmax):
    sym = np.ascontiguousarray(sym, dtype=np.uint16); ctx = np.ascontiguousarray(ctx, dtype=np.uint16)
    assert len(sym) == len(ctx) and int(sym.max()) < (1 << nbits) and int(ctx.max()) < nctx
    cap = len(sym) * nbits // 8 + len(sym) // 16 + 65536
    while True:
        buf = np.empty(cap, np.uint8)
        n = _rc2.rc2_encode(sym.ctypes.data_as(_P), ctx.ctypes.data_as(_P), ctypes.c_int64(len(sym)), ctypes.c_int(nbits),
                            ctypes.c_int64(nctx), ctypes.c_int(nmax), buf.ctypes.data_as(_P), ctypes.c_int64(cap))
        if n <= cap: return buf[:n].tobytes()
        cap = n + 65536

def rc2_decode(data, ctx, nbits, nctx, nmax):
    ctx = np.ascontiguousarray(ctx, dtype=np.uint16); buf = np.frombuffer(data, np.uint8).copy()
    sym = np.empty(len(ctx), np.uint16)
    _rc2.rc2_decode(buf.ctypes.data_as(_P), ctypes.c_int64(len(buf)), ctx.ctypes.data_as(_P), ctypes.c_int64(len(ctx)),
                    ctypes.c_int(nbits), ctypes.c_int64(nctx), ctypes.c_int(nmax), sym.ctypes.data_as(_P))
    return sym

def line_floor_bytes(D, e):
    """reference only: the per-line order-0 Shannon floor of the stored whole numbers (each line its own histogram,
    the histograms themselves not counted). Uses logs -- it is a yardstick, never part of a codec."""
    tot = 0.0
    for i in range(768):
        q = D[:, i].astype(np.int64) >> int(e[i]); _, c = np.unique(q, return_counts=True)
        tot += float((c * (np.log2(c.sum()) - np.log2(c))).sum())
    return tot / 8

def store2(D, e, nmax=1023):
    """as store(), with the counting coder. Decoded back and checked equal."""
    z = np.empty(D.size, np.uint16)
    for r in range(0, D.shape[0], dl.CH):
        v = D[r:r + dl.CH].astype(np.int64); q = v >> e[None, :]
        assert np.array_equal(q << e[None, :], v)
        zz = dl.zigzag(q).ravel(); assert int(zz.max()) < 65536; z[r * 768:r * 768 + zz.size] = zz
    w = dl.nb(int(z.max())); line = np.tile(np.arange(768, dtype=np.uint16), D.shape[0])
    bq = rc2_encode(z, line, w, 768, nmax); zb = rc2_decode(bq, line, w, 768, nmax); del line
    ok = bool(np.array_equal(zb, z)); del z
    for r in range(0, D.shape[0], dl.CH):
        back = dl.unzigzag(zb[r * 768:(r + dl.CH) * 768].astype(np.int64)).reshape(-1, 768) << e[None, :]
        ok &= bool(np.array_equal(back, D[r:r + dl.CH].astype(np.int64)))
    tot = len(bq) + 12 + 768 + 5 + 2 + LENS_BYTES
    return dict(stored_bytes=tot, MB=round(tot / 1e6, 3), bits_per_value=round(8 * tot / D.size, 4),
                roundtrip_ok=ok, coder_bytes=len(bq), width=w, nmax=nmax)

# ---------------------------------------------------------------- the parity screen (tablediffuse_lattice.c)
_lat = ctypes.CDLL(os.path.join(HERE, "tablediffuse_lattice.dylib")); _lat.lat_develop.restype = ctypes.c_int64
_lat.lat_code_word.restype = ctypes.c_int64
KINDS = {"D8": 1, "E8": 2}

def groups(seed=7):
    return np.random.default_rng(seed).permutation(768).astype(np.int32)        # 96 groups of 8 lines

def lattice_develop(E, c, kind, gseed=7, phase_seed=None, cap=12):
    """-> (D int32 on round one's grid (+ phase), e, grp, cw, o). phase_seed: lay the screen at another phase (v0)."""
    E = np.ascontiguousarray(E, dtype=np.int32); e, _ = exposure(c, cap); e64 = e.astype(np.int64)
    o = np.zeros(768, np.int64)
    if phase_seed is not None:
        r16 = np.random.default_rng(phase_seed).integers(0, 1 << 16, size=768, dtype=np.int64); o = (r16 << e64) >> 16
        E = np.ascontiguousarray(E - o[None, :].astype(np.int32))
    grp = groups(gseed); n = E.shape[0]
    out = np.empty(E.shape, np.int32); cw = np.zeros((n, 96), np.uint8)
    assert _lat.lat_develop(E.ctypes.data_as(_P), ctypes.c_int64(n), e.ctypes.data_as(_P), grp.ctypes.data_as(_P),
                            ctypes.c_int(KINDS[kind]), out.ctypes.data_as(_P), cw.ctypes.data_as(_P)) == 0
    return out, e64, grp, cw, o

def code_word_bits():
    return np.array([[(int(_lat.lat_code_word(c)) >> j) & 1 for j in range(8)] for c in range(16)], np.int64)

def lattice_streams(D, e, grp, kind, cw=None):
    """the symbols the file carries, in group order (word after word, group after group, member after member)"""
    n = D.shape[0]; sym = np.empty(D.size, np.uint16); ctx = np.empty(D.size, np.uint16)
    lines = np.tile(grp.astype(np.uint16), 1)
    for r in range(0, n, dl.CH):
        K = (D[r:r + dl.CH].astype(np.int64) >> e[None, :])[:, grp].reshape(-1, 96, 8)
        assert np.array_equal((K.reshape(-1, 768) << e[grp][None, :]), D[r:r + dl.CH].astype(np.int64)[:, grp])
        m = K.copy(); cx = np.broadcast_to(lines.reshape(96, 8), K.shape).astype(np.int64).copy()
        if kind == "D8":
            assert not (K.sum(2) & 1).any()                                          # every group's sum is even
            rr = K[:, :, :7].sum(2) & 1; m[:, :, 7] = (K[:, :, 7] - rr) >> 1; cx[:, :, 7] += 768
        else:
            B = code_word_bits()[cw[r:r + dl.CH].astype(np.int64)]                   # (rows, 96, 8) low places
            assert np.array_equal(K & 1, B); m = (K - B) >> 1
        z = dl.zigzag(m).ravel(); assert int(z.max()) < 65536
        sym[r * 768:r * 768 + z.size] = z; ctx[r * 768:r * 768 + z.size] = cx.ravel()
    return sym, ctx

def lattice_rebuild(sym, n, e, grp, kind, cwd=None):
    D = np.empty((n, 768), np.int32)
    for r in range(0, n, dl.CH):
        m = dl.unzigzag(sym[r * 768:(r + dl.CH) * 768].astype(np.int64)).reshape(-1, 96, 8)
        if kind == "D8":
            K = m.copy(); rr = K[:, :, :7].sum(2) & 1; K[:, :, 7] = 2 * m[:, :, 7] + rr
        else:
            K = 2 * m + code_word_bits()[cwd[r:r + dl.CH].astype(np.int64)]
        Kl = np.empty((K.shape[0], 768), np.int64); Kl[:, grp] = K.reshape(-1, 768)
        D[r:r + dl.CH] = (Kl << e[None, :]).astype(np.int32)
    return D

def store_lattice(D, e, grp, kind, cw=None, nmax=255, phases=False):
    """coded with the counting coder, decoded back from the bytes, rebuilt, checked equal. Side: header 12, the 768 cut
    places, the 768 group lines (2 bytes each), widths 2, the lens 3,072 (+ 768 phases x 2 when a phase is used)."""
    n = D.shape[0]; sym, ctx = lattice_streams(D, e, grp, kind, cw)
    w = dl.nb(int(sym.max())); nctx = 1536 if kind == "D8" else 768
    bq = rc2_encode(sym, ctx, w, nctx, nmax); back = rc2_decode(bq, ctx, w, nctx, nmax)
    ok = bool(np.array_equal(back, sym)); del sym
    bc = b""; cwd = None
    if kind == "E8":
        cs = cw.ravel().astype(np.uint16); cc = np.tile(np.arange(96, dtype=np.uint16), n)
        bc = rc2_encode(cs, cc, 4, 96, nmax); cwd = rc2_decode(bc, cc, 4, 96, nmax).reshape(n, 96)
        ok &= bool(np.array_equal(cwd, cw)); del cc, cs
    Db = lattice_rebuild(back, n, e, grp, kind, cwd); ok &= bool(np.array_equal(Db, D)); del Db, back
    tot = len(bq) + len(bc) + 12 + 768 + 768 * 2 + 2 + LENS_BYTES + (768 * 2 if phases else 0)
    return dict(stored_bytes=tot, MB=round(tot / 1e6, 3), bits_per_value=round(8 * tot / D.size, 4), roundtrip_ok=ok,
                coder_bytes=len(bq), code_word_bytes=len(bc), width=w, nmax=nmax)

def run_lattice(specs):
    tb = machine(); E = tb._WTE0
    for s in specs:
        t0 = time.time(); s = dict(s); name = s.pop("name"); pred = s.pop("predicted", None)
        D, e, grp, cw, o = lattice_develop(E, s["c"], s["kind"], s.get("gseed", 7), s.get("phase_seed"))
        row = dict(name=name, spec=s, predicted=pred)
        row.update(store_lattice(D, e, grp, s["kind"], cw, phases=s.get("phase_seed") is not None))
        D += o[None, :].astype(np.int32)
        row.update(judge(D)); row["grain_steps2"] = round(grain_steps2(D, E, e), 4)
        row["secs"] = round(time.time() - t0, 1); record(row); del D, cw
