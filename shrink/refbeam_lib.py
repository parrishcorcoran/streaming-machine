"""REFERENCE BEAM family: shared helpers. Whole numbers only in every coding and decoding step.

  load(name)            the stored table as whole numbers: "E" (entry, at 2^16) or "T" (top, int16), (50257, 768)
  rc(A, ctx, mant)      adaptive binary range coder (refbeam_rc.c), values written as photons: top place, direction,
                        places below. ctx: "none" (one set of odds) | "line" (odds per line) | an array of context ids
                        (one per value, same shape as A). mant=True: the 3 places under the top one also get the line's
                        own odds. Returns (bytes, roundtrip_ok) -- decoded and compared with np.array_equal.
  lz(A, order)          sign-magnitude bit pages (one page per place, a sign page), packed 8 to a byte, lzma.
                        order "line" (each line's 50,257 words side by side) or "word" (each word's 768 lines side by
                        side). Returns (bytes, roundtrip_ok).
  small(v)              side information: a short vector of whole numbers coded for real (range coder, no context);
                        returns its bytes (and checks it decodes).
  photons(A, R)         photons (1s) per value and single-wave photons, in the word A and in the residual R, and per
                        word the share of photons the reference removed.
  report(...)           prints a line and appends it to refbeam_results.json
"""
import os, json, lzma, ctypes, numpy as np

D = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/rb/"
HERE = "/Users/abundancemachine/streaming-machine/shrink/"
OUT = HERE + "refbeam_results.json"
_lib = ctypes.CDLL(HERE + "refbeam_rc.dylib")
_P = ctypes.c_void_p
_lib.rb_encode.restype = ctypes.c_int64
_lib.rb_encode.argtypes = [_P, ctypes.c_int64, _P, ctypes.c_int64, ctypes.c_uint32, ctypes.c_int, _P]
_lib.rb_decode.restype = ctypes.c_int64
_lib.rb_decode.argtypes = [_P, ctypes.c_int64, ctypes.c_int64, _P, ctypes.c_int64, ctypes.c_uint32, ctypes.c_int, _P]
N_WORDS, N_LINES = 50257, 768

def load(name):
    """E or T as whole numbers. Read from the scratch copy if it is there; otherwise taken from the harness (table_bits)
    and the scratch copy written (E as int32: its values fit in 18 bits)."""
    f = D + name + ".npy"
    if not os.path.exists(f):
        import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
        import table_bits as tb
        os.makedirs(D, exist_ok=True)
        np.save(D + "E.npy", tb.E.astype(np.int32)); np.save(D + "T.npy", tb.T)
    return np.load(f).astype(np.int64)

def _ptr(a): return a.ctypes.data_as(ctypes.c_void_p)

def rc(A, ctx="line", mant=False, check=True, give_back=False):
    A = np.ascontiguousarray(A, dtype=np.int64); flat = A.ravel(); n = flat.size
    assert np.abs(flat).max() < (1 << 31)
    if isinstance(ctx, str):
        carr = None; cdiv, cmod = (1, A.shape[-1]) if ctx == "line" else (1, 1)
    else:
        carr = np.ascontiguousarray(ctx, dtype=np.uint32).ravel(); assert carr.size == n
        cdiv, cmod = 1, int(carr.max()) + 1
    out = np.zeros(n * 5 + 1024, np.uint8)
    nb = _lib.rb_encode(_ptr(flat), n, _ptr(carr) if carr is not None else None, cdiv, cmod, int(mant), _ptr(out))
    buf = out[:nb].copy(); del out
    ok = None
    if check:
        back = np.zeros(n, np.int64)
        _lib.rb_decode(_ptr(buf), nb, n, _ptr(carr) if carr is not None else None, cdiv, cmod, int(mant), _ptr(back))
        ok = bool(np.array_equal(back.reshape(A.shape), A))
        if give_back: return int(nb), ok, back.reshape(A.shape)
        del back
    return int(nb), ok

def _bitlen_max(m):
    m = int(m); return m.bit_length()

def lz(A, order="line", check=True, preset=6):
    A = np.asarray(A, dtype=np.int64)
    X = np.ascontiguousarray(A.T) if order == "line" else np.ascontiguousarray(A)
    flat = X.ravel(); n = flat.size; mag = np.abs(flat); P = _bitlen_max(mag.max())
    pages = [np.packbits(flat < 0)] + [np.packbits(((mag >> p) & 1).astype(np.uint8)) for p in range(P - 1, -1, -1)]
    data = b"".join(p.tobytes() for p in pages); del pages
    comp = lzma.compress(data, preset=preset); del data
    header = 8                                        # n_words, n_lines, number of places: 8 bytes
    ok = None
    if check:
        raw = np.frombuffer(lzma.decompress(comp), np.uint8); pb = (n + 7) // 8
        neg = np.unpackbits(raw[:pb])[:n].astype(bool); m = np.zeros(n, np.int64)
        for i, p in enumerate(range(P - 1, -1, -1)):
            m |= np.unpackbits(raw[(i + 1) * pb:(i + 2) * pb])[:n].astype(np.int64) << p
        back = np.where(neg, -m, m).reshape(X.shape)
        if order == "line": back = back.T
        ok = bool(np.array_equal(back, A)); del back, m, neg, raw
    return len(comp) + header, ok

def small(v):
    """side information, coded for real: returns bytes (range coder, one set of odds) + 8 bytes of length/header"""
    v = np.asarray(v, dtype=np.int64).ravel()
    nb, ok = rc(v.reshape(1, -1), ctx="none"); assert ok
    return nb + 8

_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
def naf(A): A = np.abs(np.asarray(A).astype(np.int64)); return ones(A ^ (3 * A))

def photons(A, R):
    oa, orr = ones(A), ones(R)
    pw, pr = oa.sum(1), orr.sum(1)
    removed = 1 - pr / np.maximum(pw, 1)               # analysis only (a share), not part of any codec
    d = {"ones_word": round(float(oa.mean()), 4), "ones_resid": round(float(orr.mean()), 4)}
    del oa, orr
    na, nr = naf(A), naf(R)
    d.update({"naf_word": round(float(na.mean()), 4), "naf_resid": round(float(nr.mean()), 4),
              "per_word_removed_pct": {"mean": round(100 * float(removed.mean()), 2),
                                       "p10": round(100 * float(np.percentile(removed, 10)), 2),
                                       "p50": round(100 * float(np.percentile(removed, 50)), 2),
                                       "p90": round(100 * float(np.percentile(removed, 90)), 2),
                                       "max": round(100 * float(removed.max()), 2),
                                       "min": round(100 * float(removed.min()), 2),
                                       "words_over_half": int((removed > 0.5).sum())}})
    return d

def report(version, table, what, coded_bytes, side_bytes, nvals, ok, extra=None):
    tot = coded_bytes + side_bytes
    r = {"version": version, "table": table, "what": what, "coded_bytes": int(coded_bytes), "side_bytes": int(side_bytes),
         "total_MB": round(tot / 1e6, 3), "bits_per_value": round(8 * tot / nvals, 4), "roundtrip_ok": ok}
    if extra: r.update(extra)
    print(f"  {version:4s} {table:2s} {what:62s} {r['bits_per_value']:8.4f} b/v {r['total_MB']:8.3f} MB  ok={ok}"
          + (f"  {extra}" if extra else ""), flush=True)
    allr = json.load(open(OUT)) if os.path.exists(OUT) else []
    allr.append(r); json.dump(allr, open(OUT, "w"), indent=1)
    return r
