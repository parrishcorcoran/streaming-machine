"""FAX family, shared pieces. Each place (power of two) of a table is a black-and-white page: lit pixel = a 1 there.
Store only where the lit pixels are, the way a fax machine stores a page. Whole numbers everywhere in coding and
decoding: shifts, adds, subtracts, ANDs, counts, compares. Real bytes come out of real coders (Rice codes written
here, Python's lzma, and the adaptive range coder in fax_rc.c). Every lossless system decodes back and is checked.

The arrays are the same ones table_bits.py serves (E = M["wte"], T = M["head.w"].T, plate = M[name + ".w"]), read
straight from the npz so the 1.5 GB machine is not loaded when only the stored numbers are needed (fax_check.py
confirms they are identical).
"""
import os, ctypes, subprocess, lzma, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = "/Users/abundancemachine/streaming-machine/base2_gpt2.npz"

def load(which):
    M = np.load(NPZ)
    if which == "E": return M["wte"].astype(np.int32)                 # (50257, 768) whole numbers at 2^16
    if which == "T": return np.ascontiguousarray(M["head.w"].T).astype(np.int32)   # (50257, 768), all at 2^-12
    return M[which + ".w"].astype(np.int32)                           # a plate, (lines in, neurons out)

def nplanes(A): return int(np.abs(A).max()).bit_length()

# ---------------- single waves (up / hole), no two lit side by side ----------------
def naf_masks(A):
    """up and hole masks of |A| in the single-wave writing: |A| = up - hole, no two lit places side by side"""
    a = np.abs(A).astype(np.int32 if int(np.abs(A).max()) < (1 << 29) else np.int64); a3 = 3 * a
    return (a3 & ~a) >> 1, (a & ~a3) >> 1

# ---------------- Rice code of the gaps between lit pixels (vectorised, whole numbers) ----------------
def rice_k(n_lit, n_pix):
    """fixed rule, no search: k = floor(log2(dark pixels per lit pixel)), from the two counts alone"""
    if n_lit == 0: return 0
    m = (n_pix - n_lit) // n_lit
    return max(int(m).bit_length() - 1, 0)

def gaps_encode(lit):
    """lit: bool 1-D scan of a page. Returns (bytes, n_lit, k, side_bits). Each lit pixel stores the number of dark
    pixels before it: q = gap >> k in unary (q ones, one zero), then the k low bits."""
    pos = np.flatnonzero(lit).astype(np.int32); n = len(pos)
    if n == 0: return b"", 0, 0, 32 + 5
    g = np.diff(pos, prepend=np.int32(-1)) - np.int32(1); del pos
    k = rice_k(n, len(lit))
    q = g >> k
    u = np.ones(int(q.sum(dtype=np.int64)) + n, np.uint8); q += 1; u[np.cumsum(q, dtype=np.int64) - 1] = 0; del q
    ub = np.packbits(u).tobytes(); del u
    rb = b""
    if k:
        r = ((g[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
        rb = np.packbits(r.ravel()).tobytes()
    return (len(ub).to_bytes(4, "little") + ub + rb), n, k, 32 + 5 + 32   # side: count, k, split point

def gaps_decode(buf, n, k, npix):
    lit = np.zeros(npix, bool)
    if n == 0: return lit
    lu = int.from_bytes(buf[:4], "little"); ub = np.frombuffer(buf[4:4 + lu], np.uint8); rb = np.frombuffer(buf[4 + lu:], np.uint8)
    z = np.flatnonzero(np.unpackbits(ub) == 0)[:n].astype(np.int64)
    g = (np.diff(z, prepend=-1) - 1) << k; del z
    if k:
        r = np.unpackbits(rb)[:n * k].reshape(n, k)
        for j in range(k): g |= r[:, j].astype(np.int64) << (k - 1 - j)
    g += 1; lit[np.cumsum(g) - 1] = True
    return lit

def raw_bits(b):  return np.packbits(np.asarray(b, np.uint8)).tobytes()
def unraw_bits(buf, n): return np.unpackbits(np.frombuffer(buf, np.uint8))[:n].astype(bool)

# ---------------- adaptive range coder (C, whole numbers) ----------------
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = os.path.join(HERE, "fax_rc.c"), os.path.join(HERE, "fax_rc.dylib")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O2", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so)
        P = ctypes.c_void_p; I = ctypes.c_int64
        _lib.fax_encode.argtypes = [P, P, I, ctypes.c_uint32, ctypes.c_int, I, P]; _lib.fax_encode.restype = I
        _lib.fax_decode.argtypes = [P, I, P, I, ctypes.c_uint32, ctypes.c_int, I, P]; _lib.fax_decode.restype = I
    return _lib

def rc_encode(bits, ctx, nctx, tbits=0, width=1):
    bits = np.ascontiguousarray(bits, np.uint8); ctx = np.ascontiguousarray(ctx, np.uint32)
    out = np.empty(len(bits) // 4 + 65536, np.uint8)                    # room: up to 2 bits a pixel
    m = lib().fax_encode(bits.ctypes.data, ctx.ctypes.data, len(bits), nctx, tbits, width, out.ctypes.data)
    assert m <= len(out)
    return out[:m].tobytes()

def rc_decode(buf, ctx, nctx, tbits=0, width=1):
    ctx = np.ascontiguousarray(ctx, np.uint32); a = np.frombuffer(buf, np.uint8)
    bits = np.zeros(len(ctx), np.uint8)
    lib().fax_decode(a.ctypes.data, len(a), ctx.ctypes.data, len(ctx), nctx, tbits, width, bits.ctypes.data)
    return bits

def mb(nbits): return nbits / 8 / 1e6
def report(name, table, nbytes, side_bits, nvals, ok):
    tot = nbytes * 8 + side_bits
    print(f"  {name:44s} {table:4s} {tot / nvals:7.3f} bits/value  {mb(tot):8.2f} MB  (side info {side_bits:,} bits)  roundtrip {ok}", flush=True)
    return dict(name=name, table=table, bits_per_value=round(tot / nvals, 4), total_MB=round(mb(tot), 3), roundtrip_ok=ok)
