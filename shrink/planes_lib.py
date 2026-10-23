"""PLANES family, shared pieces: load a table, run the bit-plane coder (planes_rc.c, whole numbers only) both ways,
check the decode is exact, count every stored bit (coded bytes + side information), print per-place costs.

Tables are the arrays table_bits.py serves (E = M["wte"], T = M["head.w"].T, plate(name) = M[name + ".w"]), read
straight from the npz so the 1.5 GB machine is not loaded (fax_check.py in this folder confirmed they are identical).
"""
import os, ctypes, subprocess, json, time, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = "/Users/abundancemachine/streaming-machine/base2_gpt2.npz"
OUTJ = os.path.join(HERE, "planes_results.json")

M_HIGH, M_LINE, M_NEIGH, M_SIGNC, M_SIGNN, M_WEXP, M_LEXP, M_SKIP, M_SIGNLIT, M_SIGNTOP, M_REFLINE = \
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

def load(which):
    M = np.load(NPZ)
    if which == "E": return np.ascontiguousarray(M["wte"]).astype(np.int32)                 # whole numbers at 2^16
    if which == "T": return np.ascontiguousarray(M["head.w"].T).astype(np.int32)
    return np.ascontiguousarray(M[which + ".w"]).astype(np.int32)                          # a plate, (lines in, neurons out)

_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = os.path.join(HERE, "planes_rc.c"), os.path.join(HERE, "planes_rc.dylib")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O2", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so)
        P, I, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.planes.argtypes = [i32, P, I, I, i32, i32, P, P, I, P]; _lib.planes.restype = I
        _lib.naf_planes.argtypes = [i32, P, I, I, i32, i32, P, I, P]; _lib.naf_planes.restype = I
    return _lib

def nplanes(A): return int(np.abs(A).max()).bit_length()

def blocktops(A):
    """per 64 x 64 block (64 words x 64 lines, fixed tiling in table order): how many places can be lit = bit length
    of the block's largest magnitude. Whole numbers: OR of the magnitudes, then its bit length."""
    R, C = A.shape; a = np.abs(A).astype(np.int64); BR, BC = (R + 63) // 64, (C + 63) // 64
    pad = np.zeros((BR * 64, BC * 64), np.int64); pad[:R, :C] = a
    orr = np.bitwise_or.reduce(np.bitwise_or.reduce(pad.reshape(BR, 64, BC, 64), axis=3), axis=1)
    bl = np.zeros(orr.shape, np.uint8)
    x = orr.copy()
    while x.any(): bl += (x > 0); x >>= 1
    return np.ascontiguousarray(bl)

def code(A, mode, naf=False, skip=None):
    """encode then decode. Returns (nbytes, marks, decoded array, nb)."""
    L = lib(); A = np.ascontiguousarray(A, np.int32); R, C = A.shape
    nb = nplanes(A) + (1 if naf else 0)
    if mode & M_SIGNTOP: nb = int(np.where(A < 0, -A.astype(np.int64) - 1, A).max()).bit_length()
    cap = A.size * 3 + (1 << 20)
    buf = np.zeros(cap, np.uint8); mark = np.zeros(nb + 1, np.int64)
    bt = None if skip is None else np.ascontiguousarray(skip, np.uint8)
    if naf: n = L.naf_planes(0, A.ctypes.data, R, C, nb, mode, buf.ctypes.data, cap, mark.ctypes.data)
    else:   n = L.planes(0, A.ctypes.data, R, C, nb, mode, None if bt is None else bt.ctypes.data, buf.ctypes.data, cap, mark.ctypes.data)
    assert n <= cap
    data = buf[:n].copy(); del buf
    B = np.zeros_like(A); m2 = np.zeros(nb + 1, np.int64)
    if naf: L.naf_planes(1, B.ctypes.data, R, C, nb, mode, data.ctypes.data, n, m2.ctypes.data)
    else:   L.planes(1, B.ctypes.data, R, C, nb, mode, None if bt is None else bt.ctypes.data, data.ctypes.data, n, m2.ctypes.data)
    return n, mark, B, nb, data

def per_place(mark, nb, nvals, signtop=False, signlast=True):
    """bits per value spent on each place, top place first; the sign plane where it was coded"""
    out = []; prev = 0
    if signtop: out.append(("sign", (mark[nb] - prev) * 8 / nvals)); prev = mark[nb]
    for k in range(nb - 1, -1, -1):
        out.append((str(k), (mark[k] - prev) * 8 / nvals)); prev = mark[k]
    if signlast and not signtop: out.append(("sign", (mark[nb] - prev) * 8 / nvals))
    return out

def run(label, table, A, mode, naf=False, skip=False, extra_side=0, offset=None, note=""):
    """code A, decode, check exact (against A + offset if an offset was taken off), report and store the result."""
    t0 = time.time(); R, C = A.shape
    bt = blocktops(A) if skip else None
    n, mark, B, nb, data = code(A, mode, naf=naf, skip=bt)
    ok = bool(np.array_equal(B, A)); del B
    side = 2 * 32 + 5 + 16                                    # shape, number of places, mode
    if bt is not None: side += bt.size * 5                    # each block's lit-place count, 5 bits raw
    side += extra_side
    tot = n * 8 + side; bpv = tot / A.size
    pp = per_place(mark, nb, A.size, signtop=bool(mode & M_SIGNTOP), signlast=not (mode & M_SIGNLIT) and not naf)
    print(f"  {label:58s} {table:2s} {bpv:7.4f} bits/value  {tot / 8e6:7.3f} MB  side {side:,} bits  roundtrip {ok}  ({time.time() - t0:.0f} s)", flush=True)
    print("     per place (top first): " + " ".join(f"{k}:{v:.3f}" for k, v in pp), flush=True)
    res = dict(name=label, table=table, mode=mode, naf=naf, skip=skip, bits_per_value=round(bpv, 4), total_MB=round(tot / 8e6, 3),
               coded_bytes=int(n), side_bits=int(side), roundtrip_ok=ok, per_place={k: round(v, 4) for k, v in pp}, note=note)
    try: allr = json.load(open(OUTJ))
    except Exception: allr = []
    allr = [r for r in allr if not (r["name"] == label and r["table"] == table)] + [res]
    json.dump(allr, open(OUTJ, "w"), indent=1)
    return res, data
