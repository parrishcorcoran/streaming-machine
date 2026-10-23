"""CHECKER for the REFERENCE BEAM family's best claim (v1: each line's most common value stored once, every word stores
R = A - ref[line], adaptive range coder with odds per line; E without, T with the 3-places-under-the-top odds).
Writer side only. Writes ONE file per table holding everything the reader needs: shape, mode flags, the two stream
lengths, the 768 reference values (coded with the same coder, one set of odds) and the coded residual.
The stored size is the file size, nothing else.
The reference is recomputed here with my own code (sort each line, longest run, ties to the smaller value).
Coder: a fresh compile of refbeam_rc.c (scratchpad copy) -- the coder is theirs, the bookkeeping is mine.
Also prints a cross-check: the same photon writing (top place, direction, 3 places under the top, deeper places) with
the SAME contexts but counted once over the whole table (two-pass counts, odds not stored) -- the information in the
stream as the coder sees it. An adaptive coder should land just above it.
    python3 check_refbeam_enc.py
"""
import os, struct, ctypes, numpy as np
D = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/chk"
L = ctypes.CDLL(f"{D}/chk_refbeam_rc.dylib")
P = ctypes.c_void_p
L.rb_encode.restype = ctypes.c_int64
L.rb_encode.argtypes = [P, ctypes.c_int64, P, ctypes.c_int64, ctypes.c_uint32, ctypes.c_int, P]
NPZ = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")

def mode_per_line(A):
    """most common value of each line; ties go to the smallest value (sorted run lengths, whole numbers)"""
    out = np.zeros(A.shape[1], np.int64)
    for j in range(A.shape[1]):
        s = np.sort(A[:, j]); edge = np.flatnonzero(np.diff(s)) + 1
        starts = np.concatenate([[0], edge]); lens = np.diff(np.concatenate([starts, [s.size]]))
        out[j] = s[starts[np.argmax(lens)]]          # argmax gives the first (smallest value) of the longest runs
    return out

def encode(v, cmod, mant):
    v = np.ascontiguousarray(v, np.int64).ravel()
    buf = np.zeros(v.size * 5 + 1024, np.uint8)
    nb = L.rb_encode(v.ctypes.data, v.size, None, 1, cmod, mant, buf.ctypes.data)
    return buf[:nb].tobytes()

def bitlen(m):
    b = np.zeros(m.shape, np.int64); x = m.copy()
    for s in (16, 8, 4, 2, 1):
        h = (x >> s) > 0; b += h * s; x = np.where(h, x >> s, x)
    return b + (x > 0)

def H(counts0, counts1):
    """bits to say a yes/no answer, counted once (two-pass), summed over contexts"""
    n = counts0 + counts1; t = 0.0
    for c in (counts0, counts1):
        m = c > 0; t += float((c[m] * np.log2(n[m] / c[m])).sum())     # analysis only, not part of any codec
    return t

def static_bits(R, mant):
    """the coder's own writing and contexts, counted once over the whole table (a cross-check, not a codec).
    Counted in slices of 4096 words to stay small in memory."""
    nl = R.shape[1]; ncm = nl if mant else 1
    K = np.zeros((2, nl * 32), np.int64); S = np.zeros((2, nl * 64), np.int64)
    MT = np.zeros((2, ncm * 64 * 8), np.int64); ML = np.zeros((2, 4096), np.int64)
    def add(T, c, bit):
        T[0] += np.bincount(c[bit == 0], minlength=T.shape[1]); T[1] += np.bincount(c[bit == 1], minlength=T.shape[1])
    for r0 in range(0, R.shape[0], 4096):
        Rs = R[r0:r0 + 4096]
        line = np.broadcast_to(np.arange(nl), Rs.shape).ravel(); r = Rs.ravel()
        m = np.abs(r); k = bitlen(m)
        node = np.ones(r.size, np.int64)              # top place: 5 answers down the tree, odds per line
        for b in range(4, -1, -1):
            bit = (k >> b) & 1; add(K, line * 32 + node, bit); node = node * 2 + bit
        lit = k > 0                                   # direction: odds per (line, top place)
        add(S, (line * 64 + k)[lit], (r < 0)[lit].astype(np.int64))
        pre = np.ones(r.size, np.int64); cm = line if mant else np.zeros_like(line)
        for t in range(int(k.max())):                 # places under the top, top first
            j = k - 2 - t; on = j >= 0
            if not on.any(): break
            bit = (m >> np.maximum(j, 0)) & 1
            if t < 3:
                add(MT, ((cm * 64 + k) * 8 + pre)[on], bit[on]); pre = np.where(on, pre * 2 + bit, pre)
            else:
                add(ML, (k * 64 + j)[on], bit[on])
    return sum(H(T[0], T[1]) for T in (K, S, MT, ML))

for name, mant in (("E", 0), ("T", 1)):
    A = (NPZ["wte"] if name == "E" else np.ascontiguousarray(NPZ["head.w"].T)).astype(np.int64)
    Rn, C = A.shape
    ref = mode_per_line(A)
    Rr = A - ref[None, :]
    b_ref = encode(ref, 1, 0)                         # the 768 reference values, one set of odds
    b_main = encode(Rr, C, mant)                      # odds per line (context = place in the word), +mant for T
    path = f"{D}/{name}.refbeam"
    with open(path, "wb") as f:
        f.write(struct.pack("<IIBBII", Rn, C, mant, 1, len(b_ref), len(b_main))); f.write(b_ref); f.write(b_main)
    sz = os.path.getsize(path)
    print(f"{name}: ref {len(b_ref):,} B + residual {len(b_main):,} B + header 18 B = file {sz:,} B "
          f"= {sz * 8 / A.size:.4f} bits/value = {sz / 1e6:.3f} MB", flush=True)
    sb = static_bits(Rr, mant)
    print(f"   cross-check, same writing and contexts counted once (odds not stored): {sb / A.size:.4f} bits/value", flush=True)
    sb0 = static_bits(A, mant)
    print(f"   same, NO reference: {sb0 / A.size:.4f} bits/value", flush=True)
    del A, Rr
