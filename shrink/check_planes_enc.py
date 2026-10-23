"""CHECKER for the PLANES family's best claim (v7b on E, also T and the 48 plates with v9). Encoder side only.
Writes ONE file per table holding everything the reader needs: shape, number of places, mode, the 768 dark levels
(int16 each, E/T only) and the coded bytes. The stored size is the file size, nothing else.
Uses a fresh compile of planes_rc.c (scratchpad copy, same source hash) -- the coder is theirs, the bookkeeping is mine.
    python3 check_planes_enc.py ET       (E and T, v7b)
    python3 check_planes_enc.py plates   (48 plates, v9)
"""
import sys, os, ctypes, struct, numpy as np
D = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/chk"
L = ctypes.CDLL(f"{D}/chk_planes_rc.dylib")
P, I, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
L.planes.argtypes = [i32, P, I, I, i32, i32, P, P, I, P]; L.planes.restype = I
NPZ = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")

def enc(A, mode, path, dark):
    A = np.ascontiguousarray(A, np.int32); R, C = A.shape
    if dark:
        s = A[:32768].astype(np.int64).sum(0)                       # whole-number sum, rounded shift by 15
        off = ((s + (1 << 14)) >> 15)
        assert np.abs(off).max() < (1 << 15)
        off = off.astype(np.int16); F = np.ascontiguousarray(A - off.astype(np.int32)[None, :])
    else:
        off = np.zeros(0, np.int16); F = A
    nb = int(np.abs(F.astype(np.int64)).max()).bit_length()
    cap = F.size * 3 + (1 << 20); buf = np.zeros(cap, np.uint8)
    n = L.planes(0, F.ctypes.data, R, C, nb, mode, None, buf.ctypes.data, cap, None)
    assert n <= cap
    with open(path, "wb") as f:
        f.write(struct.pack("<IIBH", R, C, nb, mode)); f.write(off.tobytes()); f.write(buf[:n].tobytes())
    sz = os.path.getsize(path)
    print(f"{os.path.basename(path):22s} payload {n:,} B, file {sz:,} B = {sz * 8 / A.size:.4f} bits/value = {sz / 1e6:.3f} MB", flush=True)
    return sz, A.size

if sys.argv[1] == "ET":
    MODE = 1 | 2 | 8 | 32 | 1024                                    # HIGH LINE SIGNC WEXP REFLINE
    enc(NPZ["wte"], MODE, f"{D}/E.planes", True)
    enc(NPZ["head.w"].T, MODE, f"{D}/T.planes", True)
else:
    MODE = 1 | 32 | 64                                              # HIGH WEXP LEXP (v9)
    tot = 0; nv = 0
    for l in range(12):
        for b in ("qkv", "proj", "fc", "mproj"):
            sz, n = enc(NPZ[f"{b}.{l}.w"], MODE, f"{D}/{b}.{l}.planes", False); tot += sz; nv += n
    print(f"ALL 48 plates: {tot:,} B = {tot * 8 / nv:.4f} bits/value = {tot / 1e6:.3f} MB over {nv:,} values")
