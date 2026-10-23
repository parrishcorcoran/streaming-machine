"""CHECKER for the PLANES family, reader side: a fresh process that sees ONLY the files check_planes_enc.py wrote.
Decodes each, adds the dark levels back, compares exactly with the harness tables (table_bits: E, T, plate(name)),
then asks the machine for right next word out of 30 with the decoded E and T.
    python3 check_planes_dec.py ET
    python3 check_planes_dec.py plates
"""
import sys, os, ctypes, struct, numpy as np
D = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/chk"
L = ctypes.CDLL(f"{D}/chk_planes_rc.dylib")
P, I, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
L.planes.argtypes = [i32, P, I, I, i32, i32, P, P, I, P]; L.planes.restype = I

def dec(path, dark):
    raw = open(path, "rb").read()
    R, C, nb, mode = struct.unpack("<IIBH", raw[:11]); p = 11
    off = np.frombuffer(raw[p:p + 2 * C], np.int16).astype(np.int32) if dark else np.zeros(C, np.int32)
    if dark: p += 2 * C
    data = np.frombuffer(raw[p:], np.uint8).copy(); del raw
    B = np.zeros((R, C), np.int32)
    L.planes(1, B.ctypes.data, R, C, nb, mode, None, data.ctypes.data, len(data), None)
    B += off[None, :]
    return B

if sys.argv[1] == "ET":
    sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); os.chdir("/Users/abundancemachine/streaming-machine")
    from table_bits import *
    Ed = dec(f"{D}/E.planes", True)
    print("decoded E == harness E:", bool(np.array_equal(Ed.astype(np.int64), E)), Ed.dtype, E.dtype, flush=True)
    Td = dec(f"{D}/T.planes", True)
    print("decoded T == harness T:", bool(np.array_equal(Td.astype(np.int64), T.astype(np.int64))), flush=True)
    print("words, unchanged machine:", words(), flush=True)
    print("words(entry=decoded E, top=decoded T):", words(entry=Ed.astype(np.int64), top=Td.astype(np.int16)), flush=True)
else:                     # plates: compare with the npz arrays table_bits serves (plate(name) = M[name + ".w"]), without
    M = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")   # loading the 1.5 GB machine
    plate = lambda nm: M[nm + ".w"]
    ok = True
    for l in range(12):
        for b in ("qkv", "proj", "fc", "mproj"):
            nm = f"{b}.{l}"; Bd = dec(f"{D}/{nm}.planes", False)
            e = bool(np.array_equal(Bd.astype(np.int64), plate(nm).astype(np.int64))); ok &= e
            if not e: print("MISMATCH", nm, flush=True)
    print("all 48 plates decoded equal to harness plates:", ok)
