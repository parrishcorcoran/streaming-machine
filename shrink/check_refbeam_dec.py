"""CHECKER for the REFERENCE BEAM family, reader side: a fresh process that sees ONLY the files check_refbeam_enc.py
wrote. Decodes the 768 reference values and the residual, adds the reference back, compares exactly with the harness
tables (table_bits: E, T), then asks the machine for right next word out of 30 with the decoded E and T.
    python3 check_refbeam_dec.py
"""
import sys, os, struct, ctypes, numpy as np
D = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/chk"
L = ctypes.CDLL(f"{D}/chk_refbeam_rc.dylib")
P = ctypes.c_void_p
L.rb_decode.restype = ctypes.c_int64
L.rb_decode.argtypes = [P, ctypes.c_int64, ctypes.c_int64, P, ctypes.c_int64, ctypes.c_uint32, ctypes.c_int, P]

def decode(b, n, cmod, mant):
    a = np.frombuffer(b, np.uint8).copy(); v = np.zeros(n, np.int64)
    L.rb_decode(a.ctypes.data, a.size, n, None, 1, cmod, mant, v.ctypes.data)
    return v

def read(path):
    raw = open(path, "rb").read()
    Rn, C, mant, ctxline, lr, lm = struct.unpack("<IIBBII", raw[:18]); p = 18
    ref = decode(raw[p:p + lr], C, 1, 0); p += lr
    R = decode(raw[p:p + lm], Rn * C, C if ctxline else 1, mant).reshape(Rn, C)
    return R + ref[None, :], len(raw)

sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); os.chdir("/Users/abundancemachine/streaming-machine")
from table_bits import *
Ed, sE = read(f"{D}/E.refbeam")
print(f"decoded E == harness E: {bool(np.array_equal(Ed, E))}   file {sE:,} B = {sE * 8 / E.size:.4f} b/v", flush=True)
Td, sT = read(f"{D}/T.refbeam")
print(f"decoded T == harness T: {bool(np.array_equal(Td, T.astype(np.int64)))}   file {sT:,} B = {sT * 8 / T.size:.4f} b/v", flush=True)
print("words, unchanged machine:", words(), flush=True)
print("words(entry=decoded E, top=decoded T):", words(entry=Ed, top=Td.astype(np.int16)), flush=True)
