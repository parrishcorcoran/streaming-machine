"""CHECKER for the REFERENCE BEAM family's pair statement: "with E stored, T costs 0.074 b/v (FAX v7), so the pair takes
about 72.6 MB". Writes ONE file with everything the reader needs for T beyond E: the 768 gains (int32 raw, 4 bytes
each), the stream lengths and the coded streams. Then a reader step that uses ONLY that file and E decoded from
E.refbeam (check_refbeam_enc.py's file), rebuilds T and compares with the npz top table exactly.
FAX v7's predict/code/decode are imported (the codec is theirs); the coder is a fresh compile of fax_rc.c in the
scratchpad; the bookkeeping is mine. Nothing in shrink/ is rewritten.
    python3 check_refbeam_pair.py
"""
import sys, os, struct, ctypes, numpy as np
D = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/chk"
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import fax_common
lib = ctypes.CDLL(f"{D}/chk_fax_rc.dylib"); P_ = ctypes.c_void_p; I = ctypes.c_int64
lib.fax_encode.argtypes = [P_, P_, I, ctypes.c_uint32, ctypes.c_int, I, P_]; lib.fax_encode.restype = I
lib.fax_decode.argtypes = [P_, I, P_, I, ctypes.c_uint32, ctypes.c_int, I, P_]; lib.fax_decode.restype = I
fax_common._lib = lib
import fax_v7_top_from_entry as v7
L = ctypes.CDLL(f"{D}/chk_refbeam_rc.dylib")
L.rb_decode.restype = ctypes.c_int64
L.rb_decode.argtypes = [P_, ctypes.c_int64, ctypes.c_int64, P_, ctypes.c_int64, ctypes.c_uint32, ctypes.c_int, P_]

def rb_dec(b, n, cmod, mant):
    a = np.frombuffer(b, np.uint8).copy(); v = np.zeros(n, np.int64)
    L.rb_decode(a.ctypes.data, a.size, n, None, 1, cmod, mant, v.ctypes.data); return v

def read_E():
    raw = open(f"{D}/E.refbeam", "rb").read()
    Rn, C, mant, ctxline, lr, lm = struct.unpack("<IIBBII", raw[:18]); p = 18
    ref = rb_dec(raw[p:p + lr], C, 1, 0); p += lr
    return rb_dec(raw[p:p + lm], Rn * C, C, mant).reshape(Rn, C) + ref[None, :]

NPZ = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
T = np.ascontiguousarray(NPZ["head.w"].T).astype(np.int64)
assert bool(np.array_equal(read_E(), NPZ["wte"])), "E.refbeam does not decode to E"

# ---- writer: has E (decoded from the file), T, and the gains ----
g = np.load("/Users/abundancemachine/streaming-machine/shrink/fax_gain24.npy").astype(np.int64)
assert np.abs(g).max() < (1 << 31)
E = read_E(); Pp, dc, side = v7.predict(E, g); del E
b1, b2, rounds, mant = v7.code(T, Pp, dc, side); del Pp, dc, side
path = f"{D}/T_given_E.fax7"
with open(path, "wb") as f:
    f.write(struct.pack("<I", len(rounds))); f.write(g.astype(np.int32).tobytes())
    for b in [b1, b2] + rounds + [mant]: f.write(struct.pack("<I", len(b))); f.write(b)
sz = os.path.getsize(path)
print(f"T given E: file {sz:,} B = {sz * 8 / T.size:.4f} b/v = {sz / 1e6:.3f} MB "
      f"(lit? {len(b1):,} B, up/hole {len(b2):,} B, {len(rounds)} length pages, raw bits {len(mant):,} B)", flush=True)
del b1, b2, rounds, mant

# ---- reader: only the file and E from E.refbeam ----
raw = open(path, "rb").read(); p = 0
nr = struct.unpack("<I", raw[:4])[0]; p = 4
g2 = np.frombuffer(raw[p:p + 768 * 4], np.int32).astype(np.int64); p += 768 * 4
parts = []
for _ in range(2 + nr + 1):
    n = struct.unpack("<I", raw[p:p + 4])[0]; p += 4; parts.append(raw[p:p + n]); p += n
E = read_E(); Pp, dc, side = v7.predict(E, g2); del E
Td = v7.decode(parts[0], parts[1], parts[2:2 + nr], parts[-1], Pp, dc, side)
print("decoded T (from E + this file) == npz T:", bool(np.array_equal(Td, T)), flush=True)
e_sz = os.path.getsize(f"{D}/E.refbeam")
print(f"pair: E {e_sz:,} B + T|E {sz:,} B = {(e_sz + sz) / 1e6:.3f} MB "
      f"(T alone with refbeam v1 would be {os.path.getsize(f'{D}/T.refbeam') / 1e6:.3f} MB)")
