"""CHECKER for the develop family's extremes (whole machine, T printed from E): E as signs only (each value at the
word's mean loudness), and E as the white plate plus each word's 2 loudest departures as one photon each. Own code.
    python3 check_develop_v9_extremes.py      (run from ~/streaming-machine)
"""
import sys, json, struct, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
import table_bits as tb
p = "/Users/abundancemachine/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); a, b = h["ln_f.weight"]["data_offsets"]
f.seek(8 + n + a); G = np.round(np.frombuffer(f.read(b - a), "<f4").astype(np.float64) * 2 ** 24).astype(np.int64)
print("machine unchanged:", tb.words(), flush=True)
tb.E = None; E = tb._WTE0

def print_exact(D):
    out = np.empty(D.shape, np.int32); den = 768 * (1 << 28)
    for r in range(0, D.shape[0], 4096):
        s = D[r:r + 4096].astype(np.int64) * G[None, :]; c = 768 * s - s.sum(1, keepdims=True)
        out[r:r + 4096] = (c + den // 2) // den
    return out

# signs only
mag = (np.abs(E).sum(1) + 384) // 768
D = np.where(E < 0, -mag[:, None], mag[:, None]).astype(np.int32)
print("E signs only, T printed:", tb.words(entry=D, top=print_exact(D)), flush=True)
print("E signs only alone:", tb.words(entry=D), flush=True)
del D
# white plate + 2 loudest departures, one photon each (top digit of the single-wave writing: 2^(bitlen(3a) - 2))
wp = (E.sum(0) + 50257 // 2) // 50257
D = np.empty(E.shape, np.int32)
for r in range(0, E.shape[0], 4096):
    d = E[r:r + 4096] - wp[None, :]; ad = np.abs(d)
    idx = np.argsort(-ad, axis=1, kind="stable")[:, :2]
    av = np.take_along_axis(ad, idx, 1); sg = np.take_along_axis(d, idx, 1) < 0
    bl = np.array([[int(3 * x).bit_length() for x in row] for row in av], np.int64)
    ph = np.where(av > 0, np.int64(1) << np.maximum(bl - 2, 0), 0); ph = np.where(sg, -ph, ph)
    out = np.zeros(d.shape, np.int64); np.put_along_axis(out, idx, ph, 1)
    D[r:r + 4096] = out + wp[None, :]
print("E white plate + 2 loudest, T printed:", tb.words(entry=D, top=print_exact(D)), flush=True)
del D
# floor: the white plate alone for every word (no word-specific light at all)
D = np.broadcast_to(wp[None, :], E.shape).astype(np.int32)
print("E white plate only, T printed:", tb.words(entry=D, top=print_exact(D)), flush=True)
