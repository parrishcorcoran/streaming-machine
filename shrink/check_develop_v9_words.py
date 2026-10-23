"""CHECKER for the develop family's best claim (v9: E exposed for the lens, cap 12, T printed from E).
Rebuilds the developed E and the printed T with its own short code (not develop_lib), checks they are the very tables
develop_lib builds, then judges them with words(...) /30. Also judges a variant printed with one exact rounding
(a single nearest step instead of two), to see whether the verdict sits on a knife edge.
    python3 check_develop_v9_words.py
"""
import sys, json, struct, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import table_bits as tb
import develop_lib as dl

# the gain, read once from the checkpoint, at 2^24 (own read)
p = "/Users/abundancemachine/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); a, b = h["ln_f.weight"]["data_offsets"]
f.seek(8 + n + a); G = np.round(np.frombuffer(f.read(b - a), "<f4").astype(np.float64) * 2 ** 24).astype(np.int64)
assert np.array_equal(G, np.load("/Users/abundancemachine/streaming-machine/shrink/develop_gain24.npy"))
place = np.array([int(g).bit_length() - 1 for g in G], np.int64)

print("machine unchanged:", tb.words(), flush=True)
tb.E = None; E = tb._WTE0
assert E.dtype == np.int64
T0 = tb.T.astype(np.int64)
assert (tb.M["head.e"] == -12).all()

def develop(c, cap=12):
    e = np.minimum(np.maximum(c - place, 0), cap); q = (np.int64(1) << e)[None, :]
    D = np.empty(E.shape, np.int32)
    for r in range(0, E.shape[0], 4096):
        v = E[r:r + 4096]; m = np.abs(v); m2 = ((m + (q >> 1)) >> e) << e
        D[r:r + 4096] = np.where(v < 0, -m2, m2)
    return D, e

def print_two_step(D):     # what develop_lib does: nearest at 2^16, then divide by 3*4096 with a multiply and shift
    out = np.empty(D.shape, np.int32)
    for r in range(0, D.shape[0], 4096):
        s = D[r:r + 4096].astype(np.int64) * G[None, :]; c = 768 * s - s.sum(1, keepdims=True)
        y = (c + (1 << 23)) >> 24; out[r:r + 4096] = (y * 1431655766 + (1 << 43)) >> 44
    return out

def print_exact(D):        # one nearest step: (768 s - sum s) / (768 * 2^28), whole-number floor division
    out = np.empty(D.shape, np.int32); den = 768 * (1 << 28)
    for r in range(0, D.shape[0], 4096):
        s = D[r:r + 4096].astype(np.int64) * G[None, :]; c = 768 * s - s.sum(1, keepdims=True)
        out[r:r + 4096] = (c + den // 2) // den
    return out

Tp0 = print_two_step(E)
print(f"T printed from unchanged E vs stored T: equal share {float((Tp0 == T0).mean()):.4f}, max |diff| {int(np.abs(Tp0 - T0).max())}", flush=True)
Tx0 = print_exact(E)
print(f"exact print vs stored T: equal share {float((Tx0 == T0).mean()):.4f}, max |diff| {int(np.abs(Tx0 - T0).max())}", flush=True)
del T0
print("E unchanged + T printed (two-step):", tb.words(entry=E, top=Tp0), flush=True)
del Tp0, Tx0

res = {}
for c in (32, 34, 37):
    D, e = develop(c)
    Dl, el = dl.develop_lines(E, c, 12)
    same_D = bool(np.array_equal(D, Dl)) and bool(np.array_equal(e, el)); del Dl
    Tp = print_two_step(D)
    same_T = bool(np.array_equal(Tp, dl.print_top(D)))
    w2 = int(tb.words(entry=D, top=Tp))
    Tx = print_exact(D); ndiff = int((Tx != Tp).sum())
    wx = int(tb.words(entry=D, top=Tx))
    # E alone developed (T stored as is) and T printed alone from the unchanged E: which side carries the loss
    wE = int(tb.words(entry=D))
    wT = int(tb.words(top=Tp))
    ph = float(tb.naf(D[::7]).mean())
    res[c] = dict(same_D_as_develop_lib=same_D, same_T_as_develop_lib=same_T, words_two_step=w2, words_exact_print=wx,
                  exact_vs_two_step_values_differ=ndiff, words_E_dev_only=wE, words_T_printed_from_dev_only=wT,
                  photons_E=round(ph, 3), e_hist=np.bincount(e).tolist())
    print("RESULT", c, json.dumps(res[c]), flush=True)
    del D, Tp, Tx
json.dump(res, open("/Users/abundancemachine/streaming-machine/shrink/check_develop_v9_words.json", "w"), indent=1)
