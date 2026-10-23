"""LENS v4 / v5: keep ONE film and drop the residual altogether -- the other film is only what the lens makes of it.
These change values, so they are judged only by words(...) out of 30.

v4: store E. T := the lens applied to E (v1's T^): 2.3% of T's places off by one step (2^-12), never more.
    Stored: E film + 768 gains. Predicted: 30/30 -- each flip is half a step of the head's own rounding, the size of
    the rounding the converter already did once.
v5: store T. E := the inverse lens applied to T (v2's E^), with the word's DC (v5a) or with no DC at all (v5b: the
    centring in every bank blocks a word's DC, so the machine should not see it).
    Stored: T film (16 bits) + 768 gains (+ DC per word for v5a). Predicted: uncertain. On a gf ~1.25 line E^ is E
    rounded to 2^-12/1.25 (about 12 levels of E per step of T, ~3.7 places dropped); on the three lines with gf < 0.01
    T holds only a few levels of E, so the entry loses those lines almost entirely.
Control: v4 with the gains on randomly permuted lines (seed 0): should break the read at the top.
"""
import os, sys, time, json, numpy as np
os.chdir("/Users/abundancemachine/streaming-machine")
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from table_bits import E, T, TE, words
import lens_lib as LL

K, K2 = 24, 24
t0 = time.time()
N, L = E.shape
G = LL.whole(LL.lens_float("ln_f.weight"), K)
H = np.array([LL.div_round(np.array([1 << (K + K2)], np.int64), int(g))[0] for g in G], dtype=np.int64)


def lens(E, G):
    out = np.empty(E.shape, dtype=np.int16)
    for a in range(0, N, 4096):
        Y = E[a:a + 4096] * G[None, :]
        X = Y - LL.div_round(Y.sum(1), 768)[:, None]
        s = (K + 16 + TE[a:a + 4096]).astype(np.int64)[:, None]
        out[a:a + 4096] = np.clip((X + (np.int64(1) << (s - 1))) >> s, -32767, 32767)
    return out


def inv_lens(T, D, H):
    out = np.empty(T.shape, np.int64)
    for a in range(0, N, 4096):
        sh = (TE[a:a + 4096] + 20).astype(np.int64)[:, None]
        v = (T[a:a + 4096].astype(np.int64) << sh) + D[a:a + 4096, None]
        out[a:a + 4096] = LL.rshift_round(v * H[None, :], K2 + 4)
    return out


res = {}
print("unchanged machine:", words(), flush=True)
That = lens(E, G)
print(f"v4 T^ differs from T in {np.mean(That != T)*100:.3f}% of places, max |diff| {np.abs(That.astype(np.int64) - T).max()}", flush=True)
res["v4_store_E_T_from_lens"] = int(words(top=That)); print("v4 words:", res["v4_store_E_T_from_lens"], f"[{time.time()-t0:.0f}s]", flush=True)
Ts = lens(E, G[np.random.default_rng(0).permutation(L)])
res["v4_control_shuffled_gains"] = int(words(top=Ts)); print("v4 control (gains shuffled):", res["v4_control_shuffled_gains"], flush=True)
del Ts, That

D = np.empty(N, np.int64)
for a in range(0, N, 4096):
    D[a:a + 4096] = LL.rshift_round(LL.div_round((E[a:a + 4096] * G[None, :]).sum(1), 768), K - 4)
Eh = inv_lens(T, D, H)
dE = Eh - E
print(f"v5a E^ vs E: mean |diff| {np.abs(dE).mean():.2f} units of 2^-16, max {np.abs(dE).max()}", flush=True)
res["v5a_store_T_E_from_inverse_lens_with_DC"] = int(words(entry=Eh)); print("v5a words:", res["v5a_store_T_E_from_inverse_lens_with_DC"], f"[{time.time()-t0:.0f}s]", flush=True)
del dE
Eh0 = inv_lens(T, np.zeros(N, np.int64), H)
res["v5b_store_T_E_from_inverse_lens_no_DC"] = int(words(entry=Eh0)); print("v5b words:", res["v5b_store_T_E_from_inverse_lens_no_DC"], flush=True)
# separating the two things v5 changes: (i) the DC alone, (ii) the resolution alone
Ec = E - LL.div_round(E.sum(1), 768)[:, None]
res["check_E_with_DC_removed"] = int(words(entry=Ec)); print("check: E with each word's DC removed:", res["check_E_with_DC_removed"], flush=True)
print(json.dumps(res, indent=1)); print(f"done [{time.time()-t0:.0f}s]")
