"""Row collapse only (no bends touched): n of layer 0's 3,072 buckets (after their bends), at every position, struck
into ONE line (bucket k on harmonic k+1, T = 2n+1 ticks, whole-number cos table at 2^14), carried as that line, read
back at the far side; every other neuron and layer exactly as they are. Whole model, right next word out of 30.
    python3 row_collapse_n.py
"""
import sys, json, math, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
def bank(n):
    T = 2 * n + 1; t = np.arange(T); k = np.arange(1, n + 1)
    return T, np.round(np.cos(2 * np.pi * np.outer(k, t) / T) * 2 ** 14).astype(np.int64)   # the frequency table (whole numbers)
def run(ids, n=0, CB=None):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; err = 0
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        H = edge(gauged_bank(R, f"fc.{l}"))
        if l == 0 and n > 0:
            T, C = CB; v = H[:, :n]
            line = v @ C                                                   # the row: n buckets as one line of T ticks
            back = rshift(rshift(line, 14) @ C.T * 2, 28) // T if False else (rshift(line, 14) @ C.T * 2 + (T << 13)) // (T << 14)   # read back at the far side
            err = int(np.abs(back - v).max()); H = H.copy(); H[:, :n] = back
        R = R + plain_bank(H, f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b), err
truth = [run(ids)[0] for ids in inputs]
for n in (2, 4, 10, 100, 1000, 3072):
    CB = bank(n); res = [run(ids, n, CB) for ids in inputs]
    print(f"  {n:4d} buckets of layer 0 as one line of {CB[0]:5d} ticks: right next word {sum(r[0] == t for r, t in zip(res, truth))} / 30   (largest read-back miss: {max(r[1] for r in res)} at the stream's 2^-16)", flush=True)
