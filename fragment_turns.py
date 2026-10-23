"""A fragment of the recording reconstructs the whole. Each bank's circle (every neuron's weights end to end) is
recorded with a whole-number pattern of adds and subtracts (Walsh-Hadamard: every place of the recording made from
every weight). Keep a random fraction of the recording's places at full depth, the rest dark; play it back (the same
adds and subtracts), brighten by the fraction (a shift); turn as usual, every bend live. Compared with the same
fraction of the weights' own places kept. Base 2, whole numbers. Right next word out of 30.
    python3 fragment_turns.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
def run(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
def wht(a):                                            # the recording: adds and subtracts only, every place from every place
    a = a.copy(); h = 1
    while h < a.size:
        a = a.reshape(-1, 2, h); x = a[:, 0, :].copy(); y = a[:, 1, :].copy(); a[:, 0, :] = x + y; a[:, 1, :] = x - y; a = a.reshape(-1); h *= 2
    return a
def fragment(name, k, how, seed):                      # keep 1/2^k of the places, brighten by 2^k
    w = M[name + ".w"].astype(np.int64); nin, nout = w.shape
    circle = w.T.reshape(-1); N = 1 << (circle.size - 1).bit_length(); rng = np.random.default_rng(seed)
    if how == "places":
        keep = rng.random(circle.size) < 2.0 ** -k; out = np.where(keep, circle << k, 0)
    else:
        rec = wht(np.concatenate([circle, np.zeros(N - circle.size, np.int64)]))
        keep = rng.random(N) < 2.0 ** -k; back = wht(np.where(keep, rec, 0))
        out = rshift(back, N.bit_length() - 1 - k)[:circle.size]
    return out.reshape(nout, nin).T.astype(np.float64)
truth = [run(ids) for ids in inputs]
banks = {"fc": [f"fc.{l}" for l in range(12)], "all four": [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]}
restore = lambda n: (M[n + ".w"].astype(np.float64), M[n + ".e"], M[n + ".b"])   # one copy of the weights at a time
for which in ("fc", "all four"):
    for how in ("recording", "places"):
        for k in (0, 1, 2, 3):
            if k == 0 and how == "places": continue
            for i, n in enumerate(banks[which]): W[n] = (fragment(n, k, how, 1000 + i), M[n + ".e"], M[n + ".b"])
            right = sum(run(ids) == t for ids, t in zip(inputs, truth))
            print(f"  {which:8s} banks | {how:9s} | kept 1/{2 ** k:<2d} | right next word {right} / 30", flush=True)
            for n in banks[which]: W[n] = restore(n)
