"""Every meeting with only a share of its 1s: each value keeps the top ceil(f x its count of 1s) photons, the lowest
dropped. Weights alone, then weights and the incoming light. Every bank, every layer. Base 2. Right next word / 30.
    python3 keep_ones.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
def keep_top(A, f):                                    # each value: its top ceil(f * ones) photons, sign kept
    A = A.astype(np.int64); s = np.sign(A); m = np.abs(A); nb = int(m.max()).bit_length()
    ones = np.zeros(m.shape, np.int64)
    for b in range(nb): ones += (m >> b) & 1
    k = np.ceil(f * ones).astype(np.int64); out = np.zeros(m.shape, np.int64); seen = np.zeros(m.shape, np.int64)
    for b in range(nb - 1, -1, -1):
        bit = (m >> b) & 1; take = bit & (seen < k); out += take << b; seen += bit
    return s * out
BANKS = {id(W[n][0]): n for n in W if n != "head"}; plain_product = product; F = (1.0, False); WM = {}
def product(x16, w16):
    if id(w16) not in BANKS or F[0] >= 1: return plain_product(x16, w16)
    n = BANKS[id(w16)]
    if n not in WM: WM[n] = keep_top(w16, F[0]).astype(np.float64)
    x = keep_top(x16, F[0]) if F[1] else x16
    return (x.astype(np.float64) @ WM[n]).astype(np.int64)
def run(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(plain_product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
truth = [run(ids) for ids in inputs]
for both in (False, True):
    for f in (0.3, 0.2, 0.1):
        F = (f, both); WM = {}
        print(f"  keep {f:.0%} of the 1s of {'weights and light' if both else 'weights only':17s}: right next word {sum(run(i) == t for i, t in zip(inputs, truth))} / 30", flush=True)
