"""Rank every neuron's weights by their count of 1s, rank each piece's incoming light by its count of 1s, keep only the
meetings where both are in the top fraction, cancel the rest. Every bank of every layer. Control: same fractions, random.
Base 2. Right next word out of 30.      python3 rank_cancel.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
POP = np.array([bin(i).count("1") for i in range(1 << 16)])
ones = lambda A: POP[np.abs(A).astype(np.int64) & 0xFFFF]
FRAC, RANDOM = 1.0, False; rng = np.random.default_rng(0)
def top(A, axis, f):                                    # True where A's count of 1s ranks in the top fraction f along axis
    if f >= 1: return np.ones(A.shape, bool)
    if RANDOM: return rng.random(A.shape) < f
    c = ones(A) + rng.random(A.shape) * 0.5                 # ties split at random
    k = max(1, int(round(A.shape[axis] * f))); th = -np.sort(-c, axis=axis).take([k - 1], axis=axis)
    return c >= th
WM = {}; BANKS = {id(W[n][0]): n for n in W if n != "head"}; plain_product = product
def product(x16, w16):                                  # only the banks' meetings are ranked; the arrival's own products as they are
    if id(w16) not in BANKS: return plain_product(x16, w16)
    n = BANKS[id(w16)]
    if n not in WM: WM[n] = (w16 * top(w16.astype(np.int64), 0, FRAC))
    xm = x16 * top(x16, 1, FRAC); return (xm.astype(np.float64) @ WM[n]).astype(np.int64)
def run(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(plain_product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
truth = [run(ids) for ids in inputs]
for f in (0.75, 0.5, 0.25):
    for RANDOM in (False, True):
        FRAC = f; WM.clear(); rng = np.random.default_rng(0)
        right = sum(run(ids) == t for ids, t in zip(inputs, truth))
        print(f"  keep top {f:.2f} of weights and of light ({100 * f * f:4.1f}% of meetings) | {'random' if RANDOM else 'ranked by 1s'} | right next word {right} / 30", flush=True)
