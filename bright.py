"""Keep the bright, drop the dark. Brightness = size (where the top photon sits). Every bank of every layer; whatever is
kept keeps all its places. Light: at each piece, its brightest lines; plate: each neuron's brightest weights; or both.
Base 2. Right next word out of 30.      python3 bright.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
def brightest(A, axis, f):                             # True where |A| ranks in the top fraction f along axis
    k = max(1, int(round(A.shape[axis] * f))); m = np.abs(A.astype(np.int64))
    th = -np.sort(-m, axis=axis).take([k - 1], axis=axis); return m >= th
BANKS = {id(W[n][0]): n for n in W if n != "head"}; plain_product = product; MODE = (None, 1.0); WM = {}
def product(x16, w16):
    side, f = MODE
    if id(w16) not in BANKS or side is None: return plain_product(x16, w16)
    n = BANKS[id(w16)]
    if side in ("plate", "both"):
        if n not in WM: WM[n] = w16 * brightest(w16, 0, f)
        w = WM[n]
    else: w = w16
    x = x16 * brightest(x16, 1, f) if side in ("light", "both") else x16
    return (x.astype(np.float64) @ w).astype(np.int64)
def run(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(plain_product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
truth = [run(ids) for ids in inputs]
for side in ("light", "plate", "both"):
    for f in (0.5, 0.25, 0.1):
        MODE = (side, f); WM = {}
        print(f"  keep the brightest {f:.0%} of the {side:5s}: right next word {sum(run(i) == t for i, t in zip(inputs, truth))} / 30", flush=True)
