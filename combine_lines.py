"""Two waves, combined g lines into one on both sides: random groups, each line added up or down at random (the same on
the light and on the plate). The plate combined once; the light combined as it comes. Every bank. Base 2.
    python3 combine_lines.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
BANKS = {id(W[n][0]): n for n in W if n != "head"}; plain_product = product; G = 1; WM = {}
def plan(nin, n):
    rng = np.random.default_rng(abs(hash(n)) % 2**32); grp = rng.permutation(nin) % (nin // G); sgn = rng.choice([-1, 1], nin); return grp, sgn
def product(x16, w16):
    if id(w16) not in BANKS or G == 1: return plain_product(x16, w16)
    n = BANKS[id(w16)]; nin = w16.shape[0]
    if n not in WM:
        grp, sgn = plan(nin, n); Wc = np.zeros((nin // G, w16.shape[1])); np.add.at(Wc, grp, sgn[:, None] * w16); WM[n] = (grp, sgn, Wc)
    grp, sgn, Wc = WM[n]; Xc = np.zeros((x16.shape[0], nin // G)); np.add.at(Xc.T, grp, (sgn * x16).T)
    return (Xc @ Wc).astype(np.int64)
def run(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(plain_product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
truth = [run(ids) for ids in inputs]
for G in (2, 4, 10):
    WM = {}; print(f"  {G:2d} lines combined into one ({100 // G}% of the meetings): right next word {sum(run(i) == t for i, t in zip(inputs, truth))} / 30", flush=True)
