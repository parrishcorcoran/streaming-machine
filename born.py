"""The Born rule at every slab: instead of pouring all the grains of light into every pile, draw K grains with
probability |x_i|^2 / sum |x|^2 (one set of draws per piece per slab, shared by every pile), each weighted by 1/(K p_i).
Arrival, bend, gauge and race unchanged (base2.py's machine). Words against fp32 GPT-2's own.
    python3 born.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json"))
exact_product = product
def born(x16, w16):
    out = np.zeros((x16.shape[0], w16.shape[1]), dtype=np.int64)
    for p in range(x16.shape[0]):
        x = x16[p].astype(np.float64); pr = x * x; tot = pr.sum()
        if tot == 0: continue
        pr /= tot; idx = RNG.choice(len(x), size=K, p=pr)
        out[p] = np.round((x[idx] / (K * pr[idx])) @ w16[idx])
    return out
def gb(R, name):
    w, we, b = W[name]; x16, ex, LS = gauge_log(R)
    return power(born(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
def pbk(X, name):
    w, we, b = W[name]; x16, ex = to16(X, -ES)
    return rshift(born(x16, w), -(we[None, :] + ex[:, None] + ES)) + b
def fwd(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + pbk(attention(gb(R, f"qkv.{l}")), f"proj.{l}")
        R = R + pbk(edge(gb(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(born(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
WORDS, TEXTS = 8, 6
print(f"{TEXTS} texts x {WORDS} words; draws per piece per slab (of 768 / 3072 grains)")
for K in [64, 192, 384, 768, 3072]:
    RNG = np.random.default_rng(0); right = ident = 0
    for r in refs[:TEXTS]:
        ids, want = list(r["ids"]), r["ref"][:WORDS]; got = []
        for _ in range(WORDS):
            got.append(fwd(ids + got))
            if got[-1] != want[len(got) - 1]: break
        n = sum(1 for a, b in zip(got, want) if a == b) if got == want[:len(got)] else len(got) - 1
        right += n; ident += got == want
    print(f"  K = {K:5d}: texts identical {ident}/{TEXTS}, words right before the first wrong one {right}/{TEXTS * WORDS}")
