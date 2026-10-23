"""The light into every slab, in base 2: its 1s, against the 1s of its departure from the standing pattern (the same
machine with silence in: zero word rows, positions kept), cut at the same scale. Exact split: slab(x) = slab(standing) + slab(departure).
    python3 departure_ones.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json"))
def lights(R0):
    R = R0; L = []
    for l in range(12):
        L.append(("first", l, R)); A = attention(gauged_bank(R, f"qkv.{l}")); L.append(("second", l, A))
        R = R + plain_bank(A, f"proj.{l}"); L.append(("third", l, R)); H = edge(gauged_bank(R, f"fc.{l}")); L.append(("fourth", l, H))
        R = R + plain_bank(H, f"mproj.{l}")
    L.append(("top", 12, R)); return L
def ones(a):
    m = np.abs(a).astype(np.int64); n = 0
    while m.any(): n += int(np.count_nonzero(m & 1)); m >>= 1
    return n
tot = {}
for r in refs[:10]:
    ids = r["ids"] + r["ref"][:16]; P = len(ids); S = len(r["ids"])
    real = lights(M["wte"][ids] + M["wpe"][:P]); stand = lights(np.zeros((P, 768), np.int64) + M["wpe"][:P])
    for (k, l, X), (_, _, Y) in zip(real, stand):
        x16, ex = to16(X, -ES); sh = (ex + ES)[:, None]
        dep = rshift(X - Y, sh)                                         # the departure, cut at the very same scale as the light
        a, b = tot.get((k, l), (0, 0)); tot[(k, l)] = (a + ones(x16[S:]), b + ones(dep[S:]))
print("depth   1s of the departure as a share of the light's 1s:  first   second   third   fourth")
for l in range(12): print(f"  {l:2d}                                                    " + "  ".join(f"{100 * tot[(k, l)][1] / tot[(k, l)][0]:5.1f}%" for k in ("first", "second", "third", "fourth")))
print(f"  top                                                   {100 * tot[('top', 12)][1] / tot[('top', 12)][0]:5.1f}%")
a = sum(v[0] for v in tot.values()); b = sum(v[1] for v in tot.values()); print(f"all slabs: departure carries {100 * b / a:.1f}% of the 1s")
