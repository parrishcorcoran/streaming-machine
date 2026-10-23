"""Groups of two, whole base-2 GPT-2 plate. In every column, ups paired at random with ups, downs with downs; each pair
stored once as two lines, sum (a+b) and difference (a-b). Per piece the light's pair is (x+y, x-y); the lines meet:
(x+y)(a+b) + (x-y)(a-b) = 2(xa+yb), halved by a shift. Check every pile exact; count the stored lines' photons (1s).
    python3 pairs.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); rng = np.random.default_rng(0)
def ones(a):
    a = np.abs(a.astype(np.int64)); n = 0
    while a.any(): n += int(np.count_nonzero(a & 1)); a >>= 1
    return n
def make_pairs(w):                                         # per column: random order, ups first then downs; pair neighbours of the same sign
    key = (w < 0) * 2.0 + rng.random(w.shape); order = np.argsort(key, axis=0)
    ws = np.take_along_axis(w, order, 0); a, b = ws[0::2], ws[1::2]; ia, ib = order[0::2], order[1::2]
    if a.shape[0] > b.shape[0]: b = np.vstack([b, np.zeros((1, w.shape[1]), w.dtype)]); ib = np.vstack([ib, ia[-1:]])
    same = np.sign(a) == np.sign(b)                          # a pair across the up/down boundary is split back into two singles
    return a, b, ia, ib, same
flat_ones = pair_ones = 0; names = [f"{bk}.{l}" for l in range(12) for bk in ("qkv", "proj", "fc", "mproj")] + ["head"]
P = {}
for n in names:
    w = M[n + ".w"].astype(np.int64); a, b, ia, ib, same = make_pairs(w); P[n] = (a, b, ia, ib, same)
    flat_ones += ones(w)
    pair_ones += ones(np.where(same, a + b, a)) + ones(np.where(same, a - b, b))     # stored lines: sum and difference (singles stored as they are)
print(f"whole plate: photons (1s) as a flat sheet {flat_ones:,}   as pair-lines {pair_ones:,}   ({100 * (pair_ones / flat_ones - 1):+.1f}%)")
def pair_pile(x, n):
    a, b, ia, ib, same = P[n]; xa, xb = x[ia], x[ib]
    paired = ((xa + xb) * (a + b) + (xa - xb) * (a - b)) >> 1          # two lines meet two lines, halved by a shift
    single = xa * a + xb * b
    return np.where(same, paired, single).sum(0)
r = refs[0]; ids = r["ids"]; R = M["wte"][ids] + M["wpe"][:len(ids)]
x16, ex, LS = gauge_log(R); x = x16[-1].astype(np.int64)
for n in ["qkv.0", "fc.0", "head"]:
    print(f"  {n:6s}: piles from pair-lines == flat piles: {bool(np.array_equal(pair_pile(x, n), x @ M[n + '.w'].astype(np.int64)))}")
