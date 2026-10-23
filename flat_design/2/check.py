"""FLAT_DESIGN/2/CHECK: a second count by separate code, on one plate and a few positions, set beside measure.py's own
functions on the same positions. Shares nothing with measure.py's counting:
  - signed digits by the digit-by-digit rule (v mod 4 == 3 -> -1, == 1 -> +1), not popcount(a ^ 3a)
  - 1s by Python's bin(v).count('1')
  - every ready pile formed line by line from the light (sum of +x for up lines, -x for hole lines), no table
  - the excess: every tip summed line by line in Python ints for 3 piles, and as int8 products for the rest
  - the flat held on both sides: S rebuilt from texts 0-14 by Python's statistics.median per line
    python3 flat_design/2/check.py   -> flat_design/2/check.log
"""
import sys, json, statistics, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_design/2"
LOG = open(f"{OUT}/check.log", "w")
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
KIND, LAYER, KI = "proj", 9, 1

def naf_digits(v):            # digit by digit
    v = abs(int(v)); n = 0
    while v:
        if v & 1:
            d = 2 - (v & 3); v -= d; n += 1                    # v mod 4 == 1 -> +1, == 3 -> -1
        v >>= 1
    return n
def ones_py(v): return bin(abs(int(v))).count("1")

M = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
npos = REF["npos"].astype(int); off = np.concatenate([[0], np.cumsum(npos)])
w = np.asarray(M[f"{KIND}.{LAYER}.w"]).astype(np.int64); N, Mn = w.shape
xa = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{KIND}.npy", mmap_mode="r")[LAYER]).astype(np.int64)
pos = [int(off[15]), int(off[15]) + 1, int(off[22])]; x = xa[pos]; T = len(pos)
log(f"check on {KIND}.{LAYER}, positions {pos}")

# 1. plain, by Python ints
o_w = [sum(ones_py(v) for v in w[i]) for i in range(N)]
plain = sum(ones_py(x[t, i]) * o_w[i] for t in range(T) for i in range(N))

# 2. the excess, up/hole, line by line for 3 piles in Python ints; all piles by int8 products
def uh_signs(v): c = int(v) + 32768; return [1 if (c >> a) & 1 else -1 for a in range(16)]
tips_py = 0
for (t, j) in [(0, 0), (1, 100), (2, 767)]:
    xs = [uh_signs(x[t, i]) for i in range(N)]; ws = [uh_signs(w[i, j]) for i in range(N)]
    for a in range(16):
        for b in range(16): tips_py += abs(sum(xs[i][a] * ws[i][b] for i in range(N)))
XU = np.stack([[[1 if ((int(v) + 32768) >> a) & 1 else -1 for v in x[t]] for t in range(T)] for a in range(16)]).astype(np.int32)
WU = np.stack([np.where(((w + 32768) >> b) & 1, 1, -1) for b in range(16)]).astype(np.int32)
tips_all = 0; tips3 = 0
for a in range(16):
    for b in range(16):
        tip = XU[a] @ WU[b]; tips_all += int(np.abs(tip).sum()); tips3 += int(abs(tip[0, 0]) + abs(tip[1, 100]) + abs(tip[2, 767]))
log(f"  up/hole tips of piles (0,0),(1,100),(2,767): python {tips_py}, int products {tips3}: {'SAME' if tips_py == tips3 else 'DIFFER'}")

# 3. the group machine's reads: every ready pile formed line by line, digits by the digit rule
perm = np.random.default_rng([1, KI, LAYER, 12]).permutation(N); groups = [perm[i:i + 12] for i in range(0, N, 12)]
S = np.array([round(statistics.median(xa[off[0]:off[15], i].tolist())) for i in range(N)], np.int64)
e = x - S
def reads(light):
    tot = 0; pile2 = np.zeros((T, Mn), np.int64)
    for L in groups:
        up = (((w[L] + 32768)[:, :, None] >> np.arange(16)[None, None, :]) & 1).astype(bool)   # g x Mn x 16
        sgn = np.where(up, 1, -1).astype(np.int64)
        for t in range(T):
            V = (light[t, L][:, None, None] * sgn).sum(0)                                   # Mn x 16, line by line
            pile2[t] += (V << np.arange(16)[None, :]).sum(1)
            vals, cnts = np.unique(V, return_counts=True)
            tot += sum(naf_digits(v) * int(c) for v, c in zip(vals, cnts))
    return tot, pile2
r_x, p2x = reads(x); r_e, p2e = reads(e)
X = x.sum(1); E = e.sum(1)
ok_x = np.array_equal((p2x - X[:, None]) // 2, x @ w); ok_e = np.array_equal((p2e - E[:, None]) // 2 + (S @ w)[None, :], x @ w)
log(f"  plain {plain:,}; reads light {r_x:,} ({100 * r_x / plain:.3f}% of plain), reads departure {r_e:,} "
    f"({100 * r_e / r_x:.3f}% of the light's); piles exact: light {ok_x}, flat held on both sides {ok_e}")

# 4. measure.py's own functions on the same positions, for comparison
sys.argv = ["x"]; sys.path.insert(0, OUT); import measure as m
gl = m.grouping(N, 12, [1, KI, LAYER, 12]); PT = m.patterns(w, gl); c = m.run_collapse_places(PT, [len(L) for L in gl])
Sm = np.rint(np.median(xa[off[0]:off[15]], axis=0)).astype(np.int64)
_, cx, _ = m.group_machine(x, w, gl, PT, c); _, ce, _ = m.group_machine(x - Sm, w, gl, PT, c)
plain_m = int(m.ones(x).sum(0) @ m.ones(w).sum(1)); ex = m.excess(x, w, "uh")
log(f"  measure.py: plain {plain_m:,}  reads light {cx['read']:,}  reads departure {ce['read']:,}  up/hole tips {ex['tips_abs']:,} "
    f"(check: {tips_all:,});  S same: {np.array_equal(S, Sm)}")
same = plain == plain_m and r_x == cx["read"] and r_e == ce["read"] and tips_all == ex["tips_abs"] and np.array_equal(S, Sm)
log(f"VERDICT: {'every number the same' if same else 'NUMBERS DIFFER'}")
