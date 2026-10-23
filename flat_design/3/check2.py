"""SECOND COUNT WITH SEPARATE CODE (same session; not an independent checker). Shares nothing with measure.py.
  1. fc.3 and mproj.3, all 16 sample positions: the up/hole tips by whole-number products of +-1 planes (no popcount):
     tips[a, b, j] = sum over lines of s_a(x) t_b(w); the extra (sum |tip|), the tips' signed digits, and the pile.
  2. proj.3 at g = 11 and 12, all 16 sample positions: the ready piles built by a Gray walk (one line flipped a step),
     the plate written in the centred per-place form by its own rule, reads counted at the ready piles' signed digits.
  Compared with out/main.json.
"""
import json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; POS = np.arange(5, 648, 40)[:16]
MZ = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
MAIN = json.load(open(f"{HERE}/flat_design/3/out/main.json"))["plates"]
def nsd(v):
    v = np.abs(np.asarray(v, np.int64)); n = np.zeros(v.shape, np.int64)
    while v.any():                                       # the classic rule, digit by digit
        odd = (v & 1) == 1; d = np.where(odd, 2 - (v & 3), 0); n += d != 0; v = (v - d) >> 1
    return n
def light(kind, l): return np.asarray(np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[l][POS]).astype(np.int64)

for kind in ("fc", "mproj"):
    x = light(kind, 3); w = np.asarray(MZ[f"{kind}.3.w"]).astype(np.int64); ref = np.asarray(REF[f"{kind}_pile_3"][POS])
    nx = 16 if x.max() < 1 << 15 else 17
    Sx = np.stack([2 * (((x + (1 << (nx - 1))) >> a) & 1) - 1 for a in range(nx)], 1).astype(np.int32)   # (P, nx, N)
    Tw = np.stack([2 * (((w + (1 << 15)) >> b) & 1) - 1 for b in range(16)]).astype(np.int32)            # (16, N, M)
    ext = dig = 0; bad = 0
    for p in range(len(POS)):
        T = np.stack([Sx[p] @ Tw[b] for b in range(16)], 1).astype(np.int64)                             # (nx, 16, M)
        ext += int(np.abs(T).sum()); dig += int(nsd(T).sum())
        pile = sum(T[a, b] * (1 << (a + b)) for a in range(nx) for b in range(16))
        bad += int(((pile - 2 * x[p].sum() - 2 * w.sum(0) - x.shape[1]) // 4 != ref[p]).any())
    A = MAIN[f"{kind}.3"]["A"]["uh"]
    print(f"{kind}.3 up/hole tips: extra {ext} (main {A['extra']}), tips' digits {dig} (main {A['tipdig']}), piles differing {bad}")

x = light("proj", 3); w = np.asarray(MZ["proj.3.w"]).astype(np.int64); ref = np.asarray(REF["proj_pile_3"][POS]); N, M = w.shape
for g, var in ((12, "B12"), (11, "Bg")):
    G = -(-N // g); r = w.copy(); lit = []
    for b in range(18):
        d = np.where(r & 1, 2 - (r & 3), 0); lit.append(int((d != 0).sum())); r = (r - d) >> 1
    h = 1 + max(b for b in range(18) if lit[b] >= G * M); C = 1 << (h - 1)
    low = np.mod(w + C, 1 << h) - C; high = (w - low) // (1 << h)
    perm = np.random.default_rng([1, 1, 3, g]).permutation(N); groups = [perm[i:i + g] for i in range(0, N, g)]
    reads = 0; bad = 0
    for p in range(len(POS)):
        acc = np.zeros(M, np.int64)
        for L in groups:
            gk = len(L); V = np.zeros(1 << gk, np.int64); code = 0; cur = -x[p, L].sum(); V[0] = cur
            for i in range(1, 1 << gk):                          # Gray walk: flip line t = lowest set bit of i
                t = (i & -i).bit_length() - 1; code ^= 1 << t
                cur += 2 * x[p, L[t]] if (code >> t) & 1 else -2 * x[p, L[t]]; V[code] = cur
            pat = np.zeros((M, h), np.int64)
            for t, line in enumerate(L):
                pat += (((low[line] + C)[:, None] >> np.arange(h)) & 1) << t
            R = V[pat]; reads += int(nsd(R).sum()); acc += (R << np.arange(h)).sum(1)
        acc += -x[p].sum() + (x[p] @ high) * (1 << (h + 1))
        bad += int((acc != 2 * ref[p]).any())
    print(f"proj.3 g = {g}: h {h}, reads {reads} (main {MAIN['proj.3'][var]['reads']}), piles differing {bad}")
