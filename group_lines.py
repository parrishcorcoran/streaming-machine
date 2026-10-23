"""Groups of holes (random, fixed once) written as one sum line + difference lines (Walsh-Hadamard: all +/- patterns),
for the light at every slab of every depth of base-2 GPT-2. Photons (1s) in the lines against the flat light; how many
lines go fully dark; and the piles exact: sum_k (Hx)_k (Hw)_k = G * sum_i x_i w_i (undone by a shift).
    python3 group_lines.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); rng = np.random.default_rng(0)
def hadamard(G):
    H = np.array([[1]]); 
    while H.shape[0] < G: H = np.block([[H, H], [H, -H]])
    return H.astype(np.int64)
def ones(a):
    a = np.abs(a.astype(np.int64)); n = 0
    while a.any(): n += int(np.count_nonzero(a & 1)); a >>= 1
    return n
lights = []
for r in refs[:5]:
    ids = r["ids"] + r["ref"][:16]; P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        lights.append(gauge_log(R)[0]); A = attention(gauged_bank(R, f"qkv.{l}")); lights.append(to16(A, -ES)[0])
        R = R + plain_bank(A, f"proj.{l}"); lights.append(gauge_log(R)[0]); H = edge(gauged_bank(R, f"fc.{l}")); lights.append(to16(H, -ES)[0])
        R = R + plain_bank(H, f"mproj.{l}")
for G in (8, 16):
    Hm = hadamard(G); flat = lined = dark = lines = 0
    for X in lights:
        n = X.shape[1]; perm = rng.permutation(n); Xg = X[:, perm].reshape(X.shape[0], n // G, G).astype(np.int64)
        L = Xg @ Hm.T                                        # each group: sum line + difference lines
        flat += ones(X); lined += ones(L); dark += int((L[..., 1:] == 0).sum()); lines += L[..., 1:].size
    print(f"groups of {G:2d}: light photons flat {flat:,}  as lines {lined:,} ({100 * (lined / flat - 1):+.1f}%);  difference lines fully dark: {100 * dark / lines:.2f}%")
# exactness on one slab
w = M["qkv.0.w"].astype(np.int64); x = lights[0][-1].astype(np.int64); G = 16; Hm = hadamard(G); perm = rng.permutation(768)
xl = x[perm].reshape(-1, G) @ Hm.T; wl = np.einsum("kg,ngj->nkj", Hm, w[perm].reshape(-1, G, w.shape[1]))
print(f"piles exact through the lines (groups of 16): {bool(np.array_equal((np.einsum('nk,nkj->j', xl, wl)) >> 4, x @ w))}")
