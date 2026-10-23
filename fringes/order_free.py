"""ORDER-FREE CHECK. One bank (fc.6: 768 lines in, 3,072 neurons), the real light of the 30 refs texts (5 tokens each,
all 150 positions). The 768 lines are put in one random order, the SAME order on both sides (light and plate), and every
meeting's count of disagreeing lines is compared with the unmoved one; the piles too. For contrast: the light alone
put in that order (plate unmoved).
    python3 fringes/order_free.py
"""
import sys, os, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
plain_product = product
BANK = "fc.6"; WID = id(W[BANK][0])

def place_rows(V, n):
    U = np.ascontiguousarray((V.astype(np.int64) + (1 << (n - 1))).astype(np.uint64)); assert U.shape[1] % 64 == 0
    return np.stack([np.packbits(((U >> np.uint64(a)) & np.uint64(1)).astype(np.uint8), axis=-1, bitorder="little").view(np.uint64)
                     for a in range(n)])
def disagree(Xr, Wr):
    d = np.empty((Xr.shape[0], Xr.shape[1], Wr.shape[0], Wr.shape[1]), np.int64)
    for a in range(Xr.shape[0]): d[a] = np.bitwise_count(Xr[a][:, None, None, :] ^ Wr[None, :, :, :]).sum(-1, dtype=np.int64)
    return d
def pile_of(d, x, w, n):
    N = x.shape[1]; S = np.zeros((d.shape[1], d.shape[3]), np.int64)
    for a in range(n):
        for b in range(n): S += (N - 2 * d[a, :, b, :]) * (1 << (a + b))
    V = S - (2 * x.sum(1) + N)[:, None] - (2 * w.sum(0, dtype=np.int64) + N)[None, :] + N
    assert (V % 4 == 0).all(); return V // 4

rng = np.random.default_rng(6); perm = rng.permutation(768)
w = M[BANK + ".w"]; wp = w[perm, :]
T = dict(meetings=0, counts_changed=0, piles=0, piles_changed=0, piles_not_machine=0, oneside_counts_changed=0)
def product(x16, w16):
    pile = plain_product(x16, w16)
    if id(w16) != WID: return pile
    x = x16.astype(np.int64); n = max(int(np.abs(x).max()).bit_length(), max(int(w.max()), -int(w.min())).bit_length()) + 1
    d = disagree(place_rows(x, n), place_rows(w.T, n))
    dp = disagree(place_rows(x[:, perm], n), place_rows(wp.T, n))          # same order on both sides
    d1 = disagree(place_rows(x[:, perm], n), place_rows(w.T, n))            # the light alone moved
    p, pp = pile_of(d, x, w, n), pile_of(dp, x[:, perm], wp, n)
    T["meetings"] += d.size; T["counts_changed"] += int((dp != d).sum()); T["oneside_counts_changed"] += int((d1 != d).sum())
    T["piles"] += p.size; T["piles_changed"] += int((pp != p).sum()); T["piles_not_machine"] += int((pp != pile).sum())
    return pile
for ids in inputs:
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(7):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
print(f"bank {BANK}, 30 texts, lines in one random order on both sides: {T}")
