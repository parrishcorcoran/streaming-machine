"""Remove the flat, keep its leftover (the fringes) exactly. Each gate: w = tip + flat (flat = the balanced places, same
rule as flatcut.py). Pile = light x tip + the flat's leftover (light x flat), both whole numbers, kept exactly.
Checks: every pile identical, the words; the size of the leftover against the pile; what the leftover needs stored."""
import sys, json, time, os, numpy as np
src = open("../base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
sys.argv = ["x"]; os.chdir(".."); NS = {}; exec(src, NS); os.chdir("flatcut")
W0 = dict(NS["W"])
def split(w):
    A = np.abs(w); S = np.sign(w); n = A.shape[0]; band = 3 * 0.5 / np.sqrt(n)
    share = np.stack([((A >> p) & 1).mean(0) for p in range(15)])
    flat = np.zeros(A.shape[1], np.int64)
    for g in range(A.shape[1]):
        p = 14
        while p >= 0 and abs(share[p, g] - 0.5) > band: p -= 1
        flat[g] = p + 1
    step = np.left_shift(1, flat)[None, :]
    return S * (A - A % step), S * (A % step), flat          # tip, flat part (w = tip + flat exactly), flat place
P = {}
for k in W0:
    if k == "head": continue
    w = W0[k][0].astype(np.int64); tip, fl, f = split(w); assert np.array_equal(tip + fl, w); P[k] = (tip, fl, f)
# the machine: every bank's pile = light x tip + leftover(light x flat)
orig = NS["product"]
def product(x16, w):
    for k, (tip, fl, f) in P.items():
        if w is W0[k][0]:
            return orig(x16, tip.astype(np.float64)) + orig(x16, fl.astype(np.float64))
    return orig(x16, w)
NS["product"] = product
refs = json.load(open("../refs_many.json")); same = first = 0; t = time.time()
for r in refs:
    for i in range(16):
        got = NS["forward"](r["ids"] + r["ref"][:i]); same += got == r["ref"][i]; first += (i == 0 and got == r["ref"][0])
print(f"tips + the flat's leftover kept exactly: {same}/480 words, first word {first}/30 (uncut machine 479/480, 30/30)  ({time.time()-t:.0f}s)")
print("\nsize of the leftover (the fringes) against the whole pile, real light, every token after the first:")
for kind in ["qkv", "fc", "mproj"]:
    X = np.load(f"../groups/light_gpt2/{kind}.npy", mmap_mode="r")
    for l in [0, 6, 11]:
        x = np.asarray(X[l, 1:]).astype(np.int64); tip, fl, f = P[f"{kind}.{l}"]
        pile = x @ (tip + fl); left = x @ fl; ratio = np.abs(left) / np.maximum(np.abs(pile), 1)
        pl = np.log2(np.maximum(np.abs(left), 1)); pp = np.log2(np.maximum(np.abs(pile), 1))
        print(f"  {kind}.{l:<3d} leftover is {np.median(ratio)*100:5.1f}% of the pile (median) | pile {np.median(pp):4.1f} places, leftover {np.median(pl):4.1f} places")
tb = np.mean(np.concatenate([np.where(np.abs(t) > 0, np.floor(np.log2(np.maximum(np.abs(t) >> f[None, :], 1))) + 2, 0).ravel() for t, _, f in P.values()]))
fb = np.mean(np.concatenate([np.broadcast_to(f[None, :], t.shape).ravel() for t, _, f in P.values()]))
print(f"\nstored per weight: tip {tb:.2f} places + the flat's own places {fb:.2f} = {tb+fb:.2f} (uncut: 16)")
