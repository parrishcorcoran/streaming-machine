"""Combine the photons of the lines and divide back: (sum of the token's lines) x (sum of the neuron's lines) / lines,
over all lines at once or in random groups. Shown: real piles against combined piles, then words out of 30.
    python3 combine_ways.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
BANKS = {id(W[n][0]): n for n in W if n != "head"}; plain_product = product; G = None; PLAN = {}
def combined(x16, w16, g, key):
    nin = w16.shape[0]
    if key not in PLAN: PLAN[key] = np.random.default_rng(abs(hash(key)) % 2**32).permutation(nin) % (nin // g)
    grp = PLAN[key]; Xg = np.zeros((x16.shape[0], nin // g)); Wg = np.zeros((nin // g, w16.shape[1]))
    np.add.at(Xg.T, grp, x16.T.astype(np.float64)); np.add.at(Wg, grp, w16.astype(np.float64))
    return np.round(Xg @ Wg / g).astype(np.int64)
def product(x16, w16):
    if id(w16) not in BANKS or G is None: return plain_product(x16, w16)
    return combined(x16, w16, G if G != "all" else w16.shape[0], (BANKS[id(w16)], G))
def run(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(plain_product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
# the piles themselves: layer 0, neurons in, the first piece's last token, the first 8 neurons
ids = inputs[0]; R = M["wte"][ids] + M["wpe"][:len(ids)]; R = R + plain_bank(attention(gauged_bank(R, "qkv.0")), "proj.0")
x16, ex, LS = gauge_log(R); w = W["fc.0"][0]
print("  layer 0, neurons in, one token -- the pile at each of the first 8 neurons:")
print("    line by line      : " + "  ".join(f"{int(v):>9,}" for v in plain_product(x16, w)[-1, :8]))
for g in ("all", 64, 8, 2):
    print(f"    combined {str(g):>3s} {'lines' if g == 'all' else 'a group':7s}: " + "  ".join(f"{int(v):>9,}" for v in combined(x16, w, 768 if g == 'all' else g, ('show', g))[-1, :8]))
print(f"    (the token's 768 lines summed: {int(x16[-1].sum()):,}  against {int(np.abs(x16[-1]).sum()):,} with every sign made up)\n")
truth = [run(i) for i in inputs]
for G in ("all", 64, 8, 2):
    print(f"  every bank combined, {str(G):>3s} {'lines' if G == 'all' else 'lines a group':13s}: right next word {sum(run(i) == t for i, t in zip(inputs, truth))} / 30", flush=True)
