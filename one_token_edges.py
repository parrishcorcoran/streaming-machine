"""One token's edges used for 30 different tokens. The new piece's neurons at a layer combined on the same places,
their edges set beforehand: each neuron's share taken from ONE other token's run. Everything else live (the arrival,
the gauges, the earlier pieces' own edges). Base 2, whole numbers. 10 source tokens drawn at random, each used on the
other 29: right next word out of 29. Shuffle control: the source's shares moved to other neurons (same values).
    python3 one_token_edges.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
def share_of(H):                                       # the edge: where the value arrives picks its share (a table read)
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
def run(ids, over=None):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; got = []
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        pre = gauged_bank(R, f"fc.{l}"); s = share_of(pre); got.append(s[-1].copy())
        if over is not None and l in over:
            J, sA = over[l]; s = s.copy(); s[-1, J] = sA[J]       # the new piece's edges at J, set beforehand from the source
        R = R + plain_bank(rshift(pre * s, 16), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b), got
rng = np.random.default_rng(0)
live = [run(ids) for ids in inputs]; truth = [w for w, _ in live]; shares = [s for _, s in live]
print(f"own edges, live (control): right next word {sum(run(ids)[0] == t for ids, t in zip(inputs, truth))} / 30", flush=True)
sources = sorted(rng.choice(30, 10, replace=False).tolist()); perm = rng.permutation(3072)
rows = [("layer 0, 2 neurons", [0], 2, False), ("layer 0, 100 neurons", [0], 100, False), ("layer 0, 1,000 neurons", [0], 1000, False),
        ("layer 0, all 3,072", [0], 3072, False), ("all 12 layers, all 3,072 each", list(range(12)), 3072, False),
        ("layer 0, all 3,072, shuffled", [0], 3072, True), ("all 12 layers, shuffled", list(range(12)), 3072, True)]
print(f"source tokens: {sources}; each used on the other 29")
print(f"  {'neurons with edges from one other token':42s} | right out of 29 (mean over 10 sources) | fewest .. most")
for name, layers, n, shuf in rows:
    J = rng.choice(3072, n, replace=False); per = []
    for a in sources:
        over = {l: (J, shares[a][l][perm] if shuf else shares[a][l]) for l in layers}
        per.append(sum(run(inputs[b], over)[0] == truth[b] for b in range(30) if b != a))
    print(f"  {name:42s} | {np.mean(per):5.1f} / 29 {'':26s} | {min(per)} .. {max(per)}", flush=True)
