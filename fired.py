"""The bend as a spike: count, every layer, how many of the 3,072 neurons cross their edge (pile above zero), and pour
only those through the out-bank (everything below the edge sends nothing). Base 2. Right next word out of 30.
    python3 fired.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
fired = np.zeros(12); n = 0
def run(ids, only_fired=False, count=False):
    global n
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        pre = gauged_bank(R, f"fc.{l}"); H = edge(pre)
        if count: fired[l] += (pre[-1] > 0).mean()
        if only_fired: H = np.where(pre > 0, H, 0)                 # below the edge: silent
        R = R + plain_bank(H, f"mproj.{l}")
    if count: n += 1
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
truth = [run(ids, count=True) for ids in inputs]
print("  neurons crossing their edge, by layer: " + "  ".join(f"{100 * f / n:.0f}%" for f in fired))
print(f"  pour only the neurons that crossed: right next word {sum(run(i, True) == t for i, t in zip(inputs, truth))} / 30")
