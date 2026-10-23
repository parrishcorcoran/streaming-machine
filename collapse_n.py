"""Collapse n neurons of layer 0 (base-2 GPT-2): each collapsed neuron = its value after the row's shared delay (the
gauge) and its standing offset, then ONE fixed delay for its wave (one for every up neuron, one for every down neuron:
constants measured once over all inputs). Every other neuron and every other layer exactly as they are. Whole model,
right next word out of 30, for n = 0, 2, 4, 10, 100, 1000, 3072.      python3 collapse_n.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
def run(ids, n=0, D=None):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        pre = gauged_bank(R, f"fc.{l}"); H = edge(pre)
        if l == 0 and n > 0:                                   # the n collapsed neurons: one fixed delay per wave
            p = pre[:, :n]; d = np.where(p > 0, D[0], D[1]); H[:, :n] = power(p, -d)
        if l == 0 and D is None and n < 0: return pre, H
        R = R + plain_bank(H, f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
# the medium's two constants: the bend's delay for an up neuron and for a down neuron (log units, 65536 per doubling)
ups, dns = [], []
for ids in inputs:
    pre, H = run(ids, n=-1)
    for v, h in zip(pre.ravel(), H.ravel()):
        if v != 0 and h != 0: (ups if v > 0 else dns).append(int(log2fix([abs(int(v))])[0]) - int(log2fix([abs(int(h))])[0]))
D = (int(np.median(ups)), int(np.median(dns)))
print(f"the medium's two delays: up {D[0] * 16 / 65536:.1f} sixteenths, down {D[1] * 16 / 65536:.1f} sixteenths")
truth = [run(ids) for ids in inputs]
for n in (2, 4, 10, 100, 1000, 3072):
    right = sum(run(ids, n, D) == t for ids, t in zip(inputs, truth))
    print(f"  {n:4d} of 3,072 neurons of layer 0 collapsed: right next word {right} / 30", flush=True)
