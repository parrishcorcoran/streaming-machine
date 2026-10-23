"""Row collapse at every layer except the 12th: layers 0-10 each carry their row of 3,072 buckets (after the bends) as
one line (6,145 ticks), read back at the far side; layer 11 left whole. Second run: also the stream coming into every
layer (768 notes) as one line (1,537 ticks). Whole numbers; whole model; right next word out of 30.
    python3 row_collapse_all.py
"""
import sys, json, math, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
def bank(n):
    T = 2 * n + 1; t = np.arange(T); k = np.arange(1, n + 1)
    return T, np.round(np.cos(2 * np.pi * np.outer(k, t) / T) * 2 ** 14).astype(np.int64)
B3072, B768 = bank(3072), bank(768)
def as_line(v, B):                                     # n notes struck into one line, carried, read back (whole numbers)
    T, C = B; line = v @ C; return (rshift(line, 14) @ C.T * 2 + (T << 13)) // (T << 14)
def run(ids, buckets=False, stream=False):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        if stream and 0 < l < 11: R = as_line(R, B768)                       # the stream into layers 1-10 as one line
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        H = edge(gauged_bank(R, f"fc.{l}"))
        if buckets and l < 11: H = as_line(H, B3072)                         # layers 0-10's rows as one line each
        R = R + plain_bank(H, f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
truth = [run(ids) for ids in inputs]
for name, kw in [("layers 0-10's rows of 3,072 as lines", dict(buckets=True)), ("+ the stream into layers 1-10 as lines", dict(buckets=True, stream=True))]:
    print(f"  {name:42s}: right next word {sum(run(ids, **kw) == t for ids, t in zip(inputs, truth))} / 30", flush=True)
