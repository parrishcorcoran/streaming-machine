"""Base-2 GPT-2's top read with the light grouped by brightness: 4,096 groups going up and 4,096 going down. Every
value joins the group nearest its brightness (its 16-bit light rounded to 12 binary digits of size, sign kept); the
group's shared brightness is used for all its members. Whole numbers only. Words against the full top read.
    python3 head_levels_gpt2.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained("gpt2"); refs = json.load(open("refs_many.json"))
w, we, b = M["head.w"].astype(np.int64), M["head.e"], M["head.b"]
def grouped(x):                         # 15 binary digits of size -> 4,096 brightness groups per direction (shared brightness: multiples of 8)
    return np.sign(x) * (((np.abs(x) + 4) >> 3) << 3)
same = total = 0; used = []
for r in refs[:3]:
    ids = r["ids"] + r["ref"][:16]; S = len(r["ids"]); P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    x16, ex, LS = gauge_log(R); got = []
    for p in range(S - 1, P - 1):
        F = (we + ex[p] + ES) * 65536 - LS[p]; xg = grouped(x16[p]); used.append(len(np.unique(xg)))
        a = race(power((x16[p] @ w)[None, :], F[None, :])[0] + b); c = race(power((xg @ w)[None, :], F[None, :])[0] + b)
        same += a == c; total += 1; got.append(c)
    print(f"  {r['prompt']!r:28s} light in 4,096 up / 4,096 down: {tok.decode(got)!r}")
print(f"same word as the full top read: {same}/{total}   (groups actually lit per piece: about {int(np.median(used))} of 8,192)")
