"""QuantumQwen3's top read with the light in 4,096 up / 4,096 down brightness groups (was: 16,384 photon waves: random same-direction groups (ups with ups, downs with downs),
each ONE direction and ONE brightness (its photons' brightness taken together, shared by every photon in it).
Every photon still lands on its own pile. Words against the full top read, 3 texts x 16 pieces.
    python3 head_waves.py
"""
import sys, json, numpy as np
sys.argv = ["x", "texts=0"]; src = open("qwen_base2.py").read(); exec(src[:src.index("from transformers import AutoTokenizer")])
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); refs = json.load(open("refs_qwen.json"))
w, we = W["head"]
def grouped(x): return np.sign(x) * (((np.abs(x) + 4) >> 3) << 3)        # 4,096 up / 4,096 down brightness groups
same = total = 0; used = []
for r in refs[:3]:
    ids = r["ids"] + r["ref"][:16]; S = len(r["ids"]); R = M["embed"][ids].copy()
    for l in range(L):
        R = R + plain_slab(attention(gauged_slab(R, f"qkv.{l}"), l), f"o.{l}")
        GU = gauged_slab(R, f"gu.{l}"); R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}")
    x16, ex = to16(R); LSv = gauge(x16, ex, D)
    got = []
    for p in range(S - 1, len(ids) - 1):
        F = (we + ex[p] + ES) * 65536 - LSv[p]
        a = race(power((x16[p] @ w)[None, :], F[None, :])[0]); xg = grouped(x16[p]); used.append(len(np.unique(xg))); b = race(power((xg @ w)[None, :], F[None, :])[0])
        same += a == b; total += 1; got.append(b)
    print(f"  {r['prompt']!r:28s} light in 4,096 up / 4,096 down: {tok.decode(got)!r}")
print(f"same word as the full top read: {same}/{total}   (groups lit per piece: about {int(np.median(used))} of 8,192)")
