"""Base-2 GPT-2's top read as 16,384 photon waves: random same-direction groups, each ONE direction and ONE
brightness (its photons counted, shared over its meetings, whole numbers only). Every photon lands on its own pile.
Words against the full top read, 3 texts x 16 pieces.
    python3 head_waves_gpt2.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained("gpt2"); refs = json.load(open("refs_many.json"))
w, we, b = M["head.w"].astype(np.int64), M["head.e"], M["head.b"]; nin, nout = w.shape; G = 16384; rng = np.random.default_rng(0)
flat = w.ravel(); ups, downs = np.flatnonzero(flat > 0), np.flatnonzero(flat < 0); rng.shuffle(ups); rng.shuffle(downs)
gu = int(round(G * len(ups) / (len(ups) + len(downs))))
waves = np.zeros_like(flat)
for c in np.array_split(ups, gu): waves[c] = (int(np.abs(flat[c]).sum()) + len(c) // 2) // len(c)
for c in np.array_split(downs, G - gu): waves[c] = -((int(np.abs(flat[c]).sum()) + len(c) // 2) // len(c))
wv = waves.reshape(nin, nout); print(f"{G:,} groups of about {flat.size // G:,} meetings")
same = total = 0
for r in refs[:3]:
    ids = r["ids"] + r["ref"][:16]; S = len(r["ids"]); P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    x16, ex, LS = gauge_log(R); got = []
    for p in range(S - 1, P - 1):
        F = (we + ex[p] + ES) * 65536 - LS[p]
        a = race(power((x16[p] @ w)[None, :], F[None, :])[0] + b); c = race(power((x16[p] @ wv)[None, :], F[None, :])[0] + b)
        same += a == c; total += 1; got.append(c)
    print(f"  {r['prompt']!r:28s} with 16,384 waves: {tok.decode(got)!r}")
print(f"same word as the full top read: {same}/{total}")
