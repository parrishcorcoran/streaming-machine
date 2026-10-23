"""QuantumQwen3's top read as 16,384 random same-direction groups. The head's meetings (hole i, word j) are split by
direction (ups: positive weights, downs: negative), shuffled at random within each direction, cut into 16,384 groups of
equal size (made once). Per piece each group pours its own meetings onto their piles, all whole numbers.
Piles must equal the machine's top read exactly; the race must give the same word.
    python3 head_random_groups.py
"""
import sys, json, time, numpy as np
sys.argv = ["x", "texts=0"]; src = open("qwen_base2.py").read(); exec(src[:src.index("from transformers import AutoTokenizer")])
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); refs = json.load(open("refs_qwen.json"))
w, we = W["head"]; nin, nout = w.shape; G = 16384; rng = np.random.default_rng(0)
t0 = time.time()
flat = w.ravel()
ups, downs = np.flatnonzero(flat > 0).astype(np.int64), np.flatnonzero(flat < 0).astype(np.int64)
rng.shuffle(ups); rng.shuffle(downs)
gu = int(round(G * len(ups) / (len(ups) + len(downs)))); gd = G - gu
groups = [(c, +1) for c in np.array_split(ups, gu)] + [(c, -1) for c in np.array_split(downs, gd)]
sizes = np.array([len(c) for c, _ in groups])
print(f"made once: {len(groups):,} groups ({gu:,} up, {gd:,} down), {sizes.min():,}-{sizes.max():,} meetings each, of {len(ups) + len(downs):,} meetings ({time.time() - t0:.0f} s)")
mag = np.abs(flat).astype(np.int64)
for r in refs[:3]:
    R = M["embed"][r["ids"]].copy()
    for l in range(L):
        R = R + plain_slab(attention(gauged_slab(R, f"qkv.{l}"), l), f"o.{l}")
        GU = gauged_slab(R, f"gu.{l}"); R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}")
    x16, ex = to16(R[-1:]); x = x16[0].astype(np.int64)
    piles = np.zeros(nout, np.int64)
    for c, d in groups:                                             # each group: its own operator, pouring in its one direction
        np.add.at(piles, c % nout, d * x[c // nout] * mag[c])
    machine = x @ w
    F = (we + ex[0] + ES) * 65536 - gauge(x16, ex, D)[0]
    word_g, word_m = race(power(piles[None, :], F[None, :])[0]), race(power(machine[None, :], F[None, :])[0])
    print(f"  {r['prompt']!r:28s} piles from the 16,384 groups == machine's piles: {bool(np.array_equal(piles, machine))}   word {tok.decode([word_g])!r} (machine {tok.decode([word_m])!r}, fp32 {tok.decode([r['ref'][0]])!r})")
