"""Exactness of the photon picture. Light: each 1 at digit k of |x_i| is a photon bundle 2^k, direction sign(x_i).
Plate: each 1 at digit m of the 16-bit weight (two's complement: digit 15 points the other way, -2^15).
Event: a light photon meets a plate photon at hole i -> lands on spot j with 2^(k+m), direction = product.
Pile = sum of events. Must equal the machine's pile x16 @ w exactly, for every slab of depth 0 and the top read.
    python3 photon_exact.py
"""
import sys, json, numpy as np
sys.argv = ["x", "texts=0"]; src = open("qwen_base2.py").read(); exec(src[:src.index("from transformers import AutoTokenizer")])
r = json.load(open("refs_qwen.json"))[0]; ids = r["ids"]
def photon_pile(x, w):
    mag, sgn = np.abs(x).astype(np.int64), np.sign(x).astype(np.int64); u = w.astype(np.int64) & 0xFFFF
    pile = np.zeros(w.shape[1], dtype=object); events = 0
    for m in range(16):
        plate = ((u >> m) & 1).astype(np.int64); pdir = -1 if m == 15 else 1          # plate photons at digit m
        for k in range(16):
            light = ((mag >> k) & 1) * sgn                                            # light photons at digit k, with direction
            if not light.any(): continue
            hits = light @ plate                                                       # events landing on each spot, net of direction
            events += int(np.abs(light) @ plate.sum(1))
            pile = pile + (pdir * (1 << (k + m))) * hits.astype(object)
    return np.array(pile, dtype=np.int64), events
R = M["embed"][ids].copy()
checks = []
x16, ex = to16(R); checks.append(("qkv.0", x16[-1], W["qkv.0"][0]))
QKV = gauged_slab(R, "qkv.0"); A = attention(QKV, 0); a16, _ = to16(A); checks.append(("o.0", a16[-1], W["o.0"][0]))
R = R + plain_slab(A, "o.0"); g16, _ = to16(R); checks.append(("gu.0", g16[-1], W["gu.0"][0]))
GU = gauged_slab(R, "gu.0"); H = bend(GU[:, :3072], GU[:, 3072:]); h16, _ = to16(H); checks.append(("down.0", h16[-1], W["down.0"][0]))
checks.append(("head", g16[-1], W["head"][0]))
for name, x, w in checks:
    p, ev = photon_pile(x, w); machine = x.astype(np.int64) @ w.astype(np.int64)
    print(f"  {name:7s} spots {w.shape[1]:6d}: photon piles == machine piles: {bool(np.array_equal(p, machine))}   (events: {ev:,}; largest difference {int(np.abs(p - machine).max())})")
