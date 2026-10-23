"""The plate as a picture through a base-2 codec: whole-number waves by lifting (pairs differenced, each pair's middle
kept with a shift; then pairs of pairs; 3 levels, rows and columns). Coarseness = one right shift of the difference
waves. Size counted in bits (top 1's place + sign per value). No float anywhere. Decoded with adds and shifts, then
QuantumQwen3 runs on it.     python3 codec_base2.py [MB=40]
"""
import sys, json, numpy as np
MB = int(sys.argv[1]) if len(sys.argv) > 1 else 40
sys.argv = ["x", "texts=0"]; src = open("qwen_base2.py").read(); exec(src[:src.index("from transformers import AutoTokenizer")])
def lift(a):                                                   # along the last axis: middle s, difference d (exact, whole numbers)
    e, o = a[..., 0::2], a[..., 1::2]; d = o - e; return e + (d >> 1), d
def unlift(s, d):
    e = s - (d >> 1); o = d + e; out = np.empty(s.shape[:-1] + (2 * s.shape[-1],), np.int64); out[..., 0::2], out[..., 1::2] = e, o; return out
def encode(w, levels=3):
    bands, a = [], w.astype(np.int64)
    for _ in range(levels):
        s, d1 = lift(a); s2, d2 = lift(s.T); d3 = lift(d1.T)
        bands.append((d2.T, d3[0].T, d3[1].T)); a = s2.T                 # three difference waves per level; a = the middle of the middle
    return a, bands
def decode(a, bands):
    for d2, d3a, d3b in reversed(bands):
        s = unlift(a.T, d2.T).T; d1 = unlift(d3a.T, d3b.T).T; a = unlift(s, d1)
    return a
def bits(v):                                                   # base-2 size: top 1's place + a sign for each nonzero value, 1 for a zero
    m = np.abs(v); n = np.zeros(m.shape, np.int64); t = m.copy()
    while t.any(): n += (t > 0); t >>= 1
    return int(np.where(m > 0, n + 1, 1).sum())
enc = {k: encode(W[k][0]) for k in W}
def size_at(sh):
    tot = 0
    for a, bands in enc.values():
        tot += bits(a)
        for b3 in bands: tot += sum(bits(b >> sh) for b in b3)
    return tot // 8
for sh in range(0, 17):                                        # the smallest shift that fits
    sz = size_at(sh); print(f"  shift {sh:2d}: {sz / 1e6:8.1f} MB")
    if sz <= MB * 1_000_000: break
print(f"chosen shift {sh}: the plate as base-2 waves is {sz / 1e6:.1f} MB")
for k in W:                                                    # decode: the difference waves shifted back, then adds and shifts
    a, bands = enc[k]; W[k] = (decode(a, [tuple((b >> sh) << sh for b in b3) for b3 in bands]), W[k][1])
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); refs = json.load(open("refs_qwen.json"))
for r in refs[:3]:
    got = []
    for _ in range(8): got.append(forward(r["ids"] + got))
    want = r["ref"][:8]; print(f"  {r['prompt']!r:28s} {'IDENTICAL' if got == want else 'DIFFERENT'}  {tok.decode(got)!r}" + ("" if got == want else f"   fp32: {tok.decode(want)!r}"))
