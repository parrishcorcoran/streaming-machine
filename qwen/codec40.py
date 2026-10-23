"""The plate as a picture through a lossy picture codec (JPEG-style: each 8x8 tile turned into its waves, the waves kept
at one shared coarseness), squeezed to a target size for the whole plate, decoded back to whole numbers, then
QuantumQwen3 run on the decoded plate. Words against fp32.   python3 codec40.py [MB=40]
"""
import sys, json, math, numpy as np
MB = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
sys.argv = ["x", "texts=0"]; src = open("qwen_base2.py").read(); exec(src[:src.index("from transformers import AutoTokenizer")])
C = np.array([[math.sqrt((1 if k == 0 else 2) / 8) * math.cos(math.pi * (2 * n + 1) * k / 16) for n in range(8)] for k in range(8)])
names = [k for k in W]; total = sum(W[k][0].size for k in names)
def waves(w): b = w.reshape(w.shape[0] // 8, 8, w.shape[1] // 8, 8).transpose(0, 2, 1, 3).astype(np.float32); return np.einsum("ki,abij,lj->abkl", C, b, C, optimize=True)
def unwaves(c, shape): b = np.einsum("ki,abkl,lj->abij", C, c, C, optimize=True); return b.transpose(0, 2, 1, 3).reshape(shape)
def bits(q):                                                              # what the kept waves cost, written with an ideal code
    _, cnt = np.unique(q, return_counts=True); p = cnt / cnt.sum(); return float(-(p * np.log2(p)).sum()) * q.size
rng = np.random.default_rng(0); sample = [waves(W[k][0][:, :min(W[k][0].shape[1], 512)]) for k in names[::7]]
ss = sum(s.size for s in sample)
lo, hi = 1.0, 1e6
for _ in range(40):                                                       # the one coarseness that makes the whole plate MB megabytes
    st = math.sqrt(lo * hi); b = sum(bits(np.round(s / st)) for s in sample) / ss
    if b * total / 8 / 1e6 > MB: lo = st
    else: hi = st
st = hi; print(f"codec coarseness {st:.1f}: about {sum(bits(np.round(s / st)) for s in sample) / ss:.3f} bits per weight -> {MB:.0f} MB for the plate")
for k in names:
    w = W[k][0]; dec = np.round(unwaves(np.round(waves(w) / st) * st, w.shape)); W[k] = (np.clip(dec, -32767, 32767).astype(np.int64), W[k][1])
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); refs = json.load(open("refs_qwen.json"))
for r in refs[:3]:
    got = []
    for _ in range(8): got.append(forward(r["ids"] + got))
    want = r["ref"][:8]; print(f"  {r['prompt']!r:28s} {'IDENTICAL' if got == want else 'DIFFERENT'}  {tok.decode(got)!r}" + ("" if got == want else f"   fp32: {tok.decode(want)!r}"))
