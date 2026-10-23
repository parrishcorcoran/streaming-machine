# One piece: every 1 of the top's light is a photon (digit k of input i: brightness 2^k through hole i), landing as
# hole i's pattern on all 151,936 word spots. Brightest photons first. Watch the screen; mark when the word is certain.
import sys, json, numpy as np
sys.argv = ["x", "texts=0"]; src = open("qwen_base2.py").read(); exec(src[:src.index("from transformers import AutoTokenizer")])
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); r = json.load(open("refs_qwen.json"))[0]
w, we = W["head"]; scale = 2.0 ** we.astype(np.float64); colsum = np.abs(w).astype(np.float64).sum(0) * scale
R = M["embed"][r["ids"]].copy()
for l in range(L):
    R = R + plain_slab(attention(gauged_slab(R, f"qkv.{l}"), l), f"o.{l}")
    GU = gauged_slab(R, f"gu.{l}"); R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}")
x = to16(R)[0][-1]; mag, sgn = np.abs(x), np.sign(x); screen = np.zeros(w.shape[1]); n = 0; tot = sum(int(((mag >> k) & 1).sum()) for k in range(16))
print(f"{r['prompt']!r}: {tot} photons in all")
for k in range(15, -1, -1):
    holes = np.flatnonzero((mag >> k) & 1); n += len(holes)
    if len(holes): screen += (2.0 ** k) * (sgn[holes] @ w[holes].astype(np.float64))
    v = screen * scale; lead = int(np.argmax(v)); rest = ((2.0 ** k) - 1) * colsum
    sure = bool(np.all(v[lead] - rest[lead] >= np.delete(v + rest, lead)))
    print(f"  digit {k:2d}: {n:5d} photons ({100 * n / tot:3.0f}%)  brightest: " + " ".join(f"{tok.decode([int(j)])!r}" for j in np.argsort(-v)[:5]) + ("   <- CERTAIN" if sure else ""))
