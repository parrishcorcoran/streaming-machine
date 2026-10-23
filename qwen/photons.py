"""Photons through pinholes at the top read of QuantumQwen3. Every 1 of the light (the top state on 16 binary digits)
is a photon: a 1 at digit k of input i is a photon of brightness 2^k through hole i, landing as that hole's diffraction
pattern (the plate's row i, signed) on all 151,936 word spots at once. Photons land brightest first. Watch the screen:
the brightest word, and the moment it is certain (all photons still to come can't overturn it). Exact at the end.
    python3 photons.py
"""
import sys, json, numpy as np
sys.argv = ["x", "texts=0"]; src = open("qwen_base2.py").read(); exec(src[:src.index("from transformers import AutoTokenizer")])
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); refs = json.load(open("refs_qwen.json"))
w, we = W["head"]; scale = 2.0 ** we.astype(np.float64)                      # each word spot's own power of two (same positive gauge for all)
absrow = np.abs(w).astype(np.float64)
def top_states(ids):
    R = M["embed"][ids].copy()
    for l in range(L):
        R = R + plain_slab(attention(gauged_slab(R, f"qkv.{l}"), l), f"o.{l}")
        GU = gauged_slab(R, f"gu.{l}"); R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}")
    return to16(R)[0]
need, total, first = [], [], []
for t, r in enumerate(refs[:6]):
    ids = r["ids"] + r["ref"][:16]; S = len(r["ids"]); X = top_states(ids)
    for p in range(S - 1, len(ids) - 1):
        x = X[p]; mag, sgn = np.abs(x), np.sign(x)
        screen = np.zeros(w.shape[1]); n_ph = 0; certain_at = None; first_at = None; final = int(r["ref"][p - S + 1])
        photons_total = sum(int(((mag >> k) & 1).sum()) for k in range(16))
        for k in range(15, -1, -1):                                         # digit plane k: all its photons land together
            holes = np.flatnonzero((mag >> k) & 1); n_ph += len(holes)
            if len(holes): screen += (2.0 ** k) * (sgn[holes] @ w[holes].astype(np.float64))
            v = screen * scale; lead = int(np.argmax(v))
            rest = ((2.0 ** k) - 1) * (absrow.sum(0)) * scale if k > 0 else np.zeros_like(v)   # the most all photons still to come could add
            if first_at is None and lead == final: first_at = n_ph
            if first_at is not None and lead != final: first_at = None
            if certain_at is None and np.all(v[lead] - rest[lead] >= np.delete(v + rest, lead)): certain_at = n_ph
            if t == 0 and p == S - 1:
                top5 = np.argsort(-v)[:5]; print(f"  digit {k:2d}: {n_ph:5d} photons landed   brightest: " + "  ".join(f"{tok.decode([int(j)])!r}" for j in top5) + ("   CERTAIN" if certain_at == n_ph else ""))
        need.append(certain_at / photons_total); first.append(first_at / photons_total); total.append(photons_total)
print(f"\n{len(need)} pieces; photons per piece: about {int(np.mean(total))}")
print(f"  the final word is already the brightest (and stays) after: median {100 * np.median(first):.0f}% of the photons")
print(f"  the final word is CERTAIN (nothing still to come can overturn it) after: median {100 * np.median(need):.0f}% of the photons (best {100 * min(need):.0f}%, worst {100 * max(need):.0f}%)")
