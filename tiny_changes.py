"""The tiny changes. At every bank, the light a token brings against a reference whose meeting with the plate is
recorded once: pour only the difference (the balance), the shared part slipped from both sides. Exact, so the words
cannot move; what is counted is photons (1s, sign apart) in the difference against photons in the light itself.
References: the token before (same text), and one recorded light (another text's, drawn at random) for every token.
Base 2, whole numbers.      python3 tiny_changes.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
def lights(ids):                                     # every bank's light as it enters: 16-bit marks, one power of two a position
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; out = {}
    for l in range(12):
        x16, ex, _ = gauge_log(R); out[f"qkv.{l}"] = (x16, ex)
        A = attention(gauged_bank(R, f"qkv.{l}")); out[f"proj.{l}"] = to16(A, -ES); R = R + plain_bank(A, f"proj.{l}")
        x16, ex, _ = gauge_log(R); out[f"fc.{l}"] = (x16, ex)
        H = edge(gauged_bank(R, f"fc.{l}")); out[f"mproj.{l}"] = to16(H, -ES); R = R + plain_bank(H, f"mproj.{l}")
    return out
ones = lambda v: sum(bin(abs(int(t))).count("1") for t in v)
def photons(a, ea, b, eb):                           # photons in light a, and in a minus b at the finer of the two scales
    e = min(ea, eb); A = a.astype(object) * (1 << int(ea - e)); B = b.astype(object) * (1 << int(eb - e))
    return ones(a), ones(A - B)
L = [lights(ids) for ids in inputs]; rng = np.random.default_rng(0); other = rng.permutation(30)
tot = {}
for n in range(30):
    for name, (x16, ex) in L[n].items():
        kind = name.split(".")[0]; t = tot.setdefault(kind, [0, 0, 0])
        rx, rex = L[other[n] if other[n] != n else (n + 1) % 30][name]
        for p in range(1, 5):
            a, d_prev = photons(x16[p], ex[p], x16[p - 1], ex[p - 1]); _, d_rec = photons(x16[p], ex[p], rx[-1], rex[-1])
            t[0] += a; t[1] += d_prev; t[2] += d_rec
print(f"  {'bank':6s} | photons in the light | difference from the token before | difference from one recorded light")
for kind, (a, dp, dr) in tot.items():
    print(f"  {kind:6s} | {a:>20,} | {dp:>14,} ({100 * dp / a:5.1f}%)       | {dr:>14,} ({100 * dr / a:5.1f}%)")
