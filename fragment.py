"""The hologram's fragment test: keep a fixed random share f of every slab's input grains (the same cut plate for
every piece), scaled by 1/f. Which pieces still come out right, against how hard each piece is (its lead in fp32)?
  top only: the 12 depths exact, the read at the top from the fragment
  every slab: every slab and the read from fragments
    python3 fragment.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json"))
rng = np.random.default_rng(0)
def cut(nin, f): s = np.zeros(nin, bool); s[rng.permutation(nin)[:int(round(nin * f))]] = True; return s
def frag_product(x16, w, keep, f): return np.round((x16[:, keep].astype(np.float64) @ w[keep]) / f).astype(np.int64)
def run(ids, f, everywhere, cuts):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    def gb(R, name):
        w, we, b = W[name]; x16, ex, LS = gauge_log(R)
        pile = frag_product(x16, w, cuts[name], f) if everywhere else product(x16, w)
        return power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    def pb(X, name):
        w, we, b = W[name]; x16, ex = to16(X, -ES)
        pile = frag_product(x16, w, cuts[name], f) if everywhere else product(x16, w)
        return rshift(pile, -(we[None, :] + ex[:, None] + ES)) + b
    for l in range(12):
        R = R + pb(attention(gb(R, f"qkv.{l}")), f"proj.{l}")
        R = R + pb(edge(gb(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R)
    lg = power(frag_product(x16, w, cuts["head"], f), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return [race(row) for row in lg]
names = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]
buckets = [(0, 0.5, "hard (lead < 0.5)"), (0.5, 2, "middle (0.5-2)"), (2, 99, "easy (lead > 2)")]
for everywhere in (False, True):
    print(f"\n#### fragment {'at every slab and the top' if everywhere else 'at the top read only'}: share of pieces right")
    print("  share of plate kept   " + "   ".join(f"{t:>18s}" for _, _, t in buckets))
    for f in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        cuts = {n: cut(W[n][0].shape[0], f) for n in names}
        hits = {t: [0, 0] for _, _, t in buckets}
        for r in refs:
            ids = r["ids"] + r["ref"][:16]; S = len(r["ids"])
            got = run(ids, f, everywhere, cuts)
            for j in range(16):
                lead = r["margins"][j]; t = next(t for a, b_, t in buckets if a <= lead < b_)
                hits[t][1] += 1; hits[t][0] += got[S - 1 + j] == r["ref"][j]
        print(f"  {f:8.4f}              " + "   ".join(f"{100 * h / max(n, 1):5.1f}% of {n:3d}      " for h, n in hits.values()))
