"""The GPU machine against granite_base2.py, number for number. For each case dumped by
`./granite_stream case=c dump=dump_c.bin`, the same positions (the prompt, then the GPU's own words) go through
granite_base2.py's forward. The final stream at every position and the 8 lit experts (in race order) at every layer and
position must be equal.
    python3 compare_gpu.py 0 7 12 23
"""
import sys, json, numpy as np
cases = [int(a) for a in sys.argv[1:]]; sys.argv = sys.argv[:1]
import granite_base2 as g
refs = json.load(open("refs_granite.json")); gw = {r["case"]: r["words"] for r in json.load(open("gpu_words.json"))}
for c in cases:
    seq = refs[c]["ids"] + gw[c][:-1]; P = len(seq); pos = np.arange(P)
    raw = open(f"dump_{c}.bin", "rb").read()
    Rg = np.frombuffer(raw[:P * g.D * 8], np.int64).reshape(P, g.D)
    Lg = np.frombuffer(raw[P * g.D * 8:P * g.D * 8 + g.L * P * 8 * 4], np.int32).reshape(g.L, P, 8)
    R = np.asarray(g.EMBED[seq]).astype(np.int64); lits = []
    for l in range(g.L):
        R = R + g.plain_slab(g.attention(g.gauged_slab(R, g.plate(f"qkv.{l}")), pos), g.plate(f"o.{l}"))
        out, lit = g.experts(R, l); R = R + out; lits.append(lit)
    lits = np.stack(lits)
    print(f"case {c:2d}: {P} positions | final stream values differing {int((R != Rg).sum())} of {R.size} | "
          f"lit slots differing {int((lits != Lg).sum())} of {lits.size}", flush=True)
