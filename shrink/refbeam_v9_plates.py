"""REFERENCE BEAM family, version 9: the same reference beams on the 48 plates (lines in x neurons out, int16, one power
of two per neuron). Here the two axes are different: the coder keeps odds per NEURON (each neuron's column has its own
spread), so a value common to a LINE across all neurons is something the coder cannot learn by itself -- unlike the
word tables, where the reference and the odds were on the same axis.
  base   : as stored, rc per neuron (+mant)
  line   : minus the line's middle value over its neurons (768 or 3,072 whole numbers stored)
  neuron : minus the neuron's middle value over its lines
  both   : line, then neuron
Every plate decoded and checked. Side information coded for real and counted.
CONTROL (each plate): every neuron's column shuffled among the lines (seed 0) before the line reference -- keeps each
neuron's values, scrambles which line has which; a line offset then has nothing to hold on to.
    python3 refbeam_v9_plates.py
"""
import time, json, numpy as np
from refbeam_lib import *

M = np.load("/Users/abundancemachine/streaming-machine/base2_gpt2.npz")
def mid(X, axis): return np.sort(X, axis=axis).take(X.shape[axis] // 2, axis=axis)

tot = {"base": 0, "line": 0, "neuron": 0, "both": 0, "control_line": 0, "control_base": 0}; nvals = 0
per_kind = {}
t = time.time()
for L in range(12):
    for kind in ("qkv", "proj", "fc", "mproj"):
        nm = f"{kind}.{L}"; W = M[nm + ".w"].astype(np.int64); n = W.size; nvals += n
        res = {}
        nb, ok = rc(W, "line", True); res["base"] = (nb, 0, ok)
        rl = mid(W, 1); R1 = W - rl[:, None]; s1 = small(rl)
        nb, ok = rc(R1, "line", True); res["line"] = (nb, s1, ok and bool(np.array_equal(R1 + rl[:, None], W)))
        rn = mid(W, 0); R2 = W - rn[None, :]; s2 = small(rn)
        nb, ok = rc(R2, "line", True); res["neuron"] = (nb, s2, ok and bool(np.array_equal(R2 + rn[None, :], W)))
        rn2 = mid(R1, 0); R3 = R1 - rn2[None, :]; s3 = s1 + small(rn2)
        nb, ok = rc(R3, "line", True); res["both"] = (nb, s3, ok and bool(np.array_equal(R3 + rn2[None, :] + rl[:, None], W)))
        rng = np.random.default_rng(0); S = W.copy()
        for j in range(W.shape[1]): S[:, j] = S[rng.permutation(W.shape[0]), j]
        nb, ok = rc(S, "line", True); res["control_base"] = (nb, 8, ok)
        rs = mid(S, 1); nb, ok = rc(S - rs[:, None], "line", True); res["control_line"] = (nb, small(rs) + 8, ok)
        assert all(v[2] for v in res.values()), nm
        for k, (nb, sd, ok) in res.items():
            tot[k] += nb + sd; per_kind.setdefault(kind, {}).setdefault(k, 0); per_kind[kind][k] += nb + sd
        per_kind[kind].setdefault("n", 0); per_kind[kind]["n"] += n
        b = {k: round(8 * (v[0] + v[1]) / n, 3) for k, v in res.items()}
        ph0, ph1 = float(ones(W).mean()), float(ones(R1).mean())
        print(f"   {nm:8s} {b}  photons {ph0:.3f} -> line ref {ph1:.3f}  |line ref| mean {np.abs(rl).mean():.0f}"
              f" |neuron ref| mean {np.abs(rn).mean():.0f}", flush=True)
        del W, R1, R2, R3, S
for k in ("base", "line", "neuron", "both", "control_base", "control_line"):
    report("v9", "P", f"48 plates: {k}", tot[k], 0, nvals, True)
for kind, d in per_kind.items():
    print(f"   {kind:6s}", {k: round(8 * v / d["n"], 4) for k, v in d.items() if k != "n"})
json.dump({k: {kk: (round(8 * vv / d["n"], 4) if kk != "n" else vv) for kk, vv in d.items()} for k, d in per_kind.items()},
          open(HERE + "refbeam_v9_by_kind.json", "w"), indent=1)
print(f"   ({time.time() - t:.0f} s)")
