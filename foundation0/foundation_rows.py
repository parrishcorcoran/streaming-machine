"""The 241 input-table rows the light never reached, stored as  finished = round(c x start) + D'.  2026-10-06.

c is the one surviving share those rows carry (foundation_check.log: 6.112e-4 14m, 6.118e-4 70m, 1.1803e-2 160m), one
number per size, written as a whole number at 2^-30. Shared 16-place grid of base2_pythia-<size>.npz.
Measured on those rows and on the whole input table: lit places (signed digits) and bits a weight at the floor
(each row's own spread), for W and for D' = W - round(c x W0). Control: the same c applied to 241 rows drawn at
random from the other 50,063 rows.

PREDICTION (written before this script first ran): on the 241 rows every lane of D' is 0 or +-1, so D' keeps under
1 lit place a weight against W's 1.5-3 (14m, 70m: W there is ~3 grains; 160m: ~60 grains, W 4-6 lit places), and
the bits fall to under 1.5 a weight. The rows are 0.48% of the table, so the whole table's lit places fall by under
0.5%. Control rows: D' keeps within 1% of W's lit places and bits (c x start is ~0.06% of a built row there).
"""
import os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHK = json.load(open(os.path.join(HERE, "foundation_check.json")))
rng = np.random.default_rng(11)
def naf(x): a = np.abs(x); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def bits(x):
    sig = x.std(1); out = np.zeros(len(sig))
    for i, s in enumerate(sig):
        if s >= 2: out[i] = math.log2(s * math.sqrt(2 * math.pi * math.e))
        elif s > 0:
            k = np.arange(-int(14 * s) - 3, int(14 * s) + 4)
            p = np.array([0.5 * (math.erf((kk + .5) / (s * math.sqrt(2))) - math.erf((kk - .5) / (s * math.sqrt(2)))) for kk in k]); p = p[p > 0]
            out[i] = float(-(p * np.log2(p)).sum())
    return out.mean()
for size in ("14m", "70m", "160m"):
    r = CHK[f"{size}@143000"]["embed"]; ids = np.array(r["ids"]); c30 = int(round(r["c_med"] * 2 ** 30))
    z = np.load(os.path.join(HERE, f"base2_pythia-{size}.npz")); W = z["embed.w"].astype(np.int64); W0 = z["embed.s"].astype(np.int64)
    D = W - ((c30 * W0 + (1 << 29)) >> 30)
    others = np.setdiff1d(np.arange(W.shape[0]), ids); ctl = rng.choice(others, len(ids), replace=False)
    for lab, rows in (("never-built 241", ids), ("control 241", ctl)):
        w, d = W[rows], D[rows]
        print(f"{size:5s} {lab:16s} |W| max {np.abs(w).max():6d} |D'| max {np.abs(d).max():6d}  lit/wt W {naf(w).mean():.3f} D' {naf(d).mean():.3f}  "
              f"bits/wt W {bits(w):.3f} D' {bits(d):.3f}")
    lw, ld = naf(W).sum(), naf(D).sum()
    print(f"{size:5s} whole input table: lit places W {lw:,} D' {ld:,} ({ld / lw - 1:+.3%});  rows are {len(ids) / W.shape[0]:.2%} of the table")
