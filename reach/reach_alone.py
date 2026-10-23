"""reach/reach_alone.py: (1), CHECKED A SECOND WAY. Could any digit go even by itself?

reach.py took each neuron's digits a whole place at a time from place 0 up and found that not one neuron loses even
its place-0 sheet. That rule is one way of choosing the set. This asks the question that no choice of set can beat:
a digit at place b on line i of neuron j can be in ANY removable set only if, by itself, 2^b x |light on line i| stays
inside the bucket's near edge at every position. Any set's summed move is at least each member's own move, so the
digits that pass alone are the most that any rule could remove.
  q[i, j] = smallest, over the positions where line i is lit, of floor(m[t, j] / |light[t, i]|)  (m: the near edge)
  the digit (i, j, b) passes alone iff 2^b <= q[i, j]   (a line dark at every position: every digit passes).
Found exactly, in two steps (whole numbers): a line lit at any position where the neuron's pile sits exactly on its
bucket's edge (m = 0) has q = 0, so only the lines dark at all of those positions are measured in full.
Also, in his picture's terms, at every bank: the bucket's width (how many piles give one kept value: the places the
bank's water level or shift lets go), the light's size, and how far the piles sit from their edges.
PREDICTION (written before the run): almost no digit passes alone; under 0.01% of the 48 plates' digits, all of them
  on mproj lines whose bend is shut at the few positions where that neuron's pile sits on an edge. The bucket is
  narrower than one place-0 digit's move at the median position at every bank.
    python3 reach/reach_alone.py      -> reach/reach_alone.log, reach/reach_alone.json
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastm as F
F.ROWS = 512
from reach_lib import bucket_gauged, bucket_plain, margins
HERE = F.HERE; OUT = f"{HERE}/reach"; LOG = open(f"{OUT}/reach_alone.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
PLACES = 17; CH = 256; RES = {}

def alone(name, x16, piece):
    w16 = F.plate(name); L, N = w16.shape; T = len(x16)
    absx = np.abs(x16); lit_any = absx > 0; litf = lit_any.astype(np.float32)
    r = dict(digits=0, pass_alone=0, pairs_measured=0, neurons_without_edge0=0, width=[], edge=[])
    for c0 in range(0, N, CH):
        cs = slice(c0, min(c0 + CH, N)); n = cs.stop - cs.start
        m, width = piece(cs)
        z = (m == 0)
        r["neurons_without_edge0"] += int((~z.any(0)).sum())
        D = litf.T @ z.astype(np.float32)                         # lines x neurons: edge-0 positions where the line is lit
        w = w16[:, cs].astype(np.int64); up, dn = F.signed_digits(w); lit = up | dn
        r["digits"] += int(np.bitwise_count(lit).sum())
        q = np.zeros((L, n), np.int64)
        for j in range(n):
            cand = np.flatnonzero(D[:, j] == 0)
            if len(cand) == 0: continue
            r["pairs_measured"] += len(cand)
            ax = absx[:, cand]; mj = m[:, j][:, None]
            ratio = np.where(ax > 0, mj // np.maximum(ax, 1), np.iinfo(np.int64).max)
            q[cand, j] = ratio.min(0)
        for p in range(PLACES):
            r["pass_alone"] += int((((lit >> p) & 1).astype(bool) & ((1 << p) <= q)).sum())
        r["width"].append(np.percentile(width, [0, 50, 100], axis=None)); r["edge"].append(np.median(m))
    wd = np.array(r.pop("width")); r["bucket_width_min"] = int(wd[:, 0].min()); r["bucket_width_median"] = float(np.median(wd[:, 1]))
    r["bucket_width_max"] = int(wd[:, 2].max()); r["edge_median"] = float(np.median(r.pop("edge")))
    r["light_median"] = int(np.median(absx)); r["light_max"] = int(absx.max()); r["dark_share"] = float((~lit_any).mean())
    RES[name] = r
    log(f"  {name:8s}: digits passing alone {r['pass_alone']:>7,} of {r['digits']:>11,}; (line, neuron) pairs dark at every edge-0 "
        f"position {r['pairs_measured']:,}; neurons with no pile on an edge {r['neurons_without_edge0']}/{N}; bucket width "
        f"min {r['bucket_width_min']}, median {r['bucket_width_median']:.0f}, max {r['bucket_width_max']}; median near edge "
        f"{r['edge_median']:.0f}; |light| median {r['light_median']}, max {r['light_max']}; light exactly 0 in "
        f"{100 * r['dark_share']:.2f}% of (position, line) ({time.time() - t0:.0f}s, {F.rss()} MB)")

def hook(name, l, d):
    if name in ("stream in", "stream a", "stream"): return
    x16, P, ex, LS, we = d["x16"], d["pile"], d["ex"], d["LS"], d["we"]
    def piece(cs):
        Pc = P[:, cs]
        if LS is not None:
            F_ = (we[None, cs] + ex[:, None] + F.ES) * 65536 - LS[:, None]
            lo, hi = bucket_gauged(Pc, F_); m = margins(Pc, lambda X: F.power(X, F_), lo, hi)
        else:
            sh = -(we[None, cs] + ex[:, None] + F.ES)
            lo, hi = bucket_plain(Pc, sh); m = margins(Pc, lambda X: F.rshift(X, sh), lo, hi)
        return m, hi - lo + 1
    alone(name, x16, piece)
log("the 48 plates at all 7,848 positions")
x16, ex, LS = F.run(hook=hook, log=log)
RR = F.READ_ROWS; xt, et, lt = x16[RR], ex[RR], LS[RR]; del x16
WH = F.plate("head"); HE = F.E_["head"]
def piece_head(cs):
    P = F.product_lean(xt, WH[:, cs].astype(np.float64)); F_ = (HE[None, cs] + et[:, None] + F.ES) * 65536 - lt[:, None]
    lo, hi = bucket_gauged(P, F_); m = margins(P, lambda X: F.power(X, F_), lo, hi)
    return m, hi - lo + 1
log("the top table at the 6,480 read positions")
alone("head", xt, piece_head)
tot = sum(r["digits"] for r in RES.values()); pa = sum(r["pass_alone"] for r in RES.values())
pl = sum(r["digits"] for k, r in RES.items() if k != "head"); pal = sum(r["pass_alone"] for k, r in RES.items() if k != "head")
log(f"\nDIGITS THAT COULD GO EVEN ALONE: 48 plates {pal:,} of {pl:,} ({100 * pal / pl:.5f}%); with the top table {pa:,} of {tot:,} "
    f"({100 * pa / tot:.5f}%)")
json.dump(RES, open(f"{OUT}/reach_alone.json", "w"), indent=1)
log(f"saved reach/reach_alone.json ({time.time() - t0:.0f}s, max memory {F.rss()} MB)")
