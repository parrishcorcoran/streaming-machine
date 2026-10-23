"""A SECOND LOOK, ADDED AFTER where_blank.py: IS THE BLANK TAKEN BACK AS ITSELF, OR ONLY WITH THE WHOLE WRITE? 2026-10-06.

HIS WORDS (2026-10-06), only these are his: "The blank work must be in the embedding table equal in math to the MLP.
It has to cancel." Earlier, also his: "It doens have to be in the same spot."

WHY: where_blank.py found the later layers' writes take back the WORD part of layer 0's blank write in full:
g_word +1.04 (70m, z +5.9 against 20 scrambles) and +1.11 (160m, z +8.1); layer 1 +0.98 at 70m (z +4.4). But
blankrow.py also found layer 0's blank write lined up with the built part of the same write (g_same -1.29 and -1.51,
pooled). If the later layers take back the WHOLE layer-0 write along its own direction, the blank part comes back with
it only because it lies along that write. This look splits it, word parts only (each position less the average over
the 622 positions), at every layer l:
  W_l(p) = the whole write of layer l (attention + MLP, as the plates write it, no bias), B_l(p) its blank part,
  later_l(p) = every write of layers l+1..L-1 as the machine adds them.
  B_l(p) = c_p W_l(p) + Bperp_l(p),  c_p = <B_l(p), W_l(p)> / |W_l(p)|^2   (per position)
  g_along = -sum_p c_p <W_l(p), later_l(p)> / sum_p |B_l(p)|^2       (the blank taken back because it lies along the write)
  g_perp  = -sum_p <Bperp_l(p), later_l(p)> / sum_p |B_l(p)|^2       (the blank taken back as itself)
  g_perp_own = the same over sum_p |Bperp_l(p)|^2                     (the share of the perpendicular part taken back)
  also g_whole = -sum <W_l, later_l> / sum |W_l|^2 and the blank's lean on its write, sum <B,W>/sum |B|^2.
The 20 scrambles and 5 fresh draws (where_blank.py's seeds) are split along the same real W_l, so for them g_along is the
share that rides along by chance.

GUESS, written before this script first ran (the session's): at layer 0, at both sizes, nearly all of the blank's
take-back is g_along (0.8-1.4), and g_perp is inside +-3 scramble spreads: the later layers take back the whole first
write, and the blank part goes back only because it lies along that write. Elsewhere both inside the scrambles.

    python3 where_blank2.py size=70m
"""
import os, sys, json, time, math, resource
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(HERE), "foundation0")
sys.path.insert(0, HERE)
from common import fresh

A = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A.get("size", "70m"); NS, NF = int(A.get("scr", 20)), int(A.get("fresh", 5))
LOGF = open(os.path.join(HERE, f"where_blank2_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time(); ES = 16
Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz")); L, d = int(Z["L"]), int(Z["d"])
SH = json.load(open(os.path.join(F0, "foundation_share.json")))["sizes"][SIZE]["plates"]
PLATES = [(f"{k}.{l}", l) for l in range(L) for k in ("proj", "mproj")]

def rshift(X, s):
    s = np.asarray(s, dtype=np.int64)
    if (s < 0).any(): assert np.abs(X).max(initial=0) < 2 ** (62 + int(np.minimum(s, 0).min()))
    up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)
def prod(x16, W):
    xf = x16.astype(np.float64); Wf = np.asarray(W).astype(np.float64)
    assert float(np.abs(xf).sum(1).max()) * float(np.abs(Wf).max()) < 2 ** 53
    with np.errstate(all="ignore"): P = xf @ Wf
    assert np.isfinite(P).all() and (P == np.round(P)).all()
    return P
def recover_sh(x16, Wq, e, b, wr):
    P = prod(x16, Wq).astype(np.int64); target = wr - b[None, :]
    nb = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(P), axis=1)])
    sh = np.full(len(P), -1)
    for c in range(0, 60):
        ok = nb + max(int(e.max()) + c, 0) < 62
        if not ok.any(): break
        eq = np.zeros(len(P), bool); eq[ok] = (rshift(P[ok], -(e[None, :] + c)) == target[ok]).all(1); sh[eq & (sh < 0)] = c
    assert (sh >= 0).all(); return sh, P

z = np.load(os.path.join(HERE, f"light_{SIZE}_finished.npz")); WR = z["writes"]; P_ = WR.shape[0]
X16, SHs, FULL = [], [], np.zeros((L, P_, d))
for j, (p, l) in enumerate(PLATES):
    x16 = z[f"light.{p}"].astype(np.int64); e = Z[p + ".e"].astype(np.int64)
    sh, Pw = recover_sh(x16, np.asarray(Z[p + ".w"]), e, Z[p + ".b"].astype(np.int64), WR[:, 1 + j])
    X16.append(x16); SHs.append(sh); FULL[l] += np.ldexp(Pw.astype(np.float64), e[None, :] + sh[:, None])
log(f"THE BLANK TAKEN BACK AS ITSELF OR WITH ITS WRITE, pythia-{SIZE}: {P_} positions ({time.time() - t0:.0f}s, {rss()} MB)")

def blank_layers(kind, seed):                         # the same draws as where_blank.py
    B = np.zeros((L, P_, d))
    for j, (p, l) in enumerate(PLATES):
        W0q, e = np.asarray(Z[p + ".s"]), Z[p + ".e"].astype(np.int64)
        if kind == "scr":
            emin = int(e.min()); Wr = np.left_shift(W0q.astype(np.int64), (e - emin)[None, :])
            Wv, ev = np.random.default_rng([int(SIZE[:-1]), j, 1000 + seed]).permutation(Wr.ravel()).reshape(Wr.shape), np.full(d, emin, np.int64)
        elif kind == "fresh": Wv, ev = fresh(W0q, e, 1, np.random.default_rng([int(SIZE[:-1]), j, 2000 + seed])), e
        else: Wv, ev = W0q, e
        B[l] += SH[p]["s"] * np.ldexp(prod(X16[j], Wv), ev[None, :] + SHs[j][:, None])
    return B
cen = lambda X: X - X.mean(0)
Ww = [cen(FULL[l]) for l in range(L)]
LATw = [cen(WR[:, 3 + 2 * l:].sum(1).astype(np.float64)) for l in range(L - 1)]
WWn = [(w * w).sum(1) for w in Ww]
def split_stats(B):
    out = []
    for l in range(L - 1):
        b = cen(B[l]); w, lat = Ww[l], LATw[l]
        c = (b * w).sum(1) / WWn[l]; bp = b - c[:, None] * w; bb = (b * b).sum()
        wl = (w * lat).sum(1)
        out.append(dict(g=-float((b * lat).sum() / bb), along=-float((c * wl).sum() / bb), perp=-float((bp * lat).sum() / bb),
                        perp_own=-float((bp * lat).sum() / (bp * bp).sum()), lean=float((b * w).sum() / bb),
                        perp_share=float((bp * bp).sum() / bb)))
    return out
REAL = split_stats(blank_layers("real", 0))
SCR = [split_stats(blank_layers("scr", i)) for i in range(NS)]
FR = [split_stats(blank_layers("fresh", i)) for i in range(NF)]
log(f"  {NS} scrambles, {NF} fresh draws ({time.time() - t0:.0f}s, {rss()} MB)")
gwhole = [-float((Ww[l] * LATw[l]).sum() / (Ww[l] * Ww[l]).sum()) for l in range(L - 1)]
log("\n  layer | whole write taken back | blank: lean on its write (scr mean)  perp share of |B|^2 | g = along + perp: real  [scr mean +- spread, z] "
    "| perp's own share taken back: real  [scr mean +- spread, z]  fresh(5)")
RES = dict(size=SIZE, scrambles=NS, fresh=NF, layers=[])
for l in range(L - 1):
    r = REAL[l]; sc = {k: np.array([s[l][k] for s in SCR]) for k in r}; fr = np.array([f[l]["perp_own"] for f in FR])
    zz = {k: float((r[k] - sc[k].mean()) / sc[k].std(ddof=1)) for k in ("g", "along", "perp", "perp_own")}
    RES["layers"].append(dict(layer=l, g_whole=gwhole[l], real=r, scr_mean={k: float(v.mean()) for k, v in sc.items()},
                              scr_sd={k: float(v.std(ddof=1)) for k, v in sc.items()}, z=zz, fresh_perp_own=fr.tolist()))
    log(f"   L{l:<2d}  | {gwhole[l]:+.3f} | lean {r['lean']:+.3f} ({sc['lean'].mean():+.3f})  perp {r['perp_share']:.3f} | "
        f"{r['g']:+.3f} = {r['along']:+.3f} + {r['perp']:+.3f}  [along {sc['along'].mean():+.3f}+-{sc['along'].std(ddof=1):.3f} z {zz['along']:+.1f}; "
        f"perp {sc['perp'].mean():+.3f}+-{sc['perp'].std(ddof=1):.3f} z {zz['perp']:+.1f}] | {r['perp_own']:+.3f} "
        f"[{sc['perp_own'].mean():+.3f}+-{sc['perp_own'].std(ddof=1):.3f} z {zz['perp_own']:+.1f}]  [{' '.join(f'{x:+.2f}' for x in fr)}]")
json.dump(RES, open(os.path.join(HERE, f"where_blank2_{SIZE}.json"), "w"), indent=1)
log(f"\nwrote where_blank2_{SIZE}.json ({time.time() - t0:.0f}s, max {rss()} MB)")
