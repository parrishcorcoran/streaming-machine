"""A LOOK ADDED AFTER blankrow.py: WHERE DO THE BLANK'S WRITES GO, IF NOT INTO THE WORD ROWS? 2026-10-06.

HIS WORDS (2026-10-06), only these are his: "The blank work must be in the embedding table equal in math to the MLP.
It has to cancel." Earlier, also his: "It is zero sum because it must be. It doens have to be in the same spot."

WHY THIS LOOK: blankrow.py part 6 measured the share of the blank's write at a layer that the later writes take back,
g_later = -<B_l, later_l>/|B_l|^2, but the later writes are 30-40 times the blank's write in size, so that share swings
by +-1 to +-10 from scramble to scramble, all of it from the part every position shares (one lane vector, 512 or 768
numbers). This look splits every write into its STANDING part (its average over the 622 positions) and its WORD part
(what is left at each position), and carries 20 scrambles of the blank (each plate's real values re-dealt over the
plate) and 5 fresh bell draws, so the real blank is set against a distribution, not three draws.

On the finished machine's own light (capture.py), the blank's write at layer l, B_l = s x W0 met by the light of both
of layer l's write plates (as in blankrow.py: each position's power of two recovered digit for digit):
  a. the share of B_l's WORD part taken back by every later layer's writes (word parts), g_word, and of its STANDING
     part, g_stand; the same two shares for the BUILT part of the same writes (W - s W0 met by the same light);
  b. the summed blank write B = sum_l B_l against everything else in the top stream (top - B), word and standing;
     the standing part against the biases summed over layers;
  c. the size of the blank's writes so far against the stream each gauge reads: |sum_{l'<l} B_l'|^2 / |R_l|^2, R_l the
     stream entering layer l (word parts and whole), and |B|^2 / |top|^2 at the top;
  d. for context, cos of each layer's whole write (as added, no bias) with the word row, word parts.
Every real share is given with the scrambles' mean and spread, z = (real - mean)/spread, and the 5 fresh draws.

GUESS, written before this script first ran (the session's):
  - The blank's word part is NOT taken back by the later writes: g_word within +-0.1 at every layer and |z| < 3 at most
    layers, while the built part of the same writes is taken back (g_word 0.2-1) at the layers whose word parts
    look_writes found taken back (M2 70m, M3 160m) -- the later reads meet the built directions, not the blank's.
  - Standing parts: the real share inside the scrambles' spread at every layer (that spread is +-1 or more).
  - Everything else in the top stream: the blank's word part g within +-0.05 of the scrambles (|z| < 3).
  - The blank so far is under 1% of the squares of the stream each gauge reads after layer 1, and 0.05-0.2% at the top:
    the blank's writes are not cancelled anywhere; they ride in the stream as a small push the gauge dilutes.

    python3 where_blank.py size=70m
"""
import os, sys, json, time, math, resource
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(HERE), "foundation0")
sys.path.insert(0, HERE)
from common import fresh

A = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A.get("size", "70m"); NS, NF = int(A.get("scr", 20)), int(A.get("fresh", 5))
LOGF = open(os.path.join(HERE, f"where_blank_{SIZE}.log"), "w")
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
log(f"WHERE THE BLANK'S WRITES GO, pythia-{SIZE}: {P_} positions; powers of two recovered ({time.time() - t0:.0f}s, {rss()} MB)")

def blank_layers(kind, seed):
    """B per layer (L, P, d): kind 'real', 'scr' (re-dealt over the plate) or 'fresh'"""
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

top = WR.sum(1).astype(np.float64); row = WR[:, 0].astype(np.float64)
LATER = [WR[:, 3 + 2 * l:].sum(1).astype(np.float64) for l in range(L - 1)]
bias = sum(Z[p + ".b"].astype(np.float64) for p, _ in PLATES)
def split(X): m = X.mean(0); return X - m, m
def g_parts(a, b):                                    # share of a taken back by b: word parts, standing parts
    aw, am = split(a); bw, bm = split(b)
    return -float((aw * bw).sum() / (aw * aw).sum()), -float(am @ bm / (am @ am))

Breal = blank_layers("real", 0); Bsum_real = Breal.sum(0); REST = top - Bsum_real
def stats(B):
    out = {"later_word": [], "later_stand": []}
    for l in range(L - 1):
        gw, gs = g_parts(B[l], LATER[l]); out["later_word"].append(gw); out["later_stand"].append(gs)
    Bs = B.sum(0); out["rest_word"], out["rest_stand"] = g_parts(Bs, REST)
    m = Bs.mean(0); out["bias_stand"] = -float(m @ bias / (m @ m))
    out["row_word"], out["row_stand"] = g_parts(Bs, row)
    return out
REAL = stats(Breal)
SCR = [stats(blank_layers("scr", i)) for i in range(NS)]
log(f"  {NS} scrambles done ({time.time() - t0:.0f}s, {rss()} MB)")
FR = [stats(blank_layers("fresh", i)) for i in range(NF)]
log(f"  {NF} fresh draws done ({time.time() - t0:.0f}s, {rss()} MB)")
BUILT = FULL - Breal
built = {"later_word": [], "later_stand": []}
for l in range(L - 1):
    gw, gs = g_parts(BUILT[l], LATER[l]); built["later_word"].append(gw); built["later_stand"].append(gs)

def zline(key, idx=None):
    r = REAL[key] if idx is None else REAL[key][idx]
    sc = np.array([s[key] if idx is None else s[key][idx] for s in SCR]); fr = np.array([f[key] if idx is None else f[key][idx] for f in FR])
    zz = (r - sc.mean()) / sc.std(ddof=1)
    return r, sc.mean(), sc.std(ddof=1), zz, fr
RES = dict(size=SIZE, positions=P_, scrambles=NS, fresh=NF, per_layer=[])
log("\n== a. the share of each layer's blank write taken back by all later layers' writes ==")
log("   layer | WORD part: real  scr mean +- spread  z  fresh(5)  || built part  | STANDING part: real  scr mean +- spread  z  || built part")
for l in range(L - 1):
    rw, mw, sw, zw, fw = zline("later_word", l); rs, ms, ss, zs, fs = zline("later_stand", l)
    RES["per_layer"].append(dict(layer=l, word=dict(real=rw, scr_mean=mw, scr_sd=sw, z=zw, fresh=fw.tolist(), built=built["later_word"][l]),
                                 stand=dict(real=rs, scr_mean=ms, scr_sd=ss, z=zs, fresh=fs.tolist(), built=built["later_stand"][l])))
    log(f"   L{l:<2d}  | {rw:+.4f}  {mw:+.4f} +- {sw:.4f}  z {zw:+5.1f}  [{' '.join(f'{x:+.3f}' for x in fw)}]  || {built['later_word'][l]:+.4f}  "
        f"| {rs:+.3f}  {ms:+.3f} +- {ss:.3f}  z {zs:+5.1f}  || {built['later_stand'][l]:+.3f}")
log("\n== b. the summed blank write against everything else in the top stream, the word row, and the biases ==")
for key, lab in (("rest_word", "everything else in the top stream, word parts"), ("rest_stand", "everything else in the top stream, standing"),
                 ("row_word", "the word row, word parts"), ("row_stand", "the word row, standing"), ("bias_stand", "the biases summed over layers (standing)")):
    r, m_, s_, zz, fr = zline(key); RES[key] = dict(real=r, scr_mean=m_, scr_sd=s_, z=zz, fresh=fr.tolist())
    log(f"   {lab:48s} share taken back {r:+.4f}   scrambles {m_:+.4f} +- {s_:.4f}   z {zz:+5.1f}   fresh [{' '.join(f'{x:+.3f}' for x in fr)}]")

log("\n== c. how big the blank's writes so far are in the stream each gauge reads (pooled squares) ==")
R = row.copy(); Bso = np.zeros_like(R); sizes = []
for l in range(L + 1):
    Rw, _ = split(R); Bw, _ = split(Bso)
    sizes.append(dict(layer=l, whole=float((Bso ** 2).sum() / (R ** 2).sum()), word=float((Bw ** 2).sum() / (Rw ** 2).sum()) if l else 0.0))
    if l < L: R = R + WR[:, 1 + 2 * l] + WR[:, 2 + 2 * l]; Bso = Bso + Breal[l]
log("   entering layer l (l = L: the top): |blank so far|^2 / |stream|^2, whole and word parts:  " +
    "  ".join(f"L{q['layer']} {q['whole']:.4f}/{q['word']:.4f}" for q in sizes))
RES["sizes"] = sizes
log("\n== d. context: each layer's whole write (as added, no bias) against the word row, word parts: cos ==")
rw_, _ = split(row); cs = []
for l in range(L):
    fw_, _ = split(FULL[l]); bw_, _ = split(Breal[l])
    cs.append(dict(layer=l, whole=float((fw_ * rw_).sum() / math.sqrt((fw_ ** 2).sum() * (rw_ ** 2).sum())),
                   blank=float((bw_ * rw_).sum() / math.sqrt((bw_ ** 2).sum() * (rw_ ** 2).sum()))))
log("   " + "  ".join(f"L{q['layer']} whole {q['whole']:+.3f} blank {q['blank']:+.3f}" for q in cs))
RES["row_cos"] = cs
json.dump(RES, open(os.path.join(HERE, f"where_blank_{SIZE}.json"), "w"), indent=1)
log(f"\nwrote where_blank_{SIZE}.json ({time.time() - t0:.0f}s, max {rss()} MB)")
