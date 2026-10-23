"""STEP 1, THE TOKEN: what of the final light the top 5 needs, read top place first, proved by bounds alone.

HIS WORDS (2026-10-08): "Start backward I think that's your problem. Start on the next token and work backwards".

THE SESSION'S RESTATEMENT (the session's words). The next token is decided by the top 5 of the top read. Read the final
light (x16 after the last water level, every one of its 768 lines) from its top place down, all lines together, sign
and photons as the machine writes them. After the places >= c are read, every line is known only as a box: the read
part K (its places >= c), and below it anything from 0 to 2^c - 1 (with the sign known only where K is not 0). The
gauge (the last water level's power, LS, one number from the whole light) is likewise known only as a range, bounded
from the same boxes. The top 5 (set and order) is PROVED at place c when every word's score interval, and where
needed every pair's difference with the one shared gauge, leaves no other order possible (core.certify). The read
stops at the first (coarsest) c where it is proved. Proved at c means proved for every lower c (the boxes are nested).

WHAT IS RECORDED per position (648 refs positions, 6,000 judge5 positions):
  c5    the coarsest place for the top 5 (set and order), the gauge bounded from the same light;
  c5g   the same with the gauge handed in exactly (the light's own part alone);
  c1    the coarsest place for the race winner alone (for comparison; not the criterion);
  lanes the lines with a read place (|x| >= 2^c5), and the read places and photons;
  work  the plain read's meetings of the read places with every word's row, against the plain read's meetings,
        and on the refs the extra at the top plate (sign-and-1s tips: per place pair, the net count over the lines,
        |.| summed; flat_design/1's sm_excess) in all and in the read places.

PREDICTION (the session's, written before this ran): the read stops at place 2-4 at the median (12-14 of 16 places
read), the gauge's range costs about one place, close calls (a 5th/6th or in-order gap under a few score units) need
place 0, about 5-15% of positions. The read part is about 85-95% of the plain read's meetings when every word meets
it, against an extra of about 3-8% of the plain read at the top plate.

    python3 backward/top.py set=refs|windows      -> backward/out/top_<set>.npz / .log
"""
import sys, time, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/backward")
from core import *
ARG = dict(a.split("=") for a in sys.argv[1:]); SET = ARG.get("set", "refs"); BATCH = int(ARG.get("batch", 64))
LIGHT = ARG.get("light", "real"); ORDER = int(ARG["order"]) if "order" in ARG else None
TAG = f"top_{SET}" + ("" if LIGHT == "real" else f"_{LIGHT}") + ("" if ORDER is None else f"_order{ORDER}")
LOG = open(f"{OUT}/{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

if SET == "refs":
    Z = np.load(f"{HERE}/fringes_machine/refs_top_light.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    X, EX, LS = Z["x16"].astype(np.int64), Z["ex"].astype(np.int64), Z["LS"].astype(np.int64)
    assert np.array_equal(X, REF["x16"]) and np.array_equal(EX, REF["ex"]) and np.array_equal(LS, REF["LS"])
    T5 = REF["top5"]
else:
    Z = np.load(f"{HERE}/fringes_machine/windows_top_light.npz")
    X, EX, LS = Z["x16"].astype(np.int64), Z["ex"].astype(np.int64), Z["LS"].astype(np.int64)
    T5 = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5)
N = len(X)
if LIGHT == "shuffled":                        # CONTROL: each position's light across its lines (topband.py's draw), values kept
    rng = np.random.default_rng(2026); X = np.stack([x[rng.permutation(768)] for x in X])
import core as _core
if ORDER is not None:                          # ORDER-FREE: the lines in another order, the light and the table's rows together
    perm = np.random.default_rng(ORDER).permutation(768); X = X[:, perm]; _core.HEAD_W16 = np.ascontiguousarray(HEAD_W16[perm])
log(f"STEP 1, THE TOKEN: {SET}, {N} positions, light {LIGHT}{'' if ORDER is None else f', lines reordered (seed {ORDER})'}; "
    f"the final light read top place first, the top 5 proved by bounds")

# the machine's own top 5 on this light (check that the light is the machine's)
own = np.zeros((N, 5), np.int64)
for c0 in range(0, N, 256):
    x = X[c0:c0 + 256]; pile = head_product(x)
    S = power(pile, (HE0 + EX[c0:c0 + 256, None] + ES) * 65536 - LS[c0:c0 + 256, None]) + HEAD_B; own[c0:c0 + 256] = top5(S)
log(f"  the plain read on this light gives the saved top 5 at {int((own == T5).all(1).sum())}/{N} (set and order)")
if LIGHT == "shuffled": T5 = own.copy(); log("  (control: the shuffled light's own top 5 is the one to prove)")

def trunc_box(x, c):
    """the light known only by its places >= c: [lo, hi] per line (sign unknown where nothing is read)"""
    a = np.abs(x); K = (a >> c[:, None]) << c[:, None]; u = (np.int64(1) << c[:, None]) - 1; s = np.sign(x)
    lo = np.where(K > 0, np.where(s > 0, K, -(K + u)), -u); hi = np.where(K > 0, np.where(s > 0, K + u, -K), u)
    return lo, hi

def proved(rows, c, gauge_exact):
    """for positions rows, read down to place c (c=0: everything, the machine itself): (top 5 proved, winner proved, cand)"""
    out5 = np.zeros(len(rows), bool); outw = np.zeros(len(rows), bool); cand = np.zeros((len(rows), 5), np.int64)
    z = c == 0; out5[z] = True; outw[z] = True; cand[z] = own[rows[z]]
    nz = np.flatnonzero(~z)
    for b0 in range(0, len(nz), BATCH):
        ii = nz[b0:b0 + BATCH]; r = rows[ii]
        lo, hi = trunc_box(X[r], c[ii])
        if gauge_exact: LSlo = LShi = LS[r]
        else: LSlo, LShi = ls_box(lo, hi, EX[r])
        br = [dict(valid=np.ones(len(r), bool), xlo=lo, xhi=hi, ex=EX[r], LSlo=LSlo, LShi=LShi)]
        cr = certify(br); out5[ii] = cr["cert5"]; outw[ii] = cr["certwin"]; cand[ii] = cr["cand"]
    return out5, outw, cand

def coarsest(gauge_exact, which):
    """the largest c (0..16) proved, by halving the range per position (proved at c means proved below c)"""
    lo = np.zeros(N, np.int64); hi = np.full(N, 17, np.int64); bad = 0                  # proved at lo, not proved at hi
    while (hi - lo > 1).any():
        rows = np.flatnonzero(hi - lo > 1); mid = (lo[rows] + hi[rows]) // 2
        p5, pw, cand = proved(rows, mid, gauge_exact)
        p = p5 if which == 5 else pw
        if which == 5: bad += int((p5 & ~(cand == T5[rows]).all(1)).sum())             # a proof must give the machine's words
        else: bad += int((pw & (cand[:, 0] != T5[rows, 0])).sum())
        lo[rows[p]] = mid[p]; hi[rows[~p]] = mid[~p]
    return lo, bad

c5, b5 = coarsest(False, 5); log(f"  top 5, gauge bounded: done ({time.time() - t0:.0f}s, {rss()} MB); proofs naming other words: {b5}")
c5g, b5g = coarsest(True, 5); log(f"  top 5, gauge given: done ({time.time() - t0:.0f}s); proofs naming other words: {b5g}")
c1, b1 = coarsest(False, 1); log(f"  race winner, gauge bounded: done ({time.time() - t0:.0f}s); proofs naming another word: {b1}")
# monotone check on a sample of positions: proved at every c below the stop, not at the stop + 1
chk = np.arange(0, N, max(1, N // 64)); okm = 0
for c in range(0, 17):
    p5, _, _ = proved(chk, np.full(len(chk), c), False)
    okm += int((p5 == (c <= c5[chk])).sum())
log(f"  every c from 0 to 16 at {len(chk)} positions: proved exactly where c <= the stop at {okm}/{17 * len(chk)}")

ax = np.abs(X); pph = np.array([int(ones(np.asarray(_core.HEAD_W16[i])).sum()) for i in range(768)], np.int64)   # the plain read's 1s per line
plain_meet = ones(X) @ pph
def read_part(c): return ones(ax >> c[:, None]) @ pph
meet5 = read_part(c5); lanes = (ax >= (np.int64(1) << c5[:, None])).sum(1)
places_read = np.maximum(bitlen(ax) - c5[:, None], 0).sum(1); places_all = bitlen(ax).sum(1)
phot_read = ones(ax >> c5[:, None]).sum(1); phot_all = ones(X).sum(1)

def dist(a, name):
    q = np.percentile(a, [10, 25, 50, 75, 90]).astype(int).tolist()
    h = np.bincount(a, minlength=17)
    log(f"  {name}: median {int(np.median(a))}, 10/25/50/75/90% {q}; positions by place " + " ".join(f"{c}:{h[c]}" for c in range(17) if h[c]))
log(f"\nWHERE THE READ STOPS (place c: the light's places c..15 read; c = 0 means every place)")
dist(c5, "top 5, gauge bounded from the light ")
dist(c5g, "top 5, gauge given exactly         ")
dist(c1, "race winner alone, gauge bounded   ")
log(f"  positions needing every place (c = 0): top 5 {int((c5 == 0).sum())}/{N}, winner {int((c1 == 0).sum())}/{N}")
log(f"\nWHAT OF THE FINAL LIGHT WAS NEEDED (top 5, gauge bounded), per position: median / mean")
log(f"  lines with a read place (|x| >= 2^c): {int(np.median(lanes))} / {lanes.mean():.1f} of 768; the rest are needed only as 'below 2^c'")
log(f"  places read (top 1 of each line down to c): {places_read.sum() / places_all.sum():.2%} of the light's places; "
    f"photons read {phot_read.sum() / phot_all.sum():.2%} of its photons")
log(f"  the plain read's meetings of the read part with every word: {meet5.sum() / plain_meet.sum():.2%} of the plain read's meetings")
res = dict(c5=c5, c5g=c5g, c1=c1, lanes=lanes, places_read=places_read, places_all=places_all, phot_read=phot_read,
           phot_all=phot_all, meet5=meet5, plain_meet=plain_meet)
if SET == "refs" and ORDER is None:
    # the extra at the top plate: per place pair (a, b), the net count over the lines, |.| summed (sign and 1s)
    def planes(v, n=16):
        a = np.abs(v); s = np.sign(v)
        return np.stack([(s * ((a >> q) & 1)) for q in range(n)])
    ext_all = np.zeros(N, np.int64); ext_read = np.zeros(N, np.int64); C_all = np.zeros(N, np.int64)
    for j0 in range(0, NW, 2048):
        wb = np.asarray(HEAD_W16[:, j0:j0 + 2048]).astype(np.int64); m = wb.shape[1]
        Bm = planes(wb).astype(np.float32).transpose(1, 0, 2).reshape(768, 16 * m); Ba = np.abs(Bm)
        for p0 in range(0, N, 64):
            xb = X[p0:p0 + 64]; P_ = len(xb); px = planes(xb).astype(np.float32).reshape(16 * P_, 768)
            T = (px @ Bm).reshape(16, P_, 16 * m)                                     # tips (signed), exact in float32
            At = np.abs(T).sum(2, dtype=np.float64).astype(np.int64).T              # (P, a); sums exact in float64
            Cn = (np.abs(px) @ Ba).reshape(16, P_, 16 * m).sum((0, 2), dtype=np.float64).astype(np.int64)
            ext_all[p0:p0 + 64] += At.sum(1); C_all[p0:p0 + 64] += Cn
            cc = c5[p0:p0 + 64]
            ext_read[p0:p0 + 64] += np.where(np.arange(16)[None, :] >= cc[:, None], At, 0).sum(1)
    assert np.array_equal(C_all, plain_meet), "the tips' meetings must be the plain read's meetings"
    res.update(ext_all=ext_all, ext_read=ext_read)
    log(f"\nTHE EXTRA AT THE TOP PLATE (refs): {ext_all.sum() / plain_meet.sum():.2%} of the plain read's meetings "
        f"(per position median {np.median(ext_all / plain_meet):.2%}); in the read part {ext_read.sum() / plain_meet.sum():.2%}")
    log(f"  the read part ({meet5.sum() / plain_meet.sum():.2%} of plain) is {meet5.sum() / ext_all.sum():.1f}x the extra")
np.savez(f"{OUT}/{TAG}.npz", **res)
log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")
