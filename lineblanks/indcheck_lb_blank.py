"""INDEPENDENT CHECKER (a separate session's own code, rule 10) of blankrow.py's k and lane-total cosine at 70m. 2026-10-06.

HIS WORDS (2026-10-06), only these are his: "The blank work must be in the embedding table equal in math to the MLP.
It has to cancel."

ROUTE (different from blankrow.py and blankrow_check.py, which recover each position's power of two from the write):
the whole-number machine (../foundation0/pythia_base2.py, plates=finished, loaded as it is, no edits) is run here on the
same 622 positions, and each position's power of two at attention-out and MLP-out is taken straight from the machine's
own to16 as the plate meets the light. Checks: that light equals light_70m_finished.npz digit for digit, and the
machine's writes equal the stored writes.
  B(p) = sum over the 12 write plates of s_plate x (x16_p . W0) x 2^(e + ex_p + 16)   (stream grains, 2^-16)
  s_plate from ../foundation0/foundation_share.json, and recomputed here as (W . W0) / (W0 . W0) on real values.
  The word row: embed.w at the position's word, x 2^(e + 16).
  SAME SPOT: pooled cos(B, row) and k = -2<B,row> / (|B|^2 + |row|^2) over the 622 positions.
  SPREAD THIN: cos(mean_p B, the input table's lane totals over all 50,304 rows), with the range a cosine takes over
  10,000 random lane orders (my seed), and k of mean_p B against the table's mean row.
  ONE SCRAMBLE: every write plate's blank values re-dealt over the whole plate (my seed), the same two numbers.
  And one wrong-position draw (word rows matched to a derangement of the positions) for the same-spot numbers.

PREDICTION (written before this script's first run): blankrow_70m.log's REAL numbers come back to 4 decimals:
same spot cos +0.0218, k -0.0091; spread thin cos +0.0594 inside a lane-order range near [-0.09, +0.087], k -0.00328.
My scramble: |k| under 0.01 and |cos| under 0.03 at the same spot; its lane-total cos inside the lane-order range.
Every light and write from this run equal to the stored ones.

    python3 indcheck_lb_blank.py
"""
import os, sys, json, math, time, importlib.util, resource
import numpy as np

SIZE = "70m"; ES = 16
HERE = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(HERE), "foundation0")
LOGF = open(os.path.join(HERE, f"indcheck_lb_blank_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time()

argv = sys.argv; sys.argv = ["pythia_base2.py", f"size={SIZE}", "plates=finished"]
spec = importlib.util.spec_from_file_location("pythia_base2_ind", os.path.join(F0, "pythia_base2.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.argv = argv
assert m.PLATES == "finished" and (m.PERM == np.arange(m.d)).all()
L, d = m.L, m.d
Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz"))
refs = json.load(open(os.path.join(F0, f"refs_pythia-{SIZE}.json")))[:30]

# ------------------------------------------------------------------ run the machine, keeping each plate's light and power
X16 = {}; EX = {}; WR = []; IDS = []
for r in refs:
    ids = np.asarray(r["ids"] + r["ref"][:15]); IDS.append(ids)
    R = m.rshift(m.pick(m.EMB_F[ids], m.EMB_S[ids]), -(m.EMB_E[ids][:, None] + ES)); ws = [R.copy()]
    for l in range(L):
        Ao = m.attention(m.plain(m.gauge(R, f"ln1.{l}"), f"qkv.{l}")); At = m.plain(Ao, f"proj.{l}")
        Eo = m.edge(m.plain(m.gauge(R, f"ln2.{l}"), f"fc.{l}")); Mo = m.plain(Eo, f"mproj.{l}")
        for nm, val in ((f"proj.{l}", Ao), (f"mproj.{l}", Eo)):
            x16, ex = m.to16(val); X16.setdefault(nm, []).append(x16); EX.setdefault(nm, []).append(ex)
        ws += [At, Mo]; R = R + At + Mo
    WR.append(np.stack(ws, 1))
IDS = np.concatenate(IDS); WR = np.concatenate(WR); P_ = len(IDS)
X16 = {k: np.concatenate(v) for k, v in X16.items()}; EX = {k: np.concatenate(v) for k, v in EX.items()}
zf = np.load(os.path.join(HERE, f"light_{SIZE}_finished.npz"))
same_light = all(np.array_equal(X16[k], zf["light." + k]) for k in X16); same_writes = bool(np.array_equal(WR, zf["writes"]))
log(f"BLANK'S WRITES AGAINST THE WORD ROWS, independent route, pythia-{SIZE}: {P_} positions; machine run here "
    f"({time.time() - t0:.0f}s, {rss()} MB)")
log(f"  light at every write plate equal to light_{SIZE}_finished.npz: {same_light};  every write equal to the stored writes: {same_writes}")
log(f"  each position's power of two at the write plates (ex + 16), range: " + " ".join(f"{k} {int(EX[k].min()) + 16}..{int(EX[k].max()) + 16}" for k in EX))

# ------------------------------------------------------------------ s per plate, stored and recomputed
SH = json.load(open(os.path.join(F0, "foundation_share.json")))["sizes"][SIZE]["plates"]
PL = [f"{k}.{l}" for l in range(L) for k in ("proj", "mproj")]
s_mine = {}
for p in PL:
    e = Z[p + ".e"].astype(np.int64)
    Wr = np.ldexp(np.asarray(Z[p + ".w"]).astype(np.float64), e[None, :]); W0r = np.ldexp(np.asarray(Z[p + ".s"]).astype(np.float64), e[None, :])
    s_mine[p] = float((Wr * W0r).sum() / (W0r * W0r).sum())
log("  s per plate, stored / recomputed: " + "  ".join(f"{p} {SH[p]['s']:.4f}/{s_mine[p]:.4f}" for p in PL))

def blank_sum(scramble_seed=None):
    B = np.zeros((P_, d))
    for j, p in enumerate(PL):
        q = np.asarray(Z[p + ".s"]).astype(np.int64); e = Z[p + ".e"].astype(np.int64)
        if scramble_seed is not None:
            emin = int(e.min()); qr = q * (2 ** (e - emin))[None, :]
            q = np.random.default_rng([scramble_seed, j]).permutation(qr.ravel()).reshape(qr.shape); e = np.full(d, emin, np.int64)
        xf = X16[p].astype(np.float64); assert float(np.abs(xf).sum(1).max()) * float(np.abs(q).max()) < 2 ** 53
        with np.errstate(all="ignore"): prod = xf @ q.astype(np.float64)
        assert (prod == np.round(prod)).all()
        B += SH[p]["s"] * np.ldexp(prod, e[None, :] + (EX[p] + 16)[:, None])
    return B
EE = Z["embed.e"].astype(np.int64); EW = Z["embed.w"]
ROW = np.ldexp(EW[IDS].astype(np.float64), (EE[IDS] + ES)[:, None])
TAB = np.zeros(d)
for i0 in range(0, EW.shape[0], 4096):
    TAB += np.ldexp(np.asarray(EW[i0:i0 + 4096]).astype(np.float64), (EE[i0:i0 + 4096] + ES)[:, None]).sum(0)
NT = EW.shape[0]

def same_spot(a, b):
    ab, aa, bb = float((a * b).sum()), float((a * a).sum()), float((b * b).sum())
    return dict(cos=ab / math.sqrt(aa * bb), k=-2 * ab / (aa + bb), g=-ab / aa, size=math.sqrt(aa / bb))
def spread_thin(mB, T, n=10000, seed=31):
    c = float(mB @ T / (np.linalg.norm(mB) * np.linalg.norm(T))); Tm = T / NT
    k = -2 * float(mB @ Tm) / float(mB @ mB + Tm @ Tm)
    rng = np.random.default_rng(seed); un = mB / np.linalg.norm(mB); tn = T / np.linalg.norm(T)
    cs = np.array([un @ tn[rng.permutation(d)] for _ in range(n)])
    return dict(cos=c, k=k, range=[float(np.quantile(cs, .025)), float(np.quantile(cs, .975))], share_as_far=float((np.abs(cs) >= abs(c)).mean()))

RES = {}
Breal = blank_sum(); Bscr = blank_sum(90210)
rng = np.random.default_rng(777)
while True:
    der = rng.permutation(P_)
    if (der != np.arange(P_)).all(): break
RES["real"] = same_spot(Breal, ROW); RES["scramble"] = same_spot(Bscr, ROW); RES["wrongpos"] = same_spot(Breal, ROW[der])
RES["thin_real"] = spread_thin(Breal.mean(0), TAB); RES["thin_scramble"] = spread_thin(Bscr.mean(0), TAB)
ref = json.load(open(os.path.join(HERE, f"blankrow_{SIZE}.json")))
log("\n  SAME SPOT (pooled over the 622 positions), blank's summed write against the word's own input row")
for k_, lab in (("real", "REAL"), ("scramble", "my scramble"), ("wrongpos", "my wrong positions")):
    q = RES[k_]; log(f"    {lab:20s} cos {q['cos']:+.5f}  k {q['k']:+.5f}  g {q['g']:+.5f}  |B|/|row| {q['size']:.3f}")
st = ref["same_spot"]["whole"]
log(f"    stored REAL          cos {st['real']['cos']:+.5f}  k {st['real']['k']:+.5f}  g {st['real']['g']:+.5f}  |B|/|row| {st['real']['size']:.3f};  stored scr0 cos {st['scr0']['cos']:+.5f} k {st['scr0']['k']:+.5f}")
log("\n  SPREAD THIN: the blank's write averaged over the positions against the input table's lane totals")
for k_, lab in (("thin_real", "REAL"), ("thin_scramble", "my scramble")):
    q = RES[k_]; log(f"    {lab:12s} cos {q['cos']:+.5f}  [my lane orders 95%: {q['range'][0]:+.4f} {q['range'][1]:+.4f}; share as far out {q['share_as_far']:.3f}]  k {q['k']:+.6f}")
st3 = ref["spread_thin"]["table lane totals (whole)"]
log(f"    stored REAL  cos {st3['real']['cos']:+.5f}  [{st3['lane_order_95'][0]:+.4f} {st3['lane_order_95'][1]:+.4f}; share {st3['lane_order_p']:.3f}]  k {st3['real']['k']:+.6f};  stored scr0 cos {st3['controls']['scr0']:+.5f}")
RES["light_equal"] = same_light; RES["writes_equal"] = same_writes; RES["s_mine"] = s_mine
json.dump(RES, open(os.path.join(HERE, f"indcheck_lb_blank_{SIZE}.json"), "w"), indent=1)
log(f"\ndone ({time.time() - t0:.0f}s, max memory {rss()} MB)")
