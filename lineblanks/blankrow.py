"""THE BLANK'S WRITES INTO THE STREAM AGAINST THE WORD ROWS: DOES THE WORD TABLE CANCEL THEM? 2026-10-06.

HIS WORDS (2026-10-06), only these are his: "The blank work must be in the embedding table equal in math to the MLP.
It has to cancel. So it's either tiny bit of all the 150k tokens embeddings vs MLP, we must not be the only one to
realize this . Our work of canceling might be doing some of it to begin with. Lining up and cancelling might do most.
Can you line up the blanks and see if it all cancels?"
Earlier, also his: "It is zero sum because it must be. It doens have to be in the same spot. If you run the blank model
ALL that compute is the opposite sides canceling that we are trying to remove."

HIS PICTURE, RESTATED (the session's restatement, not his sentence): the word rows and the MLP's out-rows both write
into the same stream, so they are the same kind of thing. Each finished write plate is the surviving blank s x W0 plus
what was built on it. The blank part still writes into the stream at every position, and that write must be cancelled;
the cancel may sit in the word's own row (the same spot), or spread thin over every word row (a tiny bit of every
row: the table's shared column), and not in the MLP itself.

WHAT IS MEASURED, on the finished whole-number machine (../foundation0/pythia_base2.py, plates=finished) at the 622
positions of the 30 texts, on the machine's own light (capture.py: light_<size>_finished.npz):
 1. THE BLANK'S WRITES. At every position p and layer l, the light entering attention-out (x16 of the arrival's
    output) and the light entering MLP-out (x16 of the bend's output) meet the plate's surviving blank s_l x W0_l
    (s_l from ../foundation0/foundation_share.json, the plate's own s; W0 = the stored start .s on the shared grid):
        B_l(p) = s_l x (x16_p . W0_l) x 2^(e + ex_p + 16)       (stream grains, 2^-16; no bias: the blank's is 0)
    summed over both plates of every layer, B(p) = sum_l B_l(p). Each position's power of two ex_p is not in the
    capture file: it is recovered as the one whole number for which rshift(x16 . W_finished, -(e + ex + 16)) + bias
    equals the write the machine added, digit for digit at every lane (asserted, unique).
    Products are whole-number sums (float64 below 2^53, asserted), times exact powers of two, times s.
 2. AGAINST THE WORD'S OWN INPUT ROW at the same position: the whole row (embed.w) and its built part
    (row - c x blank row, c the shave's c from foundation_check.json: 6.118e-4 at 70m, 1.1803e-2 at 160m; also with
    the input table's own measured s, 0.0156 and 0.0381). For a = B(p), b = the row:
        cos = <a,b>/(|a||b|);   k = -2<a,b>/(|a|^2 + |b|^2)  (k > 0: the row carries the opposite of the blank's write,
        k is the share of |a|^2 + |b|^2 that cancels);   g = -<a,b>/|a|^2 (the share of the blank's write the row
        takes back along it).  Pooled over positions (sums of sums) and per position (median).
    Also cos of the word parts (each side less its average over the 622 positions: what is the word's own).
    Each layer's two plates separately too.
 3. SPREAD THIN OVER ALL WORDS: the blank's writes averaged over all positions, mean_p B(p), against the average
    word row at the 622 positions and against the input table's lane totals (sum over all 50,304 rows; k against the
    table's mean row, the tiny bit every word would carry).
 4. CONTROLS (13.3: each keeps every value and breaks the one pairing that matters):
    - the blank scrambled within each plate (its real values re-dealt over the whole plate), 3 draws: breaks W0's
      arrangement, which is what the rows could have been built against;
    - the word rows matched to the wrong positions (a random derangement), 3 draws: breaks word-to-position;
    - a fresh bell draw with W0's per-neuron spreads, through fp16, on the same grid (common.fresh);
    - for part 3, the table's lane totals in a random lane order (10,000 orders): the range a cosine takes when the
      lanes are not paired.
 5. THE FLOOR: the blank machine's own run (step 0, every plate the blank; light_<size>_start.npz): its writes
    (s = 1) against its own word rows (the blank rows), and its averaged writes against the blank table's lane totals.
    And on the finished light, the blank's writes with the shave's c in place of each plate's s.
 6. WHERE THE BLANK'S WRITES GO (added, a look the task asks for if the rows do not cancel them): at each layer, the
    share of the blank's write that all later layers' writes take back, g_later = -<B_l, later_l>/|B_l|^2 (later_l =
    every write of layers l+1..L-1 as the machine adds it); the same for the built part of the same writes; the
    share the same layer's built write takes back; the share everything else in the top stream takes back,
    -<B, top - B>/|B|^2; the share of |B|^2 that is the lane level (what every gauge takes out); and the standing
    blank write against the biases summed over layers (the one write every position gets alike).
 ORDER-FREE / SECOND PATH: blankrow_check.py recomputes the key numbers by another route (int64 sums, the plate's
 input lanes in a random order, ex found by a size ratio then verified, sums by math.fsum).

RULE 13, CHECKED BEFORE THE RUN
  13.1: the measure is the writes the machine adds into the stream and the rows; nothing counts the 1s of the light.
  13.2: the machine as it is: the light is the finished machine's own; the blank part is s x W0 exactly as
        foundation_share.py defined it; nothing in the machine is changed.
  13.3: cosines, k and g are not kept by the scrambles: each control keeps every value and breaks the pairing.
  13.4: nothing in the definitions fixes the sign or size of <B, row>: the blank's writes are fixed by W0 and the
        light, the rows by the building.
  13.5: his words quoted only above; the restatement is labelled the session's.
  13.6: every layer, every position, every lane; part 6 carries the later writes through to the top stream.

PREDICTIONS (written before this script first ran; results go to blankrow_<size>.log / .json, never here)
  The task's (the session's prediction, handed to this run, copied as given): "at 160m the word rows carry a weak
  opposite of the blank writes (k 0.02-0.15, above every scramble); at 70m about 0 (the plates' blank is smaller
  there); spread thin over all words, the input table's lane totals carry an opposite of the averaged blank writes
  (cos -0.2 to -0.6) -- if so, the cancel he predicts is in the table's lean."
  This run's (also the session's, written now):
  P1. Size: the blank's summed write |B| is 1-5 times the word row's size at a position (the word row is under 0.1% of
      the summed squares of the finished writes, look_writes; the blank part of a write plate is rho 0.02-0.22 of it).
      So the most any row could cancel, 2|a||b|/(|a|^2+|b|^2), is about 0.4-1.
  P2. Same spot: pooled cos(B, row) within +-0.03 at both sizes, inside the range of the scrambles and wrong-position
      draws; k under 0.02 and g under 0.05. Built part the same as the whole row (c x blank row is 1-3% of a row).
  P3. Spread thin: cos(mean B, the table's lane totals) inside the lane-order range (|cos| < 0.12); the table's lean
      is not the opposite of the blank's standing write.
  P4. Floor (the blank machine): pooled cos(B, row) within +-0.05; part 3 inside the lane-order range.
  P5. With c in place of s: |B| falls 150-1000x (70m) and 10-60x (160m), so k under 0.001 (70m) and 0.01 (160m)
      whatever the cosine.
  P6. Where: the later writes take back a share of the blank's write at a layer about equal to the share they take back
      of the built part of the same write (they cannot tell the two apart), largest where look_writes found the word
      parts taken back (M2 at 70m, M3 at 160m: g_later 0.2-0.6), and the scrambled blank ~0. Everything else in the top
      stream takes back g 0.1-0.4 of the summed blank write (the finished writes cancel 26% at 160m, 2% at 70m). The
      same layer's built write: g within +-0.05. The lane level: about 1/d of |B|^2.

    python3 blankrow.py size=70m
"""
import os, sys, json, time, math, resource
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(HERE), "foundation0")
sys.path.insert(0, HERE)
from common import fresh

A = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A.get("size", "70m")
LOGF = open(os.path.join(HERE, f"blankrow_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time(); ES = 16

Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz"))
L, d = int(Z["L"]), int(Z["d"])
SH = json.load(open(os.path.join(F0, "foundation_share.json")))["sizes"][SIZE]["plates"]
C = json.load(open(os.path.join(F0, "foundation_check.json")))[f"{SIZE}@143000"]["embed"]["c_med"]
S_EMB = SH["embed"]["s"]
PLATES = [(f"{k}.{l}", l) for l in range(L) for k in ("proj", "mproj")]      # order: A0 M0 A1 M1 ... (as the writes)
NAMES = [("A" if p.startswith("proj") else "M") + str(l) for p, l in PLATES]

def rshift(X, s):                                     # X x 2^-s, rounded to nearest (pythia_base2.rshift)
    s = np.asarray(s, dtype=np.int64)
    if (s < 0).any(): assert np.abs(X).max(initial=0) < 2 ** (62 + int(np.minimum(s, 0).min()))
    up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)

def prod(x16, W):                                     # whole-number sums of the meetings, exact (float64 below 2^53)
    xf = x16.astype(np.float64); Wf = np.asarray(W).astype(np.float64)
    assert float(np.abs(xf).sum(1).max()) * float(np.abs(Wf).max()) < 2 ** 53
    with np.errstate(all="ignore"): P = xf @ Wf
    assert np.isfinite(P).all() and (P == np.round(P)).all()
    return P

def recover_sh(x16, Wq, e, b, wr):                    # the one ex + 16 for which the machine's own write comes back
    P = prod(x16, Wq).astype(np.int64); target = wr - b[None, :]
    nb = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(P), axis=1)])
    sh = np.full(len(P), -1); nm = np.zeros(len(P), int)
    for c in range(0, 60):
        ok = nb + max(int(e.max()) + c, 0) < 62
        if not ok.any(): break
        R = rshift(P[ok], -(e[None, :] + c))
        eq = np.zeros(len(P), bool); eq[ok] = (R == target[ok]).all(1)
        nm += eq; sh[eq & (sh < 0)] = c
    assert (sh >= 0).all(), "no power of two gives the machine's write back"
    mx = np.abs(x16).max(1)
    assert ((mx >= 2 ** 14) | (sh == 0)).all() and ((sh > 0) | (mx < 2 ** 15)).all(), "to16's own rule broken"
    return sh, int((nm > 1).sum()), P

# ---------------------------------------------------------------------------------------------------------------- data
refs = json.load(open(os.path.join(F0, f"refs_pythia-{SIZE}.json")))[:30]
IDS = np.concatenate([np.array(r["ids"] + r["ref"][:15], dtype=np.int64) for r in refs]); P_ = len(IDS)
EW, ES_, EE = Z["embed.w"], Z["embed.s"], Z["embed.e"].astype(np.int64)
def rows_real(tab, coef=None, other=None):            # rows at the positions, real stream grains (2^-16), float64
    q = tab[IDS].astype(np.float64)
    if coef is not None: q = q - coef * other[IDS].astype(np.float64)
    return np.ldexp(q, (EE[IDS] + ES)[:, None])
ROW = {"whole": rows_real(EW), "built_c": rows_real(EW, C, ES_), "built_s": rows_real(EW, S_EMB, ES_)}
BLANKROW = rows_real(ES_)
emin_t = int(EE.min()); assert EE.max() - emin_t <= 6
def table_total(tab, coef=None, other=None):          # sum over all 50,304 rows, lane by lane, real stream grains
    T = np.zeros(d, np.int64); To = np.zeros(d, np.int64)
    for i0 in range(0, tab.shape[0], 4096):
        sh = (EE[i0:i0 + 4096] - emin_t)[:, None]
        T += np.left_shift(np.asarray(tab[i0:i0 + 4096]).astype(np.int64), sh).sum(0)
        if coef is not None: To += np.left_shift(np.asarray(other[i0:i0 + 4096]).astype(np.int64), sh).sum(0)
    return np.ldexp(T.astype(np.float64) - (0 if coef is None else coef * To.astype(np.float64)), emin_t + ES)
NTAB = EW.shape[0]
TAB = {"whole": table_total(EW), "built_c": table_total(EW, C, ES_), "built_s": table_total(EW, S_EMB, ES_)}
TAB_BLANK = table_total(ES_)
log(f"BLANK'S WRITES AGAINST THE WORD ROWS, pythia-{SIZE}: {P_} positions, {L} layers, d {d}; c (shave) {C:.4e}, "
    f"input table's s {S_EMB:.4f}; plates' s: " + " ".join(f"{n} {SH[p]['s']:.3f}" for (p, _), n in zip(PLATES, NAMES)))

# ------------------------------------------------------------------------------------------------- the blank's writes
def plate_versions(p, rng_seed):
    """the blank plate in its versions, each as (whole-number plate, power per column): real, scr0..2, fresh"""
    W0q, e = np.asarray(Z[p + ".s"]), Z[p + ".e"].astype(np.int64)
    out = {"real": (W0q, e)}
    emin = int(e.min()); Wr = np.left_shift(W0q.astype(np.int64), (e - emin)[None, :])
    for i in range(3):
        rng = np.random.default_rng(rng_seed + [10 + i])
        out[f"scr{i}"] = (rng.permutation(Wr.ravel()).reshape(Wr.shape), np.full(d, emin, np.int64))
    out["fresh"] = (fresh(W0q, e, 1, np.random.default_rng(rng_seed + [20])), e)
    return out
VERS = ["real", "scr0", "scr1", "scr2", "fresh"]

def run_writes(run):
    """the run's light; each position's power of two; the blank's writes per plate (s x W0 for finished, W0 for start)"""
    z = np.load(os.path.join(HERE, f"light_{SIZE}_{run}.npz")); WR = z["writes"]
    assert WR.shape == (P_, 2 * L + 1, d)
    assert np.array_equal(WR[:, 0], rshift((EW if run == "finished" else ES_)[IDS].astype(np.int64), -(EE[IDS] + ES)[:, None])), "ids"
    B = {v: np.zeros((2 * L, P_, d)) for v in VERS}; FULL = np.zeros((2 * L, P_, d)); multi = 0
    for j, (p, l) in enumerate(PLATES):
        x16 = z[f"light.{p}"].astype(np.int64)
        Wq = np.asarray(Z[p + (".w" if run == "finished" else ".s")]); e = Z[p + ".e"].astype(np.int64)
        b = Z[p + (".b" if run == "finished" else ".bs")].astype(np.int64)
        sh, m2, Pw = recover_sh(x16, Wq, e, b, WR[:, 1 + j]); multi += m2
        FULL[j] = np.ldexp(Pw.astype(np.float64), e[None, :] + sh[:, None])        # the plate's write, unrounded, no bias
        sfac = SH[p]["s"] if run == "finished" else 1.0
        for v, (Wv, ev) in plate_versions(p, [int(SIZE[:-1]), j]).items():
            B[v][j] = sfac * np.ldexp(prod(x16, Wv), ev[None, :] + sh[:, None])
    log(f"  {run}: each position's power of two recovered at every plate (writes back digit for digit; positions with "
        f"more than one fit: {multi}); ({time.time() - t0:.0f}s, {rss()} MB)")
    return WR, B, FULL

# ------------------------------------------------------------------------------------------------------------ measures
def pair(a, b):
    """a, b: (P, d). pooled cos, k, g, sizes; per-position medians; the word parts' cos"""
    ab = (a * b).sum(1); aa = (a * a).sum(1); bb = (b * b).sum(1)
    with np.errstate(all="ignore"): cp = ab / np.sqrt(aa * bb); kp = -2 * ab / (aa + bb)
    ac, bc = a - a.mean(0), b - b.mean(0)
    r = float(math.sqrt(aa.sum() / bb.sum()))
    return dict(cos=float(ab.sum() / math.sqrt(aa.sum() * bb.sum())), k=float(-2 * ab.sum() / (aa.sum() + bb.sum())),
                g=float(-ab.sum() / aa.sum()), size=r, ceiling=2 * r / (1 + r * r),
                cos_med=float(np.nanmedian(cp)), k_med=float(np.nanmedian(kp)),
                cos_word=float((ac * bc).sum() / math.sqrt((ac * ac).sum() * (bc * bc).sum())))
def vec(a, b):                                        # two lane vectors: cos, k, g
    ab, aa, bb = float(a @ b), float(a @ a), float(b @ b)
    return dict(cos=ab / math.sqrt(aa * bb), k=-2 * ab / (aa + bb), g=-ab / aa, size=math.sqrt(aa / bb))
def lane_range(a, b, n=10000, seed=7):                # cos of a against b in random lane orders
    rng = np.random.default_rng(seed); bn = b / np.linalg.norm(b); an = a / np.linalg.norm(a)
    cs = np.array([an @ bn[rng.permutation(d)] for _ in range(n)])
    c0 = float(an @ bn)
    return [float(np.quantile(cs, .025)), float(np.quantile(cs, .975))], float((np.abs(cs) >= abs(c0)).mean())
def derange(rng):
    while True:
        p = rng.permutation(P_)
        if (p != np.arange(P_)).all(): return p
PERMS = [derange(np.random.default_rng([int(SIZE[:-1]), 100 + i])) for i in range(3)]
def fmt(m, keys=("cos", "cos_med", "k", "k_med", "g", "size", "ceiling", "cos_word")):
    return "  ".join(f"{k} {m[k]:+.4f}" if k not in ("size", "ceiling") else f"{k} {m[k]:.3f}" for k in keys if k in m)

RES = dict(size=SIZE, positions=P_, c=C, s_embed=S_EMB, s={p: SH[p]["s"] for p, _ in PLATES})

# ============================================================================== finished machine, its own light
WR, B, FULL = run_writes("finished")
Bsum = {v: B[v].sum(0) for v in VERS}
top = WR.sum(1).astype(np.float64)
bias = np.stack([Z[p + ".b"].astype(np.float64) for p, _ in PLATES])
log("\n== 1. THE BLANK'S WRITES (finished light, s x W0 at every write plate) ==")
rms = lambda X: float(math.sqrt((X * X).sum() / X.shape[0]))
sz = dict(B=rms(Bsum["real"]), row=rms(ROW["whole"]), top=rms(top), writes=float(math.sqrt((WR[:, 1:].astype(np.float64) ** 2).sum() / P_)),
          B_scr=[rms(Bsum[f"scr{i}"]) for i in range(3)], B_fresh=rms(Bsum["fresh"]))
log(f"  rms size a position (stream units, grains/2^16): blank's summed write {sz['B'] / 2**16:.3f}; word row {sz['row'] / 2**16:.3f}; "
    f"top stream {sz['top'] / 2**16:.3f}; all writes after the row (root of summed squares) {sz['writes'] / 2**16:.3f}")
log(f"  blank's write / word row {sz['B'] / sz['row']:.2f};  scrambled blank / row {[round(x / sz['row'], 2) for x in sz['B_scr']]};  fresh / row {sz['B_fresh'] / sz['row']:.2f}")
per = []
for j, n in enumerate(NAMES):
    wj = FULL[j]; bj = B["real"][j]
    per.append(dict(name=n, blank_over_write=float(math.sqrt((bj * bj).sum() / (wj * wj).sum())), blank_over_row=rms(bj) / sz["row"],
                    share_of_B2=float((bj * bj).sum() / sum((B['real'][i] ** 2).sum() for i in range(2 * L)))))
log("  each plate, the blank part's size against the plate's whole write on the same light (|s W0 x| / |W x|): " +
    " ".join(f"{q['name']} {q['blank_over_write']:.3f}" for q in per))
log("  each plate, the blank part against the word row's size: " + " ".join(f"{q['name']} {q['blank_over_row']:.2f}" for q in per))
RES["sizes"] = sz; RES["per_plate_size"] = per

log("\n== 2. AT THE SAME POSITION: the blank's summed write against the word's own input row ==")
log("  (cos, k = -2<a,b>/(|a|^2+|b|^2), g = share of the blank's write the row takes back; pooled; _med per position; "
    "cos_word = the parts each position does not share)")
R2 = {}
for rk in ("whole", "built_c", "built_s"):
    b = ROW[rk]; R2[rk] = {}
    R2[rk]["real"] = m = pair(Bsum["real"], b)
    log(f"  row {rk:8s} REAL        {fmt(m)}")
    for i in range(3):
        R2[rk][f"wrongpos{i}"] = m = pair(Bsum["real"], b[PERMS[i]])
        log(f"  row {rk:8s} wrong pos {i}  {fmt(m)}")
    for v in VERS[1:]:
        R2[rk][v] = m = pair(Bsum[v], b)
        log(f"  row {rk:8s} {v:11s} {fmt(m)}")
log("  each plate separately (row whole): cos pooled REAL | scrambles | wrong positions;  g REAL")
R2["per_plate"] = []
for j, n in enumerate(NAMES):
    m = pair(B["real"][j], ROW["whole"]); sc = [pair(B[f"scr{i}"][j], ROW["whole"])["cos"] for i in range(3)]
    wp = [pair(B["real"][j], ROW["whole"][PERMS[i]])["cos"] for i in range(3)]
    R2["per_plate"].append(dict(name=n, real=m, scr=sc, wrongpos=wp))
    log(f"    {n:4s} cos {m['cos']:+.4f} | {' '.join(f'{x:+.4f}' for x in sc)} | {' '.join(f'{x:+.4f}' for x in wp)}   g {m['g']:+.4f}  k {m['k']:+.4f}")
log("  each layer, attention and MLP together (row whole): cos pooled REAL | scrambles;  k;  g")
R2["per_layer"] = []
for l in range(L):
    bl = B["real"][2 * l] + B["real"][2 * l + 1]; m = pair(bl, ROW["whole"])
    sc = [pair(B[f"scr{i}"][2 * l] + B[f"scr{i}"][2 * l + 1], ROW["whole"])["cos"] for i in range(3)]
    R2["per_layer"].append(dict(layer=l, real=m, scr=sc))
    log(f"    layer {l:2d}  cos {m['cos']:+.4f} | {' '.join(f'{x:+.4f}' for x in sc)}   k {m['k']:+.4f}  g {m['g']:+.4f}")
RES["same_spot"] = R2

log("\n== 3. SPREAD THIN OVER ALL WORDS: the blank's write averaged over the positions ==")
R3 = {}
mB = {v: Bsum[v].mean(0) for v in VERS}
targets = {"mean row at the positions (whole)": ROW["whole"].mean(0), "mean row at the positions (built_c)": ROW["built_c"].mean(0),
           "table lane totals (whole)": TAB["whole"], "table lane totals (built_c)": TAB["built_c"], "table lane totals (built_s)": TAB["built_s"]}
for tn, T in targets.items():
    Tm = T / NTAB if tn.startswith("table") else T
    rngc, pv = lane_range(mB["real"], T)
    m = vec(mB["real"], Tm); ctl = {v: vec(mB[v], Tm)["cos"] for v in VERS[1:]}
    R3[tn] = dict(real=m, lane_order_95=rngc, lane_order_p=pv, controls=ctl)
    log(f"  {tn:36s} cos {m['cos']:+.4f}  [lane orders 95%: {rngc[0]:+.4f} {rngc[1]:+.4f}; share as far out {pv:.3f}]  "
        f"k {m['k']:+.5f}  g {m['g']:+.5f}  |mean B|/|it| {m['size']:.3f} | scr {ctl['scr0']:+.4f} {ctl['scr1']:+.4f} {ctl['scr2']:+.4f} fresh {ctl['fresh']:+.4f}")
log("  each layer's averaged blank write against the table's lane totals (whole): cos [lane orders 95%]")
R3["per_layer"] = []
for l in range(L):
    ml = (B["real"][2 * l] + B["real"][2 * l + 1]).mean(0); c0 = vec(ml, TAB["whole"])["cos"]; rr, pv = lane_range(ml, TAB["whole"], 2000, 8 + l)
    R3["per_layer"].append(dict(layer=l, cos=c0, lane_order_95=rr, p=pv))
log("    " + "  ".join(f"L{q['layer']} {q['cos']:+.3f} [{q['lane_order_95'][0]:+.3f},{q['lane_order_95'][1]:+.3f}]" for q in R3["per_layer"]))
mBias = bias.sum(0); mRest = (top - Bsum["real"]).mean(0)
for tn, T in (("the biases summed over layers", mBias), ("everything else in the top stream, averaged", mRest),
              ("the built writes (attention + MLP, no bias), averaged", (FULL.sum(0) - Bsum["real"]).mean(0))):
    m = vec(mB["real"], T); rr, pv = lane_range(mB["real"], T); ctl = {v: vec(mB[v], T)["g"] for v in VERS[1:]}
    R3[tn] = dict(real=m, lane_order_95=rr, p=pv, controls_g=ctl)
    log(f"  vs {tn:52s} cos {m['cos']:+.4f} [{rr[0]:+.4f},{rr[1]:+.4f}] g {m['g']:+.4f} | g scr {ctl['scr0']:+.4f} {ctl['scr1']:+.4f} {ctl['scr2']:+.4f} fresh {ctl['fresh']:+.4f}")
log(f"  the table's lean: |table mean row| / rms word row at the positions {np.linalg.norm(TAB['whole'] / NTAB) / sz['row']:.3f};  "
    f"|mean B| / rms B {np.linalg.norm(mB['real']) / sz['B']:.3f}")
RES["spread_thin"] = R3

log("\n== 6. WHERE THE BLANK'S WRITES GO (finished machine) ==")
R6 = {"per_layer": []}
log("  layer: share taken back by all later layers' writes  g_later: blank REAL | scrambles | fresh || built part of the same writes || whole write;  "
    "same layer's built write takes back g_same")
for l in range(L):
    later = WR[:, 3 + 2 * l:].sum(1).astype(np.float64) if l < L - 1 else np.zeros((P_, d))
    bl = {v: B[v][2 * l] + B[v][2 * l + 1] for v in VERS}; full = FULL[2 * l] + FULL[2 * l + 1]; built = full - bl["real"]
    if l < L - 1:
        g = {v: pair(bl[v], later)["g"] for v in VERS}; gb = pair(built, later)["g"]; gf = pair(full, later)["g"]
    else: g = {v: 0.0 for v in VERS}; gb = gf = 0.0
    gs = pair(bl["real"], built)["g"]; gs_scr = [pair(bl[f"scr{i}"], built)["g"] for i in range(3)]
    R6["per_layer"].append(dict(layer=l, g_later=g, g_later_built=gb, g_later_whole=gf, g_same=gs, g_same_scr=gs_scr))
    log(f"    layer {l:2d}  g_later {g['real']:+.4f} | {g['scr0']:+.4f} {g['scr1']:+.4f} {g['scr2']:+.4f} | {g['fresh']:+.4f} || {gb:+.4f} || {gf:+.4f};   "
        f"g_same {gs:+.4f} (scr {' '.join(f'{x:+.4f}' for x in gs_scr)})")
rest = top - Bsum["real"]
g_top = {v: pair(Bsum[v], rest)["g"] for v in VERS}
lev = {v: float((Bsum[v].mean(1) ** 2).sum() * d / (Bsum[v] ** 2).sum()) for v in VERS}
keep = float((top * top).sum() / ((rest * rest).sum() + (Bsum["real"] ** 2).sum()))
R6.update(g_top=g_top, lane_level_share=lev, top_over_parts=keep)
log(f"  everything else in the top stream takes back g {g_top['real']:+.4f} of the blank's summed write (scr {g_top['scr0']:+.4f} {g_top['scr1']:+.4f} {g_top['scr2']:+.4f}, fresh {g_top['fresh']:+.4f})")
log(f"  |top|^2 / (|top - B|^2 + |B|^2) = {keep:.4f}  (1: they add like coin tosses)")
log(f"  share of |B|^2 in the lane level (what every gauge takes out; 1/d = {1 / d:.5f}): real {lev['real']:.5f}  scr {lev['scr0']:.5f} {lev['scr1']:.5f} {lev['scr2']:.5f}  fresh {lev['fresh']:.5f}")
RES["where"] = R6

# with the shave's c in place of each plate's s, on the finished light
log("\n== 5b. THE BLANK'S WRITES WITH THE SHAVE'S c IN PLACE OF EACH PLATE'S s (finished light) ==")
R5 = {}
Bc = {v: sum(B[v][j] * (C / SH[p]["s"]) for j, (p, _) in enumerate(PLATES)) for v in VERS}
for rk in ("whole", "built_c"):
    R5[rk] = {"real": pair(Bc["real"], ROW[rk])}
    for i in range(3): R5[rk][f"wrongpos{i}"] = pair(Bc["real"], ROW[rk][PERMS[i]])
    for v in VERS[1:]: R5[rk][v] = pair(Bc[v], ROW[rk])
    for kk, m in R5[rk].items(): log(f"  row {rk:8s} {kk:11s} {fmt(m)}")
m = vec(Bc["real"].mean(0), TAB["whole"] / NTAB); rr, pv = lane_range(Bc["real"].mean(0), TAB["whole"])
R5["spread_thin"] = dict(real=m, lane_order_95=rr, p=pv)
log(f"  averaged, against the table's lane totals: cos {m['cos']:+.4f} [{rr[0]:+.4f},{rr[1]:+.4f}]  k {m['k']:+.6f}  g {m['g']:+.4f}  |mean Bc|/|table mean row| {m['size']:.4f}")
RES["c_in_place_of_s"] = R5
del B, Bsum, FULL, WR, Bc

# ============================================================================== the floor: the blank machine itself
log("\n== 5a. THE FLOOR: the blank machine's own run (every plate the blank, s = 1), its own word rows (the blank rows) ==")
WRs, Bs, FULLs = run_writes("start")
Bss = {v: Bs[v].sum(0) for v in VERS}
dev = float(np.abs(Bss["real"] - WRs[:, 1:].sum(1)).max())
log(f"  check: the blank's writes as computed here against the machine's own writes (biases 0): largest gap {dev:.1f} grains "
    f"(rounding only; at most {L} x 2 x 0.5 = {L} expected)")
R5a = {"max_gap_grains": dev, "size_B_over_row": rms(Bss["real"]) / rms(BLANKROW)}
log(f"  rms blank write / rms blank row {R5a['size_B_over_row']:.2f}")
R5a["real"] = m = pair(Bss["real"], BLANKROW); log(f"  REAL        {fmt(m)}")
for i in range(3): R5a[f"wrongpos{i}"] = m = pair(Bss["real"], BLANKROW[PERMS[i]]); log(f"  wrong pos {i}  {fmt(m)}")
for v in VERS[1:]: R5a[v] = m = pair(Bss[v], BLANKROW); log(f"  {v:11s} {fmt(m)}")
R5a["per_layer"] = [pair(Bs["real"][2 * l] + Bs["real"][2 * l + 1], BLANKROW)["cos"] for l in range(L)]
log("  each layer: cos " + " ".join(f"L{l} {c:+.4f}" for l, c in enumerate(R5a["per_layer"])))
mb0 = Bss["real"].mean(0)
for tn, T in (("blank table's lane totals", TAB_BLANK), ("mean blank row at the positions", BLANKROW.mean(0))):
    Tm = T / NTAB if tn.startswith("blank table") else T
    m = vec(mb0, Tm); rr, pv = lane_range(mb0, T); ctl = {v: vec(Bss[v].mean(0), Tm)["cos"] for v in VERS[1:]}
    R5a[tn] = dict(real=m, lane_order_95=rr, p=pv, controls=ctl)
    log(f"  averaged, vs {tn:32s} cos {m['cos']:+.4f} [{rr[0]:+.4f},{rr[1]:+.4f}] k {m['k']:+.5f} g {m['g']:+.5f} | scr {ctl['scr0']:+.4f} {ctl['scr1']:+.4f} {ctl['scr2']:+.4f} fresh {ctl['fresh']:+.4f}")
R5a["g_later"] = []
for l in range(L - 1):
    later = WRs[:, 3 + 2 * l:].sum(1).astype(np.float64)
    R5a["g_later"].append({v: pair(Bs[v][2 * l] + Bs[v][2 * l + 1], later)["g"] for v in ("real", "scr0", "fresh")})
log("  share of each layer's write the later layers take back, g_later real (scr0, fresh): " +
    " ".join(f"L{l} {q['real']:+.3f} ({q['scr0']:+.3f},{q['fresh']:+.3f})" for l, q in enumerate(R5a["g_later"])))
tops = WRs.sum(1).astype(np.float64); rests = tops - Bss["real"]
R5a["g_top"] = pair(Bss["real"], rests)["g"]; R5a["top_over_parts"] = float((tops * tops).sum() / ((rests * rests).sum() + (Bss["real"] ** 2).sum()))
log(f"  the word row takes back g {R5a['g_top']:+.4f} of the blank machine's summed write; |top|^2 / (|row|^2 + |writes|^2) {R5a['top_over_parts']:.4f}")
RES["floor"] = R5a

json.dump(RES, open(os.path.join(HERE, f"blankrow_{SIZE}.json"), "w"), indent=1)
log(f"\nwrote blankrow_{SIZE}.json  ({time.time() - t0:.0f}s, max {rss()} MB)")
