"""SHAREDCOL/XCHECK.PY: AN INDEPENDENT REBUILD OF THE SHARED-COLUMN MEASURE AND CUT (measure.py, cut.py), WRITTEN BY A
SEPARATE CHECKER FROM THE DEFINITIONS, NOT FROM THEIR CODE. 2026-10-06. Whole numbers in the machine path.

HIS PICTURE, AS THE SESSION PUT IT (the session's words, not his sentence; nothing here is quoted from him): the top
table is one standing pattern shared by all words plus each word's own fringe. The light meets the standing pattern
once a position, which adds the same amount to every word, so only the fringes decide the race and the top 5.

WHAT THIS FILE REBUILDS (each phase its own process, one at a time)
  phase=weights   Pythia-70m's head (finished head.w, each word on its own power head.e) and GPT-2's T (head.w, one
                  power 2^-12). The columns rebuilt from measure.py's stated definitions with Python fractions:
                    mean    each lane's mean over the words on the finest grid, rounded (half up) to the coarsest grid
                    median  each lane's lower middle value on the finest grid, rounded (half up) to the coarsest grid
                    mean11  the mean column with 11 significant places (half up), as cut.py writes it
                  share of the summed squares (exact whole-number sums), same-sign lanes, |c| / median word column,
                  fringe / its own word column, lit places a weight in plain 1s and signed digits (both counted by a
                  digit-by-digit loop, not by popcount), each word's top place, GPT-2's column lanes.
                  SHUFFLE CONTROL: each word's values dealt to the lanes in a random order of its own (every value and
                  every word's power kept; which lane a value stands in is broken). ORDER-FREE: lanes and words
                  relabelled, every number the same.
  phase=gpt2      GPT-2's machine (base2.py, loaded as judge5.py loads it, nothing changed) on judge5 windows 0-2,
                  judged positions 3..22: the first 50. Plain read: power(product(x16, w), F) + b, the race; top 5 =
                  the race run five times (high first, then the lower word number); checked against judge5_ref.npz.
                  THE CUT FOR GPT-2 IS BUILT HERE (the session's rule did not cut GPT-2): the mean column, fringes
                  U = q - c (one grid). With S = x16.c (once) and F_w = x16.U_w, the machine's own score is
                      score_w = floor((S m + F_w m + 2^(s-1)) / 2^s) + b_w = floor((A + z_w) / 2^s),
                      z_w = F_w m + b_w 2^s,  A = S m + 2^(s-1)  (the same for every word),  m, s the position's own
                  so the race follows z. K = words with z_w > z_5 - 2^s; S is added (to K only) only where two words of
                  K stand less than one step apart.
  phase=pythia    size=160m (50 positions) or 70m (10 positions). The Pythia machine (foundation0/pythia_base2.py,
                  loaded unchanged) on refs texts 0-2, prompt + 15 fed words. Its own light y16, ey, checked against
                  the session's cutlight_<size>.npz and its head piles against the module's own forward(). Plain read:
                  rshift(y16.q_w, -(e_w + ey + 16)) by int64 sums. Cut: the mean11 column on the coarsest grid, fringes
                  u_w = q_w - (c << (emax - e_w)), z_w = F_w << (e_w - emin), S'' = S << (emax - emin),
                  s = -(emin + ey + 16), score_w = floor((z_w + S'' + 2^(s-1)) / 2^s); the same K rule.
  MEETINGS        at 10 positions of each machine: plain = the sum over every (lane, word) of ones(y_l) x ones(q_lw),
                  counted word block by word block (not the lane sums cut.py uses); cut = the same over the fringes,
                  plus ones(y_l) x ones(c_l) once, plus S's photons into each word of K where it was added. Signed
                  digits the same. For Pythia, then the session's own light file over all 622 positions by lane sums,
                  to compare with cut_<size>.json's totals digit for digit.
  CONTROLS        light shuffled across its lanes (count); the column dealt to other lanes (the read must stay exact,
                  the saving must go); ORDER-FREE for the read: lanes and words relabelled, every pile the same whole
                  number, the top 5 the same words.

RULE 13, CHECKED BEFORE THE RUN
  13.1 the measure is the finished table and the top read's work and its top 5 and race; the light's 1s only weight the
       meetings. 13.2 every machine loaded unchanged; the table only rewritten as column + fringes, digit for digit.
  13.3 the shuffle control keeps every value and breaks which lane each value stands in, so it CAN move the share and
       the lit places; exactness of the cut is an identity for any column, so it is not evidence for the pattern (the
       column-in-other-lanes control shows this). 13.4 the mean column takes the largest share of any column (definition);
       exactness is fixed by construction (u = q - c); the finding is the count. 13.5 his words are not quoted.
  13.6 every word, every lane, every position chosen; nothing stopped early.

PREDICTIONS (written before any run of this file; the session's reported numbers are the targets)
  weights, Pythia-70m: mean share 91.69%; median lane same sign 99.91%; |c| / median word column 0.956; fringe / its word
    column (median) 0.276; signed lit places: table 4.331 a weight, fewer by 1.89% (mean), 16.61% (median), 16.44%
    (mean11); plain 1s 5.774, fewer by 3.80%, 20.24%, 20.05%; top place 15.000 -> 14.706. Shuffle control: share under
    3%, median lane same sign under 60%, the lit-place saving gone (between -3% and +1%). Order-free: identical.
  weights, GPT-2: share 82.65%; median lane 56.83%; fringe / word 0.410; signed table 3.816, fewer 2.19% (mean), 2.29%
    (median); plain 1s 4.907, fewer 2.47%, 2.59%; top place 15.000 -> 12.073; the column's lanes 36, 430, 496, 314, 6 with
    18.4, 18.3, 12.4, 8.2, 7.8% of its squares, the largest 7 74.6%. Shuffle control: share under 0.1%. Order-free: same.
  gpt2: the plain top 5 = judge5_ref at 50/50 (exact ties aside); cut top 5 (set and order) and winner 50/50; every pile
    F + S = the plain pile; K = 5 at nearly every position; S added at 0-3 positions. Meetings at 10 positions: cut
    96-99.5% of plain (plain 1s), 97-99.5% (signed). Colshuffle: exact, 99-103% of plain. Order-free: piles the same.
  pythia 160m: light identical to cutlight_160m.npz at all 50 positions; plain piles = the module's own; cut top 5 and
    winner 50/50; K = 5, S added at 0 positions. At 10 positions: cut 80.5-83.5% of plain (plain 1s), 83.5-86.5%
    (signed). Over the session's 622 positions: plain, cut and shared counts equal cut_160m.json's digit for digit
    (82.03%, 84.93%). pythia 70m at 10 positions: 78-81% (plain 1s), 82-84.5% (signed); 622 totals equal cut_70m.json's
    (79.54%, 83.26%). Light shuffled: within 1.5 points of the real light. Colshuffle: exact, 97-101% of plain.
    Order-free: every pile the same, top 5 the same.

RESULTS (2026-10-06; xcheck_weights.log, xcheck_gpt2.log, xcheck_pythia_160m.log, xcheck_pythia_70m.log)
  weights, Pythia-70m: mean share 91.6894%; median lane 99.907%; |c| / median word 0.9557; fringe / word 0.2763; signed
    table 4.3309, fewer 1.89% / 16.61% / 16.44%; plain 1s 5.7743, fewer 3.80% / 20.24% / 20.05%; top place 15.000 ->
    14.706. The columns' lanes match lane for lane under the order-free relabelling.                     -> held, every digit
  weights, GPT-2: share 82.6471%; median lane 56.832%; fringe / word 0.4103; signed 3.8156, fewer 2.19% / 2.29%; plain
    1s 4.9067, fewer 2.47% / 2.59%; top place 12.073; lanes 36, 430, 496, 314, 6 with 18.44, 18.30, 12.38, 8.20, 7.81%,
    the largest 7 74.57%, the largest 10% 98.35%.                                                         -> held, every digit
  shuffle control (each word's values dealt to lanes at random): share 0.0019% (70m), 0.0018% (GPT-2); median lane 50.3%
    / 52.9%                                                                                               -> held
    lit places: GPT-2 no change (+0.02% / 0.00%), held; Pythia-70m 19.25% MORE plain 1s and 17.24% MORE signed digits
    with the mean or mean11 column of the shuffled table, FAILED my -3..+1% (the column there is a few grains; taking it
    away lights the dark low places of the fp16-written values, the session's road block 2, larger than its blank's
    +2.37%). The saving is gone, and worse.
  order-free (weights): no number differs                                                                 -> held
  gpt2 (cut built here): plain top 5 = judge5_ref 50/50, winner 50/50; pile identity 50/50; every word's score from
    fringe + S = the machine's 50/50; cut top 5 50/50, winner 50/50; K = 5 at all 50; S added at 0; 0 words outside K
    within a step                                                                                         -> held
    meetings at 10 positions: 97.21% of plain (plain 1s), 97.56% (signed); lane-sum form = block form     -> held
    colshuffle exact 50/50, 102.66% / 102.13% (held); light shuffled 97.52% / 97.82%; order-free 50/50     -> held
    largest process 3.08 GB, just OVER the 3 GB rule (four int32 tables beside the machine's float64 plates).
  pythia 160m: light = cutlight_160m.npz at 50/50, plain piles = the module's forward(); identity and every score
    50/50; cut top 5 and winner 50/50; K = 5, S added at 0                                                -> held
    meetings at 10 positions: 82.04% (plain 1s), 84.92% (signed); 622 positions: plain1, cut1, shared1, plainn, cutn,
    sharedn all EQUAL to cut_160m.json (82.03%, 84.93%)                                                    -> held
    colshuffle exact, 99.45% / 98.10% (held); light shuffled 82.16% / 85.07% (held); order-free 50/50 (held). 2.67 GB.
  pythia 70m (50 positions): exact 50/50 the same way; 10 positions 79.52% / 83.27%; 622 totals all EQUAL to
    cut_70m.json (79.54%, 83.26%); colshuffle 99.65% / 98.71%; light shuffled 79.97% / 83.54%.              -> held. 1.80 GB.

    cd ~/streaming-machine && python3 sharedcol/xcheck.py phase=weights
    python3 sharedcol/xcheck.py phase=gpt2
    python3 sharedcol/xcheck.py phase=pythia size=160m n=50 ;  python3 sharedcol/xcheck.py phase=pythia size=70m n=10
  writes sharedcol/xcheck_<phase>[_<size>].log
"""
import os, sys, json, time, math, resource, importlib.util, hashlib
from fractions import Fraction
import numpy as np

HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/sharedcol"
A_ = dict(a.split("=", 1) for a in sys.argv[1:]); PHASE = A_.get("phase", "weights"); SIZE = A_.get("size", "160m"); NPOS = int(A_.get("n", 50))
LOGF = open(f"{OUT}/xcheck_{PHASE}{'_' + SIZE if PHASE == 'pythia' else ''}.log", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
T0 = time.time()

# ---------------- counting by digit loops (not popcount) ----------------
def ones_loop(X):
    a = np.abs(np.asarray(X, np.int64)); c = np.zeros(a.shape, np.int64)
    while a.any(): c += a & 1; a = a >> 1
    return c
def signed_loop(X):                                   # signed digits: take the low digit +1 or -1 so the next is 0
    a = np.abs(np.asarray(X, np.int64)); c = np.zeros(a.shape, np.int64)
    while a.any():
        odd = a & 1; dg = np.where(odd == 1, 2 - (a & 3), 0); c += odd; a = (a - dg) >> 1
    return c
def half_up(num, den): return math.floor(Fraction(int(num), int(den)) + Fraction(1, 2))   # rounding, a half goes up
def sig11_one(v):
    b = abs(int(v)).bit_length()
    if b <= 11: return int(v)
    dr = b - 11; return half_up(int(v), 1 << dr) << dr
def topbit(x): return np.array([int(v).bit_length() for v in np.asarray(x).ravel()]).reshape(np.shape(x))

# ================================== PHASE weights ==================================
def columns(Q, E, BL=32):
    """Q [lanes][words] int16 on each word's own grid. Returns the three columns on the coarsest grid (Python whole numbers)"""
    d, V = Q.shape; emin, emax = int(E.min()), int(E.max()); G = emax - emin; up = (E - emin).astype(np.int64)
    S, MED = [], []
    for l0 in range(0, d, BL):
        B = np.asarray(Q[l0:l0 + BL]).astype(np.int64) << up[None, :]
        S += [int(x) for x in B.sum(1, dtype=np.int64)]
        Bs = np.sort(B, axis=1); MED += [int(x) for x in Bs[:, (V - 1) // 2]]
    mean = np.array([half_up(s, V << G) for s in S], np.int64)
    med = np.array([half_up(m, 1 << G) for m in MED], np.int64)
    m11 = np.array([sig11_one(v) for v in mean], np.int64)
    return dict(mean=mean, median=med, mean11=m11), emin, emax

def weigh(Q, E, cols, emin, emax, BL=32):
    """every statistic for the table and each column's fringe"""
    d, V = Q.shape; up = (E - emin).astype(np.int64); kw = (emax - E).astype(np.int64); G = emax - emin
    keys = ["T"] + list(cols)
    st = {k: dict(sq=0, n1=0, nn=0, wsq=np.zeros(V), wor=np.zeros(V, np.int64), pos=np.zeros(d, np.int64), neg=np.zeros(d, np.int64)) for k in keys}
    for l0 in range(0, d, BL):
        A = np.asarray(Q[l0:l0 + BL]).astype(np.int64)
        n = len(A)
        for k in keys:
            X = A if k == "T" else A - (cols[k][l0:l0 + n, None] << kw[None, :])
            if k != "T": assert np.array_equal(X + (cols[k][l0:l0 + n, None] << kw[None, :]), A)
            Xf = X << up[None, :]; assert np.abs(Xf).max() < 1 << 21                  # the finest grid, real sizes (sums exact in int64)
            st[k]["sq"] += sum(int(v) for v in (Xf * Xf).sum(1, dtype=np.int64))
            st[k]["wsq"] += (Xf.astype(np.float64) ** 2).sum(0)
            st[k]["n1"] += int(ones_loop(X).sum()); st[k]["nn"] += int(signed_loop(X).sum())
            st[k]["wor"] |= np.bitwise_or.reduce(np.abs(X), axis=0)
            st[k]["pos"][l0:l0 + n] = (X > 0).sum(1); st[k]["neg"][l0:l0 + n] = (X < 0).sum(1)
    N = d * V; T = st["T"]; R = {}
    R["lit_T"] = (T["n1"] / N, T["nn"] / N)
    same = np.maximum(T["pos"], T["neg"]) / V
    R["same_median_lane"] = float(np.percentile(same, 50)); R["lanes99"] = float((same >= 0.99).mean())
    R["top_T"] = float(topbit(T["wor"]).mean())
    wn = np.sqrt(T["wsq"])
    for k in cols:
        c = cols[k]; cf = [int(v) << G for v in c]
        R[k] = dict(share=V * sum(v * v for v in cf) / T["sq"], removed=1 - st[k]["sq"] / T["sq"],
                    c_over_median_word=float(np.sqrt(sum(v * v for v in cf))) / float(np.median(wn)),
                    fringe_over_word=float(np.median(np.sqrt(st[k]["wsq"] / T["wsq"]))),
                    ones=st[k]["n1"] / N, signed=st[k]["nn"] / N,
                    ones_fewer=1 - st[k]["n1"] / T["n1"], signed_fewer=1 - st[k]["nn"] / T["nn"],
                    col_signed=int(signed_loop(c).sum()), col_ones=int(ones_loop(c).sum()),
                    top=float(topbit(st[k]["wor"]).mean()),
                    fringe_same_median=float(np.percentile(np.maximum(st[k]["pos"], st[k]["neg"]) / V, 50)))
    return R

def show_w(name, R):
    log(f"  [{name}] lit a weight: plain 1s {R['lit_T'][0]:.4f}, signed {R['lit_T'][1]:.4f};  same sign, median lane {100 * R['same_median_lane']:.3f}% "
        f"(lanes at 99%+: {100 * R['lanes99']:.1f}%);  top place a word {R['top_T']:.3f}")
    for k in ("mean", "median", "mean11"):
        r = R[k]
        log(f"    {k:6s} share {100 * r['share']:.4f}%  removes {100 * r['removed']:.4f}%  |c|/median word {r['c_over_median_word']:.4f}  fringe/word {r['fringe_over_word']:.4f}"
            f"  | plain 1s {r['ones']:.4f} ({100 * r['ones_fewer']:+.2f}% fewer)  signed {r['signed']:.4f} ({100 * r['signed_fewer']:+.2f}% fewer)"
            f"  | column's own lit {r['col_signed']} signed  | top place {r['top']:.3f}  | fringe same-sign median lane {100 * r['fringe_same_median']:.2f}%")

if PHASE == "weights":
    for name in ("pythia-70m", "gpt2"):
        if name == "gpt2": Z = np.load(f"{HERE}/base2_gpt2.npz")
        else: Z = np.load(f"{HERE}/foundation0/base2_{name}.npz")
        Q = np.asarray(Z["head.w"]); E = np.asarray(Z["head.e"]).astype(np.int64); d, V = Q.shape
        log(f"== {name}: table {Q.shape} {Q.dtype}, word powers {dict(zip(*[x.tolist() for x in np.unique(E, return_counts=True)]))}  ({time.time() - T0:.0f}s, {mem()} MB)")
        cols, emin, emax = columns(Q, E)
        R = weigh(Q, E, cols, emin, emax); show_w("finished", R)
        c = cols["mean"]; csq = (c.astype(np.float64)) ** 2; o = np.argsort(-csq); fr = csq[o] / csq.sum()
        log(f"    mean column: lanes up {int((c > 0).sum())}, down {int((c < 0).sum())}; largest lanes {o[:5].tolist()} hold "
            f"{', '.join(f'{100 * x:.2f}%' for x in fr[:5])}; largest 7 {100 * fr[:7].sum():.2f}%; largest 10% of lanes {100 * fr[:d // 10].sum():.2f}%; largest 1% {100 * fr[:max(d // 100, 1)].sum():.2f}%")
        # SHUFFLE CONTROL: each word's values dealt to the lanes in its own random order
        rng = np.random.default_rng([2026, 10, 6, d])
        Qp = np.empty((d, V), np.int16)
        for w0 in range(0, V, 4096):
            blk = np.asarray(Q[:, w0:w0 + 4096]); pr = np.argsort(rng.random(blk.shape), axis=0); Qp[:, w0:w0 + 4096] = np.take_along_axis(blk, pr, axis=0)
        assert (np.sort(Qp, 0) == np.sort(np.asarray(Q), 0)).all()
        colsP, _, _ = columns(Qp, E); RP = weigh(Qp, E, colsP, emin, emax); show_w("CONTROL each word's values dealt to lanes at random", RP)
        # ORDER-FREE
        lp, wp = rng.permutation(d), rng.permutation(V); Qo = np.ascontiguousarray(np.asarray(Q)[lp][:, wp]); Eo = E[wp]
        colsO, _, _ = columns(Qo, Eo); RO = weigh(Qo, Eo, colsO, emin, emax)
        colsame = all(np.array_equal(colsO[k], cols[k][lp]) for k in cols)
        diff = [k for k in ("mean", "median", "mean11") for f in ("share", "removed", "ones", "signed") if RO[k][f] != R[k][f]]
        log(f"    ORDER-FREE (lanes and words relabelled): columns the same lane for lane {colsame}; share/lit numbers differing: {diff or 'none'}")
        del Qp, Qo
    log(f"done ({time.time() - T0:.0f}s, max {mem()} MB)"); sys.exit(0)

# ================================== the reads ==================================
def race5(vals):                                      # the race run five times: high first, then the lower word number
    o = np.lexsort((np.arange(len(vals)), -vals)); return o[:5]

def kread(z, s, exact_scores):
    """z: each word's order number (int64 or object), s: the position's shift; exact_scores(K) -> scores of the words
    K with the shared pile added (the machine's own rounding). Returns top 5, K, whether S was added."""
    n = len(z); z5 = np.partition(z, n - 5)[n - 5]
    K = np.flatnonzero(z > z5 - (1 << s)) if s >= 1 else np.flatnonzero(z >= z5)
    Ks = K[np.lexsort((K, -z[K]))]; dif = -np.diff(z[Ks])
    if not (s >= 1 and ((dif > 0) & (dif < (1 << s))).any()): return Ks[:5], K, False
    sc = exact_scores(K); return K[np.lexsort((K, -sc))][:5], K, True

def mm(Y, T, WB=4096):                                # int64 sums, word block by word block (T stored int32)
    out = np.empty((len(Y), T.shape[1]), np.int64)
    for w0 in range(0, T.shape[1], WB): out[:, w0:w0 + WB] = Y @ np.asarray(T[:, w0:w0 + WB]).astype(np.int64)
    return out
def lane_sum(T, f, BL=64): return np.concatenate([f(np.asarray(T[l0:l0 + BL])).sum(1) for l0 in range(0, T.shape[0], BL)])

def meet_blocks(Y, Tq, extra=None, f=ones_loop, WB=4096):
    """sum over every (position, lane, word) of f(y) x f(table value), word block by word block; extra: per-lane vector met once"""
    fy = f(Y); tot = 0
    for w0 in range(0, Tq.shape[1], WB): tot += int((fy @ f(Tq[:, w0:w0 + WB])).sum())
    if extra is not None: tot += int((fy @ f(extra)).sum())
    return tot

if PHASE == "gpt2":
    src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
    os.chdir(HERE); argv = sys.argv; sys.argv = ["x"]; ns = {}; exec(src, ns); sys.argv = argv
    M, W = ns["M"], ns["W"]; ES = ns["ES"]; EXP2 = ns["EXP2"]
    w, we, b = W["head"]; we = np.asarray(we, np.int64); b = np.asarray(b, np.int64); assert (we == we[0]).all()
    Q = np.asarray(M["head.w"]).astype(np.int32); d, V = Q.shape
    WIN = json.load(open(f"{HERE}/judge5_texts.json")); REF = np.load(f"{HERE}/judge5_ref.npz")["top5"]
    log(f"== GPT-2 cut read: top table {d} x {V}, one power 2^{we[0]}  ({time.time() - T0:.0f}s, {mem()} MB)")
    S_ = Q.astype(np.int64).sum(1); c = np.array([half_up(int(s), V) for s in S_], np.int64); U = (Q - c[:, None]).astype(np.int32); assert np.array_equal(U + c[:, None], Q)
    rng = np.random.default_rng([2026, 10, 6, 2]); cp = c[rng.permutation(d)]; Up = (Q - cp[:, None]).astype(np.int32)
    lp, wp = rng.permutation(d), rng.permutation(V); Uo = np.ascontiguousarray(U[lp][:, wp]); bo = b[wp]
    X16, LSs, EXs, J5REF, J5MINE = [], [], [], [], []
    for i in range(3):
        ids = WIN[i]["ids"]; P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
        for l in range(12):
            R = R + ns["plain_bank"](ns["attention"](ns["gauged_bank"](R, f"qkv.{l}")), f"proj.{l}")
            R = R + ns["plain_bank"](ns["edge"](ns["gauged_bank"](R, f"fc.{l}")), f"mproj.{l}")
        x16, ex, LS = ns["gauge_log"](R[3:23]); X16.append(x16); EXs.append(ex); LSs.append(LS); J5REF.append(REF[i])
        log(f"  window {i}: {len(x16)} judged positions  ({time.time() - T0:.0f}s, {mem()} MB)")
    X16 = np.concatenate(X16)[:NPOS].astype(np.int64); EX = np.concatenate(EXs)[:NPOS]; LS = np.concatenate(LSs)[:NPOS]; RF = np.concatenate(J5REF)[:NPOS]
    assert np.abs(X16).max() <= 1 << 15
    plain = ns["power"](ns["product"](X16, w), (we[None, :] + EX[:, None] + ES) * 65536 - LS[:, None]) + b[None, :]   # the machine's own
    PQ = mm(X16, Q); S = X16 @ c; Fw = mm(X16, U); Sp = X16 @ cp; Fp = mm(X16, Up); So = X16[:, lp] @ c[lp]; Fo = mm(X16[:, lp], Uo)
    ok = dict(pile=0, top5=0, win=0, ref=0, refwin=0, added=0, Kover5=0, cs_top5=0, cs_win=0, of_pile=0, of_top5=0, viol=0, full=0)
    Ks = []
    for p in range(NPOS):
        ok["pile"] += int(np.array_equal(Fw[p] + S[p], PQ[p])) ; ok["of_pile"] += int(np.array_equal(Fo[p][np.argsort(wp)], Fw[p]) and So[p] == S[p])
        t5p = race5(plain[p]); win = int(np.argmax(plain[p])); assert win == ns["race"](plain[p]) and win == t5p[0]
        ok["ref"] += int(np.array_equal(t5p, RF[p])); ok["refwin"] += int(win == RF[p][0])
        Fpar = int((we[0] + EX[p] + ES) * 65536 - LS[p]); I, f = Fpar >> 16, Fpar & 0xFFFF; m = int(EXP2[f]); s = 16 - I; assert s >= 1
        def run(F, Sx, bb):
            assert int(np.abs(F).max()) * m < 1 << 62 and int(np.abs(bb).max()) << s < 1 << 62
            z = F * m + (bb << s); A = int(Sx) * m + (1 << (s - 1))
            def exact(K): return np.array([(A + int(z[k])) >> s for k in K], dtype=np.int64)   # floor((A + z) / 2^s): the machine's rounding
            return kread(z, s, exact), z, A
        (t5, K, added), z, A = run(Fw[p], S[p], b)
        full = np.array([(A + int(v)) >> s for v in z], dtype=np.int64); ok["full"] += int(np.array_equal(full, plain[p]))   # every word's score rebuilt from fringe + S
        ok["top5"] += int(np.array_equal(t5, t5p)); ok["win"] += int(t5[0] == win); ok["added"] += added; ok["Kover5"] += len(K) > 5; Ks.append(len(K))
        out = np.ones(V, bool); out[K] = False; ok["viol"] += int((plain[p][out] > plain[p][t5p[4]] - 1).sum())
        (t5c, _, _), _, _ = run(Fp[p], Sp[p], b); ok["cs_top5"] += int(np.array_equal(t5c, t5p)); ok["cs_win"] += int(t5c[0] == win)
        (t5o, _, _), _, _ = run(Fo[p], So[p], bo); ok["of_top5"] += int(np.array_equal(wp[t5o], t5p))
    log(f"  plain read (the machine's power(product) + b) against judge5_ref top 5 (set and order): {ok['ref']}/{NPOS}; its winner = ref's first {ok['refwin']}/{NPOS}")
    log(f"  every pile: fringe pile + shared pile = the plain pile (int64): {ok['pile']}/{NPOS} positions; every word's score from fringe + S by floor((A+z)/2^s) = the machine's: {ok['full']}/{NPOS}")
    log(f"  CUT READ: top 5 (set and order) = plain at {ok['top5']}/{NPOS}; winner {ok['win']}/{NPOS}; K sizes {dict(zip(*[x.tolist() for x in np.unique(Ks, return_counts=True)]))}; "
        f"S added at {ok['added']} positions; words outside K within one step of the 5th: {ok['viol']}")
    log(f"  colshuffle (column in other lanes): top 5 {ok['cs_top5']}/{NPOS}, winner {ok['cs_win']}/{NPOS};  ORDER-FREE: piles same {ok['of_pile']}/{NPOS}, top 5 same {ok['of_top5']}/{NPOS}")
    # meetings at 10 positions (every position 5 apart through the 50)
    sel = np.arange(0, NPOS, NPOS // 10)[:10]; Y = X16[sel]
    for f, nm in ((ones_loop, "plain 1s"), (signed_loop, "signed digits")):
        pl = meet_blocks(Y, Q, f=f); cu = meet_blocks(Y, U, extra=c, f=f); cs = meet_blocks(Y, Up, extra=cp, f=f)
        Ysh = np.array([rng.permutation(y) for y in Y]); pls = meet_blocks(Ysh, Q, f=f); cus = meet_blocks(Ysh, U, extra=c, f=f)
        lane = int((f(Y) @ lane_sum(Q, f)).sum())
        log(f"  MEETINGS at 10 positions, {nm}: plain {pl:,} ({pl // 10:,} a position; lane-sum form {lane:,} {'EQUAL' if lane == pl else 'DIFFERENT'}); "
            f"cut {cu:,} -> {100 * cu / pl:.2f}% of plain;  colshuffle {100 * cs / pl:.2f}%;  light shuffled {100 * cus / pls:.2f}%")
    log(f"done ({time.time() - T0:.0f}s, max {mem()} MB)"); sys.exit(0)

if PHASE == "pythia":
    F0 = f"{HERE}/foundation0"; ES = 16
    argv = sys.argv; sys.argv = ["pythia_base2.py", f"size={SIZE}", "plates=finished"]
    spec = importlib.util.spec_from_file_location("pyb2_xcheck", f"{F0}/pythia_base2.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.argv = argv
    assert m.PLATES == "finished" and m.ORDER == "plain" and (m.PERM == np.arange(m.d)).all()
    Z = np.load(f"{F0}/base2_pythia-{SIZE}.npz"); Q = np.asarray(Z["head.w"]).astype(np.int32); E = np.asarray(Z["head.e"]).astype(np.int64); d, V = Q.shape
    emin, emax = int(E.min()), int(E.max()); KW = emax - E; UP = E - emin
    refs = json.load(open(f"{F0}/refs_pythia-{SIZE}.json")); CL = np.load(f"{OUT}/cutlight_{SIZE}.npz")
    log(f"== Pythia-{SIZE} cut read: top table {d} x {V}, word powers {dict(zip(*[x.tolist() for x in np.unique(E, return_counts=True)]))}  ({time.time() - T0:.0f}s, {mem()} MB)")
    Ys, EYs, LGm = [], [], []; i = 0
    while sum(len(y) for y in Ys) < NPOS:
        ids = np.asarray(refs[i]["ids"] + refs[i]["ref"][:15])
        R = m.rshift(m.pick(m.EMB_F[ids], m.EMB_S[ids]), -(m.EMB_E[ids][:, None] + ES))
        for l in range(m.L):
            At = m.plain(m.attention(m.plain(m.gauge(R, f"ln1.{l}"), f"qkv.{l}")), f"proj.{l}")
            Mo = m.plain(m.edge(m.plain(m.gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
            R = R + At + Mo
        y16, ey = m.to16(m.gauge(R, "lnf")); _, lgm = m.forward(ids, last=len(ids))
        Ys.append(y16); EYs.append(np.asarray(ey, np.int64)); LGm.append(lgm); i += 1
        log(f"  text {i - 1}: {len(ids)} positions  ({time.time() - T0:.0f}s, {mem()} MB)")
    Y = np.concatenate(Ys)[:NPOS].astype(np.int64); EY = np.concatenate(EYs)[:NPOS]; LGM = np.concatenate(LGm)[:NPOS]
    m.PL.clear()                                     # the light is in hand; free this process's copy of the plates
    light_same = np.array_equal(Y, CL["y16"][:NPOS].astype(np.int64)) and np.array_equal(EY, CL["ey"][:NPOS])
    PQ = mm(Y, Q); plain = m.rshift(PQ, -(E[None, :] + EY[:, None] + ES))
    log(f"  the light: identical to the session's cutlight_{SIZE}.npz at {NPOS} positions: {light_same};  plain piles by int64 sums = the module's own forward(): {np.array_equal(plain, LGM)}")
    # the mean11 column, rebuilt from its definition
    Ssum = [int(x) for x in (Q.astype(np.int64) << UP[None, :]).sum(1)]
    c = np.array([sig11_one(half_up(s, V << (emax - emin))) for s in Ssum], np.int64)
    U = (Q - (c[:, None] << KW[None, :])).astype(np.int32); assert np.array_equal(U + (c[:, None] << KW[None, :]), Q)
    nw = int((np.abs(U) > 32767).any(0).sum())
    log(f"  mean11 column: |c| up to {int(np.abs(c).max())}, lanes up {int((c > 0).sum())} down {int((c < 0).sum())}; words with a fringe value past 16 places: {nw}")
    rng = np.random.default_rng([2026, 10, 6, d]); cp = c[rng.permutation(d)]; Up = (Q - (cp[:, None] << KW[None, :])).astype(np.int32)
    lp, wp = rng.permutation(d), rng.permutation(V); Uo = np.ascontiguousarray(U[lp][:, wp]); Eo = E[wp]
    S = Y @ c; Fw = mm(Y, U); Sp = Y @ cp; Fp = mm(Y, Up); So = Y[:, lp] @ c[lp]; Fo = mm(Y[:, lp], Uo)
    ok = dict(pile=0, top5=0, win=0, added=0, viol=0, full=0, cs_top5=0, cs_win=0, of_pile=0, of_top5=0, refword=0, refn=0); Ks = []
    def run(F, Sx, Ew, ey):
        z = F << (Ew - emin); s = -(emin + int(ey) + ES); S2 = int(Sx) << (emax - emin)
        def exact(K): return np.array([((int(z[k]) + S2 + (1 << (s - 1))) >> s) if s >= 1 else ((int(z[k]) + S2) << -s) for k in K], dtype=np.int64)
        return kread(z, s, exact), z, s, S2
    for p in range(NPOS):
        ok["pile"] += int(np.array_equal(Fw[p] + (S[p] << KW), PQ[p]))
        ok["of_pile"] += int(np.array_equal(Fo[p][np.argsort(wp)], Fw[p]) and So[p] == S[p])
        t5p = race5(plain[p]); win = int(np.argmax(plain[p])); assert win == m.race(plain[p])
        (t5, K, added), z, s, S2 = run(Fw[p], S[p], E, EY[p]); assert s >= 1
        full = np.array([(int(v) + S2 + (1 << (s - 1))) >> s for v in z], dtype=np.int64); ok["full"] += int(np.array_equal(full, plain[p]))
        ok["top5"] += int(np.array_equal(t5, t5p)); ok["win"] += int(t5[0] == win); ok["added"] += added; Ks.append(len(K))
        out = np.ones(V, bool); out[K] = False; ok["viol"] += int((plain[p][out] > plain[p][t5p[4]] - 1).sum())
        (t5c, _, _), _, _, _ = run(Fp[p], Sp[p], E, EY[p]); ok["cs_top5"] += int(np.array_equal(t5c, t5p)); ok["cs_win"] += int(t5c[0] == win)
        (t5o, _, _), _, _, _ = run(Fo[p], So[p], Eo, EY[p]); ok["of_top5"] += int(np.array_equal(wp[t5o], t5p))
    log(f"  every pile: fringe pile + (S << (emax - e_w)) = the plain pile: {ok['pile']}/{NPOS}; every word's score from z + S'' = the plain score: {ok['full']}/{NPOS}")
    log(f"  CUT READ: top 5 (set and order) = plain at {ok['top5']}/{NPOS}; winner {ok['win']}/{NPOS}; K sizes {dict(zip(*[x.tolist() for x in np.unique(Ks, return_counts=True)]))}; "
        f"S added at {ok['added']}; words outside K within one step of the 5th: {ok['viol']}")
    log(f"  colshuffle: top 5 {ok['cs_top5']}/{NPOS}, winner {ok['cs_win']}/{NPOS};  ORDER-FREE: piles same {ok['of_pile']}/{NPOS}, top 5 same {ok['of_top5']}/{NPOS}")
    sel = np.arange(0, NPOS, max(NPOS // 10, 1))[:10]; Yt = Y[sel]
    for f, nm in ((ones_loop, "plain 1s"), (signed_loop, "signed digits")):
        pl = meet_blocks(Yt, Q, f=f); cu = meet_blocks(Yt, U, extra=c, f=f); cs = meet_blocks(Yt, Up, extra=cp, f=f)
        Ysh = np.array([rng.permutation(y) for y in Yt]); pls = meet_blocks(Ysh, Q, f=f); cus = meet_blocks(Ysh, U, extra=c, f=f)
        log(f"  MEETINGS at 10 positions ({sel.tolist()}), {nm}: plain {pl:,} ({pl // 10:,} a position); cut {cu:,} -> {100 * cu / pl:.2f}% of plain;"
            f"  colshuffle {100 * cs / pl:.2f}%;  light shuffled {100 * cus / pls:.2f}%")
    # the session's own light over all 622 positions, lane sums, against cut_<size>.json
    YA = CL["y16"].astype(np.int64); cj = json.load(open(f"{OUT}/cut_{SIZE}.json"))["counts"]
    o1, on = ones_loop(YA), signed_loop(YA)
    mine = dict(plain1=int((o1 @ lane_sum(Q, ones_loop)).sum()), cut1=int((o1 @ lane_sum(U, ones_loop)).sum()), shared1=int((o1 @ ones_loop(c)).sum()),
                plainn=int((on @ lane_sum(Q, signed_loop)).sum()), cutn=int((on @ lane_sum(U, signed_loop)).sum()), sharedn=int((on @ signed_loop(c)).sum()))
    log(f"  622 positions (the session's light file): " + "; ".join(f"{k} {v:,} {'EQUAL' if v == cj[k] else 'DIFFERS from ' + format(cj[k], ',')}" for k, v in mine.items()))
    log(f"    -> cut {100 * (mine['cut1'] + mine['shared1']) / mine['plain1']:.2f}% of plain (plain 1s), {100 * (mine['cutn'] + mine['sharedn']) / mine['plainn']:.2f}% (signed)")
    log(f"done ({time.time() - T0:.0f}s, max {mem()} MB)")
