"""THE TOP READ AS A COLLAPSING WAVEFORM, EXACT TOP 5, STAGED DOWN THE TRIANGLE ON THE FINE GRID (THE KNIFE RUN'S).

A copy of collapse_top/staged_top.py. The method is unchanged: the same saved light, the same counted piles, the same
exact bound, the same set-aside rule, the same final read and the same checks. What is new:
  1. THE STAGE LIST, fixed in advance as the knife run's grid (knife/settle.py), no search:
       sched=diag    the top 2, 4, 6, ..., 28 diagonals (a + b >= 29 - k), then all 29
       sched=light   the light's top 2, 4, ..., 12 places (a >= 15 - k), then all 15
     (staged_top.py stopped the diagonals at the top 12, a + b >= 17, which hold 4.1% of the read: places 11-14 are
     almost never lit on either side. The top 16 hold 21.3%, the top 18 35.9%, the top 20 53.8%.)
  2. A SECOND ROW, the signed (up/down) bound, run beside the plain one on the same counted piles, with its own words
     still in, its own work and its own final read:
       rest <=  sum over uncounted (a, b) of 2^(a+b) [min(n+_a, m+_wb) + min(n-_a, m-_wb)]    (up pairs: same signs)
       rest >= -sum over uncounted (a, b) of 2^(a+b) [min(n+_a, m-_wb) + min(n-_a, m+_wb)]    (down pairs)
     n+_a, n-_a: lines where the light has a photon at place a in an up value, a down value (the light's own, known at
     the start); m+_wb, m-_wb: lines where word w's stamp has a photon at place b in an up value, a down value (30 counts
     a word, stored once with the stamp). Each side is never wider than the plain bound, since
     min(n+, m+) + min(n-, m-) <= min(n+ + n-, m+ + m-) = min(n, m). It is exact; it is NOT the same cost: per word per
     stage it reads 60 table values (two tables, each read at m+ and at m-, for each side) against 15.
  3. THE BOUND WORK ALSO IN PHOTONS, by combined/machine.py's own rule, so the combined machine's top read (506.0M
     meetings + 61.5M bound work a token) can be set beside it:
       tables built from the light  sum over uncounted (a, b) of sum over m = 0..768 of ones(min(n_a, m))
       bound adds                   ones of the table values read, then pile -/+ bound (plain: 2 ones(R); signed:
                                    ones(U) + ones(D))
       divides of the bounds        (ones(lo pile) + ones(hi pile)) x the gain's photons
       standing amounts             2 x ones(the word's standing amount)
     and, for the words read to the end, the top divide and standing amount (combined's "rest" for the top).
  4. part=i/n runs the i-th of n equal slices of the positions (each run stays under the 10 minute foreground limit);
     fine_join.py puts the parts together. The bound is read from its table through a flat index (np.take) instead of
     a 2-D index; the numbers are the same, and the original bound() is kept and checked against it.

THE PICTURE (staged_top.py's, unchanged). At a read position the light x (768 whole numbers on 16 bits) meets every
word's stamp T_w (768 whole numbers). Every value is its sign and its photons (places 0..14). A light photon at place a
meeting a stamp photon at place b on the same line is a pair worth 2^(a+b), up or down, on diagonal a + b. The pairs
are the triangle; the top diagonal is 28. Reading from the top down, what the unread pairs can still add is bounded
exactly from photon counts alone, and a word is set aside when five words are surely above it (the point of no
return). score_w = power(pile_w, F) + bias_w; power never runs backwards.
  plain bound: |rest_w| <= sum over uncounted (a, b) of 2^(a+b) x min(n_a, m_wb)
  lo_w = power(counted_w - bound_w, F) + bias_w  <=  score_w  <=  hi_w = power(counted_w + bound_w, F) + bias_w
  set aside when hi_w < the 5th-best lo. After the last stage the words still in are read exactly, top 5 among them.
WORK: photon meetings for the pairs actually met, over the words still in (stage 1: every word). The bound's work
stated apart, in staged_top.py's unit (table reads and adds, gauge reads, compares) and in photons (combined's rule).

CHECKS: lo <= exact <= hi for every word at every stage, both rows; the fast bound equal to the original bound(), and
each signed side no wider than the plain bound, on the checked positions; the staged pieces against a pair-by-pair
count on sampled words; the stages' new pairs add up to the plain read's meetings; the top 5 (set and order) equal to
the plain read's at every position, both rows; on the windows also judge5's reference top 5 (and judge5.judge itself:
collapse_top/fine_judge_check.py); on the refs the race winner equal to the one saved with the light.
CONTROL: light=shuffled shuffles each position's light across its lines (every value kept, only which line it meets).
ORDER-FREE: order=1 puts the words and the lines (light and stamp together) in one shared random order first.

GATING (holographicplate CLAUDE.md, main): it measures a settling (when each word's place below the top 5 is fixed);
every read is on whole numbers (float containers hold whole numbers below 2^53, every sum checked); it is exact (bounds
only); it discards nothing (a set-aside is proven, the top 5 is compared with the full read); no fit, no word-count
comparison; not measured before on this grid (staged_top.py's diagonal stages stopped above 96% of the read).

PREDICTIONS (written before this script first ran):
  the review's: diagonal schedule 25-35% of the plain top read in meetings, its bound adds under 3%; light schedule on
    the fine grid 48-56%; shuffled light at or below real; exact everywhere. If the diagonal schedule is not below 45%,
    the block is the bound itself, and the next step is the up/hole bound.
  the session's:
    - diag, plain bound: every word still in through the top 10 diagonals; median still above 40,000 at the top 12;
      the collapse comes between the top 14 and the top 18 (median under 1,000 at the top 16, under 50 at the top 18,
      about 10 by the top 20). Meetings 20-32% of plain.
    - the bound's work: in staged_top.py's unit (table reads and adds) 0.5-1.5% of the plain meetings. In photons by
      the combined machine's rule it is not small: about 7 full stages of 50,257 words, each paying two divides of a
      pile of about 2^38 by the gain, about 450 photons a word a stage: 15-25% of the plain top read. So the diagonal
      schedule's top read with its bound work is 40-52% of plain, against the combined machine's 63.3%.
    - light, fine grid: median above 40,000 after 8 places (as before), a few hundred to a few thousand after 10, about
      10 after 12. Meetings 47-53% of plain; bound work in photons 10-15%.
    - signed bound: nearly the same as the plain one (the low places are coin tosses, n+ about n- and m+ about m-, so
      min(n+, m+) + min(n-, m-) is about min(n, m)); survivors and meetings within 1-2 points of the plain row, at 4x the
      table reads. (In the knife run the two bounds settled the same edges to 0.1%.)
    - shuffled light at or below real on both schedules; exact everywhere: top 5 6,480/6,480 both rows, bound broken 0,
      judge5 6,000/6,000.

    python3 collapse_top/fine_staged_top.py sched=diag|light [light=real|shuffled] [order=0|1] [part=i/n] [batch=50]
      -> collapse_top/fine_<sched>_<light>[_order][_p<i>of<n>].npz, summary printed
"""
import sys, os, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
SCHED, LIGHT, ORDER, SET = ARG.get("sched", "diag"), ARG.get("light", "real"), int(ARG.get("order", 0)), ARG.get("set", "both")
BATCH = int(ARG.get("batch", 50)); PI, PN = (int(v) for v in ARG.get("part", "1/1").split("/"))
import types                                           # the machine's steps, as table_bits.py / judge5.py load them:
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]     # base2.py up to its own test
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")             # never used by
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")                    # the steps (memory)
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '["head"]')                          # this process reads only the top
_argv = sys.argv; sys.argv = ["x"]; _g = {}; exec(_src, _g); sys.argv = _argv
J = types.SimpleNamespace(**_g)
def top5(S):                                           # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)
def ones(A): return np.bitwise_count(np.asarray(A, dtype=np.int64)).astype(np.int64)    # combined/machine.py's: 1s of |value|
M, W = J.M, J.W
t0 = time.time()
PL, NW, LINES = 15, 50257, 768
FLOOR = -(1 << 62); KEEPN = 256

L = np.load(f"{HERE}/collapse_top/light_top.npz")
SRC, WIN, REFW = L["src"], L["win"], L["refs_winner"]
X_ALL = L["x16"].astype(np.int64); EXA, LSA = L["ex"], L["LS"]
WH, WE, BIAS = W["head"]                               # (768, NW) float64, (NW,), (NW,)
assert (WE == WE[0]).all()
TT = np.ascontiguousarray(M["head.w"].T)              # (NW, 768) int16: each word's stamp
assert np.abs(X_ALL).max() < 1 << PL and np.abs(TT.astype(np.int32)).max() < 1 << PL
WPERM = np.arange(NW); LPERM = np.arange(LINES)
if ORDER:                                              # words and lines in another order, light and stamp together
    rng = np.random.default_rng(99); WPERM = rng.permutation(NW); LPERM = rng.permutation(LINES)
    TT = np.ascontiguousarray(TT[WPERM][:, LPERM]); WH = np.ascontiguousarray(WH[LPERM][:, WPERM]); BIAS = BIAS[WPERM]
    X_ALL = X_ALL[:, LPERM]
if LIGHT == "shuffled":                                # CONTROL: each position's light across its lines
    rng = np.random.default_rng(2026)
    X_ALL = np.stack([x[rng.permutation(LINES)] for x in X_ALL])
TA = np.abs(TT.astype(np.int32)).astype(np.uint16)    # photons' places
TS = np.sign(TT).astype(np.int8)                      # direction
NT = np.stack([((TA >> b) & 1).sum(1, dtype=np.int64) for b in range(PL)], 1)            # m_wb (NW, 15)
MP = np.stack([(((TA >> b) & 1) * (TS > 0)).sum(1, dtype=np.int64) for b in range(PL)], 1)   # m+_wb: in up values
MM = NT - MP                                                                              # m-_wb: in down values
assert (MM >= 0).all() and np.array_equal(MM, np.stack([(((TA >> b) & 1) * (TS < 0)).sum(1, dtype=np.int64) for b in range(PL)], 1))
CNT_ALL = np.stack([((TA >> b) & 1).sum(0, dtype=np.int64) for b in range(PL)])          # words with a photon at b, per line
PPH = np.bitwise_count(TA).sum(0, dtype=np.int64)     # stamp photons per line (all words)
BIAS_ONES = ones(BIAS)

# ---------- THE STAGE LIST (the one change to the method): the knife run's grid, fixed in advance ----------
KS = list(range(2, 2 * (PL - 1) + 1, 2)) if SCHED == "diag" else list(range(2, 13, 2))    # diag 2..28; light 2..12
NS = len(KS)
a_, b_ = np.arange(PL)[:, None], np.arange(PL)[None, :]
CS = []
for k in KS:
    C = (b_ >= PL - k) if SCHED == "stamp" else (a_ >= PL - k) if SCHED == "light" else (a_ + b_ >= 2 * (PL - 1) + 1 - k)
    CS.append(np.broadcast_to(C, (PL, PL)).copy())
CS.append(np.ones((PL, PL), bool))
NEW = [CS[0]] + [CS[s] & ~CS[s - 1] for s in range(1, NS + 1)]
assert all(n.any() for n in NEW) and (sum(n.astype(int) for n in NEW) == 1).all()       # every stage new; a partition
def amask(C): return np.array([sum(1 << a for a in range(PL) if C[a, b]) for b in range(PL)], np.int64)
AMASK = np.stack([amask(C) for C in CS]); NMASK = np.stack([amask(N) for N in NEW])
POWA = (1 << np.arange(PL)).astype(np.int64); NN = np.arange(LINES + 1)
TABLE_ADDS = [int((~CS[s]).any(1).sum()) * PL * (LINES + 1) for s in range(NS)]         # the bound's table, per stage
ONES_NN = ones(NN)

def bound(nx, s, rows):                               # sum over uncounted (a,b) 2^(a+b) min(n_a, m_wb), for the words in rows
    G = ((~CS[s]).T * POWA[None, :]) @ np.minimum(nx[:, None], NN[None, :])                # (15 b, 769)
    return (G[np.arange(PL)[None, :], NT[rows]] << np.arange(PL)[None, :]).sum(1)
# the same bound read through a flat index: G[b, m] << b sits at b x 769 + m
OFF = (LINES + 1) * np.arange(PL)[None, :]
IX_N, IX_P, IX_M = NT + OFF, MP + OFF, MM + OFF                                           # (NW, 15) each
HN, HP, HM = (np.stack([np.bincount(A[:, b], minlength=LINES + 1) for b in range(PL)]).ravel() for A in (NT, MP, MM))
SHB = np.arange(PL)[:, None]
def table(nvec, s):                                   # G[b, m] = sum over uncounted a of 2^a min(n_a, m)   (15 b, 769 m)
    return ((~CS[s]).T * POWA[None, :]) @ np.minimum(nvec[:, None], NN[None, :])
def table_photons(nvec, s):                           # combined/machine.py's count of building one table
    Q = np.array([ONES_NN[:n].sum() + (LINES + 1 - n) * ones(n) for n in nvec])
    return int(((~CS[s]) * Q[:, None]).sum())

def pair_by_pair(x, w, C):                            # counted pile of word w, the slow way: pair by pair, line by line
    ax, sx, tw, sw = np.abs(x), np.sign(x), TA[w].astype(np.int64), TS[w].astype(np.int64)
    return sum(int(((sx * sw) * ((ax >> a) & 1) * ((tw >> b) & 1)).sum()) << (a + b) for a in range(PL) for b in range(PL) if C[a, b])

sel = np.flatnonzero((SRC == 0) if SET == "refs" else (SRC == 1) if SET == "windows" else np.ones(len(SRC), bool))
_cut = np.linspace(0, len(sel), PN + 1).astype(np.int64); sel = sel[_cut[PI - 1]:_cut[PI]]  # this part's positions
N = len(sel)
ROWS = ("plain", "signed")
SURV = {r: np.zeros((N, NS), np.int64) for r in ROWS}; ENTER = {r: np.zeros((N, NS), np.int64) for r in ROWS}
WORK = {r: np.zeros((N, NS + 1), np.int64) for r in ROWS}
BW = {r: np.zeros((N, NS, 4), np.int64) for r in ROWS}          # photons: tables, bound adds, divides, standing amounts
FINW = {r: np.zeros((N, 2), np.int64) for r in ROWS}            # photons: top divide and standing amounts, words read to the end
TOP_ST = {r: np.zeros((N, 5), np.int64) for r in ROWS}; WIN_ST = {r: np.zeros(N, np.int64) for r in ROWS}
FIN = {r: ([], [], [0]) for r in ROWS}
PLAIN = np.zeros(N, np.int64); TOP_PL = np.zeros((N, 5), np.int64); WIN_PL = np.zeros(N, np.int64)
BAD = {"bound": 0, "bound_signed": 0, "final_vs_plain": 0, "final_vs_plain_signed": 0, "pieces": 0, "pairs_sum": 0,
       "pieces_checked": 0, "fast_bound_vs_bound": 0, "signed_wider_than_plain": 0, "bounds_checked": 0}
rng_chk = np.random.default_rng(5)

def stage(r, q, s, alive, c, F, bias, lo_p, hi_p, ex, ge):
    """lo/hi from the bounded piles; checks; set-aside; returns the words still in"""
    lo = J.power(lo_p, F) + bias; hi = J.power(hi_p, F) + bias
    BAD["bound" if r == "plain" else "bound_signed"] += int(((lo > ex) | (ex > hi)).sum())
    BW[r][q, s, 2] = (int(ones(lo_p).sum()) + int(ones(hi_p).sum())) * ge
    BW[r][q, s, 3] = 2 * int(BIAS_ONES[alive].sum())
    L5 = np.partition(lo, len(lo) - 5)[len(lo) - 5]
    ENTER[r][q, s] = len(alive); alive = alive[hi >= L5]; SURV[r][q, s] = len(alive)
    return alive

for c0 in range(0, N, BATCH):
    g = sel[c0:c0 + BATCH]; P = len(g)
    X = X_ALL[g]; ax, sx = np.abs(X), np.sign(X)
    F = (WE[0] + EXA[g] + J.ES) * 65536 - LSA[g]
    S = J.power(J.product(X, WH), F[:, None]) + BIAS[None, :]                             # the plain top read, every word
    CNTD = np.zeros((NS, P, NW), np.float64)                                              # counted piles after stages 1..NS
    for b in range(PL):                                                                   # (whole numbers below 2^41: exact)
        live = [s for s in range(NS) if AMASK[s, b]]
        if not live: continue
        Pb = (TS * ((TA >> b) & 1).astype(np.int8)).astype(np.float64)                    # the stamp's photons at b, with direction
        for s in live:
            prod = (sx * (ax & AMASK[s, b])).astype(np.float64) @ Pb.T                    # the light's places met with b: each sum
            prod *= float(1 << b); CNTD[s] += prod                                        # below 768 x 2^15, whole: exact
        del Pb, prod
    CNTD = CNTD.astype(np.int64)
    ALV = {r: np.zeros((NS + 1, P, NW), np.float32) for r in ROWS}
    for r in ROWS: ALV[r][0] = 1
    for p in range(P):
        q = c0 + p; chk = c0 == 0 and p < 4
        nx = ((ax[p][None, :] >> np.arange(PL)[:, None]) & 1).sum(1)
        nxp = (((ax[p] * (sx[p] > 0))[None, :] >> np.arange(PL)[:, None]) & 1).sum(1); nxm = nx - nxp
        ge = int(ones(J.EXP2[np.bitwise_and(-int(LSA[g[p]]), 0xFFFF)]))                # the gain's photons (combined's E_ones)
        alive = {r: np.arange(NW) for r in ROWS}
        for s in range(NS):
            # plain row
            a = alive["plain"]; full = len(a) == NW
            G = table(nx, s); G2 = (G << SHB).ravel(); OG = ones(G).ravel()
            R = (G2.take(IX_N) if full else G2.take(IX_N[a])).sum(1)
            if chk:
                BAD["fast_bound_vs_bound"] += int((R != bound(nx, s, a)).sum()); BAD["bounds_checked"] += len(a)
            c = CNTD[s, p, a]
            BW["plain"][q, s, 0] = table_photons(nx, s)
            BW["plain"][q, s, 1] = (int((HN * OG).sum()) if full else int(OG.take(IX_N[a]).sum())) + 2 * int(ones(R).sum())
            alive["plain"] = stage("plain", q, s, a, c, F[p], BIAS[a], c - R, c + R, S[p, a], ge)
            ALV["plain"][s + 1, p, alive["plain"]] = 1
            # signed row
            a = alive["signed"]; full = len(a) == NW
            Gp, Gm = table(nxp, s), table(nxm, s)
            Gp2, Gm2 = (Gp << SHB).ravel(), (Gm << SHB).ravel(); OGp, OGm = ones(Gp).ravel(), ones(Gm).ravel()
            ip, im = (IX_P, IX_M) if full else (IX_P[a], IX_M[a])
            U = Gp2.take(ip).sum(1) + Gm2.take(im).sum(1); D = Gp2.take(im).sum(1) + Gm2.take(ip).sum(1)
            if chk:
                Rs = bound(nx, s, a); BAD["signed_wider_than_plain"] += int(((U > Rs) | (D > Rs)).sum())
            c = CNTD[s, p, a]
            BW["signed"][q, s, 0] = table_photons(nxp, s) + table_photons(nxm, s)
            tv = (int(((HP + HM) * (OGp + OGm)).sum()) if full else
                  int(OGp.take(ip).sum() + OGm.take(im).sum() + OGp.take(im).sum() + OGm.take(ip).sum()))
            BW["signed"][q, s, 1] = tv + int(ones(U).sum()) + int(ones(D).sum())
            alive["signed"] = stage("signed", q, s, a, c, F[p], BIAS[a], c - D, c + U, S[p, a], ge)
            ALV["signed"][s + 1, p, alive["signed"]] = 1
        if chk:                                                                           # the pieces, pair by pair
            for w in rng_chk.choice(NW, 16, replace=False):
                for s in range(NS):
                    BAD["pieces"] += pair_by_pair(X[p], w, CS[s]) != CNTD[s, p, w]; BAD["pieces_checked"] += 1
        TOP_PL[q] = WPERM[top5(S[p:p + 1])[0]]; WIN_PL[q] = WPERM[J.race(S[p])]
        for r in ROWS:
            a = alive[r]
            pile = J.product(X[p:p + 1], WH if len(a) == NW else WH[:, a])[0]          # read exactly
            exact = J.power(pile, F[p]) + BIAS[a]
            BAD["final_vs_plain" if r == "plain" else "final_vs_plain_signed"] += int((exact != S[p, a]).sum())
            FINW[r][q] = (int(ones(pile).sum()) * ge, int(BIAS_ONES[a].sum()))
            Sp = np.full(NW, FLOOR, np.int64); Sp[a] = exact
            TOP_ST[r][q] = WPERM[top5(Sp[None])[0]]; WIN_ST[r][q] = WPERM[J.race(Sp)]
            k = np.arange(len(a)) if len(a) <= KEEPN else np.argpartition(-exact, KEEPN - 1)[:KEEPN]   # saved: the words
            ids = WPERM[a[k]]; o = np.argsort(ids)                                        # still in, or their best KEEPN
            FIN[r][0].append(ids[o]); FIN[r][1].append(exact[k][o]); FIN[r][2].append(FIN[r][2][-1] + len(k))
    del CNTD, S
    work = {r: np.zeros((NS + 1, P), np.int64) for r in ROWS}; allpairs = np.zeros(P, np.int64)
    A2 = {r: ALV[r].reshape((NS + 1) * P, NW) for r in ROWS}
    for b in range(PL):
        Bb = ((TA >> b) & 1).astype(np.float32)
        CN = {r: (A2[r] @ Bb).astype(np.int64).reshape(NS + 1, P, LINES) for r in ROWS}   # words still in with a photon at b, per line
        for s in range(NS + 1):
            Lc = np.bitwise_count(ax & NMASK[s, b]).astype(np.int64)                     # light photons at the new places met with b
            for r in ROWS: work[r][s] += (Lc * CN[r][s]).sum(1)
            allpairs += (Lc * CNT_ALL[b][None, :]).sum(1)
        del Bb, CN
    plain = (np.bitwise_count(ax).astype(np.int64) * PPH[None, :]).sum(1)
    BAD["pairs_sum"] += int((allpairs != plain).sum())
    for r in ROWS: WORK[r][c0:c0 + P] = work[r].T
    PLAIN[c0:c0 + P] = plain
    del ALV, A2
    print(f"  {c0 + P}/{N} positions ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)", flush=True)

tag = f"{SCHED}_{LIGHT}" + ("_order" if ORDER else "") + ("" if SET == "both" else f"_{SET}") + ("" if PN == 1 else f"_p{PI}of{PN}")
out = dict(sel=sel, ks=np.array(KS), plain=PLAIN, top_pl=TOP_PL, win_pl=WIN_PL,
           bad=np.array([BAD[k] for k in sorted(BAD)]), bad_keys=np.array(sorted(BAD)))
for r in ROWS:
    sx_ = "" if r == "plain" else "_s"
    out.update({f"surv{sx_}": SURV[r], f"enter{sx_}": ENTER[r], f"work{sx_}": WORK[r], f"bw{sx_}": BW[r], f"finw{sx_}": FINW[r],
                f"top_st{sx_}": TOP_ST[r], f"win_st{sx_}": WIN_ST[r], f"fin_idx{sx_}": np.concatenate(FIN[r][0]),
                f"fin_sc{sx_}": np.concatenate(FIN[r][1]), f"fin_off{sx_}": np.array(FIN[r][2])})
np.savez(f"{HERE}/collapse_top/fine_{tag}.npz", **out)

def pct(a, b): return int(round(100 * a / b))
print(f"\nfine grid: sched={SCHED} light={LIGHT} order={ORDER} set={SET} part={PI}/{PN}: {N} positions; stages top {KS} then all")
print("  checks: " + ", ".join(f"{k} {BAD[k]}" for k in sorted(BAD)))
pl = PLAIN.mean()
for r in ROWS:
    same5 = (TOP_ST[r] == TOP_PL).all(1)
    print(f"  [{r} bound] top 5 (set and order) same as the plain read: {int(same5.sum())}/{N}; same word (race): {int((WIN_ST[r] == WIN_PL).sum())}/{N}")
    rr = SRC[sel] == 0
    if rr.any() and LIGHT == "real" and ORDER == 0:
        print(f"    refs: race winner = the winner saved with the light: {int((WIN_ST[r][rr] == REFW[sel[rr]]).sum())}/{int(rr.sum())}")
    for s, k in enumerate(KS):
        v = SURV[r][:, s]
        print(f"    after top {k:2d}: survivors median {int(np.median(v)):6d}, mean {int(round(v.mean())):6d}, min {v.min():6d}, max {v.max():6d};"
              f"  stage {s + 1} meetings {int(round(WORK[r][:, s].mean() / 1e6))}M")
    print(f"    last stage (all) meetings {int(round(WORK[r][:, NS].mean() / 1e6))}M")
    tot = WORK[r].sum(1).mean()
    print(f"    plain top read here {int(round(pl / 1e6))}M per position; staged {int(round(tot / 1e6))}M = {pct(tot, pl)}% of plain;"
          f" bound work in photons {int(round(BW[r].sum((1, 2)).mean() / 1e6))}M = {pct(BW[r].sum((1, 2)).mean(), pl)}% of plain")
    if SET != "refs" and LIGHT == "real" and ORDER == 0:                                  # judge5's reference top 5
        REF = np.load(f"{HERE}/judge5_ref.npz")["top5"]; wq = np.flatnonzero(SRC[sel] == 1)
        NREF = int((SRC == 0).sum()); rows = [(q, *divmod(int(sel[q]) - NREF, 20)) for q in wq]                 # window i, place j
        assert all(WIN[sel[q]] == i for q, i, j in rows)
        same_ref = sum(int((TOP_ST[r][q] == REF[i, j]).all()) for q, i, j in rows); same_pl = sum(int((TOP_PL[q] == REF[i, j]).all()) for q, i, j in rows)
        print(f"    windows: top 5 (set and order) = judge5's reference top 5: staged {same_ref}/{len(wq)}, plain {same_pl}/{len(wq)}")
print(f"  ({time.time() - t0:.0f}s, max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)")
