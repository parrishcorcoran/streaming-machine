"""THE SPOT BOUND: THE TOP ROW DOES NOT NEED TO BE FINISHED IF THE BIG PILES ARE NOT IN THE SPOTS.

His words: "Also the top row doesn't need to be finished if the big piles are not in the spots."

HIS PICTURE, RESTATED. At the top every word's row meets the light, line by line, place by place. A word's row does not
have to be read to the end once it is sure that what is left cannot lift it into the top 5. What is left is the pairs
not yet met. The bound used so far (staged_top.py) says: at place pair (a, b) the pairs left are at most
min(light lines lit at a, word lines lit at b), as if every lit light line sat on a lit word line, all the big piles in
the same spots. Where the light's lit lines and the word's lit lines sit in different spots, that is far too much.

THE SPOT BOUND (exact). The 768 lines are split into G groups at random (a fixed seed; random only, never by likeness).
Stored once for every word: in each group, how many of its lines are lit at each place. Counted once per position: in
each group, how many of the light's lines are lit at each place. Within one group the pairs at (a, b) are at most the
smaller of the two counts, so
    rest <= sum over unread (a, b) of 2^(a+b) x sum over groups of min(light's count, word's count).
Two writings:
  signed digits (single waves, no two lit side by side): ups and downs are counted apart. An up pair is up meeting up or
    down meeting down; a down pair is up meeting down. The rest is at most the up pairs that can line up,
    sum 2^(a+b) [min(u, U) + min(d, D)], and at least minus the down pairs that can, sum 2^(a+b) [min(u, D) + min(d, U)]:
    the two sides bounded apart, each exact.
  plain photons (the machine's own sign and photons, 0/1 per place): lit counts only, the same bound on both sides.
    G = 1 is exactly the bound of staged_top.py.
The bound is computed the way the current machine does it: per stage, a table per (group, word place) built from the
light, m = 1 .. the light's largest count in the group; each word reads its stored count and adds the table value.

STAGES (fixed in advance; thresholds by the pair's place value, so both writings meet the same places at each stage):
  diag   the top 2, 4, ..., 28 diagonals (a + b >= 27, 25, ..., 1), then all
  light  the light's top 2, 4, ..., 12 places (a >= 13, 11, ..., 3), then all
(in signed digits places run to 15, so stage 1 also holds a + b = 29, 30, or the light's place 15.)

WORK, in photons:
  meetings   light photons x word photons on each line for the place pairs met, over the words still in (in the
             writing used: signed digits meet lit digits)
  the bound's own work (spot_bound.c): stored counts read, table values added, tables built, the two bound adds
             (pile + Rh, pile - Rl), the light's lit pixels counted into the groups (once a position, per G)
  stated apart: the divides of the bounds and their standing amounts (2 a word a stage), compares (a count).
  The plain read here: 916M photon meetings a position, sign and photons (the 6,480 positions).

CHECKS: the counted pile with every pair = the plain read, every word, every position; the stages' meetings add up to
the plain meetings in this writing; lo <= exact <= hi for every word still in at every stage; the pieces (counted pile,
both bounds, table photons, counts read, meetings) against a slow recount on 3 positions x 12 words, every stage; the
words read to the end re-read on their own and equal to the plain read; top 5 (set and order) equal to the plain
read's at every position; refs: the race winner equal to the one saved with the light.
CONTROLS: light=shuffled (each position's light across its lines, as staged_top.py); seed=2 (the groups drawn again).
ORDER-FREE: order=1 puts the words and the lines (light, word table and the lines' groups together) in another order.

PREDICTIONS (written before this script first ran):
  the task's:  the spot bound sets words aside several stages earlier; with G = 16-32 the top read drops to 15-30% of
               plain including the bound's own work, exact everywhere; the shuffled light gains less.
  Claude's:    exact everywhere (bound broken 0, top 5 6,480/6,480).
               At the top places the light is lit on about 5 lines (373, 447, 64, ...), where no word is above 184,
               so there the big piles are indeed not in the spots and the spot bound removes those pairs. But those
               pairs are read in the first stages. The bound that decides is the one just under the read front, at
               places where about half of all lines are lit on both sides; a group of 24 lines then nearly always
               holds lit lines of both, and min(n_g, m_g) stays close to half the group: the spot bound is only
               10-25% below the one-group bound there. One fine stage (2 diagonals) shrinks the bound about 4x, so
               words are set aside at most one stage earlier; survivors at the deciding stage fall less than 2x;
               meetings fall a few % of plain.
               The bound's own work grows with G: G x 16 table values a word a stage (x4 in signed digits, two sides
               of two counts), against about 18,000 photon meetings for a word's whole row. At G = 32 that is a large
               share of a whole row every stage, so with it G = 16-32 cost more than G = 1, likely more than plain.
               Signed digits, diagonals: G = 1 about 35-45% of 916M with its own bound work; G = 32 above G = 1.
               Plain photons, diagonals, G = 1: 50-60%. Shuffled light: about the same gain from the groups.
               Another seed: the same to a few %.

    python3 collapse_top/spot_top.py write=signed|plain [light=real|shuffled] [seed=1|2] [order=0|1] [G=1,8,16,32]
                                     [set=all|sub] [batch=16] [limit=0] [part=i/n]
      -> collapse_top/spot_<write>_<light>_s<seed>[_order][_<set>][_p<i>of<n>].npz  (parts: spot_join.py)
"""
import sys, os, time, resource, warnings, ctypes, types, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; D = f"{HERE}/collapse_top"; sys.path.insert(0, HERE); os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
WRITE, LIGHT, SEED, ORDER = ARG.get("write", "signed"), ARG.get("light", "real"), int(ARG.get("seed", 1)), int(ARG.get("order", 0))
GS = [int(g) for g in ARG.get("G", "1,8,16,32").split(",")]; SET = ARG.get("set", "all")
BATCH, LIMIT, THREADS = int(ARG.get("batch", 16)), int(ARG.get("limit", 0)), int(ARG.get("threads", 10))
PART = tuple(int(v) for v in ARG.get("part", "1/1").split("/"))                       # the i-th of n equal slices
assert WRITE in ("signed", "plain") and LIGHT in ("real", "shuffled") and SET in ("all", "sub")
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]    # the machine's own steps
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')                                  # the word table read below, int16
_argv = sys.argv; sys.argv = ["x"]; _g = {}; exec(_src, _g); sys.argv = _argv
J = types.SimpleNamespace(**_g); M = J.M
def top5(S):                                           # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
t0 = time.time()
NW, LINES = 50257, 768; FLOOR = -(1 << 62); KEEPN = 256; NPARTS = 7
PARTS = ["stored counts read", "table values added", "tables built", "bound adds", "divides of the bounds",
         "standing amounts on the bounds", "compares (a count)"]
PL = 16 if WRITE == "signed" else 15
DIAG_T = [29 - 2 * k for k in range(1, 15)]            # 27, 25, ..., 1
LIGHT_C = [15 - 2 * k for k in range(1, 7)]            # 13, 11, ..., 3
EXP2 = np.ascontiguousarray(J.EXP2, dtype=np.int64)

# ---------- the light, the word table, the order and the shuffle (as staged_top.py) ----------
L = np.load(f"{D}/light_top.npz")
SRC, WIN, REFW = L["src"], L["win"], L["refs_winner"]
X_ALL = L["x16"].astype(np.int64); EXA, LSA = L["ex"], L["LS"]
WE = M["head.e"]; BIAS = M["head.b"].astype(np.int64); assert (WE == WE[0]).all()
TT = np.ascontiguousarray(M["head.w"].T)               # (NW, 768) int16: each word's row
WPERM, LPERM = np.arange(NW), np.arange(LINES)
if ORDER:
    rng = np.random.default_rng(99); WPERM = rng.permutation(NW); LPERM = rng.permutation(LINES)
    TT = np.ascontiguousarray(TT[WPERM][:, LPERM]); BIAS = BIAS[WPERM]; X_ALL = X_ALL[:, LPERM]
if LIGHT == "shuffled":
    rng = np.random.default_rng(2026); X_ALL = np.stack([x[rng.permutation(LINES)] for x in X_ALL])
BIAS = np.ascontiguousarray(BIAS); BONES = ones(BIAS).astype(np.int32)

# ---------- the groups: random, by seed, on the machine's own line numbers (they travel with the lines) ----------
GP = np.random.default_rng(SEED).permutation(LINES); RANK = np.empty(LINES, np.int64); RANK[GP] = np.arange(LINES)
GROUP = {G: RANK[LPERM] * G // LINES for G in GS}      # group of each line, in this run's line order
for G in GS: assert np.array_equal(np.bincount(GROUP[G], minlength=G), np.full(G, LINES // G))
ONEHOT = {G: (GROUP[G][:, None] == np.arange(G)[None, :]).astype(np.float32) for G in GS}

def digits(V):                                         # every value as two sheets: its +1 places and its -1 places
    A = np.abs(V)
    if WRITE == "plain": up, dn = A, np.zeros_like(A)
    else: h = A >> 1; t = A + h; c = h ^ t; up = t & c; dn = h & c                 # single waves (no two lit side by side)
    assert np.array_equal(up - dn, A) and not (up & dn).any()
    if WRITE == "signed": assert not ((up | dn) & ((up | dn) >> 1)).any()
    neg = V < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)

# ---------- the word table in this writing: digit planes, lit counts per line, stored counts per group ----------
DIG = np.zeros((PL, NW, LINES), np.int8)               # word w's digit at place b on line i: -1, 0, +1
LITTOT = np.zeros((NW, LINES), np.int8)                # word w's lit digits on line i
PPH = np.zeros(LINES, np.int64)                        # sign-and-photons photons of all words per line (the 916M basis)
WA = {G: np.zeros((NW, PL, G), np.uint16) for G in GS}; WB = {G: np.zeros((NW, PL, G), np.uint16) for G in GS} if WRITE == "signed" else None
OH_ALL = np.concatenate([ONEHOT[G] for G in GS], 1); GOFF = np.cumsum([0] + GS)
for c0 in range(0, NW, 2048):
    T = TT[c0:c0 + 2048].astype(np.int64); up, dn = digits(T); assert ((up | dn) >> PL == 0).all()
    PPH += ones(T).sum(0)
    for b in range(PL):
        u = ((up >> b) & 1).astype(np.int8); d = ((dn >> b) & 1).astype(np.int8)
        DIG[b, c0:c0 + 2048] = u - d; LITTOT[c0:c0 + 2048] += u + d
        if WRITE == "signed":
            cu = (u.astype(np.float32) @ OH_ALL).astype(np.int64); cd = (d.astype(np.float32) @ OH_ALL).astype(np.int64)
            for i, G in enumerate(GS): WA[G][c0:c0 + 2048, b] = cu[:, GOFF[i]:GOFF[i + 1]]; WB[G][c0:c0 + 2048, b] = cd[:, GOFF[i]:GOFF[i + 1]]
        else:
            cu = ((u + d).astype(np.float32) @ OH_ALL).astype(np.int64)
            for i, G in enumerate(GS): WA[G][c0:c0 + 2048, b] = cu[:, GOFF[i]:GOFF[i + 1]]
    del T, up, dn
assert np.array_equal(PPH, np.bitwise_count(np.abs(TT.astype(np.int32))).sum(0, dtype=np.int64))
LITCOL = LITTOT.sum(0, dtype=np.int64)                 # lit digits of all words per line
STORE = {}                                             # the stored counts' size
for G in GS:
    mx = int(max(WA[G].max(), WB[G].max() if WB else 0)); room = (LINES // G).bit_length()
    STORE[G] = (mx, room, NW * PL * G * room * (2 if WB else 1) // 8, NW * PL * G * mx.bit_length() * (2 if WB else 1) // 8)
print(f"write={WRITE} light={LIGHT} seed={SEED} order={ORDER} G={GS} set={SET}: word table ready "
      f"({time.time() - t0:.0f}s, {rss():.2f} GB)", flush=True)

LIB = ctypes.CDLL(f"{D}/spot_bound.dylib")
def ptr(a): return a.ctypes.data_as(ctypes.c_void_p) if a is not None else None
def unread_masks(sched):
    a_, b_ = np.arange(PL)[:, None], np.arange(PL)[None, :]
    if sched == "diag": return np.stack([(a_ + b_ < t) for t in DIAG_T]).astype(np.uint8)
    return np.stack([np.broadcast_to(a_ < c, (PL, PL)) for c in LIGHT_C]).astype(np.uint8)
UNREAD = {s: np.ascontiguousarray(unread_masks(s)) for s in ("diag", "light")}
KS = {"diag": len(DIAG_T), "light": len(LIGHT_C)}

# ---------- positions ----------
if SET == "all": sel = np.arange(len(SRC))
else: sel = np.flatnonzero((SRC == 0) | ((SRC == 1) & (WIN % 6 == 0)))     # CONTROLS: refs + every 6th window
if LIMIT: sel = sel[:LIMIT]
sel = np.array_split(sel, PART[1])[PART[0] - 1]                                     # each part under 10 minutes
N = len(sel)
CFG = [(G, s) for G in GS for s in ("diag", "light")]
R = {c: dict(surv=np.zeros((N, KS[c[1]]), np.int64), enter=np.zeros((N, KS[c[1]]), np.int64),
             meet=np.zeros((N, KS[c[1]] + 1), np.int64), parts=np.zeros((N, KS[c[1]], NPARTS), np.int64),
             bad=np.zeros(N, np.int64), top=np.zeros((N, 5), np.int64), win=np.zeros(N, np.int64),
             final=np.zeros(N, np.int64), findiv=np.zeros(N, np.int64), finst=np.zeros(N, np.int64), kept=[], koff=[0])
     for c in CFG}
TOP_PL = np.zeros((N, 5), np.int64); WIN_PL = np.zeros(N, np.int64)
PLAIN916 = np.zeros(N, np.int64); PLAINW = np.zeros(N, np.int64); PLDIV = np.zeros(N, np.int64); PLST = np.zeros(N, np.int64)
LIGHTPH = np.zeros(N, np.int64)                        # the light's lit pixels counted into the groups (per G)
UIDS, USC, UOFF = [], [], [0]
BAD = dict(pile_all=0, meet_sum=0, meet_plain=0, final_vs_plain=0, pieces=0, pieces_checked=0)
rng_chk = np.random.default_rng(5); NSAMP = 12

def pieces(p, X, XP, XN, G, sched, w, out, CNTs, MEETs):   # the slow recount of one word at one position, every stage
    ax = [((XP[p] >> a) & 1).astype(np.int64) for a in range(PL)]; dx = [((XN[p] >> a) & 1).astype(np.int64) for a in range(PL)]
    wu = [(DIG[b, w] == 1).astype(np.int64) for b in range(PL)]; wd = [(DIG[b, w] == -1).astype(np.int64) for b in range(PL)]
    grp = GROUP[G]; bad = 0
    def gsum(v): return np.bincount(grp, weights=v, minlength=G).astype(np.int64)
    lu = [gsum(ax[a]) for a in range(PL)]; ld = [gsum(dx[a]) for a in range(PL)]
    Wu = [gsum(wu[b]) for b in range(PL)]; Wd = [gsum(wd[b]) for b in range(PL)]
    if WRITE == "plain": lu = [lu[a] + ld[a] for a in range(PL)]; Wu = [Wu[b] + Wd[b] for b in range(PL)]
    UN = UNREAD[sched]; K = KS[sched]
    for k in range(K + 1):
        read = ~UN[k].astype(bool) if k < K else np.ones((PL, PL), bool)
        prev = ~UN[k - 1].astype(bool) if k > 0 else np.zeros((PL, PL), bool)
        cnt = sum(int(((ax[a] - dx[a]) * (wu[b] - wd[b])).sum()) << (a + b) for a in range(PL) for b in range(PL) if read[a, b])
        mt = sum(int(((ax[a] + dx[a]) * (wu[b] + wd[b])).sum()) for a in range(PL) for b in range(PL) if read[a, b] and not prev[a, b])
        bad += mt != MEETs[k]; BAD["pieces_checked"] += 1
        if k == K: continue
        bad += cnt != CNTs[k]
        Jl = [max(int(lu[a][g]) for a in range(PL)) for g in range(G)]; Jd = [max(int(ld[a][g]) for a in range(PL)) for g in range(G)]
        rh = rl = adds = reads = 0
        for b in range(PL):
            if not UN[k][:, b].any(): continue
            for g in range(G):
                def tab(Lc, m, Jg): m = min(m, Jg); return sum(min(int(Lc[a][g]), m) << (a + b) for a in range(PL) if UN[k][a, b])
                if WRITE == "signed":
                    U, Dd = int(Wu[b][g]), int(Wd[b][g])
                    t1, t2, t3, t4 = tab(lu, U, Jl[g]), tab(ld, Dd, Jd[g]), tab(lu, Dd, Jl[g]), tab(ld, U, Jd[g])
                    rh += t1 + t2; rl += t3 + t4; adds += sum(int(ones(t)) for t in (t1, t2, t3, t4)); reads += int(ones(U) + ones(Dd))
                else:
                    m = int(Wu[b][g]); t1 = tab(lu, m, Jl[g]); rh += t1; rl += t1; adds += int(ones(t1)); reads += int(ones(m))
        bad += int(rh != out[k, 0]) + int(rl != out[k, 1]) + int(adds != out[k, 2]) + int(reads != out[k, 3])
    return bad

for c0 in range(0, N, BATCH):
    g = sel[c0:c0 + BATCH]; P = len(g)
    X = X_ALL[g]; XP, XN = digits(X)
    F = ((WE[0] + EXA[g] + J.ES) * 65536 - LSA[g]).astype(np.int64); GE = ones(EXP2[F & 0xFFFF])
    XD = np.stack([((XP >> a) & 1) - ((XN >> a) & 1) for a in range(PL)]).astype(np.int64)      # (PL, P, 768) digits
    LX = np.abs(XD)
    XH = np.zeros((PL + 1, P, LINES), np.int64); LXP = np.zeros((PL + 1, P, LINES), np.int64)
    for a in range(PL - 1, -1, -1): XH[a] = XH[a + 1] + (XD[a] << a); LXP[a] = LXP[a + 1] + LX[a]
    assert np.array_equal(XH[0], X)
    # the plain read (every word, exact) and the light schedule's counted piles: the light cut below place c, x the table
    rows = np.concatenate([X[None], XH[LIGHT_C]]).reshape(-1, LINES).astype(np.float64)
    OUT = np.zeros((rows.shape[0], NW), np.int64)
    for w0 in range(0, NW, 12800):
        OUT[:, w0:w0 + 12800] = J.product(rows, TT[w0:w0 + 12800].T.astype(np.float64))
    OUT = OUT.reshape(1 + len(LIGHT_C), P, NW); PILE = OUT[0]
    CNT_L = np.ascontiguousarray(OUT[1:].transpose(1, 0, 2)); del OUT, rows
    S = J.power(PILE, F[:, None]) + BIAS[None, :]
    for p in range(P):
        TOP_PL[c0 + p] = WPERM[top5(S[p:p + 1])[0]]; WIN_PL[c0 + p] = WPERM[J.race(S[p])]
    PLAIN916[c0:c0 + P] = ones(X) @ PPH; PLAINW[c0:c0 + P] = LX.sum(0) @ LITCOL
    if WRITE == "plain": BAD["meet_plain"] += int((PLAINW[c0:c0 + P] != PLAIN916[c0:c0 + P]).sum())
    PLDIV[c0:c0 + P] = ones(PILE).sum(1) * GE; PLST[c0:c0 + P] = BONES.sum()
    LIGHTPH[c0:c0 + P] = LX.sum((0, 2))
    # the diagonal schedule's counted piles (and every pair, as a check), and the meetings of both schedules
    TD = DIAG_T + [0]; ED = [np.inf] + DIAG_T                     # stage k reads diagonals [TD[k], ED[k])
    CD = np.zeros((len(TD), P, NW), np.float64); MD = np.zeros((len(TD), P, NW), np.float32)
    for b in range(PL):
        Tb = DIG[b].astype(np.float32)
        Rw = XH[np.clip(np.array(TD) - b, 0, PL)]                   # (15, P, 768): the light cut below place t - b
        lo8, hi8 = (Rw & 255).astype(np.float32), (Rw >> 8).astype(np.float32)
        O = np.concatenate([lo8, hi8]).reshape(-1, LINES) @ Tb.T     # each sum below 2^18: whole, exact
        O = O.reshape(2, len(TD), P, NW); CD += (O[1].astype(np.float64) * 256 + O[0]) * float(1 << b); del O, lo8, hi8
        alo = np.clip(np.array(TD) - b, 0, PL); ahi = np.array([PL if e == np.inf else int(np.clip(e - b, 0, PL)) for e in ED])
        ML = (LXP[alo] - LXP[ahi]).astype(np.float32)                # light photons met with place b at each stage
        MD += (ML.reshape(-1, LINES) @ np.abs(Tb).T).reshape(len(TD), P, NW); del Tb, ML
    CD = CD.astype(np.int64)
    BAD["pile_all"] += int((CD[-1] != PILE).sum())                  # every pair = the plain read
    CNT_D = np.ascontiguousarray(CD[:-1].transpose(1, 0, 2)); del CD
    MEET_D = np.ascontiguousarray(MD.transpose(1, 0, 2)).astype(np.int32); del MD
    LC = [np.inf] + LIGHT_C + [0]
    ML = np.stack([LXP[LC[k + 1]] - (0 if LC[k] == np.inf else LXP[LC[k]]) for k in range(len(LIGHT_C) + 1)]).astype(np.float32)
    MEET_L = np.ascontiguousarray((ML.reshape(-1, LINES) @ LITTOT.T.astype(np.float32)).reshape(len(LIGHT_C) + 1, P, NW).transpose(1, 0, 2)).astype(np.int32)
    del ML
    BAD["meet_sum"] += int((MEET_D.sum(1, dtype=np.int64) != MEET_L.sum(1, dtype=np.int64)).sum())
    BAD["meet_plain"] += int((MEET_D.sum((1, 2), dtype=np.int64) != PLAINW[c0:c0 + P]).sum())
    keep_u = [set() for _ in range(P)]
    for G in GS:
        if WRITE == "signed":
            LA = np.ascontiguousarray(((XD == 1).astype(np.float32).reshape(-1, LINES) @ ONEHOT[G]).reshape(PL, P, G).transpose(1, 0, 2)).astype(np.int32)
            LB = np.ascontiguousarray(((XD == -1).astype(np.float32).reshape(-1, LINES) @ ONEHOT[G]).reshape(PL, P, G).transpose(1, 0, 2)).astype(np.int32)
        else:
            LA = np.ascontiguousarray((LX.astype(np.float32).reshape(-1, LINES) @ ONEHOT[G]).reshape(PL, P, G).transpose(1, 0, 2)).astype(np.int32); LB = None
        for sched, CNT, MEET in (("diag", CNT_D, MEET_D), ("light", CNT_L, MEET_L)):
            K = KS[sched]; r = R[(G, sched)]
            EXIT = np.zeros((P, NW), np.int8); ENTER = np.zeros((P, K), np.int64); SURV = np.zeros((P, K), np.int64)
            MEETS = np.zeros((P, K + 1), np.int64); PRT = np.zeros((P, K, NPARTS), np.int64); BD = np.zeros(P, np.int64)
            ns = NSAMP if c0 == 0 else 0
            SAMP = np.ascontiguousarray(rng_chk.choice(NW, (P, NSAMP)).astype(np.int32)) if ns else np.zeros((P, 1), np.int32)
            SOUT = np.zeros((P, max(ns, 1), K, 4), np.int64)
            LIB.collapse(P, NW, PL, G, K, int(WRITE == "signed"), ptr(UNREAD[sched]), ptr(CNT), ptr(MEET), ptr(S),
                         ptr(np.ascontiguousarray(F)), ptr(np.ascontiguousarray(GE)), ptr(BIAS), ptr(BONES), ptr(EXP2),
                         ptr(WA[G]), ptr(WB[G] if WB else None), ptr(LA), ptr(LB), ptr(EXIT), ptr(ENTER), ptr(SURV),
                         ptr(MEETS), ptr(PRT), ptr(BD), ns, ptr(SAMP), ptr(SOUT), THREADS)
            r["surv"][c0:c0 + P] = SURV; r["enter"][c0:c0 + P] = ENTER; r["meet"][c0:c0 + P] = MEETS
            r["parts"][c0:c0 + P] = PRT; r["bad"][c0:c0 + P] = BD
            if ns:
                for p in range(3):
                    for s in range(NSAMP):
                        w = int(SAMP[p, s]); BAD["pieces"] += pieces(p, X, XP, XN, G, sched, w, SOUT[p, s],
                                                                     [CNT[p, k, w] for k in range(K)], [MEET[p, k, w] for k in range(K + 1)])
            for p in range(P):
                alive = np.flatnonzero(EXIT[p] == K)
                if sched == "diag":   # the words still in, read to the end on their own: the last diagonal's pairs
                    rest = (XD[0, p][None, :] * DIG[0, alive].astype(np.int64)).sum(1)
                else:                 # the light's places below the last stage, x the whole row
                    rest = J.product((X[p] - XH[LIGHT_C[-1], p])[None, :].astype(np.float64), TT[alive].T.astype(np.float64))[0]
                pile = CNT[p, K - 1, alive] + rest; exact = J.power(pile, F[p]) + BIAS[alive]
                BAD["final_vs_plain"] += int((exact != S[p, alive]).sum()) + int((pile != PILE[p, alive]).sum())
                Sp = np.full(NW, FLOOR, np.int64); Sp[alive] = exact
                r["top"][c0 + p] = WPERM[top5(Sp[None])[0]]; r["win"][c0 + p] = WPERM[J.race(Sp)]
                r["final"][c0 + p] = len(alive); r["findiv"][c0 + p] = ones(pile).sum() * GE[p]; r["finst"][c0 + p] = BONES[alive].sum()
                k = np.arange(len(alive)) if len(alive) <= KEEPN else np.argpartition(-exact, KEEPN - 1)[:KEEPN]
                ids = np.sort(WPERM[alive[k]]); r["kept"].append(ids); r["koff"].append(r["koff"][-1] + len(ids))
                keep_u[p] |= set(ids.tolist())
    for p in range(P):                                  # the exact scores of every word kept by any run, once
        u = np.array(sorted(keep_u[p]), np.int64); inv = np.empty(NW, np.int64); inv[WPERM] = np.arange(NW)
        UIDS.append(u); USC.append(S[p, inv[u]]); UOFF.append(UOFF[-1] + len(u))
    del CNT_D, CNT_L, MEET_D, MEET_L, S, PILE
    print(f"  {c0 + P}/{N} positions ({time.time() - t0:.0f}s, {rss():.2f} GB)", flush=True)

tag = f"{WRITE}_{LIGHT}_s{SEED}" + ("_order" if ORDER else "") + ("" if SET == "all" else f"_{SET}") + (f"_lim{LIMIT}" if LIMIT else "") + (f"_p{PART[0]}of{PART[1]}" if PART[1] > 1 else "")
out = dict(sel=sel, gs=np.array(GS), diag_t=np.array(DIAG_T), light_c=np.array(LIGHT_C), pl=PL, top_pl=TOP_PL, win_pl=WIN_PL,
           plain916=PLAIN916, plainw=PLAINW, pldiv=PLDIV, plst=PLST, lightph=LIGHTPH,
           uids=np.concatenate(UIDS).astype(np.int32), usc=np.concatenate(USC), uoff=np.array(UOFF),
           bad_keys=np.array(sorted(BAD)), bad=np.array([BAD[k] for k in sorted(BAD)]),
           store=np.array([STORE[G] for G in GS]), parts_names=np.array(PARTS))
for (G, s), r in R.items():
    for k, v in r.items():
        out[f"{k}_{G}_{s}"] = (np.concatenate(v).astype(np.int32) if k == "kept" else np.array(v)) if isinstance(v, list) else v
np.savez(f"{D}/spot_{tag}.npz", **out)

def M_(v): return f"{v / 1e6:.1f}M"
pl = PLAIN916.mean()
print(f"\nspot_{tag}: {N} positions; plain read {M_(pl)} a position (sign and photons), in this writing {M_(PLAINW.mean())}")
print("  checks: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())))
for (G, s), r in R.items():
    same = int((r["top"] == TOP_PL).all(1).sum()); bnd = int(r["bad"].sum())
    meet = r["meet"].sum(1).mean(); prt = r["parts"].sum(1).mean(0)
    own = prt[:4].sum() + LIGHTPH.mean()
    print(f"  G={G:2d} {s:5s}: top 5 = plain {same}/{N}, bound broken {bnd}; survivors (median) " +
          " ".join(str(int(np.median(r['surv'][:, k]))) for k in range(r["surv"].shape[1])) +
          f"; meetings {M_(meet)} + bound's own {M_(own)} = {100 * (meet + own) / pl:.1f}% of plain"
          f" (divides+standing of the bounds {M_(prt[4] + prt[5])})")
r = SRC[sel] == 0
if r.any() and LIGHT == "real":
    for (G, s), rr in R.items():
        print(f"  refs G={G} {s}: race winner = saved {int((rr['win'][r] == REFW[sel[r]]).sum())}/{int(r.sum())}", end=";")
    print()
for G in GS: print(f"  stored counts G={G}: largest {STORE[G][0]}, room {STORE[G][1]} bits -> {STORE[G][2] / 1e6:.1f} MB "
                   f"({STORE[G][3] / 1e6:.1f} MB at the largest count's width); word table 77.19 MB (16-bit)")
print(f"  ({time.time() - t0:.0f}s, max memory {rss():.2f} GB)")
