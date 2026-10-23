"""THE SPOT BOUND, A CHECKER'S OWN REBUILD (written apart from spot_top.py and spot_bound.c; whole numbers only).

His words: "Also the top row doesn't need to be finished if the big piles are not in the spots."

THE PICTURE. At a read position the light x (768 lines) meets every word's row T_w. Every value is written as digits at
places 0..PL-1: signed digits (-1, 0, +1, no two lit side by side, PL = 16) or plain photons (its sign x 0/1, PL = 15).
A light digit at place a meeting a word digit at place b on the same line is a pair worth 2^(a+b): an up pair if the
two digits agree, a down pair if they differ. The unread part of a word after a stage is the sum of 2^(a+b) x (up pairs
- down pairs) over the place pairs not yet met.
The 768 lines are cut into G groups at random: the lines in the order of default_rng(seed).permutation(768), cut into
G equal runs. Inside one group, at (a, b):
    up pairs   <= min(light +1 lines, word +1 lines) + min(light -1 lines, word -1 lines)
    down pairs <= min(light +1 lines, word -1 lines) + min(light -1 lines, word +1 lines)
(plain photons: lit pairs <= min(light lit lines, word lit lines), the same bound on both sides).

HOW IT IS COUNTED HERE (not by tables). The sum over groups of min(n_g, m_g) is the number of (group, t), t = 1..group
size, with n_g >= t and m_g >= t. Each word at place b is written once as a row of 768 marks [m_g >= t]; the light at
place a as a column of 768 marks [n_g >= t]; one product of the two gives the group bound of every word at every (a, b).
The pairs themselves are counted the same way, line by line: [word digit = +1 | = -1] against [light digit = +1 | = -1].
So the exact unread part, the bound and the meetings all come from counts at every (word, a, b), and the stages only
add them up with 2^(a+b).

CHECKS (every word, every stage, every place pair):
  - every pair added up = the plain read x . T_w (the machine's product), every word, every position
  - at every (word, a, b): up pairs <= the up bound and down pairs <= the down bound (plain: lit pairs <= the bound)
  - at every stage: -low bound <= exact unread part <= high bound, every word (not only the words still in), and at the
    score: power(counted - low) + bias <= score <= power(counted + high) + bias
  - a slow recount of the bound by direct min() group by group, and of the table photons, on sampled words
THE COLLAPSE (the same rule as the task's: set aside when the high score is below the 5th best low score):
  survivors, words entering, meetings, the bound's own work in photons (stored counts read, table values added, tables
  built, bound adds, divides, standing amounts, compares), the words read to the end, the top 5 and the race winner.
  The table photons are counted as a sum of steps: pc(table at m) = sum over t <= m of (pc(table at t) - pc(at t-1)),
  so each word's total comes out of the same mark rows as its bound.

    python3 collapse_top/spot_check_rebuild.py mode=bound write=signed|plain       (100 positions, G = 1, 8, 16, 32,
                                                                                    real light and a shuffled light)
    python3 collapse_top/spot_check_rebuild.py mode=top write=signed|plain [part=i/n]   (refs 480 + windows 0..99, G = 16)
      -> collapse_top/spot_check_<mode>_<write>[_p<i>of<n>].npz, summary printed
"""
import sys, os, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every count checked
HERE = "/Users/abundancemachine/streaming-machine"; D = f"{HERE}/collapse_top"
ARG = dict(a.split("=") for a in sys.argv[1:])
MODE, WRITE = ARG.get("mode", "bound"), ARG.get("write", "signed")
assert MODE in ("bound", "top") and WRITE in ("signed", "plain")
GS = [int(g) for g in ARG.get("G", "1,8,16,32" if MODE == "bound" else "16").split(",")]
BATCH = int(ARG.get("batch", 8)); PART = tuple(int(v) for v in ARG.get("part", "1/1").split("/")); SEED = int(ARG.get("seed", 1))
NW, LINES, FLOOR = 50257, 768, -(1 << 62)
PL = 16 if WRITE == "signed" else 15
t0 = time.time()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
def pc(V): return np.bitwise_count(np.abs(np.asarray(V, dtype=np.int64))).astype(np.int64)

# ---------- the machine's top: the word table, its bias, the gain (base2.py's numbers) ----------
M = np.load(f"{HERE}/base2_gpt2.npz")
assert "ES, SC = 16, 10" in open(f"{HERE}/base2.py").read(); ES = 16
EXP2 = M["exp2"].astype(np.int64); WE = M["head.e"].astype(np.int64); BIAS = M["head.b"].astype(np.int64)
assert (WE == WE[0]).all()
TW = M["head.w"]                                       # (768, NW) int16
L = np.load(f"{D}/light_top.npz")
X_ALL = L["x16"].astype(np.int64); EXA, LSA, SRC, WINA, REFW = L["ex"], L["LS"], L["src"], L["win"], L["refs_winner"]
assert np.abs(X_ALL).max() < (1 << 15) and int(TW.min()) > -(1 << 15) and int(TW.max()) < (1 << 15)

def power(X, F):                                       # X x 2^(F/65536): the fine part read from the table, a rounded shift
    X = np.asarray(X, np.int64); F = int(F); I, f = F >> 16, F & 0xFFFF
    assert np.abs(X).max() < (1 << 46)                 # X x 2^17 stays below 2^63
    v = X * EXP2[f]; s = 16 - I
    return (v + (1 << (s - 1))) >> s if s > 0 else v << (-s)
def race(V): return int(np.argmax(V))                 # the first of the largest
def top5(S):                                           # judge5.top5
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; o = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, o, 1)

def digits(V):                                         # (PL, ...) int8 in -1/0/+1, sum d_k 2^k = V
    A = np.abs(V); sg = np.sign(V).astype(np.int8); out = np.zeros((PL,) + A.shape, np.int8)
    if WRITE == "plain":
        assert (A >> PL == 0).all()
        for k in range(PL): out[k] = ((A >> k) & 1).astype(np.int8) * sg
    else:                                              # signed digits, place by place from the bottom: an odd remainder
        n = A.copy()                                   # takes +1 if it is 1 mod 4, -1 if 3 mod 4 (then the next place is 0)
        for k in range(PL):
            d = np.where((n & 1) == 1, 2 - (n & 3), 0); out[k] = d.astype(np.int8) * sg; n = (n - d) >> 1
        assert (n == 0).all()
        lit = out != 0; assert not (lit[1:] & lit[:-1]).any()
    return out

# ---------- the word table in this writing ----------
DW = np.zeros((PL, NW, LINES), np.int8)                # word w's digit at place b on line i
PPH = np.zeros(LINES, np.int64); LITCOL = np.zeros(LINES, np.int64)
for c0 in range(0, NW, 1024):
    V = TW[:, c0:c0 + 1024].T.astype(np.int64); d = digits(V)
    assert ((d.astype(np.int64) << np.arange(PL)[:, None, None]).sum(0) == V).all()
    DW[:, c0:c0 + 1024] = d; PPH += pc(V).sum(0); LITCOL += (d != 0).sum((0, 1))
    del V, d
BONES = pc(BIAS)

# ---------- the groups ----------
GP = np.random.default_rng(SEED).permutation(LINES)
def groups(G):
    T = LINES // G; g = np.empty(LINES, np.int64)
    for j in range(G): g[GP[j * T:(j + 1) * T]] = j
    return g
OH = {G: (groups(G)[:, None] == np.arange(G)[None, :]).astype(np.float32) for G in GS}
WC = {}                                                # stored counts: (NW, PL, G) ups, downs (plain: lit, None)
for G in GS:
    U = np.zeros((NW, PL, G), np.int16); Dn = np.zeros((NW, PL, G), np.int16)
    for b in range(PL):
        U[:, b] = ((DW[b] == 1).astype(np.float32) @ OH[G]).astype(np.int16)
        Dn[:, b] = ((DW[b] == -1).astype(np.float32) @ OH[G]).astype(np.int16)
    WC[G] = (U, Dn) if WRITE == "signed" else (U + Dn, None)
    assert int(WC[G][0].sum(2, dtype=np.int64).sum()) + (int(WC[G][1].sum(2, dtype=np.int64).sum()) if WC[G][1] is not None else 0) == int(LITCOL.sum())
print(f"mode={MODE} write={WRITE} G={GS} seed={SEED}: word table ready ({time.time() - t0:.0f}s, {rss():.2f} GB)", flush=True)

# ---------- the stages ----------
DIAG_T = list(range(27, 0, -2)); LIGHT_C = list(range(13, 2, -2))
a_, b_ = np.arange(PL)[:, None], np.arange(PL)[None, :]
UNR = {"diag": [(a_ + b_ < t) for t in DIAG_T], "light": [np.broadcast_to(a_ < c, (PL, PL)).copy() for c in LIGHT_C]}
def news(U):
    out = [~U[0]] + [U[k - 1] & ~U[k] for k in range(1, len(U))] + [U[-1].copy()]
    assert (sum(o.astype(int) for o in out) == 1).all() and all(o.any() for o in out)
    return out
NEWS = {s: news(UNR[s]) for s in UNR}
SCH = ("diag", "light"); KN = {s: len(UNR[s]) for s in SCH}
UB = [u for s in SCH for u in UNR[s]]; NB = [n for s in SCH for n in NEWS[s]]
KOFF = {"diag": 0, "light": KN["diag"]}; ROFF = {"diag": 0, "light": KN["diag"] + 1}
NK, NR = len(UB), len(NB)
P2 = (1 << (a_ + b_)).astype(np.int64)
WBI = np.stack([u * P2 for u in UB]).astype(np.int64)  # (NK, a, b) 2^(a+b) where unread
WBD = WBI.astype(np.float64); WNR = np.stack(NB).astype(np.float64)
TOUCH = np.stack([u.any(0) for u in UB])               # (NK, b): a place b with some unread a
UBI = np.stack(UB).astype(np.int64)
CUMPC = np.concatenate([[0], np.cumsum(pc(np.arange(1, LINES + 1)))])

ABUF = np.zeros((NW, 2 * LINES), np.float32)          # the word marks, written in place
def thr_word(C, T, col):                               # (NW, G) counts -> ABUF[:, col:col+G*T] = marks [C_g >= t], t = 1..T
    ABUF[:, col:col + LINES] = (C[:, :, None] > np.arange(T)[None, None, :]).reshape(len(C), -1)
def thr_light(C, T):                                   # (P, PL, G) counts -> (G*T, P*PL) marks, columns (p, a)
    P, _, G = C.shape
    return (C[:, :, :, None] > np.arange(T)).transpose(2, 3, 0, 1).reshape(G * T, P * PL).astype(np.float32)

def run_batch(X, F, G, collapse, chk):
    """X (P, 768) light, F (P,) gains. Everything for one G. Returns the checks and (if collapse) the read records."""
    P = len(X); T = LINES // G; U16, D16 = WC[G]
    XD = digits(X)
    assert ((XD.astype(np.int64) << np.arange(PL)[:, None, None]).sum(0) == X).all()
    XD = XD.transpose(1, 0, 2)                         # (P, PL, 768)
    XU, XN = XD == 1, XD == -1
    LU = (XU.astype(np.float32) @ OH[G]).astype(np.int64); LD = (XN.astype(np.float32) @ OH[G]).astype(np.int64)
    sheets = [LU, LD] if WRITE == "signed" else [LU + LD]
    # tables (for their photons only) and tables built, per position
    OCS = np.zeros((P, NK, PL, G, T + 1), np.int64); BUILT = np.zeros((P, NK), np.int64)
    for p in range(P):
        for Lc in sheets:
            Lp = Lc[p]                                 # (PL a, G)
            tab = np.einsum("kab,agm->kbgm", WBI, np.minimum(Lp[:, :, None], np.arange(T + 1)[None, None, :]))
            OCS[p] += pc(tab)
            J = Lp.max(0); q = (CUMPC[Lp] + (J[None, :] - Lp) * pc(Lp)).sum(1)        # (PL a,)
            BUILT[p] += (UBI * q[None, :, None]).sum((1, 2))
    DEL = (OCS[..., 1:] - OCS[..., :-1]).astype(np.float32)      # (P, NK, PL b, G, T)
    xu = XU.reshape(P * PL, LINES).T.astype(np.float32); xn = XN.reshape(P * PL, LINES).T.astype(np.float32)
    Rpair = np.block([[xu, xn], [xn, xu]]); del xu, xn                                   # (1536, 2 P PL): up | down
    if WRITE == "signed":
        LTU, LTD = thr_light(LU, T), thr_light(LD, T); Rb = np.block([[LTU, LTD], [LTD, LTU]]); del LTU, LTD   # high | low
    else:
        Rb = thr_light(sheets[0], T)
    nb = Rb.shape[1]
    rest = np.zeros((NW, P, NK), np.int64); pile = np.zeros((NW, P), np.int64); meet = np.zeros((NW, P, NR), np.int64)
    Rh = np.zeros((NW, P, NK), np.int64); Rl = np.zeros((NW, P, NK), np.int64); tv = np.zeros((NW, P, NK), np.int32)
    viol = cells = 0; samp = {}
    for b in range(PL):
        ABUF[:, :LINES] = DW[b] == 1; ABUF[:, LINES:] = DW[b] == -1
        O = ABUF @ Rpair
        Up = O[:, :P * PL].reshape(NW, P, PL); Dn = O[:, P * PL:].reshape(NW, P, PL); del O
        Df = (Up - Dn).astype(np.float64).reshape(NW * P, PL); Sf = (Up + Dn).astype(np.float64).reshape(NW * P, PL)
        rest += (Df @ WBD[:, :, b].T).reshape(NW, P, NK).astype(np.int64)
        pile += (Df @ P2[:, b].astype(np.float64)).reshape(NW, P).astype(np.int64)
        meet += (Sf @ WNR[:, :, b].T).reshape(NW, P, NR).astype(np.int64); del Df, Sf
        Rd = DEL[:, :, b].transpose(2, 3, 1, 0).reshape(G * T, NK * P)                   # columns (k, p)
        if WRITE == "signed":
            thr_word(U16[:, b], T, 0); thr_word(D16[:, b], T, LINES); Rd = np.concatenate([Rd, Rd], 0); Ab = ABUF
        else:
            thr_word(U16[:, b], T, 0); Ab = ABUF[:, :LINES]
        Ob = Ab @ np.concatenate([Rb, Rd], 1); del Ab, Rd
        Bh = Ob[:, :P * PL].reshape(NW, P, PL)
        Bl = Ob[:, P * PL:2 * P * PL].reshape(NW, P, PL) if WRITE == "signed" else Bh
        tv += Ob[:, nb:].reshape(NW, NK, P).transpose(0, 2, 1).astype(np.int32)
        if WRITE == "signed": viol += int(((Up > Bh) | (Dn > Bl)).sum())
        else: viol += int(((Up + Dn) > Bh).sum())
        cells += NW * P * PL
        Rh += (Bh.astype(np.float64).reshape(NW * P, PL) @ WBD[:, :, b].T).reshape(NW, P, NK).astype(np.int64)
        Rl += (Bl.astype(np.float64).reshape(NW * P, PL) @ WBD[:, :, b].T).reshape(NW, P, NK).astype(np.int64)
        if chk:
            for p in range(min(P, 2)):
                for w in chk:
                    samp[(p, w, b)] = (Up[w, p].astype(np.int64), Dn[w, p].astype(np.int64), Bh[w, p].astype(np.int64), Bl[w, p].astype(np.int64))
        del Ob, Up, Dn, Bh, Bl
    # the machine's product, every word: the pairs must add up to it
    PILE = np.zeros((P, NW), np.int64)
    for w0 in range(0, NW, 8192):
        pr = X.astype(np.float64) @ TW[:, w0:w0 + 8192].astype(np.float64); assert np.abs(pr).max() < 2 ** 52
        PILE[:, w0:w0 + 8192] = pr.astype(np.int64)
    res = dict(pile_bad=int((pile.T != PILE).sum()), viol=viol, cells=cells, stage_bad=0, score_bad=0, stage_cells=0,
               slow_bad=0, slow_checked=0)
    # the slow recount, on sampled words: direct pair counts, direct group mins, direct table photons, counts read
    for (p, w, b), (up, dn, bh, bl) in samp.items():
        for a in range(PL):
            xa, wb = XD[p, a].astype(np.int64), DW[b, w].astype(np.int64)
            u_ = int(((xa * wb) == 1).sum()); d_ = int(((xa * wb) == -1).sum())
            if WRITE == "signed":
                h_ = sum(min(int(LU[p, a, g]), int(U16[w, b, g])) + min(int(LD[p, a, g]), int(D16[w, b, g])) for g in range(G))
                l_ = sum(min(int(LU[p, a, g]), int(D16[w, b, g])) + min(int(LD[p, a, g]), int(U16[w, b, g])) for g in range(G))
            else:
                h_ = l_ = sum(min(int(sheets[0][p, a, g]), int(U16[w, b, g])) for g in range(G))
            res["slow_bad"] += int(u_ != up[a]) + int(d_ != dn[a]) + int(h_ != bh[a]) + int(l_ != bl[a]); res["slow_checked"] += 1
    if chk:
        for p in range(min(P, 2)):
            for w in chk:
                for k in range(NK):                    # table values read by word w at stage k, from scratch
                    tot = 0
                    for b in range(PL):
                        if not TOUCH[k, b]: continue
                        for g in range(G):
                            cnts = [int(U16[w, b, g])] + ([int(D16[w, b, g])] if WRITE == "signed" else [])
                            for Lc in sheets:
                                for m in cnts:
                                    tot += int(pc(sum(min(int(Lc[p, a, g]), m) << (a + b) for a in range(PL) if UB[k][a, b])))
                    res["slow_bad"] += int(tot != tv[w, p, k]); res["slow_checked"] += 1
    # every word, every stage: -low <= exact unread part <= high; and at the score
    S = np.stack([power(PILE[p], F[p]) + BIAS for p in range(P)])
    res["stage_bad"] = int(((rest > Rh) | (rest < -Rl)).sum()); res["stage_cells"] = rest.size
    for p in range(P):
        c = PILE[p][:, None] - rest[:, p]                                                    # counted pile, every stage
        lo = power(c - Rl[:, p], F[p]) + BIAS[:, None]; hi = power(c + Rh[:, p], F[p]) + BIAS[:, None]
        res["score_bad"] += int(((lo > S[p][:, None]) | (S[p][:, None] > hi)).sum())
    res["meet_all"] = meet.sum((0, 2))                                                       # every pair met once, per schedule
    res["meet_all_light"] = meet[:, :, ROFF["light"]:].sum((0, 2)); res["meet_all_diag"] = meet[:, :, :ROFF["light"]].sum((0, 2))
    res["lightph"] = (XD != 0).sum((1, 2)); res["S"] = S; res["PILE"] = PILE
    res["rest_frac"] = (np.abs(rest).sum((0, 1)), Rh.sum((0, 1)), Rl.sum((0, 1)))
    res["maxcount"] = (int(LU.max()), int(LD.max()))
    if not collapse: return res
    CR = sum(pc(C).sum(2) for C in WC[G] if C is not None)                                # (NW, PL b): 1s of the counts read
    CRS = CR @ TOUCH.T.astype(np.int64)                                                     # (NW, NK)
    out = {}
    for s in SCH:
        K = KN[s]; ko, ro = KOFF[s], ROFF[s]
        r = dict(surv=np.zeros((P, K), np.int64), enter=np.zeros((P, K), np.int64), meet=np.zeros((P, K + 1), np.int64),
                 parts=np.zeros((P, K, 7), np.int64), bad=np.zeros(P, np.int64), top=np.zeros((P, 5), np.int64),
                 win=np.zeros(P, np.int64), final=np.zeros(P, np.int64), findiv=np.zeros(P, np.int64), finst=np.zeros(P, np.int64))
        for p in range(P):
            ge = int(pc(EXP2[int(F[p]) & 0xFFFF])); alive = np.arange(NW)
            for k in range(K):
                kk = ko + k; n = len(alive)
                r["enter"][p, k] = n; r["meet"][p, k] = int(meet[alive, p, ro + k].sum())
                c = PILE[p, alive] - rest[alive, p, kk]; h, l = Rh[alive, p, kk], Rl[alive, p, kk]
                lo = power(c - l, F[p]) + BIAS[alive]; hi = power(c + h, F[p]) + BIAS[alive]
                r["bad"][p] += int(((lo > S[p, alive]) | (S[p, alive] > hi)).sum())
                r["parts"][p, k] = (int(CRS[alive, kk].sum()), int(tv[alive, p, kk].sum(dtype=np.int64)), int(BUILT[p, kk]),
                                    int(pc(h).sum() + pc(l).sum()), int((pc(c - l) + pc(c + h)).sum()) * ge,
                                    2 * int(BONES[alive].sum()), n)
                L5 = np.partition(lo, n - 5)[n - 5]
                alive = alive[hi >= L5]; r["surv"][p, k] = len(alive)
            r["meet"][p, K] = int(meet[alive, p, ro + K].sum())
            Sp = np.full(NW, FLOOR, np.int64); Sp[alive] = S[p, alive]
            r["top"][p] = top5(Sp[None])[0]; r["win"][p] = race(Sp)
            r["final"][p] = len(alive); r["findiv"][p] = int(pc(PILE[p, alive]).sum()) * ge; r["finst"][p] = int(BONES[alive].sum())
        out[s] = r
    res["collapse"] = out
    return res

# ---------- positions ----------
if MODE == "bound": sel = np.arange(0, len(SRC), 65)[:100]
else: sel = np.arange(480 + 20 * 100)                  # the 30 refs texts (480 positions) + judge5 windows 0..99
assert len(sel) == (100 if MODE == "bound" else 2480)
if MODE == "top": assert (SRC[:480] == 0).all() and (SRC[480:2480] == 1).all() and (WINA[480:2480] == np.repeat(np.arange(100), 20)).all()
sel = np.array_split(sel, PART[1])[PART[0] - 1]
if int(ARG.get("limit", 0)): sel = sel[:int(ARG["limit"])]
N = len(sel)
LIGHTS = ["real", "shuffled"] if MODE == "bound" else ["real"]
rng_sh = np.random.default_rng(31); SH = np.stack([rng_sh.permutation(LINES) for _ in range(N)])    # the checker's own shuffle
rng_w = np.random.default_rng(17)
TOT = {}; REC = {}
F_ALL = ((WE[0] + EXA + ES) * 65536 - LSA).astype(np.int64)
for light in LIGHTS:
    for G in GS:
        key = (light, G); TOT[key] = dict(pile_bad=0, viol=0, cells=0, stage_bad=0, score_bad=0, stage_cells=0, slow_bad=0,
                                          slow_checked=0, meet_sum_bad=0, collapse_bad=0, top_vs_plain=0, abs_rest=0, sum_rh=0, sum_rl=0)
        recs = {s: [] for s in SCH}; plain = []
        mxc = [0, 0]
        for c0 in range(0, N, BATCH):
            q = sel[c0:c0 + BATCH]; X = X_ALL[q].copy(); F = F_ALL[q]
            if light == "shuffled": X = np.stack([X[i][SH[c0 + i]] for i in range(len(q))])
            chk = list(rng_w.choice(NW, 3, replace=False)) if c0 == 0 else None
            res = run_batch(X, F, G, collapse=True, chk=chk)
            t = TOT[key]
            for k in ("pile_bad", "viol", "cells", "stage_bad", "score_bad", "stage_cells", "slow_bad", "slow_checked"): t[k] += res[k]
            XDl = res["lightph"]
            t["abs_rest"] += int(res["rest_frac"][0].sum()); t["sum_rh"] += int(res["rest_frac"][1].sum()); t["sum_rl"] += int(res["rest_frac"][2].sum())
            mxc = [max(mxc[0], res["maxcount"][0]), max(mxc[1], res["maxcount"][1])]
            S = res["S"]
            for i in range(len(q)):
                tp = top5(S[i:i + 1])[0]; wp = race(S[i])
                plain.append(dict(top=tp, win=wp, plain916=int(pc(X[i]) @ PPH), plainw=int((digits(X[i]) != 0).sum(0) @ LITCOL), pldiv=int(pc(res["PILE"][i]).sum()) * int(pc(EXP2[int(F[i]) & 0xFFFF])),
                                  plst=int(BONES.sum()), lightph=int(XDl[i]), meet_all_d=int(res["meet_all_diag"][i]), meet_all_l=int(res["meet_all_light"][i])))
                for s in SCH:
                    r = res["collapse"][s]
                    recs[s].append({k: v[i] for k, v in r.items()})
                    t["collapse_bad"] += int(r["bad"][i]); t["top_vs_plain"] += int((r["top"][i] != tp).any())
            print(f"  {light} G={G}: {min(c0 + BATCH, N)}/{N} positions ({time.time() - t0:.0f}s, {rss():.2f} GB)", flush=True)
        REC[key] = (recs, plain, mxc)

out = dict(sel=sel, gs=np.array(GS))
print(f"\nspot_check {MODE} {WRITE}: {N} positions (part {PART[0]}/{PART[1]}), groups seed {SEED}")
for (light, G), t in TOT.items():
    recs, plain, mxc = REC[(light, G)]
    md = np.array([p["meet_all_d"] for p in plain]); ml = np.array([p["meet_all_l"] for p in plain]); pw = np.array([p["plainw"] for p in plain])
    t["meet_sum_bad"] = int((md != pw).sum() + (ml != pw).sum())
    print(f"  {light:8s} G={G:2d}: pairs added = the machine's product: bad {t['pile_bad']};"
          f" every (word, a, b): pairs above the group bound {t['viol']} of {t['cells']:,};"
          f" every (word, stage): unread part outside [-low, high] {t['stage_bad']} of {t['stage_cells']:,}, score outside [lo, hi] {t['score_bad']};"
          f" slow recount bad {t['slow_bad']} of {t['slow_checked']}; meetings add up to the plain meetings: bad {t['meet_sum_bad']};"
          f" collapse: bound broken {t['collapse_bad']}, top 5 differs from the plain read at {t['top_vs_plain']} of {N};"
          f" |unread| / high bound {t['abs_rest'] / max(t['sum_rh'], 1):.3f}; largest light count {mxc}", flush=True)
    pfx = f"{light}_{G}"
    for k in ("top", "win", "plain916", "plainw", "pldiv", "plst", "lightph"): out[f"{pfx}_{k}_pl"] = np.array([p[k] for p in plain])
    for s in SCH:
        for k in recs[s][0]: out[f"{pfx}_{k}_{s}"] = np.array([r[k] for r in recs[s]])
    out[f"{pfx}_tot"] = np.array([t[k] for k in sorted(t)]); out[f"{pfx}_tot_keys"] = np.array(sorted(t))
for G in GS:
    U16, D16 = WC[G]
    out[f"store_{G}"] = np.array([int(max(U16.max(), D16.max() if D16 is not None else 0)), (LINES // G).bit_length()])
tag = f"{MODE}_{WRITE}" + (f"_p{PART[0]}of{PART[1]}" if PART[1] > 1 else "")
np.savez(f"{D}/spot_check_{tag}.npz", **out)
print(f"  saved collapse_top/spot_check_{tag}.npz ({time.time() - t0:.0f}s, max memory {rss():.2f} GB)")
