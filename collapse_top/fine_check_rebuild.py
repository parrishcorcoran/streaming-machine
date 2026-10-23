"""CHECKER'S OWN REBUILD of the fine-grid staged top read, diagonal schedule (fine_staged_top.py's claims), written apart
from fine_staged_top.py and staged_top.py: none of their code is imported or copied.

The picture, restated. At a read position the light x (768 whole numbers, collapse_top/light_top.npz) meets each word's
stamp T_w (768 whole numbers, head.w of base2_gpt2.npz). Every value is its sign and its photons (places 0..14). A light
photon at place a and a stamp photon at place b on the same line make a pair worth 2^(a+b), up or down, on diagonal
d = a + b (0..28). The pile is the sum of all the pairs; the score is power(pile, F) + standing amount, the machine's
own power (a table read of the fine part of F, then a shift with the half added: it never runs backwards).

How this rebuild gets there (a different route from fine_staged_top.py):
  - the piles per diagonal: for every pair of places (a, b), the signed count of lit pairs and the count of lit pairs
    (photon meetings) on each word, as two products of the light's bit plane a with the stamp's bit plane b (values
    -1, 0, 1 and 0, 1, sums below 769: whole and exact in float32). The 225 pieces are put on their diagonals.
    CHECK: the 29 diagonals together give the plain pile x . T_w of every word, exactly; the meetings of every word on
    every diagonal add up to (the light's photons) x (the stamps' photons) line by line.
  - the bound, brute force, no table: |rest_w| <= sum over unread (a, b) of 2^(a+b) min(n_a, m_wb), computed from the
    full (word, a, b) array of min(n_a, m_wb) for every word at every stage (a product with the stage's 2^(a+b) weights).
    signed: up <= sum 2^(a+b) [min(n+_a, m+_wb) + min(n-_a, m-_wb)], down <= sum 2^(a+b) [min(n+_a, m-_wb) + min(n-_a, m+_wb)].
    CHECK, on the piles of EVERY word at EVERY stage (not only those still in): -bound <= rest <= bound (plain) and
    -down <= rest <= up (signed); each signed side no wider than the plain bound.
  - stages fixed in advance: the top 2, 4, ..., 28 diagonals (d >= 29 - k), then all. A word is set aside when its hi
    is below the 5th-best lo among the words still in. After the last stage the words still in take their exact score.
  - meetings per stage: the lit pairs on the stage's new diagonals, summed over the words entering that stage; the
    last stage ("all") the words still in after the top 28, diagonal 0.
  - the bound's work in photons, combined/machine.py's rule as fine_staged_top.py states it (tables, adds, divides,
    standing amounts), recomputed here from per-word table values made by a product (not by a table lookup).
  - top 5: judge5.top5's rule on the staged scores (FLOOR for the set-aside words) against the same on the plain read of
    every word; also the tie-aware comparison (the five scores the same). On windows against judge5_ref.npz (set and
    order), and judge5.judge's own count (the staged argmax in the reference top 5, the same top word). On refs the
    race winner against the one saved with the light.

    python3 collapse_top/fine_check_rebuild.py [light=real|shuffled] [seed=2026] [set=sub|refs] [signed=1] [part=i/n] [batch=24]
      set=sub: the 480 refs positions (all 30 texts) and the judge5 windows 0, 3, 6, ..., 297 (100 windows, 2,000 positions)
      -> collapse_top/fine_check_<light>[_seed]_p<i>of<n>.npz
"""
import sys, os, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; sums checked below
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
LIGHT, SEED, SET = ARG.get("light", "real"), int(ARG.get("seed", 2026)), ARG.get("set", "sub")
SIGNED, BATCH = int(ARG.get("signed", 1)), int(ARG.get("batch", 24))
PI, PN = (int(v) for v in ARG.get("part", "1/1").split("/"))
t0 = time.time()
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
PL, NW, LN, ND = 15, 50257, 768, 29
FLOOR = -(1 << 62)

Z = np.load("base2_gpt2.npz")
HW16 = Z["head.w"]; HE = Z["head.e"]; HB = Z["head.b"].astype(np.int64); EXP2 = Z["exp2"].astype(np.int64)
assert (HE == HE[0]).all()
HWF = HW16.astype(np.float64)                                     # (768, NW) for the plain piles (whole numbers below 2^53)
T = np.ascontiguousarray(HW16.T).astype(np.int32)                 # (NW, 768) each word's stamp
TAB = np.abs(T).astype(np.int16); TSG = np.sign(T).astype(np.int8); del T
NTW = np.stack([((TAB >> b) & 1).sum(1) for b in range(PL)], 1).astype(np.int64)                   # m_wb
MPW = np.stack([(((TAB >> b) & 1) * (TSG > 0)).sum(1) for b in range(PL)], 1).astype(np.int64)    # m+_wb
MMW = np.stack([(((TAB >> b) & 1) * (TSG < 0)).sum(1) for b in range(PL)], 1).astype(np.int64)    # m-_wb
assert (MPW + MMW == NTW).all()
STAMP_PH_LINE = np.bitwise_count(TAB).astype(np.int64).sum(0)   # stamp photons per line, all words
NTWF, MPWF, MMWF = NTW.astype(np.float64), MPW.astype(np.float64), MMW.astype(np.float64)
ONES_HB = np.array([bin(int(v)).count("1") for v in np.abs(HB)], np.int64)

L = np.load("collapse_top/light_top.npz")
SRC, WIN, REFW = L["src"], L["win"], L["refs_winner"]
XALL = L["x16"].astype(np.int64); EXA, LSA = L["ex"].astype(np.int64), L["LS"].astype(np.int64)
if LIGHT == "shuffled":                                           # each position's light across its lines
    rng = np.random.default_rng(SEED)
    XALL = np.stack([x[rng.permutation(LN)] for x in XALL])
REF5 = np.load("judge5_ref.npz")["top5"]                          # (300, 20, 5)
NREF = int((SRC == 0).sum())

if SET == "sub": selall = np.flatnonzero((SRC == 0) | ((SRC == 1) & (WIN % 3 == 0)))
elif SET == "refs": selall = np.flatnonzero(SRC == 0)
else: raise SystemExit("set?")
cut = [len(selall) * i // PN for i in range(PN + 1)]
sel = selall[cut[PI - 1]:cut[PI]]; N = len(sel)

def power(X, F):                                                  # X x 2^(F/65536): table read of the fine part, then shift
    F = int(F); I, f = F >> 16, F & 0xFFFF; Y = np.asarray(X, np.int64) * int(EXP2[f]); s = 16 - I
    return (Y + (1 << (s - 1))) >> s if s > 0 else Y << (-s)
def top5(S):                                                      # judge5.top5's rule
    idx = np.argpartition(-S, 5)[:5]; return idx[np.argsort(-S[idx])]
def race(v): return int(np.flatnonzero(v == v.max())[0])          # who arrives first: the top value, first in line
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)

KS = list(range(2, 29, 2)); NS = len(KS); TH = [29 - k for k in KS]                       # stage s counts d >= TH[s]
a_, b_ = np.arange(PL)[:, None], np.arange(PL)[None, :]
WT = np.zeros((PL * PL, NS))                                      # weights 2^(a+b) of the unread pairs, per stage
WG = np.zeros((PL * PL, NS * PL))                                 # per stage and b: 2^a over the unread a (the table values)
for s, th in enumerate(TH):
    unread = (a_ + b_ < th)
    WT[:, s] = (unread * (2.0 ** (a_ + b_))).ravel()
    for b in range(PL): WG[:, s * PL + b] = ((unread & (b_ == b)) * (2.0 ** a_) * np.ones((1, PL))).ravel()
TABLE_ROWS = [min(PL, th) for th in TH]                           # light places with any unread pair (b = 0 always there)

SURV = np.zeros((N, NS), np.int64); ENTER = np.zeros((N, NS), np.int64); WORK = np.zeros((N, NS + 1), np.int64)
BW = np.zeros((N, NS, 4), np.int64); PLAIN = np.zeros(N, np.int64)
SURV_S = np.zeros((N, NS), np.int64); ENTER_S = np.zeros((N, NS), np.int64); WORK_S = np.zeros((N, NS + 1), np.int64)
TOP_ST = np.zeros((N, 5), np.int64); TOP_PL = np.zeros((N, 5), np.int64); TOP_SS = np.zeros((N, 5), np.int64)
SC_ST = np.zeros((N, 5), np.int64); SC_PL = np.zeros((N, 5), np.int64); SC_SS = np.zeros((N, 5), np.int64)
WIN_ST = np.zeros(N, np.int64); WIN_PL = np.zeros(N, np.int64); WIN_SS = np.zeros(N, np.int64)
SC56 = np.zeros((N, 2), np.int64)                                 # the plain read's 5th and 6th scores (ties)
BAD = dict(pile_vs_diagonals=0, meetings_vs_photons=0, signed_over_meetings=0, bound_pile=0, bound_signed_pile=0,
           signed_wider=0, bound_score=0, bound_score_signed=0, R_two_ways=0, UDR_by_hand=0, UDR_checked=0)
rng_chk = np.random.default_rng(11)

for c0 in range(0, N, BATCH):
    g = sel[c0:c0 + BATCH]; P = len(g)
    X = XALL[g]; ax, sx = np.abs(X), np.sign(X)
    Fv = (int(HE[0]) + EXA[g] + 16) * 65536 - LSA[g]
    PILE = (X.astype(np.float64) @ HWF); assert np.abs(PILE).max() < 2 ** 52; PILE = PILE.astype(np.int64)
    LSIG = np.stack([sx * ((ax >> a) & 1) for a in range(PL)]).reshape(PL * P, LN).astype(np.float32)
    LUNS = np.stack([((ax >> a) & 1) for a in range(PL)]).reshape(PL * P, LN).astype(np.float32)
    SD = np.zeros((ND, P, NW), np.int64); KD = np.zeros((ND, P, NW), np.int32)
    for b in range(PL):
        bitb = (TAB >> b) & 1
        CS = (LSIG @ (TSG * bitb).astype(np.float32).T).reshape(PL, P, NW)
        CU = (LUNS @ bitb.astype(np.float32).T).reshape(PL, P, NW)
        for a in range(PL):
            SD[a + b] += CS[a].astype(np.int64); KD[a + b] += CU[a].astype(np.int32)
        del CS, CU, bitb
    for p in range(P):
        q = c0 + p; F = Fv[p]
        S = power(PILE[p], F) + HB                                  # the plain read, every word
        sdp = SD[:, p, :] << np.arange(ND)[:, None]
        cum = np.cumsum(sdp[::-1], axis=0)[::-1]                    # cum[d]: the pairs on diagonals d and above
        BAD["pile_vs_diagonals"] += int((cum[0] != PILE[p]).sum())
        kd = KD[:, p, :].astype(np.int64)
        BAD["signed_over_meetings"] += int((np.abs(SD[:, p, :]) > kd).sum())
        kcum = np.cumsum(kd[::-1], axis=0)[::-1]
        plain = int((np.bitwise_count(ax[p]).astype(np.int64) * STAMP_PH_LINE).sum())
        BAD["meetings_vs_photons"] += int(kcum[0].sum() != plain); PLAIN[q] = plain
        KNEW = [kcum[TH[0]]] + [kcum[TH[s]] - kcum[TH[s - 1]] for s in range(1, NS)] + [kcum[0] - kcum[TH[-1]]]
        CNT = cum[TH]                                               # (NS, NW) counted piles
        REST = PILE[p][None, :] - CNT
        nx = np.array([int(((ax[p] >> a) & 1).sum()) for a in range(PL)], np.float64)
        MIN = np.minimum(nx[None, :, None], NTWF[:, None, :]).reshape(NW, PL * PL)
        R = (MIN @ WT).astype(np.int64).T                           # (NS, NW) the plain bound, every word, every stage
        GV = (MIN @ WG).astype(np.int64).reshape(NW, NS, PL)        # the table values each word reads
        R2 = (GV << np.arange(PL)[None, None, :]).sum(2).T
        BAD["R_two_ways"] += int((R2 != R).sum()); del MIN
        BAD["bound_pile"] += int((np.abs(REST) > R).sum())
        if SIGNED:
            nxp = np.array([int(((ax[p] * (sx[p] > 0) >> a) & 1).sum()) for a in range(PL)], np.float64); nxm = nx - nxp
            mp, mm = MPWF[:, None, :], MMWF[:, None, :]
            MU = (np.minimum(nxp[None, :, None], mp) + np.minimum(nxm[None, :, None], mm)).reshape(NW, PL * PL)
            U = (MU @ WT).astype(np.int64).T; del MU
            MD = (np.minimum(nxp[None, :, None], mm) + np.minimum(nxm[None, :, None], mp)).reshape(NW, PL * PL)
            D = (MD @ WT).astype(np.int64).T; del MD
            ws = rng_chk.choice(NW, 64, replace=False) if p == 0 else []   # first position of each batch: the bounds again, by hand, 64 words
            for w in ws:
                for s in (0, 6, 7, 13):
                    uu = sum((1 << (a + b)) * (min(int(nxp[a]), int(MPW[w, b])) + min(int(nxm[a]), int(MMW[w, b])))
                             for a in range(PL) for b in range(PL) if a + b < TH[s])
                    dd = sum((1 << (a + b)) * (min(int(nxp[a]), int(MMW[w, b])) + min(int(nxm[a]), int(MPW[w, b])))
                             for a in range(PL) for b in range(PL) if a + b < TH[s])
                    rr = sum((1 << (a + b)) * min(int(nx[a]), int(NTW[w, b])) for a in range(PL) for b in range(PL) if a + b < TH[s])
                    BAD["UDR_by_hand"] += int(uu != U[s, w]) + int(dd != D[s, w]) + int(rr != R[s, w]); BAD["UDR_checked"] += 3
            BAD["bound_signed_pile"] += int(((REST > U) | (REST < -D)).sum())
            BAD["signed_wider"] += int(((U > R) | (D > R)).sum())
        ge = bin(int(EXP2[(-int(LSA[g[p]])) & 0xFFFF])).count("1")
        QA = np.array([sum(bin(min(int(n), m)).count("1") for m in range(LN + 1)) for n in nx], np.int64)
        alive = np.arange(NW)
        for s in range(NS):                                         # plain bound
            c, Rw = CNT[s, alive], R[s, alive]
            lo = power(c - Rw, F) + HB[alive]; hi = power(c + Rw, F) + HB[alive]; ex = S[alive]
            BAD["bound_score"] += int(((lo > ex) | (ex > hi)).sum())
            BW[q, s, 0] = int(((a_ + b_ < TH[s]) * QA[:, None]).sum())
            BW[q, s, 1] = int(ones(GV[alive, s, :]).sum()) + 2 * int(ones(Rw).sum())
            BW[q, s, 2] = (int(ones(c - Rw).sum()) + int(ones(c + Rw).sum())) * ge
            BW[q, s, 3] = 2 * int(ONES_HB[alive].sum())
            WORK[q, s] = int(KNEW[s][alive].sum())
            l5 = np.sort(lo)[-5]
            ENTER[q, s] = len(alive); alive = alive[hi >= l5]; SURV[q, s] = len(alive)
        WORK[q, NS] = int(KNEW[NS][alive].sum())
        Sp = np.full(NW, FLOOR, np.int64); Sp[alive] = S[alive]
        t_st, t_pl = top5(Sp), top5(S)
        TOP_ST[q], TOP_PL[q], SC_ST[q], SC_PL[q] = t_st, t_pl, Sp[t_st], S[t_pl]
        WIN_ST[q], WIN_PL[q] = race(Sp), race(S)
        srt = np.sort(S)[::-1]; SC56[q] = srt[4], srt[5]
        if SIGNED:
            alive = np.arange(NW)
            for s in range(NS):                                     # signed bound
                c = CNT[s, alive]
                lo = power(c - D[s, alive], F) + HB[alive]; hi = power(c + U[s, alive], F) + HB[alive]; ex = S[alive]
                BAD["bound_score_signed"] += int(((lo > ex) | (ex > hi)).sum())
                WORK_S[q, s] = int(KNEW[s][alive].sum())
                l5 = np.sort(lo)[-5]
                ENTER_S[q, s] = len(alive); alive = alive[hi >= l5]; SURV_S[q, s] = len(alive)
            WORK_S[q, NS] = int(KNEW[NS][alive].sum())
            Sp = np.full(NW, FLOOR, np.int64); Sp[alive] = S[alive]
            t_ss = top5(Sp); TOP_SS[q], SC_SS[q], WIN_SS[q] = t_ss, Sp[t_ss], race(Sp)
        del sdp, cum, kd, kcum, CNT, REST, R, R2, GV
    del SD, KD, PILE, LSIG, LUNS
    print(f"  {c0 + P}/{N} positions ({time.time() - t0:.0f}s, {gb():.2f} GB)", flush=True)

tag = f"{LIGHT}" + (f"_seed{SEED}" if LIGHT == "shuffled" and SEED != 2026 else "") + ("" if SET == "sub" else f"_{SET}") + f"_p{PI}of{PN}"
np.savez(f"collapse_top/fine_check_{tag}.npz", sel=sel, ks=np.array(KS), surv=SURV, enter=ENTER, work=WORK, bw=BW, plain=PLAIN,
         surv_s=SURV_S, enter_s=ENTER_S, work_s=WORK_S, top_st=TOP_ST, top_pl=TOP_PL, top_ss=TOP_SS, sc_st=SC_ST, sc_pl=SC_PL,
         sc_ss=SC_SS, win_st=WIN_ST, win_pl=WIN_PL, win_ss=WIN_SS, sc56=SC56,
         bad=np.array(list(BAD.values())), bad_keys=np.array(list(BAD.keys())))
print(f"\nchecker's rebuild, diag schedule, light={LIGHT} seed={SEED} set={SET} part {PI}/{PN}: {N} positions")
print("  checks: " + ", ".join(f"{k} {v}" for k, v in BAD.items()))
src = SRC[sel]
for nm, tst, sst, wst in (("plain bound", TOP_ST, SC_ST, WIN_ST), ("signed bound", TOP_SS, SC_SS, WIN_SS)):
    if nm == "signed bound" and not SIGNED: continue
    same = (tst == TOP_PL).all(1); samesc = (sst == SC_PL).all(1)
    line = f"  [{nm}] top 5 (set and order) = plain read {int(same.sum())}/{N}; five scores equal {int(samesc.sum())}/{N}; race {int((wst == WIN_PL).sum())}/{N}"
    if LIGHT == "real":
        r = src == 0; line += f"; refs race = saved winner {int((wst[r] == REFW[sel[r]]).sum())}/{int(r.sum())}"
        wq = np.flatnonzero(src == 1)
        if len(wq):
            ij = [divmod(int(sel[q]) - NREF, 20) for q in wq]
            assert all(WIN[sel[q]] == i for q, (i, j) in zip(wq, ij))
            ref = np.array([REF5[i, j] for i, j in ij])
            line += (f"; windows: top 5 = judge5_ref {int((tst[wq] == ref).all(1).sum())}/{len(wq)}, staged argmax in ref top 5 "
                     f"{int((ref == wst[wq][:, None]).any(1).sum())}/{len(wq)}, = ref top word {int((ref[:, 0] == wst[wq]).sum())}/{len(wq)}")
    print(line)
print(f"  ties at 5th/6th in the plain read: {int((SC56[:, 0] == SC56[:, 1]).sum())} positions")
for s, k in enumerate(KS):
    v = SURV[:, s]
    print(f"    top {k:2d}: surv median {int(np.median(v))} mean {v.mean():.0f} min {v.min()} max {v.max()}; meetings {WORK[:, s].mean() / 1e6:.2f}M")
pl = PLAIN.mean(); tot = WORK.sum(1).mean(); ops = (ENTER.sum(1) * 15).mean() + sum(TABLE_ROWS) * PL * (LN + 1)
print(f"  plain {pl / 1e6:.1f}M; staged meetings {tot / 1e6:.1f}M = {100 * tot / pl:.2f}%; bound reads+adds {ops / 1e6:.2f}M = {100 * ops / pl:.2f}%;"
      f" bound photons {BW.sum((1, 2)).mean() / 1e6:.1f}M = {100 * BW.sum((1, 2)).mean() / pl:.2f}% (tables, adds, divides, standing: "
      + ", ".join(f"{BW[:, :, i].sum(1).mean() / 1e6:.1f}M" for i in range(4)) + ")")
if SIGNED: print(f"  signed: meetings {WORK_S.sum(1).mean() / 1e6:.1f}M = {100 * WORK_S.sum(1).mean() / pl:.2f}%")
print(f"  ({time.time() - t0:.0f}s, max memory {gb():.2f} GB)")
