"""STEP 2 (16 places) AND THE THREE LOOKS, on the 30 refs texts (first 5 tokens of ids+ref). The answer = the machine's
own word at the last position (the race over its whole-number scores).

THE PICTURE (his words): "The fringe pattern is encoded in the embedding numbers but it's encoded in a way that we can't
see the pattern until after it's ALL done. But we CAN if we reverse every step somehow." "It's predetermined, all the
way to the end." "The gates aren't gates": each bend and each water level is a dimming of the light passing through,
set by that light. Hold every dimming the light got; then the machine from the embedding numbers to word w's score is
one fixed relation, score_w = (embedding numbers) met with R_w + c_w, and R_w is word w's row sent back down every step.

PREDICTIONS (written before running):
  his:    it is predetermined, so a word's reversed pattern carries over to other texts, and the answer shows in the
          embedding numbers.
  Claude: the reversed pattern changes with the text the way the finished fringes did, so a pattern built from another
          text's run gives the answer near the control rate; the exact check holds by construction (step 2: it did,
          15/15, exact_check.log).

STEP 2, 16 places (reverse16 in machine.py: stream 32 places, everything meeting a plate or a dimming 16 places, one
power of two per word per position, the machine's own rounding): for each text, the answer, the machine's next 4 and
200 fixed random words (and the next text's answer): how close E.R_w + c_w is to the machine's score, and whether the
top 5 by the reversed patterns equal the machine's top 5.

(a) WHERE THE ANSWER SITS: the answer's score split by position (E_p met with R_w,p) and the constant; the answer's
    margin over the 2nd word and over a random word split the same way; how many (position, line) entries carry half
    of the meeting (against the lines shuffled within each position, same values); at the last position, which word
    rows of the table R_w meets most (top 10), and where the answer itself and the input word fall; the same for a
    random word's R.
(b) PREDETERMINED? B = the next text. R_w built from B's run (B's dimmings), met with A's embedding numbers + B's
    constant: is A's answer in the top 5 of the same 206 words? Control: how often a random word's R from B puts that
    word in the top 5. Also A's own words run through the machine with B's dimmings held (every dimming from B, the
    machine's own arithmetic, all 50,257 words): where A's answer and B's answer fall. Coincidence control: A's answer
    in B's own top 5. How alike R_w from A and from B are: share of (position, line) entries with the same sign, and
    photons of the difference against photons of R_w (16 places, common power of two per position). Same word, same
    text, read at the last position vs one position earlier.
(c) THE FRINGES OF THE FINAL MEETING: A's embedding numbers met with R_answer (16 places) in the up/down form
    (offset 2^(n-1), each place +1 or -1; n = the fewest places holding the position's values). For every place pair:
    d = lines that disagree, |N - 2d| = fringe, N - |N - 2d| = pairs off straight through. Against the coin-toss value
    (exact, binomial) and two shuffles of R's lines within each position. The meeting rebuilt from the counts is
    checked equal to E.R16.
    python3 reverse/looks.py            -> reverse/looks.npz, printed report (reverse/looks.log)
"""
import sys, time, math, bisect; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")                                # numpy's float products on this Mac raise spurious flags; all checked exact
from machine import *
from fractions import Fraction

def retop(rec, t):
    R = rec["R12"]; x16, ex, LS = gauge_log(R[t:t + 1]); pile = product(x16, HEADW)
    F = (HEADE[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]; scores = power(pile, F)[0] + HEADB
    r = dict(rec); r["t"] = t; r["top"] = dict(x16=x16[0], sh=int(ex[0] + ES), LS=LS, Rt=R[t], scores=scores); return r

def parts(R0, N, e):
    """E_p met with R_w,p for every word and position, exact, whole numbers at 2^EFIX: list (words) of lists (P)"""
    nw, P, n = N.shape; out = [[0] * P for _ in range(nw)]
    for gp, gs in [(N & 0xFFFF, 0), (N >> 16, 16)]:
        Sx = np.einsum("wpn,pn->wp", gp.astype(np.float64), R0.astype(np.float64)); assert np.abs(Sx).max() < 2.0 ** 53
        Si = Sx.astype(np.int64)
        for w in range(nw):
            for p in range(P): out[w][p] += int(Si[w, p]) << (int(e[w, p]) + gs - EFIX)
    return out

def rows_met(Nl, el):
    """every word row of the table met with one position's R (32 places), exact int64 (two 16-place halves)"""
    two = np.stack([(Nl & 0xFFFF), (Nl >> 16)], 1).astype(np.float64); out = np.empty(WTE.shape[0], np.int64)
    for r0 in range(0, WTE.shape[0], 8192):
        ab = WTE[r0:r0 + 8192].astype(np.float64) @ two; assert np.abs(ab).max() < 2.0 ** 53
        out[r0:r0 + 8192] = (ab[:, 1].astype(np.int64) << 16) + ab[:, 0].astype(np.int64)
    return out                                                          # x 2^el

def rank_of(V, i): return int((V > V[i]).sum())

def half_count(m):
    """fewest entries whose |m| reaches half of the total |m| (as a share of all entries)"""
    a = np.sort(np.abs(m).ravel())[::-1]; c = np.cumsum(a)
    return float((np.searchsorted(c, c[-1] / 2) + 1) / a.size)

def fvals(N, e): return N.astype(np.float64) * np.exp2(e.astype(np.float64))[..., None]   # for shares printed only

def photons(A): return int(np.bitwise_count(np.abs(A).astype(np.uint64)).sum())

def alike(Nx, ex, Ny, ey, pos):
    """(share of entries with the same sign, photons of the difference / photons of x) over positions pos"""
    same = tot = px = pd = 0
    for p in pos:
        sx, sy = np.sign(Nx[p]), np.sign(Ny[p]); same += int((sx == sy).sum()); tot += sx.size
        T = max(int(ex[p]) + int(bitlen(np.bitwise_or.reduce(np.abs(Nx[p])))), int(ey[p]) + int(bitlen(np.bitwise_or.reduce(np.abs(Ny[p]))))) - 15
        x16 = rshift(Nx[p], T - int(ex[p])); y16 = rshift(Ny[p], T - int(ey[p]))
        px += photons(x16); pd += photons(x16 - y16)
    return same / tot, pd / max(px, 1)

COIN = {}
def coin(N):                                            # E|N - 2d| / N for d ~ Binomial(N, 1/2), exact
    if N not in COIN:
        COIN[N] = Fraction(sum(math.comb(N, d) * abs(N - 2 * d) for d in range(N + 1)), (1 << N) * N)
    return COIN[N]

def bits(V, n):                                         # (n, lines) 0/1 planes of V + 2^(n-1), place 0 at the bottom
    U = V.astype(np.int64) + (1 << (n - 1)); assert U.min() >= 0 and U.max() < (1 << n)
    return ((U[None, :] >> np.arange(n)[:, None]) & 1).astype(np.float64)

def fringe(Ep, Rp):
    """place-pair disagreement counts of one position's meeting, and the check that they rebuild E.R exactly"""
    nE = int(np.abs(Ep).max()).bit_length() + 1; nR = int(np.abs(Rp).max()).bit_length() + 1
    bE, bR = bits(Ep, nE), bits(Rp, nR); N = Ep.size
    d = (bE @ (1 - bR).T + (1 - bE) @ bR.T).astype(np.int64)            # (nE, nR) lines that disagree
    S = sum((N - 2 * int(d[a, b])) << (a + b) for a in range(nE) for b in range(nR))
    ok = (S - 2 * int(Ep.sum()) - 2 * int(Rp.sum()) - N) == 4 * int((Ep.astype(object) * Rp.astype(object)).sum())
    return d, ok

t0 = time.time()
NT = int(sys.argv[1]) if len(sys.argv) > 1 else len(TEXTS); rng0 = np.random.default_rng(20261004)
cache = {}
def run(i):
    if i not in cache:
        ids = TEXTS[i]; rec = record(ids); S = machine_scores(ids)
        assert (rec["top"]["scores"] == S).all() and race(S) == forward(ids)
        res = residues(rec); kap = kappas(rec, res)
        cache[i] = (rec, res, kap)
        for k in list(cache):
            if k not in (i, (i + 1) % NT, 0): del cache[k]
    return cache[i]

OUT = {k: [] for k in ["idgap", "idgap_rel", "top5_same_order", "top5_same_set", "partsA", "constA", "scoreA",
                       "marg2", "marg2c", "margu", "marguc", "half_ans", "half_ans_shuf", "half_u", "half_marg", "half_marg_shuf",
                       "ans_rank_rows", "in_rank_rows", "u_rank_rows", "in_rank_rows_u", "ans_in10", "in_in10", "u_in10", "in_in10_u",
                       "rankAB_set", "inAB_set", "randAB_in5", "wB_rank_set", "frozen_self", "rank_frozen_wA", "rank_frozen_wB",
                       "wA_in_Btop5", "rank_wA_in_B", "alike_AB", "alike_AA_u", "alike_AB_u", "alike_t43", "alike_t43_u",
                       "alike_t43_own", "rank_wA_at3", "fr_real", "fr_u", "fr_shuf", "fr_shuf2", "fr_coin", "fr_ok",
                       "corner_real", "corner_shuf", "top10_ans", "top10_u", "answers", "wBs",
                       "top10_own", "ans_rank_own", "in_rank_own", "u_rank_own", "c_bias", "c_round", "c_top", "c_all", "rank_const_only", "rank_embed_only", "rankAB_top5", "frozen_vs_RB", "rank_frozen_top5"]}
DREAL, DSHUF = {}, {}
for i in range(NT):
    ids = TEXTS[i]; j = (i + 1) % NT
    recA, resA, kapA = run(i); recB, resB, kapB = run(j)
    SA, SB = recA["top"]["scores"], recB["top"]["scores"]; P = recA["P"]
    wA, wB = race(SA), race(SB); topA = [int(x) for x in np.argsort(-SA, kind="stable")[:5]]
    assert topA[0] == wA
    rnd = []
    while len(rnd) < 200:
        x = int(rng0.integers(50257))
        if x not in topA and x != wB and x not in rnd: rnd.append(x)
    SET = topA + ([wB] if wB not in topA else []) + rnd
    iu = SET.index(rnd[0]); iB = SET.index(wB)
    OUT["answers"].append(wA); OUT["wBs"].append(wB)
    # ---- step 2 at 16 places
    (NA, eA), cA = reverse16(recA, SET, kapA, resA)
    mA = meetE(recA["R0"], NA, eA)
    rev = [frac(a + b) for a, b in zip(mA, cA)]
    gaps = [float(r - int(SA[w])) for r, w in zip(rev, SET)]
    OUT["idgap"].append(gaps); OUT["idgap_rel"].append([abs(g) / abs(int(SA[w])) for g, w in zip(gaps, SET)])
    order_rev = sorted(range(len(SET)), key=lambda k: -rev[k])[:5]
    _c = sorted(cA); _m = sorted(mA)
    OUT["rank_const_only"].append(len(cA) - bisect.bisect_right(_c, cA[0]))          # embedding numbers dark: the constant alone
    OUT["rank_embed_only"].append(len(mA) - bisect.bisect_right(_m, mA[0]))          # the constant removed: E.R_w alone
    OUT["top5_same_order"].append([SET[k] for k in order_rev] == topA); OUT["top5_same_set"].append(set(SET[k] for k in order_rev) == set(topA))
    # ---- (a) where the answer sits
    pt = parts(recA["R0"], NA[[0, 1, iu]], eA[[0, 1, iu]])
    OUT["partsA"].append([float(frac(x)) for x in pt[0]]); OUT["constA"].append(float(frac(cA[0]))); OUT["scoreA"].append(int(SA[wA]))
    OUT["marg2"].append([float(frac(a - b)) for a, b in zip(pt[0], pt[1])]); OUT["marg2c"].append(float(frac(cA[0] - cA[1])))
    OUT["margu"].append([float(frac(a - b)) for a, b in zip(pt[0], pt[2])]); OUT["marguc"].append(float(frac(cA[0] - cA[iu])))
    E = recA["R0"]; Ra = fvals(NA[0], eA[0]); Ru = fvals(NA[iu], eA[iu]); R2 = fvals(NA[1], eA[1])
    rngs = np.random.default_rng(i)
    Rash = np.stack([rngs.permutation(Ra[p]) for p in range(P)])
    OUT["half_ans"].append(half_count(E * Ra)); OUT["half_ans_shuf"].append(half_count(E * Rash)); OUT["half_u"].append(half_count(E * Ru))
    Dm = Ra - R2; Dsh = np.stack([rngs.permutation(Dm[p]) for p in range(P)])
    OUT["half_marg"].append(half_count(E * Dm)); OUT["half_marg_shuf"].append(half_count(E * Dsh))
    t = P - 1; win = ids[-1]
    V = rows_met(NA[0, t], eA[0, t]); Vu = rows_met(NA[iu, t], eA[iu, t])
    top10 = [int(x) for x in np.argsort(-V, kind="stable")[:10]]; top10u = [int(x) for x in np.argsort(-Vu, kind="stable")[:10]]
    OUT["top10_ans"].append(top10); OUT["top10_u"].append(top10u)
    OUT["ans_rank_rows"].append(rank_of(V, wA)); OUT["in_rank_rows"].append(rank_of(V, win))
    OUT["u_rank_rows"].append(rank_of(Vu, SET[iu])); OUT["in_rank_rows_u"].append(rank_of(Vu, win))
    Do, eo = add16((NA[0, t][None], eA[0, t][None]), (-NA[iu, t][None], eA[iu, t][None]))
    Vo = rows_met(Do[0], eo[0]); top10o = [int(x) for x in np.argsort(-Vo, kind="stable")[:10]]
    OUT["top10_own"].append(top10o); OUT["ans_rank_own"].append(rank_of(Vo, wA)); OUT["in_rank_own"].append(rank_of(Vo, win))
    OUT["u_rank_own"].append(int((Vo < Vo[SET[iu]]).sum()))
    three = [SET[0], SET[1], SET[iu]]
    cb = reverse16(recA, three, kappas(recA, resA, "bias"), resA)[1]; cr_ = reverse16(recA, three, kappas(recA, resA, "round"), resA)[1]
    OUT["c_bias"].append([float(frac(x)) for x in cb]); OUT["c_round"].append([float(frac(x)) for x in cr_])
    OUT["c_top"].append([int(HEADB[w]) for w in three]); OUT["c_all"].append([float(frac(cA[k])) for k in (0, 1, iu)])
    OUT["ans_in10"].append(wA in top10); OUT["in_in10"].append(win in top10); OUT["u_in10"].append(SET[iu] in top10u); OUT["in_in10_u"].append(win in top10u)
    # ---- (b) predetermined?
    (NB, eB), cB = reverse16(recB, SET, kapB, resB)
    mAB = meetE(E, NB, eB); sAB = [a + b for a, b in zip(mAB, cB)]             # whole numbers at 2^EFIX
    _asc = sorted(sAB); rk = lambda k: len(sAB) - bisect.bisect_right(_asc, sAB[k])
    OUT["rankAB_set"].append(rk(0)); OUT["inAB_set"].append(rk(0) < 5)
    OUT["randAB_in5"].append(sum(1 for k in range(len(SET)) if SET[k] in rnd and rk(k) < 5) / len(rnd))
    OUT["wB_rank_set"].append(rk(iB))
    OUT["rankAB_top5"].append([rk(k) for k in range(5)])                         # A's 5 top words under R from B (frequency-matched control)
    OUT["frozen_self"].append(bool((record(TEXTS[j], held=recB)["top"]["scores"] == SB).all()))
    fz = record(ids, held=recB)["top"]["scores"]
    OUT["frozen_vs_RB"].append(max(abs(float(frac(sAB[k])) - int(fz[SET[k]])) for k in range(len(SET))))   # same relation, A's roundings
    OUT["rank_frozen_top5"].append([rank_of(fz, w) for w in topA])
    OUT["rank_frozen_wA"].append(rank_of(fz, wA)); OUT["rank_frozen_wB"].append(rank_of(fz, wB))
    OUT["wA_in_Btop5"].append(rank_of(SB, wA) < 5); OUT["rank_wA_in_B"].append(rank_of(SB, wA))
    allp = range(P)
    OUT["alike_AB"].append(alike(NA[0], eA[0], NB[0], eB[0], allp))
    OUT["alike_AA_u"].append(alike(NA[0], eA[0], NA[iu], eA[iu], allp))
    OUT["alike_AB_u"].append(alike(NA[0], eA[0], NB[iu], eB[iu], allp))
    rec3 = retop(recA, P - 2); S3 = rec3["top"]["scores"]
    assert (S3 == machine_scores(ids, at=P - 2)).all()
    res3 = (resA[0], dict(x=res_shift(rec3["top"]["x16"], rec3["top"]["Rt"], rec3["top"]["sh"]))); kap3 = (kapA[0], res3[1])
    (N3, e3), c3 = reverse16(rec3, [wA, SET[iu]], kap3, res3)
    OUT["alike_t43"].append(alike(NA[0], eA[0], N3[0], e3[0], range(P - 1)))
    OUT["alike_t43_u"].append(alike(NA[0], eA[0], N3[1], e3[1], range(P - 1)))
    OUT["alike_t43_own"].append(alike(NA[0][[P - 1]], eA[0][[P - 1]], N3[0][[P - 2]], e3[0][[P - 2]], [0]))
    OUT["rank_wA_at3"].append(rank_of(S3, wA))
    # ---- (c) the fringes of the final meeting
    fr = {"real": [0, 0], "u": [0, 0], "shuf": [0, 0], "shuf2": [0, 0]}; coinv = Fraction(0); okall = True; cr = [0, 0]; cs = [0, 0]
    for p in range(P):
        Ep = E[p]
        Ra16 = rshift(NA[0, p], int(max(int(bitlen(np.bitwise_or.reduce(np.abs(NA[0, p])))) - 15, 0)))
        Ru16 = rshift(NA[iu, p], int(max(int(bitlen(np.bitwise_or.reduce(np.abs(NA[iu, p])))) - 15, 0)))
        for key, Rp in [("real", Ra16), ("u", Ru16), ("shuf", rngs.permutation(Ra16)), ("shuf2", rngs.permutation(Ra16))]:
            d, ok = fringe(Ep, Rp); okall &= ok
            f = np.abs(768 - 2 * d); fr[key][0] += int(f.sum()); fr[key][1] += f.size * 768
            if key in ("real", "shuf"):
                nE, nR = d.shape
                for a_ in range(nE - 4, nE):
                    for b_ in range(nR - 4, nR):
                        k_ = (nE - 1 - a_, nR - 1 - b_)
                        (DREAL if key == "real" else DSHUF).setdefault(k_, []).append(int(f[a_, b_]))
                corner = f[nE - 2:, nR - 2:].sum()
                if key == "real": cr[0] += int(corner); cr[1] += 4 * 768
                else: cs[0] += int(corner); cs[1] += 4 * 768
        coinv += coin(768)
    for k in fr: OUT["fr_" + k].append(fr[k][0] / fr[k][1])
    OUT["fr_coin"].append(float(coinv / P)); OUT["fr_ok"].append(okall)
    OUT["corner_real"].append(cr[0] / cr[1]); OUT["corner_shuf"].append(cs[0] / cs[1])
    print(f"  text {i:2d} {''.join(word(x) for x in ids)!r:42s} answer {word(wA)!r:12s} | identity: max gap "
          f"{max(abs(g) for g in gaps):7.0f} units, top 5 same {OUT['top5_same_order'][-1]} | B's R: answer rank {rk(0):3d}/{len(SET)} "
          f"| A's words, B's dimmings: answer rank {OUT['rank_frozen_wA'][-1]}, B's answer rank {OUT['rank_frozen_wB'][-1]} "
          f"| fringe {100 * fr['real'][0] / fr['real'][1]:.2f}%  ({time.time() - t0:.0f}s)", flush=True)

np.savez("/Users/abundancemachine/streaming-machine/reverse/looks.npz", **{k: np.array(v, dtype=object) for k, v in OUT.items()},
         dreal=np.array([[k[0], k[1], np.mean(v)] for k, v in DREAL.items()]), dshuf=np.array([[k[0], k[1], np.mean(v)] for k, v in DSHUF.items()]))

# ------------------------------------------------------------------------------------------------ report
A = {k: v for k, v in OUT.items()}
pc = lambda x: f"{100 * x:.1f}%"
print(f"\n================ STEP 2 AT 16 PLACES ({NT} texts x 205-206 words) ================")
g = np.abs(np.concatenate([np.array(x, float) for x in A["idgap"]])); gr = np.concatenate([np.array(x, float) for x in A["idgap_rel"]])
print(f"  |E.R_w + c_w - machine score| in units of 2^-16: median {np.median(g):.0f}, 90th {np.percentile(g, 90):.0f}, max {g.max():.0f}"
      f"  (relative to the score: median {np.median(gr):.1e}, max {gr.max():.1e})")
print(f"  top 5 by the reversed patterns = machine's top 5: same order {sum(A['top5_same_order'])}/{NT}, same set {sum(A['top5_same_set'])}/{NT}")
print("\n================ (a) WHERE THE ANSWER SITS ================")
pa = np.array(A["partsA"]); ca = np.array(A["constA"]); sc = np.array(A["scoreA"], dtype=float)
print("  answer's score = sum over positions of E_p.R_w,p + c_w.  Mean over texts, in units (2^-16), and as share of the score:")
for p in range(5): print(f"     position {p}: {pa[:, p].mean():12.0f}   {pc((pa[:, p] / sc).mean())}")
print(f"     constant c_w: {ca.mean():10.0f}   {pc((ca / sc).mean())}")
m2 = np.array(A["marg2"]); m2c = np.array(A["marg2c"]); mu = np.array(A["margu"]); muc = np.array(A["marguc"])
print("  the answer's margin over the machine's 2nd word, split the same way (mean units; share of the margin; texts where positive):")
tot2 = m2.sum(1) + m2c
for p in range(5): print(f"     position {p}: {m2[:, p].mean():10.0f}   {pc((m2[:, p] / tot2).mean()):>8s}   positive {int((m2[:, p] > 0).sum())}/{NT}")
print(f"     constant:   {m2c.mean():10.0f}   {pc((m2c / tot2).mean()):>8s}   positive {int((m2c > 0).sum())}/{NT}      (margin itself: median {np.median(tot2):.0f})")
totu = mu.sum(1) + muc
print("  the answer's margin over a random word (control word):")
for p in range(5): print(f"     position {p}: {mu[:, p].mean():10.0f}   {pc((mu[:, p] / totu).mean()):>8s}   positive {int((mu[:, p] > 0).sum())}/{NT}")
print(f"     constant:   {muc.mean():10.0f}   {pc((muc / totu).mean()):>8s}   positive {int((muc > 0).sum())}/{NT}      (margin itself: median {np.median(totu):.0f})")
rc, re2 = np.array(A["rank_const_only"]), np.array(A["rank_embed_only"])
print(f"  the answer's rank among the same 205-206 words by the constant alone (embedding numbers dark): top 5 {int((rc < 5).sum())}/{NT}, first {int((rc == 0).sum())}/{NT}, median {np.median(rc):.0f};"
      f"  by E.R_w alone (constant removed): top 5 {int((re2 < 5).sum())}/{NT}, first {int((re2 == 0).sum())}/{NT}, median {np.median(re2):.0f}")
print(f"  (position, line) entries carrying half of |E x R|: answer {pc(np.mean(A['half_ans']))}, its lines shuffled {pc(np.mean(A['half_ans_shuf']))},"
      f" random word {pc(np.mean(A['half_u']))};  the margin over the 2nd word {pc(np.mean(A['half_marg']))}, shuffled {pc(np.mean(A['half_marg_shuf']))}")
print(f"  last position, word rows R_answer meets most (top 10 of 50,257): answer itself among them {sum(A['ans_in10'])}/{NT} "
      f"(median rank {np.median(A['ans_rank_rows']):.0f}); the input word {sum(A['in_in10'])}/{NT} (median rank {np.median(A['in_rank_rows']):.0f})")
print(f"     control, a random word's R: that word itself among its top 10 {sum(A['u_in10'])}/{NT} (median rank {np.median(A['u_rank_rows']):.0f});"
      f" the input word {sum(A['in_in10_u'])}/{NT} (median rank {np.median(A['in_rank_rows_u']):.0f})")
print(f"  the answer's own part at the last position (R_answer - R_random word): answer among its top 10 rows "
      f"{sum(1 for r in A['ans_rank_own'] if r < 10)}/{NT} (median rank {np.median(A['ans_rank_own']):.0f}); input word median rank "
      f"{np.median(A['in_rank_own']):.0f}; the random word's rank from the bottom median {np.median(A['u_rank_own']):.0f}")
cb_, cr2, ct_, ca_ = (np.array(A[k], dtype=float) for k in ("c_bias", "c_round", "c_top", "c_all"))
print(f"  the constant split (answer / 2nd / random; mean units): all {ca_.mean(0).round(0)}; the biases met with the reverse light "
      f"{cb_.mean(0).round(0)} (of which the top's standing amount {ct_.mean(0).round(0)}); the roundings {cr2.mean(0).round(0)}")
print(f"     margin answer - 2nd from the constant: biases {np.mean(cb_[:, 0] - cb_[:, 1]):.0f} (top's standing amount "
      f"{np.mean(ct_[:, 0] - ct_[:, 1]):.0f}), roundings {np.mean(cr2[:, 0] - cr2[:, 1]):.0f};  sum check vs c: "
      f"{np.abs(cb_ + cr2 - ca_).max():.3g}")
for i in range(min(3, NT)):
    print(f"     text {i}: answer {word(A['answers'][i])!r}: {[word(x) for x in A['top10_ans'][i]]}")
    print(f"             random word's R: {[word(x) for x in A['top10_u'][i]]}")
    print(f"             answer's own part: {[word(x) for x in A['top10_own'][i]]}")
print("\n================ (b) PREDETERMINED? ================")
print(f"  frozen machine with B's own dimmings on B's own words reproduces B exactly: {sum(A['frozen_self'])}/{NT}")
print(f"  R_w from B's run met with A's embedding numbers + B's constant (206 words): A's answer in the top 5 {sum(A['inAB_set'])}/{NT}"
      f" (median rank {np.median(A['rankAB_set']):.0f});  control, a random word in the top 5: {pc(np.mean(A['randAB_in5']))} of random words;"
      f"  B's answer: median rank {np.median(A['wB_rank_set']):.0f}, top 5 {sum(1 for r in A['wB_rank_set'] if r < 5)}/{NT}")
r5 = np.array(A["rankAB_top5"]); f5 = np.array(A["rank_frozen_top5"])
print(f"     frequency-matched control, A's own 2nd..5th words under R from B (206 words): in the top 5 {pc((r5[:, 1:] < 5).mean())} of them"
      f" (answer {pc((r5[:, 0] < 5).mean())})")
print(f"  R_w from B met with A's numbers + B's constant vs A's words run through B's dimmings: largest gap over the 206 words, median "
      f"{np.median(A['frozen_vs_RB']):.0f} units, max {np.max(A['frozen_vs_RB']):.0f} (the same relation; only the roundings and the 16 places differ)")
print(f"     A's 2nd..5th words, A's words through B's dimmings (50,257): in the top 5 {pc((f5[:, 1:] < 5).mean())} (median rank {np.median(f5[:, 1:]):.0f})")
rf = np.array(A["rank_frozen_wA"]); rb = np.array(A["rank_frozen_wB"])
print(f"  A's own words through B's dimmings (all 50,257 words): A's answer in the top 5 {int((rf < 5).sum())}/{NT} (first {int((rf == 0).sum())}, median rank {np.median(rf):.0f});"
      f" B's answer in the top 5 {int((rb < 5).sum())}/{NT} (first {int((rb == 0).sum())}, median rank {np.median(rb):.0f})")
print(f"  coincidence control: A's answer in B's own top 5 {sum(A['wA_in_Btop5'])}/{NT} (median rank {np.median(A['rank_wA_in_B']):.0f})")
al = lambda key: (np.mean([x[0] for x in A[key]]), np.mean([x[1] for x in A[key]]))
for key, name in [("alike_AB", "same word, A's run vs B's run"), ("alike_AA_u", "control: answer vs random word, both A's run"),
                  ("alike_AB_u", "control: answer from A vs random word from B"),
                  ("alike_t43", "same word, same text: read at last position vs one earlier (positions 0-3)"),
                  ("alike_t43_u", "control: answer at last vs random word one earlier (positions 0-3)"),
                  ("alike_t43_own", "same word, own position: last's part vs one-earlier's part")]:
    s, ph = al(key); print(f"  {name:80s} same sign {pc(s)}  photons of the difference / photons of R {pc(ph)}")
print(f"  A's answer in the machine's own scores one position earlier: median rank {np.median(A['rank_wA_at3']):.0f}, top 5 {sum(1 for r in A['rank_wA_at3'] if r < 5)}/{NT}")
print("\n================ (c) THE FRINGES OF THE FINAL MEETING ================")
print(f"  every meeting rebuilt from its place-pair counts equals E.R16 exactly: {all(A['fr_ok'])}")
print(f"  fringe share (|N - 2d| over all place pairs and lines): answer {pc(np.mean(A['fr_real']))}  random word {pc(np.mean(A['fr_u']))}"
      f"  shuffle {pc(np.mean(A['fr_shuf']))}  second shuffle {pc(np.mean(A['fr_shuf2']))}  coin toss {pc(np.mean(A['fr_coin']))}"
      f"   -> pairs off straight through: {pc(1 - np.mean(A['fr_real']))}")
print(f"  top corner (embedding's top 2 places x R's top 2 places): answer {pc(np.mean(A['corner_real']))}, shuffle {pc(np.mean(A['corner_shuf']))}")
print("  mean |N - 2d| by place from the top (embedding place, R place): real / shuffle")
for a_ in range(4):
    print("     " + "  ".join(f"({a_},{b_}) {np.mean(DREAL[(a_, b_)]):5.1f}/{np.mean(DSHUF[(a_, b_)]):5.1f}" for b_ in range(4)))
print(f"\n({time.time() - t0:.0f}s)")
