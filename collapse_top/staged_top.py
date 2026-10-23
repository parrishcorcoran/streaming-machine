"""THE TOP READ AS A COLLAPSING WAVEFORM, EXACT TOP 5, STAGED DOWN THE TRIANGLE.

The machine's top read (base2.py / judge5.py), unchanged. At a read position the light x (768 whole numbers on 16 bits,
from collapse_top/light_top.npz, saved by light_top.py with the machine's own steps) meets every word's stamp T_w (768
whole numbers, int16; head.e is one power of two for the whole table). The word's score is
    score_w = power(pile_w, F) + bias_w,   pile_w = sum over lines x_i T_wi,   F = (e + ex + 16) x 65536 - LS,
F the same for every word, and power never runs backwards (a bigger pile never gives a smaller score).

Every value is its sign and its photons (the 1s of |value|). Places 0..14: the light's top place is 14 at every
position (to16), the stamp's largest value is 24,962 (top place 14); place 15 is dark on both sides. A light photon at
place a meeting a stamp photon at place b on the same line is a pair on diagonal a + b, worth 2^(a+b), up or down.
The pairs (a, b) are the triangle; the top diagonal is 28.

STAGES, fixed in advance (k = 4, 8, 12, then all; places counted down from place 14, the top place of both sides):
  sched=stamp   stamp places 14 .. 15-k, every light place
  sched=light   light places 14 .. 15-k, every stamp place
  sched=diag    the top k diagonals: a + b >= 29 - k
After each stage, what the uncounted pairs can still add to a word is bounded exactly, without reading them:
  |rest_w| <= sum over uncounted (a, b) of 2^(a+b) x min(n_a, m_wb)
  n_a   lines where the light has a photon at place a (the light's own, known at the start)
  m_wb  lines where word w's stamp has a photon at place b (15 counts a word, stored once with the stamp)
(a pair needs a photon on both sides of the same line, so it occurs on no more lines than either side has photons).
  lo_w = power(counted_w - bound_w, F) + bias_w  <=  score_w  <=  hi_w = power(counted_w + bound_w, F) + bias_w
A word is set aside when hi_w < the 5th-best lo: five words are then surely above it. After the last stage the words
still in are read exactly (the rest of their pairs) and the top 5 taken among them (judge5.top5).

WORK, in the unit of all earlier results: photon meetings = light photons x stamp photons on each line, for the place
pairs actually met, summed over the words still in. Stage 1 meets every word with its own pairs; each later stage
meets only the words still in, with only its own new pairs. Stated apart: the bound (15 table reads and adds per word
per stage, plus a 15 x 769 table built from the light per stage), 2 extra gauge reads and 1 compare per word per stage.

CHECKS: lo <= exact <= hi for every word at every stage; the staged pieces against a pair-by-pair count on sampled
words; the stages' new pairs add up to the plain read's meetings; the top 5 (set and order) equal to the plain read's
at every position; on the windows also judge5's reference top 5 and judge5.judge itself; on the refs the race winner
equal to the one saved with the light.
CONTROL: light=shuffled shuffles each position's light across its lines (every value kept, only which line it meets).
ORDER-FREE: order=1 permutes the words and the lines (light and stamp together) first; nothing may change.

PREDICTIONS (written before this script first ran):
  the task's:  a few hundred to a few thousand words survive 8 stamp places; the staged read cuts the top read to
               30-50% of plain (12-17% of the whole machine).
  Claude's:    the exact bound has to cover the unread places without reading them, so it is wider than collapse.py's
               (collapse.py read the bottom 8 places of every word to make its bound). After 8 stamp places 2,000 to
               20,000 words survive; after 12, under 100. Stamp schedule 40-60% of plain; light schedule no better;
               diagonal schedule the best of the three, 15-35% of plain.

    python3 collapse_top/staged_top.py sched=stamp|light|diag [light=real|shuffled] [order=0|1] [set=refs|windows|both]
      -> collapse_top/staged_<sched>_<light>[_order].npz, summary printed
"""
import sys, os, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
SCHED, LIGHT, ORDER, SET = ARG.get("sched", "stamp"), ARG.get("light", "real"), int(ARG.get("order", 0)), ARG.get("set", "both")
BATCH = int(ARG.get("batch", 100))
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
CNT_ALL = np.stack([((TA >> b) & 1).sum(0, dtype=np.int64) for b in range(PL)])          # words with a photon at b, per line
PPH = np.bitwise_count(TA).sum(0, dtype=np.int64)     # stamp photons per line (all words)

a_, b_ = np.arange(PL)[:, None], np.arange(PL)[None, :]
CS = []
for k in (4, 8, 12):
    C = (b_ >= PL - k) if SCHED == "stamp" else (a_ >= PL - k) if SCHED == "light" else (a_ + b_ >= 2 * (PL - 1) + 1 - k)
    CS.append(np.broadcast_to(C, (PL, PL)).copy())
CS.append(np.ones((PL, PL), bool))
NEW = [CS[0]] + [CS[s] & ~CS[s - 1] for s in range(1, 4)]
def amask(C): return np.array([sum(1 << a for a in range(PL) if C[a, b]) for b in range(PL)], np.int64)
AMASK = np.stack([amask(C) for C in CS]); NMASK = np.stack([amask(N) for N in NEW])
POWA = (1 << np.arange(PL)).astype(np.int64); NN = np.arange(LINES + 1)
TABLE_ADDS = [int((~CS[s]).any(1).sum()) * PL * (LINES + 1) for s in range(3)]          # the bound's table, per stage

def bound(nx, s, rows):                               # sum over uncounted (a,b) 2^(a+b) min(n_a, m_wb), for the words in rows
    G = ((~CS[s]).T * POWA[None, :]) @ np.minimum(nx[:, None], NN[None, :])                # (15 b, 769)
    return (G[np.arange(PL)[None, :], NT[rows]] << np.arange(PL)[None, :]).sum(1)

def pair_by_pair(x, w, C):                            # counted pile of word w, the slow way: pair by pair, line by line
    ax, sx, tw, sw = np.abs(x), np.sign(x), TA[w].astype(np.int64), TS[w].astype(np.int64)
    return sum(int(((sx * sw) * ((ax >> a) & 1) * ((tw >> b) & 1)).sum()) << (a + b) for a in range(PL) for b in range(PL) if C[a, b])

sel = np.flatnonzero((SRC == 0) if SET == "refs" else (SRC == 1) if SET == "windows" else np.ones(len(SRC), bool))
N = len(sel)
SURV = np.zeros((N, 3), np.int64); ENTER = np.zeros((N, 3), np.int64); WORK = np.zeros((N, 4), np.int64)
PLAIN = np.zeros(N, np.int64); TOP_ST = np.zeros((N, 5), np.int64); TOP_PL = np.zeros((N, 5), np.int64)
WIN_ST = np.zeros(N, np.int64); WIN_PL = np.zeros(N, np.int64)
FIN_IDX, FIN_SC, FIN_OFF = [], [], [0]
BAD = {"bound": 0, "final_vs_plain": 0, "pieces": 0, "pairs_sum": 0, "pieces_checked": 0}
rng_chk = np.random.default_rng(5)

for c0 in range(0, N, BATCH):
    g = sel[c0:c0 + BATCH]; P = len(g)
    X = X_ALL[g]; ax, sx = np.abs(X), np.sign(X)
    F = (WE[0] + EXA[g] + J.ES) * 65536 - LSA[g]
    S = J.power(J.product(X, WH), F[:, None]) + BIAS[None, :]                             # the plain top read, every word
    CNTD = np.zeros((3, P, NW), np.float64)                                               # counted piles after stages 1..3
    for b in range(PL):                                                                   # (whole numbers below 2^41: exact)
        live = [s for s in range(3) if AMASK[s, b]]
        if not live: continue
        Pb = (TS * ((TA >> b) & 1).astype(np.int8)).astype(np.float64)                    # the stamp's photons at b, with direction
        for s in live:
            prod = (sx * (ax & AMASK[s, b])).astype(np.float64) @ Pb.T                    # the light's places met with b: each sum
            prod *= float(1 << b); CNTD[s] += prod                                        # below 768 x 2^15, whole: exact
        del Pb, prod
    CNTD = CNTD.astype(np.int64)
    ALV = np.zeros((4, P, NW), np.float32); ALV[0] = 1
    for p in range(P):
        nx = ((ax[p][None, :] >> np.arange(PL)[:, None]) & 1).sum(1)
        alive = np.arange(NW)
        for s in range(3):
            R = bound(nx, s, alive); c = CNTD[s, p, alive]
            lo = J.power(c - R, F[p]) + BIAS[alive]; hi = J.power(c + R, F[p]) + BIAS[alive]
            ex = S[p, alive]; BAD["bound"] += int(((lo > ex) | (ex > hi)).sum())
            L5 = np.partition(lo, len(lo) - 5)[len(lo) - 5]
            ENTER[c0 + p, s] = len(alive); alive = alive[hi >= L5]; SURV[c0 + p, s] = len(alive)
            ALV[s + 1, p, alive] = 1
        if c0 == 0 and p < 4:                                                             # the pieces, pair by pair
            for w in rng_chk.choice(NW, 16, replace=False):
                for s in range(3):
                    BAD["pieces"] += pair_by_pair(X[p], w, CS[s]) != CNTD[s, p, w]; BAD["pieces_checked"] += 1
        exact = J.power(J.product(X[p:p + 1], WH if len(alive) == NW else WH[:, alive]), F[p])[0] + BIAS[alive]   # read exactly
        BAD["final_vs_plain"] += int((exact != S[p, alive]).sum())
        Sp = np.full(NW, FLOOR, np.int64); Sp[alive] = exact
        TOP_ST[c0 + p] = WPERM[top5(Sp[None])[0]]; TOP_PL[c0 + p] = WPERM[top5(S[p:p + 1])[0]]
        WIN_ST[c0 + p] = WPERM[J.race(Sp)]; WIN_PL[c0 + p] = WPERM[J.race(S[p])]
        k = np.arange(len(alive)) if len(alive) <= KEEPN else np.argpartition(-exact, KEEPN - 1)[:KEEPN]   # saved: the words
        ids = WPERM[alive[k]]; o = np.argsort(ids)                                        # still in, or their best KEEPN
        FIN_IDX.append(ids[o]); FIN_SC.append(exact[k][o]); FIN_OFF.append(FIN_OFF[-1] + len(k))
    del CNTD, S
    work = np.zeros((4, P), np.int64); allpairs = np.zeros(P, np.int64)
    A2 = ALV.reshape(4 * P, NW)
    for b in range(PL):
        Bb = ((TA >> b) & 1).astype(np.float32)
        CN = (A2 @ Bb).astype(np.int64).reshape(4, P, LINES)                              # words still in with a photon at b, per line
        for s in range(4):
            Lc = np.bitwise_count(ax & NMASK[s, b]).astype(np.int64)                     # light photons at the new places met with b
            work[s] += (Lc * CN[s]).sum(1); allpairs += (Lc * CNT_ALL[b][None, :]).sum(1)
        del Bb, CN
    plain = (np.bitwise_count(ax).astype(np.int64) * PPH[None, :]).sum(1)
    BAD["pairs_sum"] += int((allpairs != plain).sum())
    WORK[c0:c0 + P] = work.T; PLAIN[c0:c0 + P] = plain
    del ALV, A2
    print(f"  {c0 + P}/{N} positions ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)", flush=True)

tag = f"{SCHED}_{LIGHT}" + ("_order" if ORDER else "") + ("" if SET == "both" else f"_{SET}")
np.savez(f"{HERE}/collapse_top/staged_{tag}.npz", sel=sel, surv=SURV, enter=ENTER, work=WORK, plain=PLAIN,
         top_st=TOP_ST, top_pl=TOP_PL, win_st=WIN_ST, win_pl=WIN_PL, fin_idx=np.concatenate(FIN_IDX),
         fin_sc=np.concatenate(FIN_SC), fin_off=np.array(FIN_OFF), bad=np.array([BAD[k] for k in sorted(BAD)]),
         bad_keys=np.array(sorted(BAD)))

def pct(a, b): return int(round(100 * a / b))
print(f"\nsched={SCHED} light={LIGHT} order={ORDER} set={SET}: {N} positions")
print(f"  checks: bound broken {BAD['bound']} times; final read != plain {BAD['final_vs_plain']}; pieces != pair-by-pair "
      f"{BAD['pieces']}/{BAD['pieces_checked']}; new pairs of the stages != plain meetings {BAD['pairs_sum']}")
same5 = (TOP_ST == TOP_PL).all(1); print(f"  top 5 (set and order) same as the plain read: {int(same5.sum())}/{N};"
                                         f" same word (race): {int((WIN_ST == WIN_PL).sum())}/{N}")
r = SRC[sel] == 0
if r.any() and LIGHT == "real": print(f"  refs: race winner = the winner saved with the light: {int((WIN_ST[r] == REFW[sel[r]]).sum())}/{int(r.sum())}")
for s, k in enumerate((4, 8, 12)):
    v = SURV[:, s]; print(f"  survivors after top {k:2d}: median {int(np.median(v))}, mean {int(round(v.mean()))}, min {v.min()}, max {v.max()}")
pl = PLAIN.mean()
for s in range(4): print(f"  stage {s + 1} work: {int(round(WORK[:, s].mean() / 1e6))}M per position")
tot = WORK.sum(1).mean()
print(f"  plain top read here {int(round(pl / 1e6))}M per position; staged {int(round(tot / 1e6))}M = {pct(tot, pl)}% of plain")
print(f"  extra: bound adds {int(round((ENTER.sum(1).mean() * PL + sum(TABLE_ADDS)) / 1e3))}K, gauge reads "
      f"{int(round(2 * ENTER.sum(1).mean() / 1e3))}K, compares {int(round(ENTER.sum(1).mean() / 1e3))}K per position")
if SET != "refs" and LIGHT == "real" and ORDER == 0:                                      # judge5's reference top 5
    REF = np.load(f"{HERE}/judge5_ref.npz")["top5"]; wq = np.flatnonzero(SRC[sel] == 1)
    NREF = int((SRC == 0).sum()); rows = [(q, *divmod(int(sel[q]) - NREF, 20)) for q in wq]                 # window i, place j
    assert all(WIN[sel[q]] == i for q, i, j in rows)
    same_ref = sum(int((TOP_ST[q] == REF[i, j]).all()) for q, i, j in rows); same_pl = sum(int((TOP_PL[q] == REF[i, j]).all()) for q, i, j in rows)
    print(f"  windows: top 5 (set and order) = judge5's reference top 5: staged {same_ref}/{len(wq)}, plain {same_pl}/{len(wq)}"
          f"  (judge5.judge itself: collapse_top/judge_check.py)")
print(f"  ({time.time() - t0:.0f}s, max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)")
