"""groups2/topg_check.py: a SECOND CODE PATH (same session, NOT the independent rebuild rule 10 asks for) for
groups2/topg.py's up/hole form on GPT-2's top table, at a few refs positions. Shares no code with topg.py or topg.c:
  - the groups by the stated rule, written again;
  - each group's ready piles as one product: the (2^g x g) matrix of +1 / -1 (line t up or a hole in pattern p) times
    the group's light, whole numbers;
  - the patterns from the binary digits of w + 2^15, one place at a time;
  - photons counted digit by digit (odd: digit 2 - (a mod 4), take it away, halve), not by popcount(a ^ 3a);
  - the half table's photons by walking the Gray order explicitly (step i turns line ctz(i)), plus the group total;
  - (ii): the set-aside written again from topband.py's rule at the doubled scale.
Compared with topg_gpt2_refs_g16_main.npz: (i) reads, the table, (ii) reads, survivors at every place, (ii) total.
    python3 groups2/topg_check.py [positions=0,166,494,647]
"""
import sys, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"
ARG = dict(a.split("=") for a in sys.argv[1:]); POS = [int(v) for v in ARG.get("positions", "0,166,494,647").split(",")]
a_ = sys.argv; sys.argv = ["x"]; sys.path.insert(0, f"{HERE}/topband"); import topband as TB; sys.argv = a_
g = 16
W = np.asarray(TB.M["head.w"]).astype(np.int64); N, M = W.shape; bias = np.asarray(TB.HEAD_B).astype(np.int64)
Z = np.load(f"{HERE}/fringes_machine/refs_top_light.npz"); R = np.load(f"{HERE}/groups2/out/topg_gpt2_refs_g16_main.npz")

def sd_count(v):
    a = np.abs(np.asarray(v, np.int64)).copy(); c = np.zeros(a.shape, np.int64)
    while a.any():
        odd = (a & 1) == 1; c += odd; a = np.where(odd, np.where((a & 2) != 0, a + 1, a - 1), a) >> 1
    return c
def ones_count(v):
    a = np.abs(np.asarray(v, np.int64)); return np.array([bin(int(t)).count("1") for t in a.ravel()]).reshape(a.shape)

order = np.random.default_rng([1, 99, N, g]).permutation(N); groups = [order[s:s + g] for s in range(0, N, g)]
U = (W + 32768).astype(np.int32); assert U.min() >= 0 and U.max() < 1 << 16
PAT = np.zeros((len(groups), M, 16), np.int32)
for k, L in enumerate(groups):
    for t, line in enumerate(L):
        for b in range(16): PAT[k, :, b] |= ((U[line] >> b) & 1) << t
SIGN = np.array([[1 if (p >> t) & 1 else -1 for t in range(g)] for p in range(1 << g)], np.int64)    # (2^g, g)
gray_turns = np.zeros(g, np.int64)
for i in range(1, 1 << (g - 1)): gray_turns[(i & -i).bit_length() - 1] += 1                         # walked, not by formula

for q in POS:
    x = Z["x16"][q].astype(np.int64); X = int(x.sum()); S1 = int(np.abs(x).sum()); sx = sd_count(x)
    A = np.zeros((M, 16), np.int64); PH = np.zeros((M, 16), np.int64); table = 0
    for k, L in enumerate(groups):
        V = SIGN @ x[L]; sV = sd_count(V)
        A += V[PAT[k]]; PH += sV[PAT[k]]
        table += int(sx[L][1:].sum()) + int((gray_turns * sx[L]).sum())
    dbl = (A << np.arange(16)[None, :]).sum(1) - X; pile = x @ W
    ok_pile = bool(np.array_equal(dbl, 2 * pile))
    # the scores and the set-aside, written again from topband.py's rule
    Fp = int((TB.HEAD_E[0] + int(Z["ex"][q]) + TB.ES) * 65536 - int(Z["LS"][q])); e = int(TB.EXP2[Fp & 0xFFFF])
    F2 = -Fp; e2 = int(TB.EXP2[F2 & 0xFFFF]); s = 16 - (Fp >> 16); s2 = 16 - (F2 >> 16)
    beta = TB.power(bias, F2); two_s = Fraction(2) ** s; two_t = Fraction(2) ** (s + s2)
    gm = Fraction(1, 2) + int(np.abs(bias).max()) * two_s * abs(e * e2 - two_t) / (e * two_t)
    mu = 2 * -(-gm.numerator // gm.denominator) + max(1, -(-two_s.numerator // (e * two_s.denominator)))
    S = TB.power(pile, Fp) + bias; s5 = np.sort(S)[-5]
    alive = np.arange(M); reads = 0; surv = {}; partial = np.zeros(M, np.int64); bound_ph = 0
    for b in range(15, -1, -1):
        reads += int(PH[alive, b].sum()); partial = partial + (A[:, b] << b)
        if b == 0: break
        R2 = ((1 << b) - 1) * S1
        assert (np.abs((2 * pile + X) - partial) <= R2).all(), "bound broken"
        V_ = partial[alive] + 2 * beta[alive]; v5 = np.sort(V_)[-5]; thr = 2 * R2 + 2 * mu
        keep = V_ > v5 - thr; assert (S[alive[~keep]] < s5).all(), "a word set aside at or above the 5th-best"
        bound_ph += bin(S1).count("1") * bin((1 << b) - 1).count("1") + bin(2 * mu).count("1") + bin(thr).count("1")
        alive = alive[keep]; surv[b] = len(alive)
    pa = (partial[alive] - X) >> 1; assert np.array_equal(pa, pile[alive])
    end = sd_count(np.array([X]))[0] * len(alive) + int(ones_count(beta[alive]).sum()) + int(ones_count(pa).sum()) * bin(e).count("1") + int(ones_count(bias[alive]).sum())
    stand = int(ones_count(bias).sum()) * bin(e2).count("1") + int(ones_count(beta).sum())
    total = table + 2 * int(sx[1:].sum()) + reads + bound_ph + end + stand
    same = dict(pile=ok_pile, i_reads=int(PH.sum()) == int(R["uh__i_reads"][q]), table=table == int(R["uh__table"][q]),
                ii_reads=reads == int(R["uh__ii_reads"][q]), surv=all(surv[b] == int(R["uh__ii_surv"][q, b]) for b in surv),
                ii_total=total == int(R["uh__ii_total"][q]))
    print(f"position {q}: " + ", ".join(f"{k} {v}" for k, v in same.items()) + f"   (ii) {total:,} survivors {[surv[b] for b in sorted(surv, reverse=True)]}", flush=True)

# ---------- the per-place form (pp), written again: up/hole at places 0-11 (offset 2^11), the rest in non-adjacent
# signed digits; the up/hole places taken from the saved run's rule record (it is a plate-only choice) ----------
import json
J = json.load(open(f"{HERE}/groups2/out/topg_gpt2_refs_g16_main.json")); ppmask = J["ppmask"]
UP = [b for b in range(16) if (ppmask >> b) & 1]; hi = max(UP); assert UP == list(range(hi + 1)), "up/hole places 0..hi expected here"
r_ = W + (1 << hi); low = (r_ & ((1 << (hi + 1)) - 1)).astype(np.int32); upper = r_ >> (hi + 1)
SD = {}                                                                         # place -> (lines, words, signs)
n_ = upper.copy(); place = hi + 1
while n_.any():
    odd = (n_ & 1) == 1; d = np.where(odd, 2 - np.mod(n_, 4), 0)
    nz = np.nonzero(d)
    if len(nz[0]): SD[place] = (nz[0], nz[1], d[nz])
    n_ = (n_ - d) // 2; place += 1
back = low.astype(np.int64) - (1 << hi)
for b, (t, j, s) in SD.items(): np.add.at(back, (t, j), s << b)
assert np.array_equal(back, W), "pp digits do not give the table back"
PATP = np.zeros((len(groups), M, hi + 1), np.int32)
for k, L in enumerate(groups):
    for t, line in enumerate(L):
        for b in range(hi + 1): PATP[k, :, b] |= ((low[line] >> b) & 1) << t
tn = lambda n: ((1 << (n + 1)) - 1) // 3 if n > 0 else 0
def bound(b):                                                                   # places below b, doubled scale, a unit of light
    v = (1 << min(b, hi + 1)) - 1
    if b > hi + 1: v += 2 * (1 << (hi + 1)) * tn(b - hi - 1)
    return v
places = sorted(set(range(hi + 1)) | set(SD), reverse=True)
for q in POS:
    x = Z["x16"][q].astype(np.int64); X = int(x.sum()); S1 = int(np.abs(x).sum()); sx = sd_count(x)
    A = {b: np.zeros(M, np.int64) for b in places}; PH = {b: np.zeros(M, np.int64) for b in places}
    for k, L in enumerate(groups):
        V = SIGN @ x[L]; sV = sd_count(V)
        for b in range(hi + 1): A[b] += V[PATP[k, :, b]]; PH[b] += sV[PATP[k, :, b]]
    for b, (t, j, s) in SD.items():
        np.add.at(A[b], j, 2 * s * x[t]); np.add.at(PH[b], j, sx[t])
    kc = ((1 << (hi + 1)) - 1) - 2 * (1 << hi)
    pile = x @ W; dbl = sum(A[b] << b for b in places) + kc * X; ok_pile = bool(np.array_equal(dbl, 2 * pile))
    Fp = int((TB.HEAD_E[0] + int(Z["ex"][q]) + TB.ES) * 65536 - int(Z["LS"][q])); e = int(TB.EXP2[Fp & 0xFFFF])
    F2 = -Fp; e2 = int(TB.EXP2[F2 & 0xFFFF]); s = 16 - (Fp >> 16); s2 = 16 - (F2 >> 16)
    beta = TB.power(bias, F2); two_s = Fraction(2) ** s; two_t = Fraction(2) ** (s + s2)
    gm = Fraction(1, 2) + int(np.abs(bias).max()) * two_s * abs(e * e2 - two_t) / (e * two_t)
    mu = 2 * -(-gm.numerator // gm.denominator) + max(1, -(-two_s.numerator // (e * two_s.denominator)))
    S = TB.power(pile, Fp) + bias; s5 = np.sort(S)[-5]
    alive = np.arange(M); reads = 0; surv = {}; partial = np.zeros(M, np.int64); bound_ph = 0
    for i, b in enumerate(places):
        reads += int(PH[b][alive].sum()); partial = partial + (A[b] << b)
        if i == len(places) - 1: break
        R2 = bound(b) * S1
        assert (np.abs((2 * pile - kc * X) - partial) <= R2).all(), "bound broken"
        V_ = partial[alive] + 2 * beta[alive]; v5 = np.sort(V_)[-5]; thr = 2 * R2 + 2 * mu
        keep = V_ > v5 - thr; assert (S[alive[~keep]] < s5).all()
        bound_ph += bin(S1).count("1") * bin(bound(b)).count("1") + bin(2 * mu).count("1") + bin(thr).count("1")
        alive = alive[keep]; surv[b] = len(alive)
    pa = (partial[alive] + kc * X) >> 1; assert np.array_equal(pa, pile[alive])
    end = sd_count(np.array([kc * X]))[0] * len(alive) + int(ones_count(beta[alive]).sum()) + int(ones_count(pa).sum()) * bin(e).count("1") + int(ones_count(bias[alive]).sum())
    stand = int(ones_count(bias).sum()) * bin(e2).count("1") + int(ones_count(beta).sum())
    table = sum(int(sx[L][1:].sum()) + int((gray_turns * sx[L]).sum()) for L in groups)
    total = table + 2 * int(sx[1:].sum()) + reads + bound_ph + end + stand
    same = dict(pile=ok_pile, i_reads=int(sum(PH[b].sum() for b in places)) == int(R["pp__i_reads"][q]),
                ii_reads=reads == int(R["pp__ii_reads"][q]), surv=all(surv[b] == int(R["pp__ii_surv"][q, b]) for b in surv),
                ii_total=total == int(R["pp__ii_total"][q]))
    print(f"[pp] position {q}: " + ", ".join(f"{k} {v}" for k, v in same.items()) + f"   (ii) {total:,} survivors {[surv[b] for b in places[:-1]]}", flush=True)
