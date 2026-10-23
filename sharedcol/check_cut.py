"""SHAREDCOL/CHECK_CUT.PY: A SECOND CODE PATH FOR cut.py (same session; not an independent rebuild by a separate checker).
No code shared with cut.py: no mm32.c, no cut_read, no rshift copy, no sig11 copy. 2026-10-06.

What it rebuilds, from the stored table (foundation0/base2_pythia-<size>.npz) and the machine's light at the top
(sharedcol/cutlight_<size>.npz, written by cut.py phase=light from pythia_base2's own steps):
  1. the light is the machine's: the plain scores of the last 16 positions of every text, by numpy's own int64 matmul and
     Python-integer rounding, hash to words_pythia-<size>.json's piles_sha256 (the module's own run): 30 of 30.
  2. the column, in Python Fractions: each lane's exact mean on the finest grid, rounded half up to the coarsest grid,
     then to 11 significant places (half up), written out digit by digit; the fringes q - (c << (emax - e_w)).
  3. the fringe piles by numpy int64 matmul; the read by a different rule from cut.py's: the 64 words with the largest
     fringe piles (finest grid) get the shared pile added and their exact score floor(v / 2^s + 1/2) in Python
     integers; the 64th must lie at least one score step below the 5th (so no other word can reach the top 5 or tie
     it); the top 5 is those words by (score high, word number low). The plain read's top 5 by a full stable sort of
     all 50,304 plain scores. Compared at every position; the race winner at the 16 read places against fp32's words.
  4. the counts: plain 1s and signed-digit meetings of the plain read and of the cut read (fringes + shared pile),
     summed position by position with Python integers (bin) on each lane's light and numpy's per-lane totals.
PREDICTION (written before it ran): every number equal to cut_<size>.json's to the digit: top 5 622/622, race 622/622,
words 480/480, sha 30/30, the four meeting totals and the shared pile's equal; the column equal lane for lane.

    python3 sharedcol/check_cut.py size=70m        (writes sharedcol/check_cut_<size>.log)
"""
import os, sys, json, time, hashlib, resource
from fractions import Fraction
import numpy as np

HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/sharedcol"; F0 = f"{HERE}/foundation0"
SIZE = dict(a.split("=", 1) for a in sys.argv[1:]).get("size", "70m")
LOGF = open(f"{OUT}/check_cut_{SIZE}.log", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
T0 = time.time()

Z = np.load(f"{F0}/base2_pythia-{SIZE}.npz")
Q = np.asarray(Z["head.w"]).astype(np.int64); E = [int(v) for v in Z["head.e"]]; d, V = Q.shape
emin, emax = min(E), max(E); Earr = np.array(E, np.int64)

def half_up(fr):                                      # nearest whole number, halves up (Fraction in, int out)
    return (fr + Fraction(1, 2)).__floor__()
def eleven(v):                                        # v written with 11 significant places, halves up
    b = abs(v).bit_length()
    if b <= 11: return v
    k = b - 11; return half_up(Fraction(v, 1 << k)) << k
lane_tot = (Q << (Earr - emin)[None, :]).sum(1)       # each lane's total on the finest grid (int64, exact: < 2^40)
col = [eleven(half_up(Fraction(int(t), V * (1 << (emax - emin))))) for t in lane_tot]
C = np.array(col, np.int64)
U = Q - (C[:, None] << (emax - Earr)[None, :])
cj = json.load(open(f"{OUT}/cut_{SIZE}.json"))
log(f"CHECK_CUT pythia-{SIZE}: column rebuilt in Fractions; |c| up to {int(np.abs(C).max())}; wide words (a fringe value past 16 places) "
    f"{int((np.abs(U) > 32767).any(0).sum())} (cut.py: {cj['wide_words']})")

L = np.load(f"{OUT}/cutlight_{SIZE}.npz"); Y = L["y16"].astype(np.int64); EY = L["ey"]; ROWS = L["rows"]
refs = json.load(open(f"{F0}/refs_pythia-{SIZE}.json"))[:30]
want = [x["piles_sha256"] for x in json.load(open(f"{F0}/words_pythia-{SIZE}.json"))["texts_detail"]]
def bitc(v): return bin(abs(int(v))).count("1")
def nafc(v): a = abs(int(v)); return bin(a ^ (3 * a)).count("1")
lane1_q = np.array([sum(bitc(x) for x in row) for row in Q[:4]])   # spot: the per-lane totals by Python ints on 4 lanes
assert np.array_equal(lane1_q, np.bitwise_count(np.abs(Q[:4])).sum(1))
W1q = np.bitwise_count(np.abs(Q)).sum(1); W1u = np.bitwise_count(np.abs(U)).sum(1)
def nafarr(A): a = np.abs(A); return np.bitwise_count(a ^ (3 * a))
Wnq = nafarr(Q).sum(1); Wnu = nafarr(U).sum(1)
tot = dict(plain1=0, plainn=0, cut1=0, cutn=0, shared1=0, sharedn=0)
same5 = race = words = nwords = sha_ok = gap_bad = 0
for ti, (a, b) in enumerate(ROWS):
    X = Y[a:b]; P = b - a
    PQ = X @ Q; F = X @ U; S = X @ C                  # numpy int64 matmul
    for j in range(P):
        p = a + j; s = -(emin + int(EY[p]) + 16)
        sh = -(Earr + int(EY[p]) + 16)                # the machine's shift for each word: positive = divide, rounded
        pq = PQ[j]
        lg = np.where(sh > 0, (pq + (np.int64(1) << np.maximum(sh - 1, 0))) >> np.maximum(sh, 0), pq << np.maximum(-sh, 0))
        order = np.lexsort((np.arange(V), -lg)); t5p = order[:5]
        Fp = F[j] << (Earr - emin); top = np.argsort(-Fp, kind="stable")[:64]
        S2 = int(S[j]) << (emax - emin)
        exact = {int(w): ((int(Fp[w]) + S2 + (1 << (s - 1))) >> s) if s >= 1 else (int(Fp[w]) + S2) << -s for w in top}
        f5 = sorted((int(Fp[w]) for w in top), reverse=True)[4]
        if not (int(Fp[top[-1]]) <= f5 - (1 << max(s, 0))): gap_bad += 1
        t5c = sorted(exact, key=lambda w: (-exact[w], w))[:5]
        same5 += list(t5c) == [int(w) for w in t5p]; race += int(t5c[0]) == int(t5p[0])
        if j >= P - 16: nwords += 1; words += int(t5c[0]) == refs[ti]["ref"][j - (P - 16)]
        if j == P - 1:
            lgs = []
            for jj in range(P - 16, P):
                shj = -(Earr + int(EY[a + jj]) + 16); pj = PQ[jj]
                lgs.append(np.where(shj > 0, (pj + (np.int64(1) << np.maximum(shj - 1, 0))) >> np.maximum(shj, 0), pj << np.maximum(-shj, 0)))
            sha_ok += hashlib.sha256(np.stack(lgs).astype(np.int64).tobytes()).hexdigest() == want[ti]
        o1 = np.array([bitc(v) for v in X[j]]); on = np.array([nafc(v) for v in X[j]])
        tot["plain1"] += int(o1 @ W1q); tot["plainn"] += int(on @ Wnq); tot["cut1"] += int(o1 @ W1u); tot["cutn"] += int(on @ Wnu)
        tot["shared1"] += sum(int(o) * bitc(c) for o, c in zip(o1, C)); tot["sharedn"] += sum(int(o) * nafc(c) for o, c in zip(on, C))
    log(f"  text {ti:2d}: top 5 same {same5}/{b}  ({time.time() - T0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} MB)")
n = len(Y); cc = cj["counts"]; ck = cj["checks"]
log(f"\nlast-16 plain scores hash to words_pythia-{SIZE}.json's (the module's own run): {sha_ok}/30")
log(f"top 5 (set and order) cut = plain: {same5}/{n} (cut.py {ck['top5_same']}); race {race}/{n} (cut.py {ck['race_same']}); words {words}/{nwords} (cut.py {ck['words_ok']}/{ck['words_n']}); "
    f"64th word not one step below the 5th: {gap_bad}")
mj = json.load(open(f"{OUT}/measure_pythia-{SIZE}.json"))["finished"]["lit"]["mean11"]
log(f"fringe lit places a weight (this rebuild): plain 1s {W1u.sum() / (d * V):.6f}, signed {Wnu.sum() / (d * V):.6f}; measure.py's mean11: {mj['ones_per_weight']:.6f}, {mj['signed_per_weight']:.6f}")
for k in tot: log(f"  {k:8s} {tot[k]:>20,}  cut.py {cc[k]:>20,}  {'EQUAL' if tot[k] == cc[k] else 'DIFFERENT'}")
log(f"done ({time.time() - T0:.0f}s)")
