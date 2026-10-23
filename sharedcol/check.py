"""SHAREDCOL/CHECK.PY: A SECOND CODE PATH FOR MEASURE.PY'S MAIN NUMBERS (same session, NOT an independent rebuild), and
two short looks the main run raised. 2026-10-06.

SECOND PATH (each number recomputed another way and compared with measure_<table>.json, "finished" on own grids):
  - the mean column: each lane's sum as Python whole numbers over 4,096-word pieces, rounded to the coarsest grid by
    Python's own divmod; compared lane for lane with measure.py's (rebuilt here from the table, not read from it)
  - the shared column's share of the summed squares and the fringe's share: exact Python whole numbers throughout
  - the same-sign share per lane: from np.sign against each lane's own majority sign
  - the fringe, rebuilt in Python whole numbers on 8 random lanes x 2,000 random words: q - C x 2^(emax - e_w)
  - lit places (signed digits): the digit-by-digit writing (odd: digit 2 - (x mod 4), else 0; halve), on 300,000
    random values of the table and of the fringe, against popcount(a ^ 3a) on the same values; then the totals
  - the pooled floor on own grids, table and fringe: np.unique counts instead of a bincount
SHORT LOOKS (predictions written before this script ran; the main run's numbers had been seen):
  (a) where the standing pattern lives: the share of the column's summed squares in its largest 1, 3, 10 lanes and
      its largest 1% and 10% of lanes (a description of the column, nothing chosen by it)
      prediction: GPT-2: concentrated, the largest 1% of lanes (8) hold 70-95% and the largest 3 lanes 50-90% (only
      6% of its lanes are one-signed, yet the column is 82.6% of the squares). Pythia 70m / 160m: spread over many
      lanes (75-79% of lanes are 99% one-signed): the largest 1% of lanes hold 3-15%. Qwen: 21.1% share with the
      median lane at 57.8%: the largest 1% of lanes hold 20-60%.
  (b) words the standing pattern does not fit: words whose fringe is larger than their own column (ratio > 1), and
      fringe values that need more than 16 places on their word's own grid (|f| >= 2^15)
      prediction: Pythia 0.5-2% of words over 1 (the words on the finer grids, whose own columns are small: 329 of
      50,304 in 160m); GPT-2 at most 0.5%; Qwen 1-5%. Over 16 places: none for GPT-2 (its highest fringe place is 14);
      fewer than 1% of words for Pythia 160m and Qwen (highest places 17).
    python3 sharedcol/check.py table=pythia-70m
"""
import os, sys, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = os.path.join(HERE, "sharedcol")
A_ = dict(a.split("=", 1) for a in sys.argv[1:]); TABLE = A_.get("table", "pythia-70m")
LOGF = open(os.path.join(OUT, f"check_{TABLE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
t0 = time.time()
if TABLE == "gpt2": Z = np.load(os.path.join(HERE, "base2_gpt2.npz"))
elif TABLE == "qwen": Z = np.load(os.path.join(HERE, "qwen", "qwen_base2.npz"))
else: Z = np.load(os.path.join(HERE, "foundation0", f"base2_{TABLE}.npz"))
Q = Z["head.w"]; E = Z["head.e"].astype(np.int64); d, V = Q.shape
J = json.load(open(os.path.join(OUT, f"measure_{TABLE}.json")))["finished"]
emin, emax = int(E.min()), int(E.max()); k = emax - emin; kw = E - emin; sw = emax - E
log(f"CHECK {TABLE}: {d} lanes x {V} words, powers 2^{emin}..2^{emax}")

# the mean column, exact
C = []; sumT2 = 0; same = []
for l in range(d):
    q = np.asarray(Q[l]).astype(np.int64); b = q << kw
    s = sum(int(b[i:i + 4096].sum()) for i in range(0, V, 4096))
    num, den = 2 * s + V * (1 << k), 2 * V * (1 << k); C.append(num // den)               # round half up, as measure.py
    sumT2 += sum(int((b[i:i + 4096] * b[i:i + 4096]).sum()) for i in range(0, V, 4096))
    sg = np.sign(q); maj = 1 if (sg > 0).sum() >= (sg < 0).sum() else -1; same.append(float((sg == maj).sum()) / V)
C = np.array(C, dtype=np.int64)
colsq = V * sum(int(c) * int(c) for c in C) * (1 << (2 * k))
fr2 = 0; nafT = nafF = 0; VF = np.empty((d, V), np.int32); wsqT = np.zeros(V); wsqF = np.zeros(V); wmaxF = np.zeros(V, np.int64)
for l in range(d):
    q = np.asarray(Q[l]).astype(np.int64); f = q - (int(C[l]) << sw); fb = f << kw
    fr2 += sum(int((fb[i:i + 4096] * fb[i:i + 4096]).sum()) for i in range(0, V, 4096))
    a = np.abs(q); nafT += int(np.bitwise_count(a ^ (3 * a)).sum()); a = np.abs(f); nafF += int(np.bitwise_count(a ^ (3 * a)).sum())
    assert np.abs(f).max() < 1 << 31; VF[l] = f
    bq = (q << kw).astype(np.float64); wsqT += bq * bq; wsqF += fb.astype(np.float64) ** 2; wmaxF = np.maximum(wmaxF, np.abs(f))
log(f"  share of the summed squares (exact whole numbers): column {colsq / sumT2 * 100:.4f}% (measure {J['shared']['mean_share_of_squares'] * 100:.4f}%), "
    f"fringe {fr2 / sumT2 * 100:.4f}% (measure {J['shared']['mean_fringe_share'] * 100:.4f}%)")
sm = np.array(same)
log(f"  same sign per lane: median lane {np.median(sm) * 100:.3f}% (measure {J['same_sign']['T']['median_lane'] * 100:.3f}%), lanes >= 99% {np.mean(sm >= 0.99) * 100:.1f}% (measure {J['same_sign']['T']['lanes_99'] * 100:.1f}%)")
N = d * V
log(f"  lit places a weight (signed digits): table {nafT / N:.4f} (measure {J['lit']['T']['signed_per_weight']:.4f}), fringe {nafF / N:.4f} (measure {J['lit']['mean']['signed_per_weight']:.4f})")

# the fringe rebuilt in Python whole numbers on a sample
rng = np.random.default_rng([3, 1]); bad = 0
for l in rng.choice(d, 8, replace=False):
    for w in rng.choice(V, 2000, replace=False):
        f = int(Q[l, w]) - int(C[l]) * 2 ** int(emax - E[w])
        bad += f != int(VF[l, w]); bad += (f * 2 ** int(E[w] - emin) + int(C[l]) * 2 ** k) != int(Q[l, w]) * 2 ** int(E[w] - emin)
log(f"  fringe rebuilt in Python whole numbers, 16,000 values: {bad} differ; table = fringe + column on the finest grid at all of them")

# signed digits digit by digit
def naf_slow(x):
    x = abs(int(x)); n = 0
    while x:
        if x & 1: dg = 2 - (x & 3); x -= dg; n += 1
        x >>= 1
    return n
idx = rng.choice(N, 300000, replace=False)
for nm, X in (("table", Q), ("fringe", VF)):
    xs = X.reshape(-1)[idx].astype(np.int64); a = np.abs(xs); fast = np.bitwise_count(a ^ (3 * a)); slow = np.array([naf_slow(v) for v in xs])
    log(f"  signed digits on 300,000 {nm} values: digit-by-digit {slow.sum():,} vs popcount(a ^ 3a) {int(fast.sum()):,}; values differing {int((slow != fast).sum())}")

# the pooled floor by np.unique
def H(X):
    _, c = np.unique(X, return_counts=True); c = c.astype(np.float64); n = c.sum(); return (n * np.log2(n) - (c * np.log2(c)).sum()) / n
hT, hF = H(Q), H(VF)
log(f"  pooled floor on own grids (np.unique): table {hT:.4f} (measure {J['bits']['T']['pooled']:.4f}), fringe {hF:.4f} (measure {J['bits']['mean']['pooled']:.4f}), saves {hT - hF:+.4f}")
del VF

# (a) where the standing pattern lives
c2 = (C.astype(np.float64) * (1 << k)) ** 2; o = np.sort(c2)[::-1]; tot = o.sum()
parts = {f"largest {n}": o[:n].sum() / tot for n in (1, 3, 10)}
parts.update({f"largest 1% ({max(1, d // 100)})": o[:max(1, d // 100)].sum() / tot, f"largest 10% ({d // 10})": o[:d // 10].sum() / tot})
log("  (a) the column's summed squares in its largest lanes: " + ", ".join(f"{kk} {v * 100:.2f}%" for kk, v in parts.items()))
top = np.argsort(c2)[::-1][:5]
log("      the 5 largest lanes: " + ", ".join(f"lane {int(l)} ({c2[l] / tot * 100:.2f}%)" for l in top))
# (b) words the standing pattern does not fit
r = np.sqrt(wsqF / np.maximum(wsqT, 1e-300))
log(f"  (b) words whose fringe is larger than their own column: {int((r > 1).sum())} of {V} ({(r > 1).mean() * 100:.3f}%); "
    f"words with a fringe value needing more than 16 places (|f| >= 2^15): {int((wmaxF >= 1 << 15).sum())} ({(wmaxF >= 1 << 15).mean() * 100:.3f}%)")
for g in range(k + 1):
    m = kw == g
    if m.any(): log(f"      words on 2^{emin + g}: {int(m.sum())}, their fringe / column median {np.median(r[m]):.3f}, over 1: {int((r[m] > 1).sum())}")
json.dump(dict(colshare=colsq / sumT2, fringeshare=fr2 / sumT2, same_median=float(np.median(sm)), naf_T=nafT / N, naf_F=nafF / N,
               pooled_T=hT, pooled_F=hF, concentration=parts, top_lanes=[int(x) for x in top], words_over1=int((r > 1).sum()),
               words_over16=int((wmaxF >= 1 << 15).sum())), open(os.path.join(OUT, f"check_{TABLE}.json"), "w"), indent=1)
log(f"done ({time.time() - t0:.0f}s)")
