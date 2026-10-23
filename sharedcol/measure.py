"""SHAREDCOL/MEASURE.PY: THE TOP TABLE AS ONE STANDING PATTERN SHARED BY ALL WORDS PLUS EACH WORD'S OWN FRINGE.
MEASURE ONLY, WEIGHTS ONLY. 2026-10-06.

WHAT WAS FOUND TODAY (lineblanks/pairs_*.log): in Pythia's finished output word table, along a lane across the 50,304
words, |net| / whole is 98.7% (the whole lane nearly one sign); the blank (step0) sits at the bell's 0.57%.

HIS PICTURE, AS THE SESSION PUT IT (the session's words, not his sentence; nothing here is quoted from him):
  the top table is one standing pattern shared by all words plus each word's own fringe. A shared column c, met by the
  light h at the top, adds h.c to every word's pile at that position: the same amount for every word. So it can never
  change who arrives first or the top 5. Only the fringes decide; store and meet only the fringes.
  Followed forward in that picture: the standing pattern is one column of d values; it is met once a position (d
  meetings), not once per word. What is left to store and meet per word is the fringe f_w = T_w - c. This run measures,
  on the plate exactly as the machine meets it, how big the standing pattern is and what the fringes cost against the
  whole table, in lit places (signed digits) and in bits a weight at the floor.

THE TABLES, AS THE MACHINE MEETS THEM (every one [lanes][words], whole numbers, each word on its own power of two):
  gpt2         base2_gpt2.npz head.w (768 x 50,257, int16), head.e (one power for every word, 2^-12): base2.py's top
               table, the last water level's gain and centring folded in (head.b, the per-word offset, is not a column
               and is left out)
  pythia-<s>   foundation0/base2_pythia-<s>.npz head.w (finished, step143000) and head.s (blank, step0), head.e (one
               power per word, the same for both), s = 14m, 70m, 160m: pythia_base2.py's head; the gauge's gain and
               offset are their own numbers there and are not in the table
  qwen         qwen/qwen_base2.npz head.w (1,024 x 151,936, int16), head.e (one power per word): qwen_base2.py's
               gauged_slab(R, "head"), the tied table with the water level's gain folded in (what qwen_fringes2's band
               read lines up)

THE SHARED COLUMN, EXACT ON EVERY WORD'S OWN GRID
  Each lane's real values q x 2^e_w are first written on the finest grid 2^emin (exact). The shared column c is then
  put on the COARSEST word grid 2^emax, so that c is a whole number on every word's own grid: on word w's grid it is
  C_l x 2^(emax - e_w). The fringe f_w = q_w - C x 2^(emax - e_w) is then exact whole numbers on word w's own grid, the
  table comes back as f + c digit for digit (asserted), and h.c is one number, the same for every word.
    mean    C_l = each lane's sum over words / V, rounded to the coarsest grid (whole-number rounding)
    median  C_l = each lane's lower middle value over words, rounded to the coarsest grid (the second choice)
  (emax - emin is 0 for GPT-2, at most 3 for Pythia and Qwen; the rounding leaves at most 4 grains of the finest grid
  in each lane's fringe mean, reported.)

WHAT IS MEASURED (every table and every version)
  - the shared column's share of the table's summed squares: V |c|^2 / sum T^2 (real sizes); the fringe's share
    sum f^2 / sum T^2; for the median column the part of the summed squares it removes, 1 - sum f^2 / sum T^2
  - the share of words with the same sign in each lane (the larger of ups and downs over V): median lane, 10th
    percentile, lowest, lanes at 99% or more; the same for the fringe
  - |c| against a typical word column's size (the median and the rms over words of |T_w|)
  - what is left per word: |f_w| / |T_w| over the lanes (median, 10th, 90th percentile; pooled)
  - lit places: signed digits (no two lit side by side) a weight, and plain 1s a weight, T against f, on the same
    grid; the shared column's own lit places (d values) counted against the whole table
  - each word's top place (the highest place any of its lanes reaches): T against f, on its own grid
  - bits a weight at the floor on the same grid: the Shannon floor of the values coded one at a time with a histogram
    (table_bits.entropy_bits' floor, model not counted), three ways:
      pooled         one histogram for the whole table (own grids; and once more on the one finest grid)
      per lane       each lane its own histogram
      per lane+grid  each lane and each word power its own histogram

RULE 13, CHECKED BEFORE THE RUN
  13.1 no light here; weights only, as asked. The measure is the finished table itself, not anything going in.
  13.2 the tables exactly as each machine meets them (stored whole numbers and each word's power); nothing changed.
  13.3 statistics that a shuffle keeping every value must return: the shared column, its share of the summed squares,
       the same-sign share, every pooled and per-lane lit-place count and floor are functions of each lane's SET of
       values over the words. The lane shuffle (each lane's values re-dealt over the words, every lane's mean and
       median kept) MUST return them digit for digit; it can show nothing about them and is not used as evidence for
       them. What it CAN show: whether the per-word numbers (each word's fringe against its own column, each word's top
       place) come from which values sit together in one word, or from the lanes alone. The evidence that the shared
       column is there at all is the blank (step0) and the fresh draws, which do not keep the lanes' levels.
  13.4 definitions that fix the answer: (a) the mean column takes the largest share of the summed squares any shared
       column can take (the mean is where the sum of squares is least), so "share of the summed squares" is an upper
       bound for every shared column, by definition. (b) the median column leaves each lane's fringe half up and half
       down, by definition: the fringe's same-sign share with the median column is about 50% whatever the table.
       (c) per lane+grid, subtracting one number from every value only relabels the histogram: that floor is EXACTLY
       unchanged by any shared column, by definition (asserted, not reported as a finding). Per lane it moves only
       through the word powers mixing. So any saving in bits from the shared column shows up only in the pooled floor,
       and there it is the cost of a histogram that does not know each lane's level. Reported as that.
  13.5 his words are not quoted anywhere here; the picture above is labelled as the session's.
  13.6 every lane, every word, every table, both columns; nothing stopped early.
  ORDER-FREE CHECK: the lanes and the words in a random order (each word keeps its power): every number the same.

CONTROLS
  blank       Pythia's head at step0 (head.s), on the finished table's grids
  laneshuffle each lane's values re-dealt at random over the words, on the one finest grid (a word's power cannot
              follow a value that moves to another word); compared with the finished table on the same finest grid
  fresh       a fresh bell draw: each word's own spread over the lanes, no level, rounded onto that word's own grid:
              what every statistic gives by chance
  freshlane   a fresh bell draw that keeps each lane's level and spread (mean and spread over the words), rounded onto
              each word's own grid: what the lane levels alone give, with bell shapes in place of the table's own

SESSION'S PREDICTION (handed to this run, written here before it ran):
  Pythia's heads: shared column 85-95% of the summed squares, fringes save 1.5-2.5 bits a weight and 30-50% of the lit
  places; GPT-2's T and Qwen's tied table: a shared column of 5-30% of the summed squares, saving 0.05-0.3 bits a weight.

THIS RUN'S PREDICTION (worked out before the run):
  - Pythia 70m and 160m, finished: |c| 0.96 of a typical column means a share of about 0.96^2 = 0.92 if "typical" is
    the rms; columns of uneven size lower it: 85-93%. 14m: 75-93%. Same sign in the median lane 99.5-100% (a level
    3.4 spreads above the fringe's spread: 99.97% for a bell). Fringe per word about sqrt(1 - 0.92) = 0.28 of its
    column (median 0.22-0.38).
  - Pythia lit places: a value's signed digits run about one per 3 places of its size, so a fringe about 1.8 places
    smaller saves about 0.6 of about 5 lit places a weight: 8-18% fewer, not 30-50%. Plain 1s the same share or a
    little more. Each word's top place falls by 1-2.5 places.
  - Pythia bits: pooled, the whole table is a mixture of lanes standing at different levels, the fringe a mixture of
    centred lanes: a saving of about log2(level / fringe spread), 1.3-2.2 bits a weight. Per lane: within 0.05 bits of
    0 (per lane+grid: exactly 0, by definition). So the pooled saving is what a one-histogram coder pays for not
    knowing the d levels, not information in the table.
  - Pythia blank and both fresh draws without a level: share about 1/V (0.001-0.01%), same sign in the median lane
    50-51%, no saving in lit places or bits (within 0.5% and 0.02 bits). freshlane: share, same sign and pooled
    saving close to the finished table's (within 0.03 of the share, 0.3 bits); its lit-place saving within a third of
    the finished table's.
  - laneshuffle: every per-lane number identical to the finished table's on the same grid (by construction); the
    per-word fringe ratios change in their spread among words but not in their pooled value.
  - GPT-2's T: shared column 3-20% of the summed squares; same sign in the median lane 55-75%; lit places 0.5-4% fewer;
    pooled bits 0.02-0.25 a weight; per lane exactly 0 (one grid). Qwen's table: 2-25%, same sign 55-80%, lit places
    0.5-5%, pooled bits 0.01-0.3, per lane within 0.02 bits.
  - order-free: every number identical.

ADDED AFTER THE FIRST RUN (pythia-14m, the smallest, run first as the code's first test; its log was overwritten by
the rerun with these additions). What it showed: the blank's column is nothing (0.0015% of the squares), yet taking it
away made the lit places 1.85% MORE and the pooled floor 0.32 bits a weight WORSE. The reason, in the table's own
terms: Pythia's plates were read once from fp16 (pythia_convert.py), so every stored value carries at most 11
significant places, and every value above 2^11 grains has its low places dark. A shared column that is not written the
same way lights those low places when it is taken away, and lanes shifted by different amounts fill each other's gaps
in a pooled histogram. So two things are added, nothing else changed:
  - each value's span: the places from its lowest lit place to its highest (plain binary), the mean over nonzero
    values, and the share of values spanning 11 places or fewer
  - a third column, mean11: the mean column written with 11 significant places (as fp16 writes a value), on the
    coarsest grid. One fixed choice, set by the table's own writing, not searched: where a value is at least the
    column's size, the column's lowest lit place is no lower than the value's, so the fringe keeps the value's dark
    low places.
  PREDICTIONS FOR THE ADDITIONS (written before any table was run with them; pythia-14m's first numbers above seen):
  - Pythia tables, finished and blank: 100% of stored values span 11 places or fewer. Fringe from the mean column:
    the share spanning 11 or fewer falls to 30-70%. Fringe from mean11: 85-100%.
  - Pythia 70m and 160m: the mean column's lit-place saving is eaten by the lit low places: -5% to +8%; mean11's
    saving 12-25%. Pooled floor: mean column 0.8-1.9 bits a weight, mean11 1.3-2.3.
  - Pythia 14m (shared column only 32.7%): mean11 lit-place saving 3-8%.
  - GPT-2 (from fp32, centred across lanes: no such writing): its values' spans are their top places (share at 11 or
    fewer 5-30%), and mean11 gives the same numbers as the mean column within 0.3%. Qwen (gain x a 16-bit writing,
    rounded to 16 places): no 11-place writing; mean11 within 0.5% of the mean column.
  - fresh and freshlane draws are not written in fp16: their spans fill their top places; their mean11 = their mean.

    python3 sharedcol/measure.py table=pythia-70m      (table = gpt2 | pythia-14m | pythia-70m | pythia-160m | qwen)
  writes sharedcol/measure_<table>.log and .json
"""
import os, sys, json, time, resource, numpy as np

HERE = "/Users/abundancemachine/streaming-machine"; OUT = os.path.join(HERE, "sharedcol")
A_ = dict(a.split("=", 1) for a in sys.argv[1:]); TABLE = A_.get("table", "pythia-70m"); NL = int(A_.get("block", 32))
LOGF = open(os.path.join(OUT, f"measure_{TABLE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
T0 = time.time()

def naf(X): a = np.abs(X); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)      # signed digits lit
def ones(X): return np.bitwise_count(np.abs(X)).astype(np.int64)                         # plain 1s
def rdiv(a, b): return (2 * a + b) // (2 * b)                                             # whole-number rounding, b > 0
def hbits(c):                                                                             # Shannon floor in bits, total
    c = c[c > 0].astype(np.float64); n = c.sum(); return float(n * np.log2(n) - (c * np.log2(c)).sum())
def sig11(x):                                                                             # each value written with 11 significant places
    x = np.asarray(x, np.int64); out = x.copy()
    for i, v in enumerate(x):
        b = int(abs(int(v))).bit_length()
        if b > 11: dr = b - 11; out[i] = rdiv(int(v), 1 << dr) << dr
    return out
def blen(x):                                                                              # top place of |x| (exact below 2^53)
    x = np.asarray(x, np.int64); assert x.max(initial=0) < 1 << 53; return np.where(x > 0, np.frexp(x.astype(np.float64))[1], 0).astype(np.int64)

# ---------------- the tables ----------------
def load():
    if TABLE == "gpt2":
        M = np.load(os.path.join(HERE, "base2_gpt2.npz")); return dict(finished=M["head.w"]), M["head.e"].astype(np.int64)
    if TABLE.startswith("pythia-"):
        Z = np.load(os.path.join(HERE, "foundation0", f"base2_{TABLE}.npz"))
        return dict(finished=Z["head.w"], blank=Z["head.s"]), Z["head.e"].astype(np.int64)
    if TABLE == "qwen":
        Z = np.load(os.path.join(HERE, "qwen", "qwen_base2.npz")); return dict(finished=Z["head.w"]), Z["head.e"].astype(np.int64)
    raise SystemExit(f"unknown table {TABLE}")

def blocks_of(Q):                                     # the stored table, lane block by lane block, as int64
    def gen():
        for l0 in range(0, Q.shape[0], NL): yield l0, np.asarray(Q[l0:l0 + NL]).astype(np.int64)
    return gen

# ---------------- the measurement ----------------
KEYS = ("T", "mean", "median", "mean11"); FKEYS = KEYS[1:]
def measure(blocks, d, V, e, name):
    """blocks(): yields (l0, A) with A the lanes l0.. on each word's own grid (int64); e: each word's power"""
    t0 = time.time(); e = np.asarray(e, np.int64); emin, emax = int(e.min()), int(e.max()); k = emax - emin
    kw = e - emin; sw = emax - e; G = k + 1
    # pass 1: each lane's sum, squares, lower middle value, signs (real sizes on the finest grid)
    S = np.zeros(d, np.int64); SQ = np.zeros(d); MED = np.zeros(d, np.int64); POS = np.zeros(d, np.int64); NEG = np.zeros(d, np.int64)
    WS = np.zeros(V); WQ = np.zeros(V); maxA = 0; maxB = 0
    for l0, A in blocks():
        n = len(A); B = A << kw[None, :]
        assert np.abs(B).max() < 1 << 40
        S[l0:l0 + n] = B.sum(1); Bf = B.astype(np.float64); SQ[l0:l0 + n] = (Bf * Bf).sum(1)
        MED[l0:l0 + n] = np.partition(B, (V - 1) // 2, axis=1)[:, (V - 1) // 2]
        POS[l0:l0 + n] = (A > 0).sum(1); NEG[l0:l0 + n] = (A < 0).sum(1)
        WS += Bf.sum(0); WQ += (Bf * Bf).sum(0); maxA = max(maxA, int(np.abs(A).max())); maxB = max(maxB, int(np.abs(B).max()))
    C = rdiv(S, V << k); CM = rdiv(MED, 1 << k)                                           # on the coarsest grid
    C11 = sig11(C); cols = dict(mean=C, median=CM, mean11=C11)
    cmax = int(np.abs(np.concatenate([C, CM, C11])).max())
    offA = maxA + cmax * (1 << k) + 1; offB = maxB + cmax * (1 << k) + 1
    R = dict(name=name, d=d, V=V, word_powers={int(emin + g): int((kw == g).sum()) for g in range(G)})
    # pass 2: the fringes
    acc = {t: dict(hA=np.zeros(2 * offA + 1, np.int64), hB=np.zeros(2 * offB + 1, np.int64), naf=0, ones=0, lane_bits=0.0,
                   lg_bits=0.0, pos=np.zeros(d, np.int64), neg=np.zeros(d, np.int64), sq=np.zeros(d), wsq=np.zeros(V),
                   wor=np.zeros(V, np.int64), lane_sum=np.zeros(d, np.int64), maxabs=0, span=0, nz=0, span11=0) for t in KEYS}
    lg_equal = True
    for l0, A in blocks():
        n = len(A); B = A << kw[None, :]
        vers = {"T": A}
        for t, col in cols.items():
            F = A - (col[l0:l0 + n, None] << sw[None, :]); vers[t] = F
            assert np.array_equal((F << kw[None, :]) + (col[l0:l0 + n, None] << k), B), "fringe + shared column is not the table"
        for t, X in vers.items():
            a = acc[t]; XB = X << kw[None, :]; XBf = XB.astype(np.float64)
            a["hA"] += np.bincount((X + offA).ravel(), minlength=2 * offA + 1)
            a["hB"] += np.bincount((XB + offB).ravel(), minlength=2 * offB + 1)
            a["naf"] += int(naf(X).sum()); a["ones"] += int(ones(X).sum()); a["maxabs"] = max(a["maxabs"], int(np.abs(X).max()))
            a["pos"][l0:l0 + n] = (X > 0).sum(1); a["neg"][l0:l0 + n] = (X < 0).sum(1)
            a["sq"][l0:l0 + n] = (XBf * XBf).sum(1); a["wsq"] += (XBf * XBf).sum(0); a["wor"] |= np.bitwise_or.reduce(np.abs(X), axis=0)
            a["lane_sum"][l0:l0 + n] = XB.sum(1)
            ax = np.abs(X); nzm = ax > 0; sp = blen(ax) - (blen(ax & -ax) - 1)                  # span: lowest lit place to highest
            a["span"] += int(sp[nzm].sum()); a["nz"] += int(nzm.sum()); a["span11"] += int((sp[nzm] <= 11).sum())
            for i in range(n):
                row = X[i] + offA
                a["lane_bits"] += hbits(np.bincount(row))
                hg = np.bincount(row + kw * (2 * offA + 1), minlength=G * (2 * offA + 1)).reshape(G, 2 * offA + 1)
                a["lg_bits"] += sum(hbits(hg[g]) for g in range(G))
                if t == "T": a.setdefault("_hg", {})[i] = [np.sort(hg[g][hg[g] > 0]) for g in range(G)]
                else: lg_equal &= all(np.array_equal(np.sort(hg[g][hg[g] > 0]), acc["T"]["_hg"][i][g]) for g in range(G))
        acc["T"].pop("_hg", None)
    N = d * V; T = acc["T"]; sumT = float(T["sq"].sum())
    # the shared column and its share
    Creal = C.astype(np.float64) * (1 << k); CMreal = CM.astype(np.float64) * (1 << k)
    wnorm = np.sqrt(T["wsq"]); cnorm = float(np.sqrt((Creal ** 2).sum()))
    same = np.maximum(POS, NEG) / V
    def q(x, p): return float(np.percentile(x, p))
    R["shared"] = dict(
        mean_share_of_squares=float(V * (Creal ** 2).sum() / sumT), mean_fringe_share=float(acc["mean"]["sq"].sum() / sumT),
        median_removed_share=float(1 - acc["median"]["sq"].sum() / sumT), median_column_share=float(V * (CMreal ** 2).sum() / sumT),
        mean11_removed_share=float(1 - acc["mean11"]["sq"].sum() / sumT),
        col_vs_median_word=cnorm / float(np.median(wnorm)), col_vs_rms_word=cnorm / float(np.sqrt((wnorm ** 2).mean())),
        fringe_lane_mean_left_max_grains=float(np.abs(acc["mean"]["lane_sum"] / V).max()),
        col_lit_places=int(naf(C).sum()), col_lit_per_weight=float(naf(C).sum() / N))
    R["same_sign"] = {t: dict(median_lane=q(np.maximum(acc[t]["pos"], acc[t]["neg"]) / V, 50), p10=q(np.maximum(acc[t]["pos"], acc[t]["neg"]) / V, 10),
                              lowest=float((np.maximum(acc[t]["pos"], acc[t]["neg"]) / V).min()),
                              lanes_99=float(((np.maximum(acc[t]["pos"], acc[t]["neg"]) / V) >= 0.99).mean()),
                              lanes_90=float(((np.maximum(acc[t]["pos"], acc[t]["neg"]) / V) >= 0.90).mean())) for t in KEYS}
    assert np.array_equal(T["pos"], POS) and np.array_equal(T["neg"], NEG)
    ratio = {t: np.sqrt(acc[t]["wsq"] / np.maximum(T["wsq"], 1e-300)) for t in FKEYS}
    R["word_fringe"] = {t: dict(median=q(ratio[t], 50), p10=q(ratio[t], 10), p90=q(ratio[t], 90),
                                pooled=float(np.sqrt(acc[t]["sq"].sum() / sumT))) for t in FKEYS}
    R["lit"] = {t: dict(signed_per_weight=acc[t]["naf"] / N, ones_per_weight=acc[t]["ones"] / N) for t in KEYS}
    for t in FKEYS:
        R["lit"][t]["signed_saving"] = 1 - acc[t]["naf"] / T["naf"]; R["lit"][t]["ones_saving"] = 1 - acc[t]["ones"] / T["ones"]
        R["lit"][t]["signed_saving_with_column"] = 1 - (acc[t]["naf"] + int(naf(cols[t]).sum())) / T["naf"]
    R["span"] = {t: dict(mean=acc[t]["span"] / max(acc[t]["nz"], 1), share_11_or_fewer=acc[t]["span11"] / max(acc[t]["nz"], 1),
                         nonzero=acc[t]["nz"] / N) for t in KEYS}
    R["top_place"] = {t: dict(mean_per_word=float(blen(acc[t]["wor"]).mean()), max=int(blen(np.array([acc[t]["maxabs"]]))[0])) for t in KEYS}
    for t in FKEYS: R["top_place"][t]["places_saved_per_word"] = R["top_place"]["T"]["mean_per_word"] - R["top_place"][t]["mean_per_word"]
    R["bits"] = {t: dict(pooled=hbits(acc[t]["hA"]) / N, pooled_finest=hbits(acc[t]["hB"]) / N, per_lane=acc[t]["lane_bits"] / N,
                         per_lane_grid=acc[t]["lg_bits"] / N) for t in KEYS}
    for t in FKEYS:
        R["bits"][t].update({f"saving_{w}": R["bits"]["T"][w] - R["bits"][t][w] for w in ("pooled", "pooled_finest", "per_lane", "per_lane_grid")})
    R["per_lane_grid_floor_unchanged_exactly"] = bool(lg_equal)
    R["lane_stats"] = dict(mu=(S / V).tolist(), sd=np.sqrt(np.maximum(SQ / V - (S / V) ** 2, 0)).tolist())
    R["word_stats"] = dict(sd=np.sqrt(np.maximum(WQ / d - (WS / d) ** 2, 0)).tolist())
    R["secs"] = round(time.time() - t0, 1)
    return R

def show(R):
    s, ss, wf, li, tp, b, sp = R["shared"], R["same_sign"], R["word_fringe"], R["lit"], R["top_place"], R["bits"], R["span"]
    log(f"  [{R['name']}]  {R['d']} lanes x {R['V']} words; word powers {R['word_powers']}  ({R['secs']}s, {rss()} MB)")
    log(f"    shared column (mean):   {s['mean_share_of_squares'] * 100:8.4f}% of the summed squares; fringe {s['mean_fringe_share'] * 100:.4f}%;  |c| / median word column {s['col_vs_median_word']:.4f}, / rms word column {s['col_vs_rms_word']:.4f}; fringe lane mean left at most {s['fringe_lane_mean_left_max_grains']:.2f} grains")
    log(f"    shared column (median): removes {s['median_removed_share'] * 100:8.4f}% of the summed squares (its own share {s['median_column_share'] * 100:.4f}%);  mean11 removes {s['mean11_removed_share'] * 100:.4f}%")
    log(f"    same sign per lane, table:  median lane {ss['T']['median_lane'] * 100:.3f}%  p10 {ss['T']['p10'] * 100:.3f}%  lowest {ss['T']['lowest'] * 100:.3f}%  lanes >= 99%: {ss['T']['lanes_99'] * 100:.1f}%  >= 90%: {ss['T']['lanes_90'] * 100:.1f}%")
    log(f"    same sign per lane, fringe (median lane): mean column {ss['mean']['median_lane'] * 100:.3f}%, median column {ss['median']['median_lane'] * 100:.3f}% (50% by definition), mean11 {ss['mean11']['median_lane'] * 100:.3f}%")
    for t in FKEYS:
        log(f"    fringe ({t:6s}) / its word column: median {wf[t]['median']:.4f}  p10 {wf[t]['p10']:.4f}  p90 {wf[t]['p90']:.4f}  pooled {wf[t]['pooled']:.4f}")
    log(f"    lit places a weight, signed digits: table {li['T']['signed_per_weight']:.4f}  " + "  ".join(
        f"{t} {li[t]['signed_per_weight']:.4f} ({li[t]['signed_saving'] * 100:+.2f}% fewer; {li[t]['signed_saving_with_column'] * 100:+.2f}% with the column)" for t in FKEYS))
    log(f"    lit places a weight, plain 1s:     table {li['T']['ones_per_weight']:.4f}  " + "  ".join(f"{t} {li[t]['ones_per_weight']:.4f} ({li[t]['ones_saving'] * 100:+.2f}%)" for t in FKEYS))
    log(f"    span (lowest lit place to highest): table {sp['T']['mean']:.3f} places, {sp['T']['share_11_or_fewer'] * 100:.2f}% at 11 or fewer;  " + "  ".join(
        f"{t} {sp[t]['mean']:.3f}, {sp[t]['share_11_or_fewer'] * 100:.2f}%" for t in FKEYS))
    log(f"    each word's top place: table {tp['T']['mean_per_word']:.3f}  " + "  ".join(f"{t} {tp[t]['mean_per_word']:.3f} ({tp[t]['places_saved_per_word']:+.3f})" for t in FKEYS)
        + f";  highest place: " + " / ".join(str(tp[t]['max']) for t in KEYS))
    for w, lab in (("pooled", "pooled, own grids"), ("pooled_finest", "pooled, finest grid"), ("per_lane", "per lane"), ("per_lane_grid", "per lane+grid")):
        log(f"    bits a weight, {lab:20s}: table {b['T'][w]:.4f}  " + "  ".join(f"{t} {b[t][w]:.4f} (saves {b[t]['saving_' + w]:+.4f})" for t in FKEYS))
    log(f"    per lane+grid floor unchanged by the column, histogram for histogram: {R['per_lane_grid_floor_unchanged_exactly']} (by definition, rule 13.4c)")

def strip(R): return {k: v for k, v in R.items() if k not in ("lane_stats", "word_stats")}

# ---------------- run ----------------
log(f"SHAREDCOL MEASURE: {TABLE}  (block {NL} lanes)")
tabs, E = load(); Q = tabs["finished"]; d, V = Q.shape; assert len(E) == V
log(f"  table {Q.shape} {Q.dtype}; word powers 2^{E.min()}..2^{E.max()}  ({rss()} MB)")
res = {}
res["finished"] = R = measure(blocks_of(Q), d, V, E, "finished, each word on its own grid"); show(R)

kwF = E - E.min()
def common_blocks(Qx, perm_rng=None):                 # the table on the one finest grid; lanes re-dealt if perm_rng
    def gen():
        rng = None if perm_rng is None else np.random.default_rng(perm_rng)
        for l0 in range(0, d, NL):
            B = np.asarray(Qx[l0:l0 + NL]).astype(np.int64) << kwF[None, :]
            if rng is not None: B = np.stack([row[rng.permutation(V)] for row in B])
            yield l0, B
    return gen
Ecom = np.full(V, int(E.min()), np.int64)
res["finished_finest"] = R = measure(common_blocks(Q), d, V, Ecom, "finished, all on the finest grid"); show(R)
res["laneshuffle"] = R = measure(common_blocks(Q, [7, 1]), d, V, Ecom, "CONTROL laneshuffle: each lane re-dealt over the words, finest grid"); show(R)
same_keys = [("shared", None), ("same_sign", None), ("lit", None), ("bits", None), ("span", None)]
def flat(x, p=""):
    if isinstance(x, dict):
        out = {}
        for kk, vv in x.items(): out.update(flat(vv, f"{p}{kk}."))
        return out
    return {p[:-1]: x}
fa, fb = flat({k: res["finished_finest"][k] for k, _ in same_keys}), flat({k: res["laneshuffle"][k] for k, _ in same_keys})
diff = [kk for kk in fa if kk != "shared.col_vs_median_word" and not np.isclose(fa[kk], fb[kk], rtol=1e-9, atol=1e-12)]
log(f"  laneshuffle against the finished table on the same grid: per-lane numbers differing (must be none, rule 13.3): {diff if diff else 'none'} (of {len(fa) - 1})")
log(f"    per-word: |c| / median word column: finished {fa['shared.col_vs_median_word']:.4f} | laneshuffle {fb['shared.col_vs_median_word']:.4f}")
for t in FKEYS:
    a, b2 = res["finished_finest"]["word_fringe"][t], res["laneshuffle"]["word_fringe"][t]
    log(f"    per-word fringe ratio ({t}): finished median {a['median']:.4f} p10 {a['p10']:.4f} p90 {a['p90']:.4f} | laneshuffle median {b2['median']:.4f} p10 {b2['p10']:.4f} p90 {b2['p90']:.4f}")
a, b2 = res["finished_finest"]["top_place"], res["laneshuffle"]["top_place"]
log(f"    each word's top place on the finest grid: finished T {a['T']['mean_per_word']:.3f} fringe {a['mean']['mean_per_word']:.3f} | laneshuffle T {b2['T']['mean_per_word']:.3f} fringe {b2['mean']['mean_per_word']:.3f}")

# order-free: lanes and words in a random order, each word keeping its power
rng = np.random.default_rng([11, 3]); lp, wp = rng.permutation(d), rng.permutation(V)
def perm_blocks():
    for l0 in range(0, d, NL): yield l0, np.asarray(Q[lp[l0:l0 + NL]]).astype(np.int64)[:, wp]
res["orderfree"] = R = measure(perm_blocks, d, V, E[wp], "ORDER-FREE: lanes and words in a random order"); show(R)
fa, fb = flat(strip(res["finished"])), flat(strip(res["orderfree"]))
diff = [kk for kk in fa if kk not in ("name", "secs") and not (fa[kk] == fb[kk] if not isinstance(fa[kk], float) else np.isclose(fa[kk], fb[kk], rtol=1e-9, atol=1e-12))]
log(f"  ORDER-FREE CHECK against the finished table: numbers differing: {diff if diff else 'none'}")

if "blank" in tabs:
    res["blank"] = R = measure(blocks_of(tabs["blank"]), d, V, E, "CONTROL blank: step0, on the finished table's grids"); show(R)

LS, WSD = res["finished"]["lane_stats"], res["finished"]["word_stats"]
mu, sdl, sdw = np.array(LS["mu"]), np.array(LS["sd"]), np.array(WSD["sd"])
def fresh_blocks(kind, seed):
    def gen():
        rng = np.random.default_rng(seed)
        for l0 in range(0, d, NL):
            n = min(NL, d - l0); z = rng.normal(0.0, 1.0, (n, V))
            v = z * sdw[None, :] if kind == "fresh" else mu[l0:l0 + n, None] + z * sdl[l0:l0 + n, None]   # real sizes, finest grid
            yield l0, np.rint(v / np.exp2(kwF)[None, :]).astype(np.int64)                                    # onto each word's own grid
    return gen
res["fresh"] = R = measure(fresh_blocks("fresh", [5, 1]), d, V, E, "CONTROL fresh: each word's own spread, no level, each word's own grid"); show(R)
res["freshlane"] = R = measure(fresh_blocks("freshlane", [5, 2]), d, V, E, "CONTROL freshlane: each lane's own level and spread as a bell, each word's own grid"); show(R)

json.dump({k: strip(v) for k, v in res.items()}, open(os.path.join(OUT, f"measure_{TABLE}.json"), "w"), indent=1)
log(f"done ({time.time() - T0:.0f}s, max memory {rss()} MB)")
