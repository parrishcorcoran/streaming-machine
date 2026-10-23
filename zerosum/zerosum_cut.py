"""THE EXACT CUT HIS SENTENCE POINTS TO -- part 1: where the machine pours the level off, and at what grain.

HIS WORDS (2026-10-06): "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2
/ A 10 that becomes a 8 is -2"

THE SESSION'S READING (not his): every read of the stream goes through a water level that pours the stream's level
(its average over the lanes) off first, so the level any write adds never reaches anything; subtracting from each
write line a constant across the stream's lanes would then change no read at all.

WHAT THE CODE DOES (read from base2.py and foundation0/pythia_base2.py before this script was written):
  GPT-2 (base2.py), every read (qkv.L, fc.L, the top):
    1. to16(R): the position's stream is cut to 16 places at its own power of two, set by the top 1 of its largest
       lane (OR of the magnitudes). Each lane is rounded at that grain, 2^sh stream units.
    2. gauge_log: S1 = sum of the cut light, S2 = sum of squares, V = 768 S2 - S1^2 (+ eps). In whole numbers
       768 (S2 + 2dS1 + 768d^2) - (S1 + 768d)^2 = 768 S2 - S1^2: V does not see a whole constant d on every lane.
    3. product(x16, w): the cut light, level and all, meets the plate. The level is poured off only because the
       plate had the centring folded in (W' = C diag(g) W) -- and the folded plate was then ROUNDED per column.
       A whole constant d on the cut light adds d x (the column's whole-number sum) to each pile.
  Pythia (pythia_base2.py), every read (ln1.L, ln2.L, lnf):
    1. the same cut, to16(R), before anything else;
    2. c = d x16 - S1 in whole numbers: the level of the CUT light poured off exactly; V = d S2 - S1^2 the same.
  Writes in both: out_j = round(pile_j x 2^(we_j + sh_w)) + b_j, each lane rounded to the stream's grain 2^-16 on
  its own; word rows and position rows and biases are added at the stream's grain directly.
  So a constant on a write line reaches a read through three roundings:
    (i)   the write's own per-lane rounding to 2^-16 (weight rows only): the row's push at a position is
          K x 2^(G + sh_w) stream units for a constant k x 2^G (real) with K = sum of the cut light x k; it is the
          same whole number on every lane only if G >= -sh_w at every position;
    (ii)  the read's 16-place cut, made BEFORE the level is poured off: a stream level L passes it as a whole
          constant on every lane only if L is a whole multiple of 2^sh_r at that position AND the top 1 of the
          largest lane does not move;
    (iii) GPT-2 only: the plate's column sums. If any column of qkv / fc / the top does not sum to exactly 0, a
          whole constant on the cut light still changes that pile.

THIS SCRIPT (no plate is changed here; it measures the plain machine):
  A. GPT-2: the column sums of every read plate (48 + the top), whole numbers at the column's power.
  B. The plain machine on the 30 refs texts (GPT-2: ids + ref, 648 positions; Pythia: ids + ref[:15], the shape
     pythia_base2.py runs), keeping at every read: the cut's power sh_r, the cut light's sum S1, the stream's sum;
     at every write: the write's own cut power sh_w and the sum of its output over the lanes.
  C. The leak test (iii): at every GPT-2 read, the cut light + 1 on every lane (V unchanged, asserted): how many of
     the read's outputs (after the gauge) change. Pythia: the same test through its own gauge: must change none.
  D. For every write line, the grain at which its constant would pass (i) and (ii) on these texts, and how much of
     its level a constant at that grain removes.
  E. The water level's own work on the level (S1's adds, S1's square, the subtract), per token, in the unit of
     combined/machine.py (photons): against GPT-2's plain 3,666,596,259 a token.

PREDICTIONS (written before the first run):
  A. GPT-2's read plates' column sums are rounding residue: zero in about 5% of columns (the sum of 768 errors each
     within +-1/2 has SD 8; P(exactly 0) = 1/(8 sqrt(2 pi)) = 5%), RMS about 8 units, at every plate and the top.
     So GPT-2's reads do NOT pour the level off exactly: the level leaks into every pile as d x colsum.
  B. sh_r (GPT-2): layer 0 about 2-3; deeper 8-13; the first position the highest (its big lanes, 13-14).
     sh_w: attention-out 1-6, MLP-out 2-8. Pythia: 14m 4-8, 70m and 160m 6-14.
     S1 at the reads: zero almost never (under 1% of reads).
  C. GPT-2: one unit on every lane of the cut light changes about 0.01% of read outputs (colsum ~8 against a pile
     of about 2^30 and an output grain of about 2^17 pile units); far from none. Pythia: exactly none.
  D. At the grain that passes (i) and (ii) on these texts, under 1% of the weight rows' level can be removed (the
     row levels are 10-1000 times finer than 2^-sh_w); word rows under 5%; position rows and biases a few lines.
     For GPT-2, (iii) leaves no grain at all: nothing nonzero is invisible digit for digit.
  E. The level's work: about 25 x 768 x 7 photons of adds plus small squares: about 0.004% of plain (session's
     prediction: 0.001%) -- but it can only be removed where S1 is exactly 0 after the cut, i.e. almost nowhere.
    python3 zerosum/zerosum_cut.py machine=gpt2|14m|70m|160m   -> zerosum/zerosum_cut_<machine>.json / .log
"""
import sys, os, json, time, resource, hashlib, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/zerosum"
A = dict(a.split("=") for a in sys.argv[1:]); MACH = A.get("machine", "gpt2")
t0 = time.time(); LOG = open(f"{OUT}/zerosum_cut_{MACH}.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(X): return np.bitwise_count(np.abs(np.asarray(X, dtype=np.int64))).astype(np.int64)
def rnd_div(a, n):                                   # whole-number a / n rounded to nearest (half away from zero)
    a = np.asarray(a, dtype=object); return np.array([(abs(int(x)) * 2 + n) // (2 * n) * (1 if x >= 0 else -1) for x in a], dtype=object)

def load_gpt2():
    src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
    src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
    banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
    assert banks in src; src = src.replace(banks, '[]')
    cwd = os.getcwd(); os.chdir(HERE); argv = sys.argv; sys.argv = ["x"]; B = {}; exec(src, B); sys.argv = argv; os.chdir(cwd)
    M = B["M"]
    class Plates(dict):                              # base2's W, each bank's float64 copy made when read
        def __missing__(self, k): return (M[k + ".w"].astype(np.float64), M[k + ".e"], M[k + ".b"])
    B["W"] = Plates(); return B

def load_pythia(size):
    path = f"{HERE}/foundation0/pythia_base2.py"; src = open(path).read(); src = src[:src.index('if __name__ == "__main__":')]
    argv = sys.argv; sys.argv = ["x", f"size={size}"]; P = {"__file__": path, "__name__": "pyb2"}; exec(src, P); sys.argv = argv
    return P

R_ = {"reads": {}, "writes": {}, "leak": {}}         # per read / write name: lists over texts of per-position arrays
def keep(kind, name, **kw):
    d = R_[kind].setdefault(name, {})
    for k, v in kw.items(): d.setdefault(k, []).append(np.asarray(v))

# ================= GPT-2 =================
if MACH == "gpt2":
    B = load_gpt2(); M, W, ES = B["M"], B["W"], B["ES"]
    to16, gauge_log, product, power, attention, edge, plain_bank = (B[k] for k in ("to16", "gauge_log", "product", "power", "attention", "edge", "plain_bank"))
    WTE, WPE = M["wte"], M["wpe"]
    log(f"GPT-2 base-2 machine loaded ({time.time() - t0:.0f}s, {rss()} MB)")
    # ---- A. column sums of every read plate ----
    log("\nA. READ PLATES' COLUMN SUMS (whole numbers at each column's own power; 0 = the plate pours a level off exactly)")
    COLSUM = {}
    for name in [f"{b}.{l}" for l in range(12) for b in ("qkv", "fc")] + ["head"]:
        w = M[name + ".w"].astype(np.int64); cs = w.sum(0); COLSUM[name] = cs
        log(f"  {name:7s} {w.shape[1]:6d} columns: exactly 0 in {int((cs == 0).sum()):5d} ({100 * (cs == 0).mean():5.2f}%), "
            f"rms {np.sqrt((cs.astype(float) ** 2).mean()):6.2f}, largest |sum| {int(np.abs(cs).max())}")
        del w
    allcs = np.concatenate([v for k, v in COLSUM.items() if k != "head"])
    log(f"  all 24 layer read plates: zero {100 * (allcs == 0).mean():.2f}%, rms {np.sqrt((allcs.astype(float) ** 2).mean()):.2f}; "
        f"rounding alone: rms sqrt(768/12) = {np.sqrt(768 / 12):.2f}, P(0) about {100 / (np.sqrt(768 / 12) * np.sqrt(2 * np.pi)):.2f}%")

    def read(R, name):                                # the plain read, kept; plus the leak test (cut light + 1)
        x16, ex, LS = gauge_log(R); w, we, b = W[name]
        S1, S2 = x16.sum(1), (x16 * x16).sum(1)
        keep("reads", name, sh=ex + ES, S1=S1, Rsum=R.sum(1), top=np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(R), axis=1)]),
             work_level=ones(x16).sum(1) + ones(S1) ** 2 + ones(S1 * S1), water=(ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2))
        F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
        out = power(product(x16, w), F) + b
        y = x16 + 1; T1, T2 = y.sum(1), (y * y).sum(1)
        assert np.array_equal(768 * T2 - T1 * T1, 768 * S2 - S1 * S1)          # V does not see the unit
        out1 = power(product(y, w), F) + b
        keep("leak", name, changed=(out1 != out).sum(1), n=np.full(len(R), out.shape[1]))
        return out
    def write(X, name):
        x16, ex = to16(X, -ES); out = plain_bank(X, name)
        keep("writes", name, sh=ex + ES, Osum=out.sum(1), Bsum=np.full(len(X), int(W[name][2].sum())))
        return out
    refs = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
    log("\nB/C. THE PLAIN MACHINE ON THE 30 REFS TEXTS (every position)")
    for t, ids in enumerate(TEXTS):
        P = len(ids); R = WTE[ids] + WPE[:P]
        keep("writes", "entry", sh=np.zeros(P, int), Osum=R.sum(1), Bsum=WPE[:P].sum(1))
        for l in range(12):
            R = R + write(attention(read(R, f"qkv.{l}")), f"proj.{l}")
            R = R + write(edge(read(R, f"fc.{l}")), f"mproj.{l}")
        read(R, "head")
        if t % 10 == 9: log(f"  text {t + 1}/30 ({time.time() - t0:.0f}s, {rss()} MB)")
    READS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "fc")] + ["head"]
    WRITES = [f"{b}.{l}" for l in range(12) for b in ("proj", "mproj")]
    AFTER = {}                                        # the reads after each write, in the machine's order
    seq = ["entry"] + [x for l in range(12) for x in (f"qkv.{l}", f"proj.{l}", f"fc.{l}", f"mproj.{l}")] + ["head"]
    for i, s in enumerate(seq):
        if s == "entry" or s in WRITES: AFTER[s] = [r for r in seq[i + 1:] if r in READS]
    D = 768
    PLATES_W = {n: (M[n + ".w"], M[n + ".e"], M[n + ".b"]) for n in WRITES}
    ROWS = {"wte": WTE, "wpe": WPE}

# ================= Pythia =================
else:
    P_ = load_pythia(MACH); D, L, ES = P_["d"], P_["L"], P_["ES"]
    to16, gauge, plain, attention, edge, rshift, power, log2fix = (P_[k] for k in ("to16", "gauge", "plain", "attention", "edge", "rshift", "power", "log2fix"))
    NORM, PL = P_["NORM"], P_["PL"]
    log(f"Pythia-{MACH} base-2 machine loaded: d {D}, {L} layers ({time.time() - t0:.0f}s, {rss()} MB)")
    def gauge_x16(x16, ex, name):                     # pythia_base2.gauge from the cut light on (the same lines)
        g, ge, b = NORM[name]; S1, S2 = x16.sum(1), (x16 * x16).sum(1)
        c = D * x16 - S1[:, None]; V = D * S2 - S1 * S1 + np.array([P_["EPS40"] >> (40 + 2 * int(e)) for e in ex])
        LS = np.array([log2fix(v) >> 1 for v in V]); return power(c * g[None, :], (ge + ES) * 65536 - LS[:, None]) + b
    def read(R, name):
        x16, ex = to16(R); S1, S2 = x16.sum(1), (x16 * x16).sum(1)
        keep("reads", name, sh=ex + ES, S1=S1, Rsum=R.sum(1), top=np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(R), axis=1)]),
             work_level=ones(x16).sum(1) + ones(S1) ** 2 + ones(S1 * S1) + D * ones(S1),
             water=(ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2) + D * ones(S1))
        out = gauge(R, name); assert np.array_equal(out, gauge_x16(x16, ex, name))
        out1 = gauge_x16(x16 + 1, ex, name)
        keep("leak", name, changed=(out1 != out).sum(1), n=np.full(len(R), out.shape[1]))
        return out
    def write(X, name):
        x16, ex = to16(X); out = plain(X, name)
        keep("writes", name, sh=ex + ES, Osum=out.sum(1), Bsum=np.full(len(X), int(PL[name][2].sum())))
        return out
    refs = json.load(open(f"{HERE}/foundation0/refs_pythia-{MACH}.json"))
    TEXTS = [r["ids"] + r["ref"][:15] for r in refs]
    EMB_F, EMB_E = P_["EMB_F"], P_["EMB_E"]
    log("\nB/C. THE PLAIN MACHINE ON THE 30 REFS TEXTS (every position)")
    for t, ids in enumerate(TEXTS):
        ids = np.asarray(ids); R = rshift(EMB_F[ids].astype(np.int64), -(EMB_E[ids][:, None] + ES))
        keep("writes", "entry", sh=np.zeros(len(ids), int), Osum=R.sum(1), Bsum=np.zeros(len(ids), int))
        for l in range(L):
            g1 = read(R, f"ln1.{l}"); g2 = read(R, f"ln2.{l}")
            At = write(attention(plain(g1, f"qkv.{l}")), f"proj.{l}")
            Mo = write(edge(plain(g2, f"fc.{l}")), f"mproj.{l}")
            R = R + At + Mo
        read(R, "lnf")
        if t % 10 == 9: log(f"  text {t + 1}/30 ({time.time() - t0:.0f}s, {rss()} MB)")
    READS = [f"ln{k}.{l}" for l in range(L) for k in (1, 2)] + ["lnf"]
    WRITES = [f"{b}.{l}" for l in range(L) for b in ("proj", "mproj")]
    AFTER = {"entry": READS}
    for l in range(L):
        for b in ("proj", "mproj"): AFTER[f"{b}.{l}"] = [f"ln{k}.{m}" for m in range(l + 1, L) for k in (1, 2)] + ["lnf"]
    Mz = P_["M"]
    PLATES_W = {n: (Mz[n + ".w"], Mz[n + ".e"], Mz[n + ".b"]) for n in WRITES}
    ROWS = {"embed": (Mz["embed.w"], Mz["embed.e"])}

for kind in R_:
    for n in R_[kind]:
        for k in R_[kind][n]: R_[kind][n][k] = np.concatenate(R_[kind][n][k])
NPOS = len(R_["reads"][READS[0]]["sh"])
log(f"  {len(TEXTS)} texts, {NPOS} positions, {len(READS)} reads and {len(WRITES)} writes a position ({time.time() - t0:.0f}s, {rss()} MB)")

# ---- B. the cut powers and the level after the cut ----
log("\nB. THE CUTS. sh = the power of two each position's lanes are rounded to before the level is poured off (stream units 2^sh)")
log(f"  {'read':8s} {'sh_r min-med-max':>17s} {'first pos max':>13s} {'S1 = 0':>8s} {'|S1| median':>11s} {'|stream sum|/D median (stream units)':>36s}")
first = np.zeros(NPOS, bool); o = 0
for ids in TEXTS: first[o] = True; o += len(ids)
for n in READS:
    r = R_["reads"][n]; sh = r["sh"]
    log(f"  {n:8s} {int(sh.min()):5d} {int(np.median(sh)):5d} {int(sh.max()):5d} {int(sh[first].max()):13d} {int((r['S1'] == 0).sum()):5d}/{NPOS} "
        f"{np.median(np.abs(r['S1'])):11.0f} {np.median(np.abs(r['Rsum'])) / D:36.1f}")
log(f"  {'write':8s} {'sh_w min-med-max':>17s}   (the write's own cut: its input's power of two)")
for n in WRITES:
    sh = R_["writes"][n]["sh"]; log(f"  {n:8s} {int(sh.min()):5d} {int(np.median(sh)):5d} {int(sh.max()):5d}")

# ---- C. the leak ----
log("\nC. THE LEAK TEST: the cut light + 1 on every lane (a whole level the water level's V cannot see), read outputs changed")
tot_c = tot_n = 0
for n in READS:
    lk = R_["leak"][n]; c, nn = int(lk["changed"].sum()), int(lk["n"].sum()); tot_c += c; tot_n += nn
    pos = int((lk["changed"] > 0).sum())
    log(f"  {n:8s} {c:10d} of {nn:11d} ({100 * c / nn:8.4f}%), positions with any change {pos}/{NPOS}")
log(f"  all reads: {tot_c} of {tot_n} ({100 * tot_c / tot_n:.4f}%)")

# ---- D. the grain each write line's constant would need ----
log("\nD. THE GRAIN A WRITE LINE'S CONSTANT NEEDS TO PASS (i) and (ii) ON THESE TEXTS (top 1 not moving is extra, not checked here)")
log("   level = the line's average over the stream's lanes (real units); the constant = the level rounded to that grain;")
log("   'level left' = sum |level - constant| / sum |level|, at the format's finest grain / at the grain that passes")
GR = {}
def shr_after(name): return np.max(np.stack([R_["reads"][r]["sh"] for r in AFTER[name]]), axis=0)
for n in WRITES:
    w, we, b = PLATES_W[n]; w = w.astype(np.int64); Gf = int(we.max())
    sa = shr_after(n); sw = R_["writes"][n]["sh"]
    Gc = max(Gf, int((sa - sw).max()))               # real units: constant = k 2^G
    gb = int(sa.max())                               # bias: stream units, multiple of 2^gb
    # row levels, real: sum_j w_ij 2^we_j / D, exact in units of 2^min(we)
    lo = int(we.min()); num = (w * (1 << (we - lo))[None, :]).sum(1)          # level x D in units of 2^lo
    lev = num / D * 2.0 ** lo
    kc = np.round(lev / 2.0 ** Gc); kf = np.round(lev / 2.0 ** Gf)
    blev = b.sum() / D
    GR[n] = dict(G_format=Gf, G_cut=Gc, bias_grain=gb, rows=len(w), rows_nonzero_cut=int((kc != 0).sum()), rows_nonzero_fine=int((kf != 0).sum()),
                 level_abs_sum=float(np.abs(lev).sum()), left_cut=float(np.abs(lev - kc * 2.0 ** Gc).sum()), left_fine=float(np.abs(lev - kf * 2.0 ** Gf).sum()),
                 bias_level=float(blev), bias_removed_cut=float(np.round(blev / 2 ** gb) * 2 ** gb), sh_w_min=int(sw.min()), sh_r_after_max=int(sa.max()))
    g = GR[n]
    log(f"  {n:8s} rows: G format 2^{Gf:3d}, G that passes 2^{Gc:3d}; nonzero constants {g['rows_nonzero_fine']:5d} (format) {g['rows_nonzero_cut']:5d} (passes) of {len(w)}; "
        f"level left {100 * g['left_fine'] / g['level_abs_sum']:6.2f}% / {100 * g['left_cut'] / g['level_abs_sum']:6.2f}%  | "
        f"bias level {blev:9.1f} units, grain 2^{gb}: constant {g['bias_removed_cut']:.0f}")
for key, val in ROWS.items():
    sa_all = np.max(np.stack([R_["reads"][r]["sh"] for r in AFTER["entry"]]), axis=0); g = int(sa_all.max())
    if key in ("wte", "wpe"):
        tab = val; lev = tab.sum(1) / D
        kc = np.round(lev / 2 ** g); kf = np.round(lev)
        GR[key] = dict(grain=g, rows=len(tab), nonzero_fine=int((kf != 0).sum()), nonzero_cut=int((kc != 0).sum()),
                       level_abs_sum=float(np.abs(lev).sum()), left_fine=float(np.abs(lev - kf).sum()), left_cut=float(np.abs(lev - kc * 2 ** g).sum()))
    else:
        tab, e = val; s = np.maximum(-(e + ES), 0)  # the row's own shift onto the stream
        lev = tab.astype(np.int64).sum(1) / D * 2.0 ** (e + ES)                 # stream units
        kc = np.round(lev / 2 ** g); kf = np.round(lev)
        GR[key] = dict(grain=g, rows=len(tab), nonzero_fine=int((kf != 0).sum()), nonzero_cut=int((kc != 0).sum()),
                       level_abs_sum=float(np.abs(lev).sum()), left_fine=float(np.abs(lev - kf).sum()), left_cut=float(np.abs(lev - kc * 2 ** g).sum()))
    x = GR[key]
    log(f"  {key:8s} {x['rows']} rows (stream units): finest grain 1 -> nonzero {x['nonzero_fine']}, level left {100 * x['left_fine'] / x['level_abs_sum']:.2f}% | "
        f"grain that passes 2^{g}: nonzero {x['nonzero_cut']}, level left {100 * x['left_cut'] / x['level_abs_sum']:.2f}%; median |level| {np.median(np.abs(lev)):.1f} units")

# ---- E. the water level's work on the level ----
log("\nE. THE WATER LEVEL'S WORK ON THE LEVEL (photons, combined/machine.py's unit), per token = mean over positions")
wl = sum(R_["reads"][n]["work_level"].sum() for n in READS) / NPOS
wa = sum(R_["reads"][n]["water"].sum() for n in READS) / NPOS
s1z = sum(int((R_["reads"][n]["S1"] == 0).sum()) for n in READS)
log(f"  the level's adds, square and subtract{' and the centring subtract' if MACH != 'gpt2' else ''}: {wl:,.0f} a token, "
    f"{100 * wl / wa:.2f}% of the water levels' own work ({wa:,.0f})" + (f", {100 * wl / 3666596259:.5f}% of plain 3,666,596,259" if MACH == "gpt2" else ""))
log(f"  removable exactly only where the cut light's sum S1 is 0: {s1z} of {NPOS * len(READS)} reads")

json.dump(dict(machine=MACH, positions=NPOS, grains=GR, work_level_per_token=wl, water_per_token=wa, s1_zero=s1z,
               reads={n: dict(sh_min=int(R_["reads"][n]["sh"].min()), sh_max=int(R_["reads"][n]["sh"].max()), sh_median=float(np.median(R_["reads"][n]["sh"])),
                              s1_zero=int((R_["reads"][n]["S1"] == 0).sum()), leak_changed=int(R_["leak"][n]["changed"].sum()), leak_n=int(R_["leak"][n]["n"].sum())) for n in READS},
               writes={n: dict(sh_min=int(R_["writes"][n]["sh"].min()), sh_max=int(R_["writes"][n]["sh"].max())) for n in WRITES},
               colsum=({n: dict(zero=int((v == 0).sum()), cols=len(v), rms=float(np.sqrt((v.astype(float) ** 2).mean()))) for n, v in COLSUM.items()} if MACH == "gpt2" else None)),
          open(f"{OUT}/zerosum_cut_{MACH}.json", "w"), indent=1)
log(f"\ndone ({time.time() - t0:.0f}s, peak {rss()} MB)")
