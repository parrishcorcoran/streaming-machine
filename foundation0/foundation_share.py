"""THE FOUNDATION INSIDE THE FINISHED WEIGHTS (weights only, no light). 2026-10-06.

HIS IDEA, his exact words (only these are his):
  "The blank untrained model with random bell curve wieghts! That's the plate and the embedding table! That's the
  "foundation" we have been looking for." "It should be 100% removable after training. We could even test on a tiny
  model." "The random weights have random values and everything. That is trained is built on top. It creates the
  topographical map. That's IT! And the entire thing pre-training could be removed"

SESSION'S RESTATEMENT (the session's words, not his): finished weight W = foundation + what was built on top. The
foundation is the random starting draw W0, worn down by the building: every building step also shaves lr_t x 0.1 of
each weight back toward zero, so the surviving foundation is s x W0, the same s for every weight that takes the shave,
set by the building schedule alone. The foundation regrows from its seed, so it costs nothing to store.

WHAT IS MEASURED, on the shared whole-number grid of base2_pythia-<size>.npz (start .s and finished .w share one power
of two per neuron) and base2_pythia-70m-steps.npz (the steps between, on the same grid):
  1. s = (W . W0) / (W0 . W0) per plate and per word table, as an exact fraction of whole-number sums (each neuron's
     sums carried at 4^power, so neurons on different powers add exactly). Against the schedule's product
     prod_t (1 - 0.1 lr_t). 70m also at steps 1000, 10000, 50000. Gains, norm offsets and biases are left out: their
     start is the constant 1 or 0, not a draw, and the shave is assumed not to touch them.
  2. Rows never built on, in the input word table (rows = words) and the output word table (columns = words):
     A: every lane of the row within 1 grain of round(s_schedule x start);  B: within 1 grain of round(c x start),
     c the row's own (w . w0)/(w0 . w0). The 27 padding rows (ids 50277-50303, past the tokenizer's last word 50276)
     are named in advance.
  3. What is left to store: D = W - W0 and D' = W - round(s x W0) (s from the schedule, and s measured per plate),
     whole numbers on the same grid. Lit places = nonzero digits of the single-wave signed-digit writing
     (table_bits.naf: popcount(|x| ^ 3|x|)). Bits a weight at the floor = the entropy of a bell curve with each
     neuron's own spread, rounded to whole numbers on its grid. (With s measured, the floor saving of D' is
     -1/2 log2(1 - rho^2), rho = cosine of the plate with its start: that follows from the definition and is not a
     separate finding. Lit places are set by the format, not by the information.)
  CONTROLS: s with W0 shuffled within the plate (same values, positions scrambled), 3 shuffles, with the spread a
  random pairing must give, |W| / (|W0| sqrt N); rows paired with another row's start (rows shuffled), counted by A and
  B; order-free: every exact s recomputed with neurons reversed and lanes permuted (the same fraction, or it fails).

PREDICTIONS, written before this script first ran (results go to foundation_share.log / .json, never here):

(0) THE SESSION'S PREDICTIONS as handed to this run, copied as given:
    "s at the finished step 0.3-0.6 at the big plates, within 10% of the schedule's product; shuffle control under
    0.02; never-built rows exactly s x start; D' saves 0-10% of the lit places at the plates and most of the lit
    places at rarely-seen rows; bits saved a weight small (0.05-0.5) at the plates."
    Noted before the run: its first two halves cannot both hold. The schedule's product from Pythia's own settings
    (PREDICTION.txt item 5, recomputed in (A1)) is 3.87e-4 (14m, 70m) and 8.96e-3 (160m), not 0.3-0.6.

(A1) The schedule's product alone (warmup 1,430 steps linear from 0, then cosine from lr to min_lr over the remaining
     141,570; shave lr_t x 0.1 a step; lr 1e-3 -> 1e-4 for 14m and 70m, 6e-4 -> 6e-5 for 160m):
       14m, 70m: 0.9656 @1000, 0.396 @10000, 0.0108 @50000, 3.87e-4 @143000;   160m: 8.96e-3 @143000.
     (If the cosine spans all 143,000 steps instead: 3.63e-4 and 8.62e-3 at the end.)
(A2) Rows the light never reaches carry only the shave.
     - Input table: the 27 padding rows are c x start with ONE c for all 27, within 7% of (A1) at every step (70m:
       0.966, 0.396, 0.0108, 3.87e-4; 14m 3.87e-4; 160m 8.96e-3); they pass A. Also '<|padding|>' (id 1) and 0-50
       other words that never came up. 160m's grid is the sharpest test of the product (its rows sit at ~50 grains).
     - Output table: 0 rows never built on. Every word's row of the output table takes part in the water level at
       every position, so every row is pushed at every step, padding rows included.
     - Shuffle control (row against another row's start): 0 rows pass A or B.
(A3) Measured s = the shave's product + whatever of the build lies along the start:
     - 70m, plates and both tables: within 10% of 0.966 at step 1000; within 25% of 0.396 at 10000; under 0.05 at
       50000; under 0.02 at 143000 at every plate kind and every size (far below the session's 0.3-0.6).
     - At the finished 14m and 70m the shave's 3.87e-4 is under a single plate's shuffle spread (~|W|/(|W0| sqrt N),
       about 1e-3 for a million-weight plate): a single plate cannot tell it from 0.
     - Shuffle control: |s| under 0.005 at every plate, size and step. Order-free check: the same fraction every time.
(A4) What is left to store (finished step, shared grid):
     - D = W - W0 costs MORE than W (lit places and bits) at every plate kind and size, since s << 1/2 puts the whole
       start back in: +0.5 to +1.5 bits a weight at 14m (the start is the larger of the two in most of its neurons),
       +0.05 to +0.5 at 70m and 160m.
     - D' with the schedule's s: lit places within +-1% of W's and under 0.01 bits a weight saved at every plate kind
       and size. At never-built rows D' is 0 or +-1 at every lane (nearly all their lit places go), but they are ~30 of
       50,304 rows, so a table as a whole saves under 0.1%.
     - D' with each plate's measured s: under 0.01 bits a weight saved.
     - Width 14m -> 70m -> 160m: no saving at any width; D's extra cost falls with width, as the finished plates
       outgrow the start at 70m and 160m.

  python3 foundation_share.py [sizes=14m,70m,160m] [steps=1]
"""
import os, sys, json, math, time, resource
import numpy as np
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
A = dict(a.split("=") for a in sys.argv[1:])
SIZES = A.get("sizes", "14m,70m,160m").split(",")
DO_STEPS = A.get("steps", "1") == "1"
LR = {"14m": (1e-3, 1e-4), "70m": (1e-3, 1e-4), "160m": (6e-4, 6e-5)}
T, WU, WD = 143000, 1430, 0.1
PAD = list(range(50277, 50304))
rng = np.random.default_rng(20261006)
t0 = time.time()
def log(*a):
    print(*a, flush=True)

def shave_product(size, upto):
    lr, mn = LR[size]; p = 1.0
    for t in range(1, upto + 1):
        l = lr * t / WU if t <= WU else mn + (lr - mn) / 2 * (math.cos(math.pi * (t - WU) / (T - WU)) + 1)
        p *= 1 - WD * l
    return p

def to30(s): return int(round(s * 2 ** 30))                     # s as a whole number at 2^-30
def times(s30, q): return (s30 * q + (1 << 29)) >> 30            # round(s x q), whole numbers only
def naf(x): a = np.abs(x); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def hbits(sig):
    """bits a value of a bell curve with spread sig (in grains), rounded to whole numbers"""
    sig = np.asarray(sig, np.float64); out = np.zeros_like(sig)
    big = sig >= 2; out[big] = np.log2(sig[big] * math.sqrt(2 * math.pi * math.e))
    for i in np.flatnonzero((sig > 0) & ~big):
        s = sig[i]; k = np.arange(-int(14 * s) - 3, int(14 * s) + 4)
        p = np.array([0.5 * (math.erf((kk + .5) / (s * math.sqrt(2))) - math.erf((kk - .5) / (s * math.sqrt(2)))) for kk in k])
        p = p[p > 0]; out[i] = float(-(p * np.log2(p)).sum())
    return out

def kind(name): return name.split(".")[0]

def mats(z, name, key="w"):
    """neurons as rows: plates [in][out] -> [out][in]; head [hidden][words] -> [words][hidden]; embed is [words][hidden]"""
    X = z[f"{name}.{key}"]
    return X if name == "embed" else X.T

def blocks(R, C):
    B = max(1, (1 << 22) // C)
    for a in range(0, R, B): yield a, min(R, a + B)

def neuron_dots(W, W0, lane_perm=None):
    R, C = W.shape; dn = np.zeros(R, np.int64); d00 = np.zeros(R, np.int64); dww = np.zeros(R, np.int64)
    for a, b in blocks(R, C):
        w = np.ascontiguousarray(W[a:b]).astype(np.int64); w0 = np.ascontiguousarray(W0[a:b]).astype(np.int64)
        if lane_perm is not None: w, w0 = w[:, lane_perm], w0[:, lane_perm]
        dn[a:b] = (w * w0).sum(1); d00[a:b] = (w0 * w0).sum(1); dww[a:b] = (w * w).sum(1)
    return dn, d00, dww

def exact(num, e, order=None):
    """sum over neurons of num * 4^(e - emin) as a Python whole number (exact, any order)"""
    emin = int(e.min()); idx = np.arange(len(e)) if order is None else order; tot = 0
    for ev in np.unique(e):
        m = idx[e[idx] == ev]; tot += int(num[m].astype(object).sum()) << (2 * (int(ev) - emin))
    return tot

def measure_s(W, W0, e, check_order=True):
    dn, d00, dww = neuron_dots(W, W0)
    num, den, ww = exact(dn, e), exact(d00, e), exact(dww, e)
    s = Fraction(num, den); rho = num / math.sqrt(den) / math.sqrt(ww)
    ok = None
    if check_order:
        lp = rng.permutation(W.shape[1]); dn2, d002, _ = neuron_dots(W, W0, lp); rev = np.arange(len(e))[::-1]
        ok = Fraction(exact(dn2, e, rev), exact(d002, e, rev)) == s
    return s, rho, dn, d00, dww, ok

def shuffle_s(W, W0, e, n=3):
    sc = np.exp2(e.astype(np.float64))[:, None]
    wr = np.ascontiguousarray(W, dtype=np.float64); wr *= sc; wr = wr.ravel()       # exact: whole number x power of two
    w0 = np.ascontiguousarray(W0, dtype=np.float64); w0 *= sc; w0 = w0.ravel()
    den = float(np.dot(w0, w0)); expect = math.sqrt(float(np.dot(wr, wr)) * den / wr.size) / den; out = []
    for _ in range(n):
        rng.shuffle(w0); out.append(float(np.dot(wr, w0)) / den)
    del wr, w0
    return out, expect

def rows_never_built(W, W0, e, s30, c30):
    """per row: max |w - round(s x w0)| (A) and max |w - round(c_row x w0)| (B), on the row's own grid"""
    R, C = W.shape; mA = np.zeros(R, np.int64); mB = np.zeros(R, np.int64)
    for a, b in blocks(R, C):
        w = np.ascontiguousarray(W[a:b]).astype(np.int64); w0 = np.ascontiguousarray(W0[a:b]).astype(np.int64)
        mA[a:b] = np.abs(w - times(s30, w0)).max(1); mB[a:b] = np.abs(w - times(c30[a:b, None], w0)).max(1)
    return mA, mB

def rows_shuffled_control(W, W0, e, s30):
    """each row against ANOTHER row's start, carried onto this row's grid; counts passing A and B"""
    R, C = W.shape; perm = rng.permutation(R); passA = passB = 0
    for a, b in blocks(R, C):
        w = np.ascontiguousarray(W[a:b]).astype(np.int64); p = perm[a:b]
        w0 = np.round(np.ascontiguousarray(W0[p]).astype(np.float64) * np.exp2((e[p] - e[a:b]).astype(np.float64))[:, None]).astype(np.int64)
        d00 = (w0 * w0).sum(1); c = np.where(d00 > 0, (w * w0).sum(1) / np.maximum(d00, 1), 0.0)
        c30 = np.round(c * 2 ** 30).astype(np.int64)
        passA += int((np.abs(w - times(s30, w0)).max(1) <= 1).sum()); passB += int((np.abs(w - times(c30[:, None], w0)).max(1) <= 1).sum())
    return passA, passB

def store_stats(W, W0, s30_sched, s30_meas):
    """per neuron: sum, sum of squares, lit places of W, W0, D, D'(schedule s), D'(measured s)"""
    R, C = W.shape; names = ["W", "W0", "D", "Ds", "Dm"]
    S = {k: np.zeros(R, np.int64) for k in names}; Q = {k: np.zeros(R, np.int64) for k in names}; L = {k: np.zeros(R, np.int64) for k in names}
    for a, b in blocks(R, C):
        w = np.ascontiguousarray(W[a:b]).astype(np.int64); w0 = np.ascontiguousarray(W0[a:b]).astype(np.int64)
        for k, x in (("W", w), ("W0", w0), ("D", w - w0), ("Ds", w - times(s30_sched, w0)), ("Dm", w - times(s30_meas, w0))):
            S[k][a:b] = x.sum(1); Q[k][a:b] = (x * x).sum(1); L[k][a:b] = naf(x).sum(1)
    sig = {k: np.sqrt(np.maximum(Q[k] / C - (S[k] / C) ** 2, 0.0)) for k in names}
    return {k: dict(lit=int(L[k].sum()), bits_neuron=hbits(sig[k]), lit_neuron=L[k]) for k in names}, C

def plate_names(L): return [f"{k}.{l}" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")] + ["head", "embed"]

def decode_ids(ids):
    t = json.load(open(os.path.join(HERE, "models", "tokenizer", "tokenizer.json")))
    inv = {v: k for k, v in t["model"]["vocab"].items()}; inv.update({x["id"]: x["content"] for x in t.get("added_tokens", [])})
    return [inv.get(int(i), "<past the tokenizer>") for i in ids]

RES = dict(prediction_file=__file__, sizes={}, steps70={})
for size in SIZES:
    z = np.load(os.path.join(HERE, f"base2_pythia-{size}.npz")); L = int(z["L"])
    s_sched = shave_product(size, T); s30s = to30(s_sched)
    log(f"\n==== pythia-{size}: schedule's product at step 143000 = {s_sched:.4e}   ({time.time() - t0:.0f}s)")
    R = dict(s_schedule=s_sched, plates={}, kinds={}, tables={})
    agg = {}
    for name in plate_names(L):
        W, W0, e = mats(z, name, "w"), mats(z, name, "s"), z[f"{name}.e"].astype(np.int64)
        s, rho, dn, d00, dww, ok = measure_s(W, W0, e)
        sh, expect = shuffle_s(W, W0, e)
        sm = float(s); s30m = to30(sm)
        st, C = store_stats(W, W0, s30s, s30m)
        row = dict(values=int(W.size), s=sm, s_num_den=[s.numerator.bit_length(), s.denominator.bit_length()], rho=rho,
                   order_free_same=ok, shuffle=sh, shuffle_spread_expected=expect,
                   lit={k: st[k]["lit"] / W.size for k in st}, bits={k: float(st[k]["bits_neuron"].mean()) for k in st})
        R["plates"][name] = row
        a = agg.setdefault(kind(name), dict(values=0, num=0, lit={k: 0 for k in st}, bits={k: 0.0 for k in st}, plates=0, s_list=[]))
        a["values"] += W.size; a["plates"] += 1; a["s_list"].append(sm)
        for k in st: a["lit"][k] += st[k]["lit"]; a["bits"][k] += float(st[k]["bits_neuron"].sum()) * C
        log(f"  {name:8s} s {sm:+.5f}  rho {rho:+.5f}  shuffle {', '.join(f'{x:+.5f}' for x in sh)} (spread {expect:.5f})  order-free {ok}  "
            f"lit/wt W {row['lit']['W']:.3f} D {row['lit']['D']:.3f} D's {row['lit']['Ds']:.3f} D'm {row['lit']['Dm']:.3f}  "
            f"bits W {row['bits']['W']:.3f} D {row['bits']['D']:.3f} D's {row['bits']['Ds']:.4f} D'm {row['bits']['Dm']:.4f}")
        if name in ("head", "embed"):
            c = np.where(d00 > 0, dn / np.maximum(d00, 1), 0.0); c30 = np.round(c * 2 ** 30).astype(np.int64)
            mA, mB = rows_never_built(W, W0, e, s30s, c30)
            shA, shB = rows_shuffled_control(W, W0, e, s30s)
            rho_row = dn / np.sqrt(np.maximum(d00, 1)) / np.sqrt(np.maximum(dww, 1))
            idsA = np.flatnonzero(mA <= 1); idsB = np.flatnonzero(mB <= 1)
            litW, litDs = st["W"]["lit_neuron"], st["Ds"]["lit_neuron"]
            save_rows = np.where(litW > 0, 1 - litDs / np.maximum(litW, 1), 0.0)
            tb = dict(rows=int(W.shape[0]), passA=len(idsA), passB=len(idsB), idsA=idsA.tolist()[:200], idsB=idsB.tolist()[:200],
                      wordsB=decode_ids(idsB[:200]), pad_c=[float(c[i]) for i in PAD], pad_maxA=[int(mA[i]) for i in PAD],
                      pad_maxB=[int(mB[i]) for i in PAD], pad_passA=int((mA[PAD] <= 1).sum()), pad_passB=int((mB[PAD] <= 1).sum()),
                      id1_c=float(c[1]), id1_maxA=int(mA[1]), id1_maxB=int(mB[1]),
                      shuffled_rows_passA=shA, shuffled_rows_passB=shB,
                      c_passB_min=float(c[idsB].min()) if len(idsB) else None, c_passB_max=float(c[idsB].max()) if len(idsB) else None,
                      rho2_rows_hist={f">={th}": int((rho_row ** 2 >= th).sum()) for th in (0.01, 0.1, 0.25, 0.5, 0.9)},
                      rows_Ds_saves={f">={th}": int((save_rows >= th).sum()) for th in (0.1, 0.5, 0.9)},
                      lit_saved_whole_table=1 - st["Ds"]["lit"] / st["W"]["lit"])
            R["tables"][name] = tb
            log(f"    {name}: rows never built (A, schedule s) {len(idsA)}, (B, own c) {len(idsB)} of {W.shape[0]};  padding rows pass A {tb['pad_passA']}/27, "
                f"B {tb['pad_passB']}/27;  padding c {min(tb['pad_c']):.4e} .. {max(tb['pad_c']):.4e};  id 1 c {tb['id1_c']:.4e} maxA {tb['id1_maxA']} maxB {tb['id1_maxB']};  "
                f"shuffled rows pass A {shA}, B {shB}")
            log(f"      rows with rho^2 >= .01/.1/.25/.5/.9: {tb['rho2_rows_hist']};  rows where D' (schedule s) removes >=10/50/90% of lit places: {tb['rows_Ds_saves']}")
            if len(idsB): log(f"      B rows: c {tb['c_passB_min']:.4e} .. {tb['c_passB_max']:.4e}; first ids {idsB[:40].tolist()}")
        del W, W0
    for k, a in agg.items():
        R["kinds"][k] = dict(values=a["values"], plates=a["plates"], s_min=min(a["s_list"]), s_max=max(a["s_list"]),
                             lit={x: a["lit"][x] / a["values"] for x in a["lit"]}, bits={x: a["bits"][x] / a["values"] for x in a["bits"]})
    log(f"  -- per kind (lit places a weight; bits a weight at the floor):")
    for k, r in R["kinds"].items():
        log(f"  {k:6s} s {r['s_min']:+.5f}..{r['s_max']:+.5f}  lit W {r['lit']['W']:.3f} W0 {r['lit']['W0']:.3f} D {r['lit']['D']:.3f} ({r['lit']['D'] / r['lit']['W'] - 1:+.2%}) "
            f"D's {r['lit']['Ds']:.3f} ({r['lit']['Ds'] / r['lit']['W'] - 1:+.3%}) D'm {r['lit']['Dm']:.3f} ({r['lit']['Dm'] / r['lit']['W'] - 1:+.3%})  "
            f"bits W {r['bits']['W']:.3f} W0 {r['bits']['W0']:.3f} D {r['bits']['D'] - r['bits']['W']:+.3f} D's {r['bits']['Ds'] - r['bits']['W']:+.5f} D'm {r['bits']['Dm'] - r['bits']['W']:+.5f}")
    RES["sizes"][size] = R
    log(f"  peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB")
    del z

if DO_STEPS and "70m" in SIZES:
    log(f"\n==== pythia-70m along the building   ({time.time() - t0:.0f}s)")
    z = np.load(os.path.join(HERE, "base2_pythia-70m.npz")); zs = np.load(os.path.join(HERE, "base2_pythia-70m-steps.npz"))
    for step in (1000, 10000, 50000, 143000):
        sp = shave_product("70m", step); s30 = to30(sp); out = dict(schedule=sp, plates={}, tables={})
        for name in plate_names(6):
            W0, e = mats(z, name, "s"), z[f"{name}.e"].astype(np.int64)
            W = mats(z, name, "w") if step == 143000 else (zs[f"{name}.step{step}"] if name == "embed" else zs[f"{name}.step{step}"].T)
            s, rho, dn, d00, dww, ok = measure_s(W, W0, e, check_order=(name in ("fc.0", "embed")))
            sh, expect = shuffle_s(W, W0, e, n=1)
            out["plates"][name] = dict(s=float(s), rho=rho, shuffle=sh[0], spread=expect, order_free_same=ok)
            if name in ("head", "embed"):
                c = np.where(d00 > 0, dn / np.maximum(d00, 1), 0.0); c30 = np.round(c * 2 ** 30).astype(np.int64)
                mA, mB = rows_never_built(W, W0, e, s30, c30); shA, shB = rows_shuffled_control(W, W0, e, s30)
                out["tables"][name] = dict(passA=int((mA <= 1).sum()), passB=int((mB <= 1).sum()), pad_c_min=float(c[PAD].min()),
                                           pad_c_max=float(c[PAD].max()), pad_passA=int((mA[PAD] <= 1).sum()), pad_passB=int((mB[PAD] <= 1).sum()),
                                           id1_c=float(c[1]), shuffled_rows_passA=shA, shuffled_rows_passB=shB)
            del W, W0
        ks = {}
        for name, r in out["plates"].items(): ks.setdefault(kind(name), []).append(r["s"])
        log(f"  step {step:6d}: schedule {sp:.4e};  s by kind " + "  ".join(f"{k} {min(v):+.4f}..{max(v):+.4f}" for k, v in ks.items()))
        log(f"     shuffle (1 each) max |s| {max(abs(r['shuffle']) for r in out['plates'].values()):.5f};  order-free {[r['order_free_same'] for r in out['plates'].values() if r['order_free_same'] is not None]}")
        for tn, tb in out["tables"].items():
            log(f"     {tn}: never built A {tb['passA']} B {tb['passB']}; padding c {tb['pad_c_min']:.4e}..{tb['pad_c_max']:.4e} (pass A {tb['pad_passA']}, B {tb['pad_passB']}/27); id1 c {tb['id1_c']:.4e}; shuffled rows A {tb['shuffled_rows_passA']} B {tb['shuffled_rows_passB']}")
        RES["steps70"][step] = out
    log(f"  peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB")

json.dump(RES, open(os.path.join(HERE, "foundation_share.json"), "w"), indent=1, default=float)
log(f"\nwrote foundation_share.json   ({time.time() - t0:.0f}s)")
