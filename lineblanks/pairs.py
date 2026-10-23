"""LINE UP THE BLANKS, PART 2: EVERY UP PAIRED WITH A DOWN BY SIZE, THE TOP AND BOTTOM CUT OFF TOGETHER. 2026-10-06.

HIS WORDS (2026-10-06), only these are his: "The blank work must be in the embedding table equal in math to the MLP.
It has to cancel. So it's either tiny bit of all the 150k tokens embeddings vs MLP, we must not be the only one to
realize this . Our work of canceling might be doing some of it to begin with. Lining up and cancelling might do most.
Can you line up the blanks and see if it all cancels?"
Earlier, his (the pairing used here): "write every single up and down pair all in a line ... cut the top and bottom off"

HIS PICTURE, RESTATED (the session's restatement, not his sentence): on a line of weights, line the ups up from the
biggest down and the downs from the biggest down, and set the biggest up against the biggest down, the next against
the next, and so on in: the top and the bottom cut off together. Where an up and a down are the same size they cancel
completely; what does not cancel is the difference of each pair and whatever is left with no partner. If the blank all
cancels, a line of the blank leaves nothing.

THE COUNT, per line of weights (whole numbers, exact):
  ups U_1 >= U_2 >= ... (the positive weights), downs D_1 >= D_2 >= ... (the sizes of the negative weights), the
  shorter list filled with 0 at the bottom; what is left = sum over k of |U_k - D_k| (the unpaired ones count whole);
  the whole = sum of |w|. Reported as left / whole, pooled over a plate's lines, and |net| / whole (net = sum of the
  line's weights), which the left can never go below (left >= |net| on every line; asserted).
  Done two ways on every plate: along each LINE (a row: across the neurons) and along each NEURON'S COLUMN (across the
  lines). Rows of [lines][neurons] plates: one input lane across the plate's neurons; columns: one neuron across its
  lines. The output word table is [lanes][words]: its rows are lanes across the 50,304 words, its columns a word across
  the lanes. The input word table is [words][lanes]: its rows are a word's row (a write into the stream), its columns
  a lane across the words.
  Sizes are real sizes: every value of a plate on one scale 2^emin (common.real), since a line crosses neurons that sit
  on different powers of two.

VERSIONS: blank (step0), finished (step143000), fresh_b and fresh_f (fresh bell-curve draws of the blank's and the
  finished plate's per-neuron spreads, through fp16 onto the same grid, common.fresh), shuffled_b and shuffled_f (the
  plate's own values re-dealt at random over the whole plate: every value kept, which line and which column it sits in
  broken).

RULE 13, CHECKED BEFORE THE RUN
  13.1: no light here; weights only, as he asked. 13.2: the plates as stored. 13.3 / 13.4: pairing by size depends only
  on the SET of values on a line (any order of the line gives the same pairs), so a shuffle within a line keeps it and
  could show nothing. The scramble that breaks the pairing that matters is the plate-wide re-deal (which values sit
  together on one line), and the fresh draw is the bell curve's own leftover. What the count decides by definition:
  for a bell line of n values it is set by n alone (below). 13.5: his words quoted only above. 13.6: every line, every
  column, every plate, both sizes.
  ORDER-FREE CHECK: every line of three plates with its values in a random order gives the same left, digit for digit.

PREDICTIONS (written before this script first ran)
  The task's (the session's, handed to this run): "the blank pairs off to the order-statistic leftover of a bell draw
  (a few % of the whole), the finished leaves more where the plates lean (output table, MLP-in lanes)."
  This run's, worked out before the run: for a bell line of n values, left / whole is at least |net| / whole, about
  1/sqrt(n) on average; the ups and downs counts differ by about sqrt(n), which shifts every pair (adds about the same
  again), and two independent lists of the same bell differ rank by rank by about 0.9/sqrt(n); the parts overlap, so
  left / whole about 1.2-1.6 / sqrt(n):
     n = 512: 5.3-7.1%   768: 4.3-5.8%   1,536: 3.1-4.1%   2,048: 2.7-3.5%   2,304: 2.5-3.3%   3,072: 2.2-2.9%
     50,304: 0.53-0.71%
  - blank = fresh_b = shuffled_b within 10% (relative) on every plate kind, rows and columns. The blank never cancels
    to 0: it pairs off to the bell's own leftover, a few % of the whole, and no further.
  - finished, from the balance counts (balance/PREDICTION.txt):
      write lines that cancel below the coin toss (attention-out rows, MLP-out rows, the input table's word rows; also
      the output table's word columns): left at 0.6-1.0 x the blank's.
      lines that lean: MLP-in rows (40-69% of input lanes beyond 3 widths) 1.3-3 x the blank's; the output table's
      rows (every lane beyond 3) and the input table's lane columns (80% of lanes beyond 3, 14-29 widths) 5-30 x the
      blank's 0.6%; MLP-out columns and the 70m qkv columns 1-2 x.
  - shuffled_f: back to fresh_f within 20%, except where the whole plate leans (the 160m output table, plate net
    +118.7 widths: its rows stay about 3 x the bell's).

    python3 pairs.py size=70m       (writes pairs_<size>.json and pairs_<size>.log)
"""
import os, sys, json, time, resource, numpy as np
from common import real, versions

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE = A.get("size", "70m")
OUT = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(OUT), "foundation0")
LOGF = open(os.path.join(OUT, f"pairs_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time()

def pairoff(B):
    """every row of B: ups and downs each lined up from the biggest, paired rank by rank; (left, whole, net) per row"""
    U = np.sort(np.where(B > 0, B, 0), axis=1)[:, ::-1]; D = np.sort(np.where(B < 0, -B, 0), axis=1)[:, ::-1]
    left = np.abs(U - D).sum(1); whole = np.abs(B).sum(1); net = B.sum(1)
    assert (U.sum(1) - D.sum(1) == net).all() and (left >= np.abs(net)).all() and (left <= whole).all()
    return left, whole, net

def count(V, along):
    """pooled over the lines (along='rows') or the columns: sums of left, whole, |net|, and per-line left/whole quantiles"""
    R, C = V.shape; ls, ws, ns = [], [], []
    if along == "rows":
        bs = max(1, (1 << 21) // C)
        for a in range(0, R, bs): l, w, n = pairoff(V[a:a + bs]); ls.append(l); ws.append(w); ns.append(n)
    else:
        bs = max(1, (1 << 21) // R)
        for a in range(0, C, bs): l, w, n = pairoff(np.ascontiguousarray(V[:, a:a + bs].T)); ls.append(l); ws.append(w); ns.append(n)
    l, w, n = np.concatenate(ls), np.concatenate(ws), np.concatenate(ns)
    assert w.sum() < 2 ** 62
    rt = l / np.maximum(w, 1)
    return dict(n=int(V.shape[1] if along == "rows" else V.shape[0]), lines=int(len(l)), left=int(l.sum()), whole=int(w.sum()),
                absnet=int(np.abs(n).sum()), q50=float(np.median(rt)), q90=float(np.quantile(rt, 0.9)))

Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz")); L_ = int(Z["L"])
PLATES = [(f"{k}.{l}", 1) for l in range(L_) for k in ("qkv", "proj", "fc", "mproj")] + [("head", 1), ("embed", 0)]
SID = {"70m": 70, "160m": 160}[SIZE]
log(f"UPS PAIRED WITH DOWNS BY SIZE, pythia-{SIZE}: {len(PLATES)} plates; left / whole pooled over each plate's lines")
RES = {}; ofc = []
for pi, (name, axis) in enumerate(PLATES):
    tp = time.time(); e = np.asarray(Z[name + ".e"]); emin = int(e.min()); r = RES[name] = {}
    for v, q in versions(Z, name, axis, [20261006, SID, pi]):
        V = real(q, e, axis, emin); del q
        r[v] = dict(rows=count(V, "rows"), cols=count(V, "cols"))
        if v in ("blank", "finished"):
            rng = np.random.default_rng([20261006, SID, pi, 7 if v == "blank" else 8])
            Vs = rng.permutation(V.ravel()).reshape(V.shape)
            r["shuffled_" + v[0]] = dict(rows=count(Vs, "rows"), cols=count(Vs, "cols")); del Vs
            if name in ("proj.0", "fc.0", "head") and v == "finished":       # order-free: each line in a random order
                Vp = np.random.default_rng([20261006, SID, pi, 9]).permuted(V, axis=1)
                ofc.append((name, count(Vp, "rows")["left"] == r[v]["rows"]["left"])); del Vp
        del V
    log(f"  {name:8s} left/whole  rows (n {r['finished']['rows']['n']:5d}): " + " ".join(f"{v} {r[v]['rows']['left'] / r[v]['rows']['whole']:.3%}" for v in r)
        + f" | cols (n {r['finished']['cols']['n']:5d}): " + " ".join(f"{v} {r[v]['cols']['left'] / r[v]['cols']['whole']:.3%}" for v in r)
        + f"  ({time.time() - tp:.0f}s, {rss()} MB)")
    json.dump(RES, open(os.path.join(OUT, f"pairs_{SIZE}.json"), "w"))

VS = ("blank", "finished", "fresh_b", "fresh_f", "shuffled_b", "shuffled_f")
KINDS = [("qkv", "qkv", "an input lane across the q, k, v neurons", "a q/k/v neuron across its input lanes"),
         ("proj", "attention out", "a write line across the stream lanes", "a stream lane across the write lines"),
         ("fc", "MLP in", "an input lane across the MLP neurons", "an MLP neuron across its input lanes"),
         ("mproj", "MLP out", "a write line (one MLP neuron) across the stream lanes", "a stream lane across the MLP neurons"),
         ("head", "output word table", "a lane across the 50,304 words", "a word across the lanes"),
         ("embed", "input word table", "a word's row across the lanes (a write)", "a lane across the 50,304 words")]
def band(n): return f"{1.2 / n ** .5:.2%}-{1.6 / n ** .5:.2%}"
for along in ("rows", "cols"):
    log(f"\n== along each {'LINE (row)' if along == 'rows' else 'NEURON COLUMN'}: left / whole pooled over the plate kind (|net| / whole after the slash) ==")
    log(f"  {'kind':18s} {'n':>6s} {'bell band':>13s} | " + "  ".join(f"{v:>15s}" for v in VS) + "  | what one line is")
    for k, kn, rd, cd in KINDS:
        names = [n for n, _ in PLATES if n.split(".")[0] == k]
        cell = []
        for v in VS:
            lf = sum(RES[n][v][along]["left"] for n in names); wh = sum(RES[n][v][along]["whole"] for n in names); nt = sum(RES[n][v][along]["absnet"] for n in names)
            cell.append(f"{lf / wh:6.2%} /{nt / wh:6.2%}")
        n = RES[names[0]]["finished"][along]["n"]
        log(f"  {kn:18s} {n:6d} {band(n):>13s} | " + "  ".join(f"{c:>15s}" for c in cell) + f"  | {rd if along == 'rows' else cd}")
log("\n== finished against blank, plate by plate: left/whole finished / blank (rows, cols) ==")
for name, _ in PLATES:
    r = RES[name]; f = lambda v, a: r[v][a]["left"] / r[v][a]["whole"]
    log(f"  {name:8s} rows {f('finished', 'rows') / f('blank', 'rows'):6.2f}x  cols {f('finished', 'cols') / f('blank', 'cols'):6.2f}x"
        f"   | blank/fresh_b rows {f('blank', 'rows') / f('fresh_b', 'rows'):5.2f} cols {f('blank', 'cols') / f('fresh_b', 'cols'):5.2f}"
        f"   | shuffled_f/fresh_f rows {f('shuffled_f', 'rows') / f('fresh_f', 'rows'):5.2f} cols {f('shuffled_f', 'cols') / f('fresh_f', 'cols'):5.2f}")
log(f"\nORDER-FREE: each line in a random order, the same left: {ofc}")
log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
