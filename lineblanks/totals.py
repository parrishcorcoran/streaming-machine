"""LINE UP THE BLANKS, PART 3: EVERY WRITE INTO THE STREAM ADDED TOGETHER. DOES THE BLANK ALL CANCEL? 2026-10-06.

HIS WORDS (2026-10-06), only these are his: "The blank work must be in the embedding table equal in math to the MLP.
It has to cancel. So it's either tiny bit of all the 150k tokens embeddings vs MLP, we must not be the only one to
realize this . Our work of canceling might be doing some of it to begin with. Lining up and cancelling might do most.
Can you line up the blanks and see if it all cancels?"
Earlier, also his: "It is zero sum because it must be. It doens have to be in the same spot. If you run the blank model
ALL that compute is the opposite sides canceling that we are trying to remove."

HIS PICTURE, RESTATED (the session's restatement, not his sentence): the word rows and the MLP's out-rows both write
into the same stream, so they are the same kind of thing. The blank's work is opposite sides cancelling, and the cancel
need not sit in the same spot: it may sit spread thin over every word row against the MLP. Then, lined up lane by lane,
the blank's writes should add to nothing, and the word table's part should stand against the MLP's part.

A. THE WEIGHTS (every line lit at 1 on the stream's scale, what the one-word light gives each word's row). The write
   lines into the stream: every word row of the input table (50,304), every attention-out row (proj, d a layer) and
   every MLP-out row (mproj, 4d a layer) of every layer; all on one scale 2^emin (common.real), whole numbers, exact.
   1. per write line: its total over the lanes against its coin-toss width sqrt(sum of squares): z.
   2. per stream lane: the total over all write lines, by group (E the word table; A all attention-out rows; M all
      MLP-out rows; and all together), against the coin-toss width sqrt(sum of squares) of the same lane: z.
   3. the word table against the MLP (his "embeddings vs MLP"), lane by lane: the agreement of the two lane-total
      lists r(E, M) = sum T_E T_M / sqrt(sum T_E^2 sum T_M^2), and the cancel ratio sum (T_E + T_M)^2 / (sum T_E^2 +
      sum T_M^2): 1 if they have nothing to do with each other, 0 if the word table cancels the MLP exactly. Also
      (E, A) and (A, M).
   Versions: blank, finished, fresh_b, fresh_f (common.fresh: per-neuron spreads, through fp16, the same grid).
   CONTROLS that break the pairing that matters (13.3): for 1, the plate's values re-dealt over the whole plate (which
   values sit on one write line); for 2, each write line's values re-dealt across its lanes (which lane each value
   writes into: every line's own total kept); for 3, the MLP's lane totals in a random lane order (2,000 orders: the
   range r takes when the lanes are not paired). ORDER-FREE: every lane total the same whole number with the lines
   summed in a random order (checked on the finished).
   The finished biases (written at every position) are reported apart: added to the finished totals as one more write
   per lane per layer.
B. THE MACHINE'S OWN WRITES (added by this run, beyond the task's three parts; lineblanks/capture.py, the 622
   positions): at every position the word row and every layer's attention write and MLP write, whole numbers at 2^-16,
   as the machine adds them. The stream at the top is their sum (asserted in capture). Q = sum of (top stream)^2 over
   (sum over writes of write^2), pooled over positions and lanes: 1 when the writes add like coin tosses, below 1 when
   they cancel, above 1 when they line up. Also with every write's level across the lanes taken out (the part a later
   gauge can meet). And at the same position: r(word row, all MLP writes), r(word row, all attention writes),
   r(attention, MLP), and r(word row, MLP write at layer l) layer by layer. For the blank machine's own run and the
   finished machine's own run. CONTROLS: every write after the word row with its lanes in a random order (each write,
   each position): Q must go to 1; every write after the word row taken from a random other position (one order of the
   positions per write): keeps whatever every position shares (the standing pattern), breaks what is the word's own.

RULE 13, CHECKED BEFORE THE RUN
  13.1: A reads weights only; B measures the writes the machine adds, not the light's 1s. 13.2: the plates and the
  machine as they are. 13.3: every count here has a control that keeps every value and breaks the one pairing it is
  about (listed above); no exact-equality statistic. 13.4: on the blank, A1 and A2 are set by its being a bell draw
  (the fresh draw shows that); the measurement is the finished against the blank and against the fresh draws, and
  A3's pairing of the word table with the MLP, which no definition fixes. 13.5: his words quoted only above.
  13.6: every write line of every layer, every lane, every position.

PREDICTIONS (written before this script first ran)
  The task's (the session's, handed to this run): "the blank's totals cancel to the coin toss and no further, the
  finished write lines cancel below the coin toss (the zero-sum game found earlier), but lane totals over all write
  lines do NOT cancel in the finished (the word tables lean 14-29 widths)."
  This run's:
  - A1 blank: rms z 0.95-1.05 in every group, |z| > 3 about 0.27%; = fresh_b. Finished: rms z 0.13-0.6 (the even part
    found in balance/even_part.log: writes 0.23-0.57 of the blank's width at 70m, 0.41-0.88 at 160m; in their own width
    0.13-0.28), the input table's rows about 0.24; the plate re-deal brings them back to about 1 (or to the plate's own
    lean). fresh_f about 1.
  - A2 blank: rms z 0.9-1.1 for E, A, M and all together, at most 2 lanes beyond 3 widths; = fresh_b.
    Finished: E rms z 10-30 (80% of lanes beyond 3); A about 1; M 2-10; all together NOT cancelled: rms z at least 5.
    The lane re-deal brings every group back to about 1.
  - A3 blank: r(E, M), r(E, A), r(A, M) inside the lane-order range (about +-2/sqrt(d): +-0.09 at 70m, +-0.07 at 160m);
    cancel ratio 1 +- 0.15. Finished: r(E, M) between -0.5 and +0.5, cancel ratio at least 0.5: the word table does not
    cancel the MLP lane by lane.
  - B blank's own run: Q 0.9-1.1 (the blank's writes add like coin tosses: nothing cancels); every r within +-0.05;
    lane scramble 1.00 +- 0.05. Finished's own run: Q above 1 (1.1-3: the standing pattern grows layer on layer);
    lane scramble 1.00 +- 0.05; the position scramble keeps most of the excess above 1. r(word row, MLP) between -0.4
    and +0.1.
  - So, to his question: the blank does not all cancel. Its writes add to the coin toss of their count, exactly as many
    random pushes do; the finished machine's write lines are the ones that cancel (each line below its coin toss), and
    the word table does not stand against the MLP lane by lane.

    python3 totals.py size=70m       (writes totals_<size>.json and totals_<size>.log)
"""
import os, sys, json, time, resource, math, numpy as np
from common import real, versions

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE = A.get("size", "70m")
OUT = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(OUT), "foundation0")
LOGF = open(os.path.join(OUT, f"totals_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time()
Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz")); L_ = int(Z["L"]); d = int(Z["d"]); SID = {"70m": 70, "160m": 160}[SIZE]
WPL = [("embed", 0, "E")] + [(f"proj.{l}", 1, "A") for l in range(L_)] + [(f"mproj.{l}", 1, "M") for l in range(L_)]
EMIN = min(int(np.asarray(Z[n + ".e"]).min()) for n, _, _ in WPL)
BOUND = 2 ** 62

def line_stats(V):
    """per write line (row): total over the lanes and sum of squares"""
    assert np.abs(V).max() < 2 ** 21
    return V.sum(1), (V * V).sum(1)
def lane_stats(V):
    return V.sum(0), (V * V).sum(0)
def zstats(t, w2):
    z = t / np.sqrt(np.maximum(w2, 1).astype(np.float64))
    return dict(rms=float(np.sqrt((z ** 2).mean())), beyond3=float((np.abs(z) > 3).mean()), maxabs=float(np.abs(z).max()), count=int(len(z)))
def agree(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())), float(((a + b) ** 2).sum() / ((a * a).sum() + (b * b).sum()))

# ================= A. the weights =================
log(f"EVERY WRITE INTO THE STREAM ADDED TOGETHER, pythia-{SIZE}: {len(WPL)} write plates on one scale 2^{EMIN}")
RES = {"A": {}}
for vname in ("blank", "finished", "fresh_b", "fresh_f"):
    tv = time.time(); LINES = {"E": [], "A": [], "M": []}; LINES_SH = {"E": [], "A": [], "M": []}
    T = {g: np.zeros(d, np.int64) for g in "EAM"}; W2 = {g: np.zeros(d, np.int64) for g in "EAM"}
    TR = {g: np.zeros(d, np.int64) for g in "EAM"}; W2R = {g: np.zeros(d, np.int64) for g in "EAM"}
    PER = {}; orderfree = None
    for pi, (name, axis, g) in enumerate(WPL):
        q = dict(versions(Z, name, axis, [20261006, SID, 500 + pi]) if vname.startswith("fresh") else
                 [("blank", np.asarray(Z[name + ".s"])), ("finished", np.asarray(Z[name + ".w"]))])[vname]
        V = real(q, np.asarray(Z[name + ".e"]), axis, EMIN); del q
        rng = np.random.default_rng([20261006, SID, 600 + pi, ("blank", "finished", "fresh_b", "fresh_f").index(vname)])
        t, w2 = line_stats(V); LINES[g].append((t, w2))
        Vs = rng.permutation(V.ravel()).reshape(V.shape); LINES_SH[g].append(line_stats(Vs)); del Vs        # re-dealt over the plate
        lt, lw = lane_stats(V); T[g] += lt; W2[g] += lw; PER[name] = zstats(lt, lw)
        Vr = rng.permuted(V, axis=1); a, b = lane_stats(Vr); TR[g] += a; W2R[g] += b; del Vr                  # each line across its lanes
        if vname == "finished" and name == "embed":                                                          # order-free: lines in a random order
            o = rng.permutation(V.shape[0]); acc = np.zeros(d, np.int64)
            for i0 in range(0, len(o), 4096): acc += V[o[i0:i0 + 4096]].sum(0)
            orderfree = bool(np.array_equal(acc, lt))
        del V
    assert all(np.abs(W2[g]).max() < BOUND and np.abs(W2R[g]).max() < BOUND for g in "EAM")
    r = RES["A"][vname] = {"per_line": {}, "per_line_redealt": {}, "lanes": {}, "lanes_redealt": {}, "agree": {}, "per_plate_lanes": PER}
    for g in "EAM":
        t = np.concatenate([x[0] for x in LINES[g]]); w2 = np.concatenate([x[1] for x in LINES[g]]); r["per_line"][g] = zstats(t, w2)
        t = np.concatenate([x[0] for x in LINES_SH[g]]); w2 = np.concatenate([x[1] for x in LINES_SH[g]]); r["per_line_redealt"][g] = zstats(t, w2)
    for gs in ("E", "A", "M", "AM", "EAM"):
        r["lanes"][gs] = zstats(sum(T[g] for g in gs), sum(W2[g] for g in gs))
        r["lanes_redealt"][gs] = zstats(sum(TR[g] for g in gs), sum(W2R[g] for g in gs))
    rng = np.random.default_rng([20261006, SID, 777])
    for a_, b_ in (("E", "M"), ("E", "A"), ("A", "M")):
        rr, cr = agree(T[a_], T[b_]); null = np.array([agree(T[a_], T[b_][rng.permutation(d)])[0] for _ in range(2000)])
        r["agree"][a_ + b_] = dict(r=rr, cancel=cr, null_lo=float(np.quantile(null, 0.025)), null_hi=float(np.quantile(null, 0.975)))
    if vname == "finished":
        r["orderfree_embed_lane_totals_equal"] = orderfree
        bias = np.zeros(d, np.int64); b2 = np.zeros(d, np.int64); sh = -16 - EMIN; assert sh >= 0
        for l in range(L_):
            for k in ("proj", "mproj"):
                bb = np.asarray(Z[f"{k}.{l}.b"]).astype(np.int64) << sh; bias += bb; b2 += bb * bb
        r["bias_lane_total"] = zstats(bias, sum(W2[g] for g in "EAM"))
        r["lanes_with_bias"] = zstats(sum(T[g] for g in "EAM") + bias, sum(W2[g] for g in "EAM") + b2)
    RES["A"][vname]["T"] = {g: T[g].tolist() for g in "EAM"}
    log(f"  {vname:8s}: per write line rms z  E {r['per_line']['E']['rms']:.3f}  A {r['per_line']['A']['rms']:.3f}  M {r['per_line']['M']['rms']:.3f}"
        f" | lane totals rms z  E {r['lanes']['E']['rms']:.2f}  A {r['lanes']['A']['rms']:.2f}  M {r['lanes']['M']['rms']:.2f}  all {r['lanes']['EAM']['rms']:.2f}"
        f" | r(E,M) {r['agree']['EM']['r']:+.3f}  cancel {r['agree']['EM']['cancel']:.3f}  ({time.time() - tv:.0f}s, {rss()} MB)")
    json.dump(RES, open(os.path.join(OUT, f"totals_{SIZE}.json"), "w"))

GN = {"E": "word rows (input table)", "A": "attention-out rows", "M": "MLP-out rows", "AM": "attention + MLP", "EAM": "ALL write lines"}
VS = ("blank", "finished", "fresh_b", "fresh_f")
log("\n== A1. each write line's total over the lanes, in its own coin-toss width (rms z; % beyond 3) ==")
log(f"  {'group':24s} " + "  ".join(f"{v:>17s}" for v in VS) + "   | re-dealt over the plate: blank  finished")
for g in "EAM":
    log(f"  {GN[g]:24s} " + "  ".join(f"{RES['A'][v]['per_line'][g]['rms']:7.3f} ({RES['A'][v]['per_line'][g]['beyond3']:5.2%})" for v in VS)
        + f"   | {RES['A']['blank']['per_line_redealt'][g]['rms']:7.3f} {RES['A']['finished']['per_line_redealt'][g]['rms']:7.3f}")
log(f"\n== A2. each stream lane's total over the write lines, in the lane's coin-toss width (rms z; % of {d} lanes beyond 3; largest |z|) ==")
log(f"  {'group':24s} " + "  ".join(f"{v:>25s}" for v in VS) + "   | each line re-dealt across its lanes: blank  finished")
for gs in ("E", "A", "M", "AM", "EAM"):
    log(f"  {GN[gs]:24s} " + "  ".join(f"{RES['A'][v]['lanes'][gs]['rms']:7.2f} ({RES['A'][v]['lanes'][gs]['beyond3']:6.2%}) {RES['A'][v]['lanes'][gs]['maxabs']:6.1f}" for v in VS)
        + f"   | {RES['A']['blank']['lanes_redealt'][gs]['rms']:7.2f} {RES['A']['finished']['lanes_redealt'][gs]['rms']:7.2f}")
f = RES["A"]["finished"]
log(f"  finished biases, summed over layers, per lane in the all-lines width: rms z {f['bias_lane_total']['rms']:.2f}; all write lines + biases: rms z {f['lanes_with_bias']['rms']:.2f} ({f['lanes_with_bias']['beyond3']:.2%} beyond 3)")
log(f"  per plate, finished lane totals rms z: " + "  ".join(f"{n} {f['per_plate_lanes'][n]['rms']:.2f}" for n, _, _ in WPL))
log(f"  per plate, blank    lane totals rms z: " + "  ".join(f"{n} {RES['A']['blank']['per_plate_lanes'][n]['rms']:.2f}" for n, _, _ in WPL))
log("\n== A3. the lane totals of one group against another's, lane by lane: r (the lanes in a random order: 95% range) and the cancel ratio ==")
for pr, pn in (("EM", "word table vs MLP"), ("EA", "word table vs attention"), ("AM", "attention vs MLP")):
    log(f"  {pn:24s} " + "  ".join(f"{v} r {RES['A'][v]['agree'][pr]['r']:+.3f} [{RES['A'][v]['agree'][pr]['null_lo']:+.3f},{RES['A'][v]['agree'][pr]['null_hi']:+.3f}] cancel {RES['A'][v]['agree'][pr]['cancel']:.3f}" for v in VS))
log(f"  order-free: the word table's lane totals with its rows summed in a random order equal: {f['orderfree_embed_lane_totals_equal']}")

# ================= B. the machine's own writes =================
def Q(W):
    """sum over positions and lanes of (sum of writes)^2 over sum of every write^2, exact whole numbers"""
    R = W.sum(1)
    return int((R.astype(object) ** 2).sum()) / int((W.astype(object) ** 2).sum())
def r_obj(a, b):
    ab = int((a.astype(object) * b.astype(object)).sum()); aa = int((a.astype(object) ** 2).sum()); bb = int((b.astype(object) ** 2).sum())
    return ab / math.sqrt(aa * bb), (aa + 2 * ab + bb) / (aa + bb)
RES["B"] = {}
for run in ("start", "finished"):
    W = np.load(os.path.join(OUT, f"light_{SIZE}_{run}.npz"))["writes"]; P, K, _ = W.shape; assert K == 1 + 2 * L_
    rng = np.random.default_rng([20261006, SID, 900, run == "finished"])
    e, a, m = W[:, 0], W[:, 1::2].sum(1), W[:, 2::2].sum(1)
    b = dict(Q=Q(W), rms_z=float(np.sqrt(((W.sum(1).astype(np.float64)) ** 2 / np.maximum((W.astype(np.float64) ** 2).sum(1), 1)).mean())))
    Wc = W * d - W.sum(2, keepdims=True); b["Q_level_out"] = Q(Wc); del Wc                                  # each write's level across the lanes out (x d, whole)
    b["r"] = {k: r_obj(x, y) for k, (x, y) in dict(EM=(e, m), EA=(e, a), AM=(a, m)).items()}
    b["r_EM_layer"] = [r_obj(e, W[:, 2 + 2 * l])[0] for l in range(L_)]; b["r_EA_layer"] = [r_obj(e, W[:, 1 + 2 * l])[0] for l in range(L_)]
    Wl = W.copy()
    for k in range(1, K):
        Wl[:, k] = np.take_along_axis(W[:, k], np.argsort(rng.random((P, d)), axis=1), axis=1)
    b["Q_lanes_scrambled"] = Q(Wl); b["r_lanes_scrambled"] = {"EM": r_obj(e, Wl[:, 2::2].sum(1))[0]}; del Wl
    Wp = W.copy()
    for k in range(1, K): Wp[:, k] = W[rng.permutation(P), k]
    b["Q_positions_scrambled"] = Q(Wp); b["r_positions_scrambled"] = {"EM": r_obj(e, Wp[:, 2::2].sum(1))[0], "EA": r_obj(e, Wp[:, 1::2].sum(1))[0]}; del Wp
    RES["B"][run] = b; del W
json.dump(RES, open(os.path.join(OUT, f"totals_{SIZE}.json"), "w"))
log(f"\n== B. the machine's own writes, {P} positions: Q = (top stream)^2 / sum of every write^2 (1: coin toss; below 1: they cancel) ==")
log(f"  {'run':28s} {'Q':>7s} {'rms z':>7s} {'Q level out':>12s} | {'lanes scrambled':>15s} {'positions scrambled':>20s} | r(word, MLP) cancel | r(word, attn) | r(attn, MLP) | r(word,MLP) lanes scr / pos scr")
for run, rn in (("start", "the blank machine's own"), ("finished", "the finished machine's own")):
    b = RES["B"][run]
    log(f"  {rn:28s} {b['Q']:7.3f} {b['rms_z']:7.3f} {b['Q_level_out']:12.3f} | {b['Q_lanes_scrambled']:15.3f} {b['Q_positions_scrambled']:20.3f} | "
        f"{b['r']['EM'][0]:+.3f} {b['r']['EM'][1]:.3f} | {b['r']['EA'][0]:+.3f} | {b['r']['AM'][0]:+.3f} | {b['r_lanes_scrambled']['EM']:+.3f} / {b['r_positions_scrambled']['EM']:+.3f}")
    log(f"  {'':28s} r(word row, MLP write at layer l): " + " ".join(f"{x:+.3f}" for x in b["r_EM_layer"]) + "   attention: " + " ".join(f"{x:+.3f}" for x in b["r_EA_layer"]))
log(f"\ndone ({time.time() - t0:.0f}s, max memory {rss()} MB)")
