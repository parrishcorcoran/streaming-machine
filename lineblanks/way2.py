"""LINE UP THE BLANKS, PART 1: WAY 2 ON THE BLANK AND THE FINISHED PLATES, ON THE SAME LIGHT. 2026-10-06.

HIS WORDS (2026-10-06), only these are his: "The blank work must be in the embedding table equal in math to the MLP.
It has to cancel. So it's either tiny bit of all the 150k tokens embeddings vs MLP, we must not be the only one to
realize this . Our work of canceling might be doing some of it to begin with. Lining up and cancelling might do most.
Can you line up the blanks and see if it all cancels?"
Earlier, also his: "It is zero sum because it must be. It doens have to be in the same spot. If you run the blank model
ALL that compute is the opposite sides canceling that we are trying to remove."

HIS PICTURE, RESTATED IN HIS WORDS (the session's restatement, not his sentence): run the blank, and all its work is
opposite sides cancelling. That cancelling does not have to happen in the same spot: the word rows of the table write
into the same stream as the MLP does, so they are the same kind of thing, and the cancel may sit spread thin over every
word row. When we line each line's weights up and let the light meet only the gaps, we are already cancelling
neighbours against each other, so our line-up may already be removing most of the blank's cancelling work.

THIS PART asks the last of those: does our line-up (way 2) take away as much of the blank's work as of the finished
work? Way 2 as in ../qwen06b/banks.py and ../qwen_fringes2.py (walk.c copied here unchanged): each line's weights are
lined up across its neurons once (the order and the gaps between neighbours in rank); per position the light on that
line meets only the gaps, the running product r_k = r_(k-1) + x gap_k is exactly x times the weight at rank k and is
poured into that neuron's pile. Counted, in photon meetings, exactly as banks.py counts them:
  plain   = sum over lines of ones(x) x (ones of every weight on the line)
  walk    = sum over lines of (signed digits of x) x (signed digits of the line's lowest weight and of every gap)
  pours   = ones(|r_k|) for every running product poured
  writing = ones(x), the light written once
  way 2   = walk + pours + writing, as a share of plain.
Zero gaps (ties) are counted per plate.

THE PLATES (shared grid of ../foundation0/base2_pythia-<size>.npz: .s the blank, step0; .w the finished, step143000;
one power of two per neuron, the same for both): qkv, proj, fc, mproj at every layer, and the output word table (head).
Four versions of every plate:
  blank     the step0 plate as it is
  finished  the step143000 plate as it is
  fresh_b   a fresh bell-curve draw, each neuron with the blank's own spread (in grains, on its own power of two)
  fresh_f   a fresh bell-curve draw, each neuron with the finished plate's own spread
Two lights (lineblanks/capture.py, the machine's own, 30 texts of texts_pythia.json, 622 positions):
  Lf  the finished machine's own light;   Lb  the blank machine's own light on the same words.
The input word table is read by one word, not by a light on all its lines, so it is counted weight-only (its rows
lined up across the lanes, and its lanes lined up across the 50,304 words) and with the one-word light (x = 1 on the
word's row, every lane poured).

CHECKS AND CONTROLS (rule 10 and rule 13)
  - exact: every walk's piles equal the direct whole-number product (mm.c, int64 sums) at every position, every plate,
    every version, both lights.
  - order-free: layer 0, the last layer and the head, (finished, Lf) and (blank, Lb): every line walked DOWN from its
    top weight in a random line order: same piles, same pours, every walk closes at 0.
  - the line-up's own control: the gaps counted with each line's neurons in a random order (count only). If way 2's
    saving is the line-up's, this goes back to about plain.
  - the light shuffled across its lines (same values at every position, line pairing broken; count only: walk/plain).
  - 13.1: the measure is the work (photon meetings), as in every earlier way 2 count; nothing here reads the 1s of the
    light as an answer. 13.2: the machine's own light and plates, every part as it is. 13.3: no agreement statistic;
    the comparison is blank against finished against a fresh draw. 13.4: the walk depends only on each line's set of
    values (any order of a line gives the same line-up) and on the light's digits; the pours are fixed by the light
    and the plate alone. So the blank's share is decided by its being a bell draw on this grid: the fresh draw of the
    same spreads is the check of that, and the finished against both is the measurement. 13.5: his sentences quoted
    only above; the restatement is labelled. 13.6: every plate, every position.

PREDICTIONS (written before this script first ran)
  The task's (the session's, handed to this run): "way 2's share on the blank equals the fresh draw's and is within a
  few points of the finished (the line-up cancels the bell curve's own pairs, which the blank and the finished both
  have)."
  This run's, from GPT-2 and Qwen's way 2 records (GPT-2 proj 58.85%, mproj 60.59%, qkv 54.06%, fc 52.51%, table
  38.07%; Qwen 44-56%, table 33.99%) and the spreads in foundation_share.log (the blank sits 1.6-2.3 bits lower on the
  shared grid at proj, mproj and the output table, 0.3-0.4 bits lower at qkv, fc and the input table):
  - exact everywhere; every order-free walk the same piles and pours, closing at 0.
  - blank = fresh_b within 0.5 points at every plate kind, both lights.
  - finished: proj and mproj (512 / 768 neurons a line) 55-65%, qkv (1,536 / 2,304) 50-58%, fc (2,048 / 3,072) 48-57%,
    the output table 30-45%.
  - blank against finished, same light: within -3..+5 points at proj, mproj and the head (fewer places on the blank
    lower the plain count and the walk together), within +-3 points at qkv and fc.
  - finished against fresh_f: within 3 points (the finished's heavier tails crowd the middle: a few more ties).
  - zero gaps: the blank's above the finished's at proj, mproj and the head (smaller spread in grains, more ties).
  - the two lights: the same plate's share within 3 points on Lf and Lb.
  - neurons in a random order: 90-110% of plain at every version (no saving without the line-up).
  - light shuffled across lines: walk/plain within 1 point of the real light's.
  - so, to his question: the line-up takes away the same share of the blank's work as of the finished work, about
    40-50% of plain in both; it cancels a bell curve's own neighbours, which every plate has, not the blank part as such.

  NOTE ADDED AFTER THE FIRST RUN (the predictions above are unchanged): the first run's fresh draws were plain bell
  draws on the grid, and they carried 6.52 ones a weight against the blank's 5.69 and the finished 5.68 (input table):
  the stored plates are fp16 numbers (at most 11 significant places, about 2.7 trailing zeros a weight on this grid),
  so a bell draw that skips fp16 is not the same number format and is no control. Each fresh value is now written as
  fp16 and put on the neuron's power of two as pythia_convert.py did for the stored plates (a value past 32,767 grains
  is held at 32,767: the stored plates never reach it, a fresh draw rarely does). The first run's numbers are kept in
  way2_70m_firstrun_nofp16.log/.json (blank and finished there are the same plates and lights as here).

    python3 way2.py size=70m       (writes way2_<size>.json and way2_<size>.log)
"""
import os, sys, json, time, ctypes, subprocess, resource, numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE = A.get("size", "70m")
OUT = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(OUT), "foundation0")
LOGF = open(os.path.join(OUT, f"way2_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(X): return np.bitwise_count(np.abs(np.asarray(X, dtype=np.int64))).astype(np.int64)
def naf(X): a = np.abs(np.asarray(X, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
t0 = time.time()

_lib = None
def lib():
    global _lib
    if _lib is None:
        so = os.path.join(OUT, "lb.dylib"); srcs = [os.path.join(OUT, "walk.c"), os.path.join(OUT, "mm.c")]
        if not os.path.exists(so) or os.path.getmtime(so) < max(os.path.getmtime(s) for s in srcs):
            subprocess.check_call(["cc", "-O2", "-shared", "-o", so] + srcs)
        _lib = ctypes.CDLL(so); P_, I_, i_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.walk.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, P_, i_, i_, I_, P_, P_, P_]; _lib.walk.restype = None
        _lib.mm16.argtypes = [I_, I_, I_, P_, P_, P_]; _lib.mm16.restype = None
    return _lib

def direct(x, w):
    """CHECK PATH: the exact product, int64 sums (mm.c)"""
    X = np.ascontiguousarray(np.asarray(x, np.int64)); W = np.ascontiguousarray(w); assert W.dtype == np.int16
    P, L = X.shape; assert W.shape[0] == L and np.abs(X).max(initial=0) <= 1 << 15
    out = np.empty((P, W.shape[1]), np.int64); lib().mm16(P, L, W.shape[1], X.ctypes.data, W.ctypes.data, out.ctypes.data)
    return out

class Lineup:
    """way 2's record of one plate (banks.py's, ranks in 32 bits): each line's weights in order of size across its
    neurons (ties in neuron order); GN the lit digits of each line's lowest weight and gaps; ZG its zero gaps; GNS the
    CONTROL: the same count with the line's neurons in a random order"""
    def __init__(self, w, rng):
        w = np.asarray(w); L, N = w.shape; self.L, self.N = L, N
        self.NR = np.empty((L, N), np.uint32); self.G = np.zeros((L, N), np.uint16)
        self.LOW = np.empty(L, np.int64); self.TOP = np.empty(L, np.int64)
        self.GN = np.empty(L, np.int64); self.ZG = np.empty(L, np.int64); self.GNS = np.empty(L, np.int64)
        B = max(16, min(256, (1 << 22) // N))                         # lines a block (memory)
        for i0 in range(0, L, B):
            wb = w[i0:i0 + B].astype(np.int64); o = np.argsort(wb, axis=1, kind="stable")
            ws = np.take_along_axis(wb, o, 1); g = np.diff(ws, axis=1)
            assert (g >= 0).all() and g.max(initial=0) < 1 << 16
            back = np.empty_like(wb); np.put_along_axis(back, o, np.concatenate([ws[:, :1], ws[:, :1] + np.cumsum(g, 1)], 1), 1)
            assert np.array_equal(back, wb), "the record does not give the plate back"
            self.NR[i0:i0 + B] = o; self.G[i0:i0 + B, 1:] = g; self.LOW[i0:i0 + B] = ws[:, 0]; self.TOP[i0:i0 + B] = ws[:, -1]
            self.GN[i0:i0 + B] = naf(ws[:, 0]) + naf(g).sum(1); self.ZG[i0:i0 + B] = (g == 0).sum(1)
            sh = rng.permuted(wb, axis=1); self.GNS[i0:i0 + B] = naf(sh[:, 0]) + naf(np.diff(sh, axis=1)).sum(1)
    def walk(self, x, line_order=None, down=False, blk=4):
        x = np.ascontiguousarray(np.asarray(x, np.int64)); P, L = x.shape; assert L == self.L and np.abs(x).max(initial=0) <= 1 << 15
        LO = np.ascontiguousarray(np.arange(L, dtype=np.int64) if line_order is None else np.asarray(line_order, np.int64))
        assert np.array_equal(np.sort(LO), np.arange(L))
        pile = np.zeros((P, self.N), np.int64); p1 = np.zeros(P, np.int64); op = np.zeros(P, np.int64)
        lib().walk(P, L, self.N, x.ctypes.data, self.LOW.ctypes.data, self.G.ctypes.data, self.NR.ctypes.data, self.TOP.ctypes.data,
                   LO.ctypes.data, -1 if down else 1, 1, blk, pile.ctypes.data, p1.ctypes.data, op.ctypes.data)
        return pile, p1, int(op.sum())

def spread(w, axis):
    """each neuron's spread in grains (axis 1: neurons are columns; 0: rows), from whole-number sums"""
    s1 = np.zeros(w.shape[axis]); s2 = np.zeros(w.shape[axis]); n = w.shape[1 - axis]
    for i0 in range(0, w.shape[0], 256):
        b = w[i0:i0 + 256].astype(np.int64)
        if axis == 1: s1 += b.sum(0); s2 += (b * b).sum(0)
        else: s1[i0:i0 + 256] = b.sum(1); s2[i0:i0 + 256] = (b * b).sum(1)
    return np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))

def fresh(w, e, axis, rng):
    """a fresh bell-curve draw: each neuron (along `axis`, the axis that holds one power of two) with that neuron's own
    spread in grains; the value written as fp16 (as the stored plates were) and put on the neuron's power of two the
    way pythia_convert.py puts every plate there (round(value / 2^e)), inside 16 places"""
    sig = spread(w, axis); out = np.empty(w.shape, np.int16)
    def put(g, ee):                                                     # grains -> real -> fp16 -> the grid
        sc = np.exp2(ee.astype(np.float64)); return np.clip(np.round((g * sc).astype(np.float16).astype(np.float64) / sc), -32767, 32767)
    if axis == 1:
        for i0 in range(0, w.shape[0], 64):
            n = min(64, w.shape[0] - i0); out[i0:i0 + n] = put(rng.normal(0.0, 1.0, (n, w.shape[1])) * sig[None, :], e[None, :])
    else:
        for i0 in range(0, w.shape[0], 4096):
            n = min(4096, w.shape[0] - i0); out[i0:i0 + n] = put(rng.normal(0.0, 1.0, (n, w.shape[1])) * sig[i0:i0 + n, None], e[i0:i0 + n, None])
    return out

Z = np.load(os.path.join(F0, f"base2_pythia-{SIZE}.npz")); L_ = int(Z["L"]); d = int(Z["d"])
LIGHT = {"Lf": np.load(os.path.join(OUT, f"light_{SIZE}_finished.npz")), "Lb": np.load(os.path.join(OUT, f"light_{SIZE}_start.npz"))}
refs = json.load(open(os.path.join(F0, f"refs_pythia-{SIZE}.json")))[:30]
WORDS = np.concatenate([np.array(r["ids"] + r["ref"][:15]) for r in refs]); P = len(WORDS)
assert all(LIGHT[k]["light.head"].shape[0] == P for k in LIGHT)
VERS = ("blank", "finished", "fresh_b", "fresh_f")
PLATES = [f"{k}.{l}" for l in range(L_) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
log(f"WAY 2 on the blank and the finished plates, pythia-{SIZE}: {len(PLATES)} plates x {len(VERS)} versions x 2 lights, {P} positions")
RES = {}
for pi, name in enumerate(PLATES):
    tp = time.time(); kind = name.split(".")[0]; layer = -1 if name == "head" else int(name.split(".")[1])
    wf, ws = np.asarray(Z[name + ".w"]), np.asarray(Z[name + ".s"])
    r = RES[name] = {}
    for vi, v in enumerate(VERS):
        rng = np.random.default_rng([20261006, {"70m": 70, "160m": 160}[SIZE], pi, vi])
        w = {"blank": ws, "finished": wf}.get(v)
        if w is None: w = fresh(ws if v == "fresh_b" else wf, np.asarray(Z[name + ".e"]), 1, rng)
        Ln, N = w.shape; lu = Lineup(w, rng); W1 = ones(w).sum(1)
        rv = r[v] = dict(lines=Ln, neurons=N, w_ones=int(W1.sum()), w_naf=int(naf(w).sum()), w_gap_lit=int(lu.GN.sum()),
                         w_gap_lit_shuffled=int(lu.GNS.sum()), zero_gaps=int(lu.ZG.sum()), gaps_total=Ln * (N - 1))
        for lt in ("Lf", "Lb"):
            x = np.asarray(LIGHT[lt]["light." + name]).astype(np.int64); assert x.shape == (P, Ln)
            ox, nx = ones(x), naf(x)
            c = dict(pairs=int(((x != 0).sum(1) * N).sum()), plain=int((ox @ W1).sum()), walk=int((nx @ lu.GN).sum()),
                     walk_neurons_shuffled=int((nx @ lu.GNS).sum()), writing=int(ox.sum()))
            dp = direct(x, w); pile, pours, _ = lu.walk(x)
            c["walk_piles_differ"] = int((pile != dp).any(1).sum()); c["pours"] = int(pours.sum()); del pile
            if v in ("finished", "blank") and (lt == "Lf") == (v == "finished") and (layer in (0, L_ - 1) or name == "head"):
                pile2, pours2, opn = lu.walk(x, line_order=rng.permutation(Ln), down=True)
                c["orderfree_piles_differ"] = int((pile2 != dp).any(1).sum()); c["orderfree_pours_equal"] = bool(pours2.sum() == pours.sum())
                c["orderfree_open"] = opn; del pile2
            del dp
            xs = rng.permuted(x, axis=1); assert (np.sort(xs, 1) == np.sort(x, 1)).all()
            c["ctl_plain"] = int((ones(xs) @ W1).sum()); c["ctl_walk"] = int((naf(xs) @ lu.GN).sum())
            c["way2"] = c["walk"] + c["pours"] + c["writing"]
            rv[lt] = c
        del lu, w
    bad = sum(r[v][lt]["walk_piles_differ"] for v in VERS for lt in ("Lf", "Lb"))
    of = [(r[v][lt]["orderfree_piles_differ"], r[v][lt]["orderfree_pours_equal"], r[v][lt]["orderfree_open"]) for v in VERS for lt in ("Lf", "Lb") if "orderfree_open" in r[v][lt]]
    log(f"  {name:8s} way 2 on Lf: " + "  ".join(f"{v} {r[v]['Lf']['way2'] / r[v]['Lf']['plain']:.2%}" for v in VERS)
        + f"   piles differing {bad}; order-free {of}  ({time.time() - tp:.0f}s, {rss()} MB)")
    json.dump(RES, open(os.path.join(OUT, f"way2_{SIZE}.json"), "w"))

# ---------------- the input word table: weight-only, and the one-word light ----------------
EW, ES_ = np.asarray(Z["embed.w"]), np.asarray(Z["embed.s"]); RES["embed"] = {}
for vi, v in enumerate(VERS):
    rng = np.random.default_rng([20261006, {"70m": 70, "160m": 160}[SIZE], 999, vi])
    w = {"blank": ES_, "finished": EW}.get(v)
    if w is None: w = fresh(ES_ if v == "fresh_b" else EW, np.asarray(Z["embed.e"]), 0, rng)   # one power of two per word: rows
    W1r = ones(w).sum(1); lur = Lineup(w, rng)                                  # each word's row lined up across its lanes
    one = dict(plain=int(W1r[WORDS].sum()), walk=int(lur.GN[WORDS].sum()), pours=int(W1r[WORDS].sum()), writing=P)
    one["way2"] = one["walk"] + one["pours"] + one["writing"]
    rv = dict(rows=dict(w_ones=int(W1r.sum()), w_naf=int(naf(w).sum()), w_gap_lit=int(lur.GN.sum()), w_gap_lit_shuffled=int(lur.GNS.sum()),
                        zero_gaps=int(lur.ZG.sum()), gaps_total=int(w.shape[0] * (w.shape[1] - 1)), one_word=one)); del lur
    luc = Lineup(np.ascontiguousarray(w.T), rng)                               # each lane lined up across the words
    rv["lanes"] = dict(w_gap_lit=int(luc.GN.sum()), w_gap_lit_shuffled=int(luc.GNS.sum()), zero_gaps=int(luc.ZG.sum()),
                       gaps_total=int(w.shape[1] * (w.shape[0] - 1))); del luc
    rv["values"] = int(w.size); RES["embed"][v] = rv; del w
json.dump(RES, open(os.path.join(OUT, f"way2_{SIZE}.json"), "w"))

# ---------------- report ----------------
def pool(names, v, lt, k): return sum(RES[n][v][lt][k] for n in names)
KINDS = [("qkv", "qkv"), ("proj", "attention out"), ("fc", "MLP in"), ("mproj", "MLP out"), ("head", "output word table")]
for lt, lname in (("Lf", "the finished machine's own light"), ("Lb", "the blank machine's own light")):
    log(f"\n== {lname} ({lt}), shares of the plain meetings of the same light; pooled over layers ==")
    log(f"  {'kind':18s} {'n/line':>7s} | way 2 share:  blank  finished  fresh_b  fresh_f | walk/plain b / f | pours/plain b / f "
        f"| zero gaps b / f / fb / ff | neurons shuffled b / f | light shuffled walk/plain f: real ctl")
    for k, kn in KINDS:
        names = [n for n in PLATES if n.split(".")[0] == k]
        sh = {v: pool(names, v, lt, "way2") / pool(names, v, lt, "plain") for v in VERS}
        wk = {v: pool(names, v, lt, "walk") / pool(names, v, lt, "plain") for v in VERS}
        po = {v: pool(names, v, lt, "pours") / pool(names, v, lt, "plain") for v in VERS}
        ns = {v: (pool(names, v, lt, "walk_neurons_shuffled") + pool(names, v, lt, "pours") + pool(names, v, lt, "writing")) / pool(names, v, lt, "plain") for v in VERS}
        zg = {v: sum(RES[n][v]["zero_gaps"] for n in names) / sum(RES[n][v]["gaps_total"] for n in names) for v in VERS}
        ctl = pool(names, "finished", lt, "ctl_walk") / pool(names, "finished", lt, "ctl_plain")
        log(f"  {kn:18s} {RES[names[0]]['finished']['neurons']:7d} |        {sh['blank']:7.2%} {sh['finished']:7.2%}  {sh['fresh_b']:7.2%}  {sh['fresh_f']:7.2%} "
            f"| {wk['blank']:6.2%} {wk['finished']:6.2%} | {po['blank']:6.2%} {po['finished']:6.2%} "
            f"| {zg['blank']:6.2%} {zg['finished']:6.2%} {zg['fresh_b']:6.2%} {zg['fresh_f']:6.2%} | {ns['blank']:7.2%} {ns['finished']:7.2%} "
            f"| {wk['finished']:6.2%} {ctl:6.2%}")
    sh = {v: pool(PLATES, v, lt, "way2") / pool(PLATES, v, lt, "plain") for v in VERS}
    pl = {v: pool(PLATES, v, lt, "plain") / P for v in VERS}
    log(f"  ALL {len(PLATES)} plates: way 2 share blank {sh['blank']:.2%}  finished {sh['finished']:.2%}  fresh_b {sh['fresh_b']:.2%}  fresh_f {sh['fresh_f']:.2%};"
        f"  plain a position: blank {pl['blank']:,.0f}  finished {pl['finished']:,.0f}  fresh_b {pl['fresh_b']:,.0f}  fresh_f {pl['fresh_f']:,.0f}")
    log(f"  work the line-up takes away a position (plain - way 2): " + "  ".join(f"{v} {(pool(PLATES, v, lt, 'plain') - pool(PLATES, v, lt, 'way2')) / P:,.0f}" for v in VERS))
bad = sum(RES[n][v][lt]["walk_piles_differ"] for n in PLATES for v in VERS for lt in ("Lf", "Lb"))
ofs = [(n, v, lt, RES[n][v][lt]["orderfree_piles_differ"], RES[n][v][lt]["orderfree_pours_equal"], RES[n][v][lt]["orderfree_open"])
       for n in PLATES for v in VERS for lt in ("Lf", "Lb") if "orderfree_open" in RES[n][v][lt]]
log(f"\nEXACT: walk piles differing from the direct product: {bad} position-plates over {len(PLATES) * len(VERS) * 2} walks; "
    f"order-free walks down in a random line order: {len(ofs)}, piles differing {sum(o[3] for o in ofs)}, pours equal {sum(o[4] for o in ofs)}/{len(ofs)}, "
    f"walks not closing at 0: {sum(o[5] for o in ofs)}")
log("\n== the input word table (one power of two per word) ==")
log(f"  {'version':9s} | ones/wt  signed/wt | rows across lanes: gap lit/wt  zero gaps  shuffled | lanes across words: gap lit/wt  zero gaps  shuffled | one-word light: way 2 / plain")
for v in VERS:
    e = RES["embed"][v]; n = e["values"]; o = e["rows"]["one_word"]
    log(f"  {v:9s} | {e['rows']['w_ones'] / n:7.3f} {e['rows']['w_naf'] / n:9.3f} |       {e['rows']['w_gap_lit'] / n:7.4f}   {e['rows']['zero_gaps'] / e['rows']['gaps_total']:7.2%}  {e['rows']['w_gap_lit_shuffled'] / n:7.4f} "
        f"|        {e['lanes']['w_gap_lit'] / n:7.4f}   {e['lanes']['zero_gaps'] / e['lanes']['gaps_total']:7.2%}  {e['lanes']['w_gap_lit_shuffled'] / n:7.4f} | {o['way2'] / o['plain']:7.2%}")
log(f"\ndone ({time.time() - t0:.0f}s, max memory {rss()} MB)")
