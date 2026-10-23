"""THE GATES, PAIRED BY SIDE, EVERY LAYER.  A MEASUREMENT ONLY; NOTHING IN THE MACHINE CHANGES.

THE PICTURE. His words for the edge: "The edge is the same backwards and forwards." The pairing by side below is the
session's restatement (DIRECTIONS.md direction 21, from checks/halves.txt), not his sentence. Every bend is a Gaussian
knife edge; write its output as the open part minus a glow:
    out = bend(H) = o - g        o = H if H > 0 else 0  (the open part: the pile passed straight on the open side)
                                 g = o - bend(H)        (the glow: on the open side what the edge holds back of H; on the
                                                         dark side the faint leak, out = -g, because o = 0 there)
Lens 2 (mproj.l) is a plain sum over the 3,072 edges, so a layer's write splits the same way, exactly:
    write(out) = write(o) - write(g)       (every line, every position; whole numbers before the water level's drop)
The question: is the glow a small edge effect, so the gates reduce to side bits (o is H times the side bit), or is it
a part of the write as large as the open part?

THE MACHINE (base2.py, unchanged; its own functions are exec'd and called, as knife/settle.py does):
  H   = gauged_bank(R, fc.l)              the value arriving at each edge, at 2^-16 (x = H / 65536)
  out = edge(H)  = rshift(H x share, 16)  share = PHI[rshift(H, 4) + 32768], 0 below the table, 65536 above
  out16, eo = to16(out, -16)              sh = eo + 16: the 16 places lens 2 meets
  mo  = plain_bank(out, mproj.l)          = rshift(product(out16, w2), -(we2 + eo + 16)) + b2: the layer's write
THE LIGHT: the stream entering fc.l at every position of the 30 refs texts (648 positions), from
recording/C_ref/plain_refs.npz (base2.py's own run, C_plain.py). Checked here: every fc pile equals that run's, every
mproj pile equals that run's, and R + mo equals that run's stream after the layer (bit-identical stream).

FIXED BY THE ROUNDING, NOT MEASURED (rule 13.4: the run checks the code here, not the machine):
  g >= 0 everywhere. bend(H) = floor((H x share + 2^15) / 2^16) with 0 <= share <= 2^16. For H > 0, H x share <= H x 2^16,
  so bend(H) <= floor(H + 1/2) = H. For H < 0, H x share <= 0, so bend(H) <= floor(1/2) = 0 and g = -bend(H) >= 0.
  The identity o - g = out, and "on the dark side the glow is the whole output", are the definition.
  The glow is a function of |H| within 5 units: for H > 0, g(-H) - g(H) = bend(H) - bend(-H) - H, which section 27
  measured as 0 for 98.22% of H and within -5..+5 otherwise (nonzero only where H = 8 mod 16). Checked here per edge.
WHAT IS MEASURED (every layer, every edge of the 30 texts, 648 x 12 x 3,072 = 23,887,872 edges):
  per layer: sum |o|, sum g, sum |out| (whole numbers at 2^-16); the share of edges with g exactly 0; the dark-side
  share of the glow; through lens 2, exactly (o, g and out met with the mproj plate at full places, every product and
  sum a whole number below 2^52): sum over lines and positions of |write(o)|, |write(g)|, |write(out)|, each line in its
  own column's power of two, all brought to the layer's finest one (whole numbers); the share of lines where the glow's
  write is larger than the open part's, and larger than the whole write; how much the two parts cancel,
  (sum |write(o)| + sum |write(g)|) / sum |write(out)|; and the machine's own write (after the water level's drop)
  against the exact one.
  Split by where H sits (x = H / 65536; settle.py's groups): far dark x < -4; -4..-2; -2..0; 0..2.66; open x >= 2.66.
  Each group's own part of the open and glow writes (the groups' writes add to the whole exactly; checked).
WORK, in the unit of all earlier results (photon meetings = light photons x plate photons on each line):
  lens 2 plain: photons of |out16| x the mproj plate's photons on that line (the machine's own lens-2 meeting).
  the pair at the machine's own places: o16 = rshift(o, sh), g16 = o16 - out16 (>= 0, since rshift never runs back), so
  out16 = o16 - g16 exactly and lens 2 meets o16 and g16 apart: photons of |o16| and of g16, each x the plate's photons.
  The subtract that joins them (one a line) is stated apart. Whole machine: 3,666,596,259 a token (FINDINGS.md).
CONTROLS:
  light=shuffled      each position's light shuffled across its 768 lines before fc (every value kept; settle.py's seeds
                      2026 + l, so the piles are settle.py's control's)
  light=shuffled447   the same, with line 447 (the first-place stone's line) held in place (section 28's note)
ORDER-FREE: order=1 moves the positions, the light's lines (with fc's rows), the neurons (fc's columns, fc's bias and
  scale, mproj's rows) and the stream's lines (mproj's columns, its scale and bias), runs base2's own functions on that,
  puts everything back, and every array's sha256 and every total must equal the real run's (gates_real.json).
THE CROSS-CHECK: gates/xchk_gates.py, a second script by another route (the glow read straight from the complement of
  the table, never from out; lens 2 met in whole-number int64 arithmetic, not float containers; its own groups, sums
  and ratios), rebuilding every total of gates_<light>.json.

GATING (holographicplate CLAUDE.md, main): a measurement of the bends' own outputs (the finished film, not the light
going in); every read is on whole numbers; it is exact (an identity, nothing approximated); it discards nothing; no
search, no best value, no fit; not measured before (direction 21's first exact test, never run).
RULE 13 against this test: (1) it measures the edges' outputs and their writes, not the 1s of the light going in
(photons appear only as the work unit); (2) the base is the machine as it runs, nothing changed, nothing split that the
machine does not already add (lens 2 is a sum); (3) every statistic changes under the shuffle (the piles move);
(4) g >= 0, o - g = out and the dark side's glow = out are fixed by the definition and stated as such above;
(5) the pairing by side is the session's, labelled so; (6) every layer, every position, every edge.

PREDICTIONS (written before this script first ran):
  THE TASK'S: the glow is nonzero on more than 99% of edges; on the dark side (83-87% of edges) the glow IS the whole
    output, so through lens 2 the glow's write is about as large as the open part's; the gates do not reduce to side bits.
    (The dark-side clause holds by the definition; the run measures the rest.)
  THE SESSION'S (from sections 4, 25, 27, 28 and the edge's shape: |x|Phi(x) and x(1 - Phi(x)) both peak at about 0.17
  near |x| = 0.75, and the table is flat only past |x| = 4.32):
    - g exactly 0 on 0.05-0.6% of edges per layer, almost all far dark past the table (share 0); glow nonzero on
      99.4-99.95%. Shuffled light: g = 0 on 1-5% (the shuffle spreads piles to both ends, section 25).
    - the glow is symmetric: g(-H) = g(H) on about 98.2% of edges, within 5 everywhere.
    - sum g / sum |o| between 0.5 and 1.5 at every layer; sum g / sum |out| 0.4-0.7; the dark side carries 75-90% of
      the glow.
    - lens 2: sum |write(g)| / sum |write(o)| between 0.4 and 1.2 at every layer (the same order: the task's "about as
      large"); the two parts cancel only mildly, (|write(o)| + |write(g)|) / |write(out)| between 1.1 and 2.5 at every
      layer, nothing like the midpoint halves (each 2.5-10 times the write, section 27). The glow's write is larger
      than the open part's on 25-50% of lines.
    - groups: -2..0 carries most of the glow's write (60-85% of the groups' summed |write(g)|), -4..-2 under 10%, far dark
      and open about 0; the open part's write comes almost all from 0..2.66.
    - work: the pair at the machine's places costs the plain lens-2 meeting plus the bright side's open and glow on top
      of its out16: 110-130% of plain lens 2. Pairing by side saves no lens-2 work by itself (the dark side's glow is
      exactly its out16).
    - control: shuffled light has more piles on the open side and at the far ends, so sum g / sum |o| and the glow's
      write against the open part's both fall, by a factor of 1.5-3; with line 447 held, between real and shuffled.

    python3 gates/gates.py [light=real|shuffled|shuffled447] [order=0|1] [layers=0,5,11]
      -> gates/gates_<tag>.log and gates/gates_<tag>.json
"""
import sys, os, time, json, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; GD = f"{HERE}/gates"; sys.path.insert(0, HERE); os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
LIGHT, ORDER = ARG.get("light", "real"), int(ARG.get("order", 0))
assert LIGHT in ("real", "shuffled", "shuffled447")
LAYERS = [int(v) for v in ARG["layers"].split(",")] if "layers" in ARG else list(range(12))
TAG = LIGHT + ("_order" if ORDER else "") + ("" if "layers" not in ARG else "_L" + "-".join(map(str, LAYERS)))
LOG = open(f"{GD}/gates_{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def el(): return int(time.time() - t0)

_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]     # base2.py up to its own test
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')        # W filled one bank at a time, by base2's own line
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W, ES, PHI = J["M"], J["W"], J["ES"], J["PHI"]
rshift, edge, to16, gauged_bank, plain_bank, product = (J[k] for k in
    ("rshift", "edge", "to16", "gauged_bank", "plain_bank", "product"))
REC = []
def _rec(x, w): r = product(x, w); REC.append(r); return r
J["product"] = _rec                                              # base2's own products, kept to check the piles
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
P, N, LINES = int(REF["npos"].sum()), 3072, 768
assert P == 648
def fill(name, perm_in=None, perm_out=None):                      # base2's own line for W, optionally with moved rows/cols
    w, e, b = M[name + ".w"], M[name + ".e"], M[name + ".b"]
    if perm_in is not None: w = w[perm_in]
    if perm_out is not None: w, e, b = w[:, perm_out], e[perm_out], b[perm_out]
    W[name] = (w.astype(np.float64), e, b)

def bucket_of(H):                                       # settle.py's groups: x = H / 65536
    return np.where(H < -4 * 65536, 0, np.where(H < -2 * 65536, 1, np.where(H < 0, 2, np.where(100 * H < 266 * 65536, 3, 4))))
BNAMES = ["far dark x<-4", "-4..-2", "-2..0", "0..2.66", "open >=2.66"]
def meet(X, wf, wa):                                    # X (whole numbers) met with the plate, exact: every partial sum
    B = np.abs(X).astype(np.float64) @ wa               # is bounded by |X|.|w|, checked below 2^52
    assert B.max(initial=0) < 2.0 ** 52
    return (X.astype(np.float64) @ wf).astype(np.int64)
def colsum(A): return np.abs(A).sum(0, dtype=object)   # per column, Python whole numbers (no overflow)
def at_scale(cs, k): return sum(int(v) << int(s) for v, s in zip(cs, k))   # columns brought to the finest power of two
def rat(a, b, d=3):                                     # a / b printed from whole numbers, rounded half up
    a, b = int(a), int(b)
    if b == 0: return "-"
    s = "-" if (a < 0) != (b < 0) and a != 0 else ""; a, b = abs(a), abs(b)
    q = (2 * a * 10 ** d + b) // (2 * b); return f"{s}{q // 10 ** d}.{q % 10 ** d:0{d}d}"
def pc(a, b, d=2): return rat(100 * int(a), b, d) + "%"
def sha(A): return hashlib.sha256(np.ascontiguousarray(A, np.int64).tobytes()).hexdigest()

BAD = {}
def bad(k, n): BAD[k] = BAD.get(k, 0) + int(n)
RES = {"light": LIGHT, "order": ORDER, "layers": LAYERS, "per_layer": {}}; DIG = {}
log(f"the gates paired by side: light={LIGHT} order={ORDER} layers={LAYERS}; {P} positions x {N} edges a layer")
for l in LAYERS:
    R = REF[f"stream_a_{l}"].astype(np.int64)
    if LIGHT == "shuffled":                             # CONTROL: each position's light across its lines (settle.py's seeds)
        rng = np.random.default_rng(2026 + l); R = np.stack([r[rng.permutation(LINES)] for r in R])
    elif LIGHT == "shuffled447":                        # CONTROL: the same, line 447 held in place
        rng = np.random.default_rng(4470 + l); keep = np.r_[0:447, 448:LINES]
        Rn = R.copy()
        for p in range(P):
            perm = rng.permutation(keep); Rn[p, keep] = R[p, perm]
        R = Rn; assert np.array_equal(R[:, 447], REF[f"stream_a_{l}"][:, 447])
    PP, LP, NP_, JP = np.arange(P), np.arange(LINES), np.arange(N), np.arange(LINES)
    if ORDER:                                           # positions, light lines, neurons and stream lines all moved
        rng = np.random.default_rng(909 + l)
        PP, LP, NP_, JP = rng.permutation(P), rng.permutation(LINES), rng.permutation(N), rng.permutation(LINES)
    fill(f"fc.{l}", LP, NP_); fill(f"mproj.{l}", NP_, JP)
    Rm = R[PP][:, LP]
    REC.clear(); Hm = gauged_bank(Rm, f"fc.{l}"); pile1 = REC[0]           # the machine's own, on what it is given
    outm = edge(Hm)
    REC.clear(); mom = plain_bank(outm, f"mproj.{l}"); pile2 = REC[0]
    out16m, eom = to16(outm, -ES)
    w2f, we2, b2 = W[f"mproj.{l}"]; w2a = np.abs(w2f)
    # ---- the pair, in the order the machine was given ----
    om = np.where(Hm > 0, Hm, 0); gm = om - outm
    bad("g_negative", (gm < 0).sum()); bad("o_minus_g_not_out", (om - gm != outm).sum())
    bad("dark_glow_not_minus_out", ((Hm < 0) & (gm != -outm)).sum())
    WOm, WGm, WXm = meet(om, w2f, w2a), meet(gm, w2f, w2a), meet(outm, w2f, w2a)
    bad("write_identity", (WOm - WGm != WXm).sum())
    BKm = bucket_of(Hm)
    WOGm = [meet(np.where(BKm == k, om, 0), w2f, w2a) for k in range(5)]
    WGGm = [meet(np.where(BKm == k, gm, 0), w2f, w2a) for k in range(5)]
    bad("groups_add_to_open_write", (sum(WOGm) != WOm).sum()); bad("groups_add_to_glow_write", (sum(WGGm) != WGm).sum())
    shm = eom + ES; o16m = rshift(om, shm[:, None]); g16m = o16m - out16m
    bad("g16_negative", (g16m < 0).sum()); bad("g16_identity", (o16m - g16m != out16m).sum())
    gmir = np.where(-Hm > 0, -Hm, 0) - edge(-Hm)         # the glow the mirrored pile would make
    # ---- put everything back in the machine's own order ----
    inv = lambda A, rows, cols: A[np.argsort(rows)][:, np.argsort(cols)]
    H, out, o, g = (inv(A, PP, NP_) for A in (Hm, outm, om, gm))
    BK, gmirror, o16, g16, out16 = inv(BKm, PP, NP_), inv(gmir, PP, NP_), inv(o16m, PP, NP_), inv(g16m, PP, NP_), inv(out16m, PP, NP_)
    WO, WG, WX, mo = (inv(A, PP, JP) for A in (WOm, WGm, WXm, mom))
    WOG = [inv(A, PP, JP) for A in WOGm]; WGG = [inv(A, PP, JP) for A in WGGm]
    sh = (eom + ES)[np.argsort(PP)]
    we2 = we2[np.argsort(JP)]; w2abs_rows = np.bitwise_count(np.abs(M[f"mproj.{l}.w"].astype(np.int64))).sum(1)
    del Hm, outm, om, gm, WOm, WGm, WXm, WOGm, WGGm, o16m, g16m, out16m, gmir
    if LIGHT == "real":                                 # the machine as it ran: piles and the stream, bit-identical
        bad("fc_pile_vs_saved_run", (inv(pile1, PP, NP_) != REF[f"fc_pile_{l}"]).sum())
        bad("mproj_pile_vs_saved_run", (inv(pile2, PP, JP) != REF[f"mproj_pile_{l}"]).sum())
        bad("stream_vs_saved_run", (REF[f"stream_a_{l}"] + mo != REF[f"stream_{l}"]).sum())
    del pile1, pile2
    # ---- measure ----
    k2 = we2 - we2.min()                                 # each line's power of two above the layer's finest
    dark, zero, brt = H < 0, H == 0, H > 0
    r = {"edges": int(H.size), "dark": int(dark.sum()), "zero": int(zero.sum()), "bright": int(brt.sum()),
         "g0": int((g == 0).sum()), "g0_dark": int(((g == 0) & dark).sum()), "g0_bright": int(((g == 0) & brt).sum()),
         "sum_o": int(o.sum()), "sum_g": int(g.sum()), "sum_out": int(np.abs(out).sum()),
         "sum_g_dark": int(g[dark].sum()), "sum_g_bright": int(g[brt].sum()), "sum_out_bright": int(out[brt].sum()),
         "glow_mirror_equal": int((gmirror == g).sum()), "glow_mirror_maxdiff": int(np.abs(gmirror - g).max()),
         "we2_min": int(we2.min())}
    r["W_open"] = at_scale(colsum(WO), k2); r["W_glow"] = at_scale(colsum(WG), k2); r["W_out"] = at_scale(colsum(WX), k2)
    aO, aG, aX = np.abs(WO), np.abs(WG), np.abs(WX)
    r["lines"] = int(WX.size); r["lines_glow_gt_open"] = int((aG > aO).sum()); r["lines_glow_gt_out"] = int((aG > aX).sum())
    r["lines_open_gt_out"] = int((aO > aX).sum()); r["lines_out_zero"] = int((WX == 0).sum())
    exact_c = WX * (np.int64(1) << k2.astype(np.int64))[None, :]                 # exact write at 2^(we2_min) stream units
    mach_c = (mo - b2[np.argsort(JP)][None, :]).astype(object) * (1 << int(-we2.min()))   # the machine's (no bias), same scale
    r["W_machine_minus_exact"] = int(np.abs(mach_c - exact_c.astype(object)).sum())
    r["groups"] = []
    for k in range(5):
        m = BK == k
        r["groups"].append({"edges": int(m.sum()), "g0": int((m & (g == 0)).sum()), "sum_o": int(o[m].sum()),
                            "sum_g": int(g[m].sum()), "sum_out": int(np.abs(out[m]).sum()),
                            "W_open": at_scale(colsum(WOG[k]), k2), "W_glow": at_scale(colsum(WGG[k]), k2),
                            "meet_plain": int((np.bitwise_count(np.abs(out16[m])) * np.broadcast_to(w2abs_rows, m.shape)[m]).sum())})
    pr = w2abs_rows[None, :]
    r["meet_plain"] = int((np.bitwise_count(np.abs(out16)) * pr).sum())
    r["meet_open"] = int((np.bitwise_count(o16) * pr).sum()); r["meet_glow"] = int((np.bitwise_count(g16) * pr).sum())
    r["meet_plain_dark"] = int((np.bitwise_count(np.abs(out16)) * pr)[dark].sum())
    r["adds_join"] = int(P * LINES)
    RES["per_layer"][str(l)] = r
    DIG[str(l)] = {"H": sha(H), "out": sha(out), "g": sha(g), "W_open": sha(WO), "W_glow": sha(WG), "W_out": sha(WX),
                   "g16": sha(g16), "sh": sha(sh)}
    del W[f"fc.{l}"], W[f"mproj.{l}"]
    log(f"  layer {l:2d}: done ({el()}s, {mb()} MB); g=0 on {pc(r['g0'], r['edges'])} of edges; "
        f"sum g / sum o {rat(r['sum_g'], r['sum_o'])}; |W(g)| / |W(o)| {rat(r['W_glow'], r['W_open'])}; "
        f"(|W(o)|+|W(g)|) / |W(out)| {rat(r['W_open'] + r['W_glow'], r['W_out'])}")

# ---------------- report ----------------
PL = RES["per_layer"]; Ls = [str(l) for l in LAYERS]
log(); log("CHECKS (every count must be 0): " + ", ".join(f"{k} {v}" for k, v in BAD.items()))
log(f"  every check 0: {all(v == 0 for v in BAD.values())}")
log(); log("EACH EDGE: out = o - g (whole numbers at 2^-16); dark = H < 0, bright = H > 0")
log(f"{'layer':>5} {'dark':>7} {'bright':>7} {'g=0':>7} {'g=0 dark':>9} {'g=0 brt':>8} {'sum o':>16} {'sum g':>16} "
    f"{'sum |out|':>16} {'g/o':>6} {'g/|out|':>7} {'dark g':>7} {'g(-H)=g(H)':>10} {'max|diff|':>9}")
for l in Ls:
    r = PL[l]
    log(f"{l:>5} {pc(r['dark'], r['edges'], 1):>7} {pc(r['bright'], r['edges'], 1):>7} {pc(r['g0'], r['edges']):>7} "
        f"{pc(r['g0_dark'], r['edges']):>9} {pc(r['g0_bright'], r['edges']):>8} {r['sum_o']:>16,} {r['sum_g']:>16,} "
        f"{r['sum_out']:>16,} {rat(r['sum_g'], r['sum_o'], 2):>6} {rat(r['sum_g'], r['sum_out'], 2):>7} "
        f"{pc(r['sum_g_dark'], r['sum_g'], 1):>7} {pc(r['glow_mirror_equal'], r['edges']):>10} {r['glow_mirror_maxdiff']:>9}")
T = {k: sum(PL[l][k] for l in Ls) for k in ("edges", "dark", "bright", "g0", "g0_dark", "g0_bright", "sum_o", "sum_g",
                                            "sum_out", "sum_g_dark", "glow_mirror_equal")}
log(f"{'all':>5} {pc(T['dark'], T['edges'], 1):>7} {pc(T['bright'], T['edges'], 1):>7} {pc(T['g0'], T['edges']):>7} "
    f"{pc(T['g0_dark'], T['edges']):>9} {pc(T['g0_bright'], T['edges']):>8} {T['sum_o']:>16,} {T['sum_g']:>16,} "
    f"{T['sum_out']:>16,} {rat(T['sum_g'], T['sum_o'], 2):>6} {rat(T['sum_g'], T['sum_out'], 2):>7} "
    f"{pc(T['sum_g_dark'], T['sum_g'], 1):>7} {pc(T['glow_mirror_equal'], T['edges']):>10} "
    f"{max(PL[l]['glow_mirror_maxdiff'] for l in Ls):>9}")
log(f"  glow nonzero on {pc(T['edges'] - T['g0'], T['edges'])} of all {T['edges']:,} edges")

log(); log("THROUGH LENS 2 (exact, before the water level's drop): sums over lines and positions of |write|, each line at "
           "its own power of two brought to the layer's finest")
log(f"{'layer':>5} {'|W(g)|/|W(o)|':>13} {'|W(o)|/|W(out)|':>15} {'|W(g)|/|W(out)|':>15} {'(o+g)/out':>9} "
    f"{'lines g>o':>9} {'lines g>out':>11} {'lines o>out':>11} {'machine drop':>12}")
GW = {"o": 0, "g": 0, "x": 0}; gmin = min(PL[l]["we2_min"] for l in Ls)
for l in Ls:
    r = PL[l]
    log(f"{l:>5} {rat(r['W_glow'], r['W_open']):>13} {rat(r['W_open'], r['W_out']):>15} {rat(r['W_glow'], r['W_out']):>15} "
        f"{rat(r['W_open'] + r['W_glow'], r['W_out']):>9} {pc(r['lines_glow_gt_open'], r['lines'], 1):>9} "
        f"{pc(r['lines_glow_gt_out'], r['lines'], 1):>11} {pc(r['lines_open_gt_out'], r['lines'], 1):>11} "
        f"{pc(r['W_machine_minus_exact'], r['W_out'], 3):>12}")
    s = r["we2_min"] - gmin; GW["o"] += r["W_open"] << s; GW["g"] += r["W_glow"] << s; GW["x"] += r["W_out"] << s
log(f"{'all':>5} {rat(GW['g'], GW['o']):>13} {rat(GW['o'], GW['x']):>15} {rat(GW['g'], GW['x']):>15} "
    f"{rat(GW['o'] + GW['g'], GW['x']):>9}   (layers summed in stream units)")

log(); log("BY WHERE H SITS (x = H / 65536): share of the layer's edges; g = 0 within the group; the group's own open and "
           "glow writes against the layer's whole write |W(out)|")
for k in range(5):
    log(f"  {BNAMES[k]}")
    log(f"  {'layer':>5} {'edges':>7} {'g=0':>8} {'sum o':>15} {'sum g':>15} {'sum |out|':>15} {'|W(o)|/|W(out)|':>15} "
        f"{'|W(g)|/|W(out)|':>15} {'lens-2 meet':>11}")
    for l in Ls:
        r = PL[l]; q = r["groups"][k]
        log(f"  {l:>5} {pc(q['edges'], r['edges']):>7} {pc(q['g0'], q['edges']):>8} {q['sum_o']:>15,} {q['sum_g']:>15,} "
            f"{q['sum_out']:>15,} {rat(q['W_open'], r['W_out']):>15} {rat(q['W_glow'], r['W_out']):>15} "
            f"{pc(q['meet_plain'], r['meet_plain'], 1):>11}")
log("  the glow's write by group, as a share of the groups' summed |W(g)| (every layer):")
for l in Ls:
    r = PL[l]; tg = sum(q["W_glow"] for q in r["groups"])
    log(f"  {l:>5}  " + "  ".join(f"{BNAMES[k]} {pc(r['groups'][k]['W_glow'], tg, 1)}" for k in range(5)))

log(); log("WORK AT LENS 2, photon meetings (light photons x plate photons on each line), at the machine's own places")
log(f"{'layer':>5} {'plain a token':>14} {'dark side':>9} {'open a token':>13} {'glow a token':>13} {'pair/plain':>10} {'joins a token':>13}")
TW = {"p": 0, "o": 0, "g": 0, "d": 0}
for l in Ls:
    r = PL[l]; TW["p"] += r["meet_plain"]; TW["o"] += r["meet_open"]; TW["g"] += r["meet_glow"]; TW["d"] += r["meet_plain_dark"]
    log(f"{l:>5} {r['meet_plain'] // P:>14,} {pc(r['meet_plain_dark'], r['meet_plain'], 1):>9} {r['meet_open'] // P:>13,} "
        f"{r['meet_glow'] // P:>13,} {pc(r['meet_open'] + r['meet_glow'], r['meet_plain'], 1):>10} {LINES:>13,}")
log(f"{'all':>5} {TW['p'] // P:>14,} {pc(TW['d'], TW['p'], 1):>9} {TW['o'] // P:>13,} {TW['g'] // P:>13,} "
    f"{pc(TW['o'] + TW['g'], TW['p'], 1):>10} {LINES * len(Ls):>13,}")
log(f"  plain lens 2 is {pc(TW['p'], 3666596259 * P, 2)} of the whole machine (3,666,596,259 a token); the pair would "
    f"move the whole machine by {pc(TW['o'] + TW['g'] - TW['p'], 3666596259 * P, 2)} plus {LINES * len(Ls):,} joining "
    f"subtracts a token")

RES["checks"] = BAD; RES["digests"] = DIG
if ORDER:                                               # order-free: everything equal to the real run's
    ref = json.load(open(f"{GD}/gates_{LIGHT}" + ("" if "layers" not in ARG else "_L" + "-".join(map(str, LAYERS))) + ".json"))
    same_d = sum(ref["digests"][l][k] == DIG[l][k] for l in Ls for k in DIG[l]); nd = sum(len(DIG[l]) for l in Ls)
    same_t = sum(ref["per_layer"][l] == PL[l] for l in Ls)
    log(); log(f"ORDER-FREE: {same_d} of {nd} array digests and {same_t} of {len(Ls)} layers' totals equal the "
               f"{LIGHT} run in the machine's own order")
    RES["order_free"] = {"digests_same": same_d, "digests": nd, "layers_same": same_t}
json.dump(RES, open(f"{GD}/gates_{TAG}.json", "w"), indent=1)
log(); log(f"saved gates/gates_{TAG}.json ({el()}s, max memory {mb()} MB)")
