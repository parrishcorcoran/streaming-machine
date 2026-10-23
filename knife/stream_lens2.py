"""LENS 2 AND THE WATER LEVELS: DOES THE STREAM FORGIVE LENS 1'S LAST DIAGONALS?  A MEASUREMENT ONLY; NOTHING IN THE
MACHINE CHANGES.

THE PICTURE, restated. Each layer's middle is a Schlieren / 4f system. The neurons-in plate fc.l is lens 1: it forms the
3,072 piles, the spectrum. The bends are Gaussian knife edges in that focal plane. The neurons-out plate mproj.l is lens
2: it gathers the 3,072 edges' outputs back onto the stream's 768 lines. knife/settle.py counted lens 1 diagonal by
diagonal from the top with exact bounds and found each edge's side fixed by the top 12-16 of 29 diagonals, but its
output (the 16 places lens 2 meets) only at D24-D28. knife/levels.py then read that lens 2's pile drops most of its own
low places when it joins the stream (median 11-13 places), and the next light keeps only the stream's top 16 places
(the next water level drops 6-9 more). So the question is whether the low places the edges are still unsure of are
places the stream throws away anyway: is the value lens 2 adds to the stream (and the next light) fixed before the edges
are?

THE MACHINE (base2.py, unchanged; its own functions are exec'd and called, as knife/settle.py does):
  H = gauged_bank(R, fc.l)              each edge's arrival, at 2^-16
  out = edge(H)                         the bend's output
  out16, eo = to16(out, -16)            sh = eo + 16, one per position, from the 3,072 outputs
  mo = plain_bank(out, mproj.l)         = rshift(product(out16, w2), -(we2 + sh)) + b2    THE VALUE THAT JOINS THE STREAM
  Rn = R + mo                           the stream after the layer
  xn, en = to16(Rn, -16)                THE NEXT LIGHT: the 16 places the next plate meets (sn = en + 16 dropped)

THE TEST, at every position of the 30 refs texts (648) and every layer, at each stage of lens 1 fixed in advance (the
diagonals from the top: D2, D4, ..., D28, all; the same stages as settle.py, so its output stages can be checked here):
  1. each edge's H interval from the signed bound (settle.py's own counting and bound, rebuilt here);
  2. each edge's output interval: the exact least and greatest out over every H in the interval (the edge's table read
     everywhere: a range read over every H the table covers, dark below it, straight above it);
  3. the position's sh: every value its 3,072 output intervals allow; for each one, out16 = rshift(out, sh) as an
     interval (at most 2^15 in size, which that sh itself guarantees);
  4. lens 2 met with intervals, exactly: centre lo + hi through the plate, width hi - lo through the plate's sizes;
     2 x pile lies in [centre.w - width.|w|, centre.w + width.|w|], the exact least and greatest pile (whole numbers,
     every sum below 2^44);
  5. the value that joins the stream: rshift(pile, -(we2 + sh)) + b2 at both ends, joined over every allowed sh.
     FIXED when the joined interval is one number;
  6. the next light: R + that interval; every next water level it allows; rshift at both ends, joined. A line FIXED
     when one number; the position's next light FIXED when all 768 lines are and the water level is one value.
  Every interval only shrinks from stage to stage (the bounds are sums over uncounted place pairs), so being fixed only
  switches on; the stages are walked from the end and a position is left once none of its lines is fixed any more.
TWO WAYS LENS 1 COULD STOP:
  (A) all 3,072 edges of a position are counted to the same stage; lens 1 stops at the first stage at which all 768
      lines of that position are fixed.
  (B) each edge also stops by itself once its own output is fixed (settle.py's freeze, exactly: its stages are checked
      against knife/stages_real.npz); the frozen edge carries its frozen interval into lens 2; lens 1 stops for every
      edge of the position once all 768 lines are fixed. (B) against settle.py's per-edge stops (91.8% of the fc
      meeting) is what lens 2 adds.
WORK, in the unit of all earlier results: photon meetings = light photons x plate photons on each line.
  lens 1: the meetings of the place pairs counted up to the stage each edge stops at, against the full fc meeting.
  lens 2 plain: photons of |out16| x the plate's photons on that line, summed (the machine's own lens-2 meeting).
  a push of intervals through lens 2: photons of |lo + hi| and of (hi - lo) on each line, each times the plate's
    photons on that line, summed over every allowed sh. A fixed edge has hi = lo, so its share is its plain meeting;
    the push costs at least the lens-2 meeting, and more on every edge still unsure. (Pushing lo and hi separately
    would cost twice the lens-2 meeting.)
  Two push rules: (i) one push, at the stop stage (the floor: as if the stop stage were known in advance);
    (ii) a push at every stage at which the position's sh is one value, until the stop (a machine that does not know
    the stop stage in advance). The push at the full pile is the lens-2 meeting itself; only the extra is counted.
  The bound's own table reads (60 per edge per stage, signed) are stated apart, as in settle.py.
  Whole machine: 3,666,596,259 photon meetings a token (FINDINGS.md); the fc meeting is 895,020,740 of it.
CHECKS (every position): settle.py's pieces add to the machine's pile and to the saved run's fc pile; the exact pile, H,
out, sh, the lens-2 pile, mo, sn and the next light inside every stage's intervals; every fixed value equal to the
machine's own; the stream rebuilt equals the saved run's stream after the layer; everything fixed at the full pile;
fixed never switching off; (B)'s edge stops identical to settle.py's; the lens-2 push at the full pile equal to the plain
lens-2 meeting; a sampled rebuild of the lens-2 intervals by other arithmetic (each edge's out16 read from the table one
H at a time; the pile ends as sums of the least and greatest products, in plain integers).
CHECK ON 3 TEXTS (light=real): base2.py's own forward on the first 3 refs texts (62 positions, every bank, the machine
unchanged): its stream entering fc.l, its mo and its next light, against the values used here, and every fixed value of
those 62 positions against the forward's own.
CONTROL: light=shuffled shuffles each position's light across its 768 lines (every value kept; settle.py's seeds) and
measures everything again on what the machine's own functions do with that light.
ORDER-FREE: order=1 permutes lens 1's lines, the edges (lens 2's lines), lens 2's output lines and the order the place
pairs are added; every stage of every line and position must be the same (sha256 per layer against the real run's).

GATING (holographicplate CLAUDE.md, main): it measures how far the exposure of lens 1 must go before what lens 2 passes
to the stream is settled (a settling, an exposure); every read is on whole numbers (the float containers hold whole
numbers, every sum below 2^44, exact; the pieces as in settle.py); it is exact (bounds only, nothing approximated); it
discards nothing (a measurement only; nothing in the machine changes); no fit and no word-count comparison; not measured
before (settle.py stopped at the edges' outputs; levels.py only read the shifts).

PREDICTIONS (written before this script first ran):
  the task's: the 3,072 interval widths add up in lens 2, so the stream value is fixed only a few diagonals before the
    end; lens 1 saves 10-25% at most, and the interval push through lens 2 costs about as much as it saves, so the net
    is near zero unless the widths are much smaller than the worst case.
  the session's, from settle.py's numbers: at D28 about 1% of the edges (about 31 per position) still have an out16
    interval of width 1, at D26 about 15% (about 455), at D24 about 58%. Lens 2 gathers each still-unsure edge with its
    plate value (typically 2^10-2^11 of the plate's 15 places) and the stream then drops 11-13 places, so a line's width
    is about 31 x 2^10.5 / 2^12 = 8-20 stream units at D28 and 100-300 at D26, where a fixed value needs under 1.
    - stream value: fixed before the full pile for under 1% of lines in every layer, in (A) and (B); no position has
      all 768 fixed before the full pile (0 of 648 in each layer).
    - next light, per line: the next water level drops sn = 6-9 places (64-512 units), so a width of 8-20 crosses a
      rounding step for roughly width / 2^sn of the lines: fixed at D28 for about 70-90% of lines in layers 0-8 and
      90-98% in layers 9-11; at D26 under 30% in layers 0-8 and up to about 60% in layer 11; by D24 under 10%.
    - next light, per position (all 768 lines and the water level): under 1% of positions before the full pile, since
      every one of 768 lines must miss its rounding step.
    - lens 1 needed when it stops at the position's stop stage: (A) 99.5-100% of the fc meeting; (B) the same as
      settle.py's per-edge output stops (91.8%) to within 0.1 point. The low diagonals hold little (D28 leaves 1 of 225
      place pairs, about 0.7% of the meetings; D26 6 pairs, about 4%), so even a stop at D26 everywhere would save
      only about 4%.
    - the push: lens 2's plain meeting a layer is about the size of lens 1's (both 768 x 3,072 plate values); a failed
      push costs about one lens-2 meeting or more. (i) one push at the stop stage: about 0 extra, about 0 saved, net
      0.0% of the whole machine. (ii) a push at every stage with one water level: 3-8 failed pushes a layer before the
      full pile, adding 100-300% of the whole machine's work.
    - control: the same within a few points (the widths come from how many edges are unsure and the plate's sizes).
    - order-free: every stage identical.
    - against the task's prediction: "fixed only a few diagonals before the end" holds for the next light's lines but
      not for the stream value; "lens 1 saves 10-25% at most" holds only as an upper limit (it saves under 1%); the net
      is not near zero under (ii) but far below it.

    python3 knife/stream_lens2.py [light=real|shuffled] [order=0|1] [layers=0,1,...]
      -> knife/stream_lens2_<light>[_order]_L<layers>.log, and knife/stream_lens2_<light>[_order]_L<l>.npz per layer
"""
import sys, os, time, json, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
LIGHT, ORDER = ARG.get("light", "real"), int(ARG.get("order", 0))
LAYERS = [int(v) for v in ARG["layers"].split(",")] if "layers" in ARG else list(range(12))
TAG = LIGHT + ("_order" if ORDER else "")
LOG = open(f"{HERE}/knife/stream_lens2_{TAG}_L{'-'.join(map(str, LAYERS))}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def el(): return int(time.time() - t0)

_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]     # base2.py up to its own test
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')        # W filled one bank at a time, by base2's own line
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W, ES, PHI, EXP2 = J["M"], J["W"], J["ES"], J["PHI"], J["EXP2"]
rshift, power, edge, to16, gauge_log, gauged_bank, product, plain_bank = (J[k] for k in
    ("rshift", "power", "edge", "to16", "gauge_log", "gauged_bank", "product", "plain_bank"))
def load(name): W[name] = (M[f"{name}.w"].astype(np.float64), M[f"{name}.e"], M[f"{name}.b"])   # base2's own line
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
P, N, LINES, PL, PW = int(REF["npos"].sum()), 3072, 768, 15, 15
ND = PL + PW - 1
assert P == 648
LIM = 1 << 15                                          # |out16| and |next light| never above 2^15 for their own sh
BIG = np.int64(1) << 62

# ---------------- the edge, read from the machine itself (settle.py's lines) ----------------
def share_of(H):
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
assert (np.diff(PHI) >= 0).all() and PHI.min() >= 0 and PHI.max() <= 65536
HA, HB = -(1 << 19) - 64, (1 << 19) + 64
GRID = np.arange(HA, HB + 1, dtype=np.int64); OUTT = edge(GRID).astype(np.int64)
assert np.array_equal(rshift(GRID * share_of(GRID), 16), OUTT) and OUTT[0] == 0 and OUTT[-1] == HB
_pos = GRID >= 0; assert (OUTT[_pos] >= 0).all() and (np.diff(OUTT[_pos]) >= 0).all()
NEGMAX = int(np.abs(OUTT[~_pos]).max())
_probe = np.concatenate([np.arange(HA - 4096, HA), -(np.int64(1) << np.arange(20, 44)),
                         np.arange(HB + 1, HB + 4096), np.int64(1) << np.arange(20, 44)])
assert np.array_equal(edge(_probe), np.where(_probe < HA, 0, _probe))                         # dark below, straight above
SMT = 21
CP = np.zeros((SMT + 1, len(GRID)), np.int32)
for _s in range(SMT + 1):
    _f = rshift(OUTT, _s); CP[_s, 1:] = np.cumsum(_f[1:] != _f[:-1])
assert not rshift(OUTT, SMT).any()
def out_ext(H): return np.where(H < HA, 0, np.where(H > HB, H, OUTT[np.clip(H - HA, 0, len(GRID) - 1)]))
def constant(lo, hi, s):
    v = rshift(out_ext(lo), s); ok = v == rshift(out_ext(hi), s)
    sc = np.minimum(s, SMT); g0 = np.clip(lo, HA, HB) - HA; g1 = np.clip(hi, HA, HB) - HA
    ok &= (hi < HA) | (lo > HB) | (CP[sc, g1] == CP[sc, g0])
    ok &= (hi <= HB) | (rshift(np.maximum(lo, HB), s) == rshift(hi, s))
    return ok, v
def bitlen(v): return np.array([int(x).bit_length() for x in v], np.int64)
def sh_range(lo, hi):                                  # settle.py's: used only to rebuild its per-edge stops exactly
    U = np.maximum(np.where(hi >= 0, out_ext(np.maximum(hi, 0)), 0), np.where(lo < 0, NEGMAX, 0)).max(1)
    L = np.where(lo >= 0, out_ext(np.maximum(lo, 0)), 0).max(1)
    return np.maximum(bitlen(L) - 15, 0), np.maximum(bitlen(U) - 15, 0)
def out_settled(lo, hi, slo, shi):
    ok, v = constant(lo, hi, slo)
    for d in range(1, int((shi - slo).max(initial=0)) + 1):
        m = ok & (slo + d <= shi)
        if not m.any(): break
        o2, v2 = constant(lo[m], hi[m], slo[m] + d); ok[m] = o2 & (v2 == v[m])
    return ok, v

# ---- the least and greatest output over every H in an interval: a range read of the edge's whole table ----
G = len(GRID); KT = G.bit_length()
TMIN = np.empty((KT, G), np.int32); TMAX = np.empty((KT, G), np.int32); TMIN[0] = OUTT; TMAX[0] = OUTT
for _k in range(1, KT):
    _h = 1 << (_k - 1)
    TMIN[_k, :G - _h] = np.minimum(TMIN[_k - 1, :G - _h], TMIN[_k - 1, _h:]); TMIN[_k, G - _h:] = TMIN[_k - 1, G - _h:]
    TMAX[_k, :G - _h] = np.maximum(TMAX[_k - 1, :G - _h], TMAX[_k - 1, _h:]); TMAX[_k, G - _h:] = TMAX[_k - 1, G - _h:]
FLOORLOG = np.zeros(G + 1, np.int64)
for _k in range(KT): FLOORLOG[1 << _k: min(1 << (_k + 1), G + 1)] = _k
def orange(lo, hi):
    mn = np.where(lo < HA, 0, BIG); mx = np.where(lo < HA, 0, -BIG)
    g0 = np.clip(lo, HA, HB) - HA; g1 = np.clip(hi, HA, HB) - HA; ing = (hi >= HA) & (lo <= HB)
    k = FLOORLOG[np.maximum(g1 - g0 + 1, 1)]; j = np.maximum(g1 - (np.int64(1) << k) + 1, 0)
    mn = np.where(ing, np.minimum(mn, np.minimum(TMIN[k, g0], TMIN[k, j])), mn)
    mx = np.where(ing, np.maximum(mx, np.maximum(TMAX[k, g0], TMAX[k, j])), mx)
    ab = hi > HB
    mn = np.where(ab, np.minimum(mn, np.maximum(lo, HB + 1)), mn); mx = np.where(ab, np.maximum(mx, hi), mx)
    return mn, mx
_rng = np.random.default_rng(1)
for _ in range(200):                                   # the range read against reading every H, on random intervals
    a = int(_rng.integers(HA - 3000, HB + 3000)); b = a + int(_rng.integers(0, 1 << int(_rng.integers(1, 21))))
    seg = edge(np.arange(a, b + 1, dtype=np.int64)); m1, m2 = orange(np.array([a]), np.array([b]))
    assert int(m1[0]) == int(seg.min()) and int(m2[0]) == int(seg.max())

def bitlen_v(v):                                       # the top place + 1 of each whole number >= 0, by smearing
    v = v.astype(np.int64).copy()
    for s in (1, 2, 4, 8, 16, 32): v |= v >> s
    return np.bitwise_count(v).astype(np.int64)
def sh_exact(lo, hi):                                  # (rows, k) intervals -> every to16 shift they allow: lowest, highest
    minabs = np.where((lo <= 0) & (hi >= 0), 0, np.minimum(np.abs(lo), np.abs(hi)))
    maxabs = np.maximum(np.abs(lo), np.abs(hi))
    return np.maximum(bitlen_v(minabs.max(1)) - 15, 0), np.maximum(bitlen_v(maxabs.max(1)) - 15, 0)
def popc(v): return np.bitwise_count(np.abs(v)).astype(np.float64)

KD = list(range(2, 29, 2)) + [ND]
_A, _B = np.arange(PL)[:, None], np.arange(PW)[None, :]
STAGES = [("all" if k == ND else f"D{k}", (_A + _B) < ND - k) for k in KD]; NS = len(STAGES); NAMES = [n for n, _ in STAGES]
WIDE = [NS - 2, NS - 3, NS - 4]                         # D28, D26, D24: every position measured, for the widths
VN = ["A", "B"]
BAD = {}; CHK = {}
def bad(k, n): BAD[k] = BAD.get(k, 0) + int(n)
def pm(a, b): q = (1000 * int(a) + int(b) // 2) // int(b) if b else 0; return f"{q // 10}.{q % 10}%"
WHOLE = 3666596259

# ---------------- CHECK ON 3 TEXTS: base2.py's own forward ----------------
TEXTS = [r["ids"] + r["ref"] for r in json.load(open(f"{HERE}/refs_many.json"))[:3]]
P3 = sum(len(t) for t in TEXTS); assert P3 == int(REF["npos"][:3].sum())
FWD = {}
if LIGHT == "real":
    Rs = [M["wte"][ids] + M["wpe"][:len(ids)] for ids in TEXTS]
    for l in range(max(LAYERS) + 1):
        for b in ("qkv", "proj", "fc", "mproj"): load(f"{b}.{l}")
        RA3, MO3 = [], []
        for t in range(3):
            R = Rs[t]
            R = R + plain_bank(J["attention"](gauged_bank(R, f"qkv.{l}")), f"proj.{l}")        # base2's forward, line by line
            RA3.append(R.copy())
            mo = plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
            R = R + mo; MO3.append(mo); Rs[t] = R
        if l in LAYERS:
            RA3, MO3 = np.concatenate(RA3), np.concatenate(MO3); xn3, en3 = to16(RA3 + MO3, -ES)
            FWD[l] = (RA3, MO3, xn3, en3 + ES)
        for b in ("qkv", "proj", "fc", "mproj"): del W[f"{b}.{l}"]
    log(f"base2's own forward on 3 texts ({P3} positions) through layer {max(LAYERS)} ({el()}s, {mb()} MB)")

PREV = np.load(f"{HERE}/knife/settle_{'shuffled' if LIGHT == 'shuffled' else 'real'}.npz")
assert list(PREV["layers"]) == list(range(12))
STPREV = np.load(f"{HERE}/knife/stages_real.npz") if LIGHT == "real" else None

CHUNKS = [(0, 216), (216, 432), (432, 648)]
log(f"lens 2: light={LIGHT} order={ORDER} layers={LAYERS}; stages {NAMES}")
TOTALS = {}
for l in LAYERS:
    load(f"fc.{l}"); load(f"mproj.{l}")
    R = REF[f"stream_a_{l}"].astype(np.int64)
    if LIGHT == "shuffled":                              # CONTROL: each position's light across its lines (settle's seeds)
        rng = np.random.default_rng(2026 + l); R = np.stack([r[rng.permutation(LINES)] for r in R])
    w16 = M[f"fc.{l}.w"]; we, bias = M[f"fc.{l}.e"], M[f"fc.{l}.b"]
    w2, we2, b2 = M[f"mproj.{l}.w"], M[f"mproj.{l}.e"], M[f"mproj.{l}.b"]
    # ---- the machine's own, in its own order ----
    x16_0, ex, LS = gauge_log(R)
    pile_x = product(x16_0, W[f"fc.{l}"][0])
    if LIGHT == "real": bad("pile_vs_saved_run", (pile_x != REF[f"fc_pile_{l}"]).sum())
    Hx = gauged_bank(R, f"fc.{l}")
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    bad("H_vs_power", (power(pile_x, F) + bias != Hx).sum())
    outx = edge(Hx); o16x, eo = to16(outx, -ES); shx = eo + ES
    pile2x = product(o16x, W[f"mproj.{l}"][0])
    mox = plain_bank(outx, f"mproj.{l}")                 # the value that joins the stream, base2's own
    bad("mo_vs_pile2", (rshift(pile2x, -(we2[None, :] + eo[:, None] + ES)) + b2 != mox).sum())
    Rn = R + mox
    if LIGHT == "real": bad("stream_after_vs_saved_run", (Rn != REF[f"stream_{l}"]).sum())
    xnx, enx = to16(Rn, -ES); snx = enx + ES
    if l in FWD:
        RA3, MO3, xn3, sn3 = FWD[l]
        bad("3texts_stream_in", (RA3 != REF[f"stream_a_{l}"][:P3]).sum()); bad("3texts_mo", (MO3 != mox[:P3]).sum())
        bad("3texts_next_light", (xn3 != xnx[:P3]).sum() + (sn3 != snx[:P3]).sum())
    del W[f"fc.{l}"], W[f"mproj.{l}"]
    # ---- order ----
    LP, NP_, OP_ = np.arange(LINES), np.arange(N), np.arange(LINES)
    if ORDER:
        rng = np.random.default_rng(777 + l); LP = rng.permutation(LINES); NP_ = rng.permutation(N)
        OP_ = np.random.default_rng(4242 + l).permutation(LINES)
        x16, ex2, LS2 = gauge_log(R[:, LP]); assert np.array_equal(ex2, ex) and np.array_equal(LS2, LS)
        wl = w16[LP][:, NP_]
    else:
        x16, wl = x16_0, w16
    Fp, bp = F[:, NP_], bias[NP_]
    Hx, outx, o16x, pile_x = Hx[:, NP_], outx[:, NP_], o16x[:, NP_], pile_x[:, NP_]
    w2p = w2[NP_][:, OP_]; we2p, b2p = we2[OP_], b2[OP_]
    WM = w2p.astype(np.float64); AWM = np.abs(WM)
    PWm = np.bitwise_count(np.abs(w2p.astype(np.int64))).sum(1).astype(np.float64)   # the plate's photons on each edge's line
    RA, mox_p, pile2p, xnp = R[:, OP_], mox[:, OP_], pile2x[:, OP_], xnx[:, OP_]
    fwd = (FWD[l][1][:, OP_], FWD[l][2][:, OP_]) if l in FWD else None
    ax, sx = np.abs(x16), np.sign(x16); aw = np.abs(wl.astype(np.int64)); sw = np.sign(wl).astype(np.int64)
    assert ax.max() < 1 << PL and aw.max() < 1 << PW
    bw = ((aw[None] >> np.arange(PW)[:, None, None]) & 1).astype(np.uint8)
    mP = (bw * (sw > 0)).sum(1, dtype=np.int64); mM = (bw * (sw < 0)).sum(1, dtype=np.int64); del bw
    pwc = np.bitwise_count(aw).astype(np.float64)
    pairs = [(a, b) for b in range(PW) for a in range(PL)]
    if ORDER: pairs = [pairs[i] for i in np.random.default_rng(55 + l).permutation(len(pairs))]

    OUT = {f"{k}_{v}": np.full((P, LINES) if k in ("FM", "FX") else (P,), -1, np.int8)
           for k in ("FM", "FX", "FPM", "FPX") for v in VN}
    FRZALL = np.full((P, N), -1, np.int8)
    T = dict(full=0, l2=0, edge_only=0, nblur=np.zeros((2, NS), np.int64), nsingle=np.zeros((2, NS), np.int64))
    for v in VN:
        for r in ("m", "x"):
            for k in ("l1", "reads", "push1", "push2", "l1_ii", "npush2", "push3", "l1_iii"): T[f"{k}_{v}_{r}"] = 0
    # a read of the plate: how far one unit of an edge's output reaches into the stream. One unit of out16 on edge i
    # moves lens 2's pile on line j by w2[i, j], and the stream by w2[i, j] / 2^(-(we2[j] + sh)): bucket k = top place
    # of |w2[i, j]| minus the places dropped, at the layer's middle sh (k = 0: under 1 unit; k: 2^(k-1) .. 2^k units)
    shmid = int(np.sort(shx)[P // 2]); dropm = -(we2[None, :] + shmid)
    T["reach"] = np.bincount(np.clip(bitlen_v(np.abs(w2.astype(np.int64))) - dropm, 0, 31).ravel(), minlength=32)[:32]
    WID = {(v, s, w): np.zeros(64, np.int64) for v in VN for s in WIDE for w in ("m", "x")}

    for r0, r1 in CHUNKS:
        Pc = r1 - r0; axc, sxc = ax[r0:r1], sx[r0:r1]; ar = np.arange(Pc)
        Hc, Fc, o16c, outc, pilec, shc = Hx[r0:r1], Fp[r0:r1], o16x[r0:r1], outx[r0:r1], pile_x[r0:r1], shx[r0:r1]
        RAc, moc, pile2c, xnc, snc = RA[r0:r1], mox_p[r0:r1], pile2p[r0:r1], xnp[r0:r1], snx[r0:r1]
        # ---- lens 1's pieces: every (light place, plate place) pair's balance and meetings, by diagonal ----
        D = np.zeros((ND, Pc, N), np.int16); MD = np.zeros((ND, Pc, N), np.int16)
        Xs = [(sxc * ((axc >> a) & 1)).astype(np.float32) for a in range(PL)]
        Xu = [((axc >> a) & 1).astype(np.float32) for a in range(PL)]
        cb = -1
        for a, b in pairs:
            if b != cb: Wbit = (aw >> b) & 1; Ws = (sw * Wbit).astype(np.float32); Wu = Wbit.astype(np.float32); cb = b
            D[a + b] += (Xs[a] @ Ws).astype(np.int16); MD[a + b] += (Xu[a] @ Wu).astype(np.int16)   # |sums| <= 768
        del Xs, Xu, Ws, Wu, Wbit
        pile = np.zeros((Pc, N), np.int64)
        for j in range(ND): pile += D[j].astype(np.int64) << j
        bad("diagonals_vs_pile", (pile != pilec).sum()); del pile
        fullm = (np.bitwise_count(axc).astype(np.float64) @ pwc).astype(np.int64)
        bad("meetings_vs_plain", (MD.sum(0, dtype=np.int64) != fullm).sum()); T["full"] += int(fullm.sum())
        L2P = (popc(o16c) @ PWm).astype(np.int64); T["l2"] += int(L2P.sum())
        # ---- the signed bound (settle.py's) ----
        bx = ((axc[None] >> np.arange(PL)[:, None, None]) & 1)
        nP, nM = (bx * (sxc > 0)).sum(2), (bx * (sxc < 0)).sum(2); del bx
        KK = np.arange(LINES + 1)
        MN = {k: np.minimum(v[:, :, None], KK[None, None, :]).reshape(PL, -1).astype(np.float64) for k, v in (("p", nP), ("m", nM))}
        def gtab(U, k):
            C = (U.T * (1 << np.arange(PL))[None, :]).astype(np.float64)
            Gt = C @ MN[k]; assert Gt.max(initial=0) < 2 ** 50; return Gt.astype(np.int64).reshape(PW, Pc, LINES + 1)
        def gsum(Gt, mm):
            o = np.zeros((Pc, N), np.int64)
            for b in range(PW): o += np.take(Gt[b], mm[b], axis=1) << b
            return o
        # ---- forward over the stages: H intervals, meetings counted, and settle.py's per-edge output stops ----
        HLO = np.zeros((NS, Pc, N), np.int64); HHI = np.zeros((NS, Pc, N), np.int64); CM = np.zeros((NS, Pc, N), np.int32)
        c = np.zeros((Pc, N), np.int64); cm = np.zeros((Pc, N), np.int32); counted = np.zeros((PL, PW), bool)
        FRZ = np.full((Pc, N), -1, np.int8); FLO = np.zeros((Pc, N), np.int64); FHI = np.zeros((Pc, N), np.int64)
        for s, (name, U) in enumerate(STAGES):
            newp = ~U & ~counted; counted |= newp
            for j in sorted(set((_A + _B)[newp].tolist())): c += D[j].astype(np.int64) << j; cm += MD[j]
            if U.any():
                Gp, Gm = gtab(U, "p"), gtab(U, "m"); up = gsum(Gp, mP) + gsum(Gm, mM); dn = gsum(Gp, mM) + gsum(Gm, mP)
                del Gp, Gm
            else:
                up = dn = np.zeros((Pc, N), np.int64)
            plo, phi = c - dn, c + up
            bad("pile_outside_bound", ((pilec < plo) | (pilec > phi)).sum())
            assert int(np.abs(plo).max()) < 1 << 45 and int(np.abs(phi).max()) < 1 << 45
            Hlo, Hhi = power(plo, Fc) + bp, power(phi, Fc) + bp
            bad("H_outside_bound", ((Hc < Hlo) | (Hc > Hhi)).sum())
            HLO[s], HHI[s], CM[s] = Hlo, Hhi, cm
            fr = FRZ >= 0                                # (B): settle.py's freeze, line for line
            LOc, HIc = np.where(fr, FLO, Hlo), np.where(fr, FHI, Hhi)
            slo, shi = sh_range(LOc, HIc)
            idx = np.flatnonzero(~fr.ravel())
            okc, vv = out_settled(Hlo.ravel()[idx], Hhi.ravel()[idx], np.repeat(slo, N)[idx], np.repeat(shi, N)[idx])
            new = idx[okc]; bad("edge_output_wrong", (vv[okc] != o16c.ravel()[new]).sum())
            FRZ.reshape(-1)[new] = s; FLO.reshape(-1)[new] = Hlo.ravel()[new]; FHI.reshape(-1)[new] = Hhi.ravel()[new]
        del D, MD, MN, c, cm, up, dn
        bad("edge_unsettled_at_all", (FRZ < 0).sum()); FRZALL[r0:r1] = FRZ
        bad("full_count_vs_plain", (CM[NS - 1] != fullm).sum())
        eo_work = CM[FRZ.astype(np.int64), ar[:, None], np.arange(N)[None, :]].sum(1, dtype=np.int64); T["edge_only"] += int(eo_work.sum())

        # ---- lens 2, walked from the end ----
        for vi, v in enumerate(VN):
            FMs = np.full((Pc, LINES), -1, np.int8); FXs = np.full((Pc, LINES), -1, np.int8)
            stm = np.ones((Pc, LINES), bool); stx = np.ones((Pc, LINES), bool)
            FPMs = np.full(Pc, -1, np.int8); FPXs = np.full(Pc, -1, np.int8); stpm = np.ones(Pc, bool); stpx = np.ones(Pc, bool)
            SINGLE = np.zeros((NS, Pc), bool); COST1 = np.zeros((NS, Pc), np.int64); COSTA = np.full((NS, Pc), -1, np.int64)
            for s in range(NS - 1, -1, -1):
                if vi == 0: Hl, Hh = HLO[s], HHI[s]
                else:
                    fz = (FRZ >= 0) & (FRZ <= s); Hl, Hh = np.where(fz, FLO, HLO[s]), np.where(fz, FHI, HHI[s])
                omin, omax = orange(Hl, Hh)
                bad("out_outside", ((outc < omin) | (outc > omax)).sum())
                slo, shi = sh_exact(omin, omax)
                bad("sh_outside", ((shc < slo) | (shc > shi)).sum())
                one = slo == shi; SINGLE[s] = one
                if one.any():                            # rule (ii): the cost of a push wherever sh is one value
                    t = slo[one][:, None]
                    lo16 = np.maximum(rshift(omin[one], t), -LIM); hi16 = np.minimum(rshift(omax[one], t), LIM)
                    COST1[s, one] = ((popc(lo16 + hi16) + popc(hi16 - lo16)) @ PWm).astype(np.int64)
                    T["nblur"][vi, s] += int((lo16 != hi16).sum()); T["nsingle"][vi, s] += int(one.sum())
                alive = stm.any(1) | stx.any(1) | (s in WIDE)
                rows = np.flatnonzero(alive)
                if rows.size == 0: continue
                nr = rows.size
                MOL = np.full((nr, LINES), BIG); MOH = np.full((nr, LINES), -BIG); costa = np.zeros(nr, np.int64)
                ncand = np.zeros(nr, np.int64)
                rl, rh = slo[rows], shi[rows]
                for d in range(int((rh - rl).max()) + 1):
                    k = np.flatnonzero(rl + d <= rh); t = (rl[k] + d)[:, None]; pr = rows[k]
                    lo16 = rshift(omin[pr], t); hi16 = rshift(omax[pr], t)
                    poss = ~((lo16 > LIM) | (hi16 < -LIM)).any(1)
                    k, t, pr, lo16, hi16 = k[poss], t[poss], pr[poss], lo16[poss], hi16[poss]
                    if k.size == 0: continue
                    lo16 = np.maximum(lo16, -LIM); hi16 = np.minimum(hi16, LIM)
                    C = lo16 + hi16; Rr = hi16 - lo16
                    PC = C.astype(np.float64) @ WM; PR = Rr.astype(np.float64) @ AWM
                    assert np.abs(PC).max() < 2 ** 52 and PR.max(initial=0) < 2 ** 52
                    PCi, PRi = PC.astype(np.int64), PR.astype(np.int64)
                    p2lo, p2hi = PCi - PRi, PCi + PRi; bad("pile2_not_even", ((p2lo & 1) != 0).sum())
                    plo2, phi2 = p2lo >> 1, p2hi >> 1
                    sft = -(we2p[None, :] + t)
                    mol, moh = rshift(plo2, sft) + b2p, rshift(phi2, sft) + b2p
                    tr = t[:, 0] == shc[pr]                  # the true sh: the true lens-2 pile and value must be inside
                    bad("pile2_outside", ((pile2c[pr[tr]] < plo2[tr]) | (pile2c[pr[tr]] > phi2[tr])).sum())
                    bad("mo_outside_at_true_sh", ((moc[pr[tr]] < mol[tr]) | (moc[pr[tr]] > moh[tr])).sum())
                    MOL[k] = np.minimum(MOL[k], mol); MOH[k] = np.maximum(MOH[k], moh); ncand[k] += 1
                    costa[k] += ((popc(C) + popc(Rr)) @ PWm).astype(np.int64)
                bad("no_possible_sh", (ncand == 0).sum())
                COSTA[s, rows] = costa
                fm = MOL == MOH
                bad("mo_outside", ((moc[rows] < MOL) | (moc[rows] > MOH)).sum())
                bad("mo_fixed_wrong", (fm & (MOL != moc[rows])).sum())
                # the next light
                Rlo, Rhi = RAc[rows] + MOL, RAc[rows] + MOH
                nlo, nhi = sh_exact(Rlo, Rhi)
                bad("sn_outside", ((snc[rows] < nlo) | (snc[rows] > nhi)).sum())
                XL = np.full((nr, LINES), BIG); XH = np.full((nr, LINES), -BIG); nposs = np.zeros(nr, np.int64)
                for d in range(int((nhi - nlo).max()) + 1):
                    k = np.flatnonzero(nlo + d <= nhi); t = (nlo[k] + d)[:, None]
                    xl, xh = rshift(Rlo[k], t), rshift(Rhi[k], t)
                    poss = ~((xl > LIM) | (xh < -LIM)).any(1); k, xl, xh = k[poss], xl[poss], xh[poss]
                    XL[k] = np.minimum(XL[k], np.maximum(xl, -LIM)); XH[k] = np.maximum(XH[k], np.minimum(xh, LIM)); nposs[k] += 1
                bad("no_possible_sn", (nposs == 0).sum())
                fx = XL == XH
                bad("next_light_outside", ((xnc[rows] < XL) | (xnc[rows] > XH)).sum())
                bad("next_light_fixed_wrong", (fx & (XL != xnc[rows])).sum())
                if fwd is not None:                      # CHECK ON 3 TEXTS: against base2's own forward
                    g = rows + r0; q = g < P3
                    if q.any():
                        bad("3texts_mo_fixed_vs_forward", (fm[q] & (MOL[q] != fwd[0][g[q]])).sum())
                        bad("3texts_next_light_fixed_vs_forward", (fx[q] & (XL[q] != fwd[1][g[q]])).sum())
                        CHK["3texts_fixed_values"] = CHK.get("3texts_fixed_values", 0) + int(fm[q].sum() + fx[q].sum())
                if s in WIDE:
                    for w, (lo_, hi_) in (("m", (MOL, MOH)), ("x", (XL, XH))):
                        WID[(v, s, w)] += np.bincount(bitlen_v(hi_ - lo_).ravel(), minlength=64)[:64]
                if vi == 0 and r0 == 0 and s in (NS - 2, NS - 3):   # sampled rebuild by other arithmetic
                    w2i = w2p.astype(np.int64)
                    for p in rows[:2]:
                        kk = int(np.flatnonzero(rows == p)[0]); lo_m, hi_m = None, None
                        for tt in range(int(slo[p]), int(shi[p]) + 1):
                            e16 = []
                            for e in range(N):
                                a_, b_ = int(Hl[p, e]), int(Hh[p, e]); vs = []
                                if a_ < HA: vs.append(0)
                                if b_ >= HA and a_ <= HB:
                                    rr = rshift(OUTT[max(a_, HA) - HA: min(b_, HB) - HA + 1], tt); vs += [int(rr.min()), int(rr.max())]
                                if b_ > HB: vs += [int(rshift(np.int64(max(a_, HB + 1)), tt)), int(rshift(np.int64(b_), tt))]
                                e16.append((min(vs), max(vs)))
                            e16 = np.array(e16, np.int64)
                            if (e16[:, 0] > LIM).any() or (e16[:, 1] < -LIM).any(): continue
                            e16 = np.clip(e16, -LIM, LIM)
                            lo_p = np.minimum(e16[:, :1] * w2i, e16[:, 1:] * w2i).sum(0)
                            hi_p = np.maximum(e16[:, :1] * w2i, e16[:, 1:] * w2i).sum(0)
                            a1 = rshift(lo_p, -(we2p + tt)) + b2p; a2 = rshift(hi_p, -(we2p + tt)) + b2p
                            lo_m = a1 if lo_m is None else np.minimum(lo_m, a1); hi_m = a2 if hi_m is None else np.maximum(hi_m, a2)
                        bad("rebuild_lens2_differs", (lo_m != MOL[kk]).sum() + (hi_m != MOH[kk]).sum())
                        CHK["rebuild_lens2_lines"] = CHK.get("rebuild_lens2_lines", 0) + LINES
                # fixed only switches on: walked from the end, a line stays only while it is fixed
                fmF = np.zeros((Pc, LINES), bool); fmF[rows] = fm; fxF = np.zeros((Pc, LINES), bool); fxF[rows] = fx
                bad("fixed_switched_off", (fmF & ~stm).sum() + (fxF & ~stx).sum())
                stm &= fmF; FMs[stm] = s; stx &= fxF; FXs[stx] = s
                pmF = np.zeros(Pc, bool); pmF[rows] = fm.all(1); pxF = np.zeros(Pc, bool); pxF[rows] = fx.all(1) & (nposs == 1)
                stpm &= pmF; FPMs[stpm] = s; stpx &= pxF; FPXs[stpx] = s
                if s == NS - 1:
                    nf = int((~fm).sum() + (~fx).sum() + (~pxF).sum())
                    if vi == 0: bad("not_fixed_at_full", nf)      # (A): every edge exact here
                    else: CHK["B_unfixed_at_full_(frozen_edges_never_exact)"] = CHK.get("B_unfixed_at_full_(frozen_edges_never_exact)", 0) + nf
            if vi == 0: bad("push_at_full_vs_lens2", (COST1[NS - 1] != L2P).sum() + (~SINGLE[NS - 1]).sum())
            else: CHK["B_positions_not_one_level_at_full"] = CHK.get("B_positions_not_one_level_at_full", 0) + int((~SINGLE[NS - 1]).sum())
            OUT[f"FM_{v}"][r0:r1], OUT[f"FX_{v}"][r0:r1] = FMs, FXs
            OUT[f"FPM_{v}"][r0:r1], OUT[f"FPX_{v}"][r0:r1] = FPMs, FPXs
            # ---- work, for each stop rule: the stream value (m) and the next light (x) ----
            for rname, stop in (("m", FPMs.astype(np.int64)), ("x", FPXs.astype(np.int64))):
                if vi == 0:
                    e = np.repeat(stop[:, None], N, 1)
                else:
                    e = np.minimum(FRZ.astype(np.int64), stop[:, None])
                l1 = CM[e, ar[:, None], np.arange(N)[None, :]].sum(1, dtype=np.int64)
                T[f"l1_{v}_{rname}"] += int(l1.sum()); T[f"reads_{v}_{rname}"] += int((np.minimum(e + 1, NS - 1) * 60).sum())
                ca = COSTA[stop, ar]; bad("no_push_at_stop", (ca < 0).sum())
                T[f"push1_{v}_{rname}"] += int((ca - L2P).sum())          # one push at the stop: the extra over lens 2
                sii = np.array([s_ for s_ in [int(np.flatnonzero(SINGLE[st_:, p_])[0]) + st_ for p_, st_ in enumerate(stop)]])
                pushes = SINGLE & (np.arange(NS)[:, None] <= sii[None, :])
                T[f"push2_{v}_{rname}"] += int((COST1 * pushes).sum() - L2P.sum()); T[f"npush2_{v}_{rname}"] += int(pushes.sum())
                e2 = np.repeat(sii[:, None], N, 1) if vi == 0 else np.minimum(FRZ.astype(np.int64), sii[:, None])
                T[f"l1_ii_{v}_{rname}"] += int(CM[e2, ar[:, None], np.arange(N)[None, :]].sum(dtype=np.int64))
                # rule (iii): the least a machine could try: one push at D28, then the full pile if that fails
                s3 = np.maximum(stop, NS - 2)
                T[f"push3_{v}_{rname}"] += int((COSTA[NS - 2] + np.where(s3 == NS - 1, COSTA[NS - 1], 0) - L2P).sum())
                e3 = np.repeat(s3[:, None], N, 1) if vi == 0 else np.minimum(FRZ.astype(np.int64), s3[:, None])
                T[f"l1_iii_{v}_{rname}"] += int(CM[e3, ar[:, None], np.arange(N)[None, :]].sum(dtype=np.int64))
        del HLO, HHI, CM, FRZ, FLO, FHI
        log(f"  layer {l} positions {r0}-{r1 - 1} done ({el()}s, {mb()} MB)")

    # ---- edge stops against settle.py's ----
    frz = np.empty((P, N), np.int8); frz[:, NP_] = FRZALL
    if STPREV is not None: bad("edge_stops_vs_settle", (frz != STPREV[f"L{l}_diag_signed_output"]).sum())
    li = list(PREV["layers"]).index(l)
    bad("edge_only_work_vs_settle", T["edge_only"] != int(PREV["work"][li, 1, 1, 2]))
    bad("full_vs_settle", T["full"] != int(PREV["full"][li]))
    # ---- back to the machine's order, digests ----
    DIG = {}
    for k, a in OUT.items():
        if a.ndim == 2: b = np.empty_like(a); b[:, OP_] = a; a = b
        OUT[k] = a; DIG[k] = hashlib.sha256(a.tobytes()).hexdigest()
    if ORDER:
        ref = np.load(f"{HERE}/knife/stream_lens2_real_L{l}.npz")
        same = sum(int(str(ref[f"dig_{k}"]) == DIG[k]) for k in DIG)
        bad("order_free_differs", len(DIG) - same); CHK["order_free_arrays"] = CHK.get("order_free_arrays", 0) + same
    np.savez(f"{HERE}/knife/stream_lens2_{TAG}_L{l}.npz", **OUT, **{f"dig_{k}": np.array(d) for k, d in DIG.items()},
             **{f"T_{k}": np.array(x) for k, x in T.items()},
             **{f"WID_{v}_{s}_{w}": a for (v, s, w), a in WID.items()}, frz=frz)
    TOTALS[l] = T

    # ---- the layer's report ----
    PL_ = P * LINES
    log(f"\nlayer {l}: full fc meeting {T['full']:,}; plain lens-2 meeting {T['l2']:,} ({pm(T['l2'], T['full'])} of fc); "
        f"settle.py's per-edge output stops need {pm(T['edge_only'], T['full'])} of fc ({el()}s, {mb()} MB)")
    log("    " + " " * 34 + "".join(f"{nm:>7s}" for nm in NAMES))
    for v in VN:
        for key, lab, n in (("FM", "stream value, lines", PL_), ("FX", "next light, lines", PL_),
                            ("FPM", "stream value, all 768 (positions)", P), ("FPX", "next light, all 768 (positions)", P)):
            cum = np.cumsum(np.bincount(OUT[f"{key}_{v}"].ravel().astype(np.int64), minlength=NS))[:NS]
            log(f"    ({v}) {lab:30s}" + "".join(f"{pm(x, n):>7s}" for x in cum))
    for v in VN:
        for rname, lab in (("m", "stream value"), ("x", "next light")):
            l1 = T[f"l1_{v}_{rname}"]
            log(f"    ({v}) stop when the {lab} is fixed: lens 1 {pm(l1, T['full'])} of fc; extra push (i) one push "
                f"{T[f'push1_{v}_{rname}']:,}, (ii) every one-level stage {T[f'push2_{v}_{rname}']:,} "
                f"({T[f'npush2_{v}_{rname}'] / P:.2f} pushes a position) = {pm(T[f'push2_{v}_{rname}'], T['l2'])} of lens 2")
    for v in VN:
        nb = T["nblur"][list(VN).index(v)]; ns_ = T["nsingle"][list(VN).index(v)]
        log(f"    ({v}) edges still unsure a position (where sh is one value): " +
            ", ".join(f"{NAMES[s]} {nb[s] // max(ns_[s], 1)} ({pm(ns_[s], P)} of positions one-level)" for s in range(NS - 6, NS)))
    for s in WIDE[::-1]:
        h = WID[("A", s, "m")]; hx = WID[("A", s, "x")]; cm_ = np.cumsum(h); tot = int(cm_[-1])
        med = int(np.searchsorted(cm_, (tot + 1) // 2))
        log(f"    (A) at {NAMES[s]}: stream value width 0: {pm(h[0], tot)}; median width 2^{med - 1 if med else '-'}..2^{med} "
            f"units; next light width 0: {pm(hx[0], int(hx.sum()))}")
    log(f"  checks so far: " + ", ".join(f"{k} {x}" for k, x in sorted(BAD.items())))

# ---------------- the layers run ----------------
log(f"\n================ LAYERS {LAYERS}, light={LIGHT} order={ORDER} ================")
FT = sum(T["full"] for T in TOTALS.values()); L2T = sum(T["l2"] for T in TOTALS.values())
EO = sum(T["edge_only"] for T in TOTALS.values())
log(f"fc meeting {FT:,} ({FT // P:,} a token); lens 2 {L2T:,} ({L2T // P:,} a token); per-edge stops {pm(EO, FT)} of fc")
for v in VN:
    for rname, lab in (("m", "stream value"), ("x", "next light")):
        l1 = sum(T[f"l1_{v}_{rname}"] for T in TOTALS.values()); p1 = sum(T[f"push1_{v}_{rname}"] for T in TOTALS.values())
        p2 = sum(T[f"push2_{v}_{rname}"] for T in TOTALS.values()); l1i = sum(T[f"l1_ii_{v}_{rname}"] for T in TOTALS.values())
        rd = sum(T[f"reads_{v}_{rname}"] for T in TOTALS.values())
        p3 = sum(T[f"push3_{v}_{rname}"] for T in TOTALS.values()); l13 = sum(T[f"l1_iii_{v}_{rname}"] for T in TOTALS.values())
        ref = FT if v == "A" else EO                     # (B) against settle.py's per-edge stops alone
        log(f"  ({v}) stop at the {lab}: lens 1 {pm(l1, FT)} of fc, saves {(ref - l1) // P:,} a token against "
            f"{'the plain fc meeting' if v == 'A' else 'the per-edge stops'}; (i) extra {p1 // P:,}, net "
            f"{pm(ref - l1 - p1, P * WHOLE)} of the whole machine; (ii) lens 1 {pm(l1i, FT)}, extra {p2 // P:,} a token, "
            f"net {pm(ref - l1i - p2, P * WHOLE)}; (iii) lens 1 {pm(l13, FT)}, extra {p3 // P:,} a token, net "
            f"{pm(ref - l13 - p3, P * WHOLE)}; bound reads {rd // P:,} a token")
log(f"\nCHECKS: " + ", ".join(f"{k} {x}" for k, x in sorted(BAD.items())))
log(f"  counted: " + ", ".join(f"{k} {x:,}" for k, x in sorted(CHK.items())))
log(f"  every interval held, every fixed value equals the machine's own: {sum(BAD.values()) == 0}")
log(f"({el()}s, max memory {mb()} MB)")
