"""THE KNIFE EDGES: HOW EARLY IS EACH BEND'S VISIBLE FRINGE DECIDED?  A MEASUREMENT ONLY; NOTHING IN THE MACHINE CHANGES.

THE PICTURE (his and the session's). Each layer's middle is a Schlieren / 4f system. The neurons-in plate fc.l is lens
1: it forms 3,072 piles, the spectrum. The bends are Gaussian knife edges in that focal plane: the share is Phi(pile),
the fraction of a Gaussian beam past a knife edge at that place. The neurons-out plate mproj.l is lens 2. The knife edge
makes the fringes visible. An edge needs only which side its light falls on, and how far. So: count each pile from the
top of the triangle down, and find the first stage at which the edge's side, its share and its output are provably
fixed.

THE MACHINE (base2.py, unchanged; its own functions are exec'd and called, the way table_bits.py / judge5.py load it):
  x16, ex, LS = gauge_log(R)          the light on 16 places (places 0..14 lit), one gauge per position
  pile = product(x16, w)              w = fc.l, int16, places 0..14, one power of two per neuron
  H = power(pile, F) + b              F = (we + ex + 16) x 65536 - LS; the value arriving at the edge, at 2^-16
  share = 0 if idx < 0, 65536 if idx > 65535, else PHI[idx];  idx = rshift(H, 4) + 32768      (the entry the edge reads)
  out = rshift(H x share, 16)         the bend's output
  out16 = rshift(out, sh)             sh from to16 over the position's 3,072 outputs: the 16 places mproj.l meets
power never runs backwards (a bigger pile never gives a smaller H); the share never runs backwards in H (PHI checked);
out never runs backwards for H >= 0 (checked); for H < 0 out is read from a table of every H the edge's table covers.

THE LIGHT: the stream entering fc.l at every position of the 30 refs texts (ids + ref, 648 positions), from
recording/C_ref/plain_refs.npz, base2.py's own run kept by C_plain.py (knife/check3.py runs base2.py's own forward on 3
texts against it). Every pile here is checked against that run's fc piles.

STAGES, fixed in advance (no search), from the top of the triangle (light place a, plate place b, a pair worth 2^(a+b)):
  sched=light   the light's places from the top: top 2, 4, 6, 8, 10, 12, all 15 (every plate place each time)
  sched=diag    the diagonals from the top: top 2, 4, ..., 28, all 29
After each stage the uncounted part is bounded exactly, without reading it:
  plain bound (collapse_top's):  |rest| <= sum over uncounted (a,b) of 2^(a+b) min(n_a, m_b)
  signed bound:   rest <= sum over uncounted 2^(a+b) [min(n+_a, m+_b) + min(n-_a, m-_b)]     (up pairs: same signs)
                  rest >= -sum over uncounted 2^(a+b) [min(n+_a, m-_b) + min(n-_a, m+_b)]   (down pairs)
  n_a, n+_a, n-_a: lines where the light has a photon at place a (any value, up values, down values): the light's own,
  known at the start.  m_b, m+_b, m-_b: lines where the neuron's plate has a photon at place b: 15 x 3 counts per neuron,
  stored once with the plate.  pile in [lo, hi]  ->  H in [power(lo, F) + b, power(hi, F) + b].
SETTLED at a stage (per neuron and position; the first stage at which it holds; the intervals only shrink):
  (a) SIDE    sign(H) the same over the whole interval: which side of the knife edge (-, 0 exactly, +)
  (b) SHARE   the table entry the edge reads the same over the whole interval
  (c) OUTPUT  out16 the same over the whole interval and over every sh the position's 3,072 intervals allow. An edge
              whose output is settled stops: its interval is frozen there, and the position's sh is bounded from the
              frozen intervals, so (c) is what a machine stopping each edge could certify. (c-full, signed bound only:
              the same with nothing frozen, to show the freezing costs nothing.)
WORK, in the unit of all earlier results: photon meetings = light photons x plate photons on each line, for the place
pairs met; per edge, the meetings of the pairs counted up to the stage it stops at; against the full fc meeting (every
pair, every edge). The bound's own table reads (15 per neuron per stage, plain; 60 signed) are stated apart.
GROUPS by where the pile sits on the edge, x = H / 65536 (exact H): far dark x < -4; -4..-2; -2..0; 0..2.66; open >= 2.66.
CHECKS (every position of the 30 texts, beyond the 3 asked): the pieces add up to the machine's pile and to the saved
fc pile; the exact pile and H inside every stage's bounds; every settled side, share and out16 equal to the machine's
own (base2's power, edge, to16); the true sh inside every stage's sh range; all the pairs' meetings equal the plain
meeting; the pieces against a pair-by-pair integer count on sampled piles; at the last stage everything settled.
CONTROL: light=shuffled shuffles each position's light across its 768 lines (every value kept; the gauge is unchanged)
and measures everything again on the piles that light makes (the machine's own functions on the shuffled light).
ORDER-FREE: order=1 permutes the lines (light and plate together), the neurons, and the order the place pairs are
added; every stage of every neuron must be the same (sha256 per layer, against the real run's).

GATING (holographicplate CLAUDE.md, main): it measures the settling of the knife edges (an exposure); every read is on
whole numbers (the products are whole numbers in float containers, every sum below 2^24, exact, as base2's product); it
is exact (bounds only, nothing approximated); it discards nothing (measurement only); no fit and no word-count
comparison; not measured before (DIRECTIONS.md 6a names it as the first exact test, never run).

PREDICTIONS (written before this script first ran):
  the task's: the side settles early only for edges whose light lands far from the edge (a small share of neurons,
    since 63-75% of piles sit at -2..0); the share and the output settle late, after most places, like the top read
    (the low places carry the same sign as the top ones), so the work saved at lens 1 is small (under 10%).
  DIRECTIONS.md 6a: under 1% of the shares fixed before their last places.
  the session's (from the bound's size: after the top k light places the plain bound is about 2^(first uncounted place)
    x half the lines x the plate's column, while the edge's unit, 2^-16 of x, sits about 9 places above the pile's
    bottom place):
    - plain and signed bounds nearly the same (the coin-toss places have n+ about n-).
    - light schedule: (a) SIDE before all for 50-85% of edges, mostly at the top 10-12 places; before the top 8 only for
      |x| > 3 (under 5%). (b) SHARE before all under 2% (only past the table's flat ends, x below about -4.4 or above
      +4.4). (c) OUTPUT before all under 3%, every one an output of exactly 0 after the water level (far dark). Work if
      each edge stops when settled: side 70-90%, share 98-100%, output 98-100% of the full fc meeting.
    - diagonal schedule: the low diagonals hold many meetings but little weight, so it does better on every criterion:
      (a) at the top 14-18 diagonals for most edges, side work 45-70%; (b) and (c) before all for 20-60% of edges, but
      only at the last 2-4 diagonals, which hold under 5% of the meetings: share and output work 95-99%.
    - groups: far dark settles first on every criterion; -2..0 last on the side; the output settles early only for far
      dark (and part of -4..-2).
    - control: the bounds are the same counts (n_a does not depend on which line a value sits on), so only where the
      piles land changes; the side settles at least as early, the share and output within a few points.
    - lens 1 saved by stopping each edge at its output: under 5% in both schedules.

    python3 knife/settle.py [light=real|shuffled] [order=0|1] [layers=0,5,11]
      -> knife/settle_<light>[_order].log and .npz; light=real also knife/stages_real.npz (every edge's stages)
"""
import sys, os, time, json, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
LIGHT, ORDER = ARG.get("light", "real"), int(ARG.get("order", 0))
LAYERS = [int(v) for v in ARG["layers"].split(",")] if "layers" in ARG else list(range(12))
TAG = LIGHT + ("_order" if ORDER else "") + ("" if "layers" not in ARG else "_L" + "-".join(map(str, LAYERS)))
LOG = open(f"{HERE}/knife/settle_{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def el(): return int(time.time() - t0)

_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]     # base2.py up to its own test
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')        # W filled one fc.l bank at a time, by base2's own line
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W, ES, PHI, EXP2 = J["M"], J["W"], J["ES"], J["PHI"], J["EXP2"]
rshift, power, edge, to16, gauge_log, gauged_bank, product = (J[k] for k in
    ("rshift", "power", "edge", "to16", "gauge_log", "gauged_bank", "product"))
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
P, N, LINES, PL, PW = int(REF["npos"].sum()), 3072, 768, 15, 15
ND = PL + PW - 1                                                       # 29 diagonals, 0..28
assert P == 648 and all(int(np.abs(M[f"fc.{l}.w"].astype(np.int64)).max()) < 1 << PW for l in range(12))

# ---------------- the edge, read from the machine itself ----------------
def share_of(H):                                       # base2.edge's own two lines: the table entry the edge reads
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
assert (np.diff(PHI) >= 0).all() and PHI.min() >= 0 and PHI.max() <= 65536                 # the share never runs back
HA, HB = -(1 << 19) - 64, (1 << 19) + 64                                                    # past the table on both sides
GRID = np.arange(HA, HB + 1, dtype=np.int64); OUTT = edge(GRID).astype(np.int64)           # every H's output
assert np.array_equal(rshift(GRID * share_of(GRID), 16), OUTT) and OUTT[0] == 0 and OUTT[-1] == HB
_pos = GRID >= 0; assert (OUTT[_pos] >= 0).all() and (np.diff(OUTT[_pos]) >= 0).all()       # out never runs back, H >= 0
NEGMAX = int(np.abs(OUTT[~_pos]).max())
_probe = np.concatenate([np.arange(HA - 4096, HA), -(np.int64(1) << np.arange(20, 44)),
                         np.arange(HB + 1, HB + 4096), np.int64(1) << np.arange(20, 44)])
assert np.array_equal(edge(_probe), np.where(_probe < HA, 0, _probe))                         # dark below, straight above
SMT = 21                                                                                     # rshift(out, >= 21) = 0 here
CP = np.zeros((SMT + 1, len(GRID)), np.int32)                                                # changes of out16, running
for _s in range(SMT + 1):
    _f = rshift(OUTT, _s); CP[_s, 1:] = np.cumsum(_f[1:] != _f[:-1])
assert not rshift(OUTT, SMT).any()

def out_ext(H): return np.where(H < HA, 0, np.where(H > HB, H, OUTT[np.clip(H - HA, 0, len(GRID) - 1)]))
def constant(lo, hi, s):                               # out16 = rshift(out, s) the same for every H in [lo, hi]?
    v = rshift(out_ext(lo), s); ok = v == rshift(out_ext(hi), s)
    sc = np.minimum(s, SMT); g0 = np.clip(lo, HA, HB) - HA; g1 = np.clip(hi, HA, HB) - HA
    ok &= (hi < HA) | (lo > HB) | (CP[sc, g1] == CP[sc, g0])
    ok &= (hi <= HB) | (rshift(np.maximum(lo, HB), s) == rshift(hi, s))
    return ok, v
def bitlen(v): return np.array([int(x).bit_length() for x in v], np.int64)
def sh_range(lo, hi):                                  # (P, N) intervals of H -> the water level's shift, lowest, highest
    U = np.maximum(np.where(hi >= 0, out_ext(np.maximum(hi, 0)), 0), np.where(lo < 0, NEGMAX, 0)).max(1)
    L = np.where(lo >= 0, out_ext(np.maximum(lo, 0)), 0).max(1)
    return np.maximum(bitlen(L) - 15, 0), np.maximum(bitlen(U) - 15, 0)
def out_settled(lo, hi, slo, shi):                     # flat: out16 the same over [lo, hi] and every sh in [slo, shi]
    ok, v = constant(lo, hi, slo)
    for d in range(1, int((shi - slo).max(initial=0)) + 1):
        m = ok & (slo + d <= shi)
        if not m.any(): break
        o2, v2 = constant(lo[m], hi[m], slo[m] + d); ok[m] = o2 & (v2 == v[m])
    return ok, v
def bucket_of(H):                                      # far dark, -4..-2, -2..0, 0..2.66, open (x = H / 65536)
    return np.where(H < -4 * 65536, 0, np.where(H < -2 * 65536, 1, np.where(H < 0, 2, np.where(100 * H < 266 * 65536, 3, 4))))
BNAMES = ["far dark x<-4", "-4..-2", "-2..0", "0..2.66", "open >=2.66"]

KL = [2, 4, 6, 8, 10, 12, PL]; KD = list(range(2, 29, 2)) + [ND]
_A, _B = np.arange(PL)[:, None], np.arange(PW)[None, :]
SCHEDS = [("light", [("all" if k == PL else f"L{k}", np.broadcast_to(_A < PL - k, (PL, PW)).copy()) for k in KL]),
          ("diag", [("all" if k == ND else f"D{k}", (_A + _B) < ND - k) for k in KD])]
NSMAX = max(len(s[1]) for s in SCHEDS); BNDS = ["plain", "signed"]; CRITS = ["side", "share", "output"]
nL = len(LAYERS)
HIST = np.zeros((nL, 2, 2, 3, 5, NSMAX), np.int64)       # edges by group whose criterion first settles at each stage
HISTF = np.zeros((nL, 2, 5, NSMAX), np.int64)            # c-full (signed bound, nothing frozen)
WORK = np.zeros((nL, 2, 2, 3), np.int64); WORKB = np.zeros((nL, 2, 2, 3, 5), np.int64)
FULL = np.zeros(nL, np.int64); FULLB = np.zeros((nL, 5), np.int64); BREADS = np.zeros((nL, 2, 2), np.int64)
SHR = np.zeros((nL, 2, 2, NSMAX, P, 2), np.int8)         # the sh range at every stage (for the checker)
DIG = {}; BAD = {}; STAGES = {}; CHK = {}
def bad(k, n): BAD[k] = BAD.get(k, 0) + int(n)
def pm(a, b): q = (1000 * int(a) + int(b) // 2) // int(b) if b else 0; return f"{q // 10}.{q % 10}%"

CHUNKS = [(0, 216), (216, 432), (432, 648)]               # every step is per position, so positions go in chunks
log(f"knife edges: light={LIGHT} order={ORDER} layers={LAYERS}; {P} positions x {N} edges per layer, "
    f"counted in {len(CHUNKS)} chunks of positions")
for li, l in enumerate(LAYERS):
    W[f"fc.{l}"] = (M[f"fc.{l}.w"].astype(np.float64), M[f"fc.{l}.e"], M[f"fc.{l}.b"])  # base2's own line, this bank only
    R = REF[f"stream_a_{l}"].astype(np.int64)
    if LIGHT == "shuffled":                              # CONTROL: each position's light across its lines
        rng = np.random.default_rng(2026 + l); R = np.stack([r[rng.permutation(LINES)] for r in R])
    w16 = M[f"fc.{l}.w"]; we, bias = M[f"fc.{l}.e"], M[f"fc.{l}.b"]
    x16_0, ex, LS = gauge_log(R)                         # the machine's own, in the machine's order
    pile_x = product(x16_0, W[f"fc.{l}"][0])
    if LIGHT == "real": bad("pile_vs_saved_run", (pile_x != REF[f"fc_pile_{l}"]).sum())
    Hx = gauged_bank(R, f"fc.{l}")
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    bad("H_vs_power", (power(pile_x, F) + bias != Hx).sum())
    outx = edge(Hx); o16x, eo = to16(outx, -ES); shx = eo + ES; sharex = share_of(Hx)
    bad("share_vs_edge", (rshift(Hx * sharex, 16) != outx).sum())
    del W[f"fc.{l}"], outx
    NP_ = np.arange(N)
    if ORDER:                                            # lines, neurons and the pairs' order all moved
        rng = np.random.default_rng(777 + l); LP = rng.permutation(LINES); NP_ = rng.permutation(N)
        x16, ex2, LS2 = gauge_log(R[:, LP]); assert np.array_equal(ex2, ex) and np.array_equal(LS2, LS)
        wl = w16[LP][:, NP_]
    else:
        x16, wl = x16_0, w16
    Fp, bp = F[:, NP_], bias[NP_]; Hx, o16x, sharex, pile_x = Hx[:, NP_], o16x[:, NP_], sharex[:, NP_], pile_x[:, NP_]
    ax, sx = np.abs(x16), np.sign(x16); aw = np.abs(wl.astype(np.int64)); sw = np.sign(wl).astype(np.int64)
    assert ax.max() < 1 << PL and aw.max() < 1 << PW
    IM = int((Fp >> 16).max()); assert IM <= 16 and int(EXP2.max()) < 1 << 17   # power: X x EXP2 then a right shift
    BK = bucket_of(Hx); NBK = np.bincount(BK.ravel(), minlength=5)
    bw = ((aw[None] >> np.arange(PW)[:, None, None]) & 1).astype(np.uint8)          # the plate's photons per place,
    mA = bw.sum(1, dtype=np.int64); mP = (bw * (sw > 0)).sum(1, dtype=np.int64)       # stored once with the plate
    mM = (bw * (sw < 0)).sum(1, dtype=np.int64); del bw
    pwc = np.bitwise_count(aw).astype(np.float64)
    pairs = [(a, b) for b in range(PW) for a in range(PL)]
    if ORDER: pairs = [pairs[i] for i in np.random.default_rng(55 + l).permutation(len(pairs))]
    SO = {(si, bi, ci): np.full((P, N), -1, np.int8) for si in range(2) for bi in range(2) for ci in range(3)}
    SOF = {si: np.full((P, N), -1, np.int8) for si in range(2)}

    for r0, r1 in CHUNKS:
        Pc = r1 - r0; axc, sxc = ax[r0:r1], sx[r0:r1]
        Hc, Fc, o16c, sharec, pilec, shc, BKc = Hx[r0:r1], Fp[r0:r1], o16x[r0:r1], sharex[r0:r1], pile_x[r0:r1], shx[r0:r1], BK[r0:r1]
        # ---- the pieces: every (light place, plate place) pair's balance and meetings ----
        D = np.zeros((ND, Pc, N), np.int16); LA = np.zeros((PL, Pc, N), np.int32)    # |D| <= 15 x 768; |LA| < 2^25
        MD = np.zeros((ND, Pc, N), np.int16); MLA = np.zeros((PL, Pc, N), np.int16)   # meetings <= 15 x 768
        Xs = [(sxc * ((axc >> a) & 1)).astype(np.float32) for a in range(PL)]
        Xu = [((axc >> a) & 1).astype(np.float32) for a in range(PL)]
        cb = -1
        for a, b in pairs:
            if b != cb: Wbit = (aw >> b) & 1; Ws = (sw * Wbit).astype(np.float32); Wu = Wbit.astype(np.float32); cb = b
            B = (Xs[a] @ Ws).astype(np.int32); D[a + b] += B.astype(np.int16); LA[a] += B << b    # |sums| <= 768: exact
            Mt = (Xu[a] @ Wu).astype(np.int16); MD[a + b] += Mt; MLA[a] += Mt
        del Xs, Xu, Ws, Wu, Wbit, B, Mt
        pile = np.zeros((Pc, N), np.int64)
        for j in range(ND): pile += D[j].astype(np.int64) << j
        bad("diagonals_vs_pile", (pile != pilec).sum())
        pile[:] = 0
        for a in range(PL): pile += LA[a].astype(np.int64) << a
        bad("light_places_vs_pile", (pile != pilec).sum()); del pile
        fullm = (np.bitwise_count(axc).astype(np.float64) @ pwc).astype(np.int64)
        bad("meetings_vs_plain", (MD.sum(0, dtype=np.int64) != fullm).sum())
        FULL[li] += fullm.sum()
        for g in range(5): FULLB[li, g] += fullm[BKc == g].sum()
        del fullm
        rs = np.random.default_rng(9 + l + r0)
        for _ in range(2):                               # pair by pair, line by line, in plain integers
            p, n = int(rs.integers(Pc)), int(rs.integers(N))
            for j in range(ND):
                want = sum(int((sxc[p] * sw[:, n] * ((axc[p] >> a) & 1) * ((aw[:, n] >> (j - a)) & 1)).sum())
                           for a in range(max(0, j - PW + 1), min(PL, j + 1)))
                bad("pieces_pair_by_pair", int(D[j, p, n]) != want); CHK["pieces"] = CHK.get("pieces", 0) + 1

        # ---- the light's photon counts per place (the light's own; with the plate's, the bound's only inputs) ----
        bx = ((axc[None] >> np.arange(PL)[:, None, None]) & 1)
        nA, nP, nM = bx.sum(2), (bx * (sxc > 0)).sum(2), (bx * (sxc < 0)).sum(2)        # (15, Pc)
        del bx
        KK = np.arange(LINES + 1)
        MN = {k: np.minimum(v[:, :, None], KK[None, None, :]).reshape(PL, -1).astype(np.float64)
              for k, v in (("a", nA), ("p", nP), ("m", nM))}
        def gtab(U, k):                                  # (15 b, Pc, 769): sum over uncounted a of 2^a min(n_a, m)
            C = (U.T * (1 << np.arange(PL))[None, :]).astype(np.float64)
            G = C @ MN[k]; assert G.max(initial=0) < 2 ** 50; return G.astype(np.int64).reshape(PW, Pc, LINES + 1)
        def gsum(G, mm):
            out = np.zeros((Pc, N), np.int64)
            for b in range(PW): out += np.take(G[b], mm[b], axis=1) << b
            return out

        for si, (sname, stages) in enumerate(SCHEDS):
            NS = len(stages)
            c = np.zeros((Pc, N), np.int64); cm = np.zeros((Pc, N), np.int64); counted = np.zeros((PL, PW), bool)
            ST = {(bi, ci): SO[(si, bi, ci)][r0:r1] for bi in range(2) for ci in range(3)}       # views
            STF = SOF[si][r0:r1]
            FLO = [np.zeros((Pc, N), np.int64) for _ in range(2)]; FHI = [np.zeros((Pc, N), np.int64) for _ in range(2)]
            for s, (name, U) in enumerate(stages):
                newp = ~U & ~counted; counted |= newp
                if sname == "light":
                    for a in np.flatnonzero(newp.any(1)): c += LA[a].astype(np.int64) << a; cm += MLA[a]
                else:
                    for j in sorted(set((_A + _B)[newp].tolist())): c += D[j].astype(np.int64) << j; cm += MD[j]
                if U.any():
                    G = gtab(U, "a"); Rb = gsum(G, mA); Gp = gtab(U, "p"); Gm = gtab(U, "m")
                    up = gsum(Gp, mP) + gsum(Gm, mM); dn = gsum(Gp, mM) + gsum(Gm, mP); del G, Gp, Gm
                else:
                    Rb = up = dn = np.zeros((Pc, N), np.int64)
                bad("signed_wider_than_plain", ((up > Rb) | (dn > Rb)).sum())
                for bi, (plo, phi) in enumerate(((c - Rb, c + Rb), (c - dn, c + up))):
                    bad("pile_outside_bound", ((pilec < plo) | (pilec > phi)).sum())
                    assert int(np.abs(plo).max()) < 1 << 45 and int(np.abs(phi).max()) < 1 << 45
                    Hlo, Hhi = power(plo, Fc) + bp, power(phi, Fc) + bp
                    bad("H_outside_bound", ((Hc < Hlo) | (Hc > Hhi)).sum())
                    # (a) side
                    st = ST[(bi, 0)]; ok = (st < 0) & (np.sign(Hlo) == np.sign(Hhi))
                    bad("side_wrong", (ok & (np.sign(Hlo) != np.sign(Hc))).sum()); st[ok] = s
                    WORK[li, si, bi, 0] += cm[ok].sum()
                    for g in range(5): WORKB[li, si, bi, 0, g] += cm[ok & (BKc == g)].sum()
                    # (b) share
                    st = ST[(bi, 1)]; sl = share_of(Hlo); ok = (st < 0) & (sl == share_of(Hhi))
                    bad("share_wrong", (ok & (sl != sharec)).sum()); st[ok] = s
                    WORK[li, si, bi, 1] += cm[ok].sum()
                    for g in range(5): WORKB[li, si, bi, 1, g] += cm[ok & (BKc == g)].sum()
                    # (c) output, edges stopping when settled
                    st = ST[(bi, 2)]; fr = st >= 0
                    LOc, HIc = np.where(fr, FLO[bi], Hlo), np.where(fr, FHI[bi], Hhi)
                    slo, shi = sh_range(LOc, HIc); SHR[li, si, bi, s, r0:r1, 0], SHR[li, si, bi, s, r0:r1, 1] = slo, shi
                    bad("sh_outside_range", ((shc < slo) | (shc > shi)).sum())
                    idx = np.flatnonzero(~fr.ravel())
                    okc, v = out_settled(Hlo.ravel()[idx], Hhi.ravel()[idx], np.repeat(slo, N)[idx], np.repeat(shi, N)[idx])
                    new = idx[okc]
                    bad("output_wrong", (v[okc] != o16c.ravel()[new]).sum())
                    st.reshape(-1)[new] = s; FLO[bi].reshape(-1)[new] = Hlo.ravel()[new]; FHI[bi].reshape(-1)[new] = Hhi.ravel()[new]
                    cmn, gn = cm.ravel()[new], BKc.ravel()[new]
                    WORK[li, si, bi, 2] += cmn.sum()
                    for g in range(5): WORKB[li, si, bi, 2, g] += cmn[gn == g].sum()
                    if bi == 1:                          # c-full: nothing frozen
                        slo2, shi2 = sh_range(Hlo, Hhi); idx = np.flatnonzero(STF.ravel() < 0)
                        okf, v2 = out_settled(Hlo.ravel()[idx], Hhi.ravel()[idx], np.repeat(slo2, N)[idx], np.repeat(shi2, N)[idx])
                        bad("output_full_wrong", (v2[okf] != o16c.ravel()[idx[okf]]).sum()); STF.reshape(-1)[idx[okf]] = s
            for bi in range(2):
                BREADS[li, si, bi] += np.minimum(ST[(bi, 2)].astype(np.int64) + 1, NS - 1).sum() * (PW if bi == 0 else 4 * PW)
            del c, cm, FLO, FHI
        del D, LA, MD, MLA, MN

    for si, (sname, stages) in enumerate(SCHEDS):
        for bi in range(2):
            for ci in range(3):
                st = SO[(si, bi, ci)]; bad(f"unsettled_at_all_{CRITS[ci]}", (st < 0).sum())
                HIST[li, si, bi, ci] = np.bincount((BK * NSMAX + st).ravel(), minlength=5 * NSMAX).reshape(5, NSMAX)
                so = np.empty((P, N), np.int8); so[:, NP_] = st                    # back to the machine's neuron order
                DIG[f"L{l} {sname} {BNDS[bi]} {CRITS[ci]}"] = hashlib.sha256(so.tobytes()).hexdigest()
                if LIGHT == "real" and not ORDER: STAGES[f"L{l}_{sname}_{BNDS[bi]}_{CRITS[ci]}"] = so
        bad("unsettled_at_all_output_full", (SOF[si] < 0).sum())
        HISTF[li, si] = np.bincount((BK * NSMAX + SOF[si]).ravel(), minlength=5 * NSMAX).reshape(5, NSMAX)
        assert (WORKB[li, si].sum(-1) == WORK[li, si]).all()
    if LIGHT == "real" and not ORDER: STAGES[f"L{l}_bucket"] = BK.astype(np.int8)
    del SO, SOF

    # ---- the layer's report ----
    log(f"\nlayer {l}: {P * N:,} edges; groups " + ", ".join(f"{BNAMES[g]} {pm(NBK[g], P * N)}" for g in range(5))
        + f"; full fc meeting {FULL[li]:,} ({el()}s, {mb()} MB)")
    for si, (sname, stages) in enumerate(SCHEDS):
        NS = len(stages); names = [nm for nm, _ in stages]
        log(f"  {sname} schedule, signed bound: edges settled by each stage (cumulative)")
        log("    " + " " * 8 + "".join(f"{nm:>7s}" for nm in names))
        for ci in range(3):
            cum = np.cumsum(HIST[li, si, 1, ci].sum(0))[:NS]
            log(f"    {CRITS[ci]:8s}" + "".join(f"{pm(v, P * N):>7s}" for v in cum))
        for bi in (1, 0):
            pre = [int(HIST[li, si, bi, ci].sum(0)[:NS - 1].sum()) for ci in range(3)]
            log(f"    {BNDS[bi]:6s} bound: settled before the full pile: side {pm(pre[0], P * N)}, share {pm(pre[1], P * N)}, "
                f"output {pm(pre[2], P * N)} ({pre[2]:,} edges); work if each edge stops when settled: side "
                f"{pm(WORK[li, si, bi, 0], FULL[li])}, share {pm(WORK[li, si, bi, 1], FULL[li])}, output {pm(WORK[li, si, bi, 2], FULL[li])}")
        fpre = int(HISTF[li, si].sum(0)[:NS - 1].sum())
        log(f"    output with nothing frozen (signed): settled before the full pile {pm(fpre, P * N)}; "
            f"bound reads for the output stops: plain {BREADS[li, si, 0]:,}, signed {BREADS[li, si, 1]:,}")
    log(f"  checks so far: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())))

# ---------------- all layers ----------------
log(f"\n================ ALL {nL} LAYERS, light={LIGHT} order={ORDER} ================")
TOT = P * N * nL; FT = int(FULL.sum())
for si, (sname, stages) in enumerate(SCHEDS):
    NS = len(stages); names = [nm for nm, _ in stages]
    for bi in (1, 0):
        log(f"{sname} schedule, {BNDS[bi]} bound: edges settled by each stage (cumulative), all layers")
        log("    " + " " * 8 + "".join(f"{nm:>7s}" for nm in names))
        for ci in range(3):
            cum = np.cumsum(HIST[:, si, bi, ci].sum((0, 1)))[:NS]
            log(f"    {CRITS[ci]:8s}" + "".join(f"{pm(v, TOT):>7s}" for v in cum))
        log(f"    work if each edge stops when settled: side {pm(WORK[:, si, bi, 0].sum(), FT)}, share "
            f"{pm(WORK[:, si, bi, 1].sum(), FT)}, output {pm(WORK[:, si, bi, 2].sum(), FT)} of {FT:,} meetings")
        pre = int(HIST[:, si, bi, 2].sum((0, 1))[:NS - 1].sum())
        log(f"    output settled before the full pile: {pre:,} of {TOT:,} edges ({pm(pre, TOT)}); bound reads for those "
            f"stops {int(BREADS[:, si, bi].sum()):,}")
    log(f"  {sname} schedule, signed bound, by group (all layers): share of the group settled before the full pile; "
        f"median stage; meetings needed as a share of the group's full meeting")
    for g in range(5):
        row = f"    {BNAMES[g]:14s} {pm(HIST[:, si, 1, 0, g].sum(), TOT):>6s} of edges:"
        for ci in range(3):
            h = HIST[:, si, 1, ci, g].sum(0)[:NS]; n = int(h.sum())
            med = names[int(np.searchsorted(np.cumsum(h), (n + 1) // 2))] if n else "-"
            row += f"  {CRITS[ci]} {pm(h[:NS - 1].sum(), n):>6s} med {med:>4s} work {pm(WORKB[:, si, 1, ci, g].sum(), FULLB[:, g].sum()):>6s}"
        log(row)
log(f"\nCHECKS: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())))
nbad = sum(BAD.values()); log(f"  pieces checked pair by pair: {CHK.get('pieces', 0)} diagonals of sampled piles")
log(f"  every bound held and every settled value equals the machine's own: {nbad == 0}")
if ORDER:
    ref = json.load(open(f"{HERE}/knife/settle_real_digests.json"))
    same = [k for k in DIG if ref.get(k) == DIG[k]]
    log(f"ORDER-FREE: every edge's stages identical to the real run's: {len(same)}/{len(DIG)} arrays")
elif LIGHT == "real" and "layers" not in ARG:
    json.dump(DIG, open(f"{HERE}/knife/settle_real_digests.json", "w"), indent=0)
np.savez(f"{HERE}/knife/settle_{TAG}.npz", layers=np.array(LAYERS), hist=HIST, histf=HISTF, work=WORK, workb=WORKB,
         full=FULL, fullb=FULLB, breads=BREADS, shr=SHR, bad_keys=np.array(sorted(BAD)), bad=np.array([BAD[k] for k in sorted(BAD)]))
if STAGES and "layers" not in ARG: np.savez_compressed(f"{HERE}/knife/stages_real.npz", **STAGES)
log(f"({el()}s, max memory {mb()} MB)")
