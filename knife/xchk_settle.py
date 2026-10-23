"""CHECKER'S OWN STAGED COUNT OF THE KNIFE EDGES (a rebuild of knife/settle.py's measurement with other arithmetic;
nothing in the machine changes).

The light: recording/C_ref/plain_refs.npz's stream entering fc.l (knife/xchk_forward.py checks it is base2's own on
all 30 texts). The machine's own functions (base2.py exec'd: gauge_log, power, edge, to16, rshift) make the exact pile,
H, share, output and water level.

Other arithmetic than settle.py at every step:
  counted part   light schedule: one product of the light cut to its top k places with the whole plate.
                 diagonal schedule: for each light place a, one product of that place's photons with the plate cut to
                 its places >= T - a (T = 29 - k); never from per-pair pieces.
  meetings       the same cuts with photon counts: popcount(light cut) x popcount(plate), or light place a's photons x
                 popcount(plate >> (T - a)).
  bounds         pair by pair, min(n_a, m_b) taken directly for each uncounted (a, b) (plain), and the up and down
                 pairs (signed); no min-tables.
  output         base2's edge at every H of a grid wider than its table, the minimum and the largest output over any
                 interval read from range tables (min/max), dark below the grid and straight above it (checked); out16 =
                 rshift(out, s) is settled when rshift(min, s) = rshift(max, s) for every s the position allows, and the
                 same value for each s. The water level's range from each edge's own |out| range (not settle.py's
                 NEGMAX).
  share          the value the edge reads (as settle.py); also the ENTRY itself (the table index, or below / above).
Stopping rule for the output as settle.py's: an edge whose out16 is settled stops and its interval is frozen; the
position's water level is bounded from the frozen intervals. Also the same with nothing frozen.
Stages, groups, work and bound reads as defined in the task. Every bound checked (pile and H inside, the true water
level inside), every settled value against the machine's own. Per-edge stages compared with knife/stages_real.npz
(real light), per-layer totals with knife/settle_real.npz / settle_shuffled.npz.
    python3 knife/xchk_settle.py [light=real|shuffled] [seed=2026] [layers=0,5,11] [texts=30]
"""
import sys, os, time, json, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
LIGHT = ARG.get("light", "real"); SEED = int(ARG.get("seed", 2026)); NT = int(ARG.get("texts", 30))
LAYERS = [int(v) for v in ARG.get("layers", "0,5,11").split(",")]
TAG = LIGHT + (f"_s{SEED}" if LIGHT == "shuffled" else "") + f"_t{NT}_L" + "-".join(map(str, LAYERS))
LOG = open(f"{HERE}/knife/xchk_settle_{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W, ES, PHI = J["M"], J["W"], J["ES"], J["PHI"]
rshift, power, edge, to16, gauge_log, gauged_bank = (J[k] for k in ("rshift", "power", "edge", "to16", "gauge_log", "gauged_bank"))
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
NPOS = REF["npos"]; P = int(NPOS[:NT].sum()); N, LINES, PL = 3072, 768, 15
BAD = {}
def bad(k, n): BAD[k] = BAD.get(k, 0) + int(n)
def bitlen(v):                                          # whole-number bit length, vectorised
    v = np.asarray(v, np.int64); n = np.zeros(v.shape, np.int64)
    for k in range(63): n += (v >> k) > 0
    return n
def pm(a, b): q = (1000 * int(a) + int(b) // 2) // int(b) if b else 0; return f"{q // 10}.{q % 10}%"

# ---------------- the edge ----------------
def entry(H): return rshift(H, ES - 12) + 32768                                   # base2.edge's index
def share_of(H): i = entry(H); return np.where(i < 0, 0, np.where(i > 65535, 65536, PHI[np.clip(i, 0, 65535)]))
def entry_read(H): return np.clip(entry(H), -1, 65536)                            # -1: below the table, 65536: above
assert (np.diff(PHI.astype(np.int64)) >= 0).all()
GA, GB = -(1 << 19) - 256, (1 << 19) + 256
GRID = np.arange(GA, GB + 1, dtype=np.int64); OUTG = edge(GRID)
assert np.array_equal(rshift(GRID * share_of(GRID), 16), OUTG)
assert entry(np.array([GA]))[0] < 0 and entry(np.array([GB]))[0] > 65535          # the grid runs past both table ends
big = np.int64(1) << np.arange(19, 46)
assert (edge(-big) == 0).all() and (edge(np.arange(GA - 70000, GA)) == 0).all()     # dark below: out = 0
assert np.array_equal(edge(big), big) and np.array_equal(edge(np.arange(GB + 1, GB + 70000)), np.arange(GB + 1, GB + 70000))
def sparse(a, f):
    T = [a.astype(np.int32)]; k = 1
    while (1 << k) <= len(a):
        h = 1 << (k - 1); T.append(f(T[-1][:-h], T[-1][h:])); k += 1
    return T
SMIN, SMAX = sparse(OUTG, np.minimum), sparse(OUTG, np.maximum)
def out_range(lo, hi):                                  # min and max of base2's edge over every H in [lo, hi]
    shp = lo.shape
    if len(shp) > 1: m1, m2 = out_range(lo.ravel(), hi.ravel()); return m1.reshape(shp), m2.reshape(shp)
    mn = np.full(lo.shape, 1 << 62, np.int64); mx = np.full(lo.shape, -(1 << 62), np.int64)
    b = lo < GA; mn[b] = np.minimum(mn[b], 0); mx[b] = np.maximum(mx[b], 0)
    a = hi > GB; mn[a] = np.minimum(mn[a], np.maximum(lo[a], GB + 1)); mx[a] = np.maximum(mx[a], hi[a])
    g0 = np.maximum(lo, GA) - GA; g1 = np.minimum(hi, GB) - GA; ins = np.flatnonzero(g0 <= g1)
    if len(ins):
        g0i, g1i = g0[ins], g1[ins]; k = bitlen(g1i - g0i + 1) - 1
        for kk in np.unique(k):
            m = k == kk; i0, i1 = g0i[m], g1i[m] - (1 << int(kk)) + 1; j = ins[m]
            mn[j] = np.minimum(mn[j], np.minimum(SMIN[kk][i0], SMIN[kk][i1]))
            mx[j] = np.maximum(mx[j], np.maximum(SMAX[kk][i0], SMAX[kk][i1]))
    return mn, mx
_t = np.random.default_rng(1); _lo = _t.integers(GA - 3000, GB + 3000, 4000); _hi = _lo + _t.integers(0, 40000, 4000)
_mn, _mx = out_range(_lo, _hi)
for i in range(0, 4000, 97):                            # the range tables against base2's edge at every H
    v = edge(np.arange(_lo[i], _hi[i] + 1)); assert v.min() == _mn[i] and v.max() == _mx[i]
def sh_range(mn, mx):                                   # (Pc, N) out ranges -> the water level's shift, lowest, highest
    amax = np.maximum(np.abs(mn), np.abs(mx)); amin = np.where((mn <= 0) & (mx >= 0), 0, np.minimum(np.abs(mn), np.abs(mx)))
    return np.maximum(bitlen(amin.max(1)) - 15, 0), np.maximum(bitlen(amax.max(1)) - 15, 0)
def settled_out(mn, mx, slo, shi):                      # out16 one value over the range and every shift allowed
    v = rshift(mn, slo[:, None]); ok = rshift(mx, slo[:, None]) == v
    for d in range(1, int((shi - slo).max(initial=0)) + 1):
        s = (slo + d)[:, None]; live = s <= shi[:, None]
        ok &= ~live | ((rshift(mn, s) == v) & (rshift(mx, s) == v))
    return ok, v
def group(H): return np.where(H < -4 * 65536, 0, np.where(H < -2 * 65536, 1, np.where(H < 0, 2, np.where(100 * H < 266 * 65536, 3, 4))))
GN = ["far dark x<-4", "-4..-2", "-2..0", "0..2.66", "open >=2.66"]

# ---------------- stages ----------------
KL = [2, 4, 6, 8, 10, 12, 15]; KD = list(range(2, 29, 2)) + [29]
SCH = [("light", ["all" if k == 15 else f"L{k}" for k in KL]), ("diag", ["all" if k == 29 else f"D{k}" for k in KD])]
def counted_at(si, a, b):                               # the first stage at which pair (a, b) is counted
    if si == 0: return next(s for s, k in enumerate(KL) if a >= 15 - k)
    return next(s for s, k in enumerate(KD) if a + b >= 29 - k)
NS = [len(KL), len(KD)]; NSM = 15
CR = ["side", "share", "entry", "output", "output_nofreeze"]; BN = ["plain", "signed"]
nL = len(LAYERS)
HIST = np.zeros((nL, 2, 2, len(CR), 5, NSM), np.int64); WORK = np.zeros((nL, 2, 2, len(CR)), np.int64)
WORKB = np.zeros((nL, 2, 2, len(CR), 5), np.int64); FULL = np.zeros(nL, np.int64); FULLB = np.zeros((nL, 5), np.int64)
BREADS = np.zeros((nL, 2, 2), np.int64); NBK = np.zeros((nL, 5), np.int64); CMP = {}
EARLY = np.zeros((nL, 2, 2, 5, 2), np.int64)              # output settled before the full pile: by group, and of those out16 = 0
STR = np.load(f"{HERE}/knife/stages_real.npz") if LIGHT == "real" else None
CHUNK = 216

log(f"checker's own staged count: light={LIGHT}{' seed ' + str(SEED) if LIGHT == 'shuffled' else ''} layers={LAYERS} "
    f"texts={NT} ({P} positions x {N} edges per layer)")
for li, l in enumerate(LAYERS):
    R = REF[f"stream_a_{l}"][:P].astype(np.int64)
    if LIGHT == "shuffled":                              # each position's light across its lines, every value kept
        rng = np.random.default_rng(SEED + l); R = np.stack([r[rng.permutation(LINES)] for r in REF[f"stream_a_{l}"].astype(np.int64)])[:P]
    w16 = M[f"fc.{l}.w"]; we, bias = M[f"fc.{l}.e"], M[f"fc.{l}.b"]
    W[f"fc.{l}"] = (w16.astype(np.float64), we, bias)
    x16, ex, LS = gauge_log(R); Hm = gauged_bank(R, f"fc.{l}"); del W[f"fc.{l}"]
    wf = w16.astype(np.float64); pile = (x16.astype(np.float64) @ wf).astype(np.int64)
    if LIGHT == "real": bad("pile_vs_saved_run", (pile != REF[f"fc_pile_{l}"][:P]).sum())
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    H = power(pile, F) + bias; bad("H_vs_gauged_bank", (H != Hm).sum()); del Hm
    out = edge(H); o16, eo = to16(out, -ES); shm = eo + ES; SHv = share_of(H); ENv = entry_read(H)
    ax, sx = np.abs(x16).astype(np.int64), np.sign(x16).astype(np.int64)
    aw, sw = np.abs(w16.astype(np.int64)), np.sign(w16.astype(np.int64))
    assert ax.max() < 1 << PL and aw.max() < 1 << PL
    toplight = bitlen(np.bitwise_or.reduce(ax, axis=1)) - 1
    BK = group(H); NBK[li] = np.bincount(BK.ravel(), minlength=5)
    mA = np.stack([((aw >> b) & 1).sum(0) for b in range(PL)]); mP = np.stack([(((aw >> b) & 1) * (sw > 0)).sum(0) for b in range(PL)])
    mM = np.stack([(((aw >> b) & 1) * (sw < 0)).sum(0) for b in range(PL)])                 # (15, N)
    WCUT = [(sw * ((aw >> c) << c)).astype(np.float64) for c in range(PL)]                  # the plate cut to places >= c
    PCUT = [np.bitwise_count(aw >> c).astype(np.float32) for c in range(PL)]               # its photons per line
    pcw = PCUT[0]
    STG = {(si, bi, ci): np.full((P, N), -1, np.int8) for si in range(2) for bi in range(2) for ci in range(len(CR))}
    for r0 in range(0, P, CHUNK):
        r1 = min(P, r0 + CHUNK); Pc = r1 - r0
        axc, sxc = ax[r0:r1], sx[r0:r1]; pc, Fc, Hc, BKc = pile[r0:r1], F[r0:r1], H[r0:r1], BK[r0:r1]
        fullm = (np.bitwise_count(axc).astype(np.float32) @ pcw).astype(np.int64)
        FULL[li] += fullm.sum(); FULLB[li] += np.array([fullm[BKc == g].sum() for g in range(5)])
        nA = np.stack([((axc >> a) & 1).sum(1) for a in range(PL)]); nP = np.stack([(((axc >> a) & 1) * (sxc > 0)).sum(1) for a in range(PL)])
        nM = np.stack([(((axc >> a) & 1) * (sxc < 0)).sum(1) for a in range(PL)])            # (15, Pc)
        XB = [(sxc * ((axc >> a) & 1)).astype(np.float64) for a in range(PL)]
        XU = [((axc >> a) & 1).astype(np.float32) for a in range(PL)]
        for si, (sname, names) in enumerate(SCH):
            ns = NS[si]
            G = {k: [np.zeros((Pc, N), np.int64) for _ in range(ns)] for k in ("pl", "up", "dn")}
            for a in range(PL):                          # each pair's bound, put at the stage it gets counted
                for b in range(PL):
                    g = counted_at(si, a, b); sh = a + b
                    G["pl"][g] += np.minimum(nA[a][:, None], mA[b][None, :]) << sh
                    G["up"][g] += (np.minimum(nP[a][:, None], mP[b][None, :]) + np.minimum(nM[a][:, None], mM[b][None, :])) << sh
                    G["dn"][g] += (np.minimum(nP[a][:, None], mM[b][None, :]) + np.minimum(nM[a][:, None], mP[b][None, :])) << sh
            for k in G:                                  # bound at stage s: every pair counted after s
                acc = np.zeros((Pc, N), np.int64)
                for s in range(ns - 1, -1, -1):
                    nxt = G[k][s]; G[k][s] = acc.copy(); acc += nxt
                del acc
            FR = {bi: [None, None] for bi in range(2)}       # frozen out ranges (output, edges stopped)
            for s in range(ns):
                if si == 0:
                    c0 = 15 - KL[s]
                    c = ((sxc * ((axc >> c0) << c0)).astype(np.float64) @ wf).astype(np.int64)
                    cm = (np.bitwise_count(axc >> c0).astype(np.float32) @ pcw).astype(np.int64)
                else:
                    T = 29 - KD[s]; c = np.zeros((Pc, N), np.int64); cm = np.zeros((Pc, N), np.int64)
                    for a in range(PL):
                        cut = max(T - a, 0)
                        if cut >= PL: continue
                        c += (XB[a] @ WCUT[cut]).astype(np.int64) << a
                        cm += (XU[a] @ PCUT[cut]).astype(np.int64)
                if s == ns - 1:
                    bad("last_stage_counted_vs_pile", (c != pc).sum()); bad("last_stage_meetings_vs_full", (cm != fullm).sum())
                Rb, up, dn = G["pl"][s], G["up"][s], G["dn"][s]
                bad("signed_wider_than_plain", ((up > Rb) | (dn > Rb)).sum())
                for bi, (plo, phi) in enumerate(((c - Rb, c + Rb), (c - dn, c + up))):
                    bad("pile_outside_bound", ((pc < plo) | (pc > phi)).sum())
                    assert np.abs(plo).max() < 1 << 45 and np.abs(phi).max() < 1 << 45
                    Hlo, Hhi = power(plo, Fc) + bias, power(phi, Fc) + bias
                    bad("H_outside_bound", ((Hc < Hlo) | (Hc > Hhi)).sum())
                    def mark(ci, ok, val=None, want=None):
                        st = STG[(si, bi, ci)][r0:r1]; new = ok & (st < 0)
                        if val is not None: bad(f"{CR[ci]}_wrong", (new & (val != want)).sum())
                        st[new] = s; WORK[li, si, bi, ci] += cm[new].sum()
                        WORKB[li, si, bi, ci] += np.array([cm[new & (BKc == g)].sum() for g in range(5)])
                        return new
                    mark(0, np.sign(Hlo) == np.sign(Hhi), np.sign(Hlo), np.sign(Hc))
                    sl = share_of(Hlo); mark(1, sl == share_of(Hhi), sl, SHv[r0:r1])
                    el = entry_read(Hlo); mark(2, el == entry_read(Hhi), el, ENv[r0:r1])
                    mn, mx = out_range(Hlo, Hhi)
                    # nothing frozen
                    slo, shi = sh_range(mn, mx); bad("sh_outside_range_nofreeze", ((shm[r0:r1] < slo) | (shm[r0:r1] > shi)).sum())
                    okf, vf = settled_out(mn, mx, slo, shi); mark(4, okf, vf, o16[r0:r1])
                    # edges stop when settled; frozen ranges for the water level
                    st = STG[(si, bi, 3)][r0:r1]; fr = st >= 0
                    if FR[bi][0] is not None: mn = np.where(fr, FR[bi][0], mn); mx = np.where(fr, FR[bi][1], mx)
                    slo, shi = sh_range(mn, mx); bad("sh_outside_range", ((shm[r0:r1] < slo) | (shm[r0:r1] > shi)).sum())
                    ok, v = settled_out(mn, mx, slo, shi)
                    new = mark(3, ok & ~fr, v, o16[r0:r1])
                    FR[bi] = [mn.copy(), mx.copy()]
            del G
        del XB, XU
    for si in range(2):
        for bi in range(2):
            st3 = STG[(si, bi, 3)].astype(np.int64); pre = st3 < NS[si] - 1
            for g in range(5):
                EARLY[li, si, bi, g] = [int((pre & (BK == g)).sum()), int((pre & (BK == g) & (o16 == 0)).sum())]
            BREADS[li, si, bi] = np.minimum(st3 + 1, NS[si] - 1).sum() * (PL if bi == 0 else 4 * PL)
            for ci in range(len(CR)):
                st = STG[(si, bi, ci)]; bad(f"unsettled_at_last_{CR[ci]}", (st < 0).sum())
                HIST[li, si, bi, ci] = np.bincount((BK.astype(np.int64) * NSM + st).ravel(), minlength=5 * NSM).reshape(5, NSM)
                if STR is not None and ci in (0, 1, 3):
                    key = f"L{l}_{SCH[si][0]}_{BN[bi]}_{['side', 'share', None, 'output'][ci]}"
                    CMP[key] = int((STR[key][:P] != st).sum())
    if STR is not None: CMP[f"L{l}_bucket"] = int((STR[f"L{l}_bucket"][:P] != BK).sum())
    del STG, WCUT, PCUT
    log(f"\nlayer {l}: light's top place {int(toplight.min())}..{int(toplight.max())}; groups " +
        ", ".join(f"{GN[g]} {pm(NBK[li, g], P * N)}" for g in range(5)) + f"; full fc meeting {FULL[li]:,} "
        f"({int(time.time() - t0)}s, {mb()} MB)")
    for si, (sname, names) in enumerate(SCH):
        ns = NS[si]
        for bi in (1, 0):
            log(f"  {sname} schedule, {BN[bi]} bound: cumulative settled by stage")
            log("    " + " " * 16 + "".join(f"{nm:>7s}" for nm in names))
            for ci in range(len(CR)):
                cum = np.cumsum(HIST[li, si, bi, ci].sum(0))[:ns]
                log(f"    {CR[ci]:16s}" + "".join(f"{pm(v, P * N):>7s}" for v in cum))
            pre = [int(HIST[li, si, bi, ci].sum(0)[:ns - 1].sum()) for ci in range(len(CR))]
            log(f"    before the full pile: " + ", ".join(f"{CR[ci]} {pm(pre[ci], P * N)}" for ci in range(len(CR)))
                + f"; output edges {pre[3]:,}")
            log(f"    work if each edge stops when settled: " + ", ".join(f"{CR[ci]} {pm(WORK[li, si, bi, ci], FULL[li])}"
                for ci in range(len(CR))) + f"; bound reads {BREADS[li, si, bi]:,}")
    log(f"  checks so far: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())))
    if CMP: log(f"  per-edge stages != knife/stages_real.npz: " + ", ".join(f"{k} {v}" for k, v in CMP.items() if k.startswith(f"L{l}_")))

# ---------------- against settle.py's own saved totals ----------------
src = np.load(f"{HERE}/knife/settle_{'real' if LIGHT == 'real' else 'shuffled'}.npz")
same_light = LIGHT == "real" or SEED == 2026
log(f"\n==== against knife/settle_{'real' if LIGHT == 'real' else 'shuffled'}.npz, per layer "
    f"({'same light' if same_light and NT == 30 else 'NOT the same light/positions: comparison only indicative'}) ====")
CI_MAP = [(0, 0), (1, 1), (2, 3)]                       # theirs (side, share, output) -> mine
for li, l in enumerate(LAYERS):
    j = list(src["layers"]).index(l); diffs = {}
    diffs["full"] = int(src["full"][j] - FULL[li]); diffs["fullb"] = int(np.abs(src["fullb"][j] - FULLB[li]).sum())
    for tci, mci in CI_MAP:
        diffs[f"hist_{CR[mci]}"] = int(np.abs(src["hist"][j][:, :, tci] - HIST[li][:, :, mci]).sum())
        diffs[f"work_{CR[mci]}"] = int(np.abs(src["work"][j][:, :, tci] - WORK[li][:, :, mci]).sum())
        diffs[f"workb_{CR[mci]}"] = int(np.abs(src["workb"][j][:, :, tci] - WORKB[li][:, :, mci]).sum())
    diffs["histf_output_nofreeze(signed)"] = int(np.abs(src["histf"][j] - HIST[li][:, 1, 4]).sum())
    diffs["bound_reads"] = int(np.abs(src["breads"][j] - BREADS[li]).sum())
    log(f"  layer {l}: differences " + ", ".join(f"{k} {v}" for k, v in diffs.items()))

# ---------------- the three layers together ----------------
TOT = P * N * nL; FT = int(FULL.sum())
log(f"\n==== layers {LAYERS} together, light={LIGHT} ====")
for si, (sname, names) in enumerate(SCH):
    ns = NS[si]
    for bi in (1, 0):
        cums = {ci: np.cumsum(HIST[:, si, bi, ci].sum((0, 1)))[:ns] for ci in range(len(CR))}
        log(f"  {sname} {BN[bi]}: " + " | ".join(f"{CR[ci]} " + " ".join(f"{names[s]}:{pm(cums[ci][s], TOT)}" for s in range(ns - 1))
                                                for ci in (0, 1, 3)))
        log(f"    work: " + ", ".join(f"{CR[ci]} {pm(WORK[:, si, bi, ci].sum(), FT)}" for ci in range(len(CR)))
            + f"; output before full pile {int(HIST[:, si, bi, 3].sum((0, 1))[:ns - 1].sum()):,} of {TOT:,}"
            + f"; per-layer side work " + ", ".join(pm(WORK[li, si, bi, 0], FULL[li]) for li in range(nL))
            + f"; per-layer output work " + ", ".join(pm(WORK[li, si, bi, 3], FULL[li]) for li in range(nL)))
    E = EARLY[:, si, 1].sum(0)
    log(f"  {sname} signed: output settled before the full pile, by group (edges, of which out16 = 0): " +
        ", ".join(f"{GN[g]} {E[g, 0]:,} ({E[g, 1]:,})" for g in range(5)) + f"; total {E[:, 0].sum():,} ({E[:, 1].sum():,})")
    log(f"  {sname} signed by group: share of all edges; per criterion: settled before the full pile, median stage, work")
    for g in range(5):
        row = f"    {GN[g]:14s} {pm(NBK[:, g].sum(), TOT):>6s}:"
        for ci in (0, 1, 2, 3):
            h = HIST[:, si, 1, ci, g].sum(0)[:ns]; n = int(h.sum())
            med = names[int(np.searchsorted(np.cumsum(h), (n + 1) // 2))] if n else "-"
            row += f"  {CR[ci]} {pm(h[:ns - 1].sum(), n):>6s} {med:>4s} {pm(WORKB[:, si, 1, ci, g].sum(), FULLB[:, g].sum()):>6s}"
        log(row)
log(f"\nCHECKS: " + ", ".join(f"{k} {v}" for k, v in sorted(BAD.items())))
log(f"every bound held and every settled value equals the machine's own: {sum(BAD.values()) == 0}")
if CMP: log(f"per-edge stages identical to knife/stages_real.npz: {sum(v == 0 for v in CMP.values())}/{len(CMP)} arrays")
np.savez(f"{HERE}/knife/xchk_settle_{TAG}.npz", layers=np.array(LAYERS), hist=HIST, work=WORK, workb=WORKB, full=FULL,
         fullb=FULLB, breads=BREADS, nbk=NBK, early=EARLY, bad_keys=np.array(sorted(BAD)), bad=np.array([BAD[k] for k in sorted(BAD)]))
log(f"({int(time.time() - t0)}s, max memory {mb()} MB)")
