"""AN INDEPENDENT CHECK OF knife/stream_lens2.py: the checker's own interval push through lens 2 and own stage search.
A measurement only; nothing in the machine changes. It shares no code with stream_lens2.py or settle.py; only base2.py's
own functions (the machine) are exec'd, the way table_bits.py / judge5.py load it.

THE PICTURE, restated. Lens 1 (fc.l) forms 3,072 piles; each pile meets its Gaussian knife edge (the bend); lens 2
(mproj.l) gathers the 3,072 edge outputs back onto the stream's 768 lines; the stream keeps lens 2's pile after its own
shift and bias (mo), and the next plate meets the stream's top 16 places after the next water level (the next light).
Lens 1 is counted diagonal by diagonal from the top; at each stage every edge's pile is known to lie in an exact
interval. Carry those intervals through the edge and through lens 2: at which stage is the value that joins the stream,
and the next light, provably one number?

HOW EACH STEP IS DONE HERE (other arithmetic than stream_lens2.py wherever there is a choice):
  stages        D2, D4, ..., D28, all: stage s counts every place pair (a, b) with a + b >= J0 = 29 - k.
  lens 1        per light place a, the plate with its places below J0 - a cleared: c = sum_a 2^a (signed photons of the
                light at place a) @ (cleared plate). [stream_lens2: per-pair diagonal pieces]
  meetings      per light place a, its photons @ the photons of the plate's kept places.
  bound         settle.py's signed bound, taken pair by pair with min() of the photon counts. [stream_lens2: tables]
  edge range    base2's edge read at every H of the table; the table falls into share blocks of 16 H values (one share
                each, so out never falls inside a block: checked over the whole table). The least out over [lo, hi] is
                then the out at lo or at a block start inside, the greatest the out at hi or at a block end inside; the
                block starts' least and the block ends' greatest over a run of blocks are read from two tables over the
                blocks. On the open side out never falls (checked), so its ends are read. Below the table dark, above it
                straight (checked on edge() itself). Checked against reading every H on 600 random intervals.
                [stream_lens2: a min/max table over every H]
  water level   every to16 shift the intervals allow, from each row's least and greatest |value|.
  lens 2        least pile = lo16 @ w+ + hi16 @ w-, greatest = hi16 @ w+ + lo16 @ w- (w+ / w- the plate's up / down
                values), for every water level allowed, then base2's shift and bias; joined. [stream_lens2: centre and
                width]
  next light    stream + that interval; every next water level allowed; rshift at both ends; joined.
  stage search  forward over every stage. At each stage a probe at ONE possible water level proves "not fixed" (the
                joined interval contains the probe's); the full join is done wherever the probe leaves any line fixed,
                and at D24, D26, D28 and all for every position. Fixed must never switch off (checked).
  (B)           settle.py's per-edge freeze, rebuilt: an edge stops once out16 is the same over its interval for every
                water level settle.py's crude range allows (the frozen edges keep their frozen intervals in that range);
                frozen edges carry their frozen interval into lens 2.
  work          the run's own definitions, rebuilt: lens 1 = meetings counted to the stop; lens 2 plain = photons of
                |out16| x the plate's photons on that edge's line; a push = photons of |lo16 + hi16| and of
                (hi16 - lo16) x the plate's photons on that line, over every allowed water level; rules (i) one push at
                the stop, (ii) a push at every one-level stage up to the first one-level stage at or after the stop,
                (iii) one push at D28 then the full pile; bound reads 60 per edge per bounded stage.
CHECKS: lens 1's count at all = the machine's pile; the pile, H, out, sh, lens-2 pile, mo, sn and the next light inside
every interval; intervals nested stage to stage; fixed never switching off; every fixed value = the machine's own; on 10
texts (0, 3, ..., 27) base2.py's own forward (every bank) against the stream used and every fixed value of those
positions. COMPARISON with the run: per-position stages (FM, FX, FPM, FPX, (A) and (B)), per-edge freezes, and every
per-layer total in knife/stream_lens2_<light>_L<l>.npz; settle_<light>.npz's fc meeting and per-edge-stop work.
    python3 knife/stream_xchk_lens2.py [light=real|shuffled] [seed=2026] [layers=0,5,11]
      -> knife/stream_xchk_lens2_<light>[_s<seed>]_L<layers>.log
"""
import sys, os, time, json, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; sums checked
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
ARG = dict(a.split("=") for a in sys.argv[1:])
LIGHT = ARG.get("light", "real"); SEED = int(ARG.get("seed", 2026)); CH = int(ARG.get("chunk", 108))
LAYERS = [int(v) for v in ARG.get("layers", "0,5,11").split(",")]
TAG = LIGHT + (f"_s{SEED}" if SEED != 2026 else "") + "_L" + "-".join(map(str, LAYERS))
LOG = open(f"{HERE}/knife/stream_xchk_lens2_{TAG}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def el(): return int(time.time() - t0)
def pm(a, b):
    if not b: return "-"
    q = (1000 * int(a) + int(b) // 2) // int(b) if a >= 0 else -((1000 * -int(a) + int(b) // 2) // int(b))
    return ("-" if q < 0 else "") + f"{abs(q) // 10}.{abs(q) % 10}%"

_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')
_argv = sys.argv; sys.argv = ["x"]; J = {}; exec(_src, J); sys.argv = _argv
M, W, ES = J["M"], J["W"], J["ES"]
rshift, power, edge, to16, gauge_log, gauged_bank, product, plain_bank, attention = (J[k] for k in
    ("rshift", "power", "edge", "to16", "gauge_log", "gauged_bank", "product", "plain_bank", "attention"))
def load(name): W[name] = (M[f"{name}.w"].astype(np.float64), M[f"{name}.e"], M[f"{name}.b"])   # base2's own line

REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); NPOS = REF["npos"].astype(np.int64); P = int(NPOS.sum())
TXT = json.load(open(f"{HERE}/refs_many.json")); assert [len(t["ids"]) + len(t["ref"]) for t in TXT] == list(NPOS)
OFF = np.concatenate([[0], np.cumsum(NPOS)])
TEN = list(range(0, 30, 3)); TENPOS = np.concatenate([np.arange(OFF[t], OFF[t + 1]) for t in TEN])
IN10 = np.zeros(P, bool); IN10[TENPOS] = True; P10 = int(IN10.sum())
N, LINES, PL, PW = 3072, 768, 15, 15
KD = list(range(2, 29, 2)) + [29]; NS = len(KD); J0 = [29 - k for k in KD]
NAMES = [f"D{k}" for k in KD[:-1]] + ["all"]
S24, S26, S28, SALL = NAMES.index("D24"), NAMES.index("D26"), NAMES.index("D28"), NS - 1
WIDE = (S24, S26, S28, SALL)
LIM = 1 << 15; BIG = np.int64(1) << 62; WHOLE = 3666596259
BAD = {}; CNT = {}
def bad(k, n): BAD[k] = BAD.get(k, 0) + int(n)
def cnt(k, n): CNT[k] = CNT.get(k, 0) + int(n)

# ---------------- whole-number helpers (own) ----------------
def blen(v):                                           # bit length of each whole number >= 0 (below 2^63)
    v = np.asarray(v, np.int64); o = np.zeros(v.shape, np.int64)
    for i in range(63): o += (v >> i) > 0
    return o
def pc(v, nb):                                         # photons (1s) of |v|, v below 2^nb
    v = np.abs(np.asarray(v, np.int64)); o = np.zeros(v.shape, np.int64)
    for i in range(nb): o += (v >> i) & 1
    return o
def shr(lo, hi):                                       # rows of intervals -> lowest, highest to16 shift allowed
    z = (lo <= 0) & (hi >= 0)
    least = np.where(z, 0, np.minimum(np.abs(lo), np.abs(hi))).max(1)
    most = np.maximum(np.abs(lo), np.abs(hi)).max(1)
    return np.maximum(blen(least) - 15, 0), np.maximum(blen(most) - 15, 0)

# ---------------- the edge, read from base2's edge() ----------------
LO_T, HI_T = -(1 << 19) - 32, (1 << 19) + 32
GH = np.arange(LO_T, HI_T + 1, dtype=np.int64); GO = edge(GH).astype(np.int64)
def gout(H): return GO[H - LO_T]
_pr = np.concatenate([np.arange(LO_T - 5000, LO_T), -(np.int64(1) << np.arange(20, 46)),
                      np.arange(HI_T + 1, HI_T + 5000), np.int64(1) << np.arange(20, 46)])
assert np.array_equal(edge(_pr), np.where(_pr < 0, 0, _pr))                  # dark below the table, straight above
assert (GO[GH < 0] <= 0).all() and (GO[GH >= 0] >= 0).all() and (np.diff(GO[GH >= 0]) >= 0).all()
_blk = (GH + 8) >> 4                                    # the share block (base2: idx = rshift(H, 4) + 32768)
assert (np.diff(GO)[_blk[1:] == _blk[:-1]] >= 0).all()  # out never falls inside a block
MA = int((LO_T + 8) >> 4)                               # the block holding LO_T
VM0 = MA + 1; V = gout(16 * np.arange(VM0, 1, dtype=np.int64) - 8)      # block starts above LO_T, up to block 0's
E = gout(16 * np.arange(MA, 0, dtype=np.int64) + 7)                      # block ends, up to block -1's
def sparse(seq, f):                                    # f over every run of 2^k blocks
    Tb = [seq.copy()]
    for k in range(1, int(len(seq)).bit_length()):
        h = 1 << (k - 1); prev = Tb[-1]; nxt = prev.copy(); nxt[:len(seq) - h] = f(prev[:len(seq) - h], prev[h:]); Tb.append(nxt)
    return np.stack(Tb)
VT, ET = sparse(V, np.minimum), sparse(E, np.maximum)
FLOG = np.zeros(len(V) + 2, np.int64)
for _k in range(1, len(V) + 2): FLOG[_k] = _k.bit_length() - 1
def rq(Tb, i, j, f):                                   # f over blocks i..j (i <= j), two overlapping runs
    k = FLOG[j - i + 1]; return f(Tb[k, i], Tb[k, j - (np.int64(1) << k) + 1])
NEGMAX = int(-GO[GH < 0].min())
def orange(lo, hi):                                    # least and greatest out over every H in [lo, hi]
    mn = np.full(lo.shape, BIG); mx = np.full(lo.shape, -BIG)
    d = lo < LO_T; mn = np.where(d, 0, mn); mx = np.where(d, 0, mx)
    a = np.maximum(lo, LO_T); b = np.minimum(hi, -1); g = a <= b              # the table's dark side
    ac = np.clip(a, LO_T, -1); bc = np.clip(b, LO_T, -1); ma = (ac + 8) >> 4; mb_ = (bc + 8) >> 4
    m1, m2 = ma + 1, mb_                                 # block starts in (a, b]
    i1 = np.clip(m1 - VM0, 0, len(V) - 1); i2 = np.maximum(np.clip(m2 - VM0, 0, len(V) - 1), i1)
    nmin = np.minimum(gout(ac), np.where(m1 <= m2, rq(VT, i1, i2, np.minimum), BIG))
    e1, e2 = ma, mb_ - 1                                 # block ends in [a, b)
    j1 = np.clip(e1 - MA, 0, len(E) - 1); j2 = np.maximum(np.clip(e2 - MA, 0, len(E) - 1), j1)
    nmax = np.maximum(gout(bc), np.where(e1 <= e2, rq(ET, j1, j2, np.maximum), -BIG))
    mn = np.where(g, np.minimum(mn, nmin), mn); mx = np.where(g, np.maximum(mx, nmax), mx)
    a = np.maximum(lo, 0); b = np.minimum(hi, HI_T); g = a <= b                 # the table's open side
    mn = np.where(g, np.minimum(mn, gout(np.clip(a, 0, HI_T))), mn); mx = np.where(g, np.maximum(mx, gout(np.clip(b, 0, HI_T))), mx)
    g = hi > HI_T                                                                # straight
    mn = np.where(g, np.minimum(mn, np.maximum(lo, HI_T + 1)), mn); mx = np.where(g, np.maximum(mx, hi), mx)
    return mn, mx
_rng = np.random.default_rng(31)
for _i in range(600):                                  # against reading every H
    a = int(_rng.integers(LO_T - 4000, HI_T + 4000)) if _i < 400 else int(_rng.integers(LO_T - 100, 0)); b = a + int(_rng.integers(0, 1 << int(_rng.integers(1, 21))))
    seg = edge(np.arange(a, b + 1, dtype=np.int64)); m1_, m2_ = orange(np.array([a]), np.array([b]))
    assert int(m1_[0]) == int(seg.min()) and int(m2_[0]) == int(seg.max()), (a, b)
def outpos(H): return np.where(H > HI_T, H, gout(np.clip(H, 0, HI_T)))   # out for H >= 0
def crude_sh(lo, hi):                                  # settle.py's water-level range for its freeze (rebuilt)
    U = np.maximum(np.where(hi >= 0, outpos(np.maximum(hi, 0)), 0), np.where(lo < 0, NEGMAX, 0)).max(1)
    L = np.where(lo >= 0, outpos(np.maximum(lo, 0)), 0).max(1)
    return np.maximum(blen(L) - 15, 0), np.maximum(blen(U) - 15, 0)

# ---------------- CHECK ON 10 TEXTS: base2.py's own forward ----------------
FWD = {}
if LIGHT == "real":
    Rs = [M["wte"][TXT[t]["ids"] + TXT[t]["ref"]] + M["wpe"][:int(NPOS[t])] for t in TEN]
    for l in range(max(LAYERS) + 1):
        for b in ("qkv", "proj", "fc", "mproj"): load(f"{b}.{l}")
        RA, MO = [], []
        for i in range(len(TEN)):
            R = Rs[i]
            R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
            RA.append(R)
            mo = plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
            R = R + mo; MO.append(mo); Rs[i] = R
        if l in LAYERS:
            RA, MO = np.concatenate(RA), np.concatenate(MO); xn, en = to16(RA + MO, -ES); FWD[l] = (RA, MO, xn, en + ES)
        for b in ("qkv", "proj", "fc", "mproj"): del W[f"{b}.{l}"]
    log(f"base2's own forward on 10 texts {TEN} ({P10} positions) through layer {max(LAYERS)} ({el()}s, {mb()} MB)")

STPREV = np.load(f"{HERE}/knife/stages_real.npz") if LIGHT == "real" else None
SETPREV = np.load(f"{HERE}/knife/settle_{'real' if LIGHT == 'real' else 'shuffled'}.npz")
RUNTAG = "real" if LIGHT == "real" else "shuffled"
log(f"light={LIGHT} seed={SEED} layers={LAYERS}; all {P} positions measured, the 10 texts reported apart; stages {NAMES}")

SUM = {}
for l in LAYERS:
    load(f"fc.{l}"); load(f"mproj.{l}")
    RALL = REF[f"stream_a_{l}"].astype(np.int64)
    if LIGHT == "shuffled":
        rng = np.random.default_rng(SEED + l); RALL = np.stack([r[rng.permutation(LINES)] for r in RALL])
    w16 = M[f"fc.{l}.w"].astype(np.int64); we1, b1 = M[f"fc.{l}.e"], M[f"fc.{l}.b"]
    w2 = M[f"mproj.{l}.w"].astype(np.int64); we2, b2 = M[f"mproj.{l}.e"], M[f"mproj.{l}.b"]
    # ---- the machine's own values ----
    x16A, exA, LSA = gauge_log(RALL)
    pileA = product(x16A, W[f"fc.{l}"][0])
    HA_ = gauged_bank(RALL, f"fc.{l}"); FA = (we1[None, :] + exA[:, None] + ES) * 65536 - LSA[:, None]
    bad("H_vs_power", (power(pileA, FA) + b1 != HA_).sum())
    outA = edge(HA_); o16A, eoA = to16(outA, -ES); shA = eoA + ES
    pile2A = product(o16A, W[f"mproj.{l}"][0])
    moA = plain_bank(outA, f"mproj.{l}")
    bad("mo_vs_pile2", (rshift(pile2A, -(we2[None, :] + shA[:, None])) + b2 != moA).sum())
    RnA = RALL + moA; xnA, enA = to16(RnA, -ES); snA = enA + ES
    if LIGHT == "real":
        bad("fc_pile_vs_saved_run", (pileA != REF[f"fc_pile_{l}"]).sum())
        bad("stream_after_vs_saved_run", (RnA != REF[f"stream_{l}"]).sum())
        RA, MO, xn, sn = FWD[l]
        bad("10texts_stream_in_vs_forward", (RA != RALL[TENPOS]).sum()); bad("10texts_mo_vs_forward", (MO != moA[TENPOS]).sum())
        bad("10texts_next_light_vs_forward", (xn != xnA[TENPOS]).sum() + (sn != snA[TENPOS]).sum())
        FMO = np.zeros((P, LINES), np.int64); FMO[TENPOS] = MO; FXN = np.zeros((P, LINES), np.int64); FXN[TENPOS] = xn
    del W[f"fc.{l}"], W[f"mproj.{l}"]
    aw, sw = np.abs(w16), np.sign(w16)
    mP = np.stack([(((aw >> b) & 1) * (sw > 0)).sum(0) for b in range(PW)]); mM = np.stack([(((aw >> b) & 1) * (sw < 0)).sum(0) for b in range(PW)])
    Wp, Wn = np.maximum(w2, 0).astype(np.float64), np.minimum(w2, 0).astype(np.float64)
    PWm = pc(w2, 16).sum(1)                              # the plate's photons on each edge's line of lens 2
    plate_full = pc(aw, 16)                              # photons of each fc plate value
    shmid = int(np.sort(shA)[P // 2]); reach = np.bincount(np.clip(blen(np.abs(w2)) + (we2[None, :] + shmid), 0, 31).ravel(), minlength=32)[:32]
    _rf = np.sort((blen(np.abs(w2)) + (we2[None, :] + shmid))[w2 != 0])                   # top place reached, in stream places
    REACH_MED = (int(_rf[len(_rf) // 4]), int(_rf[len(_rf) // 2]), int(_rf[3 * len(_rf) // 4]))   # quartiles: a unit lands below 2^k

    OUT = {f"{k}_{v}": np.full((P, LINES) if k in ("FM", "FX") else (P,), -1, np.int8) for k in ("FM", "FX", "FPM", "FPX") for v in "AB"}
    FRZALL = np.full((P, N), -1, np.int8)
    T = {}; T10 = {}; EX = {"tail28": 0, "tail26": 0, "tail24": 0}; EX10 = dict(EX)
    def add(k, x, m10):                                  # all positions, and the 10 texts apart
        x = np.asarray(x, np.int64); T[k] = T.get(k, 0) + int(x.sum()); T10[k] = T10.get(k, 0) + int(x[m10].sum())
    WID = {(v, s, w): np.zeros(64, np.int64) for v in "AB" for s in (S28, S26, S24) for w in "mx"}
    NBL = np.zeros((2, NS), np.int64); NSG = np.zeros((2, NS), np.int64)

    for r0 in range(0, P, CH):
        r1 = min(P, r0 + CH); Pc = r1 - r0; m10 = IN10[r0:r1]
        x16 = x16A[r0:r1]; ax, sx = np.abs(x16), np.sign(x16)
        Fc = FA[r0:r1]; Hc = HA_[r0:r1]; outc = outA[r0:r1]; o16c = o16A[r0:r1]; shc = shA[r0:r1]
        pilec = pileA[r0:r1]; pile2c = pile2A[r0:r1]; moc = moA[r0:r1]; Rc = RALL[r0:r1]; xnc = xnA[r0:r1]; snc = snA[r0:r1]
        # ---- lens 1 counted to each stage, and its meetings ----
        C = np.zeros((NS, Pc, N), np.int64); CM = np.zeros((NS, Pc, N), np.int64)
        Sa = [(sx * ((ax >> a) & 1)).astype(np.float64) for a in range(PL)]
        Ba = [((ax >> a) & 1).astype(np.float32) for a in range(PL)]
        for q in range(PW):
            Wq = (sw * ((aw >> q) << q)).astype(np.float64); Pq = pc(aw >> q, 16).astype(np.float32)
            for a in range(PL):
                ss = [s for s in range(NS) if max(J0[s] - a, 0) == q]
                if not ss: continue
                Tm = Sa[a] @ Wq; assert np.abs(Tm).max() < 2 ** 26
                Tm = Tm.astype(np.int64) << a; Tc = (Ba[a] @ Pq).astype(np.int64)
                for s in ss: C[s] += Tm; CM[s] += Tc
        del Sa, Ba, Wq, Pq
        bad("lens1_all_vs_pile", (C[SALL] != pilec).sum())
        fullm = (pc(ax, 16).astype(np.float64) @ plate_full.astype(np.float64)).astype(np.int64)
        bad("meetings_all_vs_plain", (CM[SALL] != fullm).sum())
        bad("meetings_not_growing", (np.diff(CM, axis=0) < 0).sum())
        # ---- the signed bound, pair by pair ----
        bx = (ax[:, :, None] >> np.arange(PL)) & 1
        nP = (bx * (sx > 0)[:, :, None]).sum(1); nM = (bx * (sx < 0)[:, :, None]).sum(1); del bx
        UP = np.zeros((NS, Pc, N), np.int64); DN = np.zeros((NS, Pc, N), np.int64)
        for j in range(max(J0)):
            bu = np.zeros((Pc, N), np.int64); bd = np.zeros((Pc, N), np.int64)
            for a in range(max(0, j - PW + 1), min(PL - 1, j) + 1):
                b = j - a; npa, nma = nP[:, a, None], nM[:, a, None]; mpb, mmb = mP[b][None, :], mM[b][None, :]
                bu += np.minimum(npa, mpb) + np.minimum(nma, mmb); bd += np.minimum(npa, mmb) + np.minimum(nma, mpb)
            for s in range(NS):
                if j < J0[s]: UP[s] += bu << j; DN[s] += bd << j
        del bu, bd
        HLO = np.empty((NS, Pc, N), np.int64); HHI = np.empty((NS, Pc, N), np.int64)
        for s in range(NS):
            plo, phi = C[s] - DN[s], C[s] + UP[s]
            assert max(int(np.abs(plo).max()), int(np.abs(phi).max())) < 1 << 45
            bad("pile_outside", ((pilec < plo) | (pilec > phi)).sum())
            HLO[s], HHI[s] = power(plo, Fc) + b1, power(phi, Fc) + b1
            bad("H_outside", ((Hc < HLO[s]) | (Hc > HHI[s])).sum())
        bad("H_not_nested", ((np.diff(HLO, axis=0) < 0) | (np.diff(HHI, axis=0) > 0)).sum())
        del C, UP, DN
        # ---- (B): settle.py's per-edge freeze, rebuilt ----
        FRZ = np.full((Pc, N), -1, np.int64); FLO = np.zeros((Pc, N), np.int64); FHI = np.zeros((Pc, N), np.int64)
        for s in range(NS):
            fr = FRZ >= 0
            a_, b_ = crude_sh(np.where(fr, FLO, HLO[s]), np.where(fr, FHI, HHI[s]))
            omin, omax = orange(HLO[s], HHI[s])
            val = rshift(omin, a_[:, None]); ok = ~fr
            for d in range(int((b_ - a_).max()) + 1):
                t = (a_ + d)[:, None]; on = t <= b_[:, None]
                lo_, hi_ = rshift(omin, t), rshift(omax, t)
                ok &= ~on | ((lo_ == hi_) & (lo_ == val))
            bad("frozen_out16_wrong", (ok & (val != o16c)).sum())
            FRZ[ok] = s; FLO[ok] = HLO[s][ok]; FHI[ok] = HHI[s][ok]
        bad("edge_never_frozen", (FRZ < 0).sum()); FRZALL[r0:r1] = FRZ
        ar = np.arange(Pc)[:, None]; nn = np.arange(N)[None, :]
        add("edge_only", CM[FRZ, ar, nn].sum(1), m10); add("full", fullm.sum(1), m10)
        L2P = pc(o16c, 18) @ PWm; add("l2", L2P, m10)
        for nm_, s_ in (("tail28", S28), ("tail26", S26), ("tail24", S24)):     # lens 1's meetings below each stage
            x_ = (CM[SALL] - CM[s_]).sum(1); EX[nm_] += int(x_.sum()); EX10[nm_] += int(x_[m10].sum())

        # ---- lens 2 and the next light, stage by stage, forward ----
        def join2(omin, omax, slo, shi, rows):           # every allowed water level; joined mo interval, cost
            nr = rows.size; MOL = np.full((nr, LINES), BIG); MOH = np.full((nr, LINES), -BIG)
            cost = np.zeros(nr, np.int64); nc = np.zeros(nr, np.int64)
            sl, sh_ = slo[rows], shi[rows]
            for d in range(int((sh_ - sl).max(initial=0)) + 1):
                k = np.flatnonzero(sl + d <= sh_)
                if k.size == 0: break
                t = (sl[k] + d)[:, None]; pr = rows[k]
                lo16 = rshift(omin[pr], t); hi16 = rshift(omax[pr], t)
                ok = ~((lo16 > LIM) | (hi16 < -LIM)).any(1)
                k, t, pr, lo16, hi16 = k[ok], t[ok], pr[ok], lo16[ok], hi16[ok]
                if k.size == 0: continue
                lo16 = np.clip(lo16, -LIM, LIM); hi16 = np.clip(hi16, -LIM, LIM); r = k.size
                A2 = np.concatenate([lo16, hi16]).astype(np.float64); Pp = A2 @ Wp; Pn = A2 @ Wn
                assert np.abs(Pp).max() < 2 ** 50 and np.abs(Pn).max() < 2 ** 50
                p2lo = (Pp[:r] + Pn[r:]).astype(np.int64); p2hi = (Pp[r:] + Pn[:r]).astype(np.int64)
                sft = -(we2[None, :] + t); mol, moh = rshift(p2lo, sft) + b2, rshift(p2hi, sft) + b2
                ls = np.maximum(-sft, 0)                  # a water level so high that lens 2's pile is shifted UP:
                over = (ls > 0) & (np.maximum(np.abs(p2lo), np.abs(p2hi)) >= np.left_shift(np.int64(1), np.clip(62 - ls, 0, 62)))
                if over.any():                            # past 2^62: the interval is left open (wider, still sound)
                    mol = np.where(over, -BIG, mol); moh = np.where(over, BIG, moh); cnt("open_interval_guard", int(over.sum()))
                tr = t[:, 0] == shc[pr]
                bad("pile2_outside", ((pile2c[pr[tr]] < p2lo[tr]) | (pile2c[pr[tr]] > p2hi[tr])).sum())
                bad("out16_outside", ((o16c[pr[tr]] < lo16[tr]) | (o16c[pr[tr]] > hi16[tr])).sum())
                MOL[k] = np.minimum(MOL[k], mol); MOH[k] = np.maximum(MOH[k], moh); nc[k] += 1
                cost[k] += pc(lo16 + hi16, 18) @ PWm + pc(hi16 - lo16, 18) @ PWm
            return MOL, MOH, cost, nc
        def joinx(rows, MOL, MOH):                       # the next light from a mo interval
            Rlo, Rhi = Rc[rows] + MOL, Rc[rows] + MOH; nlo, nhi = shr(Rlo, Rhi)
            XL = np.full(Rlo.shape, BIG); XH = np.full(Rlo.shape, -BIG); npo = np.zeros(rows.size, np.int64)
            for d in range(int((nhi - nlo).max(initial=0)) + 1):
                k = np.flatnonzero(nlo + d <= nhi)
                if k.size == 0: break
                t = (nlo[k] + d)[:, None]; xl, xh = rshift(Rlo[k], t), rshift(Rhi[k], t)
                ok = ~((xl > LIM) | (xh < -LIM)).any(1); k, xl, xh = k[ok], xl[ok], xh[ok]
                XL[k] = np.minimum(XL[k], np.clip(xl, -LIM, LIM)); XH[k] = np.maximum(XH[k], np.clip(xh, -LIM, LIM)); npo[k] += 1
            return XL, XH, npo, nlo, nhi
        for vi, v in enumerate("AB"):
            FMf = np.zeros((NS, Pc, LINES), bool); FXf = np.zeros((NS, Pc, LINES), bool)
            FPMf = np.zeros((NS, Pc), bool); FPXf = np.zeros((NS, Pc), bool)
            SINGLE = np.zeros((NS, Pc), bool); COST1 = np.zeros((NS, Pc), np.int64); COSTA = np.full((NS, Pc), -1, np.int64)
            for s in range(NS):
                if v == "A": Hl, Hh = HLO[s], HHI[s]
                else:
                    fz = (FRZ >= 0) & (FRZ <= s); Hl, Hh = np.where(fz, FLO, HLO[s]), np.where(fz, FHI, HHI[s])
                omin, omax = orange(Hl, Hh)
                bad("out_outside", ((outc < omin) | (outc > omax)).sum())
                slo, shi = shr(omin, omax); bad("sh_outside", ((shc < slo) | (shc > shi)).sum())
                one = slo == shi; SINGLE[s] = one
                if one.any():
                    t = slo[one][:, None]; lo16 = np.clip(rshift(omin[one], t), -LIM, LIM); hi16 = np.clip(rshift(omax[one], t), -LIM, LIM)
                    COST1[s, one] = pc(lo16 + hi16, 18) @ PWm + pc(hi16 - lo16, 18) @ PWm
                    NBL[vi, s] += int((lo16 != hi16).sum()); NSG[vi, s] += int(one.sum())
                allr = np.arange(Pc)
                if s in WIDE: rows = allr
                else:                                     # the probe: one possible water level per position
                    tt = np.full(Pc, -1, np.int64)
                    for d in range(int((shi - slo).max()) + 1):
                        k = np.flatnonzero((tt < 0) & (slo + d <= shi))
                        if k.size == 0: break
                        t = (slo[k] + d)[:, None]
                        ok = ~((rshift(omin[k], t) > LIM) | (rshift(omax[k], t) < -LIM)).any(1); tt[k[ok]] = slo[k[ok]] + d
                    bad("probe_no_level", (tt < 0).sum())
                    pMOL, pMOH, _, pnc = join2(omin, omax, tt, tt, allr)
                    pXL, pXH, pnp, _, _ = joinx(allr, pMOL, pMOH)
                    rows = np.flatnonzero((pMOL == pMOH).any(1) | (pXL == pXH).any(1) | (pnp == 0) | (pnc == 0))
                    cnt("probe_rows_sent_to_full_join", rows.size); cnt("probe_rows", Pc)
                if rows.size == 0: continue
                MOL, MOH, cost, nc = join2(omin, omax, slo, shi, rows)
                bad("no_possible_level", (nc == 0).sum()); COSTA[s, rows] = cost
                bad("mo_outside", ((moc[rows] < MOL) | (moc[rows] > MOH)).sum())
                fm = MOL == MOH; bad("mo_fixed_wrong", (fm & (MOL != moc[rows])).sum())
                XL, XH, npo, nlo, nhi = joinx(rows, MOL, MOH)
                bad("sn_outside", ((snc[rows] < nlo) | (snc[rows] > nhi)).sum()); bad("no_possible_next_level", (npo == 0).sum())
                bad("next_light_outside", ((xnc[rows] < XL) | (xnc[rows] > XH)).sum())
                fx = XL == XH; bad("next_light_fixed_wrong", (fx & (XL != xnc[rows])).sum())
                if LIGHT == "real":                       # every fixed value of the 10 texts against the forward's own
                    g = r0 + rows; q = IN10[g]
                    bad("10texts_mo_fixed_vs_forward", (fm[q] & (MOL[q] != FMO[g[q]])).sum())
                    bad("10texts_next_fixed_vs_forward", (fx[q] & (XL[q] != FXN[g[q]])).sum())
                    cnt("10texts_fixed_values_compared", int(fm[q].sum() + fx[q].sum()))
                FMf[s, rows] = fm; FXf[s, rows] = fx; FPMf[s, rows] = fm.all(1); FPXf[s, rows] = fx.all(1) & (npo == 1)
                if s in (S28, S26, S24):
                    WID[(v, s, "m")] += np.bincount(blen(MOH - MOL).ravel(), minlength=64)[:64]
                    WID[(v, s, "x")] += np.bincount(blen(XH - XL).ravel(), minlength=64)[:64]
                if v == "A" and s == S28 and r0 == 0:    # a sampled rebuild in plain integers, two positions
                    for p in rows[:2]:
                        kk = int(np.flatnonzero(rows == p)[0]); lo_m = hi_m = None
                        for tq in range(int(slo[p]), int(shi[p]) + 1):
                            l6, h6 = rshift(omin[p], tq), rshift(omax[p], tq)
                            if (l6 > LIM).any() or (h6 < -LIM).any(): continue
                            l6, h6 = np.clip(l6, -LIM, LIM)[:, None], np.clip(h6, -LIM, LIM)[:, None]
                            a1 = np.minimum(l6 * w2, h6 * w2).sum(0); a2 = np.maximum(l6 * w2, h6 * w2).sum(0)
                            m_lo = rshift(a1, -(we2 + tq)) + b2; m_hi = rshift(a2, -(we2 + tq)) + b2
                            lo_m = m_lo if lo_m is None else np.minimum(lo_m, m_lo); hi_m = m_hi if hi_m is None else np.maximum(hi_m, m_hi)
                        bad("rebuild_plain_integers_differs", (lo_m != MOL[kk]).sum() + (hi_m != MOH[kk]).sum()); cnt("rebuild_lines", LINES)
            # ---- fixed never switches off; first stages ----
            bad("fixed_switched_off", (FMf[:-1] & ~FMf[1:]).sum() + (FXf[:-1] & ~FXf[1:]).sum()
                + (FPMf[:-1] & ~FPMf[1:]).sum() + (FPXf[:-1] & ~FPXf[1:]).sum())
            def first(F): return np.where(F.any(0), np.argmax(F, axis=0), -1)
            FM, FX, FPM, FPX = first(FMf), first(FXf), first(FPMf), first(FPXf)
            if v == "A": bad("A_not_fixed_at_full", (FM < 0).sum() + (FX < 0).sum() + (FPM < 0).sum() + (FPX < 0).sum())
            else: cnt("B_not_fixed_at_full", int((FM < 0).sum() + (FX < 0).sum() + (FPM < 0).sum() + (FPX < 0).sum()))
            OUT[f"FM_{v}"][r0:r1], OUT[f"FX_{v}"][r0:r1], OUT[f"FPM_{v}"][r0:r1], OUT[f"FPX_{v}"][r0:r1] = FM, FX, FPM, FPX
            if v == "A": bad("push_at_full_vs_lens2", (COSTA[SALL] != L2P).sum() + (~SINGLE[SALL]).sum())
            # ---- work ----
            for rn, stop in (("m", FPM), ("x", FPX)):
                e = np.repeat(stop[:, None], N, 1) if v == "A" else np.minimum(FRZ, stop[:, None])
                add(f"l1_{v}_{rn}", CM[e, ar, nn].sum(1), m10); add(f"reads_{v}_{rn}", (np.minimum(e + 1, NS - 1) * 60).sum(1), m10)
                ca = COSTA[stop, np.arange(Pc)]; bad("no_push_at_stop", (ca < 0).sum())
                add(f"push1_{v}_{rn}", ca - L2P, m10)
                sii = np.array([st + int(np.argmax(SINGLE[st:, p])) for p, st in enumerate(stop)])
                bad("no_one_level_at_or_after_stop", sum(int(not SINGLE[st:, p].any()) for p, st in enumerate(stop)))
                pushes = SINGLE & (np.arange(NS)[:, None] <= sii[None, :])
                add(f"push2_{v}_{rn}", (COST1 * pushes).sum(0) - L2P, m10); add(f"npush2_{v}_{rn}", pushes.sum(0), m10)
                e2 = np.repeat(sii[:, None], N, 1) if v == "A" else np.minimum(FRZ, sii[:, None])
                add(f"l1_ii_{v}_{rn}", CM[e2, ar, nn].sum(1), m10)
                s3 = np.maximum(stop, S28)
                add(f"push3_{v}_{rn}", COSTA[S28] + np.where(s3 == SALL, COSTA[SALL], 0) - L2P, m10)
                e3 = np.repeat(s3[:, None], N, 1) if v == "A" else np.minimum(FRZ, s3[:, None])
                add(f"l1_iii_{v}_{rn}", CM[e3, ar, nn].sum(1), m10)
        del HLO, HHI, CM, FRZ, FLO, FHI
        log(f"  layer {l} positions {r0}-{r1 - 1} ({el()}s, {mb()} MB)")

    # ---- against the run ----
    Z = np.load(f"{HERE}/knife/stream_lens2_{RUNTAG}_L{l}.npz")
    for k in OUT: bad(f"vs_run_{k[:-2]}", (OUT[k] != Z[k]).sum())
    bad("vs_run_frz", (FRZALL != Z["frz"]).sum())
    if STPREV is not None: bad("vs_settle_edge_stops", (FRZALL != STPREV[f"L{l}_diag_signed_output"]).sum())
    li = list(SETPREV["layers"]).index(l)
    bad("vs_settle_full", T["full"] != int(SETPREV["full"][li])); bad("vs_settle_edge_only", T["edge_only"] != int(SETPREV["work"][li, 1, 1, 2]))
    diff = [k for k in T if int(Z[f"T_{k}"]) != T[k]]
    bad("vs_run_totals", len(diff)); cnt("totals_compared", len(T))
    if diff: log(f"  totals that differ from the run: " + ", ".join(f"{k} mine {T[k]:,} run {int(Z[f'T_{k}']):,}" for k in diff))
    bad("vs_run_nblur", (NBL != Z["T_nblur"]).sum()); bad("vs_run_nsingle", (NSG != Z["T_nsingle"]).sum())
    bad("vs_run_reach", (reach != Z["T_reach"]).sum())
    for (v, s, w), h in WID.items(): bad("vs_run_widths", (h != Z[f"WID_{v}_{s}_{w}"]).sum())
    SUM[l] = (T, T10, {k: a.copy() for k, a in OUT.items()}, WID, EX, EX10)

    # ---- the layer's report: the 10 texts, and all 648 ----
    log(f"\nlayer {l} ({el()}s, {mb()} MB); one unit of out16 carried onto a stream line lands below 2^k stream units, "
        f"k at the plate values' quartiles {REACH_MED} (middle water level {shmid}); lens 1's meetings left after D28 / D26 / D24: "
        f"all {pm(EX['tail28'], T['full'])} / {pm(EX['tail26'], T['full'])} / {pm(EX['tail24'], T['full'])}, "
        f"10 texts {pm(EX10['tail28'], T10['full'])} / {pm(EX10['tail26'], T10['full'])} / {pm(EX10['tail24'], T10['full'])}")
    for lab, sel, Tt in (("10 texts", IN10, T10), ("all 648", np.ones(P, bool), T)):
        npl = int(sel.sum())
        log(f"  [{lab}, {npl} positions] fc meeting {Tt['full']:,}; lens 2 {Tt['l2']:,} ({pm(Tt['l2'], Tt['full'])} of fc); per-edge stops {pm(Tt['edge_only'], Tt['full'])} of fc")
        log("    " + " " * 30 + "".join(f"{nm:>7s}" for nm in NAMES))
        for v in "AB":
            for key, lab2, n in (("FM", "stream value, lines", npl * LINES), ("FX", "next light, lines", npl * LINES),
                                 ("FPM", "stream value, positions", npl), ("FPX", "next light, positions", npl)):
                h = np.cumsum(np.bincount(OUT[f"{key}_{v}"][sel].ravel().astype(np.int64) % 256, minlength=256)[:NS])
                log(f"    ({v}) {lab2:26s}" + "".join(f"{pm(x, n):>7s}" for x in h))
        for v in "AB":
            for rn, lab2 in (("m", "stream value"), ("x", "next light")):
                ref = Tt["full"] if v == "A" else Tt["edge_only"]
                l1, p1, p2, p3 = Tt[f"l1_{v}_{rn}"], Tt[f"push1_{v}_{rn}"], Tt[f"push2_{v}_{rn}"], Tt[f"push3_{v}_{rn}"]
                log(f"    ({v}) stop at the {lab2}: lens 1 {pm(l1, Tt['full'])} of fc (saves {ref - l1:,}); extra (i) {p1:,}; "
                    f"(ii) {p2:,} = {pm(p2, Tt['l2'])} of lens 2 ({Tt[f'npush2_{v}_{rn}'] / npl:.2f} pushes a position); "
                    f"(iii) {p3:,} = {pm(p3, Tt['l2'])} of lens 2; lens 1 (ii) {pm(Tt[f'l1_ii_{v}_{rn}'], Tt['full'])}, (iii) {pm(Tt[f'l1_iii_{v}_{rn}'], Tt['full'])}")
    log(f"  checks so far: " + ", ".join(f"{k} {x}" for k, x in sorted(BAD.items())))

# ---------------- the layers together ----------------
log(f"\n================ LAYERS {LAYERS}, light={LIGHT} seed={SEED} ================")
for lab, idx in (("10 texts", 1), ("all 648", 0)):
    TT = {}
    for l in LAYERS:
        for k, x in SUM[l][idx].items(): TT[k] = TT.get(k, 0) + x
    npl = P10 if idx else P
    log(f"[{lab}] fc {TT['full']:,} ({TT['full'] // npl:,} a token); lens 2 {TT['l2']:,} ({pm(TT['l2'], TT['full'])} of fc); per-edge stops {pm(TT['edge_only'], TT['full'])}")
    for v in "AB":
        for rn, lab2 in (("m", "stream value"), ("x", "next light")):
            ref = TT["full"] if v == "A" else TT["edge_only"]
            l1 = TT[f"l1_{v}_{rn}"]
            log(f"  ({v}) {lab2}: lens 1 {pm(l1, TT['full'])}, saves {(ref - l1) // npl:,} a token; extra (i) {TT[f'push1_{v}_{rn}'] // npl:,}, "
                f"(ii) {TT[f'push2_{v}_{rn}'] // npl:,} ({pm(TT[f'push2_{v}_{rn}'], TT['l2'])} of lens 2, {TT[f'npush2_{v}_{rn}'] / (npl * len(LAYERS)):.2f} pushes a position a layer), "
                f"(iii) {TT[f'push3_{v}_{rn}'] // npl:,} ({pm(TT[f'push3_{v}_{rn}'], TT['l2'])} of lens 2) a token; "
                f"as a share of these layers' fc meeting: (i) {pm(ref - l1 - TT[f'push1_{v}_{rn}'], TT['full'])}, "
                f"(ii) {pm(ref - TT[f'l1_ii_{v}_{rn}'] - TT[f'push2_{v}_{rn}'], TT['full'])}, (iii) {pm(ref - TT[f'l1_iii_{v}_{rn}'] - TT[f'push3_{v}_{rn}'], TT['full'])}")
log(f"\nCHECKS: " + ", ".join(f"{k} {x}" for k, x in sorted(BAD.items())))
log(f"  counted: " + ", ".join(f"{k} {x:,}" for k, x in sorted(CNT.items())))
log(f"  every bound held, every fixed value the machine's own, everything equal to the run: {sum(BAD.values()) == 0}")
log(f"({el()}s, max memory {mb()} MB)")
