"""TAGSKIP: EVERY GATE TAGGED FLAT OR OPEN PER TOKEN, THE FLAT ONES SKIPPED, ON THE WHOLE-NUMBER GPT-2 MACHINE.

HIS WORDS (2026-10-08), the only sentences quoted as his:
  "we don't need to do the ectra. They need to be tagged and skipped per token THATS IT"
  "The gates are what SHOW the fringesz."  "THE FLAT IS THE CANCELLING."  "Then meet them and mark them."
  "Each gate shows the fringes, the fringes are the tips, the tips are the edges, the edges are the eccess photons the
  excess is the next token."

THE SESSION'S READING (the session's words, not his): the flat is tagged and skipped per token at the gates. For each
token, each of the 3,072 gates of a layer (the knife edges at the bends) is tagged FLAT or OPEN for this token. A FLAT
gate is skipped: its pile is not finished and its line of the down plate (mproj) is never met. Only OPEN gates are
finished and passed on. The tag must cost less than what it skips.

THE TAG (one rule, fixed before any run; nothing in it is read from results)
  - The knife edge's dark threshold is the edge's middle: H = 0, where the share table reads one half
    (PHI[32768] = 32768). Dark side H < 0, bright side H >= 0. H = power(pile, F) + b is the value arriving at the
    edge (base2.py's own gauge, divide and bias).
  - Carried back to the pile once per gate and token: T = the least pile with power(T, F) + b >= 0 (whole numbers,
    checked against base2's power at every gate: power(T) + b >= 0 and power(T - 1) + b < 0).
  - Each fc pile is read top diagonal first (place pairs a + b = 28, then 27, ... 0; one diagonal a stage), with the
    exact signed bound of knife/settle.py (FINDINGS section 25) on the unread part:
        unread <= sum over unread (a,b) of 2^(a+b) [min(n+_a, m+_b) + min(n-_a, m-_b)]      (same-sign pairs)
        unread >= -sum over unread (a,b) of 2^(a+b) [min(n+_a, m-_b) + min(n-_a, m+_b)]     (opposite-sign pairs)
    n+_a, n-_a: the light's lit lines at place a with up / down values (the light's own counts); m+_b, m-_b: the same
    for the gate's plate column (stored with the plate).
  - Before each diagonal is read (stage k = diagonals read, k = 0 .. 28): FLAT as soon as the read part + the upper
    bound < T (the pile is proved on the dark side); OPEN as soon as the read part - the lower bound >= T (proved on
    the bright side). A gate never proved either way before its last diagonal is finished in full and passed on
    (OPEN, its exact output).
  - The intervals only shrink (each diagonal's value lies inside its own share of the bound), so a gate proved at any
    stage is proved at stage 28. The run uses this as its fast path for the token runs (stage 28 alone: the pile less
    the place-0 x place-0 pair, and that pair's bound) and checks it against the full stage walk on the refs.

THE SKIP: a FLAT gate's output is taken as 0 and its line of mproj is not met. THIS IS NOT BIT-EXACT. The bend's dark
  side is exactly 0 in the share table only far down (PHI = 0 below x = H / 65536 = -4.32; 0.24% of edges sit there,
  section 25). Between x = -2 and 0 the bend gives -0.05 to -0.16 of a unit (x = -1: share 10,398 / 65,536). So the
  run is judged by THE TOKEN, and the skip carries forward: every later layer runs on the tagged machine's own light.

WORK (photon meetings and photons added, the unit of every earlier result: a token = the mean over the 648 positions
  of the 30 refs texts, ids + ref, teacher-forced; plain = 3,666,596,259 a token, rebuilt by this file with skip=none,
  part by part as confirm_groups/audit_gpt2_plain.py counts it):
  - fc at a skipped layer: FLAT gates, the meetings of the diagonals read before the tag (the tags); OPEN gates (proved
    early or finished in full), every meeting of the pile, its divide, its bias and its bend, as plain counts them.
  - the tag's own work, counted in photons (the session's model of a machine that runs it): the threshold T once a gate
    (ones(T) x ones(gain)); the bound built a gate from the counts (the four mins of every place pair added into the
    diagonal's bound, then the 29 diagonal bounds added into the whole bound); each stage the two interval ends
    (ones of each bound); each diagonal read, its bound taken off (ones of each). The mins are compares (a count, apart).
  - mproj at a skipped layer: the OPEN gates' outputs only meet the plate (ones(light) x ones(plate) per line, as plain);
    the lines met are counted.
  - qkv, proj, the arrival, the top read and the rest (water levels, divides, biases, stream adds): as plain counts
    them, on the tagged machine's own light. Beside it, the GROUP-MACHINE FORM: those five parts at base2_groups.py's
    confirmed counts a token (qkv 132,698,796; proj 62,809,289; arrival 8,020,119; top 80,689,701; rest 4,558,811;
    measured on the exact machine's light, borrowed here, not re-run), with fc and mproj as above in plain photons.
  - Reported against plain, against the group machine (19.40%, and 18.86% with the gates' table sizing), and as a
    multiple of the extra (6.24% of plain counted as uncancelled meetings over the whole machine).

JUDGE: judge5's 300 windows (6,000 positions): the machine's word in judge5's reference top 5, the same top word, the
  top 5 in set and order; the 30 refs texts: their 16 greedy words; and teacher-forced on the refs, the top word and
  top 5 against base2's own. Variants: the skip at all 12 layers, only at layer 11, only at layers 6-11.

CONTROLS: shuffledtags: the same skip with each position's tags shuffled across the 3,072 gates of the layer (the same
  count skipped, the wrong gates). shuffledlight: the tag rule run on the light shuffled across its 768 lines (every
  value kept; the gauge is unchanged, so T is too): (a) in the work run, count only, the flat share and the tag's cost
  on that light; (b) in the token run, those tags applied to the real machine. ORDER-FREE (order=1): the fc lines, the
  gates and the order the place pairs are added all moved: every gate's stage and tag, and every word, the same.

RULE 13, CHECKED BEFORE THE RUN
  13.1 the measure is the token (top 5, top word, greedy words) and the finished machine's work; the light's 1s enter
       only as the counted meetings and the bound's counts.
  13.2 nothing in the machine changes but the skip itself: every bend, water level, gauge, divide and arrival is
       base2.py's own, run on the tagged machine's own light.
  13.3 no agreement statistic: the token is judged against the fixed references; the controls move gates, not values.
  13.4 the tag's threshold and bound are fixed by the docstring; the run measures where each gate is proved, the share
       proved dark, what the skip does to the token, and the work.
  13.5 his words are quoted; the reading, the tag's threshold and the cost model are the session's.
  13.6 every diagonal down to place 0 x place 0; every position of every text; every window.

ANALYSIS BEFORE THE RUN (the session's)
  - From FINDINGS 25 and 28 (on the exact machine's light): the side settles by the top 12-16 diagonals (23.1% of
    lens 1 in steps of two diagonals); 83-87% of gates sit dark from depth 2 on. So about 85% of gates are tagged
    FLAT, at about a fifth of their meetings.
  - The dark gates' outputs are small but all one sign (the bend's dip below zero), so through the down plate they do
    not cancel like coin tosses: they push the stream together, layer after layer. Section 27 found the symmetric half
    (what makes a shut gate dark) as large as the straight half, with the lead a small remainder. So the token is
    expected to move when the skip is at every layer, much less when it is only at layer 11.
  - Work, in plain photons: qkv 679.7M + proj 248.5M + arrival 13.1M + top 904.0M + rest 4.6M = 1,849.9M a token are
    left as they are, already 50.45% of plain. So the task's "40-50% of plain in plain photons" cannot hold: the floor
    is 50.45% before fc and mproj. fc: about 0.85 x 21% + 0.15 x 100% of 895M meetings, about 300M; mproj: the open
    gates' share of the down plate's meetings, about 18-25% of 915M (bright outputs carry more 1s), about 165-230M; the
    tag's own: the bound built from the counts at every gate, roughly 2,000-5,000 photons a gate, 70-180M a token.
PREDICTIONS (written before any run of this file)
  THE TASK'S (the session's earlier prediction, quoted from the task): "80-87% of gates tagged flat from layer 2 on; the
    tags cost about 25% of the fc plate; the down plate falls to about 15%; whole machine about 40-50% of plain in plain
    photons (the qkv, proj and top read unchanged), lower if the open gates and the other plates use the group piles;
    the token: top-5 kept at 85-97% of judged positions and the same top word at 70-90% -- the small dark outputs of
    thousands of gates do add up; the shuffled tags lose most words."
  THIS FILE'S (the session's, from the analysis above):
    - FLAT 82-88% of gates from layer 2 on, within 1 point of the true dark share (H < 0); OPEN proved early 11-16%;
      finished in full under 0.5%. The tag rule on the shuffled light: a similar flat share (within 5 points).
    - tag reads: 18-24% of the FLAT gates' own meetings; fc at a skipped layer 30-38% of plain fc; mproj 18-25% of
      plain mproj; the tag's own 2-5% of plain.
    - the whole machine, skip at all 12 layers: 62-72% of plain in plain photons (about 10-12x the extra); in the
      group-machine form 22-28% of plain (above the group machine's 19.40%).
    - the token, skip at all 12 layers: top 5 kept at 25-65% of the 6,000 positions, the same top word at 15-45%, top 5
      in set and order under 5%, greedy words: 0-5 of 30 texts identical. Layers 6-11: top 5 50-80%, top word 35-65%.
      Layer 11 only: top 5 85-97%, top word 70-90%.
    - shuffled tags: top 5 under 5%; shuffled-light tags about the same as shuffled tags.
    - order-free: every stage and tag identical, every word identical.

ADDED AFTER THE FIRST RUNS (the session's way around the road block; written before glow=stored ran)
  The first runs lost the token at every-layer skipping (judge5 top 5 40.40%, top word 16.67%) and showed why: the
  skipped dark outputs are 70-97% of the summed size of a layer's outputs, and taking them away changes the layer's
  write by 65-128% of the write itself. His picture says the flat stays flat ("Nothing changes the flat stays flat").
  If the dark gates' small outputs are the standing flat, a stored copy can stand in for them.
  glow=stored (one rule): each gate's STORED GLOW c_i = its mean output over the refs positions where it sits dark on
    the exact machine (whole numbers, rounded to the nearest; 0 for a gate never dark), stored once with the plate. A
    FLAT gate is still skipped (its pile is not finished) but its output is taken as c_i instead of 0. In a machine the
    stored glows of all gates are met with mproj once and kept (one stored line of 768 a layer); each token then takes
    off the OPEN gates' stored glow (their lines met once more) and adds its OPEN gates' own outputs. The work counts
    that (estimated at each position's own water level), the token is judged as before. Built from the refs, so on
    the refs it is in-sample; judge5's windows are other texts.
  PREDICTION (the session's): the stored glow takes away 40-70% of the skip's push at each layer (the residual push
    30-60% of the write); the token at all 12 layers: top 5 55-80%, top word 30-55%; layers 6-11: top 5 80-92%;
    layer 11: top 5 95-99%. mproj rises by the open lines met twice: about 2x this design's mproj, still under 25% of
    plain mproj. Control: the same with the tags shuffled across the gates (bright gates get a stored glow, dark ones
    run): top 5 under 40%.

    python3 tagskip/tagskip.py mode=work skip=none                     the plain machine rebuilt, the unit and the checks
    python3 tagskip/tagskip.py mode=work skip=all shufmeas=1            the tags' stages, the work, the shuffled-light count
    python3 tagskip/tagskip.py mode=work skip=all order=1               order-free
    python3 tagskip/tagskip.py mode=work skip=L11 | skip=L6-11          the work of the partial skips
    python3 tagskip/tagskip.py mode=judge skip=all|L11|L6-11|none [tags=real|shuffledtags|shuffledlight]
      -> tagskip/<mode>_<skip>_<tags>[_order][_shufmeas].log and .json
"""
import sys, os, json, time, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")     # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/tagskip"
ARG = dict(a.split("=", 1) for a in sys.argv[1:])
MODE, SKIPN, TAGS = ARG.get("mode", "work"), ARG.get("skip", "all"), ARG.get("tags", "real")
ORDER, SHUFMEAS = int(ARG.get("order", 0)), int(ARG.get("shufmeas", 0))
SUB = [int(v) for v in ARG["texts"].split(",")] if "texts" in ARG else None       # a subset of the refs texts (tests)
NWIN = int(ARG.get("nwin", 300))
GLOW = ARG.get("glow", "zero"); assert GLOW in ("zero", "stored")
SKIPS = {"all": list(range(12)), "L11": [11], "L6-11": list(range(6, 12)), "none": []}
SKIP = set(SKIPS[SKIPN]); assert MODE in ("work", "judge") and TAGS in ("real", "shuffledtags", "shuffledlight")
NAME = f"{MODE}_{SKIPN}_{TAGS}" + ("_order" if ORDER else "") + ("_shufmeas" if SHUFMEAS else "") + \
       ("" if SUB is None else "_t" + "-".join(map(str, SUB))) + ("" if NWIN == 300 else f"_w{NWIN}") + \
       ("_glowstored" if GLOW == "stored" else "")
LOG = open(f"{OUT}/{NAME}.log", "w"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def el(): return int(time.time() - t0)

# ---------------- base2.py's own steps (its source up to its own test, exec'd the way settle.py loads it) ----------------
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')                  # the plates are read below, as int16
_argv, _cwd = sys.argv, os.getcwd(); sys.argv = ["x"]; os.chdir(HERE); J = {}; exec(_src, J); sys.argv = _argv; os.chdir(_cwd)
M, ES, SC, PHI, ARR, EXP2 = J["M"], J["ES"], J["SC"], J["PHI"], J["ARR"], J["EXP2"]
rshift, power, edge, to16, gauge_log, product, log2fix, race = (J[k] for k in
    ("rshift", "power", "edge", "to16", "gauge_log", "product", "log2fix", "race"))
PLATES = {f"{k}.{l}": (np.ascontiguousarray(M[f"{k}.{l}.w"]), M[f"{k}.{l}.e"].astype(np.int64), M[f"{k}.{l}.b"].astype(np.int64))
          for l in range(12) for k in ("qkv", "proj", "fc", "mproj")}
WTE, WPE = M["wte"].astype(np.int64), M["wpe"].astype(np.int64)
HW16, HE, HB = np.ascontiguousarray(M["head.w"]), M["head.e"].astype(np.int64), M["head.b"].astype(np.int64)
assert (HE == HE[0]).all()
HWF = None                                                                  # the word table in float64, on first use
N, LINES, PL, PW = 3072, 768, 15, 15; ND = PL + PW - 1                       # 29 diagonals
REFS = json.load(open(f"{HERE}/refs_many.json")); WINDOWS = json.load(open(f"{HERE}/judge5_texts.json"))
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
POP = np.bitwise_count(np.arange(1 << 16)).astype(np.int64)

def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def gain_ones(F): return ones(EXP2[np.bitwise_and(np.asarray(F, np.int64), 0xFFFF)])
def water(x16):                                     # the gauge's sums (audit_gpt2_plain.py's convention)
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    return (ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2)
def meet(x16, w16): return int((ones(x16).astype(np.float64) @ ones(w16).sum(1).astype(np.float64)).sum())
def share_of(H):
    idx = rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
def pm(a, b): return f"{100.0 * a / b:.2f}%" if b else "-"

# ---------------- the threshold: the least pile on the bright side ----------------
def threshold(F, b):
    """T (P, N): the least pile X with power(X, F) + b >= 0; and its photons ones(T) x ones(gain)"""
    I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF); E = EXP2[f].astype(np.int64); s = 16 - I
    Bb = np.broadcast_to(b[None, :], F.shape)
    assert int(s.max()) <= 40 and int(np.abs(b).max()) < 1 << 21 and int((-s).max()) <= 20
    sp = np.maximum(s, 0); sn = np.maximum(-s, 0)
    num = np.where(s > 0, -Bb * (np.int64(1) << sp) - np.where(s > 0, np.int64(1) << np.maximum(sp - 1, 0), 0), -Bb)
    den = np.where(s > 0, E, E << sn)
    T = -((-num) // den)                                                     # the ceiling
    ok_hi = power(T, F) + Bb >= 0; ok_lo = power(T - 1, F) + Bb < 0
    assert ok_hi.all() and ok_lo.all(), "threshold"
    return T, ones(T) * ones(E)

# ---------------- the light's and the plate's photon counts per place ----------------
def counts_light(x16):
    ax, sx = np.abs(x16), np.sign(x16); bx = (ax[None] >> np.arange(PL)[:, None, None]) & 1      # (15, P, 768)
    return (bx * (sx > 0)).sum(2), (bx * (sx < 0)).sum(2)                    # nP, nM: (15, P)
_PCOUNT = {}
def plate_counts(l):                                # the plate's lit lines per place, up and down values: stored once
    if l not in _PCOUNT:
        w16 = PLATES[f"fc.{l}"][0]; aw, up, dn = np.abs(w16.astype(np.int64)), w16 > 0, w16 < 0
        mP = np.zeros((PW, N), np.int64); mM = np.zeros((PW, N), np.int64)
        for b in range(PW):
            bit = ((aw >> b) & 1).astype(bool); mP[b] = (bit & up).sum(0); mM[b] = (bit & dn).sum(0)
        _PCOUNT[l] = (mP, mM)
    return _PCOUNT[l]

def tag_fast(x16, w16, pile, T, mP, mM):
    """the tags at stage 28 (every diagonal read but place 0 x place 0): FLAT, OPEN-by-then; nested intervals make
    this equal to the full stage walk's result"""
    nP, nM = counts_light(x16)
    X0 = (np.sign(x16) * (np.abs(x16) & 1)).astype(np.float32); W0 = (np.sign(w16) * (np.abs(w16.astype(np.int64)) & 1)).astype(np.float32)
    D0 = (X0 @ W0).astype(np.int64); c = pile - D0
    up = np.minimum(nP[0][:, None], mP[0][None, :]) + np.minimum(nM[0][:, None], mM[0][None, :])
    dn = np.minimum(nP[0][:, None], mM[0][None, :]) + np.minimum(nM[0][:, None], mP[0][None, :])
    return c + up < T, c - dn >= T

def tag_stages(x16, w16, T, Tph, pile, mP, mM, pair_order=None, chunk=216):
    """the full walk, diagonal by diagonal from the top, every gate; returns per gate: stage (0..28 proved, 29 finished
    in full), kind (0 FLAT, 1 OPEN proved, 2 finished in full), reads (meetings of the diagonals read before the
    stage), full meetings, and the tag's own photons by item; with checks"""
    P = x16.shape[0]; aw = np.abs(w16.astype(np.int64)); sw = np.sign(w16).astype(np.int64)
    pairs = [(a, b) for b in range(PW) for a in range(PL)] if pair_order is None else pair_order
    stage = np.full((P, N), 29, np.int8); kind = np.full((P, N), 2, np.int8)
    reads = np.zeros((P, N), np.int64); fullm = np.zeros((P, N), np.int64)
    own = {k: np.zeros((P, N), np.int64) for k in ("threshold", "bound built", "interval ends", "bound taken off")}
    own["threshold"] += Tph
    chk = {"diagonals = the pile": 0, "meetings = plain": 0, "FLAT with the pile above T": 0, "OPEN with the pile below T": 0,
           "end of walk: read part = pile, bounds 0": 0, "pair-by-pair pieces": 0, "pieces checked": 0}
    for r0 in range(0, P, chunk):
        r1 = min(P, r0 + chunk); Pc = r1 - r0; xc = x16[r0:r1]; ax, sx = np.abs(xc), np.sign(xc)
        Xs = [(sx * ((ax >> a) & 1)).astype(np.float32) for a in range(PL)]
        Xu = [((ax >> a) & 1).astype(np.float32) for a in range(PL)]
        D = np.zeros((ND, Pc, N), np.int16); MD = np.zeros((ND, Pc, N), np.int16)
        cb = -1
        for a, b in pairs:
            if b != cb:
                Wbit = (aw >> b) & 1; Ws = (sw * Wbit).astype(np.float32); Wu = Wbit.astype(np.float32); cb = b
            D[a + b] += (Xs[a] @ Ws).astype(np.int16); MD[a + b] += (Xu[a] @ Wu).astype(np.int16)    # |sums| <= 768: exact
        del Xs, Xu
        pc = np.zeros((Pc, N), np.int64)
        for j in range(ND): pc += D[j].astype(np.int64) << j
        chk["diagonals = the pile"] += int((pc != pile[r0:r1]).sum())
        fm = MD.sum(0, dtype=np.int64)
        chk["meetings = plain"] += int((fm != (ones(xc).astype(np.float64) @ ones(w16).astype(np.float64)).astype(np.int64)).sum())
        fullm[r0:r1] = fm
        rs = np.random.default_rng(31 + r0)                                   # pair by pair, line by line, plain integers
        for _ in range(2):
            p, n = int(rs.integers(Pc)), int(rs.integers(N))
            for j in range(ND):
                want = sum(int((sx[p] * sw[:, n] * ((ax[p] >> a) & 1) * ((aw[:, n] >> (j - a)) & 1)).sum())
                           for a in range(max(0, j - PW + 1), min(PL, j + 1)))
                chk["pair-by-pair pieces"] += int(D[j, p, n]) != want; chk["pieces checked"] += 1
        # the bound, built a gate from the counts
        nP, nM = counts_light(xc)
        ub = np.zeros((ND, Pc, N), np.int16); db = np.zeros((ND, Pc, N), np.int16); built = np.zeros((Pc, N), np.int64)
        for a in range(PL):
            for b in range(PW):
                pp = np.minimum(nP[a][:, None], mP[b][None, :]); mm = np.minimum(nM[a][:, None], mM[b][None, :])
                pq = np.minimum(nP[a][:, None], mM[b][None, :]); mq = np.minimum(nM[a][:, None], mP[b][None, :])
                ub[a + b] += (pp + mm).astype(np.int16); db[a + b] += (pq + mq).astype(np.int16)
                built += POP[pp] + POP[mm] + POP[pq] + POP[mq]
        U = np.zeros((Pc, N), np.int64); Dn = np.zeros((Pc, N), np.int64)
        for j in range(ND):
            U += ub[j].astype(np.int64) << j; Dn += db[j].astype(np.int64) << j
            built += POP[ub[j]] + POP[db[j]]
        own["bound built"][r0:r1] += built; del built
        # the walk
        Tc = T[r0:r1]; c = np.zeros((Pc, N), np.int64); cm = np.zeros((Pc, N), np.int64); und = np.ones((Pc, N), bool)
        st, kd, rd = stage[r0:r1], kind[r0:r1], reads[r0:r1]
        ie, bt = own["interval ends"][r0:r1], own["bound taken off"][r0:r1]
        for k in range(ND):                                                 # k diagonals read, k = 0 .. 28
            ie += np.where(und, ones(U) + ones(Dn), 0)
            fl = und & (c + U < Tc); op = und & (c - Dn >= Tc)
            st[fl] = k; kd[fl] = 0; rd[fl] = cm[fl]; st[op] = k; kd[op] = 1
            und &= ~(fl | op)
            j = ND - 1 - k
            c += D[j].astype(np.int64) << j; U -= ub[j].astype(np.int64) << j; Dn -= db[j].astype(np.int64) << j
            cm += MD[j]
            bt += np.where(und, POP[ub[j]] + POP[db[j]], 0)
        chk["end of walk: read part = pile, bounds 0"] += int(((c != pile[r0:r1]) | (U != 0) | (Dn != 0)).sum())
        pg = pile[r0:r1]
        chk["FLAT with the pile above T"] += int(((kd == 0) & (pg >= Tc)).sum())
        chk["OPEN with the pile below T"] += int(((kd == 1) & (pg < Tc)).sum())
        del D, MD, ub, db, c, cm, U, Dn
    return stage, kind, reads, fullm, own, chk

# ---------------- the arrival (base2.py's, all positions of one text at once; whole numbers, identical) ----------------
def arrival(QKV, C):
    P = QKV.shape[0]; out = np.zeros((P, 768), np.int64); tri = np.tril(np.ones((P, P), bool))
    for h in range(12):
        q, k, v = QKV[:, h * 64:(h + 1) * 64], QKV[:, 768 + h * 64:768 + (h + 1) * 64], QKV[:, 1536 + h * 64:1536 + (h + 1) * 64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        mx = np.where(tri, sc, np.iinfo(np.int64).min).max(1)                # the first to arrive (race: the largest)
        d = mx[:, None] - sc
        a = np.where(tri & (d < len(ARR)), ARR[np.clip(d, 0, len(ARR) - 1)], 0)
        num = product(a, v.astype(np.float64)); Lp = log2fix(a.sum(1))
        out[:, h * 64:(h + 1) * 64] = power(num, -Lp[:, None])
        if C is not None:
            C["arrival"] += int(np.tril(ones(q16) @ ones(k16).T).sum())
            C["arrival"] += int(ones(a).sum()) + int((ones(num).sum(1) * gain_ones(-Lp)).sum()) + int((ones(a) @ ones(v).sum(1)).sum())
    return out

# ---------------- the machine, with the tags ----------------
def forward(texts, C=None, top="all", rec=None, seed=0, skipset=None, store=None):
    """texts: lists of ids. top: 'all' | 'judged' (positions 3..22 of each) | 'last'. C: the counts (work mode).
    rec: per-layer records (work mode). Returns the scores' top 5 and race winners at the rows asked for."""
    global HWF
    npos = np.array([len(t) for t in texts]); off = np.concatenate([[0], np.cumsum(npos)]); NP = int(off[-1])
    ids = np.concatenate([np.asarray(t, np.int64) for t in texts]); pos = np.concatenate([np.arange(n) for n in npos])
    R = WTE[ids] + WPE[pos]
    if C is not None: C["rest"] += int(ones(WPE[pos]).sum())
    rng = np.random.default_rng(1000 + seed); SK = SKIP if skipset is None else skipset
    for l in range(12):
        # qkv, the arrival, proj: base2's own
        w16, we, b = PLATES[f"qkv.{l}"]; x16, ex, LS = gauge_log(R); pile = product(x16, w16.astype(np.float64))
        QKV = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        if C is not None:
            C["qkv"] += meet(x16, w16); C["rest"] += int(water(x16).sum() + (ones(pile).sum(1) * gain_ones(-LS)).sum() + NP * ones(b).sum())
        if rec is not None and "ref_rows" in rec: rec["chk"][f"qkv pile L{l} vs base2's run (rows)"] = int((pile != REF[f"qkv_pile_{l}"][rec["ref_rows"]]).any(1).sum()) if l <= rec["first_skip"] else None
        o = np.concatenate([arrival(QKV[off[t]:off[t + 1]], C) for t in range(len(texts))]); del QKV
        w16, we, b = PLATES[f"proj.{l}"]; o16, eo = to16(o, -ES); pile = product(o16, w16.astype(np.float64))
        att = rshift(pile, -(we[None, :] + eo[:, None] + ES)) + b
        if C is not None: C["proj"] += meet(o16, w16); C["rest"] += int(NP * ones(b).sum() + ones(att).sum())
        Ra = R + att
        if rec is not None and "ref_rows" in rec and l <= rec["first_skip"]:
            rec["chk"][f"stream after the arrival L{l} vs base2's run (rows)"] = int((Ra != REF[f"stream_a_{l}"][rec["ref_rows"]]).any(1).sum())
        # fc: the gates
        w16, we, b = PLATES[f"fc.{l}"]; x16, ex, LS = gauge_log(Ra); pile = product(x16, w16.astype(np.float64))
        F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]; H = power(pile, F) + b
        sh = share_of(H); G = rshift(H * sh, 16)                              # base2's edge
        if C is not None: C["rest"] += int(water(x16).sum())
        if store is not None:
            dk = H < 0; store[l][0] += np.where(dk, G, 0).sum(0); store[l][1] += dk.sum(0)
        if rec is not None and "ref_rows" in rec and l <= rec["first_skip"]:
            rec["chk"][f"fc pile L{l} vs base2's run (rows)"] = int((pile != REF[f"fc_pile_{l}"][rec["ref_rows"]]).any(1).sum())
        plain_fc_gate = None
        if C is not None:
            fullm = (ones(x16).astype(np.float64) @ ones(w16).astype(np.float64)).astype(np.int64)
            plain_fc_gate = fullm + ones(pile) * gain_ones(-LS)[:, None] + ones(b)[None, :] + ones(H) * ones(sh)
            C["plain fc on this light"] += int(plain_fc_gate.sum())
        if l in SK:
            mP, mM = plate_counts(l); T, Tph = threshold(F, b)
            flat, open28 = tag_fast(x16, w16, pile, T, mP, mM)
            if rec is not None:                                              # the full walk (work mode)
                L = rec["layers"].setdefault(l, {})
                if ORDER:                                                    # lines, gates and pair order all moved
                    LP = rng.permutation(LINES); NPm = rng.permutation(N)
                    x16p, exp_, LSp = gauge_log(Ra[:, LP]); assert np.array_equal(exp_, ex) and np.array_equal(LSp, LS)
                    w16p = np.ascontiguousarray(w16[LP][:, NPm]); pilep = product(x16p, w16p.astype(np.float64))
                    assert np.array_equal(pilep, pile[:, NPm])
                    pairs = [(a, bb) for bb in range(PW) for a in range(PL)]; pairs = [pairs[i] for i in rng.permutation(len(pairs))]
                    mPp, mMp = mP[:, NPm], mM[:, NPm]
                    sp_, kp_, rp_, fp_, op_, ck = tag_stages(x16p, w16p, T[:, NPm], Tph[:, NPm], pilep, mPp, mMp, pairs)
                    stage = np.empty_like(sp_); stage[:, NPm] = sp_; kind = np.empty_like(kp_); kind[:, NPm] = kp_
                    reads = np.empty_like(rp_); reads[:, NPm] = rp_; fullw = np.empty_like(fp_); fullw[:, NPm] = fp_
                    own = {}
                    for kk, vv in op_.items(): own[kk] = np.empty_like(vv); own[kk][:, NPm] = vv
                else:
                    stage, kind, reads, fullw, own, ck = tag_stages(x16, w16, T, Tph, pile, mP, mM)
                for kk, vv in ck.items(): rec["chk"][f"L{l} {kk}"] = vv
                rec["chk"][f"L{l} the walk's FLAT = the fast path's (gates differing)"] = int(((kind == 0) != flat).sum())
                rec["chk"][f"L{l} the walk's OPEN by stage 28 = the fast path's (gates differing)"] = int(((kind == 1) != open28).sum())
                rec["chk"][f"L{l} full meetings = plain's (gates differing)"] = int((fullw != fullm).sum())
                L["digest"] = hashlib.sha256(stage.tobytes() + kind.tobytes()).hexdigest()
                L["P"] = int(stage.size); L["flat"] = int((kind == 0).sum()); L["open"] = int((kind == 1).sum()); L["full"] = int((kind == 2).sum())
                L["dark"] = int((H < 0).sum()); L["far_dark_exact0"] = int((sh == 0).sum())
                L["stage_hist_flat"] = np.bincount(stage[kind == 0].astype(np.int64), minlength=30).tolist()
                L["stage_hist_open"] = np.bincount(stage[kind == 1].astype(np.int64), minlength=30).tolist()
                L["reads_flat"] = int(reads[kind == 0].sum()); L["fullm_flat"] = int(fullw[kind == 0].sum())
                L["fullm_all"] = int(fullw.sum())
                L["own"] = {kk: int(vv.sum()) for kk, vv in own.items()}
                L["compares"] = int(stage.size) * 900 + int(2 * np.minimum(stage.astype(np.int64) + 1, 29).sum())
                if SHUFMEAS:                                                 # the control: the rule on the light shuffled across its lines
                    rs = np.random.default_rng(2026 + l); x16s = np.stack([r[rs.permutation(LINES)] for r in x16])
                    piles = product(x16s, w16.astype(np.float64))
                    ss, ks, rds, fms, ows, cks = tag_stages(x16s, w16, T, Tph, piles, mP, mM)
                    for kk, vv in cks.items(): rec["chk"][f"L{l} shuffled light: {kk}"] = vv
                    Hs = power(piles, F) + b
                    L["shuf"] = dict(flat=int((ks == 0).sum()), open=int((ks == 1).sum()), full=int((ks == 2).sum()), dark=int((Hs < 0).sum()),
                                     reads_flat=int(rds[ks == 0].sum()), fullm_flat=int(fms[ks == 0].sum()), fullm_all=int(fms.sum()),
                                     own={kk: int(vv.sum()) for kk, vv in ows.items()},
                                     stage_hist_flat=np.bincount(ss[ks == 0].astype(np.int64), minlength=30).tolist())
                    del x16s, piles, Hs
                if C is not None:
                    C["fc: the tags (FLAT gates' diagonals read)"] += int(reads[kind == 0].sum())
                    C["fc: OPEN gates finished (meetings, divide, bias, bend)"] += int(plain_fc_gate[kind != 0].sum())
                    for kk, vv in own.items(): C[f"tag own: {kk}"] += int(vv.sum())
                flat = kind == 0
            if TAGS == "shuffledtags":
                flat = np.stack([f_[rng.permutation(N)] for f_ in flat])
            elif TAGS == "shuffledlight":
                rs = np.random.default_rng(2026 + l + 100 * seed); x16s = np.stack([r[rs.permutation(LINES)] for r in x16])
                flat, _ = tag_fast(x16s, w16, product(x16s, w16.astype(np.float64)), T, mP, mM)
            Gs = np.where(flat, CST[l][None, :], G) if GLOW == "stored" else np.where(flat, 0, G)
            if rec is not None:
                L = rec["layers"][l]
                L["glow_skipped"] = int(np.abs(G[flat]).sum()); L["out_all"] = int(np.abs(G).sum())
                L["flat_used"] = int(flat.sum()); L["glow_residual"] = int(np.abs(G - Gs)[flat].sum())
        else:
            Gs = G
            if C is not None: C["fc: plain (layer not skipped)"] += int(plain_fc_gate.sum())
        # mproj: the open gates' lines only
        w16, we, b = PLATES[f"mproj.{l}"]; g16, eg = to16(Gs, -ES); pile = product(g16, w16.astype(np.float64))
        mo = rshift(pile, -(we[None, :] + eg[:, None] + ES)) + b
        if C is not None:
            C["rest"] += int(NP * ones(b).sum() + ones(mo).sum())
            if l in SK and GLOW == "stored":                                 # the open lines met, their stored glow taken off,
                C["mproj"] += meet(np.where(flat, 0, g16), w16)              # the stored glow line added (at the position's level)
                c16 = rshift(np.broadcast_to(CST[l][None, :], Gs.shape), (eg + ES)[:, None])
                C["mproj: the OPEN lines' stored glow taken off"] += meet(np.where(flat, 0, c16), w16)
                C["mproj: the stored glow line added"] += int(ones(product(c16, w16.astype(np.float64))).sum())
            else:
                C["mproj"] += meet(g16, w16)
            if l in SK:
                C["mproj lines met"] += int((~flat).sum()); C["mproj lines, plain"] += int(flat.size)
                gx, egx = to16(G, -ES); C["plain mproj on this light"] += meet(gx, w16)
                if rec is not None:
                    mox = rshift(product(gx, w16.astype(np.float64)), -(we[None, :] + egx[:, None] + ES)) + b
                    rec["layers"][l]["push_of_skip"] = int(np.abs(mox - mo).sum()); rec["layers"][l]["write"] = int(np.abs(mox).sum())
            else:
                C["plain mproj on this light"] += meet(g16, w16)
        R = Ra + mo
        if rec is not None and "ref_rows" in rec and l < rec["first_skip"]:
            rec["chk"][f"stream after L{l} vs base2's run (rows)"] = int((R != REF[f"stream_{l}"][rec["ref_rows"]]).any(1).sum())
        if rec is not None: log(f"    layer {l:2d} done ({el()}s, {mb()} MB)")
    # the top read (plain, base2's own)
    rows = {"all": np.arange(NP), "last": off[1:] - 1,
            "judged": np.concatenate([np.arange(off[t] + 3, off[t] + 23) for t in range(len(texts))])}[top]
    x16, ex, LS = gauge_log(R[rows]); Fp = (int(HE[0]) + ex + ES) * 65536 - LS
    if C is not None:
        C["rest"] += int(water(x16).sum())
        pph = ones(HW16).sum(1); C["top"] += int((ones(x16).astype(np.float64) @ pph.astype(np.float64)).sum()) + len(rows) * int(ones(HB).sum())
    if HWF is None: HWF = HW16.astype(np.float64)
    T5, WIN = [], []
    for p0 in range(0, len(rows), 200):
        PLt = product(x16[p0:p0 + 200], HWF); S = power(PLt, Fp[p0:p0 + 200, None]) + HB[None, :]
        if C is not None: C["top"] += int((ones(PLt).sum(1) * gain_ones(Fp[p0:p0 + 200])).sum())
        idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
        T5.append(np.take_along_axis(idx, order, 1)); WIN.append(np.array([race(s) for s in S]))
    return np.concatenate(T5), np.concatenate(WIN), R

CST = {}
if GLOW == "stored":                                     # each gate's stored glow, from the exact machine on the refs
    _st = {l: [np.zeros(N, np.int64), np.zeros(N, np.int64)] for l in range(12)}
    forward([r["ids"] + r["ref"] for r in REFS], top="last", skipset=set(), store=_st)
    for l in range(12):
        s_, n_ = _st[l]; CST[l] = np.where(n_ > 0, (2 * s_ + n_) // np.maximum(2 * n_, 1), 0)
    np.savez(f"{OUT}/glow_store.npz", **{f"c{l}": CST[l] for l in range(12)}, **{f"n{l}": _st[l][1] for l in range(12)})
    log(f"stored glow built from the exact machine on the 30 refs texts: per layer, the mean stored glow (units of 2^-16) "
        + ", ".join(f"L{l} {CST[l].mean():.0f}" for l in range(12)))

PLAIN = 3666596259; EXTRA = 0.0624; GROUP = 711431068; GROUP2 = 691.5e6
GROUP_PARTS = {"qkv": 132698796, "proj": 62809289, "arrival": 8020119, "top": 80689701, "rest": 4558811}

# ============================================ WORK MODE ============================================
if MODE == "work":
    from collections import defaultdict
    ti = list(range(30)) if SUB is None else SUB
    texts = [REFS[t]["ids"] + REFS[t]["ref"] for t in ti]
    npos_all = REF["npos"]; offr = np.concatenate([[0], np.cumsum(npos_all)])
    assert [len(t) for t in [r["ids"] + r["ref"] for r in REFS]] == list(npos_all)
    ref_rows = np.concatenate([np.arange(offr[t], offr[t + 1]) for t in ti]); NP = len(ref_rows)
    C = defaultdict(int); rec = {"layers": {}, "chk": {}, "ref_rows": ref_rows, "first_skip": min(SKIP) if SKIP else 12}
    log(f"TAGSKIP work: skip at layers {sorted(SKIP)}, tags={TAGS}, order={ORDER}, shuffled-light count={SHUFMEAS}; "
        f"{len(texts)} refs texts, {NP} positions, teacher-forced")
    T5, WIN, Rtop = forward(texts, C=C, top="all", rec=rec)
    # the exact machine's top on the same positions, from base2's own saved light at the top
    x16, ex, LS = REF["x16"][ref_rows].astype(np.int64), REF["ex"][ref_rows], REF["LS"][ref_rows]
    Fp = (int(HE[0]) + ex + ES) * 65536 - LS; E5, EW = [], []
    for p0 in range(0, NP, 200):
        S = power(product(x16[p0:p0 + 200], HWF), Fp[p0:p0 + 200, None]) + HB[None, :]
        idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
        E5.append(np.take_along_axis(idx, order, 1)); EW.append(np.array([race(s) for s in S]))
    E5, EW = np.concatenate(E5), np.concatenate(EW)
    if not SKIP:
        rec["chk"]["the light at the top vs base2's run (positions)"] = int((gauge_log(Rtop)[0] != x16).any(1).sum())
    same_w = int((WIN == EW).sum()); in5 = int((E5 == WIN[:, None]).any(1).sum()); same5 = int((T5 == E5).all(1).sum())
    parts = ["qkv", "proj", "arrival", "top", "rest"]
    fcp = [k for k in C if k.startswith("fc:")]; ownp = [k for k in C if k.startswith("tag own:")]
    mpp = [k for k in C if k.startswith("mproj") and k not in ("mproj lines met", "mproj lines, plain")]
    total = sum(C[k] for k in parts) + sum(C[k] for k in fcp) + sum(C[k] for k in ownp) + sum(C[k] for k in mpp)
    log("")
    log("CHECKS: " + ", ".join(f"{k} {v}" for k, v in rec["chk"].items() if v is not None))
    nbad = sum(v for k, v in rec["chk"].items() if v is not None and k.split(" ", 1)[-1] != "pieces checked" and not k.endswith("pieces checked"))
    log(f"  every check 0: {nbad == 0}")
    log("")
    log(f"THE TOKEN (teacher-forced on the refs, against base2's own top read): same top word {same_w}/{NP}, "
        f"in base2's top 5 {in5}/{NP}, top 5 in set and order {same5}/{NP}")
    log("")
    log("THE GATES, LAYER BY LAYER (share of the layer's gates x positions)")
    log(f"  {'layer':5s} {'FLAT':>7s} {'OPEN proved':>11s} {'in full':>8s} {'dark H<0':>9s} {'exact 0':>8s} {'FLAT stage med':>14s} "
        f"{'tags/FLAT meet':>14s} {'glow skipped':>12s} {'left/glow':>9s} {'push/write':>10s}" + ("   shuffled light: FLAT, tags/FLAT meet" if SHUFMEAS else ""))
    for l in sorted(rec["layers"]):
        L = rec["layers"][l]; h = np.array(L["stage_hist_flat"]); med = int(np.searchsorted(np.cumsum(h), (h.sum() + 1) // 2)) if h.sum() else -1
        row = (f"  {l:5d} {pm(L['flat'], L['P']):>7s} {pm(L['open'], L['P']):>11s} {pm(L['full'], L['P']):>8s} {pm(L['dark'], L['P']):>9s} "
               f"{pm(L['far_dark_exact0'], L['P']):>8s} {'D' + str(med):>14s} {pm(L['reads_flat'], L['fullm_flat']):>14s} "
               f"{pm(L['glow_skipped'], L['out_all']):>12s} {pm(L['glow_residual'], L['glow_skipped']):>9s} {pm(L['push_of_skip'], L['write']):>10s}")
        if SHUFMEAS: row += f"   {pm(L['shuf']['flat'], L['P'])}, {pm(L['shuf']['reads_flat'], L['shuf']['fullm_flat'])}"
        log(row)
    if rec["layers"]:
        tot = lambda k: sum(rec["layers"][l][k] for l in rec["layers"])
        log(f"  all skipped layers: FLAT {pm(tot('flat'), tot('P'))}, OPEN proved {pm(tot('open'), tot('P'))}, in full "
            f"{pm(tot('full'), tot('P'))}, dark {pm(tot('dark'), tot('P'))}; the tags read {pm(tot('reads_flat'), tot('fullm_flat'))} "
            f"of the FLAT gates' meetings, {pm(tot('reads_flat'), tot('fullm_all'))} of the fc meetings")
        hs = np.sum([rec["layers"][l]["stage_hist_flat"] for l in rec["layers"]], 0); cs = np.cumsum(hs) / max(hs.sum(), 1)
        log("  FLAT gates proved by stage (cumulative): " + ", ".join(f"D{k} {100 * cs[k]:.1f}%" for k in (8, 10, 12, 14, 16, 18, 20, 24, 28)))
        if SHUFMEAS:
            st = lambda k: sum(rec["layers"][l]["shuf"][k] for l in rec["layers"])
            so = sum(sum(rec["layers"][l]["shuf"]["own"].values()) for l in rec["layers"])
            ro = sum(sum(rec["layers"][l]["own"].values()) for l in rec["layers"])
            log(f"  CONTROL, the rule on the light shuffled across its lines (count only): FLAT {pm(st('flat'), tot('P'))} "
                f"(dark {pm(st('dark'), tot('P'))}), the tags read {pm(st('reads_flat'), st('fullm_flat'))} of the FLAT gates' "
                f"meetings; the tag's own {so / NP:,.0f} a token against the real light's {ro / NP:,.0f}")
    log("")
    log("WORK A TOKEN (photon meetings and photons added; plain = 3,666,596,259)")
    for k in parts + sorted(fcp) + sorted(ownp) + sorted(mpp):
        log(f"  {k:58s} {C[k] / NP:16,.0f}  {pm(C[k] / NP, PLAIN):>8s} of plain")
    log(f"  {'THE WHOLE MACHINE':58s} {total / NP:16,.0f}  {pm(total / NP, PLAIN):>8s} of plain; {total / NP / (EXTRA * PLAIN):.2f}x the extra; "
        f"{total / NP / GROUP:.2f}x the group machine")
    if SKIP:
        fcw = sum(C[k] for k in fcp) + sum(C[k] for k in ownp)
        log(f"  beside it, on the same light: plain fc {C['plain fc on this light'] / NP:,.0f}, this fc with the tag's own "
            f"{fcw / NP:,.0f} ({pm(fcw, C['plain fc on this light'])}); plain mproj {C['plain mproj on this light'] / NP:,.0f}, "
            f"this mproj {sum(C[k] for k in mpp) / NP:,.0f} ({pm(sum(C[k] for k in mpp), C['plain mproj on this light'])}); mproj lines met "
            f"{pm(C['mproj lines met'], C['mproj lines, plain'])} of the skipped layers' lines")
        gform = sum(GROUP_PARTS.values()) + (fcw + sum(C[k] for k in mpp)) / NP
        log(f"  GROUP-MACHINE FORM (qkv, proj, arrival, top, rest at base2_groups' confirmed counts; fc, the tag's own and "
            f"mproj as above): {gform:,.0f} a token = {pm(gform, PLAIN)} of plain; {gform / (EXTRA * PLAIN):.2f}x the extra; "
            f"{gform / GROUP:.2f}x the group machine (19.40%), {gform / GROUP2:.2f}x its table-sized form (18.86%)")
        log(f"  the compares inside the tag (the bound's mins and the stage checks; a count, apart): "
            f"{sum(rec['layers'][l]['compares'] for l in rec['layers']) / NP:,.0f} a token")
    else:
        log(f"  the plain machine rebuilt: {total / NP:,.0f} a token; = 3,666,596,259 exactly: {total == PLAIN * NP} "
            f"(sum {total:,} against {PLAIN * NP:,})")
    js = dict(name=NAME, skip=sorted(SKIP), tags=TAGS, order=ORDER, npos=NP, counts=dict(C), checks=rec["chk"],
              layers={str(k): v for k, v in rec["layers"].items()}, token=dict(same_top=same_w, in_top5=in5, top5_set_order=same5),
              total=total, top5=T5.tolist(), win=WIN.tolist(), glow=GLOW)
    json.dump(js, open(f"{OUT}/{NAME}.json", "w"))
    if ORDER:
        try:
            ref = json.load(open(f"{OUT}/{NAME.replace('_order', '')}.json"))
            dsame = sum(ref["layers"][str(l)]["digest"] == rec["layers"][l]["digest"] for l in rec["layers"])
            log(f"ORDER-FREE: every gate's stage and tag identical to the run in machine order: {dsame}/{len(rec['layers'])} layers; "
                f"top 5 identical at {int((np.array(ref['top5']) == T5).all(1).sum())}/{NP}; counts identical: "
                f"{all(ref['counts'].get(k) == v for k, v in C.items())}")
        except FileNotFoundError:
            log("ORDER-FREE: the machine-order run's json is not there yet")
    log(f"({el()}s, max memory {mb()} MB)")

# ============================================ JUDGE MODE ============================================
else:
    ref5 = np.load(f"{HERE}/judge5_ref.npz")["top5"]
    log(f"TAGSKIP judge: skip at layers {sorted(SKIP)}, tags={TAGS}; judge5's {NWIN} windows and the 30 refs texts' greedy words")
    inn = same = so5 = tot = 0
    for w0 in range(0, NWIN, 50):
        wins = WINDOWS[w0:min(NWIN, w0 + 50)]
        T5, WIN, _ = forward([w["ids"] for w in wins], top="judged", seed=w0)
        rf = ref5[w0:w0 + len(wins)].reshape(-1, 5)
        inn += int((rf == WIN[:, None]).any(1).sum()); same += int((rf[:, 0] == WIN).sum()); so5 += int((rf == T5).all(1).sum()); tot += len(WIN)
        log(f"  windows {w0}-{w0 + len(wins)}: in the top 5 {inn}/{tot}, same top word {same}/{tot}, top 5 in set and order {so5}/{tot} ({el()}s, {mb()} MB)")
    seqs = [list(r["ids"]) for r in REFS]; gen = [[] for _ in REFS]
    for s in range(16):
        _, WIN, _ = forward([sq + g for sq, g in zip(seqs, gen)], top="last", seed=5000 + s)
        for i, w in enumerate(WIN): gen[i].append(int(w))
    want = [r["ref"] for r in REFS]
    gs = sum(int(a == b) for g, wv in zip(gen, want) for a, b in zip(g, wv))
    gt = sum(g == wv for g, wv in zip(gen, want))
    first = [next((i for i, (a, b) in enumerate(zip(g, wv)) if a != b), 16) for g, wv in zip(gen, want)]
    log("")
    log(f"JUDGE5 ({tot} positions): in the reference top 5 {inn}/{tot} ({pm(inn, tot)}), same top word {same}/{tot} ({pm(same, tot)}), "
        f"top 5 in set and order {so5}/{tot} ({pm(so5, tot)})")
    log(f"GREEDY, the 30 refs texts x 16 words: words equal at their place {gs}/480, texts identical {gt}/30, "
        f"words before the first difference: median {int(np.median(first))}, {sum(first)}/480 in all")
    json.dump(dict(name=NAME, skip=sorted(SKIP), tags=TAGS, glow=GLOW, in_top5=inn, same_top=same, top5_set_order=so5, judged=tot,
                   greedy_words=gs, greedy_texts=gt, first_diff=first, gen=gen), open(f"{OUT}/{NAME}.json", "w"))
    log(f"({el()}s, max memory {mb()} MB)")
