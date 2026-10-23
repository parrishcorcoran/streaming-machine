"""THE TOP TABLE READ BAND BY BAND: WAY 2'S WALK ONCE, EACH EXPOSURE POURED FROM ITS TOP PLACES DOWN, AND AFTER EVERY
BAND THE WORDS THAT CAN NO LONGER REACH THE TOP 5 SET ASIDE. Exact, whole numbers only.

THE PICTURE (restated; the session's words, not his).
  - Way 2 (lineup2/): each line of the top table has its 50,257 weights lined up once. Per token the light on line i is
    one whole number x_i; walking up the ranks, r_k = r_(k-1) + x_i * gap_k is exactly x_i times the weight of the word
    at rank k: that word's EXPOSURE from line i. At the top table 93% of the steps are zero gaps, so the walk costs
    1.22% of the plain read; pouring every exposure whole into its word's pile costs 36.86%.
  - The collapsing read (base2_fringes.py, 21.13% of the plain read): words are set aside once they surely cannot reach
    the top 5, with an exact bound on what is not yet read, compared at the pile's own scale.
  - Here the two are joined. The walk is made once per line, and its exposures are held. Each exposure has 30 places
    (|x| < 2^15, |w| < 2^15), split into 8 bands fixed from that format before anything ran: places 29-26, 25-22,
    21-18, 17-14, 13-10, 9-6, 5-2, 1-0. Band 0 of every exposure is poured into every word's pile; then band 1 only
    into the words still in, and so on down. After each of bands 0..6 every word that can no longer reach the top 5 is
    set aside by the collapsing read's own rule at the pile's own scale, with ONE SHARED BOUND: what is still unpoured
    in any word's pile is at most (the lit lines) x (the largest value the unpoured places can hold), R = n (2^p - 1)
    after the places p and above are poured. After band 7 the words still in are read to the end exactly.

THE RULE (base2_fringes.TopRead's, at the pile's own scale; the bound here is shared, so its test is one compare a word):
  - each word's standing amount at pile scale beta_w = power(bias_w, -F), written once a position and poured into the
    word's pile; gamma, delta and mu = 2 gamma + delta exactly as base2_fringes.py makes them (gamma covers beta's
    rounding, delta makes one pile-scale step worth at least one score step).
  - V_w = beta_w + the bands poured so far. High V_w + R, low V_w - R. With v5 the 5th-best V among the words still in,
    the 5th-best low is v5 - R; a word is set aside when its high is at or below that less mu:  V_w <= v5 - 2R - mu.
  - the words still in after band 7: pile = V - beta (the standing amount taken off), score = power(pile, F) + bias,
    the machine's own last two steps. Top 5 (judge5.top5's two lines) and the race winner (base2.race) among them.

WORK, in photons (way 2's and base2_fringes.py's rules; the plain read's basis is base2_fringes.py's 904,017,313 a token
on the 648 refs positions: its meetings, every word's divide and every word's standing amount):
  - the walk: on each line, the light's lit places (signed digits) times each gap's lit places (signed digits), the
    first rank's whole lowest weight included (way 2's gap meetings); done once a position, whatever is set aside.
  - the light's writing: ones(x) a value (way 2's rule).
  - the pours: band b's part of an exposure added into a word's pile counts the photons it adds, ones of the band's
    places of |r|. Counted only into the words still in at band b. (All 8 bands into every word is way 2's pours.)
  - the standing amounts: written at pile scale (each word's bias photons x the gain's), poured (ones(beta)).
  - the bound, per test: R = (n << p) - n adds n's photons; 2R + mu adds mu's; the threshold v5 - (2R + mu) adds
    (2R + mu)'s. Compares (the 5th-best V found among the words still in, and one compare a word) are a COUNT, kept
    apart from the photons, as base2_fringes.py keeps them.
  - the words read to the end: the standing amount taken off, the divide (the pile's photons x the gain's), the bias.
  - held, not counted (as the walk's exposures are held in way 2 until poured): the exposures between bands.

THE CHECK PATH (never used by the machine, never counted): base2's own product of the same light gives every pile and
every score; every band's part summed over the bands must give the pile; the bound must hold for EVERY word at every
test; every word set aside must score below the true 5th-best; the words read to the end must score as the plain read;
the top 5 (set and order) and the race winner must equal the plain read's and the saved references (refs: base2.py's own
run, recording/C_ref/plain_refs.npz; windows: judge5_ref.npz). The simulation forms every band of every word (the C
kernel) so it can check; only the pours into words still in are counted.

CONTROLS
  light=shuffled  each position's light shuffled across its lines (every value kept; the same seed and draw as
                  fringes_machine/top_read.py's control, so the refs control meets the very same light).
  order=SEED      ORDER-FREE: the lines walked in a random order, every walk made DOWN from the top weight, and the
                  words relabelled in a random order (the line-up laid again on the relabelled table).

PREDICTIONS (written before anything in topband/ ran)
  The task's (the way-2 run's): 12-17% of the plain read; the whole machine about 45-46%.
  The session's, reasoned from the logs already on disk (way 2: 8.57 pour photons a pair at the top table; the
  collapsing read's survivors after the top 14, 16 and 18 diagonals, whose bounds are about 6e7, 1.6e7 and 1e6 at the
  pile's scale):
  - Exact: top 5 (set and order) and race winner 6,648/6,648 against the plain read and the saved references; judge5
    6,000/6,000 and 6,000/6,000; every pile rebuilt from its bands; the bound held for every word at every test; no
    word set aside at or above the true 5th-best.
  - Words still in (median): after bands 0, 1, 2 (places 18 and up poured; 2R about 4e8 after band 2) 50,257 at
    nearly every position; after band 3 (places 14 and up; 2R about 2.5e7) a few hundred to a few thousand (median
    200-2,000); after band 4 (2R about 1.6e6) 8-30; after band 5 (2R about 1e5) 5-6; after band 6, 5.
  - Where the exposures' photons sit: a typical exposure is about 2^16 (8.57 photons), so places 14 and up hold about
    18-25% of all pour photons (a few outlier lines of the light add a little); bands 0-3 go to every word.
  - Pours 6.5-9.5% of the plain read; the walk 1.22%; the standing amounts about 0.57%; the bound under 0.01%; the
    words read to the end under 0.01%. Total 8.5-11.5% of the plain read, most likely about 10%: at or below the low
    end of the task's 12-17%. Against the collapsing read's 21.13%: about half.
  - Whole machine (base2_fringes.py's 1,745,178,133 with this read in place of its 191,031,613): 1,630-1,660M a
    token, 44.5-45.3% of plain (the task's: about 45-46%).
  - Shuffled light: the photons sit in the same bands within a point; the shared bound does not see which words the
    light favours, so the total is within 1.5 points of the real light's.
  - Order-free: every count identical at every position; the top 5 identical as a set at 648/648 refs positions, and
    in order at 647 or 648 (the exact tie at refs position 366, words 655 and 1654, may be ordered by the relabelling).

    python3 topband/topband.py set=refs|windows [part=i/n] [light=real|shuffled] [order=SEED] [batch=32]
      -> topband/top_<set>[_<light>][_order<SEED>][_p<i>of<n>].npz / .log / .json
"""
import os, sys, json, time, ctypes, subprocess, resource, glob, warnings, numpy as np
from fractions import Fraction
warnings.filterwarnings("ignore", message=".*encountered in matmul")     # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/topband"
NB = 8; PB = [26, 22, 18, 14, 10, 6, 2, 0]; NT = NB - 1                   # bands; a test after each of bands 0..6
PLAIN_READ_REFS = 904017313; FRINGES_TOP_REFS = 191031613; FRINGES_TOTAL = 1745178133; PLAIN_TOTAL = 3666596259

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def pct2(a, b): return f"{(10000 * a + b // 2) // b / 100:.2f}%" if b else "-"
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def ones1(v): return bin(abs(int(v))).count("1")

# ---------- base2.py's own steps (no plate loaded as float by its loader) ----------
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
for _a, _b in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"),
               ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('M = np.load("base2_gpt2.npz")', f'M = np.load("{HERE}/base2_gpt2.npz")'),
               ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')]:
    assert _src.count(_a) == 1, _a; _src = _src.replace(_a, _b)
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
assert B["W"] == {}
M, ES, EXP2 = B["M"], B["ES"], B["EXP2"]
power, race, product = B["power"], B["race"], B["product"]
HEAD_E, HEAD_B = M["head.e"], M["head.b"]; assert (HEAD_E == HEAD_E[0]).all()

def top5(S):                                                 # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

# ---------- the walk (band_walk.c) ----------
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{OUT}/band_walk.c", f"{OUT}/band_walk.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); P_, I_, i_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.band_walk.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, i_, P_, P_, P_, P_]; _lib.band_walk.restype = None
    return _lib

class Table:
    """the top table lined up once (way 2's record, laid here): per line the word at each rank (NR), the gaps in rank
    order (G, G[:, 0] the lowest weight), the top weight (TOP), the gaps' lit places in signed digits (GN), and the plain
    read's photons per line (PPH). word_perm: the words relabelled (ORDER-FREE): the table's columns and the biases
    taken in that order, the line-up laid again. Read back on laying: the gaps summed in rank order give the table."""
    def __init__(self, word_perm=None, log=print):
        t0 = time.time()
        T = np.asarray(M["head.w"]); bias = HEAD_B.copy()
        if word_perm is not None: T = T[:, word_perm]; bias = bias[word_perm]
        L, N = T.shape; assert N < 65536 and np.abs(T.astype(np.int64)).max() < 1 << 15
        self.L, self.N, self.bias, self.wperm = L, N, bias, (np.arange(N) if word_perm is None else np.asarray(word_perm))
        self.NR = np.empty((L, N), np.uint16); self.G = np.empty((L, N), np.int32); self.TOP = np.empty(L, np.int64)
        self.GN = np.empty(L, np.int64); self.PPH = np.empty(L, np.int64)
        for i0 in range(0, L, 64):
            wb = T[i0:i0 + 64].astype(np.int64); Lb = len(wb)
            o = np.argsort(wb, axis=1, kind="stable"); ws = np.take_along_axis(wb, o, 1)
            g = np.diff(ws, axis=1, prepend=0)
            back = np.empty_like(wb); np.put_along_axis(back, o, np.cumsum(g, axis=1), 1)
            assert np.array_equal(back, wb), "the line-up does not give the table back"
            assert (g[:, 1:] >= 0).all()
            self.NR[i0:i0 + Lb] = o; self.G[i0:i0 + Lb] = g; self.TOP[i0:i0 + Lb] = ws[:, -1]
            self.GN[i0:i0 + Lb] = naf(g).sum(1); self.PPH[i0:i0 + Lb] = ones(wb).sum(1)
        self.zero_steps = int((self.G[:, 1:] == 0).sum())
        self.HW = T.astype(np.float64)                                          # CHECK ONLY: base2's own product
        self.BIAS_ONES = ones(bias); self.BMAX = int(np.abs(bias).max())
        log(f"  top table lined up: {L} lines x {N} words; read back exact; zero steps {self.zero_steps:,} of "
            f"{L * (N - 1):,} ({pct2(self.zero_steps, L * (N - 1))}) ({time.time() - t0:.0f}s, {rss()} MB)")

    def walk(self, X, line_order=None, down=False):
        X = np.ascontiguousarray(np.asarray(X, np.int64)); P, L = X.shape; assert L == self.L
        assert np.abs(X).max(initial=0) < 1 << 15
        LO = np.ascontiguousarray(np.arange(L, dtype=np.int64) if line_order is None else np.asarray(line_order, np.int64))
        assert np.array_equal(np.sort(LO), np.arange(L))
        DN = np.zeros((P, self.N, NB), np.int16); HN = np.zeros((P, self.N, NB), np.uint16)
        op = np.zeros(P, np.int64); big = np.zeros(P, np.int64)
        lib().band_walk(P, L, self.N, X.ctypes.data, self.G.ctypes.data, self.NR.ctypes.data, self.TOP.ctypes.data,
                        LO.ctypes.data, -1 if down else 1, DN.ctypes.data, HN.ctypes.data, op.ctypes.data, big.ctypes.data)
        assert not op.any(), "a walk down did not close at 0"
        assert not big.any(), "an exposure reached 2^30"
        return DN, HN

COUNT_KEYS = ["walk", "writing", "standing_written", "standing_poured", "end", "plain_meet", "plain_div", "plain_stand", "n"]
def read(tab, X, EX, LS, line_rng=None, down=False, batch=32, log=print):
    """the band-by-band top read at every position given (light X in the table's line order). Returns a dict of
    per-position arrays (counts, survivors, top 5, race winner, the plain read's top 5 and winner) and the check's
    tally."""
    X = np.asarray(X, np.int64); N_ = len(X); NW = tab.N; bad = {}
    def nb(k, v): bad[k] = bad.get(k, 0) + int(v)
    out = {k: np.zeros(N_, np.int64) for k in COUNT_KEYS}
    out.update(pours=np.zeros((N_, NB), np.int64), hall=np.zeros((N_, NB), np.int64), surv=np.zeros((N_, NT), np.int64),
               bound=np.zeros((N_, NT), np.int64), cmp=np.zeros((N_, NT), np.int64), mu=np.zeros(N_, np.int64),
               top5=np.zeros((N_, 5), np.int64), win=np.zeros(N_, np.int64), ptop5=np.zeros((N_, 5), np.int64),
               pwin=np.zeros(N_, np.int64))
    t0 = time.time()
    for c0 in range(0, N_, batch):
        Xb = X[c0:c0 + batch]; P = len(Xb)
        F = (HEAD_E[0] + EX[c0:c0 + P] + ES) * 65536 - LS[c0:c0 + P]
        lo = None if line_rng is None else line_rng.permutation(tab.L)
        DN, HN = tab.walk(Xb, line_order=lo, down=down)
        # ---- CHECK PATH: base2's own product and scores; every band's part summed gives the pile ----
        PILE = product(Xb, tab.HW); S = power(PILE, F[:, None]) + tab.bias[None, :]
        D = DN.astype(np.int64) << np.array(PB, np.int64)[None, None, :]
        CUM = np.cumsum(D, axis=2); nb("bands summed vs the plain read's pile", (CUM[:, :, -1] != PILE).sum())
        nlit = (Xb != 0).sum(1)
        for p in range(P):
            q = c0 + p; x = Xb[p]; n = int(nlit[p])
            Fp = int(F[p]); I = Fp >> 16; e = int(EXP2[Fp & 0xFFFF]); s = 16 - I
            F2 = -Fp; I2 = F2 >> 16; e2 = int(EXP2[F2 & 0xFFFF]); s2 = 16 - I2
            beta = power(tab.bias, F2)                                            # each word's standing amount at pile scale
            twos = Fraction(2) ** s; twot = Fraction(2) ** (s + s2)
            g = Fraction(1, 2) + tab.BMAX * twos * abs(e * e2 - twot) / (e * twot)
            gamma = int(-(-g.numerator // g.denominator)); delta = max(1, int(-(-(twos.numerator) // (e * twos.denominator))))
            mu = 2 * gamma + delta; ge = ones1(e); ge2 = ones1(e2)
            out["mu"][q] = mu; out["n"][q] = n
            out["walk"][q] = int(naf(x) @ tab.GN); out["writing"][q] = int(ones(x).sum())
            out["standing_written"][q] = int(tab.BIAS_ONES.sum()) * ge2; out["standing_poured"][q] = int(ones(beta).sum())
            out["hall"][q] = HN[p].sum(0, dtype=np.int64)                          # where the photons sit (all words)
            s5 = np.partition(S[p], NW - 5)[NW - 5]                               # CHECK: the true 5th-best score
            alive = np.arange(NW); V = beta.copy()
            for b in range(NB):
                out["pours"][q, b] = int(HN[p, alive, b].sum(dtype=np.int64))       # band b poured into the words still in
                V = V + D[p, alive, b]
                if b == NT: break
                R = n * ((1 << PB[b]) - 1)                                        # the shared bound: nothing unpoured exceeds it
                nb("bound broken (any word, check)", (np.abs(PILE[p] - CUM[p, :, b]) > R).sum())
                nb("counted pile carried = the counted pile", (V - beta[alive] != CUM[p, alive, b]).sum())
                na = len(alive); v5 = np.partition(V, na - 5)[na - 5]
                thr = 2 * R + mu; LAM = v5 - thr
                keep = V > LAM
                out["bound"][q, b] = ones1(n) + ones1(mu) + ones1(thr); out["cmp"][q, b] = 2 * na
                nb("a word set aside that is not below the 5th-best", (S[p, alive[~keep]] >= s5).sum())
                alive = alive[keep]; V = V[keep]; out["surv"][q, b] = len(alive)
            pile = V - beta[alive]                                                # the standing amount taken off
            sc = power(pile, Fp) + tab.bias[alive]
            nb("words read to the end vs the plain read", (sc != S[p, alive]).sum())
            out["end"][q] = int(ones(beta[alive]).sum() + ones(pile).sum() * ge + tab.BIAS_ONES[alive].sum())
            Sp = np.full(NW, -(1 << 62), np.int64); Sp[alive] = sc
            out["top5"][q] = tab.wperm[top5(Sp[None])[0]]; out["win"][q] = tab.wperm[race(Sp)]
            out["ptop5"][q] = tab.wperm[top5(S[p:p + 1])[0]]; out["pwin"][q] = tab.wperm[race(S[p])]
            out["plain_meet"][q] = int(ones(x) @ tab.PPH); out["plain_div"][q] = int(ones(PILE[p]).sum()) * ge
            out["plain_stand"][q] = int(tab.BIAS_ONES.sum())
        del DN, HN, D, CUM, PILE, S
        if (c0 // batch) % 8 == 0 or c0 + batch >= N_:
            log(f"    {min(N_, c0 + batch)}/{N_} positions ({time.time() - t0:.0f}s, {rss()} MB)")
    out["bad"] = bad
    return out

def total(o):                                                # the band read's photons a position
    return (o["walk"] + o["writing"] + o["standing_written"] + o["standing_poured"] + o["pours"].sum(1) + o["bound"].sum(1)
            + o["end"])
def plain(o): return o["plain_meet"] + o["plain_div"] + o["plain_stand"]

if __name__ == "__main__":
    ARG = dict(a.split("=") for a in sys.argv[1:])
    SET, LIGHT = ARG.get("set", "refs"), ARG.get("light", "real"); ORDER = int(ARG["order"]) if "order" in ARG else None
    PI, PN = (int(v) for v in ARG.get("part", "1/1").split("/")); BATCH = int(ARG.get("batch", 32))
    TAG = f"{SET}" + ("" if LIGHT == "real" else f"_{LIGHT}") + ("" if ORDER is None else f"_order{ORDER}") + \
          ("" if PN == 1 else f"_p{PI}of{PN}")
    LOG = open(f"{OUT}/top_{TAG}.log", "w"); t0 = time.time()
    def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    if SET == "refs":
        Z = np.load(f"{HERE}/fringes_machine/refs_top_light.npz"); X, EX, LS, ROWS = Z["x16"], Z["ex"], Z["LS"], Z["rows"]
        REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
        assert np.array_equal(X, REF["x16"][ROWS]) and np.array_equal(EX, REF["ex"][ROWS]) and np.array_equal(LS, REF["LS"][ROWS])
        RT5, RWIN = REF["top5"][ROWS], REF["win"][ROWS]; src = "fringes_machine/refs_top_light.npz = recording/C_ref (base2.py's own run)"
    else:
        Z = np.load(f"{HERE}/fringes_machine/windows_top_light.npz"); X, EX, LS = Z["x16"], Z["ex"], Z["LS"]
        W2 = [np.load(f) for f in sorted(glob.glob(f"{HERE}/lineup2/windows_*_*.npz"))]
        assert np.array_equal(np.concatenate([w["windows"] for w in W2]), np.arange(300))
        same = all(np.array_equal(a, np.concatenate([w[k] for w in W2])) for a, k in ((X, "x16"), (EX, "ex"), (LS, "LS")))
        assert same, "the two saved lights at the top differ"
        RT5 = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5); RWIN = None
        src = "fringes_machine/windows_top_light.npz = lineup2/windows_*.npz (both checked against base2.py's own run)"
    N_ALL = len(X)
    if LIGHT == "shuffled":                                  # fringes_machine/top_read.py's draw, over the whole set
        rng = np.random.default_rng(2026); X = np.stack([x[rng.permutation(768)] for x in X])
    cut = np.linspace(0, N_ALL, PN + 1).astype(np.int64); SEL = np.arange(cut[PI - 1], cut[PI])
    X, EX, LS = X[SEL].astype(np.int64), EX[SEL], LS[SEL]
    log(f"top read band by band: set={SET} light={LIGHT} order={ORDER} part={PI}/{PN}: {len(SEL)} positions; light: {src}")
    wperm = None; line_rng = None; down = False
    if ORDER is not None:
        rng = np.random.default_rng(ORDER); wperm = rng.permutation(50257); line_rng = np.random.default_rng(ORDER + 1); down = True
    tab = Table(word_perm=wperm, log=log)
    R = read(tab, X, EX, LS, line_rng=line_rng, down=down, batch=BATCH, log=log)
    n = len(SEL)
    same5 = int((R["top5"] == R["ptop5"]).all(1).sum()); samew = int((R["win"] == R["pwin"]).sum())
    sets5 = int((np.sort(R["top5"], 1) == np.sort(R["ptop5"], 1)).all(1).sum())
    log(f"EXACT: top 5 (set and order) = base2's plain read of every word {same5}/{n} (as a set {sets5}/{n}); race winner {samew}/{n}")
    if LIGHT == "real":
        f5 = int((R["top5"] == RT5[SEL]).all(1).sum()); p5 = int((R["ptop5"] == RT5[SEL]).all(1).sum())
        log(f"  against the saved reference top 5 ({'base2.py run, plain_refs' if SET == 'refs' else 'judge5_ref'}): "
            f"band read {f5}/{n}, the plain read here {p5}/{n}")
        if RWIN is not None: log(f"  race winner = base2.py's saved winner: {int((R['win'] == RWIN[SEL]).sum())}/{n}")
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in sorted(R["bad"].items())))
    tot, pl = total(R), plain(R)
    log(f"\nWORK A POSITION (mean over {n}; photons). The plain read: meetings + every word's divide + every word's standing amount.")
    log(f"  plain read {pl.sum() / n:,.0f}  (meetings {R['plain_meet'].sum() / n:,.0f})")
    log(f"  band read  {tot.sum() / n:,.0f} = {pct2(int(tot.sum()), int(pl.sum()))} of the plain read")
    for k in ("walk", "writing", "standing_written", "standing_poured", "end"):
        log(f"    {k:18s} {R[k].sum() / n:14,.0f}  {pct2(int(R[k].sum()), int(pl.sum())):>7s}")
    log(f"    {'pours (all bands)':18s} {R['pours'].sum() / n:14,.0f}  {pct2(int(R['pours'].sum()), int(pl.sum())):>7s}"
        f"   (every band into every word, way 2's pours: {R['hall'].sum() / n:,.0f}, {pct2(int(R['hall'].sum()), int(pl.sum()))})")
    log(f"    {'bound':18s} {R['bound'].sum() / n:14,.0f}  {pct2(int(R['bound'].sum()), int(pl.sum())):>7s}   compares (a count) {R['cmp'].sum() / n:,.0f}")
    log(f"\nBY BAND (places; words still in after that band's test: median / mean / least / most; pours a position; share of the plain read; "
        f"that band's photons over every word)")
    for b in range(NB):
        hi = 29 if b == 0 else PB[b - 1] - 1
        sv = R["surv"][:, b] if b < NT else None
        svs = (f"{int(np.median(sv)):6d} {sv.mean():9.1f} {sv.min():6d} {sv.max():6d}" if sv is not None else f"{'(read to the end)':>31s}")
        log(f"  band {b} (places {hi:2d}-{PB[b]:2d}): {svs}   pours {R['pours'][:, b].sum() / n:13,.0f} "
            f"{pct2(int(R['pours'][:, b].sum()), int(pl.sum())):>7s}   all words {R['hall'][:, b].sum() / n:13,.0f} "
            f"({pct2(int(R['hall'][:, b].sum()), int(R['hall'].sum()))} of the exposures' photons)"
            + (f"   bound {R['bound'][:, b].sum() / n:5.1f}" if b < NT else ""))
    np.savez(f"{OUT}/top_{TAG}.npz", sel=SEL, **{k: v for k, v in R.items() if k != "bad"},
             bad_keys=np.array(sorted(R["bad"])), bad=np.array([R["bad"][k] for k in sorted(R["bad"])]))
    json.dump(dict(set=SET, light=LIGHT, order=ORDER, part=[PI, PN], n=int(n), bad=R["bad"], total=int(tot.sum()), plain=int(pl.sum()),
                   zero_steps=tab.zero_steps), open(f"{OUT}/top_{TAG}.json", "w"), indent=0)
    log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")
