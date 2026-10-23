"""WAY 2: THE LINE-UP ACROSS THE NEURONS ON EACH LINE, AT EVERY BANK AND AT THE TOP TABLE. Exact, whole numbers only.

THE PICTURE (the line-up, restated; the session's words, not his). Way 1 (combined/, rank_gaps.py) lined each neuron's
768 weights up DOWN the neuron and poured the light into a running sum per neuron. Way 2 lines up the other way:
  - for each line i of a plate, that line's weights over ALL its neurons are put in order of size once, and the order
    is stored (which neuron stands at each rank);
  - per token the light on line i is one whole number x_i. Walk the ranks:
        r_0 = x_i * w_(0),   r_k = r_(k-1) + x_i * gap_k   (gap_k = w_(k) - w_(k-1), tiny between neighbours in rank)
    and pour r_k into the pile of the neuron standing at rank k.
  So the running product r_k is exactly x_i times that neuron's weight, built from the tiny gaps instead of from the
  whole weight; every pile is exactly the plain pile (the sum over lines is unchanged, only how each product is made).

WHAT IS COUNTED (photons, the unit of all earlier counts; combined/machine.py's and base2_fringes.py's rules):
  - gap meetings: on each line, the light's lit places (signed digits, no two lit side by side) times each gap's lit
    places (signed digits), over every rank, the first rank's whole lowest weight included. The meeting's photons land
    in the running product r, as a plain meeting's photons land in the pile.
  - pours: each running product r_k added into its neuron's pile: an add counts the photons it adds, ones(|r_k|).
    (Side count only: the lit places of r_k in single waves, as if r_k were already written that way, its writing not
    counted.) THE POURS ARE SET BY THE LIGHT AND THE PLATE ALONE: r_k = x_i w_ij whatever the order, so the order, the
    shuffle and the walk's direction cannot change them. The run measures their size; nothing about the order.
  - the light written in signed digits: ones(x) a value, once a bank (3x = x + 2x), as base2_fringes.py counts it.
  - a dark line (x_i = 0): nothing met, nothing poured (the plain count is 0 there too).
  Against, on the same light: plain meetings ones(x) . ones(w) per line (must rebuild recording/C's 2,737,920,452 a
  token at the 48 banks and 896,556,125 at the top table); the signed-digit meetings naf(x) . naf(w) (recording/C's
  1,605,763,357, 58.65%); and base2_fringes.py's bank work from its own report (fringes_machine/report.log).
  Everything else in the machine (arrival, bends, water levels, divides, biases, stream adds) is base2.py's own and
  unchanged: 3,666,596,259 - 2,737,920,452 - 896,556,125 = 32,119,682 a token.

THE ABEL FORM ACROSS THE NEURONS (asked for: x_i gap_k added into a running sum poured once per line in a suffix form).
Stated before running. Neuron j's share from line i is r at j's rank = sum over m <= rank_i(j) of x_i gap_(i,m). In the
suffix form each x_i gap_(i,m) has to reach every neuron ranked at or above m ON LINE i. A single running sum can be
poured once only if the same suffix sets serve every line, i.e. one neuron order shared by all lines of the plate;
then D_m = sum over lines of x_i gap_(i,m) and the pile at rank k is D_0 + ... + D_k (N adds a position, the pours
nearly vanish). With each line's own order the suffix sets differ line to line, and the suffix form pours each
x_i gap_(i,m) into N - m piles: about N/2 pours a gap, hundreds of times the walk. So: exact, but NOT cheaper; it is
not run. The shared-order form is exact and pours almost nothing, but its gaps are each line's differences in an order
that is not that line's own line-up. The SHUFFLE CONTROL (each line's gaps taken in a random neuron order) measures
exactly that cost: a random order shared by all lines gives each line the same law of gaps as a random order per line.

PREDICTIONS (written before anything in lineup2/ ran)
  The task's: the gap meetings drop far below the plain meetings, but each neuron's running product still has to be
  added into its pile, and those adds (one full-size number per line per neuron) bring the bank work back near the
  signed-digit machine's 58%; storage stays at or above the 153 MB floor because the orders cost what the gaps save.
  The session's:
  - Exact: every pile (qkv, proj, fc, mproj, all 12 layers) and every half-layer stream identical to base2.py's own
    run at all 648 refs positions and every array of all 300 windows; the top table's piles identical to base2's
    product; score rows (sha256), top 5 and race winner 648/648; judge5 6,000/6,000 and 6,000/6,000. The walk down
    from the top weight with the lines in a random order: identical. The shuffled-order walk: identical piles too
    (the sum holds in any order), with far more gap meetings.
  - Gap meetings, share of the plain meetings of the same light: fc (3,072 a line) 14-19%; qkv (2,304) 16-21%; proj and
    mproj (768 a line) 25-31%; the top table (50,257 a line) 2-6%.
  - Pours: 33-40% of plain at every bank (a product of a ~10-bit light value and a ~12-bit weight has about 12 ones,
    against about 32 plain meetings a pair); 38-45% at the top table.
  - Bank work (gap meetings + pours + the light's writing): fc 50-57%, qkv 52-59%, proj and mproj 60-69%; all 48 banks
    55-62% of plain, within about 4 points of signed digits' 58.65% on the same light: no real gain at the banks.
  - The top table: 40-50% of the plain read, worse than the collapsing read's 21.13%.
  - The whole machine with way 2 at all 48 banks and the top table, everything else unchanged: 54-60% of plain, above
    base2_fringes.py's 47.60%.
  - Shuffle control (gaps from a random neuron order per line): gap meetings 55-70% of plain at every bank (a gap
    between two weights drawn at random is about as wide as a weight); pours unchanged; bank work 90-105%.
  - Storage (lineup2/storage.py): orders at their floor about 8.1 bits a weight at 768 neurons a line and 9.9-10.1 at
    2,304 and 3,072, less the ties; packed gaps 3.5-5 bits a weight; the 48 plates 150-175 MB, most likely at or above
    the 153.1 MB floor; the top table at or above T's own 54.8 MB.

    from way2 import Record, Machine     (lineup2/run_refs.py, run_windows.py, judge.py, storage.py use it)
"""
import os, sys, json, time, ctypes, subprocess, resource, hashlib, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/lineup2"
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
ALL = BANKS + ["head"]
PLAIN_TOTAL = 3666596259; PLAIN_BANKS_C = 2737920452; PLAIN_HEAD_C = 896556125; SIGNED_BANKS_C = 1605763357
REST_UNCHANGED = PLAIN_TOTAL - PLAIN_BANKS_C - PLAIN_HEAD_C              # 32,119,682 a token
FRINGES = dict(total=1745178133, qkv=362167885, proj=142256231, fc=488518533, mproj=543593368, top=191031613)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def per_token(total, n): return (total + n // 2) // n
def pct2(a, b): return f"{(10000 * a + b // 2) // b / 100:.2f}%" if b else "-"
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

# ---------- base2.py's own steps; the bank's product is the only thing replaced (BANK, the walk) ----------
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
for _a, _b in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"),
               ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('M = np.load("base2_gpt2.npz")', f'M = np.load("{HERE}/base2_gpt2.npz")'),
               ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]'),
               ("    x16, ex, LS = gauge_log(R)\n    pile = product(x16, w)\n", "    x16, ex, LS = gauge_log(R)\n    pile = BANK(x16, name)\n"),
               ("    return rshift(product(x16, w), -(we[None, :]", "    return rshift(BANK(x16, name), -(we[None, :]")]:
    assert _src.count(_a) == 1, _a; _src = _src.replace(_a, _b)
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
assert B["W"] == {}
M = B["M"]; ES = B["ES"]
rshift, to16, product, race, power, gauge_log, edge, attention, gauged_bank, plain_bank = (B[k] for k in
    ("rshift", "to16", "product", "race", "power", "gauge_log", "edge", "attention", "gauged_bank", "plain_bank"))
for _n in ALL: B["W"][_n] = (None, M[_n + ".e"], M[_n + ".b"])           # powers and biases; no plate held here
HEAD_E, HEAD_B = M["head.e"], M["head.b"]
WTE, WPE = M["wte"], M["wpe"]

# ---------- the walk (walk_rc.c) ----------
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{OUT}/walk_rc.c", f"{OUT}/walk_rc.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O2", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); P_, I_, i_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.walk.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, i_, i_, I_, P_, P_, P_, P_]; _lib.walk.restype = None
    return _lib

class Record:
    """the way-2 record of the 48 plates and the top table, laid once (never per token): for each line, the neuron at
    each rank (NR, uint16) and the gaps in rank order (G, int32; G[:, 0] the lowest weight), and per line the top
    weight (the walk down's start), the gaps' lit places (GN), and for the counts the plate's plain 1s (WROW) and lit
    places (NROW) per line. order='sorted': each line's own line-up (ties in neuron order). order='random': each line's
    neurons in a random order (THE SHUFFLE CONTROL: the same weights, the same products, the line-up broken).
    Read back on laying: the gaps summed in rank order and put back on their neurons give the plate exactly."""
    def __init__(self, names=ALL, order="sorted", seed=None, log=print, keep_shuffled_counts=True):
        t0 = time.time(); self.order = order; self.NR, self.G, self.TOP, self.GN, self.GNS, self.WROW, self.NROW = {}, {}, {}, {}, {}, {}, {}
        self.shape = {}
        for name in names:
            w = np.asarray(M[name + ".w"]); L, N = w.shape; self.shape[name] = (L, N)
            assert N < 65536 and np.abs(w.astype(np.int64)).max() < 1 << 15
            NR = np.empty((L, N), np.uint16); G = np.empty((L, N), np.int32); TOP = np.empty(L, np.int64)
            GN = np.empty(L, np.int64); GNS = np.empty(L, np.int64); WROW = np.empty(L, np.int64); NROW = np.empty(L, np.int64)
            rng = np.random.default_rng((seed or 0) + ALL.index(name)); rngs = np.random.default_rng(1000 + ALL.index(name))
            for i0 in range(0, L, 64):                               # a block of lines at a time
                wb = w[i0:i0 + 64]; Lb = len(wb)
                if order == "sorted": o = np.argsort(wb, axis=1, kind="stable")
                else: o = rng.permuted(np.tile(np.arange(N, dtype=np.int64), (Lb, 1)), axis=1)
                ws = np.take_along_axis(wb, o, 1).astype(np.int32)
                g = np.diff(ws, axis=1, prepend=np.int32(0)).astype(np.int32)
                back = np.empty((Lb, N), np.int64); np.put_along_axis(back, o, np.cumsum(g, axis=1, dtype=np.int64), 1)
                assert np.array_equal(back, wb), f"{name}: the record does not give the plate back"
                if order == "sorted": assert (g[:, 1:] >= 0).all()
                NR[i0:i0 + Lb] = o; G[i0:i0 + Lb] = g; TOP[i0:i0 + Lb] = ws[:, -1]; GN[i0:i0 + Lb] = naf(g).sum(1)
                WROW[i0:i0 + Lb] = ones(wb).sum(1); NROW[i0:i0 + Lb] = naf(wb).sum(1)
                if keep_shuffled_counts and order == "sorted":     # the shuffle control's gap lit places (counts only)
                    os_ = rngs.permuted(np.tile(np.arange(N, dtype=np.int64), (Lb, 1)), axis=1)
                    gs = np.diff(np.take_along_axis(wb.astype(np.int64), os_, 1), axis=1, prepend=0)
                    GNS[i0:i0 + Lb] = naf(gs).sum(1); del os_, gs
                del o, ws, g, back
            self.NR[name], self.G[name], self.TOP[name], self.GN[name] = NR, G, TOP, GN
            if keep_shuffled_counts and order == "sorted": self.GNS[name] = GNS
            self.WROW[name] = WROW; self.NROW[name] = NROW
            del w
        log(f"  way-2 record laid ({order}): {len(names)} tables, read back exact ({time.time() - t0:.0f}s, {rss()} MB)")

    def walk(self, name, x, line_order=None, down=False, count=True, blk=4):
        """the pile of every position at this plate, by the walk. x: (P, L) whole numbers. Returns pile, pours (P,),
        pours in single waves (P,)."""
        x = np.ascontiguousarray(np.asarray(x, np.int64)); P, L = x.shape; Ln, N = self.shape[name]; assert L == Ln
        assert np.abs(x).max(initial=0) < 1 << 16
        LO = np.ascontiguousarray(np.arange(L, dtype=np.int64) if line_order is None else np.asarray(line_order, np.int64))
        assert np.array_equal(np.sort(LO), np.arange(L))
        pile = np.zeros((P, N), np.int64); p1 = np.zeros(P, np.int64); pn = np.zeros(P, np.int64); op = np.zeros(P, np.int64)
        lib().walk(P, L, N, x.ctypes.data, self.G[name].ctypes.data, self.NR[name].ctypes.data, self.TOP[name].ctypes.data,
                   LO.ctypes.data, -1 if down else 1, 1 if count else 0, blk, pile.ctypes.data, p1.ctypes.data, pn.ctypes.data,
                   op.ctypes.data)
        assert not op.any(), f"{name}: a walk down did not close at 0"
        return pile, p1, pn

ITEMS = ("pairs", "plain", "signed", "gaps", "gaps_shuffled", "pours", "pours_single_wave", "writing")
class Machine:
    """base2.py's machine with every bank's product made by the walk. count: per-table per-position counts are kept
    (self.cnt[name][item] is a (positions,) array). line_rng: ORDER-FREE check (the lines walked in a random order at
    every bank, and the walk made DOWN from the top weight)."""
    def __init__(self, rec, count=True, line_rng=None, down=False):
        self.rec, self.count, self.line_rng, self.down = rec, count, line_rng, down
        self.cnt = {}; self.last = None
    def bank(self, x, name):
        L = x.shape[1]; lo = None if self.line_rng is None else self.line_rng.permutation(L)
        pile, p1, pn = self.rec.walk(name, x, line_order=lo, down=self.down, count=self.count)
        if self.count:
            r = self.rec; nx = naf(x); c = self.cnt.setdefault(name, {k: [] for k in ITEMS})
            N = r.shape[name][1]
            c["pairs"].append((x != 0).sum(1) * N); c["plain"].append(ones(x) @ r.WROW[name]); c["signed"].append(nx @ r.NROW[name])
            c["gaps"].append(nx @ r.GN[name]); c["gaps_shuffled"].append(nx @ r.GNS[name] if name in r.GNS else np.zeros(len(x), np.int64))
            c["pours"].append(p1); c["pours_single_wave"].append(pn); c["writing"].append(ones(x).sum(1))
        self.last = pile
        return pile
    def counts(self):
        return {n: {k: np.concatenate(v) for k, v in c.items()} for n, c in self.cnt.items()}

    def forward(self, texts, check=None, log=print):
        """every position of every text, layer by layer (each text's arrivals read only its own positions).
        check(key, value): every pile (qkv, proj, fc, mproj) and the stream after every half layer, all positions.
        Returns the light at the top (x16, ex, LS) for every position."""
        B["BANK"] = self.bank; t0 = time.time()
        lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)])
        ids = np.concatenate([np.asarray(t, np.int64) for t in texts]); pos = np.concatenate([np.arange(n) for n in lens])
        R = WTE[ids] + WPE[pos]
        def keep(k, v):
            if check is not None: check(k, v)
        for l in range(12):
            QKV = gauged_bank(R, f"qkv.{l}"); keep(f"qkv pile {l}", self.last)
            o = np.concatenate([attention(QKV[off[t]:off[t + 1]]) for t in range(len(texts))]); del QKV
            att = plain_bank(o, f"proj.{l}"); keep(f"proj pile {l}", self.last); del o
            R = R + att; keep(f"stream a {l}", R); del att
            H = gauged_bank(R, f"fc.{l}"); keep(f"fc pile {l}", self.last)
            mo = plain_bank(edge(H), f"mproj.{l}"); keep(f"mproj pile {l}", self.last); del H
            R = R + mo; keep(f"stream {l}", R); del mo
            self.last = None
            log(f"    layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
        return gauge_log(R)

    def top(self, x16, ex, LS, chunk=128):
        """the top table by the walk at the given positions: the whole-number scores (base2.py's own last two steps)
        and the piles' digests. Returns scores' sha256 rows (as bytes), top 5, race winner, and piles (as a generator
        hook for the check: self.top_check(pile, x16) if set)."""
        n = len(x16); T5 = np.zeros((n, 5), np.int64); WIN = np.zeros(n, np.int64); SHA = []
        for c0 in range(0, n, chunk):
            r = slice(c0, min(n, c0 + chunk))
            pile = self.bank(x16[r], "head")
            if getattr(self, "top_check", None) is not None: self.top_check(pile, x16[r])
            S = power(pile, (HEAD_E[None, :] + ex[r][:, None] + ES) * 65536 - LS[r][:, None]) + HEAD_B
            SHA += [hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).digest() for s in S]
            T5[r] = top5(S); WIN[r] = [race(s) for s in S]; del pile, S
        return SHA, T5, WIN

def top5(S):                                                 # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)
