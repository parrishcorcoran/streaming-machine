"""QWEN_FRINGES: THE ALL-BASE-2 QWEN3-0.6B (qwen/qwen_base2.py's own machine) WITH FOUR EXACT ADDITIONS, BUILT AS ONE.

His goal: "keep all the resolution, decrease the compute". Every water level, turn, arrival table, bend and stream add is
qwen_base2.py's own (its source is executed here, unchanged); only how the products are formed is changed, and every
pile, stream and read is checked identical to qwen_base2.py's own run. Whole numbers only: no float in the machine path
(float64 appears only in the CHECK PATH, as an exact whole-number product below 2^52, never used by the machine).

THE PICTURE (the session's words, restating Phase VI-a's; not his).
  - A plate: lines in, neurons out; on every line the light is one whole number x and the plate holds one weight for each
    neuron. Plain, every neuron's pile gets x w from every line: the 1s of x meet the 1s of w.
  - Way 2 (a): each line's weights are lined up across the plate's neurons ONCE (stored: the order and the gaps between
    neighbours in rank). Per token the light meets only the gaps: walking up the ranks, the running product
    r_k = r_(k-1) + x gap_k is exactly x times the weight of the neuron at rank k, and it is poured into that neuron's
    pile. The walk is thin where the gaps are small or zero (ties); the pours are each neuron's whole exposure.
  - The pours in signed digits (b): a running product poured as ups and holes, no two lit side by side, lands fewer
    photons in the pile (about 0.72 of its 1s). But it must be WRITTEN that way first, and the writing is counted.
  - The top read (c): the word table is walked once (way 2), and each word's exposure is poured band by band from its top
    places down, only into the words that can still reach the top 5 (topband/).
  - The arrival (d): q.k and shares x values formed from signed-digit planes (base2_fringes2 level 3).

THE FOUR ADDITIONS (each exact; the machine runs all four together; each one's work is counted apart)
 (a) WAY 2 AT EVERY PLATE (qwen06b/banks.py's walk): q, k, v, o, gate, up, down at all 28 layers (196 plates), each line's
     weights lined up across that plate's own neurons once (the line-up is laid once, never per token). Counted (the unit
     of every earlier count, photons): gap meetings = lit digits of the light (signed digits) x lit digits of each gap,
     the first rank's whole lowest weight included; pours = ones(|r_k|) for every running product added into its pile;
     the light written in signed digits = ones(x) a value, ONCE A SLAB (q, k and v meet the same light; gate and up the
     same light; Phase VI-a counted it once a kind, 0.01% of the plates).
 (b) THE POURS IN SIGNED DIGITS, THE WRITING COUNTED. The way each running product is written in signed digits is FIXED
     HERE, BEFORE ANY RUN, by a relation, not by a run:
       - a number v is written in signed digits from v and 3v (up = (3v & ~v) >> 1, down = (v & ~3v) >> 1, masks only);
         every earlier count paid 3v = v + 2v, ones(v) photons (the light's writing, the running sums of combined/).
       - way W2 (USED): the light's writing already makes 3x (3x = x + 2x). The machine walks a SECOND running product
         t_k = t_(k-1) + 3x gap_k along the same gaps, so t_k = 3 r_k at every rank (checked at every rank, never
         assumed) and each r_k's signed digits are read off r_k and t_k. Its cost: lit digits of 3x x lit digits of each
         gap (a second walk), plus 3x written in signed digits, ones(3x) a value once a slab.
       - way W1 (SIDE COUNT, not used): each new value of r on a line written once, 3r = r + 2r, ones(r) (a zero gap
         leaves r as it was, so its writing stands).
       - the relation that fixed W2: W2 per pair ~ (lit digits of 3x / lit digits of x) x the walk ~ 1.15 x the walk;
         W1 per pair ~ (1 - the zero-gap share) x the pours. With Phase VI-a's measured walk, pours and zero gaps per
         kind, summed over the 196 plates: W2 ~ 1.15 x 1,727M = 1,986M a token, W1 ~ 2,257M a token. So W2.
     Counted: the pours in signed digits = ones(|r_k| xor |t_k|) for every pour; the 3x walk; 3x's writing.
 (c) THE TOP READ, decided by the rule "whichever measured cheaper on GPT-2 with the same exactness", fixed before running:
       - GPT-2, fine diagonals with the pile-scale bound: 21.13% of the plain read (base2_fringes.py, plain photons),
         16.05% (base2_fringes2.py, signed digits and the start rule); exact top 5 and race winner at 6,648 positions,
         judge5 6000/6000 and 6000/6000.
       - GPT-2, the band pour (topband/): 10.88% (refs) and 10.99% (judged) of the plain read; exact top 5 and race
         winner at 6,648 positions, judge5 6000/6000 and 6000/6000. The same exactness, and cheaper: THE BAND POUR.
     topband's read carried to Qwen's table (1,024 lines x 151,936 words; no standing amounts: Qwen's read has no bias):
       - each line walked once (way 2, the line-up laid on the table once); every exposure (|r| < 2^30) split into
         topband's 8 bands, fixed by that format: places 29-26, 25-22, 21-18, 17-14, 13-10, 9-6, 5-2, 1-0.
       - band b is poured only into the words still in; after each of bands 0..6 the collapsing read's test at the pile's
         own scale with ONE SHARED BOUND: what is still unpoured on the n lit lines is at most R = n (2^p - 1) at the
         word's own scale (p the lowest place poured).
       - FOUR POWERS OF TWO among the words (head.e = -16..-13; GPT-2's table had one): every word is carried at one
         common scale, A_j = pile_j x 2^k_j (k_j = e_j - e_min, a shift, no photons), so score_j = power(A_j, F0)
         exactly, as qwen06b/top_read.py carried its read (checked there: every score row's sha256 equals the
         machine's). Word j's bound at that scale is R << k_j. The 5th-best low L5 is found from the five highest
         counted piles of each class less that class's bound (at most 20 lows); a word of class k is set aside when
         V_j <= L5 - mu - (R << k). That is topband's rule (high at or below the 5th-best low less mu), with the bound at
         each word's own shift. mu = 2 gamma + delta, gamma = 1, delta = max(1, ceil(2^s / e)): one pile-scale step
         worth at least one score step (top_read.py's, with no standing amount).
       - the words still in after band 7: score = power(A, F0), the machine's own last step; top 5 (by score, ties by
         word number: capture.py's rule) and the race winner (qwen_base2.race) among them.
     Counted (topband's rules): the walk (lit digits of the light x lit digits of each gap); the light's writing; band b's
     photons into the words still in; the bound (R's add, and per class: mu's add, the threshold's add, the lows'
     subtracts); the words read to the end (the divide: the pile's photons x the gain's). Compares are a count, apart.
     The plain read (qwen06b's basis): plain meetings of every word + every word's divide.
 (d) THE ARRIVAL'S PRODUCTS IN SIGNED DIGITS (base2_fringes2 level 3): q16.k16 and shares x values formed place pair by
     place pair from signed-digit planes, checked equal to qwen_base2's own products. q16 (each head), k16 and the values
     (each key/value head, once) written in signed digits once a position (ones, counted); the arrival table is stored
     in signed digits (its shares need no writing). Meetings: lit digit x lit digit.

THE PLAIN BASIS (Phase VI-a's, qwen06b/report.log, 16,397,464,589 a token on the 613 positions): the plates' plain
meetings (11,845,741,380), the plain read (4,490,888,460: meetings + every word's divide), the arrival's plain meetings
(60,834,749: q16.k16 and shares x values). Not counted on either side (unchanged parts): water levels, turns, bends, the
plates' and the arrival's divides, the shares' totals, the stream adds.

CHECKS (the check path is never used by the machine and never counted)
  - every pile of every plate at every position = the exact whole-number product; every pile digest, attention output,
    stream (both halves of every layer) and the top light = the reference run of qwen_base2.py's own steps (phase ref) AND
    capture.py's record of qwen_base2.py's own run (qwen06b/light: every plate's light, every slab's pile digests, the read
    at the top: score rows' sha256, top 5, race winner); the arrival's products = qwen_base2's int products.
  - the band read: bands summed = the pile at every word; the shared bound holds at EVERY word at every test; no word set
    aside scores at or above the true 5th best; the words read to the end score as the plain read; top 5 (set and order)
    and race winner = the plain read's and capture's, 613/613.
  - cross-checks with Phase VI-a (same light, same plates): every plate's plain, signed, gap, shuffled-gap and pour counts
    and the table's gap and all-band pour counts equal banks_*.json and table_walk.json.
  - THE 16 GREEDY WORDS: qwen_base2's own steps (the reference path) generate 16 words greedily on qwen/refs_qwen.json's
    30 prompts; they must decode to check30.log's machine words (28 = fp32's, 2 its own). The machine is run on those
    texts: every pile and stream identical to the reference, and its band read's race winner at each of the 16 places is
    the next greedy word, so its own greedy words are the same 16.
  - ORDER-FREE (3 texts, the first three, fixed): every plate walked DOWN from the top weight with its lines in a random
    order (both running products must close at 0), the arrival's lines in random orders, the word table walked down with
    its lines in a random order and the words relabelled (the line-up laid again): every pile, stream, count, top 5 and
    race winner identical to the main run.
  - CONTROLS: (i) count only: each plate line's gaps taken in a random neuron order (banks.py's draw), for the walk and
    the 3x walk: the saving is the line-up's; (ii) the band read on the light shuffled across its lines (every 4th
    position, qwen06b/top_read.py's draw, so it meets the very light of Phase VI-a's control: fine read 15.59%).

RULE 13, CHECKED BEFORE THE RUN
  13.1 nothing counts the light's 1s as the measure: the work is photons met and poured, the exactness is on the piles,
       streams and the top 5. 13.2 the base is qwen_base2.py's own machine and light, every part as it is, no float
       machine. 13.3 no agreement statistic is used. 13.4 THE POURS ARE FIXED BY THE LIGHT AND THE PLATE ALONE (r = x w in
       any order), and so are their signed digits: the run measures their size, nothing about the line-up; only the walk
       and the 3x walk depend on the line-up, and control (i) tests that. Band b's photons over every word are fixed the
       same way; which words they are poured into is what the read decides. 13.5 the task's prediction is the task's;
       the session's is labelled; nothing here is his sentence. 13.6 the band read is carried to band 7 at every position.

PREDICTIONS (written before any run of this file)
  The task's: about 37-40% of plain with (a) + (c) + (d); (b) lowers the pours by about a quarter, taking the whole to
  about 31-34%.
  The session's (from Phase VI-a's logs; two numbers were SEEN BEFORE THIS WAS WRITTEN, in a timing test of the two
  kernels: at gate.0 the pours in signed digits were 0.719 of the pours and W1's writing 0.414 of them; on 16 positions
  the bands 0-3 (places 14 and up) held about 34% of the table's exposure photons):
  - EXACT: every pile (196 plates x 613 positions), stream, attention output and the top light identical to the reference
    run and to capture's record; the band read's top 5 and race winner 613/613, bound broken 0, none set aside at or
    above the 5th best; greedy words 30/30 (28 = fp32's; 'My favorite food is' and 'Breaking news: a major earthquake'
    = check30.log's machine words); order-free identical everywhere. Every Phase VI-a cross-check equal.
  - (a): the plates at 49.13% of their plain (Phase VI-a's 49.14%, less the light's writing counted once a slab).
  - (c): words still in: all 151,936 after bands 0-2 at nearly every position (2R about 5e8 at word scale); after band
    3 a median of 30-500; after band 4, 5-10; 5 after that. The walk 0.31% of the plain read; the pours about 34% of
    the all-band pours (33.5% of the plain read) plus band 4 into a few hundred words: the band read 11.0-13.5% of the
    plain read, most likely about 12% (the fine diagonals measured 15.84% on the same light).
  - (d): the arrival 60.8M -> 34-40M a token (lit digits about 0.77 x 0.77 of the 1s on each side, plus the writing).
  - (a) + (c) + (d): plates about 5,820M + top read about 540M + arrival about 37M = about 6,400M a token, 38.3-39.8%
    of plain: inside the task's 37-40%.
  - (b): the pours from 34.55% of the plates' plain to about 0.72 of that (-1,150M a token), but the 3x walk costs
    1.10-1.20 x the gap meetings (14.58%, 1,727M): +1,900 to 2,070M. NET: (b) ADDS 750-930M a token at Qwen3-0.6B's
    widths: the plates at 55.5-57.0% of their plain, the whole machine 43.5-44.6% of plain, ABOVE the task's 31-34%.
    Per kind it loses everywhere (walk per pair 3.1-6.4 against a saving of 2.2-2.8 a pair). W1's side count about
    2,250M (worse than W2 overall; about equal at q and gate, a little better at k and v).
  - Control (i): the gaps in a random neuron order: banks' 95-111% of plain at every kind, the 3x walk likewise.
    Control (ii): the band read on the shuffled light within a point of the real light's at the same positions (the
    shared bound does not see which words the light favours; topband: 10.81% against 10.88%).

RESULTS (2026-10-05; qwen_fringes/report.log; a token on the 613 positions; plain 16,397,464,589)
  - EXACT, held everywhere: every pile (3,360 digests), attention output (840), stream (1,680) and the top light equal the
    reference run and capture's record; every walk's pile equal to the exact product; the 3x walk equal to 3r at every
    rank; the arrival's products equal the int products; the band read's top 5 and race winner 613/613 (score rows'
    sha256 613/613 against both), bound broken 0, none set aside at or above the 5th best. Greedy words 30/30 (28 = fp32's,
    2 = check30.log's machine words); the machine's race winner = the next greedy word 480/480. Order-free (texts 0-2):
    0 of 2,184 count arrays and 0 of 14 band-read arrays differ. Phase VI-a cross-checks: 0 differ.
    qwen_fringes/check.py (a second code path, same session): 1,872 plate counts and 12 band-read positions, 0 differ.
  - (a): 5,820,983,617 = 49.14% of the plates' plain. Held.
  - (c): THE BAND READ 302,579,448 = 6.74% of the plain read. MISSED (predicted 11.0-13.5%): words are set aside already
    after band 2 (places 18 and up): median 40,168 of 151,936 still in (least 113), not all of them, so band 3 (19.6% of
    the exposures' photons) is poured into about a quarter of the words; 8 still in after band 3. Against the fine
    diagonals' 15.84% on the same light: 0.43 of it.
  - (d): 34,208,215 = 56.23% of the arrival's plain. Held (34-40M).
  - (a) + (c) + (d): 6,157,771,280 = 37.55% of plain. The task's 37-40%: held; the session's 38.3-39.8%: missed low,
    because of (c).
  - (b): the pours in signed digits are 0.725 of the pours (-1,125.6M), the 3x walk is 1.139 x the walk (+1,967.1M), 3x's
    writing +1.1M: NET +842,601,328 = +5.14 points; all four 7,000,372,608 = 42.69%. The task's 31-34%: not reached;
    the session's +750-930M: held (43.5-44.6% missed only through (c)). (b) adds work at every kind (+1.17 to +4.53
    photons a pair). W1's writing (side): 0.546 of the pours, all four 44.32%.
  - Control (i): the gaps in a random neuron order: (a) 94.8-110.6% of plain per kind (banks' numbers), (a) + (b)
    155.8-181.7%. Control (ii): the band read on the shuffled light 7.76% against the real light's 6.81% at the same 154
    positions (held: within a point); words still in after band 2, median 71,848 against 42,629.
  - THE ROAD BLOCK OF (b), in the picture's own terms: a running product can be poured as ups and holes only once it is
    written that way, and writing it is a second walk along the same gaps; at 0.6B's widths the walk is 3.1-6.4 photons a
    pair while the ups and holes save 2.0-2.8 a pair. (b) pays only where the walk is thinner than about a quarter of the
    pours: the word table (walk 0.09 a pair) and wider plates (an ESTIMATE from Phase VI-a's 14B walk per pair: it would
    pay at o, gate, up and down, lose at k and v, and break about even at q).
  - Housekeeping: the first band-read run peaked at 4.17 GB (one whole-table check made int64 copies of the table), over
    the 3 GB rule; the check was made chunk by chunk and the run repeated at 1.62 GB, every array identical (the first
    run's files are kept as qwen_fringes/top_machine_over3GB.*). Every other run stayed under 2 GB.

RUNS (one python process at a time, each under 3 GB; outputs in qwen_fringes/):
    python3 qwen_fringes.py ref                 the reference run (qwen_base2.py's own steps) on the 30 refs texts
    python3 qwen_fringes.py machine             the machine on the 30 refs texts: every plate and the arrival, counted
    python3 qwen_fringes.py top set=machine     the band read at all 613 positions
    python3 qwen_fringes.py control             the band read on the shuffled light (every 4th position)
    python3 qwen_fringes.py orderfree           the machine on texts 0-2, every order random, every walk down
    python3 qwen_fringes.py top set=orderfree   the band read there, the words relabelled, walks down
    python3 qwen_fringes.py words_ref           the reference path's 16 greedy words on qwen/refs_qwen.json
    python3 qwen_fringes.py words_fr            the machine on those texts
    python3 qwen_fringes.py top set=words       the band read at the 16 read places of every text
    python3 qwen_fringes.py report
    python3 qwen_fringes/check.py               the separate recount (a sample of plates and band-read positions)
"""
import os, sys, json, time, ctypes, subprocess, resource, hashlib, zipfile, ast, warnings, numpy as np
from fractions import Fraction
warnings.filterwarnings("ignore", message=".*encountered in matmul")            # Accelerate's spurious flags; sums asserted
HERE = "/Users/abundancemachine/streaming-machine"; QD = f"{HERE}/qwen"; OUT = f"{HERE}/qwen_fringes"; NPZ = f"{QD}/qwen_base2.npz"
CAP = f"{HERE}/qwen06b"                                                          # Phase VI-a (read only)
PLAIN_PLATES, PLAIN_TOP, PLAIN_ARR = 11845741380, 4490888460, 60834749          # qwen06b/report.log, a token
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def digest(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def ones1(v): return bin(abs(int(v))).count("1")
def signed_digits(v):                                                           # (up, down) place masks, no two lit side by side
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a
    up, dn = (t & ~a) >> 1, (a & ~t) >> 1; neg = v < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)
def member(name):
    """an array stored in the npz, read in place (every member is stored flat)"""
    info = zipfile.ZipFile(NPZ).getinfo(name + ".npy"); assert info.compress_type == 0
    with open(NPZ, "rb") as f:
        f.seek(info.header_offset); h = f.read(30); assert h[:4] == b"PK\x03\x04"
        f.seek(info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        off = f.tell()
    return np.memmap(NPZ, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")

# ---------------- the two walks (qwen_fringes/walk2.c) ----------------
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{OUT}/walk2.c", f"{OUT}/walk2.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); P_, I_, i_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.walk2.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, P_, i_, I_, P_, P_, P_, P_, P_, P_]; _lib.walk2.restype = None
        _lib.band2.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, P_, i_, I_, P_, P_, P_, P_]; _lib.band2.restype = None
    return _lib

# ---------------- qwen_base2.py's own steps; only the products are hooked ----------------
Mz = np.load(NPZ)
class LazyW(dict):                                                              # W[name] -> (None, the plate's powers)
    def __getitem__(self, name):
        if name not in self.keys(): dict.__setitem__(self, name, (None, np.asarray(Mz[name + ".e"])))
        return dict.__getitem__(self, name)
ST = dict(mode="ref", order=None, rows=None, row0=0, text=0, cnt=None, dig=None, light_cmp=False, chk={}, arr_rng=None, vcache={})
def chk(k, v): ST["chk"][k] = ST["chk"].get(k, 0) + int(v)
def cadd(k, rows, v):
    C = ST["cnt"]
    if k not in C: C[k] = np.zeros(ST["P"], np.int64)
    C[k][rows] += v

def direct(x16, w):                                                             # CHECK PATH / reference: exact whole numbers
    p = np.asarray(x16, np.float64) @ np.asarray(w, np.float64); assert np.abs(p).max(initial=0) < 2 ** 52
    return p.astype(np.int64)
HEAD_W = member("head.w")
def head_product(x16):                                                          # the reference read only
    out = np.empty((len(x16), HEAD_W.shape[1]), np.int64); xf = np.asarray(x16, np.float64)
    for c in range(0, HEAD_W.shape[1], 16384):
        p = xf @ np.asarray(HEAD_W[:, c:c + 16384], np.float64); assert np.abs(p).max(initial=0) < 2 ** 52; out[:, c:c + 16384] = p.astype(np.int64)
    return out

KINDS = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
         "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}
KIDX = dict(q=0, k=1, v=2, o=3, gate=4, up=5, down=6)                          # banks.py's seed index
class Lineup:
    """way 2's record of one plate (laid once): each line's weights in order of size across the plate's neurons (ties in
    neuron order): NR the neuron at each rank, G the gaps (G[:, 0] unused), LOW, TOP; per line the gaps' lit digits GN
    (the lowest weight included), the plate's 1s WROW and lit digits WNROW (for the plain and signed counts), and the
    CONTROL's GNS: the gaps' lit digits with each line's neurons in a random order (banks.py's draw, count only)."""
    def __init__(self, w, kind, l):
        w = np.asarray(w).astype(np.int64); L, N = w.shape; self.L, self.N = L, N
        o = np.argsort(w, axis=1, kind="stable"); ws = np.take_along_axis(w, o, 1); g = np.diff(ws, axis=1)
        assert (g >= 0).all() and g.max(initial=0) < 1 << 16 and N < 1 << 16
        back = np.empty_like(w); np.put_along_axis(back, o, np.concatenate([ws[:, :1], ws[:, :1] + np.cumsum(g, 1)], 1), 1)
        assert np.array_equal(back, w), "the record does not give the plate back"
        self.NR = np.ascontiguousarray(o.astype(np.uint16)); self.G = np.zeros((L, N), np.uint16); self.G[:, 1:] = g
        self.LOW = np.ascontiguousarray(ws[:, 0]); self.TOP = np.ascontiguousarray(ws[:, -1])
        self.GN = naf(ws[:, 0]) + naf(g).sum(1); self.ZG = int((g == 0).sum())
        self.WROW = ones(w).sum(1); self.WNROW = naf(w).sum(1)
        rng = np.random.default_rng([2026, l, KIDX[kind]])
        os_ = rng.permuted(np.tile(np.arange(N), (L, 1)), axis=1); wsh = np.take_along_axis(w, os_, 1)
        self.GNS = naf(wsh[:, 0]) + naf(np.diff(wsh, axis=1)).sum(1)
    def walk(self, x, line_order=None, down=False):
        x = np.ascontiguousarray(np.asarray(x, np.int64)); P, L = x.shape; assert L == self.L and np.abs(x).max(initial=0) <= 1 << 15
        LO = np.ascontiguousarray(np.arange(L, dtype=np.int64) if line_order is None else np.asarray(line_order, np.int64))
        assert np.array_equal(np.sort(LO), np.arange(L))
        pile = np.zeros((P, self.N), np.int64); a = [np.zeros(P, np.int64) for _ in range(5)]
        lib().walk2(P, L, self.N, x.ctypes.data, self.LOW.ctypes.data, self.G.ctypes.data, self.NR.ctypes.data, self.TOP.ctypes.data,
                    LO.ctypes.data, -1 if down else 1, 4, pile.ctypes.data, *(v.ctypes.data for v in a))
        return pile, a                                                         # pours 1s, pours signed, W1 writing, open, 3x walk off

def fr_plate(x16, slab, l):
    """additions a and b at one slab: each plate of the slab walked (and its 3x walk); every pile checked exact"""
    W16 = np.asarray(Mz[f"{slab}.{l}.w"]); P = len(x16); out = np.empty((P, W16.shape[1]), np.int64)
    rows = slice(ST["row0"], ST["row0"] + P); x3 = 3 * x16
    nx, ox, n3 = naf(x16), ones(x16), naf(x3)
    cadd(f"{slab}.{l}|writing", rows, ox.sum(1)); cadd(f"{slab}.{l}|writing3", rows, ones(x3).sum(1))
    for kind, c0, c1 in KINDS[slab]:
        w = W16[:, c0:c1]; lu = Lineup(w, kind, l)
        lo = None if ST["order"] is None else np.random.default_rng([ST["order"], l, KIDX[kind]]).permutation(lu.L)
        pile, (p1, ps, w1, opn, bad3) = lu.walk(x16, line_order=lo, down=ST["order"] is not None)
        chk("plate: walk vs the exact product (rows differing)", (pile != direct(x16, w)).any(1).sum())
        chk("plate: a walk down not closed at 0", opn.sum()); chk("plate: the 3x walk not 3 r (ranks)", bad3.sum())
        k = f"{kind}.{l}|"
        for item, v in (("plain", ox @ lu.WROW), ("signed", nx @ lu.WNROW), ("gaps", nx @ lu.GN), ("gaps_shuf", nx @ lu.GNS),
                        ("walk3", n3 @ lu.GN), ("walk3_shuf", n3 @ lu.GNS), ("pours", p1), ("pours_sd", ps), ("write1", w1)):
            cadd(k + item, rows, v)
        ST["zg"][f"{kind}.{l}"] = lu.ZG
        out[:, c0:c1] = pile; del lu, pile
    return out

CAPL = {}
def PRODUCT(x16, ex, name):
    x16 = np.asarray(x16, np.int64); assert np.abs(x16).max(initial=0) <= 1 << 15
    if name == "head":
        assert ST["mode"] == "ref", "the machine reads the top by the band pour"; return head_product(x16)
    slab, l = name.split("."); l = int(l)
    if ST["light_cmp"]:                                                        # capture's record of qwen_base2.py's own light
        if slab not in CAPL: CAPL[slab] = np.load(f"{CAP}/light/{slab}_light.npy", mmap_mode="r")
        chk("light at the plates vs capture's record (rows differing)", (np.asarray(CAPL[slab][l][:len(x16)]).astype(np.int64) != x16).any(1).sum())
    pile = direct(x16, Mz[name + ".w"]) if ST["mode"] == "ref" else fr_plate(x16, slab, l)
    for t, (a, b) in enumerate(ST["rows"]): ST["dig"][f"pile|{name}|{t}"] = digest(pile[a:b])
    return pile

def sd_places(X):
    up, dn = signed_digits(X); return max(1, int(np.bitwise_or.reduce((up | dn).ravel(), initial=0)).bit_length())
def sd_planes(X, n):                                                            # X's signed digits as n planes of -1 / 0 / +1
    up, dn = signed_digits(X); assert not ((up | dn) >> n).any()
    return np.stack([((up >> a) & 1) - ((dn >> a) & 1) for a in range(n)]).astype(np.float64)
def TW(na, nb): return np.left_shift(np.int64(1), np.arange(na)[:, None] + np.arange(nb)[None, :])
def QK(q16, k16, h, l):
    """q16 . k16 (every pair of positions of this text, one head). ref: qwen_base2's own int product. fr: from signed-digit
    planes, place pair by place pair; counted; checked equal to the int product."""
    plain = q16 @ k16.T
    if ST["mode"] == "ref": return plain
    P = len(q16); rows = slice(ST["row0"], ST["row0"] + P)
    oq, ok_, nq, nk = ones(q16), ones(k16), naf(q16), naf(k16)
    cadd(f"arr.{l}|plain_scores", rows, np.tril(oq @ ok_.T).sum(1)); cadd(f"arr.{l}|sd_scores", rows, np.tril(nq @ nk.T).sum(1))
    cadd(f"arr.{l}|write_q", rows, oq.sum(1))
    if h % 2 == 0: cadd(f"arr.{l}|write_k", rows, ok_.sum(1))
    qq, kk = q16, k16
    if ST["arr_rng"] is not None: perm = ST["arr_rng"].permutation(q16.shape[1]); qq, kk = q16[:, perm], k16[:, perm]
    na, nb = sd_places(qq), sd_places(kk); assert na + nb + 8 < 62
    C = (sd_planes(qq, na).reshape(na * P, -1) @ sd_planes(kk, nb).reshape(nb * P, -1).T).reshape(na, P, nb, P)
    S = np.tensordot(C.astype(np.int64), TW(na, nb), axes=([0, 2], [0, 1]))
    chk("arrival: q.k from signed digits vs the int product", (S != plain).sum())
    return S
def AV(a, v, p, h, l):
    """shares x values at position p (one head). ref: qwen_base2's own a @ v. fr: from signed-digit planes; counted."""
    plain = a[None, :] @ v[:p + 1]
    if ST["mode"] == "ref": return plain
    key = (l, h // 2, ST["text"])
    if key not in ST["vcache"]:
        ST["vcache"].clear(); nbv = sd_places(v)
        ST["vcache"][key] = (np.ascontiguousarray(sd_planes(v, nbv).transpose(1, 0, 2)), nbv, ones(v).sum(1), naf(v).sum(1))
    VP, nbv, ovs, nvs = ST["vcache"][key]
    r = ST["row0"] + p
    cadd(f"arr.{l}|plain_values", r, int(ones(a) @ ovs[:p + 1])); cadd(f"arr.{l}|sd_values", r, int(naf(a) @ nvs[:p + 1]))
    if h % 2 == 0: cadd(f"arr.{l}|write_v", r, int(ovs[p]))
    aa, vp = a, VP[:p + 1]
    if ST["arr_rng"] is not None: perm = ST["arr_rng"].permutation(p + 1); aa, vp = a[perm], vp[perm]
    na = sd_places(aa); assert na + nbv + 6 < 62
    C = (sd_planes(aa, na) @ vp.reshape(p + 1, -1)).reshape(na, nbv, -1)
    S = np.tensordot(TW(na, nbv), C.astype(np.int64), axes=([0, 1], [0, 1]))[None, :]
    chk("arrival: shares x values from signed digits vs the int product", (S != plain).sum())
    return S

_src = open(f"{QD}/qwen_base2.py").read(); _src = _src[:_src.index("from transformers import AutoTokenizer")]
for _a, _b in [('M = np.load("qwen_base2.npz")', 'M = Mz'),
               ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = LazyW()'),
               ("w, we = W[name]; x16, ex = to16(R); pile = x16 @ w", "w, we = W[name]; x16, ex = to16(R); pile = PRODUCT(x16, ex, name)"),
               ("w, we = W[name]; x16, ex = to16(X); return rshift(x16 @ w,", "w, we = W[name]; x16, ex = to16(X); return rshift(PRODUCT(x16, ex, name),"),
               ("sc = rshift(q16 @ k16.T,", "sc = rshift(QK(q16, k16, h, l),"),
               ("out[p, h * HD:(h + 1) * HD] = power(a[None, :] @ v[:p + 1],", "out[p, h * HD:(h + 1) * HD] = power(AV(a, v, p, h, l),")]:
    assert _src.count(_a) == 1, _a; _src = _src.replace(_a, _b)
B = dict(Mz=Mz, LazyW=LazyW, PRODUCT=PRODUCT, QK=QK, AV=AV)
_argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
assert (B["GAUGE"], B["TURN"], B["BEND"], B["ARRIVAL"]) == ("int",) * 4
to16, gauge, gauged_slab, plain_slab, attention, bend, race, power = (B[k] for k in ("to16", "gauge", "gauged_slab", "plain_slab", "attention", "bend", "race", "power"))
L, D, ES, EXP2 = B["L"], B["D"], B["ES"], B["EXP2"]
EMB = member("embed")

def top5(S, labels=None):                                                       # by score, ties by word number (capture.py's rule)
    S = np.atleast_2d(S); lab = np.arange(S.shape[1]) if labels is None else labels
    return np.array([lab[np.lexsort((lab, -s))[:5]] for s in S])

def forward(texts, mode, order=None, light_cmp=False, log=print):
    """qwen_base2.py's own forward, every text layer by layer together (each text's arrival reads only its own
    positions). Records the digest of every pile, attention output and stream (both halves of every layer) per text."""
    t0 = time.time(); lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    rows = [(int(off[i]), int(off[i + 1])) for i in range(len(texts))]; P = int(off[-1])
    ST.update(mode=mode, order=order, rows=rows, row0=0, text=0, cnt={}, dig={}, light_cmp=light_cmp, P=P, zg={}, vcache={},
              arr_rng=None if order is None else np.random.default_rng([order, 99]))
    R = np.asarray(EMB[np.concatenate([np.asarray(t, np.int64) for t in texts])]).astype(np.int64)
    for l in range(L):
        QKV = gauged_slab(R, f"qkv.{l}"); outs = []
        for t, (a, b) in enumerate(rows):
            ST["row0"], ST["text"] = a, t; outs.append(attention(QKV[a:b], l))
        ST["row0"] = 0; o = np.concatenate(outs); del QKV, outs
        for t, (a, b) in enumerate(rows): ST["dig"][f"arrival|{l}|{t}"] = digest(o[a:b])
        R = R + plain_slab(o, f"o.{l}"); del o
        for t, (a, b) in enumerate(rows): ST["dig"][f"stream|{l}.0|{t}"] = digest(R[a:b])
        GU = gauged_slab(R, f"gu.{l}")
        R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}"); del GU
        for t, (a, b) in enumerate(rows): ST["dig"][f"stream|{l}.1|{t}"] = digest(R[a:b])
        if l % 7 == 6 or l == 0: log(f"    {mode}: layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    x16, ex = to16(R); LS = gauge(x16, ex, D)
    for t, (a, b) in enumerate(rows): ST["dig"][f"top light|{t}"] = digest(np.concatenate([x16[a:b], ex[a:b, None], np.asarray(LS)[a:b, None]], 1))
    return dict(R=R, x16=x16, ex=np.asarray(ex, np.int64), LS=np.asarray(LS, np.int64), rows=rows, dig=dict(ST["dig"]), cnt=dict(ST["cnt"]))

def ref_read(R, rows, sel=None):
    """the reference read: qwen_base2's own gauged_slab(R, "head") at the positions given (every position if sel is None)"""
    sel = np.arange(len(R)) if sel is None else np.asarray(sel); T5, WIN, SHA = [], [], []
    for c in range(0, len(sel), 24):
        S = gauged_slab(R[sel[c:c + 24]], "head")
        T5 += [top5(s)[0].tolist() for s in S]; WIN += [race(s) for s in S]; SHA += [hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).hexdigest() for s in S]
    return dict(top5=T5, win=WIN, sha=SHA)

# ---------------- (c) the band read ----------------
NB = 8; PB = [26, 22, 18, 14, 10, 6, 2, 0]; NT = NB - 1
class Table:
    """the word table lined up once (way 2's record): per line the word at each rank (NR, uint32), the gaps (G, uint16,
    G[:, 0] unused), LOW, TOP, the gaps' lit digits GN; the plain read's 1s per line PPH; each word's shift KW.
    word_perm: the words relabelled (ORDER-FREE), the line-up laid again on the relabelled table."""
    def __init__(self, word_perm=None, log=print):
        t0 = time.time()
        T = np.ascontiguousarray(member("head.w")); HE = np.asarray(Mz["head.e"]).astype(np.int64)
        if word_perm is not None: T = np.ascontiguousarray(T[:, word_perm]); HE = HE[word_perm]
        self.wperm = np.arange(T.shape[1]) if word_perm is None else np.asarray(word_perm)
        self.L, self.N = T.shape; self.EMIN = int(HE.min()); self.KW = HE - self.EMIN; assert self.KW.max() <= 3
        self.NR = np.empty((self.L, self.N), np.uint32); self.G = np.zeros((self.L, self.N), np.uint16)
        self.LOW = np.empty(self.L, np.int64); self.TOP = np.empty(self.L, np.int64); self.GN = np.empty(self.L, np.int64)
        self.PPH = np.empty(self.L, np.int64); zg = 0
        for i0 in range(0, self.L, 16):
            wb = T[i0:i0 + 16].astype(np.int64); o = np.argsort(wb, axis=1, kind="stable"); ws = np.take_along_axis(wb, o, 1); g = np.diff(ws, axis=1)
            assert np.abs(wb).max() < 1 << 15
            assert (g >= 0).all() and g.max(initial=0) < 1 << 16
            back = np.empty_like(wb); np.put_along_axis(back, o, np.concatenate([ws[:, :1], ws[:, :1] + np.cumsum(g, 1)], 1), 1)
            assert np.array_equal(back, wb), "the record does not give the table back"
            self.NR[i0:i0 + 16] = o; self.G[i0:i0 + 16, 1:] = g; self.LOW[i0:i0 + 16] = ws[:, 0]; self.TOP[i0:i0 + 16] = ws[:, -1]
            self.GN[i0:i0 + 16] = naf(ws[:, 0]) + naf(g).sum(1); self.PPH[i0:i0 + 16] = ones(wb).sum(1); zg += int((g == 0).sum())
        self.zero_steps = zg; self.T = T
        log(f"  the word table lined up: {self.L} lines x {self.N} words, shifts {np.bincount(self.KW).tolist()}; read back exact; "
            f"zero steps {zg / (self.L * (self.N - 1)) * 100:.2f}% ({time.time() - t0:.0f}s, {rss()} MB)")
    def walk(self, X, line_order=None, down=False):
        X = np.ascontiguousarray(np.asarray(X, np.int64)); P = len(X); assert np.abs(X).max(initial=0) < 1 << 15
        LO = np.ascontiguousarray(np.arange(self.L, dtype=np.int64) if line_order is None else np.asarray(line_order, np.int64))
        DN = np.zeros((P, self.N, NB), np.int16); HN = np.zeros((P, self.N, NB), np.uint16); op = np.zeros(P, np.int64); big = np.zeros(P, np.int64)
        lib().band2(P, self.L, self.N, X.ctypes.data, self.LOW.ctypes.data, self.G.ctypes.data, self.NR.ctypes.data, self.TOP.ctypes.data,
                    LO.ctypes.data, -1 if down else 1, 1, DN.ctypes.data, HN.ctypes.data, op.ctypes.data, big.ctypes.data)
        assert not op.any(), "a walk down did not close at 0"; assert not big.any(), "an exposure reached 2^30"
        return DN, HN
    def pile(self, X):                                                          # CHECK PATH: the exact product, every word
        out = np.empty((len(X), self.N), np.int64); xf = np.asarray(X, np.float64)
        for c in range(0, self.N, 8192):
            p = xf @ self.T[:, c:c + 8192].astype(np.float64); assert np.abs(p).max(initial=0) < 2 ** 52; out[:, c:c + 8192] = p.astype(np.int64)
        return out

TOPKEYS = ["walk", "writing", "end", "plain_meet", "plain_div", "n", "mu"]
def band_read(tab, X, EX, LS, line_rng=None, down=False, batch=10, log=print):
    X = np.asarray(X, np.int64); N_ = len(X); NW = tab.N; bad = {}
    def nb(k, v): bad[k] = bad.get(k, 0) + int(v)
    out = {k: np.zeros(N_, np.int64) for k in TOPKEYS}
    out.update(pours=np.zeros((N_, NB), np.int64), hall=np.zeros((N_, NB), np.int64), surv=np.zeros((N_, NT), np.int64),
               bound=np.zeros((N_, NT), np.int64), cmp=np.zeros((N_, NT), np.int64), top5=np.zeros((N_, 5), np.int64),
               win=np.zeros(N_, np.int64), ptop5=np.zeros((N_, 5), np.int64), pwin=np.zeros(N_, np.int64))
    sha = []; inv = np.argsort(tab.wperm); KWall = tab.KW; PBa = np.array(PB, np.int64); t0 = time.time()
    for c0 in range(0, N_, batch):
        Xb = X[c0:c0 + batch]; P = len(Xb)
        F0 = (tab.EMIN + EX[c0:c0 + P] + ES) * 65536 - LS[c0:c0 + P]
        DN, HN = tab.walk(Xb, line_order=None if line_rng is None else line_rng.permutation(tab.L), down=down)
        PILE = tab.pile(Xb); A = PILE << KWall[None, :]; S = power(A, F0[:, None])          # CHECK PATH: every word's score
        for p in range(P):
            q = c0 + p; x = Xb[p]; n = int((x != 0).sum())
            sha.append(hashlib.sha256(np.ascontiguousarray(S[p][inv], np.int64).tobytes()).hexdigest())
            Fp = int(F0[p]); I = Fp >> 16; e = int(EXP2[Fp & 0xFFFF]); s = 16 - I
            twos = Fraction(2) ** s; delta = max(1, int(-(-(twos.numerator) // (e * twos.denominator)))); mu = 2 * 1 + delta; ge = ones1(e)
            out["mu"][q] = mu; out["n"][q] = n
            out["walk"][q] = int(naf(x) @ tab.GN); out["writing"][q] = int(ones(x).sum())
            out["plain_meet"][q] = int(ones(x) @ tab.PPH); out["plain_div"][q] = int(ones(PILE[p]).sum()) * ge
            Dp = DN[p].astype(np.int64) << PBa[None, :]; CUM = np.cumsum(Dp, axis=1); Hp = HN[p].astype(np.int64)
            nb("bands summed vs the exact pile", (CUM[:, -1] != PILE[p]).sum())
            out["hall"][q] = Hp.sum(0)
            s5 = np.partition(S[p], NW - 5)[NW - 5]
            alive = np.arange(NW); V = np.zeros(NW, np.int64); K = KWall.copy()
            for b in range(NB):
                out["pours"][q, b] = int(Hp[alive, b].sum())                    # band b poured into the words still in
                V = V + (Dp[alive, b] << K)
                if b == NT: break
                R = n * ((1 << PB[b]) - 1)
                nb("bound broken (any word, check)", (np.abs(PILE[p] - CUM[:, b]) > R).sum())
                nb("counted pile carried = the counted pile", (V != (CUM[alive, b] << K)).sum())
                lows = []; ph = ones1(n); cmp = 2 * len(alive)
                for k in range(4):
                    Vk = V[K == k]
                    if len(Vk) == 0: continue
                    t5 = np.partition(Vk, len(Vk) - min(5, len(Vk)))[len(Vk) - min(5, len(Vk)):]
                    lows += list(t5 - (R << k)); ph += len(t5) * ones1(R)
                lows = np.sort(np.array(lows, np.int64)); L5 = int(lows[-5]); cmp += len(lows)
                THR = np.zeros(4, np.int64)
                for k in range(4):
                    if (K == k).any():
                        thr = (R << k) + mu; THR[k] = L5 - thr; ph += ones1(mu) + ones1(thr)
                keep = V > THR[K]
                out["bound"][q, b] = ph; out["cmp"][q, b] = cmp
                nb("a word set aside that is not below the 5th-best", (S[p, alive[~keep]] >= s5).sum())
                alive, V, K = alive[keep], V[keep], K[keep]; out["surv"][q, b] = len(alive)
            sc = power(V, Fp)                                                   # the words still in, read to the end
            nb("words read to the end vs the plain read", (sc != S[p, alive]).sum())
            out["end"][q] = int(ones(V).sum()) * ge
            Sp = np.full(NW, -(1 << 62), np.int64); Sp[alive] = sc
            out["top5"][q] = top5(Sp, tab.wperm)[0]; out["win"][q] = tab.wperm[race(Sp)]
            out["ptop5"][q] = top5(S[p], tab.wperm)[0]; out["pwin"][q] = tab.wperm[race(S[p])]
        del DN, HN, PILE, A, S
        if (c0 // batch) % 10 == 0 or c0 + batch >= N_: log(f"    {min(N_, c0 + batch)}/{N_} positions read ({time.time() - t0:.0f}s, {rss()} MB)")
    out["bad"] = bad; out["sha"] = sha
    return out
def band_total(o): return o["walk"] + o["writing"] + o["pours"].sum(1) + o["bound"].sum(1) + o["end"]
def band_plain(o): return o["plain_meet"] + o["plain_div"]

# ---------------- phases ----------------
def opener(tag):
    f = open(f"{OUT}/{tag}.log", "w")
    def log(s=""): print(s, flush=True); f.write(s + "\n"); f.flush()
    return log
def META(): return json.load(open(f"{CAP}/light/meta.json"))
def save_counts(path, cnt): np.savez(path, **{k.replace("|", "__"): v for k, v in cnt.items()})
def load_counts(path): Z = np.load(path); return {k.replace("__", "|"): Z[k] for k in Z.files}

def phase_ref():
    log = opener("ref"); t0 = time.time(); M_ = META(); texts = M_["texts"]
    log(f"REFERENCE: qwen_base2.py's own steps (exact whole-number products) on the 30 refs texts in Qwen's words, {M_['npos']} positions")
    res = forward(texts, "ref", light_cmp=True, log=log)
    pd = sum(res["dig"][f"pile|{n}|{t}"] != M_["dig"][n][t] for n in M_["dig"] for t in range(len(texts)))
    log(f"  piles vs capture's digests: {pd} of {len(M_['dig']) * len(texts)} differ; checks: {ST['chk']}")
    rd = ref_read(res["R"], res["rows"])
    log(f"  the read at every position vs capture's: score rows (sha256) {sum(a == b for a, b in zip(rd['sha'], M_['sha']))}/{M_['npos']}, "
        f"top 5 {sum(a == b for a, b in zip(rd['top5'], M_['top5']))}/{M_['npos']}, race winner {sum(a == b for a, b in zip(rd['win'], M_['win']))}/{M_['npos']}")
    json.dump(dict(dig=res["dig"], read=rd, chk=ST["chk"]), open(f"{OUT}/ref.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_machine(order=None, tag="machine", ntext=None):
    log = opener(tag); t0 = time.time(); M_ = META(); texts = M_["texts"][:ntext]; RF = json.load(open(f"{OUT}/ref.json"))
    log(f"THE MACHINE ({tag}): way 2 + the 3x walk at every plate, the arrival in signed digits; {len(texts)} texts, "
        f"{sum(map(len, texts))} positions" + ("" if order is None else f"; ORDER-FREE (seed {order}): every line order random, every walk down"))
    res = forward(texts, "fr", order=order, light_cmp=True, log=log)
    dd = {}
    for k, v in res["dig"].items():
        part = k.split("|")[0]; dd.setdefault(part, [0, 0]); dd[part][0] += 1; dd[part][1] += v != RF["dig"][k]
    log("  against the reference run: " + ", ".join(f"{p} {d} of {n} differ" for p, (n, d) in dd.items()))
    pd = sum(res["dig"][f"pile|{n}|{t}"] != M_["dig"][n][t] for n in M_["dig"] for t in range(len(texts)))
    log(f"  piles against capture's record of qwen_base2.py's own run: {pd} of {len(M_['dig']) * len(texts)} differ")
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in ST["chk"].items()))
    save_counts(f"{OUT}/{tag}_counts.npz", res["cnt"]); json.dump(dict(zg=ST["zg"], chk=ST["chk"], dig=res["dig"]), open(f"{OUT}/{tag}.json", "w"))
    np.savez(f"{OUT}/{tag}_top_light.npz", x16=res["x16"].astype(np.int32), ex=res["ex"], LS=res["LS"], rows=np.array(res["rows"]))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_top(SET):
    log = opener(f"top_{SET}"); t0 = time.time()
    Z = np.load(f"{OUT}/{'words_fr' if SET == 'words' else SET}_top_light.npz"); X, EX, LS, rows = Z["x16"].astype(np.int64), Z["ex"], Z["LS"], Z["rows"]
    sel = np.arange(len(X)); wperm = None; line_rng = None; down = False
    if SET == "words":
        W = json.load(open(f"{OUT}/words_ref.json")); sel = np.array(W["read_rows"]); REF = dict(top5=W["top5"], win=W["win"], sha=W["sha"])
    else:
        RF = json.load(open(f"{OUT}/ref.json"))["read"]; REF = {k: v[:len(X)] for k, v in RF.items()}
    if SET == "orderfree":
        wperm = np.random.default_rng([7, 151936]).permutation(151936); line_rng = np.random.default_rng([7, 1024]); down = True
    log(f"THE BAND READ, set {SET}: {len(sel)} positions" + ("" if wperm is None else "; ORDER-FREE: words relabelled, lines in random orders, walks down"))
    tab = Table(word_perm=wperm, log=log)
    R = band_read(tab, X[sel], EX[sel], LS[sel], line_rng=line_rng, down=down, log=log); n = len(sel)
    log(f"EXACT: top 5 (set and order) = the plain read's {int((R['top5'] == R['ptop5']).all(1).sum())}/{n}; race winner {int((R['win'] == R['pwin']).sum())}/{n}")
    log(f"  against the reference run of qwen_base2.py's own read: score rows (sha256) {sum(a == b for a, b in zip(R['sha'], REF['sha']))}/{n}, "
        f"top 5 {int((R['top5'] == np.array(REF['top5'])).all(1).sum())}/{n}, race winner {int((R['win'] == np.array(REF['win'])).sum())}/{n}")
    if SET == "machine":
        M_ = META(); log(f"  against capture's record: score rows {sum(a == b for a, b in zip(R['sha'], M_['sha']))}/{n}, top 5 {int((R['top5'] == np.array(M_['top5'])).all(1).sum())}/{n}, "
                         f"race winner {int((R['win'] == np.array(M_['win'])).sum())}/{n}")
    if SET == "words": log(f"  race winner = the next greedy word (qwen_base2's own): {int((R['win'] == np.array(W['next'])).sum())}/{n}")
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in sorted(R["bad"].items())))
    tot, pl = band_total(R), band_plain(R)
    log(f"WORK A POSITION (mean over {n}): plain read {pl.sum() / n:,.0f}; band read {tot.sum() / n:,.0f} = {tot.sum() / pl.sum() * 100:.2f}%")
    for k in ("walk", "writing", "end"): log(f"    {k:10s} {R[k].sum() / n:14,.0f}  {R[k].sum() / pl.sum() * 100:6.3f}%")
    log(f"    pours      {R['pours'].sum() / n:14,.0f}  {R['pours'].sum() / pl.sum() * 100:6.3f}%   (every band into every word: {R['hall'].sum() / n:,.0f}, {R['hall'].sum() / pl.sum() * 100:.2f}%)")
    log(f"    bound      {R['bound'].sum() / n:14,.0f}  {R['bound'].sum() / pl.sum() * 100:6.3f}%   compares (a count) {R['cmp'].sum() / n:,.0f}")
    for b in range(NB):
        hi = 29 if b == 0 else PB[b - 1] - 1; sv = R["surv"][:, b] if b < NT else None
        svs = f"{int(np.median(sv)):7d} {sv.mean():10.1f} {sv.min():7d} {sv.max():7d}" if sv is not None else f"{'(read to the end)':>34s}"
        log(f"  band {b} (places {hi:2d}-{PB[b]:2d}): words still in after its test (median/mean/least/most) {svs}   pours {R['pours'][:, b].sum() / n:13,.0f} "
            f"({R['pours'][:, b].sum() / pl.sum() * 100:5.2f}%)   its photons over every word {R['hall'][:, b].sum() / R['hall'].sum() * 100:5.2f}%")
    np.savez(f"{OUT}/top_{SET}.npz", sel=sel, **{k: v for k, v in R.items() if k not in ("bad", "sha")})
    json.dump(dict(bad=R["bad"], sha=R["sha"], zero_steps=tab.zero_steps), open(f"{OUT}/top_{SET}.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_control():
    log = opener("control"); t0 = time.time()
    Z = np.load(f"{OUT}/machine_top_light.npz"); idx = np.arange(0, len(Z["x16"]), 4)
    X = Z["x16"].astype(np.int64)[idx]; EX, LS = Z["ex"][idx], Z["LS"][idx]
    X = np.random.default_rng([2026, 7]).permuted(X, axis=1)                     # qwen06b/top_read.py's draw
    log(f"CONTROL: the band read on the light shuffled across its lines, every 4th position ({len(idx)})")
    tab = Table(log=log); R = band_read(tab, X, EX, LS, log=log); n = len(idx)
    log(f"  exact on its own light: top 5 = its plain read {int((R['top5'] == R['ptop5']).all(1).sum())}/{n}; checks: " + ", ".join(f"{k} {v}" for k, v in sorted(R["bad"].items())))
    np.savez(f"{OUT}/top_control.npz", sel=idx, **{k: v for k, v in R.items() if k not in ("bad", "sha")})
    json.dump(dict(bad=R["bad"]), open(f"{OUT}/top_control.json", "w"))
    tot, pl = band_total(R), band_plain(R); log(f"  band read {tot.sum() / pl.sum() * 100:.2f}% of its plain read; survivors (median) after bands 2, 3, 4: "
                                                 f"{np.median(R['surv'][:, 2]):.0f}, {np.median(R['surv'][:, 3]):.0f}, {np.median(R['surv'][:, 4]):.0f}")
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def check30_words():
    """check30.log's printed machine words (qwen_base2.py's own greedy run), by prompt"""
    out = {}
    for line in open(f"{QD}/check30.log"):
        for tag in ("IDENTICAL", "DIFFERENT"):
            if f" {tag}  " in line:
                pr, rest = line.strip().split(f" {tag}  ", 1); rest = rest.split("   model: ")[0]
                out[ast.literal_eval(pr.strip())] = ast.literal_eval(rest.strip())
    return out
def phase_words_ref():
    log = opener("words_ref"); t0 = time.time(); RQ = json.load(open(f"{QD}/refs_qwen.json"))
    words = [list(r["ref"]) for r in RQ]; done = [0] * len(RQ); it = 0
    log(f"THE REFERENCE PATH'S GREEDY WORDS (qwen_base2.py's own steps) on qwen/refs_qwen.json: {len(RQ)} prompts, 16 words each; "
        "start from fp32's words as the teacher; each pass fixes the first word that differs and checks the rest")
    while True:
        act = [i for i in range(len(RQ)) if done[i] < 16]
        if not act: break
        it += 1; texts = [RQ[i]["ids"] + words[i][:15] for i in act]
        res = forward(texts, "ref", log=lambda s: None)
        sel = [res["rows"][j][0] + len(RQ[i]["ids"]) - 1 + t for j, i in enumerate(act) for t in range(16)]
        win = ref_read(res["R"], res["rows"], sel)["win"]
        for j, i in enumerate(act):
            w = win[16 * j:16 * (j + 1)]; t = next((t for t in range(done[i], 16) if w[t] != words[i][t]), 16)
            if t < 16: words[i] = words[i][:t] + [w[t]] + words[i][t + 1:]
            done[i] = 16 if t == 16 else t + 1
        log(f"  pass {it}: {len(act)} texts read; still open: {sum(d < 16 for d in done)}  ({time.time() - t0:.0f}s)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); c30 = check30_words()
    same30 = [tok.decode(words[i]) == c30[RQ[i]["prompt"]] for i in range(len(RQ))]
    log(f"  decoded greedy words = check30.log's machine words: {sum(same30)}/{len(RQ)}; = fp32's 16 words: {sum(words[i] == RQ[i]['ref'] for i in range(len(RQ)))}/{len(RQ)}")
    for i in range(len(RQ)):
        if words[i] != RQ[i]["ref"]: log(f"    {RQ[i]['prompt']!r}: {tok.decode(words[i])!r}   (fp32: {tok.decode(RQ[i]['ref'])!r})")
    texts = [RQ[i]["ids"] + words[i][:15] for i in range(len(RQ))]
    res = forward(texts, "ref", log=log)
    sel = [res["rows"][i][0] + len(RQ[i]["ids"]) - 1 + t for i in range(len(RQ)) for t in range(16)]
    rd = ref_read(res["R"], res["rows"], sel)
    log(f"  final pass: race winner = the greedy word at {sum(a == b for a, b in zip(rd['win'], sum(words, [])))}/{len(sel)} read places")
    json.dump(dict(words=words, texts=texts, read_rows=sel, next=sum(words, []), top5=rd["top5"], win=rd["win"], sha=rd["sha"], dig=res["dig"],
                   same_check30=same30), open(f"{OUT}/words_ref.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
def phase_words_fr():
    log = opener("words_fr"); t0 = time.time(); W = json.load(open(f"{OUT}/words_ref.json"))
    log(f"THE MACHINE on the 30 greedy texts (prompt + the first 15 greedy words): {sum(map(len, W['texts']))} positions")
    res = forward(W["texts"], "fr", log=log)
    dd = {}
    for k, v in res["dig"].items():
        part = k.split("|")[0]; dd.setdefault(part, [0, 0]); dd[part][0] += 1; dd[part][1] += v != W["dig"][k]
    log("  against the reference path on the same texts: " + ", ".join(f"{p} {d} of {n} differ" for p, (n, d) in dd.items()))
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in ST["chk"].items()))
    np.savez(f"{OUT}/words_fr_top_light.npz", x16=res["x16"].astype(np.int32), ex=res["ex"], LS=res["LS"], rows=np.array(res["rows"]))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_report():
    """reads only qwen_fringes/'s own results and Phase VI-a's (for the cross-checks); prints and writes report.log"""
    import glob
    log = opener("report"); M_ = META(); NP = M_["npos"]
    C = load_counts(f"{OUT}/machine_counts.npz"); T = np.load(f"{OUT}/top_machine.npz"); MJ = json.load(open(f"{OUT}/machine.json"))
    BK = {}
    for f in sorted(glob.glob(f"{CAP}/banks_*_*.json")): BK.update(json.load(open(f)))
    TW_ = json.load(open(f"{CAP}/table_walk.json")); FINE = json.load(open(f"{CAP}/top_read_main.json"))
    KN = ["q", "k", "v", "o", "gate", "up", "down"]; SL = dict(q="qkv", k="qkv", v="qkv", o="o", gate="gu", up="gu", down="down")
    def S(kind, item): return int(sum(C[f"{kind}.{l}|{item}"].sum() for l in range(L)))
    def slab_sum(item, slabs=("qkv", "o", "gu", "down")): return int(sum(C[f"{s}.{l}|{item}"].sum() for s in slabs for l in range(L)))
    def A(item): return int(sum(C[f"arr.{l}|{item}"].sum() for l in range(L)))
    def pt(v): return v / NP
    # ---- cross-checks with Phase VI-a ----
    bad = []
    for kind in KN:
        for l in range(L):
            r = BK[f"{kind}.{l}"]
            for mine, theirs in (("plain", "plain"), ("signed", "signed"), ("gaps", "gaps"), ("gaps_shuf", "gaps_shuffled"), ("pours", "pours")):
                if int(C[f"{kind}.{l}|{mine}"].sum()) != r[theirs]: bad.append(f"{kind}.{l} {mine}")
            if MJ["zg"][f"{kind}.{l}"] != r["w_zero_gaps"]: bad.append(f"{kind}.{l} zero gaps")
    tbad = [k for k, a, b in (("walk", int(T["walk"].sum()), TW_["gaps"]), ("all-band pours", int(T["hall"].sum()), TW_["pours"]),
                               ("plain meetings", int(T["plain_meet"].sum()), TW_["plain"])) if a != b]
    plates_plain = sum(S(k, "plain") for k in KN)
    top_plain = int(T["plain_meet"].sum() + T["plain_div"].sum()); arr_plain = A("plain_scores") + A("plain_values")
    fine_plain = sum(v for k, v in FINE["count"].items() if k.startswith("plain|")); fine_cut = sum(v for k, v in FINE["count"].items() if k.startswith("cut|"))
    log(f"QWEN_FRINGES: the all-base-2 Qwen3-0.6B with way 2 at every plate (a), the pours in signed digits (b), the band read (c) and")
    log(f"the arrival in signed digits (d), built as one machine. 30 refs texts in Qwen's words, {NP} positions; photons a token.")
    log("")
    log("EXACT")
    for f in ("ref", "machine", "top_machine", "control", "orderfree", "top_orderfree", "words_ref", "words_fr", "top_words"):
        for line in open(f"{OUT}/{f}.log"):
            if any(w in line for w in ("against", "EXACT", "checks:", "decoded", "final pass", "race winner = the next", "exact on its own")): log(f"  [{f}] {line.strip()}")
    log(f"  cross-check with Phase VI-a (same light, same plates; plain, signed, gaps, shuffled gaps, pours, zero gaps at all 196 plates): "
        f"{len(bad)} differ{(': ' + ', '.join(bad[:10])) if bad else ''}; the table (walk, all-band pours, plain meetings): {len(tbad)} differ")
    log(f"  the plain basis rebuilt: plates {pt(plates_plain):,.0f}, read {pt(top_plain):,.0f} (fine read's basis {pt(fine_plain):,.0f}), arrival {pt(arr_plain):,.0f} "
        f"(capture: {pt(M_['arrival_scores_total'] + M_['arrival_values_total']):,.0f})")
    # ---- order-free: every count against the main run ----
    O = load_counts(f"{OUT}/orderfree_counts.npz"); n3 = len(next(iter(O.values())))
    od = [k for k in C if k in O and not np.array_equal(C[k][:n3], O[k])]; miss = set(C) ^ set(O)
    TO = np.load(f"{OUT}/top_orderfree.npz"); tdiff = [k for k in ("walk", "writing", "pours", "hall", "bound", "end", "plain_meet", "plain_div", "surv", "cmp", "n", "mu", "top5", "win")
                                                       if not np.array_equal(TO[k], T[k][:n3])]
    log(f"  ORDER-FREE (texts 0-2, {n3} positions): plate and arrival counts differing from the main run: {len(od)} of {len(O)} arrays (keys missing {len(miss)}); "
        f"band read arrays differing: {len(tdiff)} of 14 {tdiff if tdiff else ''}")
    log("")
    # ---- the work, part by part ----
    gaps, pours, pours_sd, walk3, write1 = (sum(S(k, it) for k in KN) for it in ("gaps", "pours", "pours_sd", "walk3", "write1"))
    writing, writing3 = slab_sum("writing"), slab_sum("writing3")
    signed = sum(S(k, "signed") for k in KN)
    plates_a = gaps + pours + writing; plates_ab = gaps + pours_sd + walk3 + writing + writing3; plates_w1 = gaps + pours_sd + write1 + writing
    top_c = int(band_total(dict(T)).sum()); arr_d = A("sd_scores") + A("sd_values") + A("write_q") + A("write_k") + A("write_v")
    P0 = plates_plain + top_plain + arr_plain
    def row(name, plain_, cut_, note=""): log(f"  {name:44s} {pt(plain_):16,.0f} {pt(cut_):16,.0f} {cut_ / plain_ * 100:8.2f}%  {note}")
    log("WORK A TOKEN BY PART (photons; plain = Phase VI-a's basis)")
    log(f"  {'part':44s} {'plain':>16s} {'the machine':>16s} {'share':>9s}")
    row("the 196 plates, (a) way 2", plates_plain, plates_a, f"gap meetings {pt(gaps):,.0f} + pours {pt(pours):,.0f} + the light's writing {pt(writing):,.0f}")
    row("the 196 plates, (a) + (b)", plates_plain, plates_ab, f"gap meetings {pt(gaps):,.0f} + pours in signed digits {pt(pours_sd):,.0f} + the 3x walk {pt(walk3):,.0f} + writing {pt(writing + writing3):,.0f}")
    row("  (side) the plates, signed digits alone", plates_plain, signed, "Phase VI-a's other cut")
    row("  (side) (a) + (b) with W1's writing", plates_plain, plates_w1, f"each new running product written once, 3r = r + 2r: {pt(write1):,.0f}")
    row("the top read, (c) the band read", top_plain, top_c, f"walk {pt(T['walk'].sum()):,.0f} + pours {pt(T['pours'].sum()):,.0f} + bound {pt(T['bound'].sum()):,.0f} + the rest {pt(T['writing'].sum() + T['end'].sum()):,.0f}")
    row("  (side) the fine diagonals, measured (qwen06b)", fine_plain, fine_cut, "not used: the rule chose the band read")
    row("the arrival, (d) signed digits", arr_plain, arr_d, f"scores {pt(A('sd_scores')):,.0f} (plain {pt(A('plain_scores')):,.0f}) + shares x values {pt(A('sd_values')):,.0f} "
        f"(plain {pt(A('plain_values')):,.0f}) + writing {pt(A('write_q') + A('write_k') + A('write_v')):,.0f}")
    log("")
    log("WHICH ADDITION GAVE WHAT (the whole machine, a token; plain 100%)")
    steps = [("plain", P0), ("+ (a) way 2 at every plate", plates_a + top_plain + arr_plain), ("+ (c) the band read", plates_a + top_c + arr_plain),
             ("+ (d) the arrival in signed digits  = (a)+(c)+(d)", plates_a + top_c + arr_d), ("+ (b) the pours in signed digits, 3x walk counted = all four", plates_ab + top_c + arr_d)]
    prev = None
    for name, v in steps:
        log(f"  {name:62s} {pt(v):16,.0f}  {v / P0 * 100:7.2f}%" + ("" if prev is None else f"   ({(v - prev) / P0 * 100:+.2f} points)")); prev = v
    log(f"  (side) all four with W1's writing in place of the 3x walk: {(plates_w1 + top_c + arr_d) / P0 * 100:.2f}%")
    log("")
    log("BY PLATE KIND (28 layers each; shares of that kind's plain; per pair = per lit light value x neuron, Phase VI-a's pairs)")
    log(f"  {'kind':5s} {'plain/pr':>8s} {'walk/pr':>7s} {'pour/pr':>7s} {'sd pour/pr':>10s} {'3x walk/pr':>10s} {'W1/pr':>6s} {'zero gaps':>9s} | {'(a)':>7s} {'(a)+(b)':>8s} {'(b) net/pr':>10s} | {'ctl gaps':>8s} {'ctl 3x':>7s}")
    for k in KN:
        pairs = sum(BK[f"{k}.{l}"]["pairs"] for l in range(L)); pl = S(k, "plain"); N_, out_ = BK[f"{k}.0"]["N"], BK[f"{k}.0"]["out"]
        zgs = sum(MJ["zg"][f"{k}.{l}"] for l in range(L)) / (L * N_ * (out_ - 1))
        wr = sum(C[f"{SL[k]}.{l}|writing"].sum() for l in range(L)) / len(KINDS[SL[k]]); wr3 = sum(C[f"{SL[k]}.{l}|writing3"].sum() for l in range(L)) / len(KINDS[SL[k]])
        a_ = S(k, "gaps") + S(k, "pours") + wr; ab = S(k, "gaps") + S(k, "pours_sd") + S(k, "walk3") + wr + wr3
        log(f"  {k:5s} {pl / pairs:8.2f} {S(k, 'gaps') / pairs:7.3f} {S(k, 'pours') / pairs:7.2f} {S(k, 'pours_sd') / pairs:10.2f} {S(k, 'walk3') / pairs:10.3f} {S(k, 'write1') / pairs:6.2f} "
            f"{zgs * 100:8.2f}% | {a_ / pl * 100:6.2f}% {ab / pl * 100:7.2f}% {(ab - a_) / pairs:+10.2f} | {(S(k, 'gaps_shuf') + S(k, 'pours') + wr) / pl * 100:7.2f}% {(S(k, 'gaps_shuf') + S(k, 'pours_sd') + S(k, 'walk3_shuf') + wr + wr3) / pl * 100:6.2f}%")
    log(f"  all   pours in signed digits / pours {pours_sd / pours:.4f}; the 3x walk / the walk {walk3 / gaps:.4f}; W1's writing / pours {write1 / pours:.4f}")
    log("  ctl: each line's gaps in a random neuron order (banks.py's draw, count only); (a) then (a)+(b) with those gaps.")
    log("")
    log("THE BAND READ AGAINST ITS CONTROL (the light shuffled across its lines, every 4th position: qwen06b/top_read.py's draw)")
    TC = np.load(f"{OUT}/top_control.npz"); idx = TC["sel"]; Tr = {k: T[k][idx] for k in T.files if k != "sel"}
    for name, o in (("real light", Tr), ("shuffled", dict(TC))):
        log(f"  {name:10s} {band_total(o).sum() / band_plain(o).sum() * 100:6.2f}% of its plain read; words still in after bands 2, 3, 4 (median): "
            f"{np.median(o['surv'][:, 2]):.0f}, {np.median(o['surv'][:, 3]):.0f}, {np.median(o['surv'][:, 4]):.0f}; positions with any word set aside after band 1: {int((o['surv'][:, 1] < 151936).sum())}, after band 2: {int((o['surv'][:, 2] < 151936).sum())}")
    log(f"  the fine diagonals on the same lights (qwen06b): real 15.84%, shuffled 15.59%")
    log("")
    log("ESTIMATE (not measured; Phase VI-a's 14B walk per pair, qwen06b/report.log estimate 2): (b) pays at a plate when its 3x walk")
    log("  per pair is below what the signed pours save per pair. Measured here at 0.6B: the 3x walk is about 1.1x the walk; the saving")
    log("  per pair is carried over from 0.6B as measured per kind.")
    est = dict(q=(2.33, 3.54), k=(6.39, 6.39), v=(6.42, 6.42), o=(1.22, 1.30), gate=(0.82, 1.97), up=(0.82, 1.97), down=(1.08, 1.14))
    for k in KN:
        pairs = sum(BK[f"{k}.{l}"]["pairs"] for l in range(L)); sv = (S(k, "pours") - S(k, "pours_sd")) / pairs; rt = S(k, "walk3") / S(k, "gaps")
        lo_, hi_ = est[k]; log(f"    {k:5s} saving {sv:5.2f} a pair; 3x walk at 14B {rt * lo_:5.2f}-{rt * hi_:5.2f} a pair: (b) net {rt * lo_ - sv:+5.2f} to {rt * hi_ - sv:+5.2f} a pair")

if __name__ == "__main__":
    ph = sys.argv[1]; ARG = dict(a.split("=") for a in sys.argv[2:])
    if ph == "ref": phase_ref()
    elif ph == "machine": phase_machine()
    elif ph == "orderfree": phase_machine(order=7, tag="orderfree", ntext=3)
    elif ph == "top": phase_top(ARG["set"])
    elif ph == "control": phase_control()
    elif ph == "words_ref": phase_words_ref()
    elif ph == "words_fr": phase_words_fr()
    elif ph == "report": phase_report()
