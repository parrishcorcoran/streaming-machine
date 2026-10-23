"""QWEN_FRINGES2: QWEN_FRINGES.PY'S EXACT MACHINE (way 2 + the band read + the arrival in signed digits, 37.55% of plain)
WITH TWO CHANGES: THE ARRIVAL'S SIGNED-DIGIT PRODUCTS AS WHOLE NUMBERS, AND EACH SLAB LINED UP AS ONE.

His goal: "keep all the resolution, decrease the compute". Every water level, turn, arrival table, bend and stream add is
qwen_base2.py's own (its source is executed here); only how the products are formed changes, and every pile, stream,
arrival output and read is checked identical to qwen_base2.py's own run. Whole numbers only, in the machine path AND in
the check path: no float array anywhere (qwen_fringes.py's check path used float64 holding whole numbers; here the
check products are int64 sums in qwen_fringes2/kern.c).

THE PICTURE (the session's words, restating qwen_fringes.py's; not his)
  - A plate: lines in, neurons out; on every line the light is one whole number x. Way 2: each line's weights are lined
    up across the neurons ONCE (stored: the order and the gaps between neighbours in rank). Per token the light meets only
    the gaps: walking up the ranks, the running product r_k = r_(k-1) + x gap_k is exactly x times the weight at rank k,
    and it is poured into that neuron's pile. The walk is thin where the gaps are small or zero; the pours are each
    neuron's whole exposure and do not depend on the line-up.

CHANGE 1. THE ARRIVAL'S SIGNED-DIGIT PRODUCTS AS WHOLE-NUMBER PRODUCTS.
  qwen_fringes.py formed q16.k16 and shares x values from signed-digit planes held in float64 and summed them through a
  float64 product (exact, every value a whole number below 2^52, but a float: against rule 9). Here the planes are int8
  (-1 / 0 / +1) and every place-pair sum is an int64 sum (kern.c pp8); the place pairs are then shifted together in
  int64 (2^(a+b), a tensordot of whole numbers). Same planes, same place pairs, same numbers: checked equal to
  qwen_base2's own int products at every head, layer and position. The count (lit digit x lit digit, plus the writing) is
  not touched.
  One more float found in the machine path while building this: qwen_base2.py's gauge takes half of log2(n) through
  math.log2 for n = 1,024 and 128. Both are powers of two, so the value is a whole number (655,360 / 2 and 458,752 / 2);
  it is computed here as ((bit_length(n) - 1) << 16) // 2, checked equal to qwen_base2's own value for both n at load.
  No number changes.

CHANGE 2. EACH SLAB LINED UP AS ONE (Phase VI-a: the walk per pair falls with the neurons on a line: 1,024 -> 6.4,
  2,048 -> 4.5, 3,072 -> 3.5). q, k and v meet the same light, and gate and up meet the same light, so each line's weights
  are lined up across the whole slab at once: q + k + v together (4,096 neurons a line: 2,048 + 1,024 + 1,024) and
  gate + up together (6,144). One walk per line serves the whole slab: every r_k is poured into its own neuron, whichever
  plate it belongs to (each neuron's own power of two is applied after, as qwen_base2 does, to the whole-number pile).
  o and down are one plate a slab and keep their own line-ups.

  THE RULE (fixed here, before any run): at each slab (q+k+v at layer l, gate+up at layer l: 56 slabs), the machine uses
  the slab's one line-up only if, summed over the 613 positions of the 30 refs texts,
        the slab's walk + its pours   <   the per-plate walks + their pours,   on the same light.
  The walk is gap meetings (lit digits of the light x lit digits of each gap, each line's lowest weight included), the
  pours ones(|r|) of every pour. Otherwise that slab keeps its per-plate line-ups, and the order-free and greedy-word runs
  follow the same choice. The light's writing is once a slab either way and is left out of both sides.
  What the rule can see (rule 13.4, stated so the run is not read as more than it is): the pours are fixed by the light
  and the plate alone (r = x w in any order), so they are the same number on both sides; the rule is decided by the walks.
  The machine walks the slab's line-up at every slab (that walk is what is checked exact here); a slab the rule sent back
  would be counted with its per-plate walks, whose exact walk on this very light is qwen_fringes.py's (every pile
  identical there).

UNCHANGED (qwen_fringes.py's, copied): way 2 at o and down; the band read (c) on the word table; the arrival's count;
  every count's definition. The 3x walk (b) is kept only as a SIDE count (its kernel is unchanged): not used by the
  machine, reported to see whether the wider line-ups change its balance.

THE PLAIN BASIS (Phase VI-a's, qwen06b/report.log, 16,397,464,589 a token on the 613 positions): the plates' plain
meetings (11,845,741,380), the plain read (4,490,888,460), the arrival's plain meetings (60,834,749). Not counted on
either side (unchanged parts): water levels, turns, bends, the divides of the plates and the arrival, the stream adds.

CHECKS (the check path is never used by the machine and never counted)
  - every slab's pile at every position = the exact whole-number product (kern.c mm16, int64 sums); every pile digest,
    attention output, stream (both halves of every layer) and the top light = qwen_fringes.py's reference run of
    qwen_base2.py's own steps (qwen_fringes/ref.json, read only, not rerun: nothing it depends on is changed here) AND
    capture's record of qwen_base2.py's own run (qwen06b/light); the arrival's products = qwen_base2's int products.
  - the band read at all 613 positions: top 5 and race winner = the plain read's, the reference's and capture's; score
    rows' sha256; and every band-read array equal to qwen_fringes.py's (same light, same read).
  - THE 16 GREEDY WORDS: the machine on qwen_fringes/words_ref.json's 30 greedy texts (qwen_base2's own greedy words,
    30/30 = check30.log's machine words): every pile, stream and arrival output = the reference path's, and the band
    read's race winner at the 16 read places of every text = the next greedy word (480/480).
  - ORDER-FREE (texts 0-2): every slab walked DOWN from the top weight with its lines in a random order (both running
    products must close at 0), the arrival's lines in random orders, the word table walked down in a random line order
    with the words relabelled: every pile, stream, count, top 5 and race winner identical to the main run.
  - CROSS-CHECKS with qwen_fringes.py (same light, same plates; code checks, not findings): every plate's plain, signed
    and per-plate gap counts; the slab's pours (= the sum of its plates' pours, by construction); the arrival's counts;
    the light's writing; o's and down's every count.
  - CONTROL (i), count only: each slab line's gaps with the slab's neurons in a random order: the saving is the
    line-up's. (The band read's control (ii) is qwen_fringes.py's: the read is unchanged and so is its light.)
  - qwen_fringes2/check.py: a second code path (same session, NOT an independent rebuild): a sample of slab and plate
    walk counts recounted from scratch, the rule recomputed, and the arrival tested for using its own products.

RULE 13, CHECKED BEFORE THE RUN
  13.1 the light's 1s are not the measure: the work is photons met and poured; exactness is on the piles, streams and
       top 5. 13.2 the base is qwen_base2.py's own machine and light, every part as it is (the gauge's constant is the
       same whole number, written without a float). 13.3 no agreement statistic. 13.4 the pours are fixed by the light
       and the plate alone and the same on both sides of the rule; the run measures the walk, which depends on the
       line-up, and control (i) tests that. Change 1 fixes no count; it changes how the same numbers are formed, and the
       run checks only that they are the same numbers. 13.5 the task's prediction is the task's; the session's is
       labelled; nothing here is his sentence. 13.6 the band read is carried to band 7 at every position.

PREDICTIONS (written before any run of this file)
  The task's: (1) changes no number and no count; (2) the walk at q/k/v falls from 4.5-6.4 to about 3.0 a pair and at
  gate/up from 3.5 to about 2.5, the pours unchanged, taking the whole to about 35-36%.
  The session's (from Phase VI-a's records; nothing of this file run yet):
  - (1): every product, pile, stream, attention output, top light and read identical; every count identical to
    qwen_fringes.py's (the arrival 34,208,215 a token, 56.23% of its plain). The gauge's constant: no number changes.
  - (2), by the relation Phase VI-a measured (walk per pair = the light's signed digits per lit value x gap lit digits
    per weight, within 0.4% at every kind) with the gap lit digits carried along the word table's own width law
    (qwen06b/table_walk.json: 1,024 1.47, 2,048 0.99, 3,072 0.75, 5,120 0.51; log-log between them: 4,096 about 0.61,
    6,144 about 0.44) and scaled from each kind's own measured value:
      q+k+v (4,096): gap lit about 0.69-0.71 a weight -> the walk about 2.7 a pair if q's, k's and v's weights on a line
        spread alike, up to about 3.0 if they do not (Phase VI-a's lines pooled across layers kept 0.91-0.95 against
        the law's 0.61 at 5,120: lines from plates that spread differently keep part of their gaps);
      gate+up (6,144): gap lit about 0.52 a weight -> the walk about 2.0 a pair, up to about 2.5.
    Zero gaps about 62-65% (q+k+v) and 72-74% (gate+up), against 30-47% and 58.6% per plate.
    THE RULE: the slab below its plates at all 56 slabs, and on every line (0 of 28,672 lines at q+k+v and at gate+up
    with the slab's gap digits above its plates' summed), so the choice holds on any light, not only this one.
  - The whole: the walk at q+k+v 636.1M -> 316-352M a token (117.18M pairs), at gate+up 613.8M -> 352-439M (175.78M
    pairs); o, down, the pours (4,092.5M), the writing, the read and the arrival unchanged: 34.0-34.8% of plain
    (2.7 and 2.0 give 34.00%; the task's own 3.0 and 2.5 give 34.75%), a little below the task's 35-36%.
  - Control (i): the slab's gaps in a random neuron order: (a) at 94-96% of plain at q+k+v and gate+up, as per plate.
  - (b), side: the 3x walk about 1.14 x the walk; at q+k+v 3.1-3.4 a pair against the signed pours' saving of about
    2.74 a pair, so (b) still adds work there; at gate+up 2.3-2.85 against about 2.79: about even, at most a small gain.

RESULTS (2026-10-05; qwen_fringes2/report.log and check.log; a token on the 613 positions; plain 16,397,464,589)
  - EXACT, held everywhere: every pile (3,360 digests), attention output (840), stream (1,680) and the top light (30) equal
    the reference run and capture's record (piles); every slab walk equal to the whole-number product; every walk down
    closed at 0; the arrival's products from int8 planes equal the int products at every head, layer and position. The
    band read: top 5 and race winner 613/613, score rows' sha256 613/613 against both; every one of its 16 arrays equal
    to qwen_fringes.py's. Greedy words: every pile, stream and arrival output equal on the 30 greedy texts, and the race
    winner = the next greedy word 480/480. Order-free (texts 0-2): 0 of 2,128 count arrays and 0 of 14 band-read arrays
    differ. Cross-checks with qwen_fringes.py: 0 differ (per-plate plain, signed, own gaps, zero gaps; pours; writing; o
    and down; the arrival's 196 count arrays).
  - CHANGE 1 (the task's (1): held): no number and no count changed (arrival 34,208,215 = 56.23% of its plain, as before).
    check.py: the place-pair weights doubled change the arrival output at 28 of 28 layers (the machine uses its own
    sums); a trace of every return in 37 machine-path functions (183,153 returns, one text) found no float but the known
    clock reading in forward() (its positive control). The band read was not traced; its code holds no float (its one
    non-integer is an exact Fraction).
  - CHANGE 2: the walk per pair at q+k+v 2.820 (task: about 3.0; session: 2.7-3.0, held) and at gate+up 2.069 (task: about
    2.5; session: 2.0-2.5, held at its low end). The relation (light's signed digits x gap lit digits per weight) gives
    2.828 and 2.076. Gap lit digits a weight 0.7393 and 0.5358 (session's law carried: 0.69-0.71 and about 0.52: a little
    above, so lining q, k and v up together kept a little of their gaps). Zero gaps 64.48% and 73.96% (session: 62-65% and
    72-74%, held). Pours unchanged (by construction: 13.4).
  - THE RULE: the slab's line-up at 56 of 56 slabs, (walk + pours) at 0.821-0.843 (q+k+v) and 0.888-0.907 (gate+up) of the
    plates'; the slab's walk below its plates' at every position and layer (0 of 17,164 each); weights only: 0 of 28,672
    lines at each with the slab's gap digits above its plates' summed (54 equal at gate+up), so it holds on any light.
  - THE WHOLE: 5,602,023,037 = 34.16% of plain (qwen_fringes.py 37.55%: -3.39 points, 90.97% of it). The task's 35-36%:
    missed low (the walks fell further than its 3.0 and 2.5); the session's 34.0-34.8%: held. By part: the walk at
    q+k+v 636.1M -> 330.4M (51.94%), at gate+up 613.8M -> 363.7M (59.26%); o and down 205.95M and 271.62M, the pours
    4,092.5M, the writing 0.97M, the band read 302.6M, the arrival 34.2M, all unchanged. The plates 44.45% of their plain
    (49.14% before). The pours are now 73% of the machine and are untouched by any line-up.
  - CONTROL (i): the slab's gaps in a random neuron order: (a) at 95.71% (q+k+v) and 94.78% (gate+up) of plain, against
    42.98% and 39.74% real (session: 94-96%, held): the saving is the line-up's.
  - SIDE (b) (not used): at the wider line-ups the 3x walk is 3.210 a pair at q+k+v against a saving of 2.741 (+0.470, adds
    work), and 2.352 at gate+up against 2.795 (-0.443 a pair, -77.6M a token: it now PAYS there; session: about even, at
    most a small gain). (b) only at gate+up would give 33.69%: shown only, no rule for it was fixed before this run.
  - NOT YET: an independent rebuild by a separate checker (check.py is a second code path of the same session).
  - Housekeeping: every run under 2 GB (largest 1,836 MB, the order-free band read); 29 MB written.

RUNS (one python process at a time, foreground, each under 3 GB; outputs in qwen_fringes2/):
    python3 qwen_fringes2.py machine             the machine on the 30 refs texts (613 positions): every slab, counted
    python3 qwen_fringes2.py top set=machine     the band read at all 613 positions
    python3 qwen_fringes2.py orderfree           texts 0-2, every order random, every walk down
    python3 qwen_fringes2.py top set=orderfree   the band read there, the words relabelled, walks down
    python3 qwen_fringes2.py words               the machine on qwen_fringes/words_ref.json's 30 greedy texts
    python3 qwen_fringes2.py top set=words       the band read at the 16 read places of every text
    python3 qwen_fringes2.py report
    python3 qwen_fringes2/check.py               the second code path
"""
import os, sys, json, time, math, ctypes, subprocess, resource, hashlib, zipfile, glob, numpy as np
from fractions import Fraction
HERE = "/Users/abundancemachine/streaming-machine"; QD = f"{HERE}/qwen"; NPZ = f"{QD}/qwen_base2.npz"
OUT = f"{HERE}/qwen_fringes2"; OUT1 = f"{HERE}/qwen_fringes"; CAP = f"{HERE}/qwen06b"     # qwen_fringes/ and qwen06b/: read only
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

# ---------------- the kernels (qwen_fringes2/kern.c), whole numbers only ----------------
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{OUT}/kern.c", f"{OUT}/kern.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); P_, I_, i_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.walk2.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, P_, i_, I_, P_, P_, P_, P_, P_, P_]; _lib.walk2.restype = None
        _lib.band2.argtypes = [I_, I_, I_, P_, P_, P_, P_, P_, P_, i_, I_, P_, P_, P_, P_]; _lib.band2.restype = None
        _lib.pp8.argtypes = [I_, I_, I_, P_, I_, P_, I_, P_]; _lib.pp8.restype = None
        _lib.mm16.argtypes = [I_, I_, I_, P_, P_, P_]; _lib.mm16.restype = None
    return _lib
def pp8(A, B):
    """THE MACHINE: C[i, j] = sum over k of A[i, k] B[j, k]; A, B planes of -1 / 0 / +1 in int8 (rows may be slices of a
    wider plane, read in place), every sum an int64 sum"""
    assert A.dtype == np.int8 and B.dtype == np.int8 and A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    assert A.shape[1] >= 1 and (A.shape[1] == 1 or (A.strides[1] == 1 and B.strides[1] == 1))   # a single column: any stride
    M, K = A.shape; N = B.shape[0]; C = np.empty((M, N), np.int64)
    lib().pp8(M, N, K, A.ctypes.data, A.strides[0] if M > 1 else K, B.ctypes.data, B.strides[0] if N > 1 else K, C.ctypes.data)
    return C
def exact_product(x16, w16):
    """CHECK PATH: the exact product of the light and a plate as stored (int16), int64 sums (kern.c mm16)"""
    X = np.ascontiguousarray(np.asarray(x16, np.int64)); W = np.ascontiguousarray(w16); assert W.dtype == np.int16
    P, L_ = X.shape; assert W.shape[0] == L_ and np.abs(X).max(initial=0) <= 1 << 15
    out = np.empty((P, W.shape[1]), np.int64); lib().mm16(P, L_, W.shape[1], X.ctypes.data, W.ctypes.data, out.ctypes.data)
    return out

# ---------------- qwen_base2.py's own steps; only the products are hooked ----------------
Mz = np.load(NPZ)
class LazyW(dict):                                                              # W[name] -> (None, the plate's powers)
    def __getitem__(self, name):
        if name not in self.keys(): dict.__setitem__(self, name, (None, np.asarray(Mz[name + ".e"])))
        return dict.__getitem__(self, name)
ST = dict(order=None, rows=None, row0=0, text=0, cnt=None, dig=None, light_cmp=False, chk={}, arr_rng=None, vcache={})
def chk(k, v): ST["chk"][k] = ST["chk"].get(k, 0) + int(v)
def cadd(k, rows, v):
    C = ST["cnt"]
    if k not in C: C[k] = np.zeros(ST["P"], np.int64)
    C[k][rows] += v

SLABS = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
         "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}
SEED = dict(qkv=7, o=3, gu=8, down=6)                                           # o, down: banks.py's / qwen_fringes.py's draw
class Lineup:
    """way 2's record of one line-up (laid once): each line's weights in order of size across ALL the neurons given (a
    whole slab here; ties in neuron order): NR the neuron at each rank, G the gaps (G[:, 0] unused), LOW, TOP; per line the
    gaps' lit digits GN (the lowest weight included) and ZG the zero gaps; the CONTROL's GNS: the gaps' lit digits with
    each line's neurons in a random order (count only)."""
    def __init__(self, w, seed):
        w = np.asarray(w).astype(np.int64); L, N = w.shape; self.L, self.N = L, N
        o = np.argsort(w, axis=1, kind="stable"); ws = np.take_along_axis(w, o, 1); g = np.diff(ws, axis=1)
        assert (g >= 0).all() and g.max(initial=0) < 1 << 16 and N < 1 << 16
        back = np.empty_like(w); np.put_along_axis(back, o, np.concatenate([ws[:, :1], ws[:, :1] + np.cumsum(g, 1)], 1), 1)
        assert np.array_equal(back, w), "the record does not give the slab back"
        self.NR = np.ascontiguousarray(o.astype(np.uint16)); self.G = np.zeros((L, N), np.uint16); self.G[:, 1:] = g
        self.LOW = np.ascontiguousarray(ws[:, 0]); self.TOP = np.ascontiguousarray(ws[:, -1])
        self.GN = naf(ws[:, 0]) + naf(g).sum(1); self.ZG = int((g == 0).sum())
        rng = np.random.default_rng(seed)
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
def plate_counts(w):
    """count only: one plate's own line-up (each line across that plate's neurons), its plain 1s and lit digits per line"""
    w = np.asarray(w).astype(np.int64); ws = np.sort(w, axis=1); g = np.diff(ws, axis=1)
    return dict(GN=naf(ws[:, 0]) + naf(g).sum(1), ZG=int((g == 0).sum()), WROW=ones(w).sum(1), WNROW=naf(w).sum(1))

def fr_slab(x16, slab, l):
    """change 2: the slab lined up as one and walked once (and its 3x walk, side); the pile checked exact; the per-plate
    line-ups counted beside it on the same light (count only) for the rule"""
    W16 = np.asarray(Mz[f"{slab}.{l}.w"]); assert W16.dtype == np.int16
    P = len(x16); rows = slice(ST["row0"], ST["row0"] + P); x3 = 3 * x16
    nx, ox, n3 = naf(x16), ones(x16), naf(x3)
    cadd(f"{slab}.{l}|writing", rows, ox.sum(1)); cadd(f"{slab}.{l}|writing3", rows, ones(x3).sum(1))
    lu = Lineup(W16, [2026, l, SEED[slab]])
    lo = None if ST["order"] is None else np.random.default_rng([ST["order"], l, SEED[slab]]).permutation(lu.L)
    pile, (p1, ps, w1, opn, bad3) = lu.walk(x16, line_order=lo, down=ST["order"] is not None)
    chk("slab: walk vs the exact product (rows differing)", (pile != exact_product(x16, W16)).any(1).sum())
    chk("slab: a walk down not closed at 0", opn.sum()); chk("slab: the 3x walk not 3 r (ranks)", bad3.sum())
    k = f"{slab}.{l}|"                                                          # slab items: walk*, pours*, write1, pairs, lit, writing*
    for item, v in (("walk", nx @ lu.GN), ("walk_shuf", nx @ lu.GNS), ("walk3", n3 @ lu.GN), ("walk3_shuf", n3 @ lu.GNS),
                    ("pours", p1), ("pours_sd", ps), ("write1", w1), ("pairs", (x16 != 0).sum(1) * lu.N), ("lit", nx.sum(1))):
        cadd(k + item, rows, v)
    gsum = np.zeros(lu.L, np.int64)
    for kind, c0, c1 in SLABS[slab]:                                            # plate items: plain, signed, gaps (own line-up)
        pc_ = plate_counts(W16[:, c0:c1]); gsum += pc_["GN"]
        for item, v in (("plain", ox @ pc_["WROW"]), ("signed", nx @ pc_["WNROW"]), ("gaps", nx @ pc_["GN"])):
            cadd(f"{kind}.{l}|{item}", rows, v)
        ST["zg"][f"{kind}.{l}"] = pc_["ZG"]; ST["wgap"][f"{kind}.{l}"] = int(pc_["GN"].sum())
    cadd(k + "walk_plates", rows, nx @ gsum)
    ST["zg"][f"slab {slab}.{l}"] = lu.ZG; ST["wgap"][f"slab {slab}.{l}"] = int(lu.GN.sum()); ST["wgap_shuf"][f"slab {slab}.{l}"] = int(lu.GNS.sum())
    ST["lines"][f"{slab}.{l}"] = [int((lu.GN > gsum).sum()), int((lu.GN == gsum).sum()), int(lu.L)]
    del lu
    return pile

CAPL = {}
def PRODUCT(x16, ex, name):
    x16 = np.asarray(x16, np.int64); assert np.abs(x16).max(initial=0) <= 1 << 15
    assert name != "head", "the machine reads the top by the band pour"
    slab, l = name.split("."); l = int(l)
    if ST["light_cmp"]:                                                        # capture's record of qwen_base2.py's own light
        if slab not in CAPL: CAPL[slab] = np.load(f"{CAP}/light/{slab}_light.npy", mmap_mode="r")
        chk("light at the slabs vs capture's record (rows differing)", (np.asarray(CAPL[slab][l][:len(x16)]).astype(np.int64) != x16).any(1).sum())
    pile = fr_slab(x16, slab, l)
    for t, (a, b) in enumerate(ST["rows"]): ST["dig"][f"pile|{name}|{t}"] = digest(pile[a:b])
    return pile

# ---------------- change 1: the arrival's products from int8 signed-digit planes, int64 sums ----------------
def sd_places(X):
    up, dn = signed_digits(X); return max(1, int(np.bitwise_or.reduce((up | dn).ravel(), initial=0)).bit_length())
def sd_planes8(X, n):                                                           # X's signed digits as n planes of -1 / 0 / +1, int8
    up, dn = signed_digits(X); assert not ((up | dn) >> n).any()
    return np.stack([((up >> a) & 1) - ((dn >> a) & 1) for a in range(n)]).astype(np.int8)
def TW(na, nb): return np.left_shift(np.int64(1), np.arange(na)[:, None] + np.arange(nb)[None, :])
def QK(q16, k16, h, l):
    """q16 . k16 (every pair of positions of this text, one head), formed from int8 signed-digit planes place pair by
    place pair (int64 sums), shifted together in int64; counted; checked equal to qwen_base2's own int product"""
    plain = q16 @ k16.T                                                         # CHECK PATH: qwen_base2's own int64 product
    P = len(q16); rows = slice(ST["row0"], ST["row0"] + P)
    oq, ok_, nq, nk = ones(q16), ones(k16), naf(q16), naf(k16)
    cadd(f"arr.{l}|plain_scores", rows, np.tril(oq @ ok_.T).sum(1)); cadd(f"arr.{l}|sd_scores", rows, np.tril(nq @ nk.T).sum(1))
    cadd(f"arr.{l}|write_q", rows, oq.sum(1))
    if h % 2 == 0: cadd(f"arr.{l}|write_k", rows, ok_.sum(1))
    qq, kk = q16, k16
    if ST["arr_rng"] is not None: perm = ST["arr_rng"].permutation(q16.shape[1]); qq, kk = q16[:, perm], k16[:, perm]
    na, nb = sd_places(qq), sd_places(kk); assert na + nb + 8 < 62
    C = pp8(sd_planes8(qq, na).reshape(na * P, -1), sd_planes8(kk, nb).reshape(nb * P, -1)).reshape(na, P, nb, P)
    S = np.tensordot(C, TW(na, nb), axes=([0, 2], [0, 1])); assert S.dtype == np.int64
    chk("arrival: q.k from int8 signed-digit planes vs the int product", (S != plain).sum())
    return S
def AV(a, v, p, h, l):
    """shares x values at position p (one head), formed from int8 signed-digit planes (int64 sums); counted; checked"""
    plain = a[None, :] @ v[:p + 1]                                              # CHECK PATH: qwen_base2's own int64 product
    key = (l, h // 2, ST["text"])
    if key not in ST["vcache"]:
        ST["vcache"].clear(); nbv = sd_places(v); VP = sd_planes8(v, nbv)         # [place, position, value line]
        VT = np.ascontiguousarray(VP.transpose(0, 2, 1)).reshape(nbv * v.shape[1], len(v))   # row (place, value line), column position
        ST["vcache"][key] = (VT, nbv, ones(v).sum(1), naf(v).sum(1))
    VT, nbv, ovs, nvs = ST["vcache"][key]
    r = ST["row0"] + p
    cadd(f"arr.{l}|plain_values", r, int(ones(a) @ ovs[:p + 1])); cadd(f"arr.{l}|sd_values", r, int(naf(a) @ nvs[:p + 1]))
    if h % 2 == 0: cadd(f"arr.{l}|write_v", r, int(ovs[p]))
    aa, B = a, VT[:, :p + 1]
    if ST["arr_rng"] is not None: perm = ST["arr_rng"].permutation(p + 1); aa, B = a[perm], np.ascontiguousarray(B[:, perm])
    na = sd_places(aa); assert na + nbv + int(p + 1).bit_length() < 62
    C = pp8(sd_planes8(aa, na), B).reshape(na, nbv, -1)
    S = np.tensordot(TW(na, nbv), C, axes=([0, 1], [0, 1]))[None, :]; assert S.dtype == np.int64
    chk("arrival: shares x values from int8 signed-digit planes vs the int product", (S != plain).sum())
    return S

def half_log2(n):                                                               # half of log2(n) at 2^-16, n a power of two
    n = int(n); assert n > 0 and n & (n - 1) == 0; return ((n.bit_length() - 1) << 16) // 2
for _n in (1024, 128): assert half_log2(_n) == int(round(math.log2(_n) * 65536)) // 2   # CHECK PATH, once: qwen_base2's own value

_src = open(f"{QD}/qwen_base2.py").read(); _src = _src[:_src.index("from transformers import AutoTokenizer")]
for _a, _b in [('M = np.load("qwen_base2.npz")', 'M = Mz'),
               ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = LazyW()'),
               ("w, we = W[name]; x16, ex = to16(R); pile = x16 @ w", "w, we = W[name]; x16, ex = to16(R); pile = PRODUCT(x16, ex, name)"),
               ("w, we = W[name]; x16, ex = to16(X); return rshift(x16 @ w,", "w, we = W[name]; x16, ex = to16(X); return rshift(PRODUCT(x16, ex, name),"),
               ("sc = rshift(q16 @ k16.T,", "sc = rshift(QK(q16, k16, h, l),"),
               ("out[p, h * HD:(h + 1) * HD] = power(a[None, :] @ v[:p + 1],", "out[p, h * HD:(h + 1) * HD] = power(AV(a, v, p, h, l),"),
               ("half = int(round(math.log2(n) * 65536)) // 2", "half = half_log2(n)")]:
    assert _src.count(_a) == 1, _a; _src = _src.replace(_a, _b)
B = dict(Mz=Mz, LazyW=LazyW, PRODUCT=PRODUCT, QK=QK, AV=AV, half_log2=half_log2)
_argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
assert (B["GAUGE"], B["TURN"], B["BEND"], B["ARRIVAL"]) == ("int",) * 4
to16, gauge, gauged_slab, plain_slab, attention, bend, race, power = (B[k] for k in ("to16", "gauge", "gauged_slab", "plain_slab", "attention", "bend", "race", "power"))
L, D, ES, EXP2 = B["L"], B["D"], B["ES"], B["EXP2"]
EMB = member("embed")

def top5(S, labels=None):                                                       # by score, ties by word number (capture.py's rule)
    S = np.atleast_2d(S); lab = np.arange(S.shape[1]) if labels is None else labels
    return np.array([lab[np.lexsort((lab, -s))[:5]] for s in S])

def forward(texts, order=None, light_cmp=False, log=print):
    """qwen_base2.py's own forward, every text layer by layer together (each text's arrival reads only its own
    positions). Records the digest of every pile, attention output and stream (both halves of every layer) per text."""
    t0 = time.time(); lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    rows = [(int(off[i]), int(off[i + 1])) for i in range(len(texts))]; P = int(off[-1])
    ST.update(order=order, rows=rows, row0=0, text=0, cnt={}, dig={}, light_cmp=light_cmp, P=P, zg={}, wgap={}, wgap_shuf={}, lines={},
              vcache={}, chk={}, arr_rng=None if order is None else np.random.default_rng([order, 99]))
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
        if l % 7 == 6 or l == 0: log(f"    layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    x16, ex = to16(R); LS = gauge(x16, ex, D)
    for t, (a, b) in enumerate(rows): ST["dig"][f"top light|{t}"] = digest(np.concatenate([x16[a:b], ex[a:b, None], np.asarray(LS)[a:b, None]], 1))
    return dict(R=R, x16=x16, ex=np.asarray(ex, np.int64), LS=np.asarray(LS, np.int64), rows=rows, dig=dict(ST["dig"]), cnt=dict(ST["cnt"]))

# ---------------- (c) the band read: qwen_fringes.py's, its check product now whole numbers ----------------
NB = 8; PB = [26, 22, 18, 14, 10, 6, 2, 0]; NT = NB - 1
class Table:
    """the word table lined up once (way 2's record): per line the word at each rank (NR, uint32), the gaps (G, uint16,
    G[:, 0] unused), LOW, TOP, the gaps' lit digits GN; the plain read's 1s per line PPH; each word's shift KW.
    word_perm: the words relabelled (ORDER-FREE), the line-up laid again on the relabelled table."""
    def __init__(self, word_perm=None, log=print):
        t0 = time.time()
        T = np.ascontiguousarray(member("head.w")); HE = np.asarray(Mz["head.e"]).astype(np.int64); assert T.dtype == np.int16
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
    def pile(self, X): return exact_product(X, self.T)                          # CHECK PATH: the exact product, every word, int64 sums

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
def diff_digests(dig, ref):
    dd = {}
    for k, v in dig.items():
        part = k.split("|")[0]; dd.setdefault(part, [0, 0]); dd[part][0] += 1; dd[part][1] += v != ref[k]
    return ", ".join(f"{p} {d} of {n} differ" for p, (n, d) in dd.items())

def phase_machine(order=None, tag="machine", ntext=None):
    log = opener(tag); t0 = time.time(); M_ = META(); texts = M_["texts"][:ntext]; RF = json.load(open(f"{OUT1}/ref.json"))
    log(f"THE MACHINE ({tag}): every slab lined up as one and walked once, the arrival from int8 signed-digit planes; {len(texts)} texts, "
        f"{sum(map(len, texts))} positions" + ("" if order is None else f"; ORDER-FREE (seed {order}): every line order random, every walk down"))
    res = forward(texts, order=order, light_cmp=True, log=log)
    log("  against the reference run of qwen_base2.py's own steps (qwen_fringes/ref.json): " + diff_digests(res["dig"], RF["dig"]))
    pd = sum(res["dig"][f"pile|{n}|{t}"] != M_["dig"][n][t] for n in M_["dig"] for t in range(len(texts)))
    log(f"  piles against capture's record of qwen_base2.py's own run: {pd} of {len(M_['dig']) * len(texts)} differ")
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in ST["chk"].items()))
    save_counts(f"{OUT}/{tag}_counts.npz", res["cnt"])
    json.dump(dict(zg=ST["zg"], wgap=ST["wgap"], wgap_shuf=ST["wgap_shuf"], lines=ST["lines"], chk=ST["chk"], dig=res["dig"]), open(f"{OUT}/{tag}.json", "w"))
    np.savez(f"{OUT}/{tag}_top_light.npz", x16=res["x16"].astype(np.int32), ex=res["ex"], LS=res["LS"], rows=np.array(res["rows"]))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_words():
    log = opener("words"); t0 = time.time(); W = json.load(open(f"{OUT1}/words_ref.json"))
    log(f"THE MACHINE on the 30 greedy texts of qwen_fringes/words_ref.json (prompt + the first 15 greedy words): {sum(map(len, W['texts']))} positions")
    res = forward(W["texts"], log=log)
    log("  against the reference path on the same texts: " + diff_digests(res["dig"], W["dig"]))
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in ST["chk"].items()))
    save_counts(f"{OUT}/words_counts.npz", res["cnt"])
    np.savez(f"{OUT}/words_top_light.npz", x16=res["x16"].astype(np.int32), ex=res["ex"], LS=res["LS"], rows=np.array(res["rows"]))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_top(SET):
    log = opener(f"top_{SET}"); t0 = time.time()
    Z = np.load(f"{OUT}/{SET}_top_light.npz"); X, EX, LS = Z["x16"].astype(np.int64), Z["ex"], Z["LS"]
    sel = np.arange(len(X)); wperm = None; line_rng = None; down = False
    if SET == "words":
        W = json.load(open(f"{OUT1}/words_ref.json")); sel = np.array(W["read_rows"]); REF = dict(top5=W["top5"], win=W["win"], sha=W["sha"])
    else:
        RF = json.load(open(f"{OUT1}/ref.json"))["read"]; REF = {k: v[:len(X)] for k, v in RF.items()}
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
    old = np.load(f"{OUT1}/top_{SET}.npz"); keys = [k for k in R if k not in ("bad", "sha")]
    log(f"  every band-read array against qwen_fringes.py's on the same set: {sum(not np.array_equal(R[k], old[k]) for k in keys)} of {len(keys)} differ")
    log("  checks: " + ", ".join(f"{k} {v}" for k, v in sorted(R["bad"].items())))
    tot, pl = band_total(R), band_plain(R)
    log(f"WORK A POSITION (mean over {n}): plain read {pl.sum() / n:,.0f}; band read {tot.sum() / n:,.0f} = {tot.sum() / pl.sum() * 100:.2f}%")
    np.savez(f"{OUT}/top_{SET}.npz", sel=sel, **{k: v for k, v in R.items() if k not in ("bad", "sha")})
    json.dump(dict(bad=R["bad"], sha=R["sha"], zero_steps=tab.zero_steps), open(f"{OUT}/top_{SET}.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")

def phase_report():
    """reads only qwen_fringes2/'s own results, qwen_fringes/'s (the 37.55% machine) and Phase VI-a's; writes report.log"""
    log = opener("report"); M_ = META(); NP = M_["npos"]
    C = load_counts(f"{OUT}/machine_counts.npz"); T = np.load(f"{OUT}/top_machine.npz"); MJ = json.load(open(f"{OUT}/machine.json"))
    C1 = load_counts(f"{OUT1}/machine_counts.npz"); T1 = np.load(f"{OUT1}/top_machine.npz"); MJ1 = json.load(open(f"{OUT1}/machine.json"))
    KN = ["q", "k", "v", "o", "gate", "up", "down"]; SN = ["qkv", "o", "gu", "down"]
    KS = dict(qkv=["q", "k", "v"], o=["o"], gu=["gate", "up"], down=["down"]); NN = dict(q=2048, k=1024, v=1024, o=1024, gate=3072, up=3072, down=1024)
    NS = dict(qkv=4096, o=1024, gu=6144, down=1024)
    def S(key): return int(C[key].sum())
    def sl(item, slabs=SN): return sum(S(f"{s}.{l}|{item}") for s in slabs for l in range(L))
    def kd(item, kinds=KN, src=None): src = C if src is None else src; return sum(int(src[f"{k}.{l}|{item}"].sum()) for k in kinds for l in range(L))
    def A(item, src=None): src = C if src is None else src; return sum(int(src[f"arr.{l}|{item}"].sum()) for l in range(L))
    def pt(v): return v / NP
    log("QWEN_FRINGES2: qwen_fringes.py's machine with the arrival's products from int8 signed-digit planes (int64 sums) and every")
    log(f"slab lined up as one (q+k+v 4,096 neurons a line, gate+up 6,144). 30 refs texts in Qwen's words, {NP} positions; photons a token.")
    log("")
    log("EXACT")
    for f in ("machine", "top_machine", "orderfree", "top_orderfree", "words", "top_words"):
        for line in open(f"{OUT}/{f}.log"):
            if any(w in line for w in ("against", "EXACT", "checks:", "race winner = the next")): log(f"  [{f}] {line.strip()}")
    # ---- cross-checks with qwen_fringes.py (same light) ----
    bad = []
    for k in KN:
        for l in range(L):
            for item in ("plain", "signed", "gaps"):
                if not np.array_equal(C[f"{k}.{l}|{item}"], C1[f"{k}.{l}|{item}"]): bad.append(f"{k}.{l} {item}")
            if MJ["zg"][f"{k}.{l}"] != MJ1["zg"][f"{k}.{l}"]: bad.append(f"{k}.{l} zero gaps")
    for s in SN:
        for l in range(L):
            for item in ("pours", "pours_sd"):
                if not np.array_equal(C[f"{s}.{l}|{item}"], sum(C1[f"{k}.{l}|{item}"] for k in KS[s])): bad.append(f"{s}.{l} {item}")
            for item in ("writing", "writing3"):
                if not np.array_equal(C[f"{s}.{l}|{item}"], C1[f"{s}.{l}|{item}"]): bad.append(f"{s}.{l} {item}")
    for s in ("o", "down"):
        for l in range(L):
            for mine, theirs in (("walk", "gaps"), ("walk_plates", "gaps"), ("walk_shuf", "gaps_shuf"), ("walk3", "walk3"), ("walk3_shuf", "walk3_shuf"), ("write1", "write1")):
                if not np.array_equal(C[f"{s}.{l}|{mine}"], C1[f"{s}.{l}|{theirs}"]): bad.append(f"{s}.{l} {mine} (own)")
    abad = [k for k in C1 if k.startswith("arr.") and not np.array_equal(C[k], C1[k])]
    tbad = [k for k in T1.files if not np.array_equal(T[k], T1[k])]
    log(f"  cross-check with qwen_fringes.py (same light; code checks): per-plate plain, signed, own-line-up gaps and zero gaps, the slabs'")
    log(f"    pours and signed pours (= their plates' summed), the writing, every count at o and down: {len(bad)} differ{(': ' + ', '.join(bad[:8])) if bad else ''};")
    log(f"    the arrival's {len([k for k in C1 if k.startswith('arr.')])} count arrays: {len(abad)} differ; the band read's {len(T1.files)} arrays: {len(tbad)} differ")
    O = load_counts(f"{OUT}/orderfree_counts.npz"); n3 = len(next(iter(O.values())))
    od = [k for k in C if k in O and not np.array_equal(C[k][:n3], O[k])]; miss = set(C) ^ set(O)
    TO = np.load(f"{OUT}/top_orderfree.npz"); tdiff = [k for k in ("walk", "writing", "pours", "hall", "bound", "end", "plain_meet", "plain_div", "surv", "cmp", "n", "mu", "top5", "win")
                                                       if not np.array_equal(TO[k], T[k][:n3])]
    log(f"  ORDER-FREE (texts 0-2, {n3} positions): count arrays differing from the main run: {len(od)} of {len(O)} (keys missing {len(miss)}); "
        f"band read arrays differing: {len(tdiff)} of 14 {tdiff if tdiff else ''}")
    log("")
    # ---- the rule ----
    log("THE RULE (fixed before the run): a slab uses its one line-up only if its walk + pours < its plates' walks + pours, summed over")
    log("  the 613 positions on the same light. The pours are the same number on both sides (checked above), so the walks decide.")
    choose = {}
    for s in ("qkv", "gu"):
        marg = []
        for l in range(L):
            a_, b_ = S(f"{s}.{l}|walk") + S(f"{s}.{l}|pours"), S(f"{s}.{l}|walk_plates") + S(f"{s}.{l}|pours")
            choose[f"{s}.{l}"] = a_ < b_; marg.append(a_ / b_)
        lw, le, ln = (sum(MJ["lines"][f"{s}.{l}"][i] for l in range(L)) for i in range(3))
        pos_worse = int(sum((C[f"{s}.{l}|walk"] >= C[f"{s}.{l}|walk_plates"]).sum() for l in range(L)))
        log(f"  {s:4s}: the slab's line-up chosen at {sum(choose[f'{s}.{l}'] for l in range(L))} of {L} layers; (walk + pours) slab / plates "
            f"{min(marg):.4f}-{max(marg):.4f}; positions x layers where the slab's walk is not below its plates': {pos_worse} of {NP * L}")
        log(f"        weight-only: lines whose slab gap digits exceed their plates' summed: {lw} of {ln} (equal: {le})"
            + (": so the choice holds on any light" if lw == 0 else ""))
    def walk_chosen(s, l):
        if s in ("qkv", "gu") and not choose[f"{s}.{l}"]: return S(f"{s}.{l}|walk_plates")
        return S(f"{s}.{l}|walk")
    log("")
    # ---- the work ----
    plates_plain = kd("plain"); top_plain = int(band_plain(dict(T)).sum()); arr_plain = A("plain_scores") + A("plain_values"); P0 = plates_plain + top_plain + arr_plain
    walk_new = sum(walk_chosen(s, l) for s in SN for l in range(L)); pours = sl("pours"); writing = sl("writing")
    walk_old = kd("gaps", src=C1); pours_old = kd("pours", src=C1); writing_old = sum(int(C1[f"{s}.{l}|writing"].sum()) for s in SN for l in range(L))
    plates_new = walk_new + pours + writing; plates_old = walk_old + pours_old + writing_old
    top_c = int(band_total(dict(T)).sum()); top_c1 = int(band_total(dict(T1)).sum())
    arr_d = A("sd_scores") + A("sd_values") + A("write_q") + A("write_k") + A("write_v")
    arr_d1 = sum(A(i, C1) for i in ("sd_scores", "sd_values", "write_q", "write_k", "write_v"))
    whole, whole1 = plates_new + top_c + arr_d, plates_old + top_c1 + arr_d1
    log(f"  the plain basis rebuilt: plates {pt(plates_plain):,.0f}, read {pt(top_plain):,.0f}, arrival {pt(arr_plain):,.0f}; total {pt(P0):,.0f} "
        f"(Phase VI-a: {PLAIN_PLATES + PLAIN_TOP + PLAIN_ARR:,})")
    log("")
    log("WORK A TOKEN BY PART (photons)")
    log(f"  {'part':34s} {'plain':>16s} {'qwen_fringes.py':>16s} {'this machine':>16s} {'of plain':>9s} {'of before':>9s}")
    def row(name, plain_, old_, new_):
        log(f"  {name:34s} {pt(plain_):16,.0f} {pt(old_):16,.0f} {pt(new_):16,.0f} {new_ / plain_ * 100:8.2f}% {new_ / old_ * 100:8.2f}%")
    for s in SN:
        row(f"  {s} walk (of the slab's plain)", kd("plain", KS[s]), kd("gaps", KS[s], C1), sum(walk_chosen(s, l) for l in range(L)))
    for s in SN:
        row(f"  {s} pours", kd("plain", KS[s]), kd("pours", KS[s], C1), sl("pours", [s]))
    row("  the light's writing", plates_plain, writing_old, writing)
    row("the 196 plates, (a) way 2", plates_plain, plates_old, plates_new)
    row("the top read, (c) the band read", top_plain, top_c1, top_c)
    row("the arrival, (d) signed digits", arr_plain, arr_d1, arr_d)
    row("THE WHOLE MACHINE", P0, whole1, whole)
    log(f"  the whole: {whole / P0 * 100:.2f}% of plain (qwen_fringes.py {whole1 / P0 * 100:.2f}%: {(whole - whole1) / P0 * 100:+.2f} points); "
        f"the plates {plates_new / plates_plain * 100:.2f}% of their plain (before {plates_old / plates_plain * 100:.2f}%)")
    log("")
    log("THE WALK PER PAIR (per lit light value x neuron) AND THE RELATION (light's signed digits per lit value x gap lit digits per weight)")
    log(f"  {'line-up':22s} {'neurons':>7s} {'walk/pair':>9s} {'light sd/lit':>12s} {'gap lit/w':>9s} {'relation':>8s} {'zero gaps':>9s} {'ctl walk/pr':>11s}")
    for s in ("qkv", "gu", "o", "down"):
        N_ = NS[s]; pairs = sl("pairs", [s]); litv = pairs // N_; sl_ = sl("lit", [s]) / litv
        LN = dict(qkv=1024, o=2048, gu=1024, down=3072)[s]; gl = sum(MJ["wgap"][f"slab {s}.{l}"] for l in range(L)) / (LN * N_ * L)
        zg = sum(MJ["zg"][f"slab {s}.{l}"] for l in range(L)) / (LN * (N_ - 1) * L)
        log(f"  {s + ' (one line-up)':22s} {N_:7d} {sl('walk', [s]) / pairs:9.3f} {sl_:12.3f} {gl:9.4f} {sl_ * gl:8.3f} {zg * 100:8.2f}% {sl('walk_shuf', [s]) / pairs:11.3f}")
        if s in ("qkv", "gu"):
            for k in KS[s]:
                Nk = NN[k]; pk = litv * Nk
                glk = sum(MJ["wgap"][f"{k}.{l}"] for l in range(L)) / (1024 * Nk * L); zgk = sum(MJ["zg"][f"{k}.{l}"] for l in range(L)) / (1024 * (Nk - 1) * L)
                log(f"  {'  ' + k + ' (own line-up)':22s} {Nk:7d} {kd('gaps', [k]) / pk:9.3f} {sl_:12.3f} {glk:9.4f} {sl_ * glk:8.3f} {zgk * 100:8.2f}%")
    log("")
    log("CONTROL (i), count only: each slab line's gaps with the slab's neurons in a random order; (a) at that slab, share of its plain")
    for s in SN:
        pl = kd("plain", KS[s]); log(f"  {s:4s} real {(sl('walk', [s]) + sl('pours', [s]) + sl('writing', [s])) / pl * 100:7.2f}%   gaps in a random neuron order "
                                     f"{(sl('walk_shuf', [s]) + sl('pours', [s]) + sl('writing', [s])) / pl * 100:7.2f}%")
    log("")
    log("SIDE (not used by the machine): (b) the pours in signed digits with the 3x walk, at the wider line-ups")
    nets = {}
    for s in SN:
        pairs = sl("pairs", [s]); sv = (sl("pours", [s]) - sl("pours_sd", [s])) / pairs; w3 = sl("walk3", [s]) / pairs
        nets[s] = sl("walk3", [s]) + sl("writing3", [s]) - (sl("pours", [s]) - sl("pours_sd", [s]))
        log(f"  {s:4s} 3x walk {w3:6.3f} a pair (walk {sl('walk', [s]) / pairs:.3f}); the signed pours save {sv:5.3f} a pair: (b) net {w3 - sv:+6.3f} a pair, {pt(nets[s]):+,.0f} a token")
    log(f"  (b) at every slab: {(whole + sum(nets.values())) / P0 * 100:.2f}% of plain; (b) only at the slab kinds where its net is below 0: "
        f"{(whole + sum(min(0, v) for v in nets.values())) / P0 * 100:.2f}% (shown only: no rule for it was fixed before the run)")

if __name__ == "__main__":
    ph = sys.argv[1]; ARG = dict(a.split("=") for a in sys.argv[2:])
    if ph == "machine": phase_machine()
    elif ph == "orderfree": phase_machine(order=7, tag="orderfree", ntext=3)
    elif ph == "words": phase_words()
    elif ph == "top": phase_top(ARG["set"])
    elif ph == "report": phase_report()
