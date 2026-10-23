"""SHAREDCOL/CUT.PY: THE EXACT CUT. THE TOP TABLE STORED AS ONE SHARED COLUMN PLUS EACH WORD'S FRINGE, AND THE TOP READ
MADE FROM THE FRINGES. Whole numbers in the machine path. 2026-10-06.

HIS PICTURE, AS THE SESSION PUT IT (the session's words, not his sentence; nothing here is quoted from him):
  the top table is one standing pattern shared by all words plus each word's own fringe. The light h meets the standing
  pattern once a position, and that adds the same h.c to every word, so only the fringes decide the race and the top
  5. Store and meet only the fringes.
  Followed forward: per position the light meets the column once (one pile, shared) and meets every word's fringe in
  the machine's own top read. The shared pile is the same for every word, so it cannot change who arrives first. It
  is added to a word only where the machine's own rounding could make two words arrive together (below).

WHICH MACHINES (THE RULE, FIXED BEFORE ANY RUN OF THIS FILE): a machine takes the cut only if the measure step found
the fringes' lit places a weight at least 5% below the table's. From measure_<table>.log (signed digits, the column
written like the values, mean11 | median):
    pythia-70m   16.44% | 16.61%   -> cut
    pythia-160m  14.92% | 15.16%   -> cut
    pythia-14m    1.20% |  1.24%   -> not cut
    gpt2          2.19% |  2.29%   -> not cut   (its top read is the band read; nothing here touches it)
    qwen          0.65% |  0.75%   -> not cut   (its top read is the band read; nothing here touches it)
So this file runs on Pythia 70m and 160m only. Their machine is foundation0/pythia_base2.py, loaded as it is, no
edits; its top read is the plain read: every word's pile is the light y16 (16 places, the position's own power 2^ey)
met with the word's column q_w (stored whole numbers, the word's own power 2^e_w), then put on the score grid 2^-16
with the machine's own rounding shift, and the race (pythia_base2.race: top digits first, the lowest word number among
equals).

THE COLUMN (one fixed choice, made before this run, not searched): mean11, the measure step's column written the way
the plate writes its values. Each lane's mean over the 50,304 words on the finest word grid, rounded to the COARSEST
word grid 2^emax (so it is a whole number on every word's own grid), then written with 11 significant places as fp16
writes a value (Pythia's plates were read once from fp16). The measure step's median column gave the same saving
within 0.25 points; it is not run here (no second choice, no winner picked).

THE FRINGES, EXACT. Word w's fringe on its own grid: u_w = q_w - (c << (emax - e_w)), whole numbers. Asserted at every
value: u_w + (c << (emax - e_w)) = q_w. A fringe value can need more than 16 places (the words on finer grids, whose
columns are far smaller than the standing pattern): those words are counted, and their whole fringe is stored in 32
places (exact); every other word's fringe fits 16 places.

THE PER-WORD SCALE, HANDLED EXACTLY. The machine's score for word w is
    lg_w = rshift(y16 . q_w, -(e_w + ey + 16))          (round to nearest: add half, shift)
         = floor(x_w + 1/2),   x_w = (y16 . q_w) 2^(e_w + ey + 16)
With F_w = y16 . u_w (the fringe pile, word w's own grid) and S = y16 . c (the shared pile, the coarsest grid),
    x_w = (F'_w + S'') 2^-s,   F'_w = F_w << (e_w - emin),   S'' = S << (emax - emin),   s = -(emin + ey + 16)
F'_w is the fringe pile moved to the finest grid by a shift (exact). S'' is ONE number for every word. So x_w is in the
order of F'_w, and lg_w = floor(x_w + 1/2) is in the same order, except that two words less than one score step apart
(0 < F'_a - F'_b < 2^s) can round to the same score; then the race takes the lower word number. The read:
  1. f5 = the 5th largest F'. K = the words with F'_w > f5 - 2^s. Every word outside K has x_w <= x_5 - 1, so its
     score is at least one step below the 5th's: it can neither be in the top 5 nor tie with it. (Checked at every
     word, every position, on the plain scores.)
  2. If no two words of K are less than one step apart (other than exactly equal), the scores of K are in the order of
     F' (equal F', equal score): the top 5 is K in the order (F' high first, then the lower word number), the race
     winner is its first. The shared pile is added to no word.
  3. Otherwise (a rounding tie is possible) the shared pile is added to the words of K only: lg_w = rshift(F'_w + S'',
     s), the machine's own rounding, and the top 5 is K in the order (score high first, then the lower word number).
  The plain read's top 5 is taken by the same rule on its own scores: the race run five times (judge5.top5's
  argpartition order is reported beside it; the two differ only where scores are exactly equal).

COUNTED (per position; the plain read's own form: plain 1s, as lineblanks/way2.py counts the plain read)
  plain read   sum over lanes of ones(y_l) x (ones of every word's value on lane l)
  cut read     sum over lanes of ones(y_l) x (ones of every word's fringe on lane l)  +  ones(y_l) x ones(c_l) (the
               shared pile, once)  +  where step 3 runs, the shared pile's photons added into each word of K (adds)
  the same in signed digits (naf(y) x naf of the values), the recording's form
  the light's writing (ones(y)) and the compares of the race and the top 5 are the same in both reads: not counted
  in the shares, reported apart.
  WAY 2 FORM (count only, never the machine here; lineblanks/way2.py's rule): walk = naf(y) x (lit digits of each
  lane's lowest value and its gaps, the lane lined up across the words, own grids as stored); pours = ones(|x w|) for
  every exposure; writing = ones(y). The same for the fringe table (+ the shared pile, plain photons).
  WHOLE MACHINE: the plates' plain meetings, lineblanks/way2_<size>.json (25 plates at 70m, 49 at 160m, finished,
  the finished machine's own light Lf; arrival products not in it), with the head's part replaced by the cut's.
  FOOTPRINT: the stored top table as whole numbers: 16 places a value, the wide words' fringes in 32 places with their
  word numbers, the column in 16 places, one power a word.

CHECKS (the check path is never used by the read and never counted)
  - the light is the machine's own: pythia_base2's own steps run here (its functions, loaded as capture.py loads them)
    on the 30 texts, prompt + the 15 fed-in words (622 positions); the last 16 positions' scores, by the module's own
    product, have the same sha256 as words_pythia-<size>.json (its own run): 30 of 30.
  - every pile: F_w + (S << (emax - e_w)) = y16 . q_w (the plain pile, int64 sums), every word, every position.
  - the plain scores by this file's int64 product equal the module's own (float64-proven) product, every position.
  - top 5 (set and order) and race winner of the cut read = the plain read's, every position; the race winner at the
    16 read places of every text = fp32 Pythia's own word (480 of 480, as pythia_base2 gets).
  - where step 3 runs: the cut's scores at K = the plain scores.
  - every word outside K scores at least one step below the plain 5th: violations counted (must be 0).
CONTROLS AND THE ORDER-FREE CHECK
  light=shuffled    each position's light shuffled across its lanes (every value kept, the lane pairing broken):
                    count only, the cut's share against the plain read's on the same shuffled light.
  colshuffle        the column's values dealt to other lanes (a random permutation of the lanes): the fringes
                    u = q - (c_perm << k). The read is run in full (it must stay exact: the identity holds for ANY
                    column) and counted: the saving belongs to the standing pattern only if it vanishes here.
  blank             Pythia's step0 head (head.s) with its own mean11 column, met by the finished light: count only
                    (the blank's words are not the machine's).
  moved             64 words given another of the 64's fringe (pairing broken, values kept): the pile identity must
                    fail at exactly the words whose fringe changed, at every position (the check has teeth).
  ORDER-FREE        the lanes summed in a random order and the words relabelled in a random order (each word keeps
                    its power and its fringe): every pile the same whole number, top 5 and race the same words.

RULE 13, CHECKED BEFORE THE RUN
  13.1 the measure is the top read's work (photon meetings) and the finished top 5 and race; the 1s of the light only
       weight the meetings, they are not an answer.
  13.2 the machine as it is: its own light, its own head product, rounding and race; the table is only rewritten as
       column + fringes, digit for digit. Nothing is dropped.
  13.3 the shuffle controls keep every value and break one pairing each: light with lanes, column with lanes, fringe
       with word. No exact-equality statistic is used as evidence of agreement; exactness is an identity checked per
       pile.
  13.4 definitions that fix the answer: (a) EXACTNESS IS FIXED BY CONSTRUCTION FOR ANY COLUMN (u = q - c is an
       identity), so "exact" says nothing about the standing pattern; colshuffle shows this. The finding is the
       count. (b) the weight-only part of the saving is the measure step's lit places (fixed before this run); what
       this run adds is the light's weighting across lanes, the per-word scale, the rounding ties, and the whole
       machine.
  13.5 his words are not quoted anywhere here; the picture above is the session's.
  13.6 every position, every word, every lane, both sizes; nothing stopped early.

SESSION'S PREDICTION (handed to this run, written here before it ran): exact everywhere; Pythia's top read photon
meetings fall 30-50%; GPT-2 and Qwen fall under 10%, so the whole machine moves under 1 point there.

THIS RUN'S PREDICTION (worked out before the run, from the measure step's weight-only numbers and way2_<size>.json):
  - Exact: top 5 (set and order) and race winner 622/622 at both sizes; 480/480 words; every pile identity holds;
    30/30 sha256; 0 bound violations.
  - K: 5 words at nearly every position (one score step is 2^-16 of a logit, the top 5 usually stand thousands of
    steps apart); step 3 (the shared pile added) at 0-3 positions of 622.
  - Plain read, plain 1s: the cut is 79-82% of the plain read at 70m (weight-only 20.05% fewer; the column is spread
    evenly over the lanes, so the light's weighting moves it by under 1.5 points) and 81-84% at 160m (17.84%).
    Signed digits: 82.5-85.5% (70m, weight-only 16.44%) and 83.5-86.5% (160m, 14.92%). The session's 30-50% fall:
    fails (about 16-20%, not 30-50%).
  - The shared pile's own meetings: under 0.01% of the plain read; step 3's adds under 0.001%.
  - Whole machine (plates' plain meetings; the head is 64.76% of them at 70m, 38.45% at 160m): 85.5-88.5% at 70m,
    92-94.5% at 160m.
  - Wide words (a fringe value past 16 places): 40-120 at 70m, 300-500 at 160m; footprint +0.1-0.3% (70m) and
    +0.6-1.0% (160m) in bytes. Their own meetings rise (their fringe is mostly the column).
  - Way 2 form (count only): walk within 5% of the table's (gaps between words on one grid are unchanged by a shared
    column; only words on other grids interleave differently); pours 5-12% fewer; the head's way-2 share from
    29.84% / 29.81% of plain to 26-28.5%.
  - Controls: light shuffled: the cut's share within 1 point of the real light's. colshuffle: exact, and meetings
    5-15% ABOVE the plain read (no saving). blank: 1.5-3% above the plain read (no standing pattern; the column's
    taking-away lights low places). moved: the identity fails at exactly the moved words (64 a position, fewer only if
    two moved words had equal fringes). Order-free: every pile identical; top 5 and race identical at every position
    except any exact score tie in the top 5 (0-2 positions).

RESULTS (2026-10-06; cut_70m.log, cut_160m.log; the second code path check_cut.py agrees on every number, see below)
  Exact at both sizes: top 5 (set and order) and race winner 622/622; words 480/480; every pile identity holds
  (0 of 31,289,088); the module's last-16 scores 30/30; bound violations 0; K = 5 at all 622 positions; step 3 (the
  shared pile added to any word) at 0 positions.                                                      -> held
  Top read, plain 1s: 70m 79.54% of the plain read (20.46% fewer), 160m 82.03% (17.97% fewer)       -> held (79-82, 81-84)
  Signed digits: 70m 83.26%, 160m 84.93%                                                          -> held
  The session's 30-50% fall                                                                      -> failed (18-20%)
  Shared pile: 21,995 / 35,895 meetings a position (0.002% of the plain read); adds 0                  -> held
  Whole machine (plates' plain meetings): 70m 86.75%, 160m 93.09%                                  -> held
  Wide words: 60 (70m), 396 (160m); footprint +0.122% / +0.791%                                    -> held
  Their own meetings rise: the 324 / 436 words off the pattern rise (plain 1s 5.544 -> 6.799, 5.252 -> 7.383); the
  wide words fall at 70m (5.768 -> 4.861) and rise at 160m (5.209 -> 7.400)                        -> half held
  Way 2 form: pours 10.2% / 8.8% fewer (held); walk +3.6% (70m, held) and +11.9% (160m, FAILED: 1,083 words on other
  grids interleave); head way-2 share 26.93% / 27.42% of plain (held); whole machine in way-2 form 39.13 -> 37.24%,
  44.31 -> 43.39%.
  Controls: light shuffled 79.95% / 82.16% (held); colshuffle exact 622/622 but 99.30% / 99.75% of plain, NOT 5-15%
  above (FAILED: a level of the wrong lane but the same sign still takes about two places off a value, one of the
  opposite sign adds about one; the lane levels are 255 up / 257 down at 70m, 396 / 372 at 160m); blank 102.71% /
  101.64% plain 1s (held), 102.37% / 101.44% signed (160m just under the 1.5% low end); moved: exactly the 64 moved
  words flagged at 622/622 (held); order-free: 0 piles differ, top 5 622/622 (held).
  The 70m cut was first run with the whole-table steps not split into lane blocks (2.41 GB) and the moved check counted
  per text; it was rerun with the blocks (994 MB) and every number came out the same.

    python3 sharedcol/cut.py size=70m phase=light    (runs the machine once; writes sharedcol/cutlight_<size>.npz)
    python3 sharedcol/cut.py size=70m phase=cut      (the cut, the checks, the controls; writes cut_<size>.log/.json)
"""
import os, sys, json, time, ctypes, subprocess, resource, hashlib, importlib.util, numpy as np

HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/sharedcol"; F0 = f"{HERE}/foundation0"
A_ = dict(a.split("=", 1) for a in sys.argv[1:]); SIZE = A_.get("size", "70m"); PHASE = A_.get("phase", "cut")
ES = 16
LOGF = open(f"{OUT}/cut_{SIZE}{'' if PHASE == 'cut' else '_' + PHASE}.log", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(X): return np.bitwise_count(np.abs(np.asarray(X, dtype=np.int64))).astype(np.int64)
def naf(X): a = np.abs(np.asarray(X, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def rdiv(a, b): return (2 * a + b) // (2 * b)                                             # whole-number rounding, b > 0
def sig11(x):                                                                             # measure.py's: 11 significant places
    x = np.asarray(x, np.int64); out = x.copy()
    for i, v in enumerate(x):
        b = int(abs(int(v))).bit_length()
        if b > 11: dr = b - 11; out[i] = rdiv(int(v), 1 << dr) << dr
    return out
def rshift(X, s):                                     # pythia_base2.rshift, copied unchanged: X x 2^-s, rounded to nearest
    s = np.asarray(s, dtype=np.int64)
    if (s < 0).any(): assert np.abs(X).max(initial=0) < 2 ** (62 + int(np.minimum(s, 0).min()))
    up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)
def pct(a, b): return f"{100 * a / b:.2f}%" if b else "-"
T0 = time.time()

# =============================== PHASE 1: THE MACHINE'S OWN LIGHT AT THE TOP ===============================
LIGHTF = f"{OUT}/cutlight_{SIZE}.npz"
if PHASE == "light":
    argv = sys.argv; sys.argv = ["pythia_base2.py", f"size={SIZE}", "plates=finished"]
    spec = importlib.util.spec_from_file_location("pythia_base2_cut", f"{F0}/pythia_base2.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.argv = argv
    assert m.PLATES == "finished" and m.ORDER == "plain" and m.MATMUL == "float64" and (m.PERM == np.arange(m.d)).all()
    refs = json.load(open(f"{F0}/refs_pythia-{SIZE}.json"))[:30]
    want_sha = [x["piles_sha256"] for x in json.load(open(f"{F0}/words_pythia-{SIZE}.json"))["texts_detail"]]
    w, e, _, wmax = m.PL["head"]
    Y, EY, ROWS, SHA, RACE, same = [], [], [], [], [], 0; p0 = 0
    for i, r in enumerate(refs):
        ids = np.asarray(r["ids"] + r["ref"][:15])
        R = m.rshift(m.pick(m.EMB_F[ids], m.EMB_S[ids])[:, m.PERM], -(m.EMB_E[ids][:, None] + ES))   # forward's own steps
        for l in range(m.L):
            At = m.plain(m.attention(m.plain(m.gauge(R, f"ln1.{l}"), f"qkv.{l}")), f"proj.{l}")
            Mo = m.plain(m.edge(m.plain(m.gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
            R = R + At + Mo
        y16, ey = m.to16(m.gauge(R, "lnf"))
        lg = m.rshift(m.product(y16, w, wmax), -(e[None, :] + ey[:, None] + ES))          # the module's own head, every position
        ok = hashlib.sha256(lg[-16:].tobytes()).hexdigest() == want_sha[i]; same += ok
        Y.append(y16.astype(np.int32)); EY.append(np.asarray(ey, np.int64)); ROWS.append((p0, p0 + len(ids))); p0 += len(ids)
        SHA += [hashlib.sha256(row.tobytes()).hexdigest() for row in lg]; RACE += [m.race(row) for row in lg]
        log(f"  text {i:2d} {len(ids):3d} positions  last-16 scores sha256 {'SAME' if ok else 'DIFFERENT'} as words_pythia-{SIZE}.json  ({time.time() - T0:.0f}s, {rss()} MB)")
    Y = np.concatenate(Y); assert np.abs(Y).max() <= 1 << 15
    np.savez(LIGHTF, y16=Y, ey=np.concatenate(EY), rows=np.array(ROWS), sha=np.array(SHA), race=np.array(RACE))
    log(f"{SIZE}: {p0} positions; the module's last-16 scores the same as its own run in {same} of 30 texts; wrote {LIGHTF} "
        f"({time.time() - T0:.0f}s, max {rss()} MB)")
    sys.exit(0)

# =============================== PHASE 2: THE CUT ===============================
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{OUT}/mm32.c", f"{OUT}/mm32.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); P_, I_, i_ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        _lib.mm32.argtypes = [I_, I_, I_, P_, P_, P_, P_, i_]; _lib.mm32.restype = None
    return _lib
def mm(X, W, count=False):
    """whole-number product, int64 sums (mm32.c); with count, the photons of every exposure x w a position"""
    X = np.ascontiguousarray(np.asarray(X, np.int64)); assert W.dtype == np.int32 and W.flags.c_contiguous
    P, L = X.shape; N = W.shape[1]; assert W.shape[0] == L and np.abs(X).max(initial=0) <= 1 << 15
    out = np.empty((P, N), np.int64); pour = np.zeros(P, np.int64)
    lib().mm32(P, L, N, X.ctypes.data, W.ctypes.data, out.ctypes.data, pour.ctypes.data, int(count))
    return out, pour

Z = np.load(f"{F0}/base2_pythia-{SIZE}.npz")
Q32 = np.ascontiguousarray(np.asarray(Z["head.w"]).astype(np.int32)); E = np.asarray(Z["head.e"]).astype(np.int64)
d, V = Q32.shape; emin, emax = int(E.min()), int(E.max()); G = emax - emin; KW = emax - E; UP = E - emin
log(f"SHAREDCOL CUT: pythia-{SIZE}, top table {d} lanes x {V} words, word powers {dict(zip(*[x.tolist() for x in np.unique(E, return_counts=True)]))}")
BL = 64                                               # lanes a block (memory: every whole-table step is done block by block)
def blocks(T):
    for l0 in range(0, d, BL): yield l0, np.asarray(T[l0:l0 + BL]).astype(np.int64)
def column(T):
    """mean11: each lane's mean over the words on the finest grid, rounded to the coarsest grid, 11 significant places"""
    S = np.zeros(d, np.int64)
    for l0, A in blocks(T): S[l0:l0 + len(A)] = (A << UP[None, :]).sum(1)
    return sig11(rdiv(S, V << G))
def fringes(T, c):
    """u_w = q_w - (c << (emax - e_w)) on each word's own grid, int32; the identity asserted at every value"""
    U = np.empty(T.shape, np.int32)
    for l0, A in blocks(T):
        cc = c[l0:l0 + len(A), None] << KW[None, :]; F = A - cc
        assert np.array_equal(F + cc, A), "fringe + column is not the table"; assert np.abs(F).max() < 1 << 24
        U[l0:l0 + len(A)] = F
    return U
def lane_lit(T):
    N1, Nn = np.zeros(d, np.int64), np.zeros(d, np.int64)
    for l0, A in blocks(T): N1[l0:l0 + len(A)] = ones(A).sum(1); Nn[l0:l0 + len(A)] = naf(A).sum(1)
    return N1, Nn
def word_sq(T):                                       # each word's summed squares on the finest grid (real sizes)
    acc = np.zeros(V)
    for l0, A in blocks(T): Af = (A << UP[None, :]).astype(np.float64); acc += (Af * Af).sum(0)
    return acc
def wide_of(T):
    w = np.zeros(V, bool)
    for l0, A in blocks(T): w |= (np.abs(A) > 32767).any(0)
    return np.flatnonzero(w)
def lineup_lit(T):
    """way 2's weight-only record: per lane, lit digits (signed) of the lowest value and of every gap, the lane lined up
    across the words as stored (own grids); and its zero gaps"""
    GN = np.empty(d, np.int64); ZG = 0
    for l0, A in blocks(T):
        s = np.sort(A, axis=1); g = np.diff(s, axis=1); GN[l0:l0 + len(A)] = naf(s[:, 0]) + naf(g).sum(1); ZG += int((g == 0).sum())
    return GN, ZG

C = column(Q32); assert np.abs(C).max() < 1 << 15
U32 = fringes(Q32, C)
wide = wide_of(U32); nw = len(wide)
off = np.flatnonzero(word_sq(U32) > word_sq(Q32))
NW1, NWn = lane_lit(Q32); NU1, NUn = lane_lit(U32); C1, Cn = ones(C), naf(C)
mj = json.load(open(f"{OUT}/measure_pythia-{SIZE}.json"))["finished"]
def per_w(T, ws): return ones(np.asarray(T[:, ws]).astype(np.int64)).sum() / max(len(ws) * d, 1)
log(f"  the column (mean11) on the coarsest grid 2^{emax}: |c| up to {int(np.abs(C).max())}; {int((C > 0).sum())} lanes up, {int((C < 0).sum())} down; its lit places {int(Cn.sum())} signed, {int(C1.sum())} plain 1s")
log(f"  fringe + column = table at every value (asserted).  words with a fringe value past 16 places: {nw}; words whose fringe is larger than their own column: {len(off)}")
log(f"    the wide words' powers: {dict(zip(*[x.tolist() for x in np.unique(E[wide], return_counts=True)]))}; their plain 1s a weight, table {per_w(Q32, wide):.3f} -> fringe {per_w(U32, wide):.3f}")
log(f"    the words off the pattern: powers {dict(zip(*[x.tolist() for x in np.unique(E[off], return_counts=True)]))}; their plain 1s a weight, table {per_w(Q32, off):.3f} -> fringe {per_w(U32, off):.3f}")
log(f"  lit places a weight, weight only: plain 1s table {NW1.sum() / (d * V):.4f} fringe {NU1.sum() / (d * V):.4f} ({pct(NW1.sum() - NU1.sum(), NW1.sum())} fewer);"
    f"  signed table {NWn.sum() / (d * V):.4f} fringe {NUn.sum() / (d * V):.4f} ({pct(NWn.sum() - NUn.sum(), NWn.sum())} fewer)"
    f"   [measure step, mean11: plain {mj['lit']['mean11']['ones_per_weight']:.4f}, signed {mj['lit']['mean11']['signed_per_weight']:.4f}]")
assert abs(NUn.sum() / (d * V) - mj["lit"]["mean11"]["signed_per_weight"]) < 1e-9 and abs(NU1.sum() / (d * V) - mj["lit"]["mean11"]["ones_per_weight"]) < 1e-9
B_plain = d * V * 2 + V; B_cut = d * (V - nw) * 2 + d * nw * 4 + nw * 4 + d * 2 + V + 1
log(f"  footprint, the stored top table: plain {B_plain:,} bytes ({B_plain / 1e6:.2f} MB: 16 places a value, one power a word);"
    f" cut {B_cut:,} bytes ({B_cut / 1e6:.2f} MB: fringes in 16 places, the {nw} wide words' in 32 places with their numbers, the column, the powers): {B_cut / B_plain - 1:+.3%}")
GNq, ZGq = lineup_lit(Q32); GNu, ZGu = lineup_lit(U32)
log(f"  way 2 record (count only): lit digits of the lowest values and gaps, table {int(GNq.sum()):,} ({ZGq / (d * (V - 1)):.2%} zero gaps), fringes {int(GNu.sum()):,} ({ZGu / (d * (V - 1)):.2%} zero gaps)")

# controls' tables
rng = np.random.default_rng([20261006, {"70m": 70, "160m": 160}[SIZE]])
cperm = rng.permutation(d); Cp = C[cperm]; Up32 = fringes(Q32, Cp)
moved = np.sort(rng.choice(np.setdiff1d(np.arange(V), wide), 64, replace=False)); mperm = moved[rng.permutation(64)]
QS32 = np.asarray(Z["head.s"]).astype(np.int32); CB = column(QS32); UB32 = fringes(QS32, CB)
(NBW1, NBWn), (NBU1, NBUn) = lane_lit(QS32), lane_lit(UB32); del QS32, UB32
NP1, NPn = lane_lit(Up32)
lperm = rng.permutation(d); wperm = rng.permutation(V)                         # ORDER-FREE: lanes and words relabelled
Uo32 = np.ascontiguousarray(U32[lperm][:, wperm]); Eo = E[wperm]; Co = C[lperm]
log(f"  ({time.time() - T0:.0f}s, {rss()} MB)")

# =============================== PHASE 3: THE READS ===============================
LZ = np.load(LIGHTF); Y = LZ["y16"].astype(np.int64); EY = LZ["ey"]; ROWS = LZ["rows"]; MSHA = LZ["sha"]; MRACE = LZ["race"]
refs = json.load(open(f"{F0}/refs_pythia-{SIZE}.json"))[:30]; NPOS = len(Y)
W2 = json.load(open(f"{HERE}/lineblanks/way2_{SIZE}.json"))
PL_TOTAL = sum(v["finished"]["Lf"]["plain"] for v in W2.values() if isinstance(v, dict) and "finished" in v and "Lf" in v["finished"])
W2_TOTAL = sum(v["finished"]["Lf"]["way2"] for v in W2.values() if isinstance(v, dict) and "finished" in v and "Lf" in v["finished"])
W2H = W2["head"]["finished"]["Lf"]

def top5_rule(vals):                                  # the race run five times: high first, then the lower word number
    n = len(vals); v5 = np.partition(vals, n - 5)[n - 5]; cand = np.flatnonzero(vals >= v5)
    return cand[np.lexsort((cand, -vals[cand]))[:5]]
def judge5_top5(S):                                   # judge5.top5's two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

def cut_read(F, S, ey, Ew):
    """THE MACHINE PATH. F: every word's fringe pile (own grid), S: the shared pile (coarsest grid), ey: the position's
    power, Ew: each word's power. Returns top 5, the words in K, whether the shared pile was added, the adds' photons,
    and the scores computed (word -> score) where it was."""
    Fp = F << (Ew - emin); s = -(emin + int(ey) + ES); n = len(Fp)
    f5 = np.partition(Fp, n - 5)[n - 5]
    K = np.flatnonzero(Fp > f5 - (1 << s)) if s >= 1 else np.flatnonzero(Fp >= f5)
    Ks = K[np.lexsort((K, -Fp[K]))]; dif = -np.diff(Fp[Ks])
    amb = bool(s >= 1 and ((dif > 0) & (dif < (1 << s))).any())
    if not amb: return Ks[:5], K, False, 0, None
    S2 = int(S) << (emax - emin); lgK = rshift(Fp[K] + S2, s)
    return K[np.lexsort((K, -lgK))][:5], K, True, len(K) * int(ones(S2)), dict(zip(K.tolist(), lgK.tolist()))

R = dict(size=SIZE, positions=NPOS, wide_words=int(nw), wide_list=wide.tolist(), off_pattern_words=int(len(off)), footprint=dict(plain=B_plain, cut=B_cut))
cnt = {k: 0 for k in ("plain1", "plainn", "cut1", "cutn", "shared1", "sharedn", "adds", "writing1", "writingn",
                      "walk_q", "walk_u", "pours_q", "pours_u", "ctl_plain1", "ctl_cut1", "ctl_plainn", "ctl_cutn",
                      "cs_cut1", "cs_cutn", "blank_plain1", "blank_cut1", "blank_plainn", "blank_cutn")}
chk = {k: 0 for k in ("pile_identity_bad", "module_scores_bad", "top5_same", "race_same", "module_race_same", "judge5_top5_differs",
                      "bound_violations", "amb_positions", "amb_scores_bad", "K_sizes_over5", "words_ok", "words_n",
                      "cs_top5_same", "cs_race_same", "cs_amb", "of_piles_bad", "of_top5_same", "of_race_same",
                      "moved_flagged_exact", "moved_flagged_words")}
Ksz, amb_at, of_diff_at = [], [], []
for ti, (a, b) in enumerate(ROWS):
    X = Y[a:b]; P = b - a; ey = EY[a:b]; o1, on = ones(X), naf(X)
    # ---- the check path: the plain read (int64 product of the stored table), its scores, its top 5 and race ----
    PQ, pourq = mm(X, Q32, count=True)
    lg = rshift(PQ, -(E[None, :] + ey[:, None] + ES))
    chk["module_scores_bad"] += sum(hashlib.sha256(lg[j].tobytes()).hexdigest() != MSHA[a + j] for j in range(P))
    # ---- the machine path: the shared pile once, every word's fringe pile ----
    S = X @ C                                                                     # d meetings a position
    F, pouru = mm(X, U32, count=True)
    chk["pile_identity_bad"] += int((F + (S[:, None] << KW[None, :]) != PQ).sum())
    # moved: 64 words given another's fringe pile (the product of the moved table is the same piles, moved)
    Fm = F.copy(); Fm[:, moved] = F[:, mperm]
    bad = (Fm + (S[:, None] << KW[None, :]) != PQ)
    chk["moved_flagged_words"] += int(bad.sum()); chk["moved_flagged_exact"] += sum(bool(np.array_equal(np.flatnonzero(bad[j]), np.sort(moved[F[j, moved] != F[j, mperm]]))) for j in range(P))
    # colshuffle: the column's values dealt to other lanes; the read run in full
    Sp = X @ Cp; Fq, _ = mm(X, Up32)
    # order-free: lanes in a random order, words relabelled
    So = X[:, lperm] @ Co; Fo, _ = mm(X[:, lperm], Uo32)
    J5 = judge5_top5(lg)
    for j in range(P):
        p = a + j; t5p = top5_rule(lg[j]); rp = int(t5p[0])
        chk["module_race_same"] += rp == int(MRACE[p]); chk["judge5_top5_differs"] += not np.array_equal(J5[j], t5p)
        t5, K, amb, adds, lgK = cut_read(F[j], S[j], ey[j], E)
        chk["top5_same"] += np.array_equal(t5, t5p); chk["race_same"] += int(t5[0]) == rp
        Ksz.append(len(K)); chk["K_sizes_over5"] += len(K) > 5; cnt["adds"] += adds
        if amb:
            chk["amb_positions"] += 1; amb_at.append(int(p)); chk["amb_scores_bad"] += sum(lgK[w] != lg[j, w] for w in K)
        out = np.ones(V, bool); out[K] = False
        chk["bound_violations"] += int((lg[j, out] > lg[j, t5p[4]] - 1).sum())
        if j >= P - 16:
            chk["words_n"] += 1; chk["words_ok"] += int(t5[0]) == refs[ti]["ref"][j - (P - 16)]
        t5c, _, ambc, _, _ = cut_read(Fq[j], Sp[j], ey[j], E)
        chk["cs_top5_same"] += np.array_equal(t5c, t5p); chk["cs_race_same"] += int(t5c[0]) == rp; chk["cs_amb"] += ambc
        Fo_back = np.empty(V, np.int64); Fo_back[wperm] = Fo[j]
        chk["of_piles_bad"] += int((Fo_back != F[j]).sum()) + int(So[j] != S[j])
        t5o, _, _, _, _ = cut_read(Fo[j], So[j], ey[j], Eo)
        same5 = np.array_equal(wperm[t5o], t5p); chk["of_top5_same"] += same5; chk["of_race_same"] += int(wperm[t5o[0]]) == rp
        if not same5: of_diff_at.append(int(p))
    # ---- counts ----
    cnt["plain1"] += int((o1 @ NW1).sum()); cnt["plainn"] += int((on @ NWn).sum())
    cnt["cut1"] += int((o1 @ NU1).sum()); cnt["cutn"] += int((on @ NUn).sum())
    cnt["shared1"] += int((o1 @ C1).sum()); cnt["sharedn"] += int((on @ Cn).sum())
    cnt["writing1"] += int(o1.sum()); cnt["writingn"] += int(on.sum())
    cnt["walk_q"] += int((on @ GNq).sum()); cnt["walk_u"] += int((on @ GNu).sum()); cnt["pours_q"] += int(pourq.sum()); cnt["pours_u"] += int(pouru.sum())
    cnt["cs_cut1"] += int((o1 @ NP1).sum() + (o1 @ ones(Cp)).sum()); cnt["cs_cutn"] += int((on @ NPn).sum() + (on @ naf(Cp)).sum())
    cnt["blank_plain1"] += int((o1 @ NBW1).sum()); cnt["blank_cut1"] += int((o1 @ NBU1).sum() + (o1 @ ones(CB)).sum())
    cnt["blank_plainn"] += int((on @ NBWn).sum()); cnt["blank_cutn"] += int((on @ NBUn).sum() + (on @ naf(CB)).sum())
    Xs = rng.permuted(X, axis=1); assert (np.sort(Xs, 1) == np.sort(X, 1)).all(); s1, sn = ones(Xs), naf(Xs)
    cnt["ctl_plain1"] += int((s1 @ NW1).sum()); cnt["ctl_cut1"] += int((s1 @ NU1).sum() + (s1 @ C1).sum())
    cnt["ctl_plainn"] += int((sn @ NWn).sum()); cnt["ctl_cutn"] += int((sn @ NUn).sum() + (sn @ Cn).sum())
    log(f"  text {ti:2d} {P:3d} positions: top 5 same {chk['top5_same']}/{b} so far, piles bad {chk['pile_identity_bad']}, "
        f"K>5 {chk['K_sizes_over5']}, step 3 {chk['amb_positions']}  ({time.time() - T0:.0f}s, {rss()} MB)")

R["checks"] = {k: int(v) for k, v in chk.items()}; R["counts"] = {k: int(v) for k, v in cnt.items()}; R["K"] = dict(median=float(np.median(Ksz)), max=int(max(Ksz)), sizes=dict(zip(*[x.tolist() for x in np.unique(Ksz, return_counts=True)])))
R["amb_at"] = amb_at; R["orderfree_differs_at"] = of_diff_at
n = NPOS
log(f"\n== EXACT (pythia-{SIZE}, {n} positions) ==")
log(f"  module's own scores rebuilt by the int64 product: positions differing {chk['module_scores_bad']}; plain race = module's race at {chk['module_race_same']}/{n}")
log(f"  every pile: fringe pile + shared pile (shifted to the word's grid) = the plain pile; values differing {chk['pile_identity_bad']} of {n * V:,}")
log(f"  CUT READ: top 5 (set and order) the same as the plain read at {chk['top5_same']}/{n}; race winner {chk['race_same']}/{n}; "
    f"words at the 16 read places = fp32 Pythia's {chk['words_ok']}/{chk['words_n']}")
log(f"  K (words within one score step of the 5th, by fringes alone): median {R['K']['median']:.0f}, largest {R['K']['max']}, sizes {R['K']['sizes']}")
log(f"  step 3 (shared pile added to K, a rounding tie possible): {chk['amb_positions']} positions {amb_at}; their scores = plain: {chk['amb_scores_bad']} differ")
log(f"  words outside K scoring within one step of the plain 5th (bound violations): {chk['bound_violations']}")
log(f"  judge5.top5's order differs from the race-run-five-times order at {chk['judge5_top5_differs']} positions (exact score ties only)")
log(f"\n== WORK OF THE TOP READ, a position (plain read = this file's count; way2_{SIZE}.json's head plain {W2H['plain'] / n:,.0f}) ==")
log(f"  plain 1s:      plain read {cnt['plain1'] / n:,.0f}   cut: fringes {cnt['cut1'] / n:,.0f} + shared pile {cnt['shared1'] / n:,.0f} + adds {cnt['adds'] / n:,.2f}"
    f" = {(cnt['cut1'] + cnt['shared1'] + cnt['adds']) / n:,.0f}  -> {pct(cnt['cut1'] + cnt['shared1'] + cnt['adds'], cnt['plain1'])} of the plain read")
log(f"  signed digits: plain read {cnt['plainn'] / n:,.0f}   cut: fringes {cnt['cutn'] / n:,.0f} + shared pile {cnt['sharedn'] / n:,.0f} + adds {cnt['adds'] / n:,.2f}"
    f" = {(cnt['cutn'] + cnt['sharedn'] + cnt['adds']) / n:,.0f}  -> {pct(cnt['cutn'] + cnt['sharedn'] + cnt['adds'], cnt['plainn'])} of the plain read in signed digits")
log(f"  the same light as way2_{SIZE}.json: plain read {'EQUAL' if cnt['plain1'] == W2H['plain'] else 'DIFFERENT'}, walk {'EQUAL' if cnt['walk_q'] == W2H['walk'] else 'DIFFERENT'}, pours {'EQUAL' if cnt['pours_q'] == W2H['pours'] else 'DIFFERENT'} (its head counts, the same plate and light)")
log(f"  (the light's writing, the same in both: plain {cnt['writing1'] / n:,.0f}, signed {cnt['writingn'] / n:,.0f}; the race's and top 5's compares the same in both)")
cut_w = cnt["cut1"] + cnt["shared1"] + cnt["adds"]
log(f"  WHOLE MACHINE (plates' plain meetings, way2_{SIZE}.json, {PL_TOTAL / n:,.0f} a position; head {W2H['plain'] / PL_TOTAL:.2%} of it): "
    f"{(PL_TOTAL - W2H['plain'] + cut_w) / n:,.0f} -> {pct(PL_TOTAL - W2H['plain'] + cut_w, PL_TOTAL)}")
w2q = cnt["walk_q"] + cnt["pours_q"] + cnt["writing1"]; w2u = cnt["walk_u"] + cnt["pours_u"] + cnt["writing1"] + cnt["shared1"]
log(f"  WAY 2 FORM (count only): table walk {cnt['walk_q'] / n:,.0f} pours {cnt['pours_q'] / n:,.0f} writing {cnt['writing1'] / n:,.0f} = {pct(w2q, cnt['plain1'])} of plain"
    f"  [way2_{SIZE}.json: walk {W2H['walk'] / n:,.0f} pours {W2H['pours'] / n:,.0f}];  fringes walk {cnt['walk_u'] / n:,.0f} pours {cnt['pours_u'] / n:,.0f} + shared {cnt['shared1'] / n:,.0f}"
    f" = {pct(w2u, cnt['plain1'])} of plain ({pct(w2u, w2q)} of the table's way 2); whole machine in way 2 form {pct(W2_TOTAL, PL_TOTAL)} -> {pct(W2_TOTAL - W2H['way2'] + w2u, PL_TOTAL)}")
log(f"\n== CONTROLS ==")
log(f"  light shuffled across its lanes (count only): cut {pct(cnt['ctl_cut1'], cnt['ctl_plain1'])} of plain (plain 1s), {pct(cnt['ctl_cutn'], cnt['ctl_plainn'])} (signed); real light {pct(cut_w, cnt['plain1'])} / {pct(cnt['cutn'] + cnt['sharedn'] + cnt['adds'], cnt['plainn'])}")
log(f"  colshuffle (the column's values in other lanes): top 5 same {chk['cs_top5_same']}/{n}, race {chk['cs_race_same']}/{n}, step 3 at {chk['cs_amb']}; "
    f"meetings {pct(cnt['cs_cut1'], cnt['plain1'])} of plain (plain 1s), {pct(cnt['cs_cutn'], cnt['plainn'])} (signed)")
log(f"  blank head (step0, its own mean11 column, the finished light; count only): {pct(cnt['blank_cut1'], cnt['blank_plain1'])} of its plain read (plain 1s), {pct(cnt['blank_cutn'], cnt['blank_plainn'])} (signed)")
log(f"  moved (64 words given another's fringe): piles flagged {chk['moved_flagged_words']:,} = {chk['moved_flagged_words'] / n:.2f} a position; flagged exactly the moved words whose fringe pile changed at {chk['moved_flagged_exact']}/{n} positions")
log(f"  ORDER-FREE (lanes in a random order, words relabelled): piles differing {chk['of_piles_bad']}; top 5 same {chk['of_top5_same']}/{n} (differs at {of_diff_at}); race {chk['of_race_same']}/{n}")
json.dump(R, open(f"{OUT}/cut_{SIZE}.json", "w"), indent=1)
log(f"done ({time.time() - T0:.0f}s, max memory {rss()} MB)")
