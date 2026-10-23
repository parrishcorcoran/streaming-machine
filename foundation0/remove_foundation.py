"""HIS SENTENCE ON THE RUNNING MACHINE. 2026-10-06.

HIS IDEA, his exact words (only these are his):
  "The blank untrained model with random bell curve wieghts! That's the plate and the embedding table! That's the
  "foundation" we have been looking for." "It should be 100% removable after training. We could even test on a tiny
  model." "The random weights have random values and everything. That is trained is built on top. It creates the
  topographical map. That's IT! And the entire thing pre-training could be removed"

THE SENTENCE TESTED HERE, his: "It should be 100% removable after training."

SESSION'S RESTATEMENT (the session's words, not his): finished weight W = foundation + what was built on top. The
foundation is the random starting draw W0, worn down by the building, so what survives of it in W is s x W0, one s per
plate (measured by foundation_share.py: s = (W . W0)/(W0 . W0), an exact fraction on the shared grid).

WHAT IS RUN: the base-2 machine of pythia_base2.py (its parts copied here unchanged: gauge, turns, arrival with its water
level, bend, race), on the shared whole-number grid of base2_pythia-<size>.npz, with the plates and word tables changed
case by case, against the finished machine, at every position of the 30 texts (each text's own words, then the finished
machine's 16 greedy words fed in), and the case's own 16 greedy words.
  fin    the finished machine (must give back pythia_base2.py's own head piles: checked by sha256)
  a      every plate and both word tables with the whole start removed:          W - W0
  b      the surviving foundation removed:                                       W - round(s x W0), s per plate
  ca,cb  CONTROLS: a and b with W0 shuffled within each plate (the same values, positions scrambled; a value moved to
         a neuron on another power of two is written on that neuron's grid, rounded to nearest where it loses places)
  X_tab  the foundation removed only from the two word tables (X = a, b, ca, cb)
  X_pl   only from the plates (every layer's qkv, proj, fc, mproj)
  X_L<l> only from layer l's four plates
  Gains, norm offsets and biases are never touched: their start is the constant 1 or 0, not a draw.
Measured for each case, against the finished machine:
  - top-1: the case's word (the race's winner) equals the finished machine's word, at every position
  - top-5 (judge-style): the finished machine's word is inside the case's top 5
  - the 16 greedy words: the case run on its own words from each text; words equal position by position, texts
    identical over all 16
  - where it moves: in each layer, the share of arrival rows whose water level is set by another position (the race's
    winner in the row moves; rows with one position are left out), and the share of knife edges (each bend's input,
    open if above 0) that changed side. Both on the same fed-in words as the finished machine.
  - statistic controls (pairing broken, rule 13.3): the case's words against the NEXT text's finished words at the
    same continuation position; the finished machine's water level and knife edges against the next text's.

PREDICTIONS, written before the first run of this script (results go to remove_<size>.log / .json, never here):

(0) THE SESSION'S PREDICTIONS as handed to this run, copied as given:
    "(a) and (b) lose most words (top-1 under 50%), because what was built was built while the light passed through
    the foundation, so the map's hills sit at heights that assume the terrain is there; (c) breaks about as much or
    more than (a)/(b); tables only keep most words (top-1 80% or more); one layer at a time loses some words at every
    layer."

(1) THIS RUN'S OWN ANALYSIS, from the stored step's numbers (foundation_share.json), before any of these cases ran.
    The start's size against the finished plate, |W0|/|W| = rho/s: 14m 0.27-2.40 (above 1 at 21 of 24 plates),
    70m 0.13-0.97, 160m 0.19-0.84; word tables (head, embed) 14m 1.52/1.76, 70m 0.24/0.90, 160m 0.20/0.80.
    So case a lays a fresh random relief of 0.13-2.4 times each plate's own size over every plate (sign reversed).
    Case b takes away the surviving foundation, a relief of size rho against the plate: 0.02-0.55 (14m), 0.02-0.25
    (70m), 0.02-0.36 (160m); 0.003-0.03 on the word tables.
    For his sentence to hold at a, W0 . x would have to be about 0 for every light x the machine meets; a random plate
    has no such blind side, so a cannot hold. b is the fair test of "the light does not need the foundation".
    - a: the words go. Top-1 at all positions under 25% at every size (14m near the pairing-broken control), top-5
      under 50%, 0 of 30 greedy texts identical.
    - b: a much smaller cut than a, but in every plate at once. Top-1 40-75% at 70m and 160m, 20-50% at 14m; 0-5 of 30
      greedy texts identical.
    - controls: ca and cb lay the same values, scrambled. If the light runs along the foundation's own grooves (the
      session's restatement), removing the real start moves the piles more than removing a scrambled copy of it:
      b loses at least 5 points more top-1 than cb at 70m and 160m; a at or below ca. If b and cb come out equal, the
      surviving foundation is to the light just another random relief of its size.
    - tables only: b_tab keeps top-1 at 95% or more (a 0.3-3% relief). a_tab: 14m under 30%; 70m and 160m 30-70%
      (the input table's start is 0.8-0.9 of its size; the gauge after it rescales the row but cannot undo its turn).
    - plates only: a_pl about as broken as a; b_pl nearly as b.
    - one layer at a time: a_L loses words at every layer, layer 0 costs most at 70m and 160m and the last layers
      least (their qkv start is 0.13-0.22 of the plate). b_L loses some words at every layer (top-1 75-97%), layer 0
      costing most at 14m and 70m (fc.0 rho 0.55, 0.25), at 160m layer 0 or layer 2 (fc rho 0.36, 0.24).
    - where it moves first: in every case that touches layer 0 the knife edges change side from layer 0 on; in b the
      water level holds better than the knife edges in the first layers (a smaller share of rows' levels move than of
      edges' sides).
    - his sentence holds on the running machine if a case that removes the foundation keeps every word (literally
      100%); this run calls it near-holding at top-1 95% or more with 25 or more of 30 greedy texts identical.

(2) ADDED AFTER THE FIRST RUNS (70m and 14m done, 160m running), written before these extra 70m runs (tag=extra):
    switches added: seed= (another scramble draw), parts _head and _emb (one word table alone), and a look at the
    finished machine's arrival rows: the first arrival's share of its row (ARR[0] over the row's total; a flat row,
    where many positions sit near the level, gives a small share).
    - second scramble draw (seed=2): cb, cb_tab, cb_pl within 3 points of top-1 of the first draw (51.6, 89.4, 52.9);
      ca 0-1%. So the b-against-cb gap (+1.0 point at 70m, -2.3 at 14m) sits inside the draw-to-draw spread.
    - one table alone: in b_tab the water levels move in 1-1.5% of rows and 0.3% of knife edges change side; those
      come only from the input table (the output table comes after every layer). So the 12% of words lost in b_tab
      are mostly lost at the output table: b_head 88-93% top-1, b_emb 95-99%; cb_head, cb_emb within 3 points of them.
      a_head 20-50% (its start is 0.24 of it), a_emb 0-10% (0.90 of it).
    - flat rows: in the finished 70m the first arrival's share of its row is smallest in the last two layers, and in b
      the rows whose level moved have a smaller first-arrival share than the rows that held, at every layer.
(3) ADDED after the 160m run (written before the seed=2 runs at 160m and 14m, and the layer-0 cases at 70m):
    the first 160m run found b 7.7 points of top-1 below cb (50.5 vs 58.2), b_pl 7.3 below cb_pl, and at layer 0
    a_L0 24.4 against ca_L0 48.2 and b_L0 80.7 against cb_L0 86.7; 14m b_L0 37.8 against cb_L0 44.4; 70m layer 0 equal.
    Prediction for the second scramble draw (seed=2): 160m cb and cb_pl within 3 points of 58.2 and 60.5, ca_L0 within
    5 of 48.2, cb_L0 within 3 of 86.7; 14m cb_L0 within 3 of 44.4; 70m ca_L0, cb_L0 within 3 of 14.6, 81.8. If so, at
    160m (and at layer 0 of 14m) the light leans on the real foundation more than on a random relief of its size, and
    at 70m it does not.
(4) ADDED after the 70m seed=2 run (written before the seed=3,4,5 runs at every size). That run moved some controls by
    more than (3) allowed: 70m cb_tab 89.4 -> 94.1, ca_L0 14.6 -> 25.1 (cb 51.6 -> 52.6, cb_pl 52.9 -> 52.7, cb_L0
    81.8 -> 84.4). It also found b_emb at 100.0% (every word, 30 of 30 greedy texts) against cb_emb 99.5%, b_head 90.0
    against cb_head 94.5, a_emb 26.7, a_head 0.0; and the finished 70m's last two layers are NOT flat: the first
    arrival takes a median 96% and 99.5% of its row there (prediction (2) on flat rows was wrong).
    So the scramble's own spread must be measured. Five draws in all (seeds 20261006, 2, 3, 4, 5):
    - 70m: cb spread (largest minus smallest top-1) 3 points or less; cb_tab and ca_L0 up to 12 points. b (52.6) inside
      the cb draws. b_emb 100% at or above every cb_emb draw; b_head (90.0) at or below every cb_head draw.
    - 160m: b (50.5) below every cb draw and a_L0 (24.4) below every ca_L0 draw.
    - 14m and 160m: b_emb 99% or more; b_head below b_emb.
    - who sets the level (counted for every case from now on): in the finished 70m's layers 4-5 the first word, the
      row itself and the row before together set the level in 70% or more of rows, and the same three hold in b.
(5) ADDED after the five draws (written before the tag=setter runs of b at 14m and 160m, greedy=0): at 70m b moved
    the level off the first word in layer 4 (57% -> 35% of rows; other positions 33% -> 48%). In the finished 160m the
    first word sets the level in 53-71% of rows at layers 4-11, where the first arrival takes 69-100% of its row.
    Prediction: in b at 160m the first word's share of the levels falls at every layer from 4 to 11, by 10 points or
    more at layers 9-11 (where 52-66% of levels move); the same direction at 14m in its last two layers.

  python3 remove_foundation.py size=70m [cases=all|fin,a,b,...] [greedy=1] [order=plain|shuffle] [texts=30] [tag=] [seed=]
"""
import os, sys, json, math, time, hashlib, zlib, resource
import numpy as np

A = dict(a.split("=") for a in sys.argv[1:])
SIZE, ORDER = A.get("size", "70m"), A.get("order", "plain")
GREEDY = A.get("greedy", "1") == "1"
SEED = int(A.get("seed", 20261006))                                    # the scramble's draw (added after the first runs)
TEXTS, WORDS = int(A.get("texts", 30)), int(A.get("words", 16))
HERE = os.path.dirname(os.path.abspath(__file__))
M = np.load(os.path.join(HERE, f"base2_pythia-{SIZE}.npz"))
d, L, NH, HS, ROT = (int(M[k]) for k in ("d", "L", "heads", "hs", "rot"))
ES, SC = 16, 10
PHI, ARR, LOG2, EXP2, COS, SIN = (M[k] for k in ("phi", "arrival", "log2", "exp2", "cos", "sin"))
EPS40, HALF = int(M["eps40"]), int(M["half_log2_hs"])
PERM = np.random.default_rng(12345).permutation(d) if ORDER == "shuffle" else np.arange(d)
SHUF = ORDER == "shuffle"
SHARE = json.load(open(os.path.join(HERE, "foundation_share.json")))["sizes"][SIZE]["plates"]
PLATES = [f"{nm}.{l}" for l in range(L) for nm in ("qkv", "proj", "fc", "mproj")]
TABLES = ["head", "embed"]
t0 = time.time()
LOGF = open(os.path.join(HERE, f"remove_{SIZE}{('-' + A['tag']) if A.get('tag') else ''}{'-shuffle' if SHUF else ''}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes

# ---------------------------------------------------------------- the machine, copied unchanged from pythia_base2.py
def rshift(X, s):
    s = np.asarray(s, dtype=np.int64)
    if (s < 0).any(): assert np.abs(X).max(initial=0) < 2 ** (62 + int(np.minimum(s, 0).min()))
    up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)

_shift_py = np.frompyfunc(lambda x, s: (x << -s) if s <= 0 else ((x + (1 << (s - 1))) >> s), 2, 1)

def to16(X):
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(X), axis=1)])
    sh = np.maximum(top - 15, 0)
    return rshift(X, sh[:, None]), sh - ES

def product(x16, w, wmax):
    assert float(np.abs(x16).sum(1).max()) * wmax < 2 ** 53
    with np.errstate(all="ignore"):
        p = x16.astype(np.float64) @ w
    assert np.isfinite(p).all() and (p == np.round(p)).all()
    return p.astype(np.int64)

def log2fix(v):
    v = int(v); t = v.bit_length() - 1
    m = (v >> (t - 16)) & 0xFFFF if t >= 16 else (v << (16 - t)) & 0xFFFF
    return t * 65536 + int(LOG2[m])

def power(X, F):
    F = np.broadcast_to(np.asarray(F, dtype=np.int64), X.shape); I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF)
    if np.abs(X).max(initial=0) < 2 ** 45: return rshift(X * EXP2[f], 16 - I)
    return _shift_py(X.astype(object) * EXP2[f].astype(object), (16 - I).astype(object)).astype(np.int64)

def race(vals):
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)
    alive = np.ones(len(u), dtype=bool)
    for bit in range(63, -1, -1):
        has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        if has.any(): alive = has
    return int(np.flatnonzero(alive)[0])

NORM = {}
for l in range(L):
    for nm in ("ln1", "ln2"):
        k = f"{nm}.{l}"; NORM[k] = (M[k + ".g"].astype(np.int64)[PERM], int(M[k + ".ge"]), M[k + ".b"].astype(np.int64)[PERM])
NORM["lnf"] = (M["lnf.g"].astype(np.int64)[PERM], int(M["lnf.ge"]), M["lnf.b"].astype(np.int64)[PERM])

def gauge(R, name):
    g, ge, b = NORM[name]
    x16, ex = to16(R)
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    c = d * x16 - S1[:, None]
    V = d * S2 - S1 * S1 + np.array([EPS40 >> (40 + 2 * int(e)) for e in ex])
    LS = np.array([log2fix(v) >> 1 for v in V])
    return power(c * g[None, :], (ge + ES) * 65536 - LS[:, None]) + b

PL = {}       # name -> (float64 plate, e, b, wmax) for plates and head; "embed" -> (int64 digits [words][lanes], e)
CUR = {}      # name -> which form is loaded now

def plain(X, name):
    w, e, b, wmax = PL[name]; x16, ex = to16(X)
    return rshift(product(x16, w, wmax), -(e[None, :] + ex[:, None] + ES)) + b

def turn(X, pos):
    r = ROT // 2; a, b = X[:, :r], X[:, r:ROT]
    assert np.abs(X[:, :ROT]).max() < 2 ** 31
    c, s = COS[pos], SIN[pos]
    return np.concatenate([rshift(a * c - b * s, 30), rshift(b * c + a * s, 30), X[:, ROT:]], 1)

def attention(QKV, rec=None):
    P = len(QKV); pos = np.arange(P); out = np.zeros((P, d), np.int64); live = np.tril(np.ones((P, P), dtype=bool))
    win = np.zeros((NH, P), np.int64); fs = np.zeros((NH, P))
    for h in range(NH):
        o = h * 3 * HS; q, k, v = QKV[:, o:o + HS], QKV[:, o + HS:o + 2 * HS], QKV[:, o + 2 * HS:o + 3 * HS]
        q, k = turn(q, pos), turn(k, pos)
        q16, eq = to16(q); k16, ek = to16(k)
        sc = power(product(q16, k16.T.astype(np.float64), 2.0 ** 15), (eq[:, None] + ek[None, :] + SC) * 65536 - HALF)
        lev = np.where(live, sc, np.iinfo(np.int64).min).max(1)
        win[h] = np.where(live & (sc == lev[:, None]), pos[None, :], P).min(1)   # who sets the water level (first at it)
        dd = lev[:, None] - sc
        a = np.where(live & (dd < len(ARR)), ARR[np.clip(dd, 0, len(ARR) - 1)], 0)
        if rec is not None: fs[h] = ARR[0] / a.sum(1)                    # a look, not the machine: the first arrival's share of the row
        assert float(np.abs(v).max(initial=0)) * float(a.sum(1).max()) < 2 ** 62
        out[:, h * HS:(h + 1) * HS] = power(a @ v, -np.array([log2fix(t) for t in a.sum(1)])[:, None])
    if rec is not None: rec["wl"].append(win); rec["fs"].append(fs)
    return out

def edge(H, rec=None):
    if rec is not None: rec["ke"].append(H > 0)                         # the knife edge's side: open above 0
    assert np.abs(H).max(initial=0) < 2 ** 46                            # H x share stays inside int64
    idx = rshift(H, ES - 12) + 32768
    share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
    return rshift(H * share, 16)

def forward(ids, last=1, rec=None):
    ids = np.asarray(ids); EW, EE = PL["embed"]
    R = rshift(EW[ids].astype(np.int64)[:, PERM], -(EE[ids][:, None] + ES))
    for l in range(L):
        At = plain(attention(plain(gauge(R, f"ln1.{l}"), f"qkv.{l}"), rec), f"proj.{l}")
        Mo = plain(edge(plain(gauge(R, f"ln2.{l}"), f"fc.{l}"), rec), f"mproj.{l}")
        R = R + At + Mo
    y16, ey = to16(gauge(R[-last:], "lnf")); w, e, _, wmax = PL["head"]
    lg = rshift(product(y16, w, wmax), -(e[None, :] + ey[:, None] + ES))
    return [race(row) for row in lg], lg

# ---------------------------------------------------------------- the plates for each case
def to30(s): return int(round(s * 2 ** 30))                     # s as a whole number at 2^-30 (as foundation_share.py)
def times(s30, q): return (s30 * q + (1 << 29)) >> 30            # round(s x q), whole numbers only

LOST = {}
def shuffled_start(name):
    """W0 with its values scrambled over the whole plate. A value keeps its size: digit x 2^(its own neuron's power),
    written on the grid of the neuron it lands in (exact where that grid is coarser-or-equal in places kept; rounded to
    nearest where it would lose places, counted in LOST)."""
    S = M[name + ".s"]; e = M[name + ".e"].astype(np.int8)
    E = np.broadcast_to(e[:, None] if name == "embed" else e[None, :], S.shape).ravel()
    rng = np.random.default_rng(SEED + zlib.crc32(name.encode()))
    P = rng.permutation(S.size); Sf = S.ravel(); out = np.empty(S.size, np.int32); lost = 0
    for a in range(0, S.size, 1 << 22):
        b = min(S.size, a + (1 << 22)); src = Sf[P[a:b]].astype(np.int64)
        sh = E[P[a:b]].astype(np.int64) - E[a:b].astype(np.int64)        # source power minus destination power
        r = rshift(src, -sh)                                             # left shift where sh >= 0, rounded right shift else
        lost += int(((sh < 0) & (rshift(r, sh) != src)).sum())
        out[a:b] = r
    LOST[name] = lost
    return out.reshape(S.shape)

def form(case, name):
    """which form plate `name` takes in `case`: W, D (W - W0), Dm (W - round(s W0)), Ds, Dms (the same, W0 shuffled)"""
    if case == "fin": return "W"
    kind, _, part = case.partition("_")
    f = {"a": "D", "b": "Dm", "ca": "Ds", "cb": "Dms"}[kind]
    if part == "": return f
    if part == "tab": return f if name in TABLES else "W"
    if part == "pl": return f if name in PLATES else "W"
    if part == "head": return f if name == "head" else "W"                # added after the first runs: one table alone
    if part == "emb": return f if name == "embed" else "W"
    if part.startswith("L"): return f if name in PLATES and name.split(".")[1] == part[1:] else "W"
    raise ValueError(case)

def load(name, f):
    if CUR.get(name) == f: return
    PL.pop(name, None)
    W = M[name + ".w"]
    S = None if f == "W" else (M[name + ".s"] if f in ("D", "Dm") else shuffled_start(name))
    s30 = to30(SHARE[name]["s"]) if f in ("Dm", "Dms") else None
    out = np.empty(W.shape, np.int32 if name == "embed" else np.float64)
    for a in range(0, W.shape[0], max(1, (1 << 22) // W.shape[1])):
        b = min(W.shape[0], a + max(1, (1 << 22) // W.shape[1])); w = W[a:b].astype(np.int64)
        if f == "W": out[a:b] = w
        elif f in ("D", "Ds"): out[a:b] = w - S[a:b].astype(np.int64)
        else: out[a:b] = w - times(s30, S[a:b].astype(np.int64))
    del S
    e = M[name + ".e"].astype(np.int64)
    if name == "embed": PL[name] = (out, e)
    elif name == "head": PL[name] = (out[PERM] if SHUF else out, e, 0, float(np.abs(out).max()))
    else:
        b = M[name + ".b"].astype(np.int64)
        if name.startswith(("qkv", "fc")): out = out[PERM] if SHUF else out
        else: out, e, b = (out[:, PERM], e[PERM], b[PERM]) if SHUF else (out, e, b)
        PL[name] = (out, e, b, float(np.abs(out).max()))
    CUR[name] = f

def set_case(case):
    for name in TABLES + PLATES: load(name, form(case, name))

# ---------------------------------------------------------------- the runs
refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))[:TEXTS]
oldw = json.load(open(os.path.join(HERE, f"words_pythia-{SIZE}.json")))["texts_detail"]
CASES = A.get("cases", "all")
if CASES == "all":
    CASES = ["fin", "a", "b", "ca", "cb", "a_tab", "b_tab", "ca_tab", "cb_tab", "a_pl", "b_pl", "ca_pl", "cb_pl"] + \
            [f"{k}_L{l}" for l in range(L) for k in ("a", "b", "ca", "cb")]
else:
    CASES = CASES.split(",")
    if CASES[0] != "fin": CASES = ["fin"] + CASES
from tokenizers import Tokenizer                                         # the word list only (no torch in this process)
tok = Tokenizer.from_file(os.path.join(HERE, "models", "tokenizer", "tokenizer.json"))
log(f"remove_foundation.py  size {SIZE}  order {ORDER}  texts {TEXTS}  greedy {GREEDY}  cases {len(CASES)}")

def top5(row):
    return np.lexsort((np.arange(len(row)), -row))[:5]                  # largest piles first, lower word first on a tie

FIN = None; RES = {}
for case in CASES:
    tc = time.time(); set_case(case)
    per = []
    for t, r in enumerate(refs):
        seq = r["ids"] + r["ref"][:WORDS - 1]; n0 = len(r["ids"]) - 1    # position n0 + j gives continuation word j
        rec = dict(wl=[], ke=[], fs=[])
        got, lg = forward(seq, last=len(seq), rec=rec)
        t5 = np.array([top5(row) for row in lg])
        assert (t5[:, 0] == np.array(got)).all()                         # the sort's first word is the race's winner
        srt = np.sort(lg, axis=1)[:, -2:]; lead = ((srt[:, 1] - srt[:, 0]) / 2.0 ** ES)
        x = dict(top1=got, top5=t5.tolist(), lead=lead.tolist(), wl=rec["wl"], ke=rec["ke"], fs=rec["fs"],
                 sha_all=hashlib.sha256(np.ascontiguousarray(lg).tobytes()).hexdigest(),
                 sha_last=hashlib.sha256(np.ascontiguousarray(lg[-WORDS:]).tobytes()).hexdigest(), n0=n0)
        if GREEDY:
            cont = got[n0:n0 + WORDS]; bad = [j for j in range(WORDS) if cont[j] != r["ref"][j]]
            own = list(r["ref"]) if not bad else r["ref"][:bad[0]] + [cont[bad[0]]]
            while len(own) < WORDS: own.append(forward(r["ids"] + own)[0][0])
            x["greedy"] = own[:WORDS]
        per.append(x)
    if case == "fin":
        FIN = per
        ok_ref = sum(int(a == b) for x, r in zip(per, refs) for a, b in zip(x["top1"][x["n0"]:x["n0"] + WORDS], r["ref"]))
        ok_sha = sum(x["sha_last"] == o["piles_sha256"] for x, o in zip(per, oldw)) if not SHUF else None
        log(f"  fin: the finished machine's own word = fp32's at {ok_ref} of {TEXTS * WORDS} continuation positions; "
            f"head piles of the last {WORDS} positions identical (sha256) to pythia_base2.py's run in "
            f"{ok_sha} of {TEXTS} texts" if not SHUF else f"  fin (order shuffle): words = fp32's at {ok_ref}")
        # statistic controls with the pairing broken: finished text t against finished text t+1
        wlc = np.zeros((L, 2)); kec = np.zeros((L, 2))
        for t in range(TEXTS):
            u, v = per[t], per[(t + 1) % TEXTS]; P = min(len(u["top1"]), len(v["top1"]))
            for l in range(L):
                wlc[l] += [(u["wl"][l][:, 1:P] != v["wl"][l][:, 1:P]).sum(), NH * (P - 1)]
                kec[l] += [(u["ke"][l][:P] != v["ke"][l][:P]).sum(), u["ke"][l][:P].size]
        RES["control_finished_vs_next_text"] = dict(wl_moved=(wlc[:, 0] / wlc[:, 1]).round(4).tolist(),
                                                    ke_flipped=(kec[:, 0] / kec[:, 1]).round(4).tolist())
        log(f"  pairing-broken control (finished text t vs text t+1, same row): water level moved per layer "
            f"{RES['control_finished_vs_next_text']['wl_moved']}; knife edges changed side "
            f"{RES['control_finished_vs_next_text']['ke_flipped']}")
    # ---- against the finished machine
    n_all = t1 = t5c = 0; c1 = c5 = n_c = 0; k1 = k5 = 0; gw = gt = 0; first = []
    kept_lead, lost_lead = [], []
    wl = np.zeros((L, 2)); ke = np.zeros((L, 2)); fs_mv = [[] for _ in range(L)]; fs_hd = [[] for _ in range(L)]
    setter = np.zeros((L, 4))                                            # who sets the level: first word / the row itself / the row before / other
    for t in range(TEXTS):
        x, f = per[t], FIN[t]; nx = FIN[(t + 1) % TEXTS]
        for p in range(len(x["top1"])):
            n_all += 1; same = x["top1"][p] == f["top1"][p]; t1 += same; t5c += f["top1"][p] in x["top5"][p]
            (kept_lead if same else lost_lead).append(f["lead"][p])
        for j in range(WORDS):
            p, q = x["n0"] + j, nx["n0"] + j; n_c += 1
            c1 += x["top1"][p] == f["top1"][p]; c5 += f["top1"][p] in x["top5"][p]
            k1 += x["top1"][p] == nx["top1"][q]; k5 += nx["top1"][q] in x["top5"][p]
        if GREEDY:
            eq = [a == b for a, b in zip(x["greedy"], refs[t]["ref"])]; gw += sum(eq); gt += all(eq)
            first.append(eq.index(False) if not all(eq) else WORDS)
        for l in range(L):
            wl[l] += [(x["wl"][l][:, 1:] != f["wl"][l][:, 1:]).sum(), x["wl"][l][:, 1:].size]
            ke[l] += [(x["ke"][l] != f["ke"][l]).sum(), x["ke"][l].size]
            mv = x["wl"][l][:, 1:] != f["wl"][l][:, 1:]; fsh = f["fs"][l][:, 1:]
            wv = x["wl"][l][:, 1:]; row = np.arange(1, wv.shape[1] + 1)[None, :]
            q0 = wv == 0; q1 = ~q0 & (wv == row); q2 = ~q0 & ~q1 & (wv == row - 1)
            setter[l] += [q0.sum(), q1.sum(), q2.sum(), (~q0 & ~q1 & ~q2).sum()]
            fs_mv[l] += fsh[mv].tolist(); fs_hd[l] += fsh[~mv].tolist()
    o = dict(top1_all=t1 / n_all, top5_all=t5c / n_all, positions=n_all, top1_cont=c1 / n_c, top5_cont=c5 / n_c,
             control_top1_cont=k1 / n_c, control_top5_cont=k5 / n_c,
             greedy_words=gw, greedy_texts=gt, greedy_first_diff_mean=float(np.mean(first)) if GREEDY else None,
             fin_lead_median_kept=float(np.median(kept_lead)) if kept_lead else None,
             fin_lead_median_lost=float(np.median(lost_lead)) if lost_lead else None,
             fs_moved=[round(float(np.median(v)), 4) if v else None for v in fs_mv],
             fs_held=[round(float(np.median(v)), 4) if v else None for v in fs_hd],
             setter=(setter / setter.sum(1, keepdims=True)).round(3).tolist(),
             wl_moved=(wl[:, 0] / wl[:, 1]).round(4).tolist(), ke_flipped=(ke[:, 0] / ke[:, 1]).round(4).tolist(),
             sha_all=[x["sha_all"] for x in per], top1=[x["top1"] for x in per],
             greedy=[x["greedy"] for x in per] if GREEDY else None, seconds=round(time.time() - tc, 1))
    RES[case] = o
    log(f"  {case:8s} top-1 {o['top1_all']*100:5.1f}%  top-5 {o['top5_all']*100:5.1f}%  (of {n_all});  "
        f"16 continuation: top-1 {o['top1_cont']*100:5.1f}% top-5 {o['top5_cont']*100:5.1f}% [pairing-broken {o['control_top1_cont']*100:4.1f}% / {o['control_top5_cont']*100:4.1f}%];  "
        + (f"greedy words {gw}/{TEXTS * WORDS} texts {gt}/{TEXTS} first diff {o['greedy_first_diff_mean']:.1f};  " if GREEDY else "")
        + f"finished lead kept/lost {o['fin_lead_median_kept'] or 0:.2f}/{(o['fin_lead_median_lost'] or 0):.2f};  "
        f"{o['seconds']}s  peak {mem():.2f} GB")
    log(f"  {'':8s} water level moved by layer {o['wl_moved']}")
    log(f"  {'':8s} knife edges changed side by layer {o['ke_flipped']}")
    log(f"  {'':8s} who sets the level, by layer [first word, the row itself, the row before, other]: {o['setter']}")
    log(f"  {'':8s} the finished machine's first arrival's share of its row (median), rows whose level moved {o['fs_moved']} / held {o['fs_held']}")
    if GREEDY and case != "fin":
        for t in [u for u in (0, 6, 20) if u < TEXTS]:
            log(f"  {'':8s}   {refs[t]['prompt']!r} -> {tok.decode(per[t]['greedy'])!r}")
    if case != "fin":
        for x in per: x.pop("wl"); x.pop("ke"); x.pop("fs")
RES["lost_places_in_shuffle"] = LOST
RES["s30"] = {k: to30(v["s"]) for k, v in SHARE.items()}
json.dump(dict(size=SIZE, order=ORDER, texts=TEXTS, words=WORDS, cases=RES), open(LOGF.name[:-4] + ".json", "w"))
log(f"done {time.time() - t0:.0f}s  peak memory {mem():.2f} GB")
