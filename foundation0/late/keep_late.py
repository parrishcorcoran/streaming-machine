"""KEEP: the surviving foundation kept only on the plates the look found carrying the late water level, removed from
every other plate and from both word tables. 2026-10-06.

HIS SENTENCES (his exact words, the only ones here that are his): "It should be 100% removable after training."
  "Any failure is not a pivot."
SESSION'S RESTATEMENT (the session's words, not his), as in ../remove_foundation.py and plate_look.py: the finished
plate W = what survives of the random start, round(s x W0) with one s per plate (../foundation_share.json), + what was
built on it. "Removing the foundation" here means W - round(s x W0), on the shared whole-number grid.

WHAT THE LOOK FOUND (plate_look_<size>.log, one plate at a time, three scrambled draws each): only qkv.l can move layer
l's level (wiring); each late layer's level is moved by its own qkv (160m qkv.8-11: 25, 42, 54, 61% of rows at its own
layer; 70m qkv.4-5: 60, 67%); every other plate moves any late layer's level by at most 7% (160m: fc.0, qkv.8 at
layer 9) or 11% (70m: qkv.3); at the late level the real foundation is no more load-bearing than a scramble.

THE KEEP SET, fixed here before any case of this script runs, from the look's level numbers only (no word count was
read to choose it): K = { qkv.l : l in the look's late layers }, the last third of the layers as plate_look.py defines
them (LATEL = range(2L//3, L)):  160m  qkv.8, qkv.9, qkv.10, qkv.11      70m  qkv.4, qkv.5
  The rule, as a check the script runs on plate_look_<size>.json before anything else (it stops if the set differs):
  every plate in K moves some late layer's level, with its surviving foundation alone removed, in more rows than any
  plate outside K moves any late layer (160m: smallest in K 24.5% against largest outside 7%; 70m: 59.9% against 11%).
Everything else loses its surviving foundation: 160m 44 of 48 plates, 70m 22 of 24, and both word tables.
Share of weights whose foundation is removed (a count fixed by the set, not a finding):
  160m  plates 84,934,656 values, kept 7,077,888 (8.33%); with both tables 162,201,600, removed 155,123,712 (95.64%)
  70m   plates 18,874,368 values, kept 1,572,864 (8.33%); with both tables  70,385,664, removed  68,812,800 (97.77%)
  The script also prints the share of the surviving foundation's own size (sum of (s w0)^2) that is removed.

CASES (the base-2 machine of ../pythia_base2.py, its parts copied unchanged through plate_look.py; same 30 texts, same
fed-in words: each text's own words, then the finished machine's 16 greedy words; and each case's own 16 greedy words):
  fin       the finished machine (head piles must match pythia_base2.py's run by sha256, 30 of 30 texts)
  b         every plate and both tables lose round(s W0) (a reproduction: top-1 lists must equal ../remove_<size>.json's b)
  k         THE TEST: K keeps its surviving foundation; every other plate and both word tables lose round(s W0)
  k_pl      diagnostic: as k, but the two word tables untouched (splits the tables' share of k's loss)
  ck@n      CONTROL, scrambled start: as k, with W0 scrambled within each plate (same values, positions moved), draws
            n = 20261006, 2, 3 (the same scrambles ../remove_foundation.py and plate_look.py made for those seeds)
  rk@n      CONTROL, pairing of "kept" and "the look's plates" broken: the same number of plates (4 at 160m, 2 at 70m)
            keep their surviving foundation, drawn at random from all plates (draws n = 1, 2, 3, numpy default_rng(
            20261006 + 1000 n)); every other plate and both word tables lose round(s W0)
Measured for each case against the finished machine: top-1 and top-5 (the finished word inside the case's top 5) at all
622 positions; the 16 continuation positions against the NEXT text's finished words (statistic control, pairing
broken); the case's own 16 greedy words and texts identical; rows whose water level moves, at every layer; the text's
first word's share of the levels; knife edges that change side, at every layer; the finished lead of word 1 over word 2
(median) at positions kept and lost.
Wiring facts (a check, not a finding; rule 13.4): k and b remove the same plates up to the first late layer, so k's
levels at the layers before it and k's knife edges up to and including it must equal b's row for row.

PREDICTIONS, written before the first run of this script (results go to keep_late_<size>.log / .json, never here).
  Basis: the look's single plates and kinds combine close to "losses add as squares" (160m qkv kind 28.0 points of
  top-1 lost against 26.9 from its twelve single plates; fc 28.5 against 25.6); every plate and table at once (b) lost
  49.5 (160m) and 47.4 (70m). Taking K's share out of b's: 160m about 44 points lost, 70m about 39.
  (1) the late level mostly holds: rows whose level moves at 160m layers 8-11, each 8-28% (b: 36, 52, 56, 66); 70m
      layers 4-5, each 10-30% (b: 59, 66). The text's first word's share of the levels at those layers within 6 points
      of the finished machine's (160m 61.2, 65.8, 70.5, 53.4; 70m 57.3, 32.9).
  (2) the words are still lost: k top-1 160m 53-63%, 70m 56-68%; top-5 160m 82-90%, 70m 85-92%; greedy texts identical
      0-2 of 30 at each size, greedy words 55-150 of 480. Keeping the level keeps 4-12 points of top-1 over b, not the
      words: the loss outside K runs through the bend's feed (fc kind alone 28.5 / 21.1 points), the bend's write
      (mproj 12.9 / 14.6) and the output word table (b_head 14.1 / 10.0), none of which moves the late level much.
  (3) k_pl 2-5 points of top-1 above k.
  (4) scrambled starts: at 160m k below every ck draw by 4-10 points (the real-foundation excess sat in the fc and
      mproj kinds, all removed here); at 70m k inside the ck spread or within 3 points of it.
  (5) random keep sets: top-1 160m 50-58%, 70m 52-60% (a draw holding a late qkv may sit higher); the late level moves
      as in b (160m layers 9-11 above 40%, 70m layers 4-5 above 45%) at every draw that keeps no late qkv. k keeps the
      late level better than every such draw; on words k above the rk draws by 2-8 points.
  (6) knife edges: k's equal b's exactly up to the first late layer (wiring); after it within 2 points of b's.
  (7) order-free check (order=shuffle, the stream's lanes in a random order): k's head piles identical in 30 of 30
      texts at 160m.
  (8) independent rebuild (keep_checker.py, fp32 Pythia from its own fp16 files, no shared code): checker word =
      base-2 k's word at 95% or more of positions; the share of finished words kept within 3 points.
His sentence holds for this set if k keeps every word (100%, 30 of 30 greedy texts); near-holding at top-1 95% or more
with 25 or more of 30 greedy texts.

  python3 keep_late.py size=160m [cases=all|<list>] [order=plain|shuffle] [greedy=1] [texts=30] [tag=]
"""
import os, sys, json, math, time, hashlib, zlib, resource
import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE, ORDER = A.get("size", "160m"), A.get("order", "plain")
GREEDY = A.get("greedy", "1") == "1"
SEEDS = [20261006, 2, 3]
RDRAWS = [1, 2, 3]
TEXTS, WORDS = int(A.get("texts", 30)), int(A.get("words", 16))
LATE = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LATE)
M = np.load(os.path.join(HERE, f"base2_pythia-{SIZE}.npz"))
d, L, NH, HS, ROT = (int(M[k]) for k in ("d", "L", "heads", "hs", "rot"))
ES, SC = 16, 10
PHI, ARR, LOG2, EXP2, COS, SIN = (M[k] for k in ("phi", "arrival", "log2", "exp2", "cos", "sin"))
EPS40, HALF = int(M["eps40"]), int(M["half_log2_hs"])
PERM = np.random.default_rng(12345).permutation(d) if ORDER == "shuffle" else np.arange(d)
SHUF = ORDER == "shuffle"
SHARE = json.load(open(os.path.join(HERE, "foundation_share.json")))["sizes"][SIZE]["plates"]
KINDS = ("qkv", "proj", "fc", "mproj")
PLATES = [f"{nm}.{l}" for l in range(L) for nm in KINDS]
TABLES = ["head", "embed"]
LATEL = list(range((2 * L) // 3, L))                                    # the last third of the layers (as plate_look.py)
t0 = time.time()
STEM = os.path.join(LATE, f"keep_late_{SIZE}{('-' + A['tag']) if A.get('tag') else ''}{'-shuffle' if SHUF else ''}")
LOGF = open(STEM + ".log", "w")
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
CUR = {}      # name -> (form, seed) loaded now

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
    win = np.zeros((NH, P), np.int64); gap = np.full((NH, P), -1, np.int64)
    for h in range(NH):
        o = h * 3 * HS; q, k, v = QKV[:, o:o + HS], QKV[:, o + HS:o + 2 * HS], QKV[:, o + 2 * HS:o + 3 * HS]
        q, k = turn(q, pos), turn(k, pos)
        q16, eq = to16(q); k16, ek = to16(k)
        sc = power(product(q16, k16.T.astype(np.float64), 2.0 ** 15), (eq[:, None] + ek[None, :] + SC) * 65536 - HALF)
        lev = np.where(live, sc, np.iinfo(np.int64).min).max(1)
        win[h] = np.where(live & (sc == lev[:, None]), pos[None, :], P).min(1)   # who sets the water level (first at it)
        if rec is not None and P > 1:                                    # a look, not the machine: the next score below the level
            srt = np.sort(np.where(live, sc, np.iinfo(np.int64).min), axis=1)
            gap[h, 1:] = lev[1:] - srt[1:, -2]
        dd = lev[:, None] - sc
        a = np.where(live & (dd < len(ARR)), ARR[np.clip(dd, 0, len(ARR) - 1)], 0)
        assert float(np.abs(v).max(initial=0)) * float(a.sum(1).max()) < 2 ** 62
        out[:, h * HS:(h + 1) * HS] = power(a @ v, -np.array([log2fix(t) for t in a.sum(1)])[:, None])
    if rec is not None: rec["wl"].append(win); rec["gap"].append(gap)
    return out

def edge(H, rec=None):
    if rec is not None: rec["ke"].append(H > 0)                         # the knife edge's side: open above 0
    assert np.abs(H).max(initial=0) < 2 ** 46
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

# ---------------------------------------------------------------- the plates for each case (as ../remove_foundation.py)
def to30(s): return int(round(s * 2 ** 30))
def times(s30, q): return (s30 * q + (1 << 29)) >> 30

LOST = {}
def shuffled_start(name, seed):
    """W0 with its values scrambled over the whole plate, exactly as ../remove_foundation.py does for the same seed."""
    S = M[name + ".s"]; e = M[name + ".e"].astype(np.int8)
    E = np.broadcast_to(e[:, None] if name == "embed" else e[None, :], S.shape).ravel()
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
    P = rng.permutation(S.size); Sf = S.ravel(); out = np.empty(S.size, np.int32); lost = 0
    for a in range(0, S.size, 1 << 22):
        b = min(S.size, a + (1 << 22)); src = Sf[P[a:b]].astype(np.int64)
        sh = E[P[a:b]].astype(np.int64) - E[a:b].astype(np.int64)
        r = rshift(src, -sh)
        lost += int(((sh < 0) & (rshift(r, sh) != src)).sum())
        out[a:b] = r
    LOST[f"{name}@{seed}"] = lost
    return out.reshape(S.shape)


# ---------------------------------------------------------------- the keep set, from the look's level numbers (before any case)
LOOK = json.load(open(os.path.join(LATE, f"plate_look_{SIZE}.json")))["cases"]
KEEP = [f"qkv.{l}" for l in LATEL]
late_max = {p: max(LOOK[f"b:{p}"]["wl_moved"][m] for m in LATEL) for p in PLATES}
inside = min(late_max[p] for p in KEEP); outside = max(late_max[p] for p in PLATES if p not in KEEP)
worst = max((p for p in PLATES if p not in KEEP), key=lambda p: late_max[p])
own_at = {p: int(np.argmax(LOOK[f"b:{p}"]["wl_moved"])) for p in KEEP}
log(f"keep_late.py  size {SIZE}  order {ORDER}  texts {TEXTS}  greedy {GREEDY}  late layers {LATEL}")
log(f"  keep set K = {KEEP}: each moves its own late layer's level in "
    f"{[round(100 * LOOK[f'b:{p}']['wl_moved'][int(p.split('.')[1])], 1) for p in KEEP]}% of rows (the look, b:<plate>)")
log(f"  rule check: smallest late move inside K {100 * inside:.1f}% > largest late move outside K {100 * outside:.1f}% "
    f"({worst}); each K plate's largest move is at its own layer: {all(own_at[p] == int(p.split('.')[1]) for p in KEEP)}")
assert inside > outside and all(own_at[p] == int(p.split(".")[1]) for p in KEEP), "the look's rule does not give K"
RK = {}
for n in RDRAWS:
    RK[n] = sorted(np.random.default_rng(20261006 + 1000 * n).choice(PLATES, size=len(KEEP), replace=False).tolist(),
                   key=PLATES.index)
    log(f"  random keep draw rk@{n}: {RK[n]}  (holds a late qkv: {any(p in KEEP for p in RK[n])})")

def keepset(case):
    if case in ("k", "k_pl") or case.startswith("ck@"): return set(KEEP)
    if case.startswith("rk@"): return set(RK[int(case[3:])])
    return set()

def form(case, name):
    if case == "fin" or name in keepset(case): return ("W", None)
    if case == "k_pl" and name in TABLES: return ("W", None)
    if case.startswith("ck@"): return ("Dms", int(case[3:]))
    return ("Dm", None)

# the count of weights and the surviving foundation's own size (sum of (s w0)^2 in real numbers), per plate and table
COUNT, RELIEF = {}, {}
for name in TABLES + PLATES:
    S = M[name + ".s"]; e = M[name + ".e"].astype(np.float64); COUNT[name] = int(S.size)
    sq = (S.astype(np.float64) ** 2).sum(1 if name == "embed" else 0)
    RELIEF[name] = float((sq * 4.0 ** e).sum()) * SHARE[name]["s"] ** 2; del S, sq

def shares(case):
    rem = [n for n in TABLES + PLATES if form(case, n)[0] != "W"]
    return dict(all=sum(COUNT[n] for n in rem) / sum(COUNT.values()),
                plates=sum(COUNT[n] for n in rem if n in PLATES) / sum(COUNT[n] for n in PLATES),
                relief=sum(RELIEF[n] for n in rem) / sum(RELIEF.values()),
                relief_plates=sum(RELIEF[n] for n in rem if n in PLATES) / sum(RELIEF[n] for n in PLATES),
                removed_values=sum(COUNT[n] for n in rem), all_values=sum(COUNT.values()))

def load(name, fs):
    if CUR.get(name) == fs: return
    f, seed = fs
    PL.pop(name, None)
    W = M[name + ".w"]
    S = None if f == "W" else (M[name + ".s"] if f == "Dm" else shuffled_start(name, seed))
    s30 = to30(SHARE[name]["s"]) if f in ("Dm", "Dms") else None
    out = np.empty(W.shape, np.int32 if name == "embed" else np.float64)
    for a in range(0, W.shape[0], max(1, (1 << 22) // W.shape[1])):
        b = min(W.shape[0], a + max(1, (1 << 22) // W.shape[1])); w = W[a:b].astype(np.int64)
        if f == "W": out[a:b] = w
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
    CUR[name] = fs

def set_case(case):
    for name in TABLES + PLATES: load(name, form(case, name))


# ---------------------------------------------------------------- the runs
refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))[:TEXTS]
oldw = json.load(open(os.path.join(HERE, f"words_pythia-{SIZE}.json")))["texts_detail"]
CA = A.get("cases", "all")
ALL = ["b", "k", "k_pl"] + [f"ck@{s}" for s in SEEDS] + [f"rk@{n}" for n in RDRAWS]
CASES = ALL if CA == "all" else CA.split(",")
if CASES[0] != "fin": CASES = ["fin"] + CASES
log(f"  cases {CASES}")

def top5(row):
    return np.lexsort((np.arange(len(row)), -row))[:5]

def med(v): return round(float(np.median(v)), 3) if len(v) else None

FIN = None; RES = {}; BREC = None
for case in CASES:
    tc = time.time(); set_case(case)
    per = []
    for t, r in enumerate(refs):
        seq = r["ids"] + r["ref"][:WORDS - 1]; n0 = len(r["ids"]) - 1
        rec = dict(wl=[], ke=[], gap=[])
        got, lg = forward(seq, last=len(seq), rec=rec)
        t5 = np.array([top5(row) for row in lg])
        assert (t5[:, 0] == np.array(got)).all()
        srt = np.sort(lg, axis=1)[:, -2:]; lead = ((srt[:, 1] - srt[:, 0]) / 2.0 ** ES)
        x = dict(top1=got, top5=t5.tolist(), wl=rec["wl"], ke=rec["ke"], lead=lead.tolist(),
                 sha_all=hashlib.sha256(np.ascontiguousarray(lg).tobytes()).hexdigest(),
                 sha_last=hashlib.sha256(np.ascontiguousarray(lg[-WORDS:]).tobytes()).hexdigest(), n0=n0)
        if GREEDY and case != "fin":
            cont = got[n0:n0 + WORDS]; bad = [j for j in range(WORDS) if cont[j] != r["ref"][j]]
            own = list(r["ref"]) if not bad else r["ref"][:bad[0]] + [cont[bad[0]]]
            while len(own) < WORDS: own.append(forward(r["ids"] + own)[0][0])
            x["greedy"] = own[:WORDS]
        elif case == "fin": x["greedy"] = list(r["ref"])
        per.append(x)
    if case == "fin":
        FIN = per
        ok_ref = sum(int(a == b) for x, r in zip(per, refs) for a, b in zip(x["top1"][x["n0"]:x["n0"] + WORDS], r["ref"]))
        ok_sha = sum(x["sha_last"] == o["piles_sha256"] for x, o in zip(per, oldw)) if not SHUF else None
        log(f"  fin: words = fp32's at {ok_ref} of {TEXTS * WORDS}; head piles identical (sha256) to pythia_base2.py's "
            f"run in {ok_sha} of {TEXTS} texts" if not SHUF else f"  fin (order shuffle): words = fp32's at {ok_ref}")
        wlc = np.zeros((L, 2)); kec = np.zeros((L, 2))
        for t in range(TEXTS):
            u, v = per[t], per[(t + 1) % TEXTS]; P = min(len(u["top1"]), len(v["top1"]))
            for l in range(L):
                wlc[l] += [(u["wl"][l][:, 1:P] != v["wl"][l][:, 1:P]).sum(), NH * (P - 1)]
                kec[l] += [(u["ke"][l][:P] != v["ke"][l][:P]).sum(), u["ke"][l][:P].size]
        RES["control_finished_vs_next_text"] = dict(wl_moved=(wlc[:, 0] / wlc[:, 1]).round(4).tolist(),
                                                    ke_flipped=(kec[:, 0] / kec[:, 1]).round(4).tolist())
        log(f"  pairing-broken control (finished text t vs t+1): level moved {RES['control_finished_vs_next_text']['wl_moved']}; "
            f"edges changed side {RES['control_finished_vs_next_text']['ke_flipped']}")
    n_all = t1 = t5c = 0; gw = gt = 0; c1 = c5 = k1 = k5 = n_c = 0; kept_lead, lost_lead = [], []
    wl = np.zeros((L, 2)); ke = np.zeros((L, 2)); setter = np.zeros((L, 4))
    for t in range(TEXTS):
        x, f, nx = per[t], FIN[t], FIN[(t + 1) % TEXTS]
        for p in range(len(x["top1"])):
            n_all += 1; same = x["top1"][p] == f["top1"][p]; t1 += same; t5c += f["top1"][p] in x["top5"][p]
            (kept_lead if same else lost_lead).append(f["lead"][p])
        for j in range(WORDS):
            p, q = x["n0"] + j, nx["n0"] + j; n_c += 1
            c1 += x["top1"][p] == f["top1"][p]; c5 += f["top1"][p] in x["top5"][p]
            k1 += x["top1"][p] == nx["top1"][q]; k5 += nx["top1"][q] in x["top5"][p]
        if "greedy" in x: eq = [a == b for a, b in zip(x["greedy"], refs[t]["ref"])]; gw += sum(eq); gt += all(eq)
        for l in range(L):
            mv = x["wl"][l][:, 1:] != f["wl"][l][:, 1:]
            wl[l] += [mv.sum(), mv.size]
            ke[l] += [(x["ke"][l] != f["ke"][l]).sum(), x["ke"][l].size]
            wv = x["wl"][l][:, 1:]; row = np.arange(1, wv.shape[1] + 1)[None, :]
            q0 = wv == 0; q1 = ~q0 & (wv == row); q2 = ~q0 & ~q1 & (wv == row - 1)
            setter[l] += [q0.sum(), q1.sum(), q2.sum(), (~q0 & ~q1 & ~q2).sum()]
    S = setter / setter.sum(1, keepdims=True); sh = shares(case)
    o = dict(top1_all=t1 / n_all, top5_all=t5c / n_all, positions=n_all, top1_cont=c1 / n_c, top5_cont=c5 / n_c,
             control_top1_cont=k1 / n_c, control_top5_cont=k5 / n_c, greedy_words=gw, greedy_texts=gt,
             fin_lead_median_kept=med(kept_lead), fin_lead_median_lost=med(lost_lead),
             wl_moved=(wl[:, 0] / wl[:, 1]).round(4).tolist(), ke_flipped=(ke[:, 0] / ke[:, 1]).round(4).tolist(),
             setter=S.round(4).tolist(), first_word=S[:, 0].round(4).tolist(), shares=sh,
             kept=sorted(keepset(case), key=PLATES.index), sha_all=[x["sha_all"] for x in per],
             top1=[x["top1"] for x in per], greedy=[x.get("greedy") for x in per], seconds=round(time.time() - tc, 1))
    if case == "b":
        BREC = [(x["wl"], x["ke"]) for x in per]
        if not SHUF:
            ref_b = json.load(open(os.path.join(HERE, f"remove_{SIZE}.json")))["cases"]["b"]["top1"]
            o["b_reproduces_remove_json"] = sum(x["top1"] == y for x, y in zip(per, ref_b))
            log(f"  b reproduction: top-1 lists equal to ../remove_{SIZE}.json's b in {o['b_reproduces_remove_json']} of {TEXTS} texts")
    if case in ("k", "k_pl") and BREC is not None:                      # wiring check: the same as b before the first late layer
        l0 = LATEL[0]
        same_wl = all((x["wl"][l] == bw[l]).all() for x, (bw, bk) in zip(per, BREC) for l in range(l0))
        same_ke = all((x["ke"][l] == bk[l]).all() for x, (bw, bk) in zip(per, BREC) for l in range(l0 + 1))
        o["wiring_same_as_b"] = dict(levels_before_late=bool(same_wl), edges_through_first_late=bool(same_ke))
        log(f"  wiring check ({case} against b): levels at layers 0-{l0 - 1} identical: {same_wl}; "
            f"knife edges at layers 0-{l0} identical: {same_ke}" + ("" if case == "k" else "  (k_pl: tables differ, not expected)"))
    RES[case] = o
    pc = lambda v: " ".join(f"{100 * a:3.0f}" for a in v)
    log(f"  {case:12s} top-1 {o['top1_all']*100:5.1f}  top-5 {o['top5_all']*100:5.1f}  greedy {gw:3d}/{TEXTS * WORDS} "
        f"texts {gt:2d}/{TEXTS}  | cont top-1 {o['top1_cont']*100:5.1f} [pairing-broken {o['control_top1_cont']*100:4.1f}] "
        f"| removed: {100*sh['all']:.2f}% of all weights, {100*sh['plates']:.2f}% of plate weights, "
        f"{100*sh['relief']:.2f}% of the foundation's size | lead kept/lost {o['fin_lead_median_kept']}/{o['fin_lead_median_lost']} "
        f"| {o['seconds']}s {mem():.2f} GB")
    log(f"  {'':12s} level moved {pc(o['wl_moved'])} | first word {pc(o['first_word'])} | edges {pc(o['ke_flipped'])}")
    if case != "fin":
        for x in per: x.pop("wl"); x.pop("ke")
    RES["lost_places_in_shuffle"] = LOST
    json.dump(dict(size=SIZE, order=ORDER, texts=TEXTS, words=WORDS, seeds=SEEDS, late=LATEL, keep=KEEP, rk=RK,
                   rule=dict(inside_min=inside, outside_max=outside, outside_worst=worst), cases=RES),
              open(STEM + ".json", "w"))

# ---------------------------------------------------------------- order-free check against the plain run
if SHUF and os.path.exists(os.path.join(LATE, f"keep_late_{SIZE}.json")):
    P0 = json.load(open(os.path.join(LATE, f"keep_late_{SIZE}.json")))["cases"]
    for c in CASES:
        if c in P0 and c in RES:
            log(f"  order-free check {c}: every head pile the same whole number in "
                f"{sum(a == b for a, b in zip(RES[c]['sha_all'], P0[c]['sha_all']))} of {TEXTS} texts; "
                f"words the same in {sum(a == b for a, b in zip(RES[c]['top1'], P0[c]['top1']))} of {TEXTS}")

# ---------------------------------------------------------------- summary
log("\nSUMMARY (top-1 / top-5 of all positions; late layers " + ", ".join(map(str, LATEL)) + ")")
fin = RES["fin"]
fw = " ".join("%.1f" % (100 * fin["first_word"][m]) for m in LATEL)
log(f"  finished first word at the late layers: {fw}")
for c in CASES:
    if c == "fin": continue
    o = RES[c]
    lm = " ".join("%4.1f" % (100 * o["wl_moved"][m]) for m in LATEL); lf = " ".join("%4.1f" % (100 * o["first_word"][m]) for m in LATEL)
    log(f"  {c:12s} top-1 {100*o['top1_all']:5.1f}  top-5 {100*o['top5_all']:5.1f}  greedy texts {o['greedy_texts']:2d}  "
        f"words {o['greedy_words']:3d} | late level moved {lm} | "
        f"late first word {lf} | edges "
        f"{100*min(o['ke_flipped']):.1f}..{100*max(o['ke_flipped']):.1f} | kept {o['kept']}")
ck = [RES[f"ck@{s}"]["top1_all"] for s in SEEDS if f"ck@{s}" in RES]
rk = [RES[f"rk@{n}"]["top1_all"] for n in RDRAWS if f"rk@{n}" in RES]
if "k" in RES and ck:
    k = RES["k"]["top1_all"]
    log(f"  k against the scrambled draws: k {100*k:.1f}, ck {100*min(ck):.1f}..{100*max(ck):.1f} "
        f"({'below every draw' if k < min(ck) else 'above every draw' if k > max(ck) else 'inside'})")
if "k" in RES and rk:
    log(f"  k against the random keep sets: k {100*RES['k']['top1_all']:.1f}, rk {100*min(rk):.1f}..{100*max(rk):.1f}")
log(f"done {time.time() - t0:.0f}s  peak memory {mem():.2f} GB")
