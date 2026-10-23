"""ONE PLATE AT A TIME: which plate carries the water level in the late layers. 2026-10-06.

HIS SENTENCE (his exact words, the only ones here that are his): "It should be 100% removable after training."
  and "Any failure is not a pivot."
SESSION'S RESTATEMENT (the session's words, not his), as in ../remove_foundation.py: the finished plate W = what
survives of the random start, round(s x W0) with one s per plate (../foundation_share.json), + what was built on it.

WHAT WAS FOUND BEFORE THIS SCRIPT (../remove_<size>.log, -extra*, -setter): removing round(s x W0) from every plate
loses about half the words at 70m and 160m; the knife edges hold (4-13% change side) but the water level is lost in
the late layers (160m layers 9-11: 52-66% of rows move), where the text's first word stops setting the level in many
rows; at 160m the real foundation costs more than a scrambled copy (b 50.5 against cb 57.9-60.9 top-1).

WHAT IS RUN HERE: the same base-2 machine (pythia_base2.py's parts, copied unchanged through ../remove_foundation.py),
on the same shared grid, the same 30 texts and 16 words. Only one plate, or one plate kind, changes per case:
  fin             the finished machine (its head piles must match pythia_base2.py's run: sha256, 30 of 30 texts)
  b:<kind>.<l>    the surviving foundation removed from that one plate only: W - round(s x W0)
  cb:<kind>.<l>@n CONTROL: the same with W0 scrambled within that plate (the same values, positions moved; draw n),
                  three draws: seeds 20261006, 2, 3 (the same draws ../remove_foundation.py made for those seeds)
  b:K=<kind>      the surviving foundation removed from every plate of that kind (every qkv, every proj, ...)
  cb:K=<kind>@n   CONTROL: the same, scrambled, three draws
  kinds: qkv (makes the arrival's question, key and value), proj (writes the arrival into the stream), fc (feeds the
  bend), mproj (writes the bend into the stream). Gains, offsets, biases and both word tables are never touched.
Measured against the finished machine, on the same fed-in words (each text's own words, then the finished machine's
16 greedy words), plus the case's own 16 greedy words:
  - rows whose water level moves, at every layer (the row's first position at the level is another position)
  - who sets the level at every layer: the text's first word / the row itself / the row before / other; the summary
    reads the first word's share at the last third of the layers (160m layers 8-11, 70m layers 4-5)
  - knife edges that change side, at every layer
  - top-1 and top-5 (the finished word inside the case's top 5) over all 622 positions; 16 greedy words and texts
  - a look, not the machine: in the finished machine, how far the next position's score stands below the level
    (in e-folds of the arrival table), for the rows that moved and the rows that held
  - statistic control (pairing broken, rule 13.3): finished text t against finished text t+1, same row
Wiring facts (fixed by the machine, a check and not a finding; rule 13.4): Pythia's attention and bend both read the
stream before layer l writes, so only qkv.l can move layer l's level; proj.l, fc.l, mproj.l reach the level from layer
l+1 on. Only fc.l can move layer l's knife edges; qkv.l, proj.l, mproj.l reach them from layer l+1 on.

PREDICTIONS, written before the first run of this script (results go to plate_look_<size>.log / .json, never here).
  Basis: ../remove_160m.log one layer at a time (b_L<l>: four plates of layer l): the level moves at layer l itself in
  9, 6, 10, 7, 7, 7, 17, 21, 25, 42, 54, 61% of rows (l = 0..11) and at the later layers by 4-10%; cb_L<l> (one draw)
  within 4 points of b_L<l> at every own layer. 70m: 8, 7, 6, 25, 60, 67% (cb 7, 7, 5, 29, 64, 72%). The qkv plate's
  surviving foundation is small in the late layers (rho = |s W0|/|W|: 160m qkv.9-11 0.033, 0.034, 0.042; 70m qkv.4-5
  0.020, 0.032), the fc plates' largest (160m 0.17-0.36, 70m 0.12-0.25).
  160m:
  (1) the late level is carried by each layer's own qkv: b:qkv.l moves layer l's level in 35-60% of rows at l = 9-11
      (within 8 points of b_L<l>'s 42, 54, 61%), and in 15-30% at l = 6-8. b:proj.l, b:fc.l, b:mproj.l each move every
      later layer's level by under 10% of rows.
  (2) real against scramble at the late level: the three cb:qkv.l draws at l = 9-11 sit within 6 points of b:qkv.l on
      layer l's moved rows, and b inside their spread or within 3 points of it: there the surviving foundation is to
      the level just a random relief of its size.
  (3) the first word's share of the levels at layers 8-11 (finished 61.2, 65.8, 70.5, 53.4%): b:qkv.l at l = 9-11
      takes 10-30 points off at layer l; cb:qkv.l the same within 5 points. Every other single plate moves it by under
      5 points at every layer.
  (4) knife edges: b:fc.l changes the side of 3-7% of layer l's edges; every single plate changes 1-5% of every later
      layer's edges.
  (5) words (top-1 of 622): b:fc.0 is the largest single-plate loss in layers 0-8 (8-15 points); b:qkv.11 loses 8-16
      points; every b:proj.l keeps 95% or more.
  (6) real against scramble on words: the layer-0 excess (b_L0 80.7 against cb_L0 86.7-88.1) sits in fc.0 or mproj.0
      (the plates whose surviving start is largest: s 0.46, 0.67): b below all three draws there by 3 points or more.
      At every other plate b within 2 points of the draws' spread.
  (7) by kind: b:K=qkv moves the late level (layers 9-11) in 45-65% of rows (b: 52, 56, 66%), changes 3-8% of knife
      edges per layer, top-1 60-75%. b:K=fc changes 6-10% of edges per layer, moves the late level in 10-25%, top-1
      65-80%. b:K=mproj top-1 75-88%; b:K=proj 88-95%. The late level is carried by the qkv kind. b:K=fc or b:K=mproj
      below all three cb draws (holding the 160m excess); b:K=qkv within the draws' spread.
  70m:
  (8) b:qkv.4, b:qkv.5 move their own layer's level in 50-70% of rows, b:qkv.3 in 15-30%; cb draws within 6 points.
      b:K=qkv moves layers 4-5 by 55-70%. b inside or within 3 points of the cb spread for every plate and kind
      (the 70m run found no real-vs-scramble excess: b 52.6 inside cb 51.4-60.9).
  (9) the look: in the finished machine the rows whose level moves stand with the next position closer below the
      level than the rows that hold, at every layer.

  python3 plate_look.py size=160m [cases=all|plates|kinds|<list>] [seeds=20261006,2,3] [greedy=1] [order=plain|shuffle]
                         [texts=30] [tag=]
"""
import os, sys, json, math, time, hashlib, zlib, resource
import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE, ORDER = A.get("size", "160m"), A.get("order", "plain")
GREEDY = A.get("greedy", "1") == "1"
SEEDS = [int(x) for x in A.get("seeds", "20261006,2,3").split(",")]
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
LATEL = list(range((2 * L) // 3, L))                                    # the last third of the layers
t0 = time.time()
STEM = os.path.join(LATE, f"plate_look_{SIZE}{('-' + A['tag']) if A.get('tag') else ''}{'-shuffle' if SHUF else ''}")
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

def parse(case):
    """fin | b:<plate> | cb:<plate>@<seed> | b:K=<kind> | cb:K=<kind>@<seed>  ->  (kind of cut, set of plates, seed)"""
    if case == "fin": return None, set(), None
    cut, _, rest = case.partition(":"); rest, _, seed = rest.partition("@")
    names = {p for p in PLATES if p.split(".")[0] == rest[2:]} if rest.startswith("K=") else {rest}
    assert names and names <= set(PLATES), case
    return cut, names, (int(seed) if seed else None)

def form(case, name):
    cut, names, seed = parse(case)
    if name not in names: return ("W", None)
    return ("Dm", None) if cut == "b" else ("Dms", seed)

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
PLATE_CASES = [c for p in PLATES for c in [f"b:{p}"] + [f"cb:{p}@{s}" for s in SEEDS]]
KIND_CASES = [c for k in KINDS for c in [f"b:K={k}"] + [f"cb:K={k}@{s}" for s in SEEDS]]
CASES = {"all": PLATE_CASES + KIND_CASES, "plates": PLATE_CASES, "kinds": KIND_CASES}.get(CA) or CA.split(",")
if CASES[0] != "fin": CASES = ["fin"] + CASES
log(f"plate_look.py  size {SIZE}  order {ORDER}  texts {TEXTS}  greedy {GREEDY}  seeds {SEEDS}  cases {len(CASES)}  "
    f"late layers {LATEL}")

def top5(row):
    return np.lexsort((np.arange(len(row)), -row))[:5]

def med(v): return round(float(np.median(v)), 3) if len(v) else None

FIN = None; RES = {}
for case in CASES:
    tc = time.time(); set_case(case)
    per = []
    for t, r in enumerate(refs):
        seq = r["ids"] + r["ref"][:WORDS - 1]; n0 = len(r["ids"]) - 1
        rec = dict(wl=[], ke=[], gap=[])
        got, lg = forward(seq, last=len(seq), rec=rec)
        t5 = np.array([top5(row) for row in lg])
        assert (t5[:, 0] == np.array(got)).all()
        x = dict(top1=got, top5=t5.tolist(), wl=rec["wl"], ke=rec["ke"], gap=rec["gap"],
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
        lv = [np.concatenate([x["gap"][l][:, 1:].ravel() for x in per]) / 1024.0 for l in range(L)]
        RES["fin_gap_median_efolds"] = [med(v) for v in lv]
        log(f"  pairing-broken control (finished text t vs t+1): level moved {RES['control_finished_vs_next_text']['wl_moved']}; "
            f"edges changed side {RES['control_finished_vs_next_text']['ke_flipped']}")
        log(f"  finished: next score below the level, median e-folds by layer {RES['fin_gap_median_efolds']}")
    n_all = t1 = t5c = 0; gw = gt = 0
    wl = np.zeros((L, 2)); ke = np.zeros((L, 2)); setter = np.zeros((L, 4))
    gap_mv = [[] for _ in range(L)]; gap_hd = [[] for _ in range(L)]
    for t in range(TEXTS):
        x, f = per[t], FIN[t]
        for p in range(len(x["top1"])):
            n_all += 1; t1 += x["top1"][p] == f["top1"][p]; t5c += f["top1"][p] in x["top5"][p]
        eq = [a == b for a, b in zip(x["greedy"], refs[t]["ref"])]; gw += sum(eq); gt += all(eq)
        for l in range(L):
            mv = x["wl"][l][:, 1:] != f["wl"][l][:, 1:]
            wl[l] += [mv.sum(), mv.size]
            ke[l] += [(x["ke"][l] != f["ke"][l]).sum(), x["ke"][l].size]
            wv = x["wl"][l][:, 1:]; row = np.arange(1, wv.shape[1] + 1)[None, :]
            q0 = wv == 0; q1 = ~q0 & (wv == row); q2 = ~q0 & ~q1 & (wv == row - 1)
            setter[l] += [q0.sum(), q1.sum(), q2.sum(), (~q0 & ~q1 & ~q2).sum()]
            g = f["gap"][l][:, 1:] / 1024.0; gap_mv[l] += g[mv].tolist(); gap_hd[l] += g[~mv].tolist()
    S = setter / setter.sum(1, keepdims=True)
    o = dict(top1_all=t1 / n_all, top5_all=t5c / n_all, positions=n_all, greedy_words=gw, greedy_texts=gt,
             wl_moved=(wl[:, 0] / wl[:, 1]).round(4).tolist(), ke_flipped=(ke[:, 0] / ke[:, 1]).round(4).tolist(),
             setter=S.round(4).tolist(), first_word=S[:, 0].round(4).tolist(),
             gap_moved=[med(v) for v in gap_mv], gap_held=[med(v) for v in gap_hd],
             top1=[x["top1"] for x in per], greedy=[x["greedy"] for x in per], seconds=round(time.time() - tc, 1))
    RES[case] = o
    pc = lambda v: " ".join(f"{100 * a:3.0f}" for a in v)
    log(f"  {case:16s} top-1 {o['top1_all']*100:5.1f}  top-5 {o['top5_all']*100:5.1f}  greedy {gw:3d}/{TEXTS * WORDS} "
        f"texts {gt:2d}/{TEXTS}  | level moved {pc(o['wl_moved'])} | first word {pc(o['first_word'])} | "
        f"edges {pc(o['ke_flipped'])} | {o['seconds']}s {mem():.2f} GB")
    if case != "fin":
        for x in per: x.pop("wl"); x.pop("ke"); x.pop("gap")
    RES["lost_places_in_shuffle"] = LOST
    json.dump(dict(size=SIZE, order=ORDER, texts=TEXTS, words=WORDS, seeds=SEEDS, late=LATEL, cases=RES),
              open(STEM + ".json", "w"))

# ---------------------------------------------------------------- summary tables (from RES only)
def row(c): return RES.get(c)
log("\nSUMMARY  (b = surviving foundation removed; cb = scrambled copy, three draws as min..max)")
log(f"  finished first-word share by layer: {' '.join(f'{100*a:.1f}' for a in RES['fin']['first_word'])}")
log(f"  {'plate':8s} {'top-1 b':>8s} {'cb draws':>12s} | {'own-layer level moved b / cb':>30s} | "
    f"{'late level moved b (layers ' + str(LATEL[0]) + '-' + str(LATEL[-1]) + ')':>34s} | {'late first word b':>20s} | "
    f"{'edges own/next b':>16s} | greedy texts b")
for p in [c[2:] for c in CASES if c.startswith("b:")]:
    b = row(f"b:{p}"); cbs = [row(f"cb:{p}@{s}") for s in SEEDS]; cbs = [c for c in cbs if c]
    if not b: continue
    if p.startswith("K="): l0, lr = 0, 0
    else:
        k, l = p.split("."); l = int(l); l0 = l if k == "qkv" else l + 1; lr = l if k == "fc" else l + 1
    own_b = b["wl_moved"][l0] if l0 < L else None
    own_c = [c["wl_moved"][l0] for c in cbs] if l0 < L else []
    ct = [c["top1_all"] for c in cbs]
    s_own = (f"{100*own_b:5.1f} / {100*min(own_c):5.1f}..{100*max(own_c):5.1f}" if own_b is not None and own_c
             else (f"{100*own_b:5.1f}" if own_b is not None else "-"))
    if p.startswith("K="): s_own = "(every layer)"
    lt = " ".join(f"{100*b['wl_moved'][l]:3.0f}" for l in LATEL)
    lf = " ".join(f"{100*b['first_word'][l]:3.0f}" for l in LATEL)
    ek = (f"{100*b['ke_flipped'][lr]:4.1f}/{100*b['ke_flipped'][lr+1]:4.1f}" if (not p.startswith("K=") and lr + 1 < L)
          else (f"{100*b['ke_flipped'][lr]:4.1f}/-" if lr < L else "-"))
    if p.startswith("K="): ek = f"{100*min(b['ke_flipped']):.1f}..{100*max(b['ke_flipped']):.1f}"
    log(f"  {p:8s} {100*b['top1_all']:8.1f} {(f'{100*min(ct):5.1f}..{100*max(ct):5.1f}' if ct else '-'):>12s} | "
        f"{s_own:>30s} | {lt:>34s} | {lf:>20s} | {ek:>16s} | {b['greedy_texts']}")
log(f"done {time.time() - t0:.0f}s  peak memory {mem():.2f} GB")
