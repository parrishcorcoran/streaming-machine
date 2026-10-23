"""HIS SENTENCE ON THE RUNNING MACHINE: the blank weight removed exactly as it stands in the finished weight. 2026-10-06.

His words: "If the blank slate were saved somewhere you would be able to remove EXACTLY each blank weight exactly"
SESSION'S READING (the session's words, not his): what is left of each blank weight inside the finished weight is
c x that blank weight, c the shave's one number per size, read from the 241 never-built input-table rows
(../foundation_check.json c_med, the number ../foundation_rows.py uses). The predictions are in PREDICTION.txt, written
before this script first ran.

Every plate (qkv, proj, fc, mproj of every layer) and both word tables written as W - round(v x W0), whole numbers on the
shared 16-place grid of ../base2_pythia-<size>.npz, v as a whole number at 2^-30. Gains, offsets, biases untouched.
  fin    the finished machine (its head piles must equal ../pythia_base2.py's own run: sha256)
  k      v = c from the never-built rows (14m 6.112e-4, 70m 6.118e-4, 160m 1.1803e-2)
  p      v = the settings' product (3.865910e-4 for 14m and 70m, 8.961508e-3 for 160m)
  ck1-3  k with W0 scrambled within each plate (3 draws);  cp1-3  p with W0 scrambled the same way.
         Scramble: every value of the plate's blank keeps its size (digit x 2^(its own neuron's power)), positions
         scrambled over the whole plate, each written on the grid of the neuron it lands in (exact where that grid
         keeps its places, rounded to nearest where it would lose places; counted).
The machine is ../pythia_base2.py's, copied unchanged (gauge, turns, arrival with its water level, bend, race).

    python3 remove_exact.py size=14m [cases=fin,k,p,ck1,ck2,ck3,cp1,cp2,cp3] [order=plain|shuffle] [texts=30] [greedy=1]
"""
import os, sys, json, time, hashlib, zlib, resource
import numpy as np

A = dict(a.split("=") for a in sys.argv[1:])
SIZE, ORDER = A.get("size", "14m"), A.get("order", "plain")
GREEDY = A.get("greedy", "1") == "1"
TEXTS, WORDS = int(A.get("texts", 30)), int(A.get("words", 16))
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
M = np.load(os.path.join(UP, f"base2_pythia-{SIZE}.npz"))
d, L, NH, HS, ROT = (int(M[k]) for k in ("d", "L", "heads", "hs", "rot"))
ES, SC = 16, 10
PHI, ARR, LOG2, EXP2, COS, SIN = (M[k] for k in ("phi", "arrival", "log2", "exp2", "cos", "sin"))
EPS40, HALF = int(M["eps40"]), int(M["half_log2_hs"])
SHUF = ORDER == "shuffle"
PERM = np.random.default_rng(12345).permutation(d) if SHUF else np.arange(d)
PLATES = [f"{nm}.{l}" for l in range(L) for nm in ("qkv", "proj", "fc", "mproj")]
TABLES = ["head", "embed"]
CHK = json.load(open(os.path.join(UP, "foundation_check.json")))[f"{SIZE}@143000"]
C_NB, PROD = CHK["embed"]["c_med"], CHK["product"]
def to30(s): return int(round(s * 2 ** 30))
def times(v30, q): return (v30 * q + (1 << 29)) >> 30            # round(v x q), whole numbers only
V30 = {"k": to30(C_NB), "p": to30(PROD)}
DRAWS = {"1": 20261006, "2": 2, "3": 3}
t0 = time.time()
LOGF = open(os.path.join(HERE, f"remove_exact_{SIZE}{'-shuffle' if SHUF else ''}{('-' + A['tag']) if A.get('tag') else ''}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes

# ---------------------------------------------------------------- the machine, copied unchanged from ../pythia_base2.py
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

PL = {}; CUR = {}

def plain(X, name):
    w, e, b, wmax = PL[name]; x16, ex = to16(X)
    return rshift(product(x16, w, wmax), -(e[None, :] + ex[:, None] + ES)) + b

def turn(X, pos):
    r = ROT // 2; a, b = X[:, :r], X[:, r:ROT]
    assert np.abs(X[:, :ROT]).max() < 2 ** 31
    c, s = COS[pos], SIN[pos]
    return np.concatenate([rshift(a * c - b * s, 30), rshift(b * c + a * s, 30), X[:, ROT:]], 1)

def attention(QKV):
    P = len(QKV); pos = np.arange(P); out = np.zeros((P, d), np.int64); live = np.tril(np.ones((P, P), dtype=bool))
    for h in range(NH):
        o = h * 3 * HS; q, k, v = QKV[:, o:o + HS], QKV[:, o + HS:o + 2 * HS], QKV[:, o + 2 * HS:o + 3 * HS]
        q, k = turn(q, pos), turn(k, pos)
        q16, eq = to16(q); k16, ek = to16(k)
        sc = power(product(q16, k16.T.astype(np.float64), 2.0 ** 15), (eq[:, None] + ek[None, :] + SC) * 65536 - HALF)
        lev = np.where(live, sc, np.iinfo(np.int64).min).max(1)
        dd = lev[:, None] - sc
        a = np.where(live & (dd < len(ARR)), ARR[np.clip(dd, 0, len(ARR) - 1)], 0)
        assert float(np.abs(v).max(initial=0)) * float(a.sum(1).max()) < 2 ** 62
        out[:, h * HS:(h + 1) * HS] = power(a @ v, -np.array([log2fix(t) for t in a.sum(1)])[:, None])
    return out

def edge(H):
    assert np.abs(H).max(initial=0) < 2 ** 46
    idx = rshift(H, ES - 12) + 32768
    share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
    return rshift(H * share, 16)

def forward(ids, last=1):
    ids = np.asarray(ids); EW, EE = PL["embed"]
    R = rshift(EW[ids].astype(np.int64)[:, PERM], -(EE[ids][:, None] + ES))
    for l in range(L):
        At = plain(attention(plain(gauge(R, f"ln1.{l}"), f"qkv.{l}")), f"proj.{l}")
        Mo = plain(edge(plain(gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
        R = R + At + Mo
    y16, ey = to16(gauge(R[-last:], "lnf")); w, e, _, wmax = PL["head"]
    lg = rshift(product(y16, w, wmax), -(e[None, :] + ey[:, None] + ES))
    return [race(row) for row in lg], lg

# ---------------------------------------------------------------- the plates for each case
LOST = {}
def scrambled_start(name, seed):
    """the plate's blank with its values scrambled over the whole plate; each value keeps digit x 2^(own power) and
    is written on the grid of the neuron it lands in (rounded to nearest where that loses places: counted in LOST)"""
    S = M[name + ".s"]; e = M[name + ".e"].astype(np.int16); assert -128 <= int(e.min()) and int(e.max()) < 128
    shape, size = S.shape, S.size
    E = np.ascontiguousarray(np.broadcast_to(e[:, None] if name == "embed" else e[None, :], shape)).ravel()
    pk = S.ravel().astype(np.int32); del S; pk <<= 8; pk |= (E + 128).astype(np.int32)   # value and its own power, packed
    np.random.default_rng(seed + zlib.crc32(name.encode())).shuffle(pk)
    out = np.empty(size, np.int32); lost = 0
    for a in range(0, size, 1 << 22):
        b = min(size, a + (1 << 22)); src = (pk[a:b] >> 8).astype(np.int64)
        sh = ((pk[a:b] & 255) - 128).astype(np.int64) - E[a:b].astype(np.int64)   # source power minus destination power
        r = rshift(src, -sh)
        lost += int(((sh < 0) & (rshift(r, sh) != src)).sum())
        out[a:b] = r
    LOST[f"{name}@{seed}"] = lost
    return out.reshape(shape)

def kind(name): return name.split(".")[0]
KINDS = ["qkv", "proj", "fc", "mproj", "head", "embed"]

def load(name, f):
    """f: 'W' or (v30, seed or None). Returns the change's counts for this plate (None for W)."""
    if CUR.get(name) == f: return None
    PL.pop(name, None)
    W = M[name + ".w"]; st = None
    if f != "W":
        v30, seed = f
        S = M[name + ".s"] if seed is None else scrambled_start(name, seed)
        st = np.zeros(6, np.int64)          # values, changed, sum |change|, max |change|, inside the finished weight's own grain, |change| >= 2
    out = np.empty(W.shape, np.int32 if name == "embed" else np.float64)
    step = max(1, (1 << 22) // W.shape[1])
    for a in range(0, W.shape[0], step):
        b = min(W.shape[0], a + step); w = W[a:b].astype(np.int64)
        if f == "W": out[a:b] = w; continue
        r = times(v30, S[a:b].astype(np.int64)); out[a:b] = w - r
        ar = np.abs(r); bl = np.floor(np.log2(np.maximum(np.abs(w), 1))).astype(np.int64) + 1   # bit length of |w|
        grain = np.left_shift(np.int64(1), np.maximum(bl - 11, 0))                             # the finished value's own grain (11 places)
        st += [r.size, int((r != 0).sum()), int(ar.sum()), 0, int((2 * ar <= grain).sum()), int((ar >= 2).sum())]
        st[3] = max(int(st[3]), int(ar.max(initial=0)))                  # the largest change (kept as a largest, not summed)
    if f != "W": del S
    e = M[name + ".e"].astype(np.int64)
    if name == "embed": PL[name] = (out, e)
    elif name == "head": PL[name] = (out[PERM] if SHUF else out, e, 0, float(np.abs(out).max()))
    else:
        b = M[name + ".b"].astype(np.int64)
        if name.startswith(("qkv", "fc")): out = out[PERM] if SHUF else out
        else: out, e, b = (out[:, PERM], e[PERM], b[PERM]) if SHUF else (out, e, b)
        PL[name] = (out, e, b, float(np.abs(out).max()))
    CUR[name] = f
    return st

def set_case(case):
    if case == "fin": f = "W"
    elif case in ("k", "p"): f = (V30[case], None)
    else: f = (V30[case[1]], DRAWS[case[2]])
    agg = {k: np.zeros(6, np.int64) for k in KINDS}
    for name in TABLES + PLATES:
        st = load(name, f)
        if st is not None: m3 = max(int(agg[kind(name)][3]), int(st[3])); agg[kind(name)] += st; agg[kind(name)][3] = m3
    return agg

# ---------------------------------------------------------------- the runs
refs = json.load(open(os.path.join(UP, f"refs_pythia-{SIZE}.json")))[:TEXTS]
oldw = json.load(open(os.path.join(UP, f"words_pythia-{SIZE}.json")))["texts_detail"]
CASES = A.get("cases", "fin,k,p,ck1,ck2,ck3,cp1,cp2,cp3").split(",")
if CASES[0] != "fin": CASES = ["fin"] + CASES
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(UP, "models", "tokenizer", "tokenizer.json"))
log(f"remove_exact.py  size {SIZE}  order {ORDER}  texts {TEXTS}  greedy {GREEDY}  cases {CASES}")
log(f"  c from the never-built rows {C_NB:.6e} -> {V30['k']} x 2^-30;  the settings' product {PROD:.6e} -> {V30['p']} x 2^-30")

def top5(row): return np.lexsort((np.arange(len(row)), -row))[:5]
def rowsha(lg): return [hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()[:16] for r in lg]

FIN = None; RES = dict(size=SIZE, order=ORDER, c=C_NB, product=PROD, v30=V30, draws=DRAWS, cases={})
for case in CASES:
    tc = time.time(); agg = set_case(case); per = []
    for t, r in enumerate(refs):
        seq = r["ids"] + r["ref"][:WORDS - 1]; n0 = len(r["ids"]) - 1
        got, lg = forward(seq, last=len(seq))
        t5 = np.array([top5(row) for row in lg]); assert (t5[:, 0] == np.array(got)).all()
        x = dict(top1=got, top5=t5.tolist(), n0=n0, rows=rowsha(lg),
                 sha_all=hashlib.sha256(np.ascontiguousarray(lg).tobytes()).hexdigest(),
                 sha_last=hashlib.sha256(np.ascontiguousarray(lg[-WORDS:]).tobytes()).hexdigest())
        if FIN is None:                                                  # the finished machine: its lead at every position
            srt = np.sort(lg, axis=1)[:, -2:]; x["lead"] = ((srt[:, 1] - srt[:, 0]) / 2.0 ** ES).tolist()
        else:                                                            # the finished word's lead over the best other word, here
            fw = np.array(FIN[t]["top1"]); pw = lg[np.arange(len(lg)), fw]; tmp = lg.copy(); tmp[np.arange(len(lg)), fw] = np.iinfo(np.int64).min
            x["lead_here"] = ((pw - tmp.max(1)) / 2.0 ** ES).tolist(); del tmp
        if GREEDY:
            cont = got[n0:n0 + WORDS]; bad = [j for j in range(WORDS) if cont[j] != r["ref"][j]]
            own = list(r["ref"][:WORDS]) if not bad else r["ref"][:bad[0]] + [cont[bad[0]]]
            while len(own) < WORDS: own.append(forward(r["ids"] + own)[0][0])
            x["greedy"] = own[:WORDS]
        per.append(x)
    if case == "fin":
        FIN = per
        ok_ref = sum(int(a == b) for x, r in zip(per, refs) for a, b in zip(x["top1"][x["n0"]:x["n0"] + WORDS], r["ref"]))
        ok_sha = sum(x["sha_last"] == o["piles_sha256"] for x, o in zip(per, oldw))
        log(f"  fin: the finished machine's word = fp32's at {ok_ref} of {TEXTS * WORDS} continuation positions; head piles of the "
            f"last {WORDS} positions identical (sha256) to ../pythia_base2.py's run in {ok_sha} of {TEXTS} texts" + ("" if not SHUF else " (lanes shuffled)"))
        RES["cases"]["fin"] = dict(sha_all=[x["sha_all"] for x in per], top1=[x["top1"] for x in per],
                                   lead=[x["lead"] for x in per], seconds=round(time.time() - tc, 1))
        continue
    n = t1 = t5c = rows_same = 0; c1 = c5 = k1 = n_c = 0; gw = gt = 0; first = []; lostpos = []; moves = []
    for t in range(TEXTS):
        x, f = per[t], FIN[t]; nx = FIN[(t + 1) % TEXTS]
        for p in range(len(x["top1"])):
            n += 1; same = x["top1"][p] == f["top1"][p]; t1 += same; t5c += f["top1"][p] in x["top5"][p]
            rows_same += x["rows"][p] == f["rows"][p]; moves.append(x["lead_here"][p] - f["lead"][p])
            if not same: lostpos.append(dict(text=t, pos=p, fin_lead=round(f["lead"][p], 5), here=round(x["lead_here"][p], 5)))
        for j in range(WORDS):
            p, q = x["n0"] + j, nx["n0"] + j; n_c += 1
            c1 += x["top1"][p] == f["top1"][p]; c5 += f["top1"][p] in x["top5"][p]; k1 += x["top1"][p] == nx["top1"][q]
        if GREEDY:
            eq = [a == b for a, b in zip(x["greedy"], refs[t]["ref"])]; gw += sum(eq); gt += all(eq)
            first.append(eq.index(False) if not all(eq) else WORDS)
    am = np.abs(np.array(moves))
    ch = {k: dict(values=int(v[0]), changed=int(v[1]), changed_share=float(v[1] / v[0]), mean_abs=float(v[2] / v[0]), max_abs=int(v[3]),
                  inside_own_grain=float(v[4] / v[0]), two_or_more=float(v[5] / v[0])) for k, v in agg.items()}
    tot = np.sum([v for v in agg.values()], axis=0)                     # (its [3] is not used: the largest is taken below)
    o = dict(top1_all=t1 / n, top5_all=t5c / n, positions=n, top1_cont=c1 / n_c, top5_cont=c5 / n_c, control_top1_cont=k1 / n_c,
             rows_identical=rows_same, texts_identical_piles=sum(x["sha_all"] == f["sha_all"] for x, f in zip(per, FIN)),
             greedy_words=gw if GREEDY else None, greedy_texts=gt if GREEDY else None,
             greedy_first_diff=first if GREEDY else None, lost=lostpos,
             lead_move_median=float(np.median(am)), lead_move_p99=float(np.percentile(am, 99)), lead_move_max=float(am.max()),
             changed=ch, changed_all=dict(values=int(tot[0]), changed=int(tot[1]), changed_share=float(tot[1] / tot[0]),
                                          mean_abs=float(tot[2] / tot[0]), max_abs=int(max(v[3] for v in agg.values())),
                                          inside_own_grain=float(tot[4] / tot[0])),
             sha_all=[x["sha_all"] for x in per], top1=[x["top1"] for x in per], greedy=[x.get("greedy") for x in per],
             seconds=round(time.time() - tc, 1))
    RES["cases"][case] = o
    log(f"  {case:4s} top-1 {o['top1_all']*100:6.2f}%  top-5 {o['top5_all']*100:6.2f}%  (of {n});  16 continuation top-1 {o['top1_cont']*100:6.2f}% "
        f"[pairing-broken {o['control_top1_cont']*100:4.1f}%];  greedy words {gw}/{TEXTS * WORDS} texts {gt}/{TEXTS};  piles identical: "
        f"positions {rows_same}/{n} texts {o['texts_identical_piles']}/{TEXTS};  {o['seconds']}s  peak {mem():.2f} GB")
    log(f"       the finished word's lead moved by: median {o['lead_move_median']:.5f}  99th pct {o['lead_move_p99']:.5f}  max {o['lead_move_max']:.5f} logits")
    if lostpos: log(f"       positions lost (finished lead -> here): " + ", ".join(f"t{q['text']}p{q['pos']} {q['fin_lead']:.4f}->{q['here']:.4f}" for q in lostpos[:12]) + (" ..." if len(lostpos) > 12 else ""))
    log(f"       weights changed on the grid: {o['changed_all']['changed']:,} of {o['changed_all']['values']:,} ({o['changed_all']['changed_share']*100:.2f}%), "
        f"mean |change| {o['changed_all']['mean_abs']:.2f} grains, largest {o['changed_all']['max_abs']}; inside the finished weight's own 11-place grain "
        f"{o['changed_all']['inside_own_grain']*100:.1f}%")
    log("       by kind: " + "  ".join(f"{k} {v['changed_share']*100:.1f}% mean {v['mean_abs']:.2f} max {v['max_abs']} in-grain {v['inside_own_grain']*100:.0f}%" for k, v in ch.items()))
    if GREEDY:
        for t in [u for u in range(TEXTS) if per[u]["greedy"] != refs[u]["ref"][:WORDS]][:4]:
            log(f"       text {t} {refs[t]['prompt']!r}: here {tok.decode(per[t]['greedy'])!r}  finished {tok.decode(refs[t]['ref'][:WORDS])!r}")
RES["lost_places_in_scramble"] = LOST
if SHUF:                                                                  # the order-free check against the plain run
    pj = os.path.join(HERE, f"remove_exact_{SIZE}.json")
    if os.path.exists(pj):
        P = json.load(open(pj))["cases"]
        for case in CASES:
            if case in P:
                same = sum(a == b for a, b in zip(RES["cases"][case]["sha_all"], P[case]["sha_all"]))
                log(f"  order-free check, {case}: every head pile at every position the same whole number as the plain run in {same} of {TEXTS} texts")
                RES["cases"][case]["order_free_same_texts"] = same
json.dump(RES, open(LOGF.name[:-4] + ".json", "w"))
log(f"done {time.time() - t0:.0f}s  peak memory {mem():.2f} GB")
