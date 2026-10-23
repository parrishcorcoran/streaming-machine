"""QuantumPythia: the all-base-2 Pythia, checked against fp32 Pythia's own greedy words (refs_pythia-<size>.json).
Every value a whole number. Products are whole-number products: float64 only where every partial sum is proven below
2^53 (the bound is asserted), int64 for the arrival, Python whole numbers where a product would pass 2^63.
Follows ../base2.py (GPT-2) and ../qwen/qwen_base2.py (exact turns, gauge, water level). Pythia's own parts as they are:
  - parallel: the attention and the MLP both read the same stream and both write back:  R <- R + A(g1 R) + M(g2 R)
  - the gauge (layer norm), with its gain and offset as their own numbers (not folded into the plate after it, so
    the plate stays the plate the start was drawn on): each lane's n x - sum x, over sqrt(n sum x^2 - (sum x)^2 +
    eps n^2) -- the log of the size halved, no root and no divide -- times the gain, plus the offset
  - the turns (rotary) on the first quarter of each head: pair (i, i + quarter/2) turned by position x
    10000^(-2i/quarter); cos and sin whole numbers at 2^-30
  - the score's 1/sqrt(head size) is a half power of two (2^-2.5 for the 14m's head of 32, 2^-3 for 64): a table read
    of the fine part and a shift
  - the arrival: the clock starts at whoever arrives first in the row (the water level), e^-d from its table, the
    sum divided by the total through its log
  - the bend: Pythia's own erf form, x Phi(x), Phi read from the knife-edge table built the way base2.py builds PHI
  - separate word tables: embed in (a word's row, one power of two per word), head out (a word's column, one power
    of two per word); the word: whoever arrives first (the race)
Every position's word is read in one pass with fp32's own words fed in; that is the same as running each prefix alone,
because nothing in the machine reaches forward in the text (every gauge, 16-place cut and arrival is per position).
Texts that differ are then run on with the machine's own words, as base2.py does.
Switches:
    size=14m|70m|160m   plates=finished|start|built (built = finished - start, digit for digit)
    order=plain|shuffle (shuffle: the stream's lanes in a random order, every plate permuted to match: an order-free
                         check, every head pile must be the same whole number)
    gauge= turn= edge= arrival= float   (one part back to float, to trace a difference to its place)
    matmul=int          (every product as int64 sums instead of float64: must give the same whole numbers)
    texts=30 words=16
    python3 pythia_base2.py size=14m
"""
import os, sys, json, math, hashlib, numpy as np

A = dict(a.split("=") for a in sys.argv[1:])
SIZE, PLATES, ORDER = A.get("size", "14m"), A.get("plates", "finished"), A.get("order", "plain")
GAUGE, TURN, EDGE, ARRIVAL = (A.get(k, "int") for k in ("gauge", "turn", "edge", "arrival"))
TEXTS, WORDS = int(A.get("texts", 30)), int(A.get("words", 16)); MATMUL = A.get("matmul", "float64")
HERE = os.path.dirname(os.path.abspath(__file__))
M = np.load(os.path.join(HERE, f"base2_pythia-{SIZE}.npz"))
d, L, NH, HS, ROT = (int(M[k]) for k in ("d", "L", "heads", "hs", "rot"))
ES, SC = 16, 10
PHI, ARR, LOG2, EXP2, COS, SIN = (M[k] for k in ("phi", "arrival", "log2", "exp2", "cos", "sin"))
EPS40, HALF = int(M["eps40"]), int(M["half_log2_hs"])
PERM = np.random.default_rng(12345).permutation(d) if ORDER == "shuffle" else np.arange(d)

def pick(f, s):                                       # which plate: the finished, the start, or what was built on it
    f, s = f.astype(np.int64), s.astype(np.int64)
    return {"finished": f, "start": s, "built": f - s}[PLATES]

PL, NORM = {}, {}
for l in range(L):
    for nm in ("ln1", "ln2"):
        k = f"{nm}.{l}"; NORM[k] = (pick(M[k + ".g"], M[k + ".gs"])[PERM], int(M[k + ".ge"]), pick(M[k + ".b"], M[k + ".bs"])[PERM])
    for nm in ("qkv", "proj", "fc", "mproj"):
        k = f"{nm}.{l}"; w, e, b = pick(M[k + ".w"], M[k + ".s"]), M[k + ".e"], pick(M[k + ".b"], M[k + ".bs"])
        if nm in ("qkv", "fc"): w = w[PERM]                                   # reads the stream: rows follow its lanes
        else: w, e, b = w[:, PERM], e[PERM], b[PERM]                          # writes the stream: columns follow its lanes
        PL[k] = (w.astype(np.float64), e, b, float(np.abs(w).max())); del w
NORM["lnf"] = (pick(M["lnf.g"], M["lnf.gs"])[PERM], int(M["lnf.ge"]), pick(M["lnf.b"], M["lnf.bs"])[PERM])
w = pick(M["head.w"], M["head.s"])[PERM]; PL["head"] = (w.astype(np.float64), M["head.e"], 0, float(np.abs(w).max())); del w
EMB_F, EMB_S, EMB_E = M["embed.w"], M["embed.s"], M["embed.e"]

def rshift(X, s):                                     # X x 2^-s, rounded to nearest: add half, shift (s may be 0 or negative)
    s = np.asarray(s, dtype=np.int64)
    if (s < 0).any(): assert np.abs(X).max(initial=0) < 2 ** (62 + int(np.minimum(s, 0).min()))
    up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)

_shift_py = np.frompyfunc(lambda x, s: (x << -s) if s <= 0 else ((x + (1 << (s - 1))) >> s), 2, 1)

def to16(X):                                          # each position on 16 places with its own power of two (value = x16 x 2^ex)
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(X), axis=1)])
    sh = np.maximum(top - 15, 0)
    return rshift(X, sh[:, None]), sh - ES

def product(x16, w, wmax):                            # whole-number meetings and sums; exact: every partial sum below 2^53
    assert float(np.abs(x16).sum(1).max()) * wmax < 2 ** 53
    if MATMUL == "int": return x16 @ w.astype(np.int64)                  # the check: int64 sums, no float anywhere
    with np.errstate(all="ignore"):                                      # macOS's float64 product raises stray flags; the
        p = x16.astype(np.float64) @ w                                   # result is checked whole and finite here, and
    assert np.isfinite(p).all() and (p == np.round(p)).all()             # equal to the int64 sums by the matmul=int run
    return p.astype(np.int64)

def log2fix(v):                                       # log2 of a positive whole number: top 1's place + a table read
    v = int(v); t = v.bit_length() - 1
    m = (v >> (t - 16)) & 0xFFFF if t >= 16 else (v << (16 - t)) & 0xFFFF
    return t * 65536 + int(LOG2[m])

def power(X, F):                                      # X x 2^(F / 65536): a table read of the fine part, then a shift
    F = np.broadcast_to(np.asarray(F, dtype=np.int64), X.shape); I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF)
    if np.abs(X).max(initial=0) < 2 ** 45: return rshift(X * EXP2[f], 16 - I)
    return _shift_py(X.astype(object) * EXP2[f].astype(object), (16 - I).astype(object)).astype(np.int64)   # past 2^63: Python whole numbers

def race(vals):                                       # who arrives first: top digits first; at each digit, if anyone still in
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)   # has a 1 there, everyone with a 0 drops out
    alive = np.ones(len(u), dtype=bool)
    for bit in range(63, -1, -1):
        has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        if has.any(): alive = has
    return int(np.flatnonzero(alive)[0])

def gauge(R, name):                                   # the layer norm, whole: centre, size by its halved log, gain, offset
    g, ge, b = NORM[name]
    if GAUGE == "float":
        x = R / 2.0 ** ES; mu = x.mean(1, keepdims=True); sg = np.sqrt(((x - mu) ** 2).mean(1, keepdims=True) + 1e-5)
        return np.round((x - mu) / sg * g * 2.0 ** ge * 2 ** ES).astype(np.int64) + b
    x16, ex = to16(R)
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    c = d * x16 - S1[:, None]                                            # d x each lane's distance from the level
    V = d * S2 - S1 * S1 + np.array([EPS40 >> (40 + 2 * int(e)) for e in ex])
    LS = np.array([log2fix(v) >> 1 for v in V])                          # log2 of the size, halved: the root never taken
    return power(c * g[None, :], (ge + ES) * 65536 - LS[:, None]) + b

def plain(X, name):                                   # a plate: 16 places in, whole-number product, onto the stream's scale, bias
    w, e, b, wmax = PL[name]; x16, ex = to16(X)
    return rshift(product(x16, w, wmax), -(e[None, :] + ex[:, None] + ES)) + b

def turn(X, pos):                                     # the first quarter of the head: pair (i, i + quarter/2) turns by its angle
    r = ROT // 2; a, b = X[:, :r], X[:, r:ROT]
    if TURN == "float":
        ang = pos[:, None] * (10000.0 ** (-np.arange(0, ROT, 2) / ROT))[None, :]; c, s = np.cos(ang), np.sin(ang)
        return np.concatenate([np.round(a * c - b * s).astype(np.int64), np.round(b * c + a * s).astype(np.int64), X[:, ROT:]], 1)
    assert np.abs(X[:, :ROT]).max() < 2 ** 31
    c, s = COS[pos], SIN[pos]
    return np.concatenate([rshift(a * c - b * s, 30), rshift(b * c + a * s, 30), X[:, ROT:]], 1)

def attention(QKV):                                   # the arrival, head by head (Pythia's order: per head q, k, v)
    P = len(QKV); pos = np.arange(P); out = np.zeros((P, d), np.int64); live = np.tril(np.ones((P, P), dtype=bool))
    for h in range(NH):
        o = h * 3 * HS; q, k, v = QKV[:, o:o + HS], QKV[:, o + HS:o + 2 * HS], QKV[:, o + 2 * HS:o + 3 * HS]
        q, k = turn(q, pos), turn(k, pos)
        if ARRIVAL == "float":
            s = (q / 2.0 ** ES) @ (k / 2.0 ** ES).T / math.sqrt(HS); s = np.where(live, s, -np.inf)
            a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
            out[:, h * HS:(h + 1) * HS] = np.round(a @ (v / 2.0 ** ES) * 2 ** ES).astype(np.int64); continue
        q16, eq = to16(q); k16, ek = to16(k)
        sc = power(product(q16, k16.T.astype(np.float64), 2.0 ** 15), (eq[:, None] + ek[None, :] + SC) * 65536 - HALF)   # scores at 2^-10
        lev = np.where(live, sc, np.iinfo(np.int64).min).max(1)          # the water level: the race's winner in each row
        dd = lev[:, None] - sc
        a = np.where(live & (dd < len(ARR)), ARR[np.clip(dd, 0, len(ARR) - 1)], 0)
        assert float(np.abs(v).max(initial=0)) * float(a.sum(1).max()) < 2 ** 62
        out[:, h * HS:(h + 1) * HS] = power(a @ v, -np.array([log2fix(t) for t in a.sum(1)])[:, None])   # over the total: its log negated
    return out

def edge(H):                                          # the bend: where the value arrives picks the share past the knife edge
    if EDGE == "float":
        from math import erf
        h = H / 2.0 ** ES; return np.round(h * 0.5 * (1 + np.vectorize(erf)(h / math.sqrt(2))) * 2 ** ES).astype(np.int64)
    idx = rshift(H, ES - 12) + 32768
    share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
    return rshift(H * share, 16)

def forward(ids, last=1):                             # the words after the last `last` positions; and the head's piles
    ids = np.asarray(ids)
    R = rshift(pick(EMB_F[ids], EMB_S[ids])[:, PERM], -(EMB_E[ids][:, None] + ES))   # each word's row onto the stream's scale 2^-16
    for l in range(L):
        At = plain(attention(plain(gauge(R, f"ln1.{l}"), f"qkv.{l}")), f"proj.{l}")
        Mo = plain(edge(plain(gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
        R = R + At + Mo
    y16, ey = to16(gauge(R[-last:], "lnf")); w, e, _, wmax = PL["head"]
    lg = rshift(product(y16, w, wmax), -(e[None, :] + ey[:, None] + ES))  # every word's pile on one scale, 2^-16
    return [race(row) for row in lg], lg

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "tokenizer"))
    refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))[:TEXTS]
    print(f"QuantumPythia {SIZE}, all base 2: plates {PLATES}, order {ORDER}; gauge {GAUGE}, turn {TURN}, edge {EDGE}, arrival {ARRIVAL}")
    res = []
    for r in refs:
        want = r["ref"][:WORDS]
        got, lg = forward(r["ids"] + want[:-1], last=WORDS)
        bad = [i for i in range(WORDS) if got[i] != want[i]]
        own = None
        if bad:                                       # run on with the machine's own words from the first difference
            own = want[:bad[0]] + [got[bad[0]]]
            while len(own) < WORDS: own.append(forward(r["ids"] + own)[0][0])
        top2 = np.sort(lg, axis=1)[:, -2:]; mm = ((top2[:, 1] - top2[:, 0]) / 2.0 ** ES).tolist()   # the machine's own lead of word 1 over word 2
        res.append(dict(prompt=r["prompt"], ref=want, got=got, agree=WORDS - len(bad), differ_at=bad, machine_margins=mm,
                        margin_moved=max(abs(a - b) for a, b in zip(mm, r["margins"])),
                        margins_at_differ=[r["margins"][i] for i in bad], own=own, piles_sha256=hashlib.sha256(lg.tobytes()).hexdigest()))
        print(f"  {r['prompt']!r:44s} {'IDENTICAL' if not bad else 'DIFFERENT'} {WORDS - len(bad):2d}/{WORDS}  {tok.decode(own or got)!r}"
              + ("" if not bad else f"\n  {'':44s} fp32 {'':10s}{tok.decode(want)!r}   (differs at {bad}, fp32 margin {[r['margins'][i] for i in bad]})"), flush=True)
    n = len(res); agree = sum(x["agree"] for x in res); same = sum(not x["differ_at"] for x in res)
    ctrl = sum(int(a == b) for i, x in enumerate(res) for a, b in zip(x["got"], res[(i + 1) % n]["ref"]))   # pairing broken
    print(f"words equal to fp32 Pythia's own (its words fed in): {agree} of {n * WORDS}; texts identical over all {WORDS}: {same} of {n}")
    print(f"shuffle control (each text's machine words against the NEXT text's fp32 words): {ctrl} of {n * WORDS}")
    mv = max(x["margin_moved"] for x in res); small = sorted((m, i, j) for i, r in enumerate(refs) for j, m in enumerate(r["margins"][:WORDS]))[:3]
    print(f"the lead of word 1 over word 2 (logits) moved from fp32's by at most {mv:.5f} at any of the {n * WORDS} positions; "
          f"fp32's three closest: " + ", ".join(f"{m:.5f} (machine {res[i]['machine_margins'][j]:.5f})" for m, i, j in small))
    tag = "" if (PLATES, ORDER, GAUGE, TURN, EDGE, ARRIVAL, MATMUL) == ("finished", "plain", "int", "int", "int", "int", "float64") else \
          "-" + "-".join(x for x in [PLATES if PLATES != "finished" else "", ORDER if ORDER != "plain" else "", "matmulint" if MATMUL == "int" else ""] +
                         [f"{k}float" for k, v in (("gauge", GAUGE), ("turn", TURN), ("edge", EDGE), ("arrival", ARRIVAL)) if v == "float"] if x)
    dst = os.path.join(HERE, f"words_pythia-{SIZE}{tag}.json")
    json.dump(dict(size=SIZE, plates=PLATES, order=ORDER, switches=dict(gauge=GAUGE, turn=TURN, edge=EDGE, arrival=ARRIVAL),
                   agree=agree, positions=n * WORDS, identical_texts=same, texts=n, shuffle_control=ctrl,
                   margin_moved_max=mv, texts_detail=res), open(dst, "w"), indent=0)
    base = os.path.join(HERE, f"words_pythia-{SIZE}.json")
    if (ORDER == "shuffle" or MATMUL == "int") and os.path.exists(base):
        b = json.load(open(base))["texts_detail"]
        print(f"{'order-free check' if ORDER == 'shuffle' else 'int64 sums check'} against the plain run: words the same in {sum(x['got'] == y['got'] for x, y in zip(res, b))} of {n} texts, "
              f"every head pile the same whole number in {sum(x['piles_sha256'] == y['piles_sha256'] for x, y in zip(res, b))} of {n}")
