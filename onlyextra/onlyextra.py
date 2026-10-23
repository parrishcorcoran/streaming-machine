"""ONLY THE EXTRA: EVERY WEIGHT KEEPS ONLY ITS TOP PHOTONS, EVERY LIGHT VALUE MEETS WITH ONLY ITS TOP PHOTONS.

HIS WORDS (2026-10-08), the only sentences quoted as his:
  "TERNARY and 4 bit quantization proves it's not necessary to store all the extras."
  "Ternary is 10x smaller and is the worst way to get rid of the extra. We already proved the extra is all we need"
  Earlier: "THE FLAT IS THE CANCELLING."  "it has to be BOTH SIDES"  "The embedding table and the mlp"

THE PICTURE (the session's restatement): every weight and every light value is a stack of photons on circles of 2.
The low photons are speckle that cancels in every pile (below the top 5-7 diagonals the tips match the coin toss,
blank-slate/docs/08). The extra stands above the flat in the top corner, where the light's top photons meet the plate's
top photons. So store only each weight's top photons, meet only the light's top photons -- both sides -- and the token
should come through.

THE MACHINE: base2.py's own steps (its source exec'd up to its own test, as tagskip.py loads it), base2_gpt2.npz, all
positions of a batch of texts through each plate at once, the arrival one text at a time (whole numbers, identical to
base2's per-position loop: the clock starts at the largest score, which is all the race's winner is used for there).
  PLATE SIDE (k): every weight of the 48 plates, and every value of the word table, written in single waves (each lit
    place up +1 or a hole -1, no two lit places side by side: up = a3 & c, hole = ah & c with ah = a >> 1, a3 = a + ah,
    c = ah ^ a3), keeps only its top k lit places; the rest are dropped. k = 1, 2, 3, fixed now.
    The top table is printed from the cut word table through the 768 whole-number gains, as the machine builds it
    (shrink/develop_lib.print_top: s = G x E; c = 768 s - sum_lines s; y = (c + 2^23) >> 24; T = (y x 1431655766 + 2^43)
    >> 44, the gains written once at 2^24 in shrink/develop_gain24.npy).
  LIGHT SIDE (j): at every plate and at the top read, each light value entering the plate (after base2's to16) keeps
    only its top j lit places in single waves; j = full (no cut), 3, 2. The water level (gauge) is read from the light
    as base2 makes it; the arrival's own products are not plates and are left whole.
  Nothing else changes: every bend, gauge, divide, arrival and bias is base2's.

THE TOKEN: judge5's 300 windows (6,000 judged positions): the full machine's word in the case's top 5 (asked), the
  case's word in the full machine's top 5 (judge5's own measure), the same top word; the 30 refs texts: 16 greedy
  words, against the full base-2 machine's own greedy words and against fp32's (refs_many.json).
STORED SIZE, at the floor, for exactly what is kept (analysis, entropies from static counts, no coder run):
  chain: per value, where its top kept photon sits and its sign (or dark), counts per plate; then for each further
  kept photon the gap down from the photon above (2 or more) and up/hole, or stop, counts per plate and per place of
  the photon above (the reader knows k, so no stop after the k-th). Joint: each kept value as one symbol, counts per
  plate. Each count cell charged 40 bits. The lens (768 gains, 2,892 bytes) is charged to the word table.
WORK per token: the meetings of kept light photons with kept plate photons, lit(light) x lit(weight) over every product
  of the 48 plates and the top read, per position; against the full machine in plain base 2 (ones x ones, the records'
  count) and in single waves.
CONTROLS: random: each weight keeps k of its lit places chosen at random (the same count), j = full.
  shuffle: the top-k values put on random places within their own plate (and the word table within itself), j = full.
  order: k = 2, j = 3 with every plate's input lines reordered together with the light (the gauge read on the reordered
  light) and the top read's words relabelled: every word and every count must come out identical.
Predictions: onlyextra/PREDICTION.txt (written before this file ran).
    python3 onlyextra/onlyextra.py [cases=name,name,...] [nwin=300]
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")     # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/onlyextra"
ARG = dict(a.split("=", 1) for a in sys.argv[1:])
NWIN = int(ARG.get("nwin", 300)); TAG = ARG.get("tag", "")
LOG = open(f"{OUT}/onlyextra{TAG}.log", "a"); t0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb():                                          # peak so far / now, MB
    now = int(os.popen(f"ps -o rss= -p {os.getpid()}").read().strip() or 0) // 1000
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} peak, {now} now"
def el(): return int(time.time() - t0)

# ---------------- base2.py's own steps (its source up to its own test; the plates are read below as int16) ----------------
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')
_argv, _cwd = sys.argv, os.getcwd(); sys.argv = ["x"]; os.chdir(HERE); J = {}; exec(_src, J); sys.argv = _argv; os.chdir(_cwd)
M, ES, SC, ARR = J["M"], J["ES"], J["SC"], J["ARR"]
rshift, power, edge, to16, gauge_log, product, log2fix, race = (J[k] for k in
    ("rshift", "power", "edge", "to16", "gauge_log", "product", "log2fix", "race"))
assert J["EDGE"] == "int" and J["GAUGE"] == "int" and J["ARRIVAL"] == "int"
NAMES = [f"{k}.{l}" for l in range(12) for k in ("qkv", "proj", "fc", "mproj")]
SHAPE = {n: tuple(M[f"{n}.w"].shape) for n in NAMES}                         # the plates are read from the file per case
EXP = {n: M[f"{n}.e"].astype(np.int64) for n in NAMES}; BIAS = {n: M[f"{n}.b"].astype(np.int64) for n in NAMES}
_w = M["wte"]; assert int(_w.max()) < 1 << 30 and int(_w.min()) > -(1 << 30)
WTE = _w.astype(np.int32); del _w; WPE = M["wpe"].astype(np.int64)
HE = M["head.e"].astype(np.int64); HB = M["head.b"].astype(np.int64)
assert (HE == HE[0]).all()
GAIN = np.load(f"{HERE}/shrink/develop_gain24.npy").astype(np.int64)
REFS = json.load(open(f"{HERE}/refs_many.json")); WINDOWS = json.load(open(f"{HERE}/judge5_texts.json"))
REF5 = np.load(f"{HERE}/judge5_ref.npz")["top5"]
NV_PLATES = sum(int(np.prod(SHAPE[n])) for n in NAMES); NV_E = WTE.size; NV = NV_PLATES + NV_E
CH = 4096

# ---------------- single waves: up or hole, never two side by side ----------------
def naf_split(a):                                   # a >= 0 -> (up, hole), a = up - hole
    ah = a >> 1; a3 = a + ah; c = ah ^ a3
    return a3 & c, ah & c

def lit_count(V):                                   # lit places of each value in single waves
    up, hole = naf_split(np.abs(np.asarray(V, dtype=np.int64)))
    return np.bitwise_count(up | hole).astype(np.int64)

def ones_count(V):                                  # lit places of each value in plain base 2
    return np.bitwise_count(np.abs(np.asarray(V, dtype=np.int64))).astype(np.int64)

def top_k(V, k):
    """each value keeps its top k lit places (single waves); the rest dropped"""
    V = np.asarray(V, dtype=np.int64); a = np.abs(V); up, hole = naf_split(a); lit = up | hole
    cnt = np.bitwise_count(lit).astype(np.int64)
    for _ in range(max(int(cnt.max()) - k, 0) if cnt.size else 0):
        sel = cnt > k; lit = np.where(sel, lit & (lit - 1), lit); cnt = cnt - sel   # the lowest lit place put out
    kept = (up & lit) - (hole & lit)
    return np.where(V < 0, -kept, kept)

def random_k(V, k, rng):
    """CONTROL: each value keeps k of its lit places chosen at random (all of them when it has k or fewer)"""
    V = np.asarray(V, dtype=np.int64); a = np.abs(V); up, hole = naf_split(a); lit = up | hole
    cnt = np.bitwise_count(lit).astype(np.int64)
    while True:
        sel = np.flatnonzero(cnt > k)
        if sel.size == 0: break
        l = lit[sel]; r = rng.integers(0, cnt[sel]); run = np.zeros(sel.size, np.int64); pick = np.zeros(sel.size, np.int64)
        for b in range(26):                         # the r-th lit place from the bottom
            bit = (l >> b) & 1; pick |= np.where((bit == 1) & (run == r), np.int64(1) << b, 0); run += bit
        assert np.bitwise_count(pick).max() == 1
        lit[sel] = l ^ pick; cnt[sel] -= 1
    kept = (up & lit) - (hole & lit)
    return np.where(V < 0, -kept, kept)

def print_top(E2):                                  # shrink/develop_lib.print_top, whole numbers
    out = np.empty(E2.shape, np.int32)
    for r in range(0, E2.shape[0], CH):
        sv = E2[r:r + CH].astype(np.int64) * GAIN[None, :]
        c = 768 * sv - sv.sum(1, keepdims=True); y = (c + (1 << 23)) >> 24
        assert int(np.abs(y).max()) < (1 << 32)
        out[r:r + CH] = (y * 1431655766 + (1 << 43)) >> 44
    return out

# ---------------- stored size at the floor ----------------
def bits_of(counts):
    """total bits of symbols at their static floor; counts: (contexts, symbols). And the count cells used."""
    c = np.asarray(counts, dtype=np.float64); tot = c.sum(1, keepdims=True); nz = c > 0
    return float(-(c[nz] * np.log2((c / np.where(tot > 0, tot, 1))[nz])).sum()), int(nz.sum())

def place_top(x):                                   # place of the top lit place, plus one (0 for 0)
    t = np.zeros(x.shape, np.int64); x = x.copy()
    for s in (32, 16, 8, 4, 2, 1):
        m = (x >> s) > 0; t += m * s; x = np.where(m, x >> s, x)
    return t + (x > 0)

ZZ = 1 << 22                                        # zigzag range for the joint symbol (checked)
def floor_counts(V, k, acc):
    """add one slice's counts. V: kept values (at most k lit places each).
    round 1: top kept place + sign, or dark (counts per plate); round r: the gap down from the photon above (2+) and
    up/hole, or stop; counts per place of the photon above. joint: the kept value as one symbol."""
    V = np.asarray(V, dtype=np.int64).ravel(); a = np.abs(V); up, hole = naf_split(a); lit = up | hole
    assert np.array_equal(up - hole, a) and int(np.bitwise_count(lit).max()) <= k
    t = place_top(lit)
    acc["r1"] += np.bincount(np.where(t > 0, 2 * t - 1 + (V < 0), 0), minlength=64)[None, :64]
    prev = t - 1; alive = t > 0; rest = np.where(alive, lit ^ (np.int64(1) << np.maximum(prev, 0)), 0)
    for r in range(2, k + 1):
        room = alive & (prev >= 2); l = rest[room]; pr = prev[room]; tt = place_top(l); has = tt > 0
        hb = np.where(has, (hole[room] >> np.maximum(tt - 1, 0)) & 1, 0)
        sym = np.where(has, 1 + 2 * (pr - (tt - 1) - 2) + hb, 0)
        assert sym.size == 0 or (int(sym.max()) < 64 and int(pr.max()) < 64)
        acc[f"r{r}"] += np.bincount(pr * 64 + sym, minlength=64 * 64)[:64 * 64].reshape(64, 64)
        prev[room] = np.where(has, tt - 1, -1); alive[~room] = False; alive[room] = has
        rest[room] = np.where(has, l ^ (np.int64(1) << np.maximum(tt - 1, 0)), 0)
    assert not (alive & (rest != 0)).any()
    z = (V << 1) ^ (V >> 63); assert int(z.max()) < ZZ
    acc["joint"] += np.bincount(z, minlength=ZZ)[None, :]

def floor_bits(V, k, rows=CH):
    """(chain bits, chain cells, joint bits, joint cells) for one plate, a slice of rows at a time"""
    acc = {"r1": np.zeros((1, 64), np.int64), "joint": np.zeros((1, ZZ), np.int64),
           **{f"r{r}": np.zeros((64, 64), np.int64) for r in range(2, k + 1)}}
    for r0 in range(0, V.shape[0], rows): floor_counts(V[r0:r0 + rows], k, acc)
    cb = cc = 0
    for key in ["r1"] + [f"r{r}" for r in range(2, k + 1)]:
        b, c = bits_of(acc[key]); cb += b; cc += c
    jb, jc = bits_of(acc["joint"])
    return cb, cc, jb, jc

# ---------------- the machine ----------------
def cut_light(x16, j):
    return x16 if j is None else top_k(x16, j)

class Work:
    def __init__(s): s.plates = 0; s.plates_ones = 0; s.top = 0; s.top_ones = 0; s.npos = 0; s.nrows = 0
def met(x, rowlit): return int((lit_count(x).sum(0).astype(np.float64) @ rowlit.astype(np.float64)))
def met1(x, rowones): return int((ones_count(x).sum(0).astype(np.float64) @ rowones.astype(np.float64)))

def arrival(QKV):                                   # base2's arrival, all positions of one text at once
    P = QKV.shape[0]; out = np.zeros((P, 768), np.int64); tri = np.tril(np.ones((P, P), bool))
    for h in range(12):
        q, k, v = QKV[:, h * 64:(h + 1) * 64], QKV[:, 768 + h * 64:768 + (h + 1) * 64], QKV[:, 1536 + h * 64:1536 + (h + 1) * 64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        mx = np.where(tri, sc, np.iinfo(np.int64).min).max(1)            # the clock starts at the first to arrive
        d = mx[:, None] - sc
        a = np.where(tri & (d < len(ARR)), ARR[np.clip(d, 0, len(ARR) - 1)], 0)
        num = product(a, v.astype(np.float64)); Lp = log2fix(a.sum(1))
        out[:, h * 64:(h + 1) * 64] = power(num, -Lp[:, None])
    return out

def forward(texts, C, top, W=None):
    """C: the case (plates, E, Tf, j, counts; C["order"]: the reordering check). top: 'judged' | 'last'. W: Work or None."""
    j = C["j"]; P8 = C["plates"]; order = C.get("order")
    npos = np.array([len(t) for t in texts]); off = np.concatenate([[0], np.cumsum(npos)]); NP = int(off[-1])
    ids = np.concatenate([np.asarray(t, np.int64) for t in texts]); pos = np.concatenate([np.arange(n) for n in npos])
    R = C["E"][ids].astype(np.int64) + WPE[pos]
    if W is not None: W.npos += NP
    def bank(n, X, gauged):
        w = P8[n]; we, b = EXP[n], BIAS[n]; rl, ro = C["rowlit"][n], C["rowones"][n]
        if order is not None:                       # the plate's lines reordered, the light's lines with them
            LP = order[n]; X = X[:, LP]; w = w[LP]; rl, ro = rl[LP], ro[LP]
        if gauged: x16, ex, LS = gauge_log(X)
        else: x16, ex = to16(X, -ES)
        xc = cut_light(x16, j); pile = product(xc, w.astype(np.float64))
        if W is not None: W.plates += met(xc, rl); W.plates_ones += met1(xc, ro)
        if gauged: return power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        return rshift(pile, -(we[None, :] + ex[:, None] + ES)) + b
    for l in range(12):
        QKV = bank(f"qkv.{l}", R, True)
        o = np.concatenate([arrival(QKV[off[t]:off[t + 1]]) for t in range(len(texts))]); del QKV
        R = R + bank(f"proj.{l}", o, False)
        G = edge(bank(f"fc.{l}", R, True))
        R = R + bank(f"mproj.{l}", G, False)
    rows = {"last": off[1:] - 1, "judged": np.concatenate([np.arange(off[t] + 3, off[t] + 23) for t in range(len(texts))])}[top]
    X = R[rows]; Tf = C["Tf"]                       # under the check, Tf, toplit, topones and HB were reordered once
    if order is not None: X = X[:, order["top_lines"]]
    x16, ex, LS = gauge_log(X); xc = cut_light(x16, j); Fp = (int(HE[0]) + ex + ES) * 65536 - LS
    if W is not None:
        W.nrows += len(rows); W.top += met(xc, C["toplit"]); W.top_ones += met1(xc, C["topones"])
    T5, WIN = [], []
    HBo = HB if order is None else C["HBo"]
    for p0 in range(0, len(rows), 64):
        S = power(product(xc[p0:p0 + 64], Tf), Fp[p0:p0 + 64, None]) + HBo[None, :]
        if order is not None:
            S2 = np.empty_like(S); S2[:, order["words"]] = S; S = S2
        idx = np.argpartition(-S, 5, axis=1)[:, :5]; od = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
        T5.append(np.take_along_axis(idx, od, 1)); WIN.append(np.array([race(s) for s in S]))
    return np.concatenate(T5), np.concatenate(WIN)

# ---------------- building a case ----------------
def build(spec):
    """spec: k (None = full), j (None = full), how ('top' | 'random' | 'shuffle'), where ('both' | 'plates' | 'table'),
    printed (print T from E even when E is whole)"""
    k, how, where = spec.get("k"), spec.get("how", "top"), spec.get("where", "both")
    rng = np.random.default_rng(spec.get("seed", 1))
    C = dict(j=spec.get("j"), plates={}, rowlit={}, rowones={}, size={})
    lit_full = lit_kept = changed = 0; pbits = pcells = jbits = jcells = 0.0
    for n in NAMES:
        w = M[f"{n}.w"].astype(np.int64)
        if k is not None and where in ("both", "plates"):
            wc = top_k(w, k) if how in ("top", "shuffle") else random_k(w, k, rng)
            if how == "shuffle": wc = wc.ravel()[rng.permutation(wc.size)].reshape(wc.shape)
        else: wc = w
        lw = lit_count(wc); lit_kept += int(lw.sum()); lit_full += int(lit_count(w).sum()); changed += int((wc != w).sum())
        C["rowlit"][n] = lw.sum(1); C["rowones"][n] = ones_count(wc).sum(1)
        if k is not None and where in ("both", "plates") and how != "shuffle" and spec.get("size", True):
            cb, cc, jb, jc = floor_bits(wc, k); pbits += cb; pcells += cc; jbits += jb; jcells += jc
        C["plates"][n] = wc.astype(np.int32)
    log(f"    plates built ({mb()} MB)")
    C["size"]["plates"] = dict(lit_full=lit_full / NV_PLATES, lit_kept=lit_kept / NV_PLATES, changed=changed / NV_PLATES,
                               chain_bits=pbits, chain_cells=pcells, joint_bits=jbits, joint_cells=jcells)
    if k is not None and where in ("both", "table"):
        E = np.empty(WTE.shape, np.int32)
        for r in range(0, WTE.shape[0], CH):
            e = WTE[r:r + CH].astype(np.int64)
            E[r:r + CH] = top_k(e, k) if how in ("top", "shuffle") else random_k(e, k, rng)
        if how == "shuffle": E = E.ravel()[rng.permutation(E.size)].reshape(E.shape)
        T = print_top(E)
    else:
        E = WTE; T = print_top(E) if spec.get("printed") else M["head.w"].T.astype(np.int32)
    elit_full = sum(int(lit_count(WTE[r:r + CH]).sum()) for r in range(0, WTE.shape[0], CH))
    elit_kept = sum(int(lit_count(E[r:r + CH]).sum()) for r in range(0, E.shape[0], CH))
    echanged = sum(int((E[r:r + CH] != WTE[r:r + CH]).sum()) for r in range(0, E.shape[0], CH))
    es = dict(lit_full=elit_full / NV_E, lit_kept=elit_kept / NV_E, changed=echanged / NV_E)
    if k is not None and where in ("both", "table") and how != "shuffle" and spec.get("size", True):
        cb, cc, jb, jc = floor_bits(E, k); es.update(chain_bits=cb, chain_cells=cc, joint_bits=jb, joint_cells=jc)
    C["size"]["table"] = es
    C["E"] = E
    tl = np.zeros(768, np.int64); to = np.zeros(768, np.int64)
    for r in range(0, T.shape[0], CH): tl += lit_count(T[r:r + CH]).sum(0); to += ones_count(T[r:r + CH]).sum(0)
    C["toplit"], C["topones"] = tl, to
    C["size"]["top_lit_per_value"] = float(tl.sum() / T.size)
    H16 = M["head.w"]
    C["size"]["top_changed"] = float(sum(int((T[r:r + CH] != H16.T[r:r + CH]).sum()) for r in range(0, T.shape[0], CH)) / T.size)
    del H16
    log(f"    word table and top table built ({mb()} MB)")
    Tf = np.empty((T.shape[1], T.shape[0]), np.float64)
    for r in range(0, T.shape[0], CH): Tf[:, r:r + CH] = T[r:r + CH].T
    C["Tf"] = Tf; del T
    return C

def judge(C, W):
    inn = full_in = same = so5 = tot = 0; allT5, allWIN = [], []
    for w0 in range(0, NWIN, 25):
        wins = WINDOWS[w0:min(NWIN, w0 + 25)]
        T5, WIN = forward([w["ids"] for w in wins], C, "judged", W)
        rf = REF5[w0:w0 + len(wins)].reshape(-1, 5)
        inn += int((rf == WIN[:, None]).any(1).sum()); full_in += int((T5 == rf[:, :1]).any(1).sum())
        same += int((rf[:, 0] == WIN).sum()); so5 += int((rf == T5).all(1).sum()); tot += len(WIN)
        allT5.append(T5); allWIN.append(WIN)
    seqs = [list(r["ids"]) for r in REFS]; gen = [[] for _ in REFS]
    for s in range(16):
        _, WIN = forward([sq + g for sq, g in zip(seqs, gen)], C, "last", None)
        for i, w in enumerate(WIN): gen[i].append(int(w))
    return dict(in_full_top5=inn, full_word_in_case_top5=full_in, same_top=same, top5_set_order=so5, judged=tot,
                gen=gen), np.concatenate(allT5), np.concatenate(allWIN)

def greedy_vs(gen, want):
    words = sum(int(a == b) for g, wv in zip(gen, want) for a, b in zip(g, wv))
    texts = sum(g == wv for g, wv in zip(gen, want))
    first = [next((i for i, (a, b) in enumerate(zip(g, wv)) if a != b), 16) for g, wv in zip(gen, want)]
    return dict(words=words, texts=texts, before_first_diff=sum(first))

CASES = {
    "full":            dict(k=None, j=None),
    "full_printed":    dict(k=None, j=None, printed=True),
    "light_j3":        dict(k=None, j=3),
    "light_j2":        dict(k=None, j=2),
    **{f"k{k}_j{'full' if j is None else j}": dict(k=k, j=j) for k in (1, 2, 3) for j in (None, 3, 2)},
    **{f"k{k}_plates_only": dict(k=k, j=None, where="plates", size=False) for k in (1, 2)},
    **{f"k{k}_table_only": dict(k=k, j=None, where="table", size=False) for k in (1, 2)},
    **{f"ctrl_random_k{k}": dict(k=k, j=None, how="random", seed=11 + k) for k in (1, 2, 3)},
    **{f"ctrl_shuffle_k{k}": dict(k=k, j=None, how="shuffle", seed=21 + k) for k in (1, 2, 3)},
    "order_k2_j3":     dict(k=2, j=3, order=True, size=False),
}
ORDER = list(CASES)

def pm(a, b): return f"{100.0 * a / b:.2f}%" if b else "-"

if __name__ == "__main__":
    names = ARG["cases"].split(",") if "cases" in ARG else ORDER
    RES = f"{OUT}/results{TAG}.json"
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    log(f"=== ONLYEXTRA {time.strftime('%Y-%m-%d %H:%M:%S')}  cases {names}  windows {NWIN}  (loaded, {mb()} MB)")
    for name in names:
        while True:                                 # one heavy process at a time: wait while another python over 1.5 GB runs
            ps = os.popen("ps -axo pid=,rss=,comm=").read().split("\n")
            other = [p for p in ps if p.strip() and "ython" in p and int(p.split()[0]) != os.getpid() and int(p.split()[1]) > 1_500_000]
            if not other: break
            log(f"  waiting: another python process over 1.5 GB is running ({other[0].split()[0]})"); time.sleep(60)
        spec = CASES[name]; tc = time.time()
        C = build(spec)
        log(f"[{name}] built ({int(time.time() - tc)}s, {mb()} MB): plates lit {C['size']['plates']['lit_full']:.3f} -> "
            f"{C['size']['plates']['lit_kept']:.3f} a weight, changed {100 * C['size']['plates']['changed']:.1f}%; word table lit "
            f"{C['size']['table']['lit_full']:.3f} -> {C['size']['table']['lit_kept']:.3f}, changed {100 * C['size']['table']['changed']:.1f}%; "
            f"top table {C['size']['top_lit_per_value']:.3f} lit a value, {100 * C['size']['top_changed']:.1f}% of its values differ from the machine's")
        order = None
        if spec.get("order"):
            rng = np.random.default_rng(777)
            order = {n: rng.permutation(SHAPE[n][0]) for n in NAMES}
            order["top_lines"] = rng.permutation(768); order["words"] = rng.permutation(50257)
            tl, wd = order["top_lines"], order["words"]
            Tf = C.pop("Tf"); Tn = np.empty_like(Tf)
            for r0 in range(0, 768, 64): Tn[r0:r0 + 64] = Tf[tl[r0:r0 + 64]][:, wd]
            C["Tf"] = Tn; del Tf; C["toplit"], C["topones"] = C["toplit"][tl], C["topones"][tl]
            C["HBo"] = HB[wd]; C["order"] = order
        W = Work()
        r, T5, WIN = judge(C, W)
        np.savez_compressed(f"{OUT}/top5_{name}{TAG}.npz", top5=T5, win=WIN)
        full_gen = res.get("full", {}).get("gen") if name != "full" else r["gen"]
        row = dict(spec={k_: v for k_, v in spec.items()}, **{k_: v for k_, v in r.items()},
                   greedy_vs_fp32=greedy_vs(r["gen"], [x["ref"] for x in REFS]),
                   greedy_vs_full=greedy_vs(r["gen"], full_gen) if full_gen else None,
                   size=C["size"],
                   work=dict(plates=W.plates / W.npos, plates_ones=W.plates_ones / W.npos, top=W.top / W.nrows,
                             top_ones=W.top_ones / W.nrows, npos=W.npos, nrows=W.nrows),
                   seconds=int(time.time() - tc), max_mb=mb())
        if order is not None and "k2_j3" in res:
            ref = np.load(f"{OUT}/top5_k2_j3{TAG}.npz")
            row["order_free"] = dict(top5_identical=bool(np.array_equal(ref["top5"], T5)), win_identical=bool(np.array_equal(ref["win"], WIN)),
                                     greedy_identical=r["gen"] == res["k2_j3"]["gen"],
                                     work_identical=row["work"] == res["k2_j3"]["work"])
        res[name] = row; json.dump(res, open(RES, "w"))
        gv = row["greedy_vs_full"]
        log(f"[{name}] full machine's word in the case's top 5 {r['full_word_in_case_top5']}/{r['judged']} ({pm(r['full_word_in_case_top5'], r['judged'])}); "
            f"case's word in the full top 5 {r['in_full_top5']}/{r['judged']} ({pm(r['in_full_top5'], r['judged'])}); same top word {r['same_top']}/{r['judged']} "
            f"({pm(r['same_top'], r['judged'])}); top 5 in set and order {r['top5_set_order']}"
            + (f"; greedy vs full {gv['words']}/480 words, {gv['texts']}/30 texts" if gv else "")
            + f"; vs fp32 {row['greedy_vs_fp32']['words']}/480, {row['greedy_vs_fp32']['texts']}/30")
        log(f"[{name}] work a token: plates {row['work']['plates']:,.0f} single-wave meetings ({row['work']['plates_ones']:,.0f} in plain ones), "
            f"top read {row['work']['top']:,.0f} ({row['work']['top_ones']:,.0f})"
            + (f"; ORDER-FREE {row['order_free']}" if "order_free" in row else "") + f"  ({row['seconds']}s, {mb()} MB)")
        del C
    log(f"=== done ({el()}s, max memory {mb()} MB)")
