"""ONE COMBINED EXACT MACHINE (the library; run_refs.py, run_windows.py, run_controls.py and run_order.py use it).

The base-2 GPT-2 of base2.py. Every bend, water level and arrival exactly as it is. Three exact savings found earlier,
put together in one machine:

 1. THE ENTRY (extra/extra_layer0.py, V6).
    - qkv.0: the word's row and the position's row are stored rounded the way the water level rounds them, with their
      meetings with the plate (tips). Only the -1/0/+1 carry meets the plate.
    - proj.0: the pair's own value is stored with its tip. Only the rest of the arrival's light meets the plate.
    - fc.0: the word's row and the (position + proj bias) row are stored rounded, with their tips. Only the extra meets
      the plate.
 2. THE LINE-UP OF UPS AND DOWNS (rank_gaps.py, lit_formats.py, and the running-sum count of 2026-10-04).
    - Each neuron's plate is put in order of size once. Stored: that order, and the gap from each weight to the next one
      down (the last gap is the smallest weight itself).
    - The light is poured in that order into the neuron's running sum T_k, the light of every line whose weight is at
      or above rank k. Then pile = sum over k of gap_k x T_k (Abel's sum, exact on whole numbers).
    - Gaps and running sums are written in single waves: up and down pixels, no two lit side by side. Lit pixels of n
      = ones(n ^ 3n).
    - Applied wherever full light meets a plate: the 45 banks past the entry, and what still meets proj.0 and fc.0.
    - Not applied to the qkv.0 carry. The carry is lit on a quarter of the lines, so its running sum would be lit at
      nearly every rank. It meets the plate written in single waves. (Its line-up count is printed beside, not used.)
 3. THE COLLAPSING TOP READ (collapse_top/staged_top.py, the light schedule, which was the cheaper one there).
    - Every word's stamp is read down the light's places 14..11, then 10..7, then 6..3.
    - After each stage, an exact bound covers what the unread places can still add.
    - A word is set aside when five words are surely above it. The words still in are read to the end. Exact top 5.

WORK, in the unit of all earlier results (extra/extra_layer0.py's count, whose whole machine is 3,666,596,259 a token
on the 648 positions of the 30 refs texts):
  - A meeting is light photons x plate photons for the places actually met.
  - An add counts the photons it adds.
  - A divide counts the pile's photons x the gain's photons.
Parts:
  entry    layer 0's qkv.0, proj.0 and fc.0: what still meets their plates, the stored tips added, the splits
  banks    the other 45 banks' meetings
  arrival  scores, shares' total, shares x values, the arrival's divides (unchanged)
  top read the word stamps' meetings
  extra    the line-up's pours into the running sums, its single-wave writing, the top read's bounds
  rest     water levels, bank divides, biases, bends, stream adds, the top's divide and standing amounts (unchanged,
           except that the top's divide and standing amounts are paid for the words read to the end only)
Compares (the top read's set-asides) are stated as a count, not photons.

PREDICTIONS (written before anything here ran):
  The task's: about half the plain machine's photon meetings, every word bit-identical.
  Claude's:
  - Every pile, the stream and every word identical to the plain machine. Abel's sum is exact on whole numbers, and
    the entry and the top read were exact before.
  - The line-up's meetings: 34-38% of the plain meetings at the banks.
  - The pours into the running sums: about 13% more.
  - Writing the running sums in single waves (never counted before): another 15-20% of the plain bank meetings.
  - Top read: about 60% of its 904M, plus 30-60M of bound work (in photons).
  - Arrival and rest unchanged.
  - Whole machine: 55-60% of 3,667M counting meetings and pours; 65-72% with every extra add counted.
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")     # Accelerate's spurious flags; every sum checked
HERE = "/Users/abundancemachine/streaming-machine"

# ---------- the machine's own steps: base2.py, read the way collapse_top/staged_top.py reads it ----------
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
_src = _src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
_banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert _banks in _src; _src = _src.replace(_banks, '[]')                           # plates are read below, as int16
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; B = {}; exec(_src, B); sys.argv = _argv; os.chdir(_cwd)
M = B["M"]; ES, SC, ARR, EXP2, PHI = B["ES"], B["SC"], B["ARR"], B["EXP2"], B["PHI"]
rshift, to16, product, race, log2fix, power, gauge_log = (B[k] for k in ("rshift", "to16", "product", "race", "log2fix", "power", "gauge_log"))
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
ENTRY = ("qkv.0", "proj.0", "fc.0")
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
WINDOWS = json.load(open(f"{HERE}/judge5_texts.json")); FIRST, LAST = 3, 23
WORDS = sorted({t for t in sum(TEXTS, [])} | {t for w in WINDOWS for t in w["ids"]})
IDX = np.full(50257, -1, np.int64); IDX[WORDS] = np.arange(len(WORDS))
_wte = M["wte"]; WTE = _wte[WORDS].copy(); del _wte          # entry rows of the words used (int64 at 2^16)
PL16 = {n: M[n + ".w"] for n in BANKS}                     # the plates as stored: int16, (lines in, neurons out)
EB = {n: (M[n + ".e"], M[n + ".b"]) for n in BANKS}
class _Plates(dict):                                         # base2's W: a bank's float64 copy is made only when read
    def __missing__(self, k): return (PL16[k].astype(np.float64),) + EB[k]
B["W"] = _Plates(); W = B["W"]
PMAX = 24; WPE = M["wpe"][:PMAX].copy()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6   # MB

# ---------- photons ----------
def ones(A): return np.bitwise_count(np.asarray(A, dtype=np.int64)).astype(np.int64)          # 1s of |value|
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return ones(a ^ (3 * a))               # lit pixels in single waves
def _rowsum(f, w): return np.concatenate([f(w[i:i + 64]).sum(1) for i in range(0, w.shape[0], 64)])
HEAD16 = M["head.w"]                                        # (768 lines, 50,257 words) int16: the word stamps
WROW = {n: _rowsum(ones, PL16[n]) for n in BANKS}; WROW["head"] = _rowsum(ones, HEAD16)
NWROW = {"qkv.0": _rowsum(naf, PL16["qkv.0"])}               # the entry plate's lit pixels per line, in single waves
def E_ones(LS): return ones(EXP2[np.bitwise_and(-np.asarray(LS, dtype=np.int64), 0xFFFF)])     # the gain's photons
def water(x16):                                              # the gauge's sums: S1 adds, S2 squares, 768 S2 - S1^2
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    return (ones(x16) ** 2).sum(1) + ones(x16).sum(1) + ones(S1) ** 2 + 2 * ones(S2)
class Count(dict):                                           # key "who|part|name"; who: both / plain / comb / side / ctl
    def add(self, k, v): self[k] = self.get(k, 0) + int(np.sum(v))

# ---------- the line-up: each neuron's plate in order of size, stored once ----------
def lineup_store(w16, how="rank", seed=0):
    w = w16.astype(np.int32)
    if how == "rank": o = np.argsort(~w16, axis=0, kind="stable")                      # largest weight first (~w = -w - 1)
    else: o = np.random.default_rng(seed).permuted(np.tile(np.arange(w.shape[0])[:, None], (1, w.shape[1])), axis=0)  # CONTROL: any order
    ws = np.take_along_axis(w, o, 0); g = ws.copy(); g[:-1] -= ws[1:]                    # gap to the next rank down
    assert g[:-1].min() >= 0 if how == "rank" else True
    ng = naf(g).astype(np.uint8)
    if how == "rank": return o.astype(np.int16), (g[:-1].astype(np.uint16), g[-1].astype(np.int64)), ng
    return o.astype(np.int16), (g[:-1], g[-1].astype(np.int64)), ng
LU = {}; ALT_COUNT = False                                  # ALT_COUNT: also count gaps (single waves) x running sums as they are
def build_lineups(how="rank", perm_seed=None):
    """the stored line-up of every bank. perm_seed: ORDER-FREE check, each bank's lines (plate rows, and the light with
    them at run time) put in a random order first."""
    LU.clear()
    for i, n in enumerate(BANKS):
        perm = None if perm_seed is None else np.random.default_rng(perm_seed + i).permutation(PL16[n].shape[0])
        LU[n] = lineup_store(PL16[n] if perm is None else PL16[n][perm], how, seed=1000 + i) + (perm,)
def lineup(X, name, C=None, part="banks", who="comb", store=None, shuffled=None):
    """pile = sum over ranks of gap x running sum, every position. Counted (C): lit-pixel meetings of gap and running
    sum, both in single waves; the light poured into each neuron's own running sum; the running sums written in single
    waves where they meet a lit gap; the light written in single waves. shuffled: a CONTROL count with each position's
    light shuffled across its lines (counted only; the real pile is returned)."""
    o, g, ng, perm = (store or LU)[name]; X = np.asarray(X, np.int64)
    if perm is not None: X = X[:, perm]
    oi = o.astype(np.intp); P = len(X); out = np.empty((P, o.shape[1]), np.int64)
    label = name if part == "entry" else name.split(".")[0]
    g, glast = g
    if C is not None: ng32 = ng.astype(np.int32); lit = np.concatenate([g != 0, (glast != 0)[None, :]], 0)
    for p in range(P):
        T = np.cumsum(X[p][oi], axis=0)                       # the light at or above each rank, per neuron
        out[p] = (T[:-1] * g).sum(0) + T[-1] * glast
        if C is not None:
            a = np.abs(T)
            C.add(f"{who}|{part}|{label} meetings, line-up", (ng32 * np.bitwise_count(a ^ (3 * a))).sum())
            C.add(f"{who}|extra|line-up: running sums written in single waves", (np.bitwise_count(a) * lit).sum())
            if ALT_COUNT: C.add(f"{'alt' if who == 'comb' else who + '-alt'}|{part}|{label} meetings, line-up, running sums as they are (not rewritten)", (ng32 * np.bitwise_count(a)).sum())
            if shuffled is not None:
                Ts = np.abs(np.cumsum(X[p][shuffled.permutation(X.shape[1])][oi], axis=0))
                C.add(f"ctl|{part}|{label} meetings, line-up, light shuffled across its lines", (ng32 * np.bitwise_count(Ts ^ (3 * Ts))).sum())
    if C is not None:
        C.add(f"{who}|extra|line-up: light poured into each neuron's running sum", o.shape[1] * naf(X).sum())
        C.add(f"{who}|extra|line-up: light written in single waves", ones(X).sum())
    return out

# ---------- the entry's stores (extra/extra_layer0.py): only the words and positions the texts use ----------
BPROJ = EB["proj.0"][1]; _STORE = {}
def exact_meet(X, name):                                     # stored tips: whole-number meetings, every partial sum below 2^53
    w = W[name][0]; assert (np.abs(X).astype(np.float64) @ np.abs(w)).max() < 2 ** 52
    return product(X, w)
def store(kind, s):
    key = (kind, int(s))
    if key not in _STORE:
        if kind == "A": v = rshift(WTE, s)                                   # word rows rounded at s
        elif kind == "B": v = rshift(WPE, s)                                 # position rows rounded at s
        elif kind == "B1": v = rshift(WPE + BPROJ[None, :], s)               # position + proj bias, rounded at s
        elif kind == "TQ": v = exact_meet(store("A", s), "qkv.0")
        elif kind == "PQ": v = exact_meet(store("B", s), "qkv.0")
        elif kind == "TF": v = exact_meet(store("A", s), "fc.0")
        elif kind == "PF": v = exact_meet(store("B1", s), "fc.0")
        _STORE[key] = v
    return _STORE[key]
def take(kind, s, rows):                                     # per position: the stored row at that position's own s
    out = None
    for sv in np.unique(s):
        m = s == sv; t = store(kind, sv)
        if out is None: out = np.zeros((len(rows), t.shape[1]), np.int64)
        out[m] = t[rows[m]]
    return out

# ---------- the plain machine itself (base2.py's own steps), every pile kept for the comparison ----------
_product = B["product"]; _REC = []
def _rec(x, w): r = _product(x, w); _REC.append(r); return r
def plain_run(ids, keep_top=True):
    B["product"] = _rec; out = {}
    try:
        P = len(ids); R = WTE[IDX[np.asarray(ids)]] + WPE[:P]
        for l in range(12):
            _REC.clear(); QKV = B["gauged_bank"](R, f"qkv.{l}"); out[f"qkv pile {l}"] = _REC[0]; out[f"qkv {l}"] = QKV
            o = B["attention"](QKV); out[f"o {l}"] = o
            _REC.clear(); att = B["plain_bank"](o, f"proj.{l}"); out[f"proj pile {l}"] = _REC[0]; out[f"att {l}"] = att
            R = R + att; out[f"stream a {l}"] = R
            _REC.clear(); H = B["gauged_bank"](R, f"fc.{l}"); out[f"fc pile {l}"] = _REC[0]; out[f"fc {l}"] = H
            _REC.clear(); mo = B["plain_bank"](B["edge"](H), f"mproj.{l}"); out[f"mproj pile {l}"] = _REC[0]
            R = R + mo; out[f"stream {l}"] = R
        _REC.clear(); x16, ex, LS = gauge_log(R); out["top light"] = (x16, ex, LS)
        if keep_top:
            out["top scores"] = power(head_pile(x16), (WE[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + BIAS
    finally: B["product"] = _product
    return out

# ---------- the combined machine ----------
def arrival(QKV, C=None):                                    # base2's attention step for step, counted
    P = QKV.shape[0]; out = np.zeros((P, 768), dtype=np.int64)
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        if C is not None: C.add("both|arrival|scores", np.tril(ones(q16) @ ones(k16).T).sum()); ov = ones(v).sum(1)
        for p in range(P):
            row = sc[p, :p+1]; d = row[race(row)] - row
            a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            num = product(a[None, :], v[:p+1].astype(np.float64))[0]; Lp = log2fix([a.sum()])[0]
            out[p, h*64:(h+1)*64] = power(num, -Lp)
            if C is not None:
                C.add("both|arrival|shares' total", ones(a).sum()); C.add("both|arrival|shares x values", (ones(a) * ov[:p+1]).sum())
                C.add("both|arrival|divides", ones(num).sum() * E_ones([Lp])[0])
    return out
def bend(H, C=None):                                         # base2's edge, counted
    idx = rshift(H, ES - 12) + 32768
    share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, PHI[np.clip(idx, 0, 65535)]))
    if C is not None: C.add("both|rest|bends", (ones(H) * ones(share)).sum())
    return rshift(H * share, 16)
def _part(name): return "entry" if name in ENTRY else "banks"
def _plain_meeting(C, name, x16):
    if C is not None: C.add(f"plain|{_part(name)}|{name if name in ENTRY else name.split('.')[0]} meetings, plain", ones(x16) @ WROW[name])
def gauged(R, name, C, pile_fn):                             # a bank after a folded layer norm
    we, b = EB[name]; x16, ex, LS = gauge_log(R)
    if C is not None: C.add("both|rest|water levels", water(x16))
    _plain_meeting(C, name, x16); pile = pile_fn(x16, ex)
    if C is not None: C.add("both|rest|bank divides", ones(pile).sum(1) * E_ones(LS)); C.add("both|rest|biases", len(R) * ones(b).sum())
    return power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b, pile
def plainb(X, name, C, pile_fn):                             # a bank with no layer norm before it
    we, b = EB[name]; x16, ex = to16(X, -ES)
    _plain_meeting(C, name, x16); pile = pile_fn(x16, ex)
    if C is not None: C.add("both|rest|biases", len(X) * ones(b).sum())
    return rshift(pile, -(we[None, :] + ex[:, None] + ES)) + b, pile

def forward(ids, C=None, rec=None, side=True, shuffled=None, parts_rng=None):
    """the combined machine on every position of one text: the stream after layer 11 and the light at the top.
    rec: every pile and value kept for the comparison. parts_rng: ORDER-FREE check, each pile's parts added in a random
    order."""
    def add_parts(parts):
        tot = np.zeros_like(parts[0])
        for i in (range(len(parts)) if parts_rng is None else parts_rng.permutation(len(parts))): tot = tot + parts[i]
        return tot
    def keep(k, v):
        if rec is not None: rec[k] = v
    P = len(ids); wi = IDX[np.asarray(ids)]; assert (wi >= 0).all(); pos = np.arange(P)
    R = WTE[wi] + WPE[:P]
    if C is not None: C.add("both|rest|entry: word + position", ones(WPE[:P]).sum())
    # layer 0, the entry
    def qkv0(x16, ex):
        s = ex + ES; A_, B_ = take("A", s, wi), take("B", s, pos); c = x16 - A_ - B_; assert np.abs(c).max() <= 1
        t1, t2 = take("TQ", s, wi), take("PQ", s, pos)
        wq = PL16["qkv.0"]; m = np.zeros((P, wq.shape[1]), np.int64)
        for p in range(P): m[p] = wq[c[p] == 1].astype(np.int64).sum(0) - wq[c[p] == -1].astype(np.int64).sum(0)   # lit lines pour their plate rows, up or down
        if C is not None:
            C.add("comb|entry|qkv.0 split (stored rounded rows taken off)", ones(A_).sum() + ones(B_).sum())
            C.add("comb|entry|qkv.0 stored tips added", ones(t1).sum() + ones(t2).sum())
            C.add("comb|entry|qkv.0 carry meetings (plate in single waves)", ones(c) @ NWROW["qkv.0"])
            if side:
                Cs = Count(); assert np.array_equal(lineup(c, "qkv.0", Cs, "entry", "side"), m)
                for k, v in Cs.items(): C.add(k, v)
        return add_parts([t1, t2, m])
    QKV, pile = gauged(R, "qkv.0", C, qkv0); keep("qkv pile 0", pile); keep("qkv 0", QKV)
    o = arrival(QKV, C); keep("o 0", o)
    def proj0(o16, eo):
        so = eo + ES; ref = rshift(QKV[:, 1536:], so[:, None])               # the pair's own value, at o's s
        tip = product(ref, W["proj.0"][0])                                    # stored per (word, position, s)
        if C is not None:
            C.add("comb|entry|proj.0 split (own value taken off)", ones(ref).sum()); C.add("comb|entry|proj.0 stored tips added", ones(tip).sum())
        return add_parts([tip, lineup(o16 - ref, "proj.0", C, "entry", shuffled=shuffled)])
    att, pile = plainb(o, "proj.0", C, proj0); keep("proj pile 0", pile); keep("att 0", att)
    R = R + att; keep("stream a 0", R)
    if C is not None: C.add("both|rest|stream adds", ones(att).sum())
    def fc0(x1, e1):
        s1 = e1 + ES; A_, B_ = take("A", s1, wi), take("B1", s1, pos); m = x1 - A_ - B_
        t1, t2 = take("TF", s1, wi), take("PF", s1, pos)
        if C is not None:
            C.add("comb|entry|fc.0 split (stored rounded rows taken off)", ones(A_).sum() + ones(B_).sum())
            C.add("comb|entry|fc.0 stored tips added", ones(t1).sum() + ones(t2).sum())
        return add_parts([t1, t2, lineup(m, "fc.0", C, "entry", shuffled=shuffled)])
    H, pile = gauged(R, "fc.0", C, fc0); keep("fc pile 0", pile); keep("fc 0", H)
    lu = lambda name: (lambda x, e: lineup(x, name, C, shuffled=shuffled))
    mo, pile = plainb(bend(H, C), "mproj.0", C, lu("mproj.0")); keep("mproj pile 0", pile)
    R = R + mo; keep("stream 0", R)
    if C is not None: C.add("both|rest|stream adds", ones(mo).sum())
    for l in range(1, 12):                                   # layers 1-11: every bank through the line-up
        QKV, pile = gauged(R, f"qkv.{l}", C, lu(f"qkv.{l}")); keep(f"qkv pile {l}", pile); keep(f"qkv {l}", QKV)
        o = arrival(QKV, C); keep(f"o {l}", o)
        att, pile = plainb(o, f"proj.{l}", C, lu(f"proj.{l}")); keep(f"proj pile {l}", pile); keep(f"att {l}", att)
        R = R + att; keep(f"stream a {l}", R)
        if C is not None: C.add("both|rest|stream adds", ones(att).sum())
        H, pile = gauged(R, f"fc.{l}", C, lu(f"fc.{l}")); keep(f"fc pile {l}", pile); keep(f"fc {l}", H)
        mo, pile = plainb(bend(H, C), f"mproj.{l}", C, lu(f"mproj.{l}")); keep(f"mproj pile {l}", pile)
        R = R + mo; keep(f"stream {l}", R)
        if C is not None: C.add("both|rest|stream adds", ones(mo).sum())
    x16, ex, LS = gauge_log(R); keep("top light", (x16, ex, LS))
    if C is not None: C.add("both|rest|top water level", water(x16))
    return R, (x16, ex, LS)
KEYS = [f"{k} {l}" for l in range(12) for k in ("qkv pile", "qkv", "o", "proj pile", "att", "stream a", "fc pile", "fc", "mproj pile", "stream")]
def same_as_plain(rec, ref):                                 # every pile and value, and the light at the top
    bad = [k for k in KEYS if not np.array_equal(rec[k], ref[k])]
    if not all(np.array_equal(a, b) for a, b in zip(rec["top light"], ref["top light"])): bad.append("top light")
    return bad

# ---------- the collapsing top read (collapse_top/staged_top.py, light schedule), counted ----------
NW, PLACES, LINES = 50257, 15, 768; FLOOR = -(1 << 62)
WE, BIAS = M["head.e"], M["head.b"]; assert (WE == WE[0]).all()
CHUNKS = [np.arange(i, min(i + 12800, NW)) for i in range(0, NW, 12800)]                  # words, a quarter at a time
def head_pile(X, cols=None):                                 # the plain read's piles (whole numbers, exact), chunk by chunk
    if cols is not None: return _product(np.asarray(X), HEAD16[:, cols].astype(np.float64))
    return np.concatenate([_product(np.asarray(X), HEAD16[:, c].astype(np.float64)) for c in CHUNKS], 1)
_TT = np.ascontiguousarray(HEAD16.T)                         # (NW, 768) int16: each word's stamp
TA = np.abs(_TT.astype(np.int32)).astype(np.uint16); TS = np.sign(_TT).astype(np.int8); del _TT
NT = np.stack([((TA >> b) & 1).sum(1, dtype=np.int64) for b in range(PLACES)], 1)           # m_wb: word w's lines lit at b
CNT_ALL = np.stack([((TA >> b) & 1).sum(0, dtype=np.int64) for b in range(PLACES)])         # words lit at b, per line
PPH = np.bitwise_count(TA).sum(0, dtype=np.int64); assert np.array_equal(PPH, WROW["head"])
BIAS_ONES = ones(BIAS)
_a, _b = np.arange(PLACES)[:, None], np.arange(PLACES)[None, :]
CS = [np.broadcast_to(_a >= PLACES - k, (PLACES, PLACES)).copy() for k in (4, 8, 12)] + [np.ones((PLACES, PLACES), bool)]
NEW = [CS[0]] + [CS[s] & ~CS[s - 1] for s in range(1, 4)]
def _amask(C): return np.array([sum(1 << a for a in range(PLACES) if C[a, b]) for b in range(PLACES)], np.int64)
AMASK = np.stack([_amask(c) for c in CS]); NMASK = np.stack([_amask(c) for c in NEW])
POWA = (1 << np.arange(PLACES)).astype(np.int64); NN = np.arange(LINES + 1); ONES_NN = ones(NN)
def top5(S):                                                 # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

def top_read(X, EX, LS, C=None, keepn=256, batch=48, log=None):
    """the collapsing top read at every position given (X: the light x16 at the top, EX, LS). Returns a dict: top5,
    winner (race), survivors and words entering per stage, pair meetings per stage, the words read to the end (their
    best keepn) with exact scores, and the checks against the plain read (every word read in full)."""
    N = len(X); R = dict(top5=np.zeros((N, 5), np.int64), win=np.zeros(N, np.int64), surv=np.zeros((N, 3), np.int64),
                         enter=np.zeros((N, 3), np.int64), work=np.zeros((N, 4), np.int64), plain=np.zeros(N, np.int64),
                         plain_top5=np.zeros((N, 5), np.int64), plain_win=np.zeros(N, np.int64), fin_idx=[], fin_sc=[], fin_off=[0])
    BAD = dict(bound=0, final_vs_plain=0, pairs_sum=0)
    for c0 in range(0, N, batch):
        X_ = np.asarray(X[c0:c0 + batch], np.int64); P = len(X_); ax, sx = np.abs(X_), np.sign(X_)
        assert ax.max() < 1 << PLACES
        F = (WE[0] + EX[c0:c0 + P] + ES) * 65536 - LS[c0:c0 + P]
        PILE = head_pile(X_)                                 # CHECK ONLY: the plain read, every word in full
        S = power(PILE, F[:, None]) + BIAS[None, :]
        if C is not None:
            C.add("plain|top read|top meetings, plain", ones(X_) @ WROW["head"])
            C.add("plain|rest|top divide (every word)", ones(PILE).sum(1) * E_ones(LS[c0:c0 + P]))
            C.add("plain|rest|top standing amounts (every word)", P * BIAS_ONES.sum())
        del PILE
        CNTD = np.zeros((3, P, NW), np.float64)              # counted piles after stages 1..3 (whole numbers below 2^41)
        for c in CHUNKS:
            for b in range(PLACES):
                live = [s for s in range(3) if AMASK[s, b]]
                if not live: continue
                Pb = (TS[c] * ((TA[c] >> b) & 1).astype(np.int8)).astype(np.float64)   # the stamps' photons at b, with direction
                for s in live: CNTD[s][:, c] += ((sx * (ax & AMASK[s, b])).astype(np.float64) @ Pb.T) * float(1 << b)
                del Pb
        CNTD = CNTD.astype(np.int64)
        ALV = np.zeros((4, P, NW), np.float32); ALV[0] = 1
        for p in range(P):
            q = c0 + p; nx = ((ax[p][None, :] >> np.arange(PLACES)[:, None]) & 1).sum(1); ge = E_ones([LS[q]])[0]
            alive = np.arange(NW)
            for s in range(3):
                unc = ~CS[s]; G = (unc.T * POWA[None, :]) @ np.minimum(nx[:, None], NN[None, :])          # (15 b, 769)
                Gw = G[np.arange(PLACES)[None, :], NT[alive]]; Rw = (Gw << np.arange(PLACES)[None, :]).sum(1)
                c = CNTD[s, p, alive]; lo_p, hi_p = c - Rw, c + Rw
                lo = power(lo_p, F[p]) + BIAS[alive]; hi = power(hi_p, F[p]) + BIAS[alive]
                ex_ = S[p, alive]; BAD["bound"] += int(((lo > ex_) | (ex_ > hi)).sum())
                if C is not None:
                    Q = np.array([ONES_NN[:n].sum() + (LINES + 1 - n) * ones(n) for n in nx])     # sum over n of ones(min(n_a, n))
                    C.add("comb|extra|top read: bound tables built from the light", (unc * Q[:, None]).sum())
                    C.add("comb|extra|top read: bound adds (15 table values, then pile -/+ bound)", ones(Gw).sum() + 2 * ones(Rw).sum())
                    C.add("comb|extra|top read: divides of the bounds", (ones(lo_p).sum() + ones(hi_p).sum()) * ge)
                    C.add("comb|extra|top read: standing amounts on the bounds", 2 * BIAS_ONES[alive].sum())
                    C.add("comp|top read: compares (a count, not photons)", len(alive))
                L5 = np.partition(lo, len(lo) - 5)[len(lo) - 5]
                R["enter"][q, s] = len(alive); alive = alive[hi >= L5]; R["surv"][q, s] = len(alive)
                ALV[s + 1, p, alive] = 1
            low = sx[p] * (ax[p] & AMASK[3, 0] & ~AMASK[2, 0])                       # the light's places not yet read
            pile = CNTD[2, p, alive] + head_pile(low[None, :], alive)[0]             # the words still in, read to the end
            exact = power(pile, F[p]) + BIAS[alive]
            BAD["final_vs_plain"] += int((exact != S[p, alive]).sum())
            if C is not None:
                C.add("comb|rest|top divide (words read to the end)", ones(pile).sum() * ge)
                C.add("comb|rest|top standing amounts (words read to the end)", BIAS_ONES[alive].sum())
            Sp = np.full(NW, FLOOR, np.int64); Sp[alive] = exact
            R["top5"][q] = top5(Sp[None])[0]; R["win"][q] = race(Sp)
            R["plain_top5"][q] = top5(S[p:p + 1])[0]; R["plain_win"][q] = race(S[p])
            k = np.arange(len(alive)) if len(alive) <= keepn else np.argpartition(-exact, keepn - 1)[:keepn]
            ids = alive[k]; o = np.argsort(ids)
            R["fin_idx"].append(ids[o]); R["fin_sc"].append(exact[k][o]); R["fin_off"].append(R["fin_off"][-1] + len(k))
        del CNTD, S
        work = np.zeros((4, P), np.int64); allpairs = np.zeros(P, np.int64); A2 = ALV.reshape(4 * P, NW)
        for b in range(PLACES):
            CN = np.zeros((4 * P, LINES), np.int64)
            for c in CHUNKS: CN += (A2[:, c] @ ((TA[c] >> b) & 1).astype(np.float32)).astype(np.int64)   # words still in, lit at b
            CN = CN.reshape(4, P, LINES)
            for s in range(4):
                Lc = np.bitwise_count(ax & NMASK[s, b]).astype(np.int64)            # light photons at the new places
                work[s] += (Lc * CN[s]).sum(1); allpairs += (Lc * CNT_ALL[b][None, :]).sum(1)
            del CN
        plain = (np.bitwise_count(ax).astype(np.int64) * PPH[None, :]).sum(1)
        BAD["pairs_sum"] += int((allpairs != plain).sum())
        R["work"][c0:c0 + P] = work.T; R["plain"][c0:c0 + P] = plain
        if C is not None:
            for s in range(4): C.add(f"comb|top read|top meetings, stage {s + 1}", work[s])
        del ALV, A2
        if log: log(f"    top read {c0 + P}/{N} positions ({rss()} MB)")
    R["fin_idx"] = np.concatenate(R["fin_idx"]); R["fin_sc"] = np.concatenate(R["fin_sc"]); R["fin_off"] = np.array(R["fin_off"])
    R["bad"] = BAD
    return R

def per_token(total, n): return (total + n // 2) // n
def pct(a, b): return (200 * a + b) // (2 * b)
