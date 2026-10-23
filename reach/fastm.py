"""reach/fastm.py: THE BASE-2 MACHINE ON EVERY POSITION AT ONCE (the library for reach/).

Every step is base2.py's own function, read from base2.py's source (gauge_log, to16, product, power, rshift, edge,
log2fix), with its plate list emptied so each plate is read from base2_gpt2.npz only when its bank is met. The one
change of form: base2's arrival runs its race one row at a time; here the race's result for every row is found at
once (the row's largest score; ties give the same distance, 0, so the winner's index never enters). Everything else
in the arrival is base2's: the clock starts at the largest score, each earlier position's distance below it reads
the arrival table (0 past its end), the values are met by the arrivals, and the total's log is negated.

The texts: the 30 refs texts (ids + ref, 648 positions; the top is read at len(ids)-1 .. len(ids)+14, 480 positions)
and judge5's 300 windows (24 positions each, 7,200; the top is read at 3..22, 6,000 positions). Every position of
every text runs through the 12 layers (each text's arrivals read only its own positions).

run(texts, plates=None, hook=None): layer by layer over all texts at once.
  plates: {bank name: whole-number plate (lines x neurons)} to use in place of base2_gpt2.npz's.
  hook(bank, l, d): called at every bank with d = dict(x16, ex, LS (None for plain banks), pile, we, kept) where kept
    is what the bank hands on (after its water level or shift and its bias), and with ("stream a"/"stream", l, R).
  Returns (x16, ex, LS) at the top for every position.
"""
import os, sys, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags (light_top.py does the same)
HERE = "/Users/abundancemachine/streaming-machine"
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
for _a, _b in [("import sys, math, numpy as np, torch", "import sys, math, numpy as np"),
               ("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", ""),
               ('M = np.load("base2_gpt2.npz")', f'M = np.load("{HERE}/base2_gpt2.npz")'),
               ('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]')]:
    assert _a in _src, _a; _src = _src.replace(_a, _b)
B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv
assert B["W"] == {}
M = B["M"]; ES, SC = B["ES"], B["SC"]
ARR, PHI, LOG2, EXP2 = (np.asarray(B[k]) for k in ("ARR", "PHI", "LOG2", "EXP2"))
rshift, to16, product, race, log2fix, power, gauge_log, edge = (B[k] for k in
    ("rshift", "to16", "product", "race", "log2fix", "power", "gauge_log", "edge"))
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
PLATES = BANKS + ["head"]
E_ = {n: M[n + ".e"] for n in PLATES}; B_ = {n: M[n + ".b"] for n in PLATES}
WPE = M["wpe"]
def plate(name): return M[name + ".w"]                         # int16, lines x neurons

REFS = json.load(open(f"{HERE}/refs_many.json")); REF_TEXTS = [r["ids"] + r["ref"] for r in REFS]
REF_READ = [np.arange(len(r["ids"]) - 1, len(r["ids"]) + 15) for r in REFS]          # 16 a text, 480 in all
WINDOWS = json.load(open(f"{HERE}/judge5_texts.json")); WIN_TEXTS = [w["ids"] for w in WINDOWS]
WIN_READ = [np.arange(3, 23) for _ in WINDOWS]                                         # 20 a window, 6,000 in all
TEXTS = REF_TEXTS + WIN_TEXTS; READ = REF_READ + WIN_READ
LENS = np.array([len(t) for t in TEXTS]); OFF = np.concatenate([[0], np.cumsum(LENS)])
READ_ROWS = np.concatenate([OFF[t] + READ[t] for t in range(len(TEXTS))])             # 6,480 rows of the 7,848
NREF_ROWS = 480
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6

def log2fix_v(V):                                              # base2's log2fix for a whole array (values below 2^53)
    V = np.asarray(V, np.int64); assert (V > 0).all() and (V < (1 << 53)).all()
    t = np.frexp(V.astype(np.float64))[1].astype(np.int64) - 1     # the top 1's place (exact below 2^53)
    m = np.where(t >= 16, np.right_shift(V, np.maximum(t - 16, 0)), np.left_shift(V, np.maximum(16 - t, 0))) & 0xFFFF
    return t * 65536 + LOG2[m]

def attention(QKV):                                            # base2's arrival, every row's race at once
    P = QKV.shape[0]; out = np.zeros((P, 768), dtype=np.int64); low = np.tril(np.ones((P, P), bool))
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        top = np.where(low, sc, np.iinfo(np.int64).min).max(1)
        d = np.where(low, top[:, None] - sc, len(ARR))
        a = np.where(d < len(ARR), ARR[np.clip(d, 0, len(ARR) - 1)], 0)
        num = product(a, v.astype(np.float64))
        out[:, h*64:(h+1)*64] = power(num, -log2fix_v(a.sum(1))[:, None])
    return out

ROWS = 1024                                                    # elementwise steps a piece of rows at a time (same arithmetic)
def _rows(fn, n): return np.concatenate([fn(slice(r, r + ROWS)) for r in range(0, n, ROWS)])
def gauged_kept(pile, we, ex, LS, b):
    return _rows(lambda r: power(pile[r], (we[None, :] + ex[r, None] + ES) * 65536 - LS[r, None]) + b, len(pile))
def plain_kept(pile, we, ex, b):
    return _rows(lambda r: rshift(pile[r], -(we[None, :] + ex[r, None] + ES)) + b, len(pile))
def edge_rows(Hp): return _rows(lambda r: edge(Hp[r]), len(Hp))
def to16_rows(X, e):                                           # base2's to16, a piece of rows at a time (each row its own)
    parts = [to16(X[r0:r0 + ROWS], e) for r0 in range(0, len(X), ROWS)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
def product_lean(x16, w):                                      # base2's product: the same float64 sums, the same check
    p = x16.astype(np.float64) @ w
    assert max(p.max(initial=0), -p.min(initial=0)) < 2 ** 52
    return p.astype(np.int64)

def run(texts=None, plates=None, hook=None, log=None, start=None):
    """start: (l, half, R) to begin at layer l (half 0: before qkv, 1: before fc) from the stream R."""
    texts = TEXTS if texts is None else texts; plates = plates or {}
    lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)]); t0 = time.time()
    if start is None:
        WTE = M["wte"]; R = np.concatenate([WTE[np.asarray(ids)] + WPE[:len(ids)] for ids in texts]); l0, h0 = 0, 0; del WTE
    else: l0, h0, R = start
    def W(name): return np.asarray(plates[name] if name in plates else plate(name)).astype(np.float64)
    def H(*a):
        if hook is not None: hook(*a)
    for l in range(l0, 12):
        if not (l == l0 and h0 == 1):
            H("stream in", l, R)
            x16, ex, LS = gauge_log(R); n = f"qkv.{l}"; we = E_[n]; pile = product_lean(x16, W(n))
            QKV = gauged_kept(pile, we, ex, LS, B_[n])
            H(n, l, dict(x16=x16, ex=ex, LS=LS, pile=pile, we=we, kept=QKV)); del pile
            o = np.concatenate([attention(QKV[off[t]:off[t + 1]]) for t in range(len(texts))]); del QKV
            x16, ex = to16_rows(o, -ES); n = f"proj.{l}"; we = E_[n]; pile = product_lean(x16, W(n))
            att = plain_kept(pile, we, ex, B_[n])
            H(n, l, dict(x16=x16, ex=ex, LS=None, pile=pile, we=we, kept=att)); del pile, o
            R = R + att; H("stream a", l, R); del att
        x16, ex, LS = gauge_log(R); n = f"fc.{l}"; we = E_[n]; pile = product_lean(x16, W(n))
        Hp = gauged_kept(pile, we, ex, LS, B_[n])
        H(n, l, dict(x16=x16, ex=ex, LS=LS, pile=pile, we=we, kept=Hp)); del pile
        Hb = edge_rows(Hp); del Hp
        x16, ex = to16_rows(Hb, -ES); n = f"mproj.{l}"; we = E_[n]; pile = product_lean(x16, W(n))
        mo = plain_kept(pile, we, ex, B_[n])
        H(n, l, dict(x16=x16, ex=ex, LS=None, pile=pile, we=we, kept=mo)); del pile, Hb
        R = R + mo; H("stream", l, R); del mo
        if log: log(f"    layer {l:2d} ({time.time() - t0:.0f}s, {rss()} MB)")
    return gauge_log(R)

def top_scores(x16, ex, LS, head=None, rows_per=128):
    """base2's read at the given positions: power(product(x16, head), (head.e + ex + ES) x 65536 - LS) + standing
    amount. Yields (row slice, scores) a piece of positions at a time."""
    Wh = np.asarray(plate("head") if head is None else head).astype(np.float64); he, hb = E_["head"], B_["head"]
    for r0 in range(0, len(x16), rows_per):
        sl = slice(r0, r0 + rows_per)
        S = power(product(x16[sl], Wh), (he[None, :] + ex[sl, None] + ES) * 65536 - LS[sl, None]) + hb
        yield sl, S
        del S

def top6(S):                                                   # the top 6 in order (judge5's top5 rule, one further)
    idx = np.argpartition(-S, 6, axis=1)[:, :6]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, 1)
def top5(S):                                                   # judge5.top5, the same two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)

def signed_digits(v):                                          # (up, down) place masks; no two lit side by side
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a
    up, dn = (t & ~a) >> 1, (a & ~t) >> 1
    neg = v < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
