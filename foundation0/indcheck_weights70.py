"""SEPARATE CHECKER, weights only, pythia-70m. 2026-10-06. Shares no code with foundation_share.py, foundation_check.py,
foundation_rows.py or pythia_convert.py.

It reads Pythia's own fp16 files (step0 = start, step143000 = finished) and turns every fp16 number into a whole number
at 2^-24 by its bits alone (sign, exponent field, 10 mantissa bits): every fp16 value is an exact multiple of 2^-24, so
nothing is rounded. Everything after that is whole-number arithmetic (int64 with asserted bounds, Python ints past them).

  1. s = (W . W0)/(W0 . W0) per plate and per word table, an exact fraction of whole numbers. Shuffle control: W0's
     values permuted over the whole plate (3 draws of this script's own seeds), the same fraction; the spread a random
     pairing gives, |W| / (|W0| sqrt N).
  2. Rows of the input word table the light never reached: a row passes if every lane satisfies
     |w - c w0| <= 1/2 ulp16(w) + 1/2 c ulp16(w0), c the row's own exact (w . w0)/(w0 . w0) (fp16's own half grain:
     what rounding a row that is exactly c x start would leave). Also counted at a full grain. The same for the output
     table. Shuffle control: each row against another row's start.
  3. The shared 16-place grid rebuilt from the whole numbers (one power of two per neuron, the larger of start and
     finished in |q| <= 32767, rounding half to even) and compared digit for digit with base2_pythia-70m.npz.
     Lit places (signed-digit writing with no two neighbouring nonzero digits, counted by stepping the digits out one at
     a time, not by the popcount identity) of W and of D' = W - round(s x W0) at three plates (qkv.0, fc.0, head), and on
     the input table (D' = W - round(c x W0), c the never-built rows' common share) for the never-built rows, 241 rows
     drawn from the rest, and the whole table.

PREDICTION (written before this script first ran; from the delivered numbers it checks):
  - s per plate equal to foundation_share.json's to 1e-3 relative (the delivered s is on the shared grid, where the start
    is rounded; here nothing is rounded). qkv.0 0.1338, fc.0 0.2834, head 0.01433, embed 0.01555. Every plate's s at
    least 10 x its shuffle spread; shuffle |s| under 4 x the expected spread at every plate.
  - never-built input rows at half a grain: 241 (all 27 padding rows, id 1 '<|padding|>', id 126), the same 241 ids as
    foundation_check.json; their c median 6.118e-4, (max - min)/median under 0.1%. At a full grain also 241. Output
    table 0. Shuffled control 0.
  - grid rebuilt: .w, .s and .e identical to the npz at every plate (0 digits differ).
  - lit places a weight (delivered): qkv.0 W 4.322 D'm 4.828, fc.0 W 4.345 D'm 4.914, head W 4.331 D'm 5.392 (within
    0.005); never-built rows W 1.336 D' 0.008; control rows W 4.344 D' 4.932; whole input table W 111,601,955
    D' 126,571,053 (+13.4%). (Noted before the run: D' has MORE lit places than W on built rows because fp16 keeps 11
    significant bits, so W on the 16-place grid ends in zeros that subtracting round(s x W0) fills. A format fact.)
"""
import os, sys, json, time, resource
import numpy as np
from fractions import Fraction
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(HERE, "models")
LOGF = open(os.path.join(HERE, "indcheck_weights70.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
t0 = time.time()
f1 = safe_open(os.path.join(MD, "pythia-70m-step143000", "model.safetensors"), "np")
f0 = safe_open(os.path.join(MD, "pythia-70m-step0", "model.safetensors"), "np")
L = 6
NAMES = {}
for l in range(L):
    p = f"gpt_neox.layers.{l}."
    NAMES[f"qkv.{l}"] = p + "attention.query_key_value.weight"; NAMES[f"proj.{l}"] = p + "attention.dense.weight"
    NAMES[f"fc.{l}"] = p + "mlp.dense_h_to_4h.weight"; NAMES[f"mproj.{l}"] = p + "mlp.dense_4h_to_h.weight"
NAMES["head"] = "embed_out.weight"; NAMES["embed"] = "gpt_neox.embed_in.weight"

def whole24(a16):
    """fp16 -> (whole number at 2^-24, ulp at 2^-24), from the bits only"""
    b = a16.view(np.uint16).astype(np.int64)
    sg, ex, mn = b >> 15, (b >> 10) & 31, b & 1023
    assert (ex != 31).all()
    mag = np.where(ex == 0, mn, (mn | 1024) << np.maximum(ex - 1, 0))
    ulp = np.where(ex == 0, 1, np.int64(1) << np.maximum(ex - 1, 0))
    return np.where(sg == 1, -mag, mag), ulp

def xdot(x, y):
    """exact sum of x*y over all elements (int64 products, chunked so no partial sum can pass 2^62)"""
    x, y = x.ravel(), y.ravel()
    mp = int(np.abs(x).max(initial=0)) * int(np.abs(y).max(initial=0))
    assert mp < 2 ** 62
    step = max(1, (2 ** 62) // max(mp, 1)); tot = 0
    for a in range(0, x.size, step):
        tot += int(np.dot(x[a:a + step], y[a:a + step]))
    return tot

def rowdot(x, y):
    """exact per-row sums, int64 (bound asserted)"""
    assert int(np.abs(x).max(initial=0)) * int(np.abs(y).max(initial=0)) * x.shape[1] < 2 ** 62
    return (x * y).sum(1)

SHARE = json.load(open(os.path.join(HERE, "foundation_share.json")))["sizes"]["70m"]["plates"]
CHK = json.load(open(os.path.join(HERE, "foundation_check.json")))["70m@143000"]
rng = np.random.default_rng(31337)
OUT = {"plates": {}}

# ---------------------------------------------------------------- 1. s per plate and its shuffle control
log("1. s = (W . W0)/(W0 . W0), exact, from fp16 bits at 2^-24; shuffle control 3 draws (W0 permuted over the plate)")
worst = 0.0; S_EXACT = {}
for name, key in NAMES.items():
    W, _ = whole24(f1.get_tensor(key)); W0, _ = whole24(f0.get_tensor(key))
    num, den, ww = xdot(W, W0), xdot(W0, W0), xdot(W, W)
    s = Fraction(num, den); S_EXACT[name] = s
    rho2 = Fraction(num * num, den * ww)
    sh = []
    for k in range(3):
        P = rng.permutation(W0.size)
        sh.append(float(Fraction(xdot(W, W0.ravel()[P]), den)))
    spread = (ww ** 0.5) / (den ** 0.5) / (W.size ** 0.5)
    ref = SHARE[name]["s"]; rel = float(s) / ref - 1; worst = max(worst, abs(rel))
    ratio = float(s) / spread
    OUT["plates"][name] = dict(s=float(s), rho2=float(rho2), shuffle=sh, spread=spread, delivered=ref, rel=rel,
                               s_over_spread=ratio, shuffle_max_over_spread=max(abs(v) for v in sh) / spread)
    log(f"  {name:8s} s {float(s):+.6f}  delivered {ref:+.6f}  rel {rel:+.2e}  rho^2 {float(rho2):.5f}  shuffle "
        + ", ".join(f"{v:+.5f}" for v in sh) + f"  (expected spread {spread:.5f}; s/spread {ratio:7.1f}; "
        f"max|shuffle|/spread {max(abs(v) for v in sh) / spread:.2f})")
    del W, W0
log(f"  worst relative difference against foundation_share.json: {worst:.2e}   ({time.time() - t0:.0f}s, peak {mem():.2f} GB)")
OUT["worst_rel"] = worst

# ---------------------------------------------------------------- 2. never-built rows
def never_built(W, U, W0, U0, pair, half=True):
    """rows passing |w - c w0| <= f*(ulp(w) + c ulp(w0)), f = 1/2 or 1; float64 screen at 1.5x, then exact whole numbers"""
    R = W.shape[0]; out = []; cs = {}
    f = 1 if half else 2                                                  # compare 2|w den - num w0| <= f (ulp den + num ulp0)
    for a in range(0, R, 2048):
        b = min(R, a + 2048); w, u = W[a:b], U[a:b]; w0, u0 = W0[pair[a:b]], U0[pair[a:b]]
        num, den = rowdot(w, w0), rowdot(w0, w0)
        wf, w0f = w.astype(np.float64), w0.astype(np.float64)
        c = num / np.maximum(den, 1)
        lhs = 2 * np.abs(wf - c[:, None] * w0f); rhs = f * (u + np.abs(c)[:, None] * u0)
        cand = np.flatnonzero((lhs <= 1.5 * rhs).all(1))
        for i in cand:
            n_, d_ = int(num[i]), int(den[i])
            if d_ == 0: continue
            wi = [int(v) for v in w[i]]; w0i = [int(v) for v in w0[i]]; ui = [int(v) for v in u[i]]; u0i = [int(v) for v in u0[i]]
            if all(2 * abs(x * d_ - n_ * y) <= f * (uu * d_ + abs(n_) * uu0) for x, y, uu, uu0 in zip(wi, w0i, ui, u0i)):
                out.append(a + int(i)); cs[a + int(i)] = Fraction(n_, d_)
    return out, cs

log("2. rows never built (every lane within fp16's half grain of c x start, c the row's own exact share)")
for name in ("embed", "head"):
    W, U = whole24(f1.get_tensor(NAMES[name])); W0, U0 = whole24(f0.get_tensor(NAMES[name]))
    ident = np.arange(W.shape[0]); perm = rng.permutation(W.shape[0])
    ids, cs = never_built(W, U, W0, U0, ident, half=True)
    ids1, _ = never_built(W, U, W0, U0, ident, half=False)
    shf, _ = never_built(W, U, W0, U0, perm, half=False)
    pad = [i for i in range(50277, 50304)]
    r = dict(half_grain=len(ids), full_grain=len(ids1), shuffled_full_grain=len(shf),
             pad_in=sum(i in set(ids) for i in pad), id1=1 in ids, id126=126 in ids)
    if ids:
        c = np.array([float(cs[i]) for i in ids]); med = float(np.median(c))
        r.update(c_median=med, c_spread=float((c.max() - c.min()) / med), c_min=float(c.min()), c_max=float(c.max()),
                 same_ids_as_foundation_check=(sorted(ids) == sorted(CHK[name]["ids"])) if name in CHK else None)
        # rho^2 view of the same rows
    num_r, den_r, ww_r = rowdot(W, W0).astype(object), rowdot(W0, W0).astype(object), rowdot(W, W).astype(object)
    rho2 = np.array([float(Fraction(int(n) * int(n), int(d) * int(q))) if d and q else 0.0 for n, d, q in zip(num_r, den_r, ww_r)])
    r["rho2_ge_0.9"] = int((rho2 >= 0.9).sum()); r["rho2_ge_0.5"] = int((rho2 >= 0.5).sum())
    OUT[f"never_built_{name}"] = r; OUT[f"never_built_{name}_ids"] = ids
    log(f"  {name:6s} {r}")
    if name == "embed":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(MD, "tokenizer", "tokenizer.json"))
        shown = [repr(tok.id_to_token(int(i))) for i in ids if i < 50277][:30]
        log(f"         first words: {' '.join(shown)}")
        C_EMB = Fraction(int(round(r['c_median'] * 2 ** 40)), 2 ** 40)
    del W, W0, U, U0
log(f"   ({time.time() - t0:.0f}s, peak {mem():.2f} GB)")

# ---------------------------------------------------------------- 3. the shared grid rebuilt, and lit places
def grid(Wf, Ws):
    """per row (neuron, HF orientation): smallest k with max(|Wf|,|Ws|) <= 32767 * 2^k (k at 2^-24); q = round half even"""
    M = np.maximum(np.abs(Wf).max(1), np.abs(Ws).max(1))
    k = np.zeros(len(M), np.int64)
    for i, m in enumerate(M.tolist()):
        kk = 0
        if m > 32767:
            while m > (32767 << kk): kk += 1
        else:
            while m > 0 and (m << (1 - kk)) <= 32767: kk -= 1
        k[i] = kk
    def q(V):
        out = np.empty_like(V)
        for kk in np.unique(k):
            rows = k == kk; X = V[rows]
            if kk <= 0: out[rows] = X << -kk; continue
            fl = X >> kk; r = X - (fl << kk); h = np.int64(1) << (kk - 1)
            up = (r > h) | ((r == h) & ((fl & 1) == 1)); out[rows] = fl + up
        return out
    return q(Wf), q(Ws), k - 24

def lit(X):
    """nonzero digits of the signed-digit writing with no two neighbouring nonzero digits, stepped out one digit at a time"""
    x = np.abs(X).astype(np.int64).copy(); n = np.zeros(x.shape, np.int64)
    while x.any():
        odd = (x & 1) == 1
        z = np.where(odd, 2 - (x & 3), 0)                                   # +1 where x = 1 mod 4, -1 where x = 3 mod 4
        n += odd; x = (x - z) >> 1
    return n

def rnd(s40, Q):
    """round(s x Q), s = s40 / 2^40, half away from zero, whole numbers (|Q| <= 2^15, s40 < 2^47: inside int64)"""
    assert 0 <= s40 < 2 ** 47 and np.abs(Q).max() <= 32768
    P = Q.astype(np.int64) * s40
    return np.sign(P) * ((np.abs(P) + (1 << 39)) >> 40)

Z = np.load(os.path.join(HERE, "base2_pythia-70m.npz"))
log("3. the shared 16-place grid rebuilt from the whole numbers, against base2_pythia-70m.npz; lit places of W and D'")
for name in ("qkv.0", "fc.0", "head", "embed"):
    W, _ = whole24(f1.get_tensor(NAMES[name])); W0, _ = whole24(f0.get_tensor(NAMES[name]))
    qf, qs, e = grid(W, W0)
    zw, zs, ze = Z[name + ".w"].astype(np.int64), Z[name + ".s"].astype(np.int64), Z[name + ".e"].astype(np.int64)
    if name != "embed": zw, zs = zw.T, zs.T
    diff = dict(w=int((qf != zw).sum()), s=int((qs != zs).sum()), e=int((e != ze).sum()))
    if name != "embed":
        s40 = int(round(float(S_EXACT[name]) * 2 ** 40))
        D = qf - rnd(s40, qs)
        lw, ld, lw0 = lit(qf).mean(), lit(D).mean(), lit(qs).mean()
        dl = SHARE[name]["lit"]
        log(f"  {name:6s} grid digits differing from npz: {diff};  lit a weight W {lw:.4f} (delivered {dl['W']:.4f})  "
            f"W0 {lw0:.4f} ({dl['W0']:.4f})  D'm {ld:.4f} ({dl['Dm']:.4f})  D'm/W {ld / lw - 1:+.2%}")
        OUT["plates"][name].update(grid_diff=diff, lit_W=float(lw), lit_W0=float(lw0), lit_Dm=float(ld))
    else:
        ids = np.array(OUT["never_built_embed_ids"]); s40 = int(C_EMB * 2 ** 40)
        D = qf - rnd(s40, qs)
        LW, LD = lit(qf), lit(D)
        rest = np.setdiff1d(np.arange(qf.shape[0]), ids); ctl = np.random.default_rng(5).choice(rest, len(ids), replace=False)
        log(f"  embed  grid digits differing from npz: {diff};  c = {float(C_EMB):.6e}")
        log(f"         never-built rows: |W| max {np.abs(qf[ids]).max()}  |D'| max {np.abs(D[ids]).max()}  lit a weight W {LW[ids].mean():.3f} D' {LD[ids].mean():.3f}")
        log(f"         241 other rows:   |W| max {np.abs(qf[ctl]).max()}  |D'| max {np.abs(D[ctl]).max()}  lit a weight W {LW[ctl].mean():.3f} D' {LD[ctl].mean():.3f}")
        log(f"         whole table: lit places W {int(LW.sum()):,}  D' {int(LD.sum()):,}  ({LD.sum() / LW.sum() - 1:+.3%})")
        OUT["embed_lit"] = dict(grid_diff=diff, nb_W=float(LW[ids].mean()), nb_D=float(LD[ids].mean()), nb_Dmax=int(np.abs(D[ids]).max()),
                                ctl_W=float(LW[ctl].mean()), ctl_D=float(LD[ctl].mean()), table_W=int(LW.sum()), table_D=int(LD.sum()))
    del W, W0, qf, qs
log(f"done {time.time() - t0:.0f}s  peak memory {mem():.2f} GB")
json.dump(OUT, open(os.path.join(HERE, "indcheck_weights70.json"), "w"), default=str)
