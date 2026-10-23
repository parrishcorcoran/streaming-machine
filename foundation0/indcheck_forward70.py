"""SEPARATE CHECKER'S OWN WHOLE-NUMBER MACHINE for pythia-70m, 2026-10-06. Shares no code and no tables with
pythia_base2.py, remove_foundation.py or the npz files: it reads Pythia's fp16 files and turns every value into a whole
number at 2^-24 by its bits alone (exact: every fp16 value is a multiple of 2^-24). Its own tables are built here with
Python's decimal arithmetic (base-10 at 50 digits, no binary float) and written as whole numbers at 2^-30.

THE MACHINE (a different design from the delivered one on purpose: fixed scale instead of 16 places a row):
  - the stream R: whole numbers at 2^-16. Word row in: round(row / 2^8).
  - plate: R' = round((W x + b<<16) / 2^24), W, b whole numbers at 2^-24; int64 sums with the bound asserted
    (max|x| x the largest row sum of |W| < 2^62); past it, W split into two halves and recombined exactly.
  - gauge: c = d x - S1, V = d S2 - S1^2 + eps' d^2 (eps' = round(1e-5 x 2^32)), y = round(c 2^32 / isqrt(V 2^32))
    (Python whole numbers, exact integer root), then round((y g + b<<16) / 2^24).
  - turns: first 16 lanes of each head, pairs (i, i+8), cos and sin at 2^-30 from the decimal table.
  - score: q . k at 2^-32, 1/8 = 3 places; water level: the row's largest score; arrival e^-t, t at 2^-16, as the
    product of two tables (e^-(k/256) and e^-(j/65536)) at 2^-30; the pile divided by the row total, rounded.
  - bend: x Phi(x), Phi from a 2^-12 table on [-8, 8] at 2^-30, read between its points by a straight line in whole
    numbers; outside [-8, 8] Phi is 0 or 1.
  - head piles at 2^-16; the race: the largest pile, the lower word on a tie; top 5 by the same order.
CASES (gains, norm offsets and biases are the finished ones in every case, as delivered):
  fin  finished plates;  a  W - W0 on every plate and both word tables;  b  W - round(s W0), s per plate (this checker's
  exact s, indcheck_weights70.json, carried at 2^-30);  ca, cb  a and b with W0's values permuted over each whole plate
  (this checker's own draw, so only the shares can be compared with the delivered controls, not word by word).
Positions: each text's own words, then fp32's 16 greedy words fed in (as remove_foundation.py counts them), first 10
texts. Measured: top-1 (the case's word = this machine's finished word) and top-5 (the finished word inside the case's
top 5) at all positions; against fp32 (indcheck_fp32_70m.json) and against the delivered per-position words
(remove_70m.json) and the delivered 10-text run (remove_70m-indcheck10.json, made by the delivered script, if present).

PREDICTION (written before this script first ran):
  - fin: equal to fp32's word at 99% or more of the 10 texts' positions (all 160 continuation positions), and to the
    delivered finished machine's word at the same share.
  - a: top-1 under 2%, top-5 under 5% (delivered over 30 texts: 0.3% / 1.0%); this machine's a-words = fp32's a-words
    at 95% or more of positions.
  - b: top-1 within 5 points of the delivered 10-text share (30-text share 52.6%), top-5 within 5 points (82.0%);
    this machine's b-words = delivered b-words at 95% or more of positions.
  - ca: top-1 and top-5 at 0-2% (delivered 0.0 / 0.0); cb: top-1 within 10 points of b (delivered draws 51.4-60.9%).
"""
import os, sys, json, time, math, resource
import numpy as np
from decimal import Decimal, getcontext
from safetensors import safe_open

A = dict(a.split("=") for a in sys.argv[1:])
TEXTS = int(A.get("texts", 10)); CASES = A.get("cases", "fin,a,b,ca,cb").split(",")
SEED = int(A.get("seed", 424242)); TAG = ("-" + A["tag"]) if A.get("tag") else ""   # added after the first run: more scramble draws
HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(HERE, "models")
LOGF = open(os.path.join(HERE, f"indcheck_forward70{TAG}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
t0 = time.time()
d, L, NH, HS, ROT = 512, 6, 8, 64, 16

# ---------------------------------------------------------------- tables, decimal arithmetic -> whole numbers at 2^-30
TPATH = os.path.join(HERE, "indcheck_tables70.npz")
if os.path.exists(TPATH):
    T = np.load(TPATH); PHI, EH, EL, COS, SIN = (T[k].astype(np.int64) for k in ("phi", "eh", "el", "cos", "sin"))
else:
    getcontext().prec = 50
    PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
    ONE30 = Decimal(2) ** 30
    def r30(x): return int((x * ONE30).to_integral_value())                    # round half even
    SQ2, SQPI = Decimal(2).sqrt(), PI.sqrt()
    def erf(z):
        if z == 0: return Decimal(0)
        z2 = 2 * z * z; term = z; s = z; n = 0
        while True:
            n += 1; term = term * z2 / (2 * n + 1); s += term
            if term < s * Decimal("1e-46"): break
        return 2 / SQPI * (-(z * z)).exp() * s
    half = [r30((1 + erf(Decimal(i) / 4096 / SQ2)) / 2) for i in range(32769)]     # x = 0 .. 8 at 2^-12
    PHI = np.array([(1 << 30) - half[32768 - i] for i in range(32768)] + half, dtype=np.int64)  # x = -8 .. 8, 65537 points
    EH = np.array([r30((-Decimal(k) / 256).exp()) for k in range(24 * 256 + 1)], dtype=np.int64)
    EL = np.array([r30((-Decimal(j) / 65536).exp()) for j in range(256)], dtype=np.int64)
    def cs(x):
        x = x % (2 * PI); c = s = Decimal(0); tc = Decimal(1); ts = x; n = 0
        while abs(tc) > Decimal("1e-48") or abs(ts) > Decimal("1e-48"):
            c += tc; s += ts; tc = -tc * x * x / ((2 * n + 1) * (2 * n + 2)); ts = -ts * x * x / ((2 * n + 2) * (2 * n + 3)); n += 1
        return c, s
    inv = [Decimal(10000) ** (Decimal(-i) / Decimal(8)) for i in range(8)]
    COS = np.zeros((64, 8), np.int64); SIN = np.zeros((64, 8), np.int64)
    for p in range(64):
        for i in range(8):
            c, s = cs(p * inv[i]); COS[p, i], SIN[p, i] = r30(c), r30(s)
    np.savez(TPATH, phi=PHI, eh=EH, el=EL, cos=COS, sin=SIN)
log(f"tables ready ({time.time() - t0:.0f}s): PHI {len(PHI)} points, Phi(0) = {PHI[32768]} / 2^30, e^-t {len(EH)} x {len(EL)}")

# ---------------------------------------------------------------- whole numbers from fp16 bits
f1 = safe_open(os.path.join(MD, "pythia-70m-step143000", "model.safetensors"), "np")
f0 = safe_open(os.path.join(MD, "pythia-70m-step0", "model.safetensors"), "np")
def whole24(a16):
    """fp16 -> whole numbers at 2^-24, from the bits only, in row blocks (keeps the temporaries small)"""
    out = np.empty(a16.shape, np.int64); R = a16.shape[0] if a16.ndim > 1 else 1
    flat = a16.reshape(R, -1); o2 = out.reshape(R, -1); B = max(1, (1 << 20) // flat.shape[1])
    for a in range(0, R, B):
        b = flat[a:a + B].view(np.uint16).astype(np.int64); sg, ex, mn = b >> 15, (b >> 10) & 31, b & 1023
        assert (ex != 31).all()
        mag = np.where(ex == 0, mn, (mn | 1024) << np.maximum(ex - 1, 0))
        o2[a:a + B] = np.where(sg == 1, -mag, mag)
    return out
def get(f, key): return whole24(f.get_tensor(key))
NAMES = {"embed": "gpt_neox.embed_in.weight", "head": "embed_out.weight"}
for l in range(L):
    p = f"gpt_neox.layers.{l}."
    NAMES[f"qkv.{l}"] = p + "attention.query_key_value.weight"; NAMES[f"proj.{l}"] = p + "attention.dense.weight"
    NAMES[f"fc.{l}"] = p + "mlp.dense_h_to_4h.weight"; NAMES[f"mproj.{l}"] = p + "mlp.dense_4h_to_h.weight"
BIAS = {}
for l in range(L):
    p = f"gpt_neox.layers.{l}."
    for nm, k in (("qkv", "attention.query_key_value.bias"), ("proj", "attention.dense.bias"),
                  ("fc", "mlp.dense_h_to_4h.bias"), ("mproj", "mlp.dense_4h_to_h.bias")):
        BIAS[f"{nm}.{l}"] = get(f1, p + k)
    for nm, k in (("ln1", "input_layernorm"), ("ln2", "post_attention_layernorm")):
        BIAS[f"{nm}.{l}"] = (get(f1, p + k + ".weight"), get(f1, p + k + ".bias"))
BIAS["lnf"] = (get(f1, "gpt_neox.final_layer_norm.weight"), get(f1, "gpt_neox.final_layer_norm.bias"))
BIAS["head"] = None
SFR = {k: v["s"] for k, v in json.load(open(os.path.join(HERE, "indcheck_weights70.json")))["plates"].items()}

PL = {}
def build(case):
    PL.clear(); kind = case
    for i, (name, key) in enumerate(NAMES.items()):
        W = get(f1, key)
        if kind != "fin":
            W0 = get(f0, key)
            if kind in ("ca", "cb"):
                P = np.random.default_rng(SEED + i).permutation(W0.size); W0 = W0.ravel()[P].reshape(W0.shape); del P
            if kind in ("a", "ca"): W -= W0
            else:
                s30 = int(round(SFR[name] * 2 ** 30)); assert int(np.abs(W0).max()) * s30 < 2 ** 62
                B = max(1, (1 << 20) // W.shape[1])
                for a in range(0, W.shape[0], B):
                    X = W0[a:a + B] * s30; W[a:a + B] -= np.sign(X) * ((np.abs(X) + (1 << 29)) >> 30)
            del W0
        rs = int(np.abs(W).sum(1).max()) if name != "embed" else None
        PL[name] = (np.ascontiguousarray(W), rs)

SPLITS = [0]
def rnd_shift(X, k): return (X + (np.int64(1) << (k - 1))) >> k                 # round half up, k >= 1
def plate(x, name):
    """x: [P][in] whole numbers at 2^-16 -> [P][out] at 2^-16"""
    W, rs = PL[name]; b = BIAS[name]; mx = int(np.abs(x).max(initial=0))
    bb = 0 if b is None else (b.astype(np.int64) << 16)[None, :]
    if mx * rs < 2 ** 62:
        return rnd_shift((W @ x.T).T + bb, 24)
    SPLITS[0] += 1                                                                # exact recombination past the bound
    Wh = W >> 16; Wl = W - (Wh << 16)
    assert mx * int(np.abs(Wh).sum(1).max()) < 2 ** 62 and mx * int(Wl.sum(1).max()) < 2 ** 62
    Ah, Al = (Wh @ x.T).T, (Wl @ x.T).T
    q, r = Ah >> 8, Ah & 255                                                      # total = q 2^24 + (r 2^16 + Al + bb)
    return q + rnd_shift((r << 16) + Al + bb, 24)

EPS = int(round(1e-5 * 2 ** 32))
def gauge(R, name):
    g, b = BIAS[name]; P = R.shape[0]; out = np.zeros_like(R)
    for i in range(P):
        x = [int(v) for v in R[i]]; S1 = sum(x); S2 = sum(v * v for v in x)
        V = d * S2 - S1 * S1 + EPS * d * d; sd = math.isqrt(V << 32)
        y = [((d * v - S1) * (1 << 32) * 2 + sd) // (2 * sd) for v in x]         # round((d x - S1) 2^32 / sd)
        out[i] = [((yy * int(gg) + (int(bv) << 16)) + (1 << 23)) >> 24 for yy, gg, bv in zip(y, g, b)]
    return out

def turn(X, pos):
    c, s = COS[pos], SIN[pos]; x1, x2 = X[:, :8], X[:, 8:16]
    assert int(np.abs(X[:, :16]).max(initial=0)) < 2 ** 31
    return np.concatenate([rnd_shift(x1 * c - x2 * s, 30), rnd_shift(x2 * c + x1 * s, 30), X[:, 16:]], 1)

def expneg(t):
    """e^-t at 2^-30 for t >= 0 at 2^-16"""
    hi, lo = t >> 8, t & 255
    ok = hi < len(EH)
    return np.where(ok, rnd_shift(EH[np.minimum(hi, len(EH) - 1)] * EL[lo], 30), 0)

def attention(QKV):
    P = QKV.shape[0]; pos = np.arange(P); out = np.zeros((P, d), np.int64); live = np.tril(np.ones((P, P), bool))
    for h in range(NH):
        o = h * 3 * HS; q, k, v = QKV[:, o:o + HS], QKV[:, o + HS:o + 2 * HS], QKV[:, o + 2 * HS:o + 3 * HS]
        q, k = turn(q, pos), turn(k, pos)
        assert int(np.abs(q).max()) * int(np.abs(k).max()) * HS < 2 ** 62
        sc = q @ k.T                                                              # 2^-32; the 1/8 makes it 2^-35
        lev = np.where(live, sc, np.iinfo(np.int64).min).max(1)
        t = np.where(live, (lev[:, None] - sc + (1 << 18)) >> 19, 0)             # depth below the level at 2^-16
        a = np.where(live, expneg(t), 0)
        den = a.sum(1)
        assert int(np.abs(v).max(initial=0)) * int(den.max()) < 2 ** 62
        num = a @ v
        out[:, h * HS:(h + 1) * HS] = (2 * num + den[:, None]) // (2 * den[:, None])
    return out

def bend(H):
    u = H + (8 << 16); inside = (u >= 0) & (u < (16 << 16))
    uc = np.clip(u, 0, (16 << 16) - 1); idx, fr = uc >> 4, uc & 15
    ph = PHI[idx] + (((PHI[idx + 1] - PHI[idx]) * fr + 8) >> 4)
    ph = np.where(inside, ph, np.where(u < 0, 0, 1 << 30))
    assert int(np.abs(H).max(initial=0)) < 2 ** 32
    return rnd_shift(H * ph, 30)

def forward(ids):
    E = PL["embed"][0]; R = rnd_shift(E[np.asarray(ids)].astype(np.int64), 8)
    for l in range(L):
        At = plate(attention(plate(gauge(R, f"ln1.{l}"), f"qkv.{l}")), f"proj.{l}")
        Mo = plate(bend(plate(gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
        R = R + At + Mo
    return plate(gauge(R, "lnf"), "head")

def top5(row): return np.lexsort((np.arange(len(row)), -row))[:5]

refs = json.load(open(os.path.join(HERE, "refs_pythia-70m.json")))[:TEXTS]
FP = json.load(open(os.path.join(HERE, "indcheck_fp32_70m.json")))
DEL = json.load(open(os.path.join(HERE, "remove_70m.json")))["cases"]
p10 = os.path.join(HERE, "remove_70m-indcheck10.json")
DEL10 = json.load(open(p10))["cases"] if os.path.exists(p10) else None
RES = {}; FIN = None
for case in CASES:
    tc = time.time(); build(case); SPLITS[0] = 0; t1, t5 = [], []
    for r in refs:
        lg = forward(r["ids"] + r["ref"][:15])
        o = [top5(row) for row in lg]; t1.append([int(x[0]) for x in o]); t5.append([[int(v) for v in x] for x in o])
    if case == "fin": FIN = (t1, t5)
    n = sum(len(x) for x in t1)
    keep1 = sum(a == b for x, y in zip(t1, FIN[0]) for a, b in zip(x, y))
    keep5 = sum(f in s for x5, y in zip(t5, FIN[0]) for s, f in zip(x5, y))
    line = f"  {case:4s} top-1 {keep1 / n * 100:5.1f}%  top-5 {keep5 / n * 100:5.1f}%  (of {n} positions)"
    if case in FP:
        sf = sum(a == b for x, y in zip(t1, FP[case]["top1"]) for a, b in zip(x, y)); line += f";  word = fp32 {case}'s at {sf}/{n}"
    if case in DEL:
        dd = DEL[case]["top1"][:TEXTS]; df = DEL["fin"]["top1"][:TEXTS]
        sd = sum(a == b for x, y in zip(t1, dd) for a, b in zip(x, y))
        dk = sum(a == b for x, y in zip(dd, df) for a, b in zip(x, y))
        line += f";  word = delivered {case}'s at {sd}/{n};  delivered top-1 on these 10 texts {dk / n * 100:5.1f}%"
    if DEL10 and case in DEL10:
        line += f" (delivered 10-text run: top-1 {DEL10[case]['top1_all'] * 100:5.1f}% top-5 {DEL10[case]['top5_all'] * 100:5.1f}%)"
    line += f";  split products {SPLITS[0]};  {time.time() - tc:.0f}s  peak {mem():.2f} GB"
    log(line)
    if case == "fin":
        cont = sum(int(a == b) for x, rr in zip(t1, refs) for a, b in zip(x[len(rr["ids"]) - 1:len(rr["ids"]) + 15], rr["ref"]))
        log(f"       finished words = fp32 greedy words (refs) at {cont} of {16 * TEXTS} continuation positions")
    RES[case] = dict(top1=t1, top5=t5, top1_share=keep1 / n, top5_share=keep5 / n)
    # pairing-broken control for the statistic: the case's words against the NEXT text's finished words, same position
    if case != "fin":
        k1 = k5 = m = 0
        for t in range(TEXTS):
            u, v = t1[t], FIN[0][(t + 1) % TEXTS]; u5 = t5[t]
            for p in range(min(len(u), len(v))):
                m += 1; k1 += u[p] == v[p]; k5 += v[p] in u5[p]
        log(f"       pairing-broken (next text's finished words): top-1 {k1 / m * 100:4.1f}%  top-5 {k5 / m * 100:4.1f}%")
        RES[case].update(ctl_top1=k1 / m, ctl_top5=k5 / m)
json.dump(RES, open(os.path.join(HERE, f"indcheck_forward70{TAG}.json"), "w"))
log(f"done {time.time() - t0:.0f}s  peak memory {mem():.2f} GB")
