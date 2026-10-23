"""Independent checker's own code (written by the checking session, not the building session). Shares NOTHING with
model.py, snapio.py, bmachine.py, machine.py, work.py, inplace.py, remove.py, dodge_exact.py or workcheck.py: it reads
only the files the builds wrote (blank_*.npz, snaps/*.npz, build_*.json, data.npz, machines/*.npz) and the JSON
results it is checking.

  - its own list of the parts and its own redraw of the blank from the seed
  - its own step sizes and its own replay of the shave (recipe C)
  - its own fingerprint of a set of parts (the same recipe build.py logs: sha256 of each part's first 16 hex digits)
  - its own float64 model (layer norm, causal arrival, tanh bend written out by hand)
  - its own whole-number machine, written from the arithmetic described in machine.py's docstring and SPEC: 16 places
    per position, whole-number products (int64, no float), rounding shifts, the water level through the halved log,
    the arrival through e^-d, the bend through the share table, the word by the largest pile (first on a tie)
  - its own tables, regenerated from their definitions and compared with machines/tables.npz
"""
import os, math, json, hashlib, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOC, CTX, HS16 = 8193, 128, 16
ES = 16                      # the stream's scale: a value v is held as v x 2^16
SCORE = 10                   # the arrival's distance scale 2^-10


# ---------------------------------------------------------------------------------------------- parts and the blank
def part_list(d, L):
    big, small = 0.02, 0.02 / math.sqrt(2 * L)
    out = [("wte", (VOC, d), big), ("wpe", (CTX, d), big)]
    for l in range(L):
        out += [(f"ln1.{l}.g", (d,), "one"), (f"ln1.{l}.b", (d,), "zero"),
                (f"qkv.{l}.w", (d, 3 * d), big), (f"qkv.{l}.b", (3 * d,), "zero"),
                (f"proj.{l}.w", (d, d), small), (f"proj.{l}.b", (d,), "zero"),
                (f"ln2.{l}.g", (d,), "one"), (f"ln2.{l}.b", (d,), "zero"),
                (f"fc.{l}.w", (d, 4 * d), big), (f"fc.{l}.b", (4 * d,), "zero"),
                (f"mproj.{l}.w", (4 * d, d), small), (f"mproj.{l}.b", (d,), "zero")]
    out += [("lnf.g", (d,), "one"), ("lnf.b", (d,), "zero"), ("head", (d, VOC), big)]
    return out


def drawn_names(d, L): return [n for n, _, s in part_list(d, L) if not isinstance(s, str)]


def redraw_blank(d, L, seed=20261006):
    g = np.random.default_rng(seed); out = {}
    for n, shp, s in part_list(d, L):
        if s == "one": out[n] = np.full(shp, 1 << 30, np.int64)
        elif s == "zero": out[n] = np.zeros(shp, np.int64)
        else: out[n] = np.rint(g.standard_normal(shp) * s * (1 << 30)).astype(np.int64)
    return out


def step_lr(t, peak, total=3000, warm=30):
    if t <= warm: return peak * t / warm
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - warm) / (total - warm))))


def shaved_blank(B0, drawn, peak, wd, upto):
    """the blank after `upto` shaves, every shave a whole number rint(B x step x wd), drawn parts only"""
    B = {n: a.copy() for n, a in B0.items()}
    if not wd: return B
    for t in range(1, upto + 1):
        f = step_lr(t, peak) * wd
        for n in drawn: B[n] = B[n] - np.rint(B[n] * f).astype(np.int64)
    return B


def fingerprint(D, names):
    return hashlib.sha256(b"".join(hashlib.sha256(np.ascontiguousarray(D[n]).tobytes()).hexdigest()[:16].encode()
                                   for n in names)).hexdigest()[:16]


def exact_dot(a, b):
    """exact whole-number dot product of two int64 arrays (split into 2^20 halves so no int64 sum can overflow)"""
    a = a.ravel().astype(np.int64); b = b.ravel().astype(np.int64)
    assert np.abs(a).max(initial=0) < 2 ** 40 and np.abs(b).max(initial=0) < 2 ** 40 and a.size < 2 ** 21
    ah, al = a >> 20, a & ((1 << 20) - 1); bh, bl = b >> 20, b & ((1 << 20) - 1)
    def s(x, y): return sum(int(v) for v in np.add.reduceat(x * y, np.arange(0, x.size, 4096)))   # each block < 2^52
    return (s(ah, bh) << 40) + ((s(ah, bl) + s(al, bh)) << 20) + s(al, bl)


def scr_within_neuron(name, a, draw):
    """the checker's own scramble: values permuted within each neuron (plates/head: a column; wte/wpe: a row);
    seeds unrelated to snapio.scramble's"""
    rng = np.random.default_rng(99991 + 7919 * draw + 31 * len(name) + sum(ord(c) * (i + 1) for i, c in enumerate(name)))
    ax = 1 if name in ("wte", "wpe") else 0
    return np.take_along_axis(a, np.argsort(rng.random(a.shape), axis=ax), axis=ax)


def held_part():
    z = np.load(os.path.join(HERE, "data.npz")); return z["held"].astype(np.int64)


def loss_windows(held, n=128):
    st = np.linspace(0, len(held) - CTX - 2, n).astype(np.int64)
    return np.stack([held[s:s + CTX + 1] for s in st])


# ---------------------------------------------------------------------------------------------- float64 model
def float_forward(Wf, ids, d, L):
    import torch
    nh = d // HS16; B, T = ids.shape
    def ln(x, g, b):
        m = x.mean(-1, keepdim=True); v = ((x - m) ** 2).mean(-1, keepdim=True)
        return (x - m) / torch.sqrt(v + 1e-5) * g + b
    def bend(x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    x = Wf["wte"][ids] + Wf["wpe"][:T]
    for l in range(L):
        h = ln(x, Wf[f"ln1.{l}.g"], Wf[f"ln1.{l}.b"])
        y = h @ Wf[f"qkv.{l}.w"] + Wf[f"qkv.{l}.b"]
        q, k, v = y[..., :d], y[..., d:2 * d], y[..., 2 * d:]
        q, k, v = (z.reshape(B, T, nh, HS16).permute(0, 2, 1, 3) for z in (q, k, v))
        s = (q @ k.transpose(-1, -2)) / 4.0
        s = s.masked_fill(mask, -math.inf)
        a = torch.exp(s - s.max(-1, keepdim=True).values); a = a / a.sum(-1, keepdim=True)
        o = (a @ v).permute(0, 2, 1, 3).reshape(B, T, d)
        x = x + o @ Wf[f"proj.{l}.w"] + Wf[f"proj.{l}.b"]
        h = ln(x, Wf[f"ln2.{l}.g"], Wf[f"ln2.{l}.b"])
        x = x + bend(h @ Wf[f"fc.{l}.w"] + Wf[f"fc.{l}.b"]) @ Wf[f"mproj.{l}.w"] + Wf[f"mproj.{l}.b"]
    return ln(x, Wf["lnf.g"], Wf["lnf.b"]) @ Wf["head"]


def float_loss(Wint, wins, d, L, chunk=16):
    import torch
    Wf = {n: torch.from_numpy(a.astype(np.float64) / 2.0 ** 30) for n, a in Wint.items()}
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(wins), chunk):
            w = torch.from_numpy(wins[i:i + chunk]); lg = float_forward(Wf, w[:, :-1], d, L)
            lse = torch.logsumexp(lg, -1); tgt = lg.gather(-1, w[:, 1:, None])[..., 0]
            tot += float((lse - tgt).sum()); cnt += tgt.numel()
    return tot / cnt


# ---------------------------------------------------------------------------------------------- tables
def make_tables():
    c = math.sqrt(2 / math.pi)
    share = np.array([round(0.5 * (1 + math.tanh(c * (x + 0.044715 * x ** 3))) * 65536)
                      for x in ((i - 32768) / 4096 for i in range(65536))], np.int64)
    arr = []
    t = 0
    while True:
        a = round(math.exp(-t / 1024) * 2 ** 24)
        if a == 0: break
        arr.append(a); t += 1
    log2 = np.array([round(math.log2(1 + m / 65536) * 65536) for m in range(65536)], np.int64)
    exp2 = np.array([round(2 ** (f / 65536) * 65536) for f in range(65536)], np.int64)
    return dict(share=share, arrival=np.array(arr, np.int64), log2=log2, exp2=exp2)


TAB = None
def tables():
    global TAB
    if TAB is None:
        mine = make_tables(); disk = np.load(os.path.join(HERE, "machines", "tables.npz"))
        for k in mine: assert np.array_equal(mine[k], disk[k]), f"table {k} differs from its definition"
        TAB = mine
    return TAB


# ---------------------------------------------------------------------------------------------- whole-number machine
def rshift_round(X, s):
    """X x 2^-s rounded to nearest (half up), s >= 0 or < 0 (then an exact left shift); whole numbers only"""
    X = np.asarray(X); s = np.broadcast_to(np.asarray(s, np.int64), X.shape)
    if X.dtype == object or np.abs(X).max(initial=0) >= 2 ** 61:
        f = np.frompyfunc(lambda x, k: (int(x) << -int(k)) if k <= 0 else ((int(x) + (1 << (int(k) - 1))) >> int(k)), 2, 1)
        r = f(X.astype(object), s)
        return np.array([int(v) for v in r.ravel()], dtype=np.int64).reshape(X.shape)
    X = X.astype(np.int64); out = np.empty(X.shape, np.int64)
    left = s <= 0
    if left.any():
        k = -s[left]; v = X[left]
        assert (np.abs(v) < (np.int64(1) << (62 - k))).all()
        out[left] = v << k
    if (~left).any():
        k = s[~left]; v = X[~left]
        out[~left] = (v + (np.int64(1) << (k - 1))) >> k
    return out


def on16(X):
    """each position (row) on 16 places with its own power: X ~ x16 x 2^sh, returns (x16, sh - 16)"""
    top = np.array([int(m).bit_length() for m in np.abs(X).max(1)], np.int64)
    sh = np.maximum(top - 15, 0)
    return rshift_round(X, sh[:, None]), sh - ES


def log2_fine(v):
    v = int(v); t = v.bit_length() - 1
    m = ((v >> (t - 16)) if t >= 16 else (v << (16 - t))) & 0xFFFF
    return t * 65536 + int(tables()["log2"][m])


def times_pow2(X, F):
    """X x 2^(F/65536): the fine part read from the 2^f table, the whole part a rounding shift"""
    F = np.broadcast_to(np.asarray(F, np.int64), np.shape(X))
    whole, fine = F >> 16, F & 0xFFFF
    e2 = tables()["exp2"][fine]
    if np.abs(X).max(initial=0) < 2 ** 45: Y = X.astype(np.int64) * e2
    else: Y = X.astype(object) * e2.astype(object)
    return rshift_round(Y, 16 - whole)


def imatmul(a, b):
    """whole-number product; float64 BLAS only where every partial sum is provably below 2^53, else int64"""
    bound = float(np.abs(a).sum(1).max(initial=0)) * float(np.abs(b).max(initial=0))
    if bound < 2 ** 53:
        with np.errstate(all="ignore"):                   # Accelerate raises spurious flags; the result is checked
            r = a.astype(np.float64) @ b.astype(np.float64)
        assert np.isfinite(r).all() and (r == np.round(r)).all()
        return r.astype(np.int64)
    return a.astype(np.int64) @ b.astype(np.int64)


class WholeMachine:
    """variant: 'finished' (.w), 'pushes' (.w - .s on every drawn part), or a dict of per-part callables"""
    def __init__(self, path, variant="finished", capture=False):
        M = np.load(path); self.M = M
        self.d, self.L, self.nh, self.hs = (int(M[k]) for k in ("d", "L", "heads", "hs"))
        self.eps40, self.half = int(M["eps40"]), int(M["half_log2_hs"])
        assert self.half == 2 * 65536                         # log2(16)/2 = 2 exactly: the score's 1/4 is a shift of 2
        self.capture, self.cap = capture, {}
        def take(key):
            w, s = M[key + ".w"].astype(np.int64), M[key + ".s"].astype(np.int64)
            if variant == "finished": return w
            if variant == "pushes": return w - s
            return variant(key, w, s)
        self.P = {}
        for l in range(self.L):
            for nm in ("qkv", "proj", "fc", "mproj"):
                k = f"{nm}.{l}"; self.P[k] = (take(k), M[k + ".e"].astype(np.int64), M[k + ".b"].astype(np.int64))
        self.P["head"] = (take("head"), M["head.e"].astype(np.int64), None)
        self.wte, self.wte_e = take("wte"), M["wte.e"].astype(np.int64)
        self.wpe, self.wpe_e = take("wpe"), M["wpe.e"].astype(np.int64)
        self.N = {}
        for nm in [f"ln1.{l}" for l in range(self.L)] + [f"ln2.{l}" for l in range(self.L)] + ["lnf"]:
            self.N[nm] = (M[nm + ".g"].astype(np.int64), int(M[nm + ".ge"]), M[nm + ".b"].astype(np.int64))

    def level(self, R, nm):
        g, ge, b = self.N[nm]; d = self.d
        x, ex = on16(R)
        S1 = x.sum(1); S2 = (x * x).sum(1)
        c = d * x - S1[:, None]
        sh = 40 + 2 * ex
        epsu = np.array([(self.eps40 >> int(k)) if k >= 0 else (self.eps40 << int(-k)) for k in sh], np.int64)
        V = d * S2 - S1 * S1 + epsu
        half_log = np.array([log2_fine(v) >> 1 for v in V], np.int64)
        return times_pow2(c * g[None, :], (ge + ES) * 65536 - half_log[:, None]) + b

    def plate(self, X, key):
        w, e, b = self.P[key]; x, ex = on16(X)
        if self.capture: self.cap[key] = (x, ex)
        return rshift_round(imatmul(x, w), -(e[None, :] + ex[:, None] + ES)) + b

    def attend(self, QKV):
        n = len(QKV); d, hs = self.d, self.hs; ARR = tables()["arrival"]
        live = np.tril(np.ones((n, n), bool)); out = np.zeros((n, d), np.int64)
        for h in range(self.nh):
            q = QKV[:, h * hs:(h + 1) * hs]; k = QKV[:, d + h * hs:d + (h + 1) * hs]; v = QKV[:, 2 * d + h * hs:2 * d + (h + 1) * hs]
            q16, eq = on16(q); k16, ek = on16(k)
            sc = times_pow2(imatmul(q16, k16.T), (eq[:, None] + ek[None, :] + SCORE) * 65536 - self.half)
            top = np.where(live, sc, np.iinfo(np.int64).min).max(1)
            gap = top[:, None] - sc
            a = np.where(live & (gap < len(ARR)), ARR[np.clip(gap, 0, len(ARR) - 1)], 0)
            tot = a.sum(1)
            av = a @ v
            out[:, h * hs:(h + 1) * hs] = times_pow2(av, -np.array([log2_fine(t) for t in tot], np.int64)[:, None])
        return out

    def bend(self, H):
        SH = tables()["share"]
        i = rshift_round(H, ES - 12) + 32768
        sh = np.where(i < 0, 0, np.where(i > 65535, 65536, SH[np.clip(i, 0, 65535)]))
        return rshift_round(H * sh, 16)

    def run(self, ids, last):
        ids = np.asarray(ids); pos = np.arange(len(ids))
        R = rshift_round(self.wte[ids], -(self.wte_e[ids][:, None] + ES)) + rshift_round(self.wpe[pos], -(self.wpe_e[pos][:, None] + ES))
        for l in range(self.L):
            R = R + self.plate(self.attend(self.plate(self.level(R, f"ln1.{l}"), f"qkv.{l}")), f"proj.{l}")
            R = R + self.plate(self.bend(self.plate(self.level(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
        y, ey = on16(self.level(R[-last:], "lnf"))
        if self.capture: self.cap["head"] = (y, ey)
        w, e, _ = self.P["head"]
        piles = rshift_round(imatmul(y, w), -(e[None, :] + ey[:, None] + ES))
        return piles.argmax(1), piles
