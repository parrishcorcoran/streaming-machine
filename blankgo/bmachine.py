"""The whole-number machine of machine.py, copied part for part so it can be imported (machine.py runs on import), with
one addition: it can keep the light that meets each plate. Nothing in the arithmetic is changed: every function below is
machine.py's own, line for line; the only new lines are the ones marked CAPTURE and the extra plate choices.

  Machine(shape d, L, recipe, step t, plates, capture=False)
    plates  'finished' .w   'pushes' .w - .s   'scrK' .w - .s + scramble(.s, K)    (as machine.py)
            'blank'    .s on every drawn part (plates, both word tables, position rows); gains, offsets, biases finished
            a callable f(key, w, s) -> whole numbers on the finished grid (for the blank with its counter removed)
  capture=True keeps, for every plate the light meets, (x16, ex): the light on 16 places with its power per position,
  exactly the numbers the plate's product uses. The pile is x16 @ w; the blank's part x16 @ s and the built part
  x16 @ (w - s) add to it digit for digit.

identity.py checks that this copy gives every head pile of machine.py's runs, sha256 for sha256.
"""
import os, json, math, hashlib, numpy as np
from model import HERE, tag
from snapio import scramble

TB = np.load(os.path.join(HERE, "machines", "tables.npz"))
SHARE, ARR, LOG2, EXP2 = TB["share"], TB["arrival"], TB["log2"], TB["exp2"]
ES, SC = 16, 10
MATMUL = "float64"

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
    if MATMUL == "int": return x16 @ w.astype(np.int64)
    with np.errstate(all="ignore"):
        p = x16.astype(np.float64) @ w
    assert np.isfinite(p).all() and (p == np.round(p)).all()
    return p.astype(np.int64)

def log2fix(v):                                       # log2 of a positive whole number: top 1's place + a table read
    v = int(v); t = v.bit_length() - 1
    m = (v >> (t - 16)) & 0xFFFF if t >= 16 else (v << (16 - t)) & 0xFFFF
    return t * 65536 + int(LOG2[m])

def power(X, F):                                      # X x 2^(F / 65536): a table read of the fine part, then a shift
    F = np.broadcast_to(np.asarray(F, dtype=np.int64), X.shape); I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF)
    if np.abs(X).max(initial=0) < 2 ** 45: return rshift(X * EXP2[f], 16 - I)
    return _shift_py(X.astype(object) * EXP2[f].astype(object), (16 - I).astype(object)).astype(np.int64)

def race(vals):                                       # who arrives first: top digits first; at each digit, if anyone still in
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)   # has a 1 there, everyone with a 0 drops out
    alive = np.ones(len(u), dtype=bool)
    for bit in range(63, -1, -1):
        has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        if has.any(): alive = has
    return int(np.flatnonzero(alive)[0])

def machine_file(d, L, recipe, t):
    return np.load(os.path.join(HERE, "machines", f"{tag(d, L)}_{recipe}_step{t}.npz"))

class Machine:
    def __init__(self, d0, L0, recipe, t, variant, capture=False, M=None):
        M = machine_file(d0, L0, recipe, t) if M is None else M
        self.d, self.L, self.NH, self.HS = (int(M[k]) for k in ("d", "L", "heads", "hs"))
        self.EPS40, self.HALF = int(M["eps40"]), int(M["half_log2_hs"])
        self.capture, self.cap = capture, {}                                     # CAPTURE
        d = self.d
        perm = np.arange(d)
        def plate(key, fname):                        # the variant's whole numbers on the finished plate's grid
            w, s = M[key + ".w"].astype(np.int64), M[key + ".s"].astype(np.int64)
            if callable(variant): return variant(key, w, s)
            if variant == "finished": return w
            if variant == "pushes": return w - s
            if variant == "blank": return s
            if variant.startswith("scr"): return w - s + scramble(fname, s, int(variant[3:]))
            raise ValueError(variant)
        self.over16 = 0
        self.PL, self.NORM = {}, {}
        for l in range(self.L):
            for nm in ("ln1", "ln2"):
                k = f"{nm}.{l}"; self.NORM[k] = (M[k + ".g"][perm], int(M[k + ".ge"]), M[k + ".b"][perm])
            for nm in ("qkv", "proj", "fc", "mproj"):
                k = f"{nm}.{l}"; w = plate(k, k + ".w"); e, b = M[k + ".e"], M[k + ".b"]
                self.over16 += int((np.abs(w) > 32767).sum())
                if nm in ("qkv", "fc"): w = w[perm]
                else: w, e, b = w[:, perm], e[perm], b[perm]
                self.PL[k] = (w.astype(np.float64), e, b, float(np.abs(w).max()))
        self.NORM["lnf"] = (M["lnf.g"][perm], int(M["lnf.ge"]), M["lnf.b"][perm])
        w = plate("head", "head")[perm]; self.over16 += int((np.abs(w) > 32767).sum())
        self.PL["head"] = (w.astype(np.float64), M["head.e"], 0, float(np.abs(w).max()))
        self.WTE, self.WTE_E = plate("wte", "wte")[:, perm], M["wte.e"]
        self.WPE, self.WPE_E = plate("wpe", "wpe")[:, perm], M["wpe.e"]

    def gauge(self, R, name):
        g, ge, b = self.NORM[name]; d = self.d
        x16, ex = to16(R)
        S1, S2 = x16.sum(1), (x16 * x16).sum(1)
        c = d * x16 - S1[:, None]
        V = d * S2 - S1 * S1 + np.array([self.EPS40 >> (40 + 2 * int(e)) if 40 + 2 * int(e) >= 0 else self.EPS40 << -(40 + 2 * int(e)) for e in ex])
        LS = np.array([log2fix(v) >> 1 for v in V])
        return power(c * g[None, :], (ge + ES) * 65536 - LS[:, None]) + b

    def plain(self, X, name):
        w, e, b, wmax = self.PL[name]; x16, ex = to16(X)
        if self.capture: self.cap.setdefault(name, []).append((x16, ex))        # CAPTURE
        return rshift(product(x16, w, wmax), -(e[None, :] + ex[:, None] + ES)) + b

    def arrival(self, QKV):
        P = len(QKV); d, HS = self.d, self.HS; out = np.zeros((P, d), np.int64); live = np.tril(np.ones((P, P), dtype=bool))
        for h in range(self.NH):
            q, k, v = QKV[:, h * HS:(h + 1) * HS], QKV[:, d + h * HS:d + (h + 1) * HS], QKV[:, 2 * d + h * HS:2 * d + (h + 1) * HS]
            q16, eq = to16(q); k16, ek = to16(k)
            sc = power(product(q16, k16.T.astype(np.float64), 2.0 ** 15), (eq[:, None] + ek[None, :] + SC) * 65536 - self.HALF)
            lev = np.where(live, sc, np.iinfo(np.int64).min).max(1)
            dd = lev[:, None] - sc
            a = np.where(live & (dd < len(ARR)), ARR[np.clip(dd, 0, len(ARR) - 1)], 0)
            assert float(np.abs(v).max(initial=0)) * float(a.sum(1).max()) < 2 ** 62
            out[:, h * HS:(h + 1) * HS] = power(a @ v, -np.array([log2fix(t) for t in a.sum(1)])[:, None])
        return out

    def edge(self, H):
        idx = rshift(H, ES - 12) + 32768
        share = np.where(idx < 0, 0, np.where(idx > 65535, 65536, SHARE[np.clip(idx, 0, 65535)]))
        return rshift(H * share, 16)

    def forward(self, ids, last):
        ids = np.asarray(ids); pos = np.arange(len(ids))
        R = rshift(self.WTE[ids], -(self.WTE_E[ids][:, None] + ES)) + rshift(self.WPE[pos], -(self.WPE_E[pos][:, None] + ES))
        for l in range(self.L):
            R = R + self.plain(self.arrival(self.plain(self.gauge(R, f"ln1.{l}"), f"qkv.{l}")), f"proj.{l}")
            R = R + self.plain(self.edge(self.plain(self.gauge(R, f"ln2.{l}"), f"fc.{l}")), f"mproj.{l}")
        y16, ey = to16(self.gauge(R[-last:], "lnf")); w, e, _, wmax = self.PL["head"]
        if self.capture: self.cap.setdefault("head", []).append((y16, ey))      # CAPTURE
        lg = rshift(product(y16, w, wmax), -(e[None, :] + ey[:, None] + ES))
        return [race(row) for row in lg], lg

def pile_sha(lg): return hashlib.sha256(lg.tobytes()).hexdigest()
