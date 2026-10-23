"""QuantumQwen3, all base 2, checked against fp32 Qwen3-0.6B's own greedy words. Every value a whole number.
Switches put one part back to float, to trace any difference to its place:
    gauge=float  turn=float  bend=float  arrival=float   words=8  texts=3
"""
import sys, json, math, numpy as np
A = dict(a.split("=") for a in sys.argv[1:])
GAUGE, TURN, BEND, ARRIVAL = A.get("gauge", "int"), A.get("turn", "int"), A.get("bend", "int"), A.get("arrival", "int")
WORDS, TEXTS = int(A.get("words", 8)), int(A.get("texts", 3))
M = np.load("qwen_base2.npz"); ES, SC, L, D, HD = 16, 10, 28, 1024, 128
W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}
LOG2, EXP2, ARR, SIG, COS, SIN, EPS40 = M["log2"], M["exp2"], M["arrival"], M["sig"], M["cos"], M["sin"], int(M["eps40"])
def rshift(X, s):
    s = np.asarray(s, dtype=np.int64); up = np.left_shift(X, np.maximum(-s, 0)); sp = np.maximum(s, 0)
    return np.right_shift(up + np.where(sp > 0, np.left_shift(np.int64(1), np.maximum(sp - 1, 0)), 0), sp)
def to16(X):                                                   # each row on 16 bits, its own power of two (from the OR of its sizes)
    top = np.array([int(v).bit_length() for v in np.bitwise_or.reduce(np.abs(X), axis=1)]); sh = np.maximum(top - 15, 0)
    return rshift(X, sh[:, None]), sh - ES
def log2fix(v):
    v = int(v); t = v.bit_length() - 1; m = (v >> (t - 16)) & 0xFFFF if t >= 16 else (v << (16 - t)) & 0xFFFF; return t * 65536 + int(LOG2[m])
def power(X, F):
    F = np.asarray(F, dtype=np.int64); I, f = np.right_shift(F, 16), np.bitwise_and(F, 0xFFFF); return rshift(X * EXP2[f], 16 - I)
def gauge(x16, ex, n):                                         # log2 of the root mean square, no root and no divide: halve the log
    V = (x16 * x16).sum(1); half = int(round(math.log2(n) * 65536)) // 2
    return np.array([(log2fix(V[p] + ((EPS40 * n) >> (40 + 2 * int(ex[p])) if 40 + 2 * int(ex[p]) >= 0 else (EPS40 * n) << -(40 + 2 * int(ex[p])))) >> 1) - half + int(ex[p]) * 65536 for p in range(len(V))])
def gauged_slab(R, name):
    w, we = W[name]; x16, ex = to16(R); pile = x16 @ w
    if GAUGE == "float":
        Rf = R / 2.0 ** ES; rms = np.sqrt((Rf ** 2).mean(1) + 1e-6); return np.round(pile * 2.0 ** (we[None, :] + ex[:, None] + ES) / rms[:, None]).astype(np.int64)
    return power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - gauge(x16, ex, D)[:, None])
def plain_slab(X, name):
    w, we = W[name]; x16, ex = to16(X); return rshift(x16 @ w, -(we[None, :] + ex[:, None] + ES))
def head_gauge(X, g, ge):                                      # per head: x / rms(x) * gain
    if GAUGE == "float":
        Xf = X / 2.0 ** ES; rms = np.sqrt((Xf ** 2).mean(1) + 1e-6); return np.round(Xf / rms[:, None] * g * 2.0 ** ge * 2 ** ES).astype(np.int64)
    x16, ex = to16(X); return power(x16 * g[None, :], (ge + ex[:, None] + ES) * 65536 - gauge(x16, ex, HD)[:, None])
def turn(X, pos):                                              # pair (i, i+64) turns by its position's angle
    a, b = X[:, :64], X[:, 64:]
    if TURN == "float":
        ang = pos[:, None] * (1e6 ** (-np.arange(0, HD, 2) / HD))[None, :]; c, s = np.cos(ang), np.sin(ang)
        return np.round(np.concatenate([a * c - b * s, b * c + a * s], 1)).astype(np.int64)
    c, s = COS[pos], SIN[pos]; return np.concatenate([rshift(a * c - b * s, 30), rshift(b * c + a * s, 30)], 1)
def race(vals):
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63); alive = np.ones(len(u), bool)
    for bit in range(63, -1, -1):
        has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        if has.any(): alive = has
    return int(np.flatnonzero(alive)[0])
def attention(QKV, l):
    P = QKV.shape[0]; pos = np.arange(P); out = np.zeros((P, 16 * HD), np.int64)
    qg, qge, kg, kge = M[f"qg.{l}"], int(M[f"qge.{l}"]), M[f"kg.{l}"], int(M[f"kge.{l}"])
    K = [turn(head_gauge(QKV[:, 2048 + j * HD:2048 + (j + 1) * HD], kg, kge), pos) for j in range(8)]
    V = [QKV[:, 3072 + j * HD:3072 + (j + 1) * HD] for j in range(8)]
    for h in range(16):
        q = turn(head_gauge(QKV[:, h * HD:(h + 1) * HD], qg, qge), pos); k, v = K[h // 2], V[h // 2]
        if ARRIVAL == "float":
            s = (q / 2.0 ** ES) @ (k / 2.0 ** ES).T; s = np.where(np.tril(np.ones((P, P))) > 0, s, -1e30)
            a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True); out[:, h * HD:(h + 1) * HD] = np.round(a @ (v / 2.0 ** ES) * 2 ** ES).astype(np.int64); continue
        q16, eq = to16(q); k16, ek = to16(k); sc = rshift(q16 @ k16.T, -(eq[:, None] + ek[None, :] + SC))
        for p in range(P):
            row = sc[p, :p + 1]; d = row[race(row)] - row; a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            out[p, h * HD:(h + 1) * HD] = power(a[None, :] @ v[:p + 1], -log2fix(a.sum()))[0]
    return out
def bend(G, U):                                                # the gated bend: g * sigma(g) * u
    if BEND == "float":
        g = G / 2.0 ** ES; return np.round(g / (1 + np.exp(-g)) * (U / 2.0 ** ES) * 2 ** ES).astype(np.int64)
    idx = rshift(G, ES - 12) + 65536; share = np.where(idx < 0, 0, np.where(idx > 131071, 65536, SIG[np.clip(idx, 0, 131071)]))
    return rshift(rshift(G * share, 16) * U, ES)
def forward(ids):
    R = M["embed"][ids].copy()
    for l in range(L):
        R = R + plain_slab(attention(gauged_slab(R, f"qkv.{l}"), l), f"o.{l}")
        GU = gauged_slab(R, f"gu.{l}"); R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}")
    return race(gauged_slab(R[-1:], "head")[0])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); refs = json.load(open("refs_qwen.json"))
print(f"QuantumQwen3, all base 2: gauge {GAUGE}, turn {TURN}, bend {BEND}, arrival {ARRIVAL}")
for r in refs[:TEXTS]:
    got = []
    for _ in range(WORDS): got.append(forward(r["ids"] + got))
    want = r["ref"][:WORDS]; print(f"  {r['prompt']!r:30s} {'IDENTICAL' if got == want else 'DIFFERENT'}  {tok.decode(got)!r}" + ("" if got == want else f"   model: {tok.decode(want)!r}"))
