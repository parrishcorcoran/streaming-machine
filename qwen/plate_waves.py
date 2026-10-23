"""The collapse formula on the plate, before any light: every row of every slab struck once onto a whole-number lap
(number-theoretic transform: exact turns on a prime circle; two primes, combined, so every pile comes back whole).
Per piece the light is struck onto a lap; every pile = one lap of the light's wave against the row's stored wave.
Checked against QuantumQwen3's piles to the last digit.
    python3 plate_waves.py
"""
import sys, json, time, numpy as np
sys.argv = ["x", "texts=0"]; src = open("qwen_base2.py").read(); exec(src[:src.index("from transformers import AutoTokenizer")])
PR = [(998244353, 3), (985661441, 3)]                                   # two prime circles with 2^k turns
def ntt(a, p, g, inv=False):                                            # whole-number lap: a turned onto n ticks (a: [..., n])
    a = a.copy() % p; n = a.shape[-1]; j = 0
    rev = np.zeros(n, np.int64)
    for i in range(1, n):
        bit = n >> 1
        while j & bit: j ^= bit; bit >>= 1
        j |= bit; rev[i] = j
    a = a[..., rev]; length = 2
    while length <= n:
        w = pow(g, (p - 1) // length, p)
        if inv: w = pow(w, p - 2, p)
        ws = np.array([pow(w, k, p) for k in range(length // 2)], np.uint64)
        a = a.reshape(a.shape[:-1] + (n // length, length))
        u = a[..., :length // 2].copy(); v = (a[..., length // 2:] * ws) % p
        a = np.concatenate([(u + v) % p, (u + p - v) % p], -1).reshape(a.shape[:-2] + (n,)); length <<= 1
    if inv: a = (a * pow(n, p - 2, p)) % p
    return a
def lap(a, n): out = np.zeros(a.shape[:-1] + (n,), np.uint64); out[..., :a.shape[-1]] = a; return out
def plate_wave(w, n, p, g):                                            # each row (pile j's weights) reversed onto the lap: stored once
    rows = (w.T.astype(np.int64) % p).astype(np.uint64); r = np.zeros((w.shape[1], n), np.uint64); r[:, 0] = rows[:, 0]; r[:, n - w.shape[0] + 1:] = rows[:, :0:-1]      # weight i at tick -i (mod n)
    return ntt(r, p, g)
def piles_by_waves(x, w):
    n = 1 << (w.shape[0] - 1).bit_length(); res = []
    for p, g in PR:
        Wv = plate_wave(w, n, p, g)                                     # the plate as waves (once, before any light)
        X = ntt(lap((x.astype(np.int64) % p).astype(np.uint64)[None, :], n), p, g)[0]   # the light as a wave (per piece)
        acc = np.zeros(w.shape[1], np.uint64)
        for c in range(0, n, 8): acc = (acc + ((Wv[:, c:c + 8] * X[c:c + 8]) % p).sum(1)) % p    # one lap: the light's wave against each row's wave
        res.append((acc * pow(n, p - 2, p)) % p)
    (p1, _), (p2, _) = PR; a1, a2 = res[0].astype(object), res[1].astype(object)
    t = ((a2 - a1) * pow(p1, -1, p2)) % p2; v = a1 + p1 * t; M2 = p1 * p2
    return np.array([int(z) - M2 if z > M2 // 2 else int(z) for z in v], dtype=np.int64)   # the two circles together: the true signed pile
r = json.load(open("refs_qwen.json"))[0]; R = M["embed"][r["ids"]].copy()
x16, ex = to16(R); xq = x16[-1]
QKV = gauged_slab(R, "qkv.0"); A = attention(QKV, 0); a16, _ = to16(A)
R2 = R + plain_slab(A, "o.0"); g16, _ = to16(R2); GU = gauged_slab(R2, "gu.0"); H = bend(GU[:, :3072], GU[:, 3072:]); h16, _ = to16(H)
for name, x, w in [("qkv.0", xq, W["qkv.0"][0]), ("o.0", a16[-1], W["o.0"][0]), ("gu.0", g16[-1], W["gu.0"][0]), ("down.0", h16[-1], W["down.0"][0]), ("head[:8192]", g16[-1], W["head"][0][:, :8192])]:
    t0 = time.time(); pw = piles_by_waves(x, w); same = np.array_equal(pw, x.astype(np.int64) @ w.astype(np.int64))
    n = 1 << (w.shape[0] - 1).bit_length()
    print(f"  {name:12s} {w.shape[0]:5d} -> {w.shape[1]:6d}: piles from one lap per row == machine's piles: {same}   (lap of {n} ticks per row; {time.time() - t0:.1f} s)")
